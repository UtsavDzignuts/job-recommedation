"""Unit tests for the Sync Service.

Tests sync_entity, full_resync, mark_failed, and queue behavior
using an in-memory SQLite database and mocked dependencies.
"""

import pytest
import pytest_asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.ai.db import Base
from app.ai.exceptions import EmbeddingGenerationError, VectorDBUnavailableError
from app.ai.models import SyncReport
from app.ai.sync_models import EmbeddingSyncStatus
from app.ai.sync_service import (
    ENTITY_COLLECTION_MAP,
    QueuedOperation,
    SyncService,
    get_entity_text,
)


@pytest_asyncio.fixture
async def async_session():
    """Create an in-memory SQLite async session for testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest.fixture
def mock_embedding_service():
    """Create a mocked EmbeddingService."""
    service = AsyncMock()
    service.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3] * 512)
    service.generate_embeddings_batch = AsyncMock(
        return_value=[[0.1, 0.2, 0.3] * 512]
    )
    return service


@pytest.fixture
def mock_vector_store():
    """Create a mocked VectorStoreInterface."""
    store = AsyncMock()
    store.upsert = AsyncMock()
    store.delete = AsyncMock()
    store.health_check = AsyncMock(return_value=True)
    return store


@pytest_asyncio.fixture
async def sync_service(async_session, mock_embedding_service, mock_vector_store):
    """Create a SyncService instance with mocked dependencies."""
    return SyncService(
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
        db_session=async_session,
    )


class TestGetEntityText:
    """Tests for the get_entity_text helper function."""

    def test_job_post_text(self):
        entity = {
            "title": "Software Engineer",
            "description": "Build great software",
            "requirements": "5 years Python",
        }
        result = get_entity_text("job_post", entity)
        assert "Software Engineer" in result
        assert "Build great software" in result
        assert "5 years Python" in result

    def test_company_text(self):
        entity = {
            "name": "Acme Corp",
            "description": "Leading tech company",
            "industry": "Technology",
        }
        result = get_entity_text("company", entity)
        assert "Acme Corp" in result
        assert "Leading tech company" in result
        assert "Technology" in result

    def test_candidate_text(self):
        entity = {
            "skills": "Python, FastAPI",
            "experience": "5 years backend",
            "bio": "Passionate developer",
        }
        result = get_entity_text("candidate", entity)
        assert "Python, FastAPI" in result
        assert "5 years backend" in result
        assert "Passionate developer" in result

    def test_missing_fields_handled(self):
        entity = {"title": "Engineer"}
        result = get_entity_text("job_post", entity)
        assert "Engineer" in result

    def test_unknown_entity_type_fallback(self):
        entity = {"field1": "value1", "field2": "value2"}
        result = get_entity_text("unknown", entity)
        assert "value1" in result
        assert "value2" in result


class TestSyncEntity:
    """Tests for sync_entity method."""

    @pytest.mark.asyncio
    async def test_create_entity_upserts_to_vector_db(
        self, sync_service, mock_embedding_service, mock_vector_store, async_session
    ):
        entity_data = {
            "title": "Data Scientist",
            "description": "ML role",
            "requirements": "PhD preferred",
        }

        await sync_service.sync_entity(
            "job_post", "job123", "create", entity_data
        )

        # Verify embedding was generated
        mock_embedding_service.generate_embedding.assert_called_once()
        # Verify upsert was called
        mock_vector_store.upsert.assert_called_once()
        call_args = mock_vector_store.upsert.call_args
        assert call_args[0][0] == "job_posts"  # collection name
        docs = call_args[0][1]
        assert len(docs) == 1
        assert docs[0].id == "job_post_job123"
        assert docs[0].metadata["entity_type"] == "job_post"
        assert docs[0].metadata["entity_id"] == "job123"

    @pytest.mark.asyncio
    async def test_update_entity_upserts_to_vector_db(
        self, sync_service, mock_embedding_service, mock_vector_store
    ):
        entity_data = {
            "name": "Tech Inc",
            "description": "Updated description",
            "industry": "SaaS",
        }

        await sync_service.sync_entity(
            "company", "comp456", "update", entity_data
        )

        mock_embedding_service.generate_embedding.assert_called_once()
        mock_vector_store.upsert.assert_called_once()
        call_args = mock_vector_store.upsert.call_args
        assert call_args[0][0] == "companies"

    @pytest.mark.asyncio
    async def test_delete_entity_removes_from_vector_db(
        self, sync_service, mock_vector_store
    ):
        await sync_service.sync_entity("candidate", "cand789", "delete")

        mock_vector_store.delete.assert_called_once_with(
            "candidates", ["candidate_cand789"]
        )

    @pytest.mark.asyncio
    async def test_invalid_entity_type_raises_value_error(self, sync_service):
        with pytest.raises(ValueError, match="Invalid entity_type"):
            await sync_service.sync_entity("invalid_type", "id1", "create")

    @pytest.mark.asyncio
    async def test_invalid_action_raises_value_error(self, sync_service):
        with pytest.raises(ValueError, match="Invalid action"):
            await sync_service.sync_entity("job_post", "id1", "invalid_action")

    @pytest.mark.asyncio
    async def test_vector_db_unavailable_queues_operation(
        self, sync_service, mock_vector_store
    ):
        mock_vector_store.upsert.side_effect = VectorDBUnavailableError(
            "Connection refused"
        )
        entity_data = {"title": "Engineer", "description": "Role", "requirements": ""}

        await sync_service.sync_entity(
            "job_post", "job1", "create", entity_data
        )

        # Should be queued for retry
        assert len(sync_service.retry_queue) == 1
        queued = sync_service.retry_queue[0]
        assert queued.entity_type == "job_post"
        assert queued.entity_id == "job1"
        assert queued.action == "create"

    @pytest.mark.asyncio
    async def test_embedding_generation_failure_marks_failed(
        self, sync_service, mock_embedding_service, async_session
    ):
        mock_embedding_service.generate_embedding.side_effect = (
            EmbeddingGenerationError("API error")
        )
        entity_data = {"title": "Role", "description": "Desc", "requirements": "Reqs"}

        await sync_service.sync_entity(
            "job_post", "job2", "create", entity_data
        )

        # Verify marked as failed in DB
        stmt = select(EmbeddingSyncStatus).where(
            EmbeddingSyncStatus.entity_id == "job2"
        )
        result = await async_session.execute(stmt)
        record = result.scalar_one_or_none()
        assert record is not None
        assert record.status == "failed"
        assert "API error" in record.last_error
        assert record.retry_count == 1

    @pytest.mark.asyncio
    async def test_sync_entity_marks_synced_on_success(
        self, sync_service, async_session
    ):
        entity_data = {
            "skills": "Python",
            "experience": "3 years",
            "bio": "Developer",
        }

        await sync_service.sync_entity(
            "candidate", "cand1", "create", entity_data
        )

        stmt = select(EmbeddingSyncStatus).where(
            EmbeddingSyncStatus.entity_id == "cand1"
        )
        result = await async_session.execute(stmt)
        record = result.scalar_one_or_none()
        assert record is not None
        assert record.status == "synced"
        assert record.last_synced_at is not None


class TestFullResync:
    """Tests for full_resync method."""

    @pytest.mark.asyncio
    async def test_full_resync_creates_embeddings_for_all_entities(
        self, sync_service, mock_embedding_service, mock_vector_store
    ):
        async def entity_fetcher():
            return {
                "job_post": [
                    {"id": "j1", "title": "Dev", "description": "Build", "requirements": "Python"},
                    {"id": "j2", "title": "PM", "description": "Lead", "requirements": "MBA"},
                ],
                "company": [
                    {"id": "c1", "name": "Corp", "description": "Tech", "industry": "IT"},
                ],
            }

        report = await sync_service.full_resync(entity_fetcher=entity_fetcher)

        assert isinstance(report, SyncReport)
        assert report.total_entities == 3
        assert report.created == 3
        assert report.failed == 0
        assert report.duration_seconds >= 0
        assert mock_embedding_service.generate_embedding.call_count == 3
        assert mock_vector_store.upsert.call_count == 3

    @pytest.mark.asyncio
    async def test_full_resync_is_idempotent(
        self, sync_service, mock_embedding_service, mock_vector_store, async_session
    ):
        async def entity_fetcher():
            return {
                "job_post": [
                    {"id": "j1", "title": "Dev", "description": "Build", "requirements": "Python"},
                ],
            }

        # First sync
        report1 = await sync_service.full_resync(entity_fetcher=entity_fetcher)
        assert report1.created == 1

        # Second sync - same entities
        report2 = await sync_service.full_resync(entity_fetcher=entity_fetcher)
        # Should be updated (not duplicated)
        assert report2.updated == 1
        assert report2.created == 0

    @pytest.mark.asyncio
    async def test_full_resync_handles_entity_without_id(
        self, sync_service, mock_embedding_service
    ):
        async def entity_fetcher():
            return {
                "job_post": [
                    {"title": "No ID Job", "description": "Missing", "requirements": ""},
                ],
            }

        report = await sync_service.full_resync(entity_fetcher=entity_fetcher)
        assert report.total_entities == 1
        assert report.failed == 1
        # Embedding should not be generated for entity without ID
        mock_embedding_service.generate_embedding.assert_not_called()

    @pytest.mark.asyncio
    async def test_full_resync_continues_on_individual_failure(
        self, sync_service, mock_embedding_service, mock_vector_store
    ):
        # First call succeeds, second fails
        mock_embedding_service.generate_embedding.side_effect = [
            [0.1] * 1536,
            EmbeddingGenerationError("Timeout"),
        ]

        async def entity_fetcher():
            return {
                "job_post": [
                    {"id": "j1", "title": "Job 1", "description": "D1", "requirements": "R1"},
                    {"id": "j2", "title": "Job 2", "description": "D2", "requirements": "R2"},
                ],
            }

        report = await sync_service.full_resync(entity_fetcher=entity_fetcher)
        assert report.total_entities == 2
        assert report.created == 1
        assert report.failed == 1

    @pytest.mark.asyncio
    async def test_full_resync_with_no_entities(self, sync_service):
        async def entity_fetcher():
            return {}

        report = await sync_service.full_resync(entity_fetcher=entity_fetcher)
        assert report.total_entities == 0
        assert report.created == 0
        assert report.updated == 0
        assert report.failed == 0

    @pytest.mark.asyncio
    async def test_full_resync_queues_on_vector_db_unavailable(
        self, sync_service, mock_vector_store
    ):
        mock_vector_store.upsert.side_effect = VectorDBUnavailableError("Down")

        async def entity_fetcher():
            return {
                "job_post": [
                    {"id": "j1", "title": "Job", "description": "D", "requirements": "R"},
                ],
            }

        report = await sync_service.full_resync(entity_fetcher=entity_fetcher)
        assert report.failed == 1
        assert len(sync_service.retry_queue) == 1


class TestMarkFailed:
    """Tests for mark_failed method."""

    @pytest.mark.asyncio
    async def test_mark_failed_creates_record(self, sync_service, async_session):
        await sync_service.mark_failed("job_post", "j99", "Connection timeout")

        stmt = select(EmbeddingSyncStatus).where(
            EmbeddingSyncStatus.entity_id == "j99"
        )
        result = await async_session.execute(stmt)
        record = result.scalar_one_or_none()
        assert record is not None
        assert record.status == "failed"
        assert record.last_error == "Connection timeout"
        assert record.retry_count == 1

    @pytest.mark.asyncio
    async def test_mark_failed_increments_retry_count(
        self, sync_service, async_session
    ):
        await sync_service.mark_failed("company", "c1", "Error 1")
        await sync_service.mark_failed("company", "c1", "Error 2")

        stmt = select(EmbeddingSyncStatus).where(
            EmbeddingSyncStatus.entity_id == "c1"
        )
        result = await async_session.execute(stmt)
        record = result.scalar_one_or_none()
        assert record is not None
        assert record.retry_count == 2
        assert record.last_error == "Error 2"


class TestRetryQueue:
    """Tests for the retry queue mechanism."""

    @pytest.mark.asyncio
    async def test_queue_populated_on_vector_db_failure(
        self, sync_service, mock_vector_store
    ):
        mock_vector_store.upsert.side_effect = VectorDBUnavailableError("Unavailable")
        entity_data = {"title": "Job", "description": "Desc", "requirements": "Req"}

        await sync_service.sync_entity("job_post", "j1", "create", entity_data)
        await sync_service.sync_entity("job_post", "j2", "create", entity_data)

        assert len(sync_service.retry_queue) == 2
        assert sync_service.retry_queue[0].entity_id == "j1"
        assert sync_service.retry_queue[1].entity_id == "j2"

    @pytest.mark.asyncio
    async def test_delete_on_vector_db_unavailable_queues(
        self, sync_service, mock_vector_store
    ):
        mock_vector_store.delete.side_effect = VectorDBUnavailableError("Down")

        await sync_service.sync_entity("candidate", "c1", "delete")

        assert len(sync_service.retry_queue) == 1
        assert sync_service.retry_queue[0].action == "delete"
