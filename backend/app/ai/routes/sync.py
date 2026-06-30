"""Sync API endpoint for the AI Intelligence Layer.

Provides the POST /sync/full endpoint that triggers a full idempotent
re-sync of all entity embeddings from PostgreSQL to the vector database.
"""

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status

from app.ai.config import AIConfig
from app.ai.db import create_session_factory
from app.ai.embedding_service import EmbeddingService
from app.ai.models import SyncReport
from app.ai.sync_service import SyncService
from app.ai.vectorstore.factory import create_vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sync", tags=["sync"])


def _get_config() -> AIConfig:
    """Provide AIConfig instance."""
    return AIConfig()


async def _get_sync_service(
    config: AIConfig = Depends(_get_config),
) -> SyncService:
    """Dependency that provides a configured SyncService instance.

    Creates the embedding service, vector store, and database session
    needed by the SyncService.
    """
    embedding_service = EmbeddingService(config=config)
    vector_store = create_vector_store(config=config)

    # Create a database session for this request
    session_factory = create_session_factory(
        database_url=f"postgresql+asyncpg://localhost/kra_kpa"
    )
    session = session_factory()

    return SyncService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        db_session=session,
    )


async def _placeholder_entity_fetcher() -> Dict[str, List[Dict[str, Any]]]:
    """Entity fetcher that returns realistic sample data for the AI Intelligence Layer.

    In production, this would query all job_posts, companies, and candidates
    from the PostgreSQL database. Currently returns curated sample data
    for demonstration and testing purposes.

    Returns:
        A dict mapping entity_type -> list of entity dicts.
        Each entity dict has an 'id' field and relevant content fields.
    """
    from seed_data import SAMPLE_CANDIDATES, SAMPLE_COMPANIES, SAMPLE_JOB_POSTS

    return {
        "job_post": SAMPLE_JOB_POSTS,
        "company": SAMPLE_COMPANIES,
        "candidate": SAMPLE_CANDIDATES,
    }


@router.post(
    "/full",
    response_model=SyncReport,
    status_code=status.HTTP_200_OK,
    summary="Trigger full embedding re-sync",
    description=(
        "Triggers an idempotent full re-sync of all entity embeddings from "
        "PostgreSQL to the vector database. Creates missing embeddings, updates "
        "stale ones, and returns a report of what changed. Calling this endpoint "
        "multiple times produces the same vector database state."
    ),
)
async def full_sync(
    sync_service: SyncService = Depends(_get_sync_service),
) -> SyncReport:
    """Trigger a full idempotent re-sync of all embeddings.

    This endpoint is idempotent: calling it multiple times produces the same
    vector database state without duplicating embeddings.

    Returns:
        SyncReport with counts of created, updated, deleted, and failed entities.
    """
    try:
        report = await sync_service.full_resync(
            entity_fetcher=_placeholder_entity_fetcher
        )
        return report
    except Exception as exc:
        logger.error("Full sync failed: %s", str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Sync operation failed: {exc}",
        ) from exc
