"""Sync Service for the AI Intelligence Layer.

Keeps the vector database aligned with PostgreSQL data by detecting changes
and triggering embedding generation for job posts, companies, and candidates.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.embedding_service import EmbeddingService
from app.ai.exceptions import EmbeddingGenerationError, VectorDBUnavailableError
from app.ai.models import SyncReport, VectorDocument
from app.ai.sync_models import EmbeddingSyncStatus
from app.ai.vectorstore import VectorStoreInterface

logger = logging.getLogger(__name__)

# Mapping from entity_type to the vector DB collection name
ENTITY_COLLECTION_MAP: Dict[str, str] = {
    "job_post": "job_posts",
    "company": "companies",
    "candidate": "candidates",
}

# Valid entity types
VALID_ENTITY_TYPES = set(ENTITY_COLLECTION_MAP.keys())

# Valid actions for sync_entity
VALID_ACTIONS = {"create", "update", "delete"}


@dataclass
class QueuedOperation:
    """An embedding operation queued for later retry when vector DB is unavailable."""

    entity_type: str
    entity_id: str
    action: str
    text_content: Optional[str] = None
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


def get_entity_text(entity_type: str, entity: Dict[str, Any]) -> str:
    """Extract and combine relevant text fields from an entity for embedding.

    Args:
        entity_type: The type of entity ('job_post', 'company', 'candidate').
        entity: Dictionary with the entity's fields.

    Returns:
        Combined text suitable for embedding generation.
    """
    if entity_type == "job_post":
        title = entity.get("title", "")
        description = entity.get("description", "")
        requirements = entity.get("requirements", "")
        return f"{title} {description} {requirements}".strip()
    elif entity_type == "company":
        name = entity.get("name", "")
        description = entity.get("description", "")
        industry = entity.get("industry", "")
        return f"{name} {description} {industry}".strip()
    elif entity_type == "candidate":
        skills = entity.get("skills", "")
        experience = entity.get("experience", "")
        bio = entity.get("bio", "")
        return f"{skills} {experience} {bio}".strip()
    else:
        # Fallback: concatenate all string values
        parts = [str(v) for v in entity.values() if isinstance(v, str) and v]
        return " ".join(parts).strip()


class SyncService:
    """Service responsible for synchronizing entity embeddings between PostgreSQL
    and the vector database.

    Handles individual entity sync operations, full re-sync, failure tracking,
    and queuing operations when the vector DB is unavailable.

    Args:
        embedding_service: Service for generating text embeddings.
        vector_store: Vector database interface for storing/deleting embeddings.
        db_session: SQLAlchemy async session for accessing PostgreSQL.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreInterface,
        db_session: AsyncSession,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._db_session = db_session
        self._retry_queue: List[QueuedOperation] = []

    @property
    def retry_queue(self) -> List[QueuedOperation]:
        """Access the in-memory retry queue for queued operations."""
        return self._retry_queue

    async def sync_entity(
        self,
        entity_type: str,
        entity_id: str,
        action: str,
        entity_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Synchronize a single entity's embedding with the vector database.

        Handles create, update, and delete actions:
        - create/update: generates embedding from entity data and upserts to vector DB
        - delete: removes embedding from vector DB

        On success, updates the sync status to 'synced'.
        On failure, logs the error, increments retry_count, and marks as 'failed'.

        Args:
            entity_type: The entity type ('job_post', 'company', 'candidate').
            entity_id: Unique identifier of the entity.
            action: The sync action ('create', 'update', 'delete').
            entity_data: Dictionary of entity fields (required for create/update).

        Raises:
            ValueError: If entity_type or action is invalid.
        """
        if entity_type not in VALID_ENTITY_TYPES:
            raise ValueError(
                f"Invalid entity_type '{entity_type}'. "
                f"Must be one of: {sorted(VALID_ENTITY_TYPES)}"
            )
        if action not in VALID_ACTIONS:
            raise ValueError(
                f"Invalid action '{action}'. Must be one of: {sorted(VALID_ACTIONS)}"
            )

        collection = ENTITY_COLLECTION_MAP[entity_type]

        try:
            if action == "delete":
                await self._delete_entity(collection, entity_type, entity_id)
            else:
                # create or update
                await self._upsert_entity(
                    collection, entity_type, entity_id, entity_data or {}
                )

            # Mark as synced on success
            await self._update_sync_status(
                entity_type, entity_id, status="synced"
            )

        except VectorDBUnavailableError as exc:
            # Queue for later retry and log warning
            logger.warning(
                "Vector DB unavailable during sync of %s/%s (action=%s). "
                "Queuing for retry. Error: %s",
                entity_type,
                entity_id,
                action,
                str(exc),
            )
            self._retry_queue.append(
                QueuedOperation(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    action=action,
                    text_content=(
                        get_entity_text(entity_type, entity_data)
                        if entity_data
                        else None
                    ),
                )
            )
            await self._update_sync_status(
                entity_type,
                entity_id,
                status="pending",
                error=f"Queued: Vector DB unavailable - {exc}",
            )

        except (EmbeddingGenerationError, Exception) as exc:
            error_msg = str(exc)
            logger.error(
                "Failed to sync %s/%s (action=%s): %s",
                entity_type,
                entity_id,
                action,
                error_msg,
            )
            await self.mark_failed(entity_type, entity_id, error_msg)

    async def full_resync(
        self,
        entity_fetcher: Optional[Any] = None,
    ) -> SyncReport:
        """Perform a full idempotent re-sync of all entities.

        Queries all entities from PostgreSQL (via the entity_fetcher callable),
        creates missing embeddings, updates stale ones, and returns a SyncReport.

        This operation is idempotent: running it multiple times produces the same
        vector database state without duplicating embeddings.

        Args:
            entity_fetcher: An async callable that returns a dict mapping
                entity_type -> list of entity dicts. Each entity dict must
                have an 'id' field and relevant content fields.
                If None, fetches entities from sync status table only.

        Returns:
            SyncReport with counts of created, updated, deleted, failed entities
            and the operation duration.
        """
        start_time = time.time()

        created = 0
        updated = 0
        deleted = 0
        failed = 0
        total = 0

        if entity_fetcher is not None:
            # Fetch all entities from PostgreSQL via the provided fetcher
            all_entities = await entity_fetcher()
        else:
            # No fetcher provided — use empty dict
            all_entities = {}

        for entity_type, entities in all_entities.items():
            if entity_type not in VALID_ENTITY_TYPES:
                logger.warning(
                    "Skipping unknown entity_type '%s' during full_resync",
                    entity_type,
                )
                continue

            collection = ENTITY_COLLECTION_MAP[entity_type]

            for entity in entities:
                total += 1
                entity_id = str(entity.get("id", ""))
                if not entity_id:
                    logger.warning(
                        "Skipping entity of type '%s' with no 'id' field",
                        entity_type,
                    )
                    failed += 1
                    continue

                # Check current sync status
                current_status = await self._get_sync_status(entity_type, entity_id)

                try:
                    text_content = get_entity_text(entity_type, entity)
                    if not text_content:
                        logger.warning(
                            "Skipping %s/%s: no text content to embed",
                            entity_type,
                            entity_id,
                        )
                        failed += 1
                        continue

                    # Generate embedding
                    embedding = await self._embedding_service.generate_embedding(
                        text_content
                    )

                    # Create metadata
                    metadata = {
                        "entity_type": entity_type,
                        "entity_id": entity_id,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "text_snippet": text_content[:200],
                    }

                    # Upsert to vector DB (idempotent via document ID)
                    doc = VectorDocument(
                        id=f"{entity_type}_{entity_id}",
                        embedding=embedding,
                        metadata=metadata,
                        text_snippet=text_content[:200],
                    )
                    await self._vector_store.upsert(collection, [doc])

                    # Update sync status
                    if current_status is None:
                        created += 1
                    else:
                        updated += 1

                    await self._update_sync_status(
                        entity_type, entity_id, status="synced"
                    )

                except VectorDBUnavailableError as exc:
                    logger.warning(
                        "Vector DB unavailable during full_resync for %s/%s. "
                        "Queuing for retry. Error: %s",
                        entity_type,
                        entity_id,
                        str(exc),
                    )
                    self._retry_queue.append(
                        QueuedOperation(
                            entity_type=entity_type,
                            entity_id=entity_id,
                            action="update",
                            text_content=get_entity_text(entity_type, entity),
                        )
                    )
                    failed += 1

                except (EmbeddingGenerationError, Exception) as exc:
                    logger.error(
                        "Failed to sync %s/%s during full_resync: %s",
                        entity_type,
                        entity_id,
                        str(exc),
                    )
                    await self.mark_failed(entity_type, entity_id, str(exc))
                    failed += 1

        duration = time.time() - start_time

        return SyncReport(
            total_entities=total,
            created=created,
            updated=updated,
            deleted=deleted,
            failed=failed,
            duration_seconds=round(duration, 3),
        )

    async def mark_failed(
        self, entity_type: str, entity_id: str, error: str
    ) -> None:
        """Mark an entity as failed in the sync status table.

        Increments the retry_count and records the error message.

        Args:
            entity_type: The entity type.
            entity_id: The entity identifier.
            error: Description of the failure.
        """
        await self._update_sync_status(
            entity_type, entity_id, status="failed", error=error, increment_retry=True
        )

    async def _upsert_entity(
        self,
        collection: str,
        entity_type: str,
        entity_id: str,
        entity_data: Dict[str, Any],
    ) -> None:
        """Generate embedding for an entity and upsert it to the vector DB.

        Args:
            collection: Target vector DB collection name.
            entity_type: The entity type.
            entity_id: The entity identifier.
            entity_data: Dictionary with entity fields.
        """
        text_content = get_entity_text(entity_type, entity_data)
        if not text_content:
            raise ValueError(
                f"Cannot generate embedding for {entity_type}/{entity_id}: "
                "no text content extracted from entity data"
            )

        # Generate embedding
        embedding = await self._embedding_service.generate_embedding(text_content)

        # Build metadata
        metadata = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "text_snippet": text_content[:200],
        }

        # Create vector document and upsert
        doc = VectorDocument(
            id=f"{entity_type}_{entity_id}",
            embedding=embedding,
            metadata=metadata,
            text_snippet=text_content[:200],
        )
        await self._vector_store.upsert(collection, [doc])

    async def _delete_entity(
        self, collection: str, entity_type: str, entity_id: str
    ) -> None:
        """Delete an entity's embedding from the vector DB.

        Args:
            collection: Target vector DB collection name.
            entity_type: The entity type.
            entity_id: The entity identifier.
        """
        doc_id = f"{entity_type}_{entity_id}"
        await self._vector_store.delete(collection, [doc_id])

    async def _get_sync_status(
        self, entity_type: str, entity_id: str
    ) -> Optional[EmbeddingSyncStatus]:
        """Fetch the current sync status record for an entity.

        Args:
            entity_type: The entity type.
            entity_id: The entity identifier.

        Returns:
            The EmbeddingSyncStatus record, or None if not found.
        """
        stmt = select(EmbeddingSyncStatus).where(
            EmbeddingSyncStatus.entity_type == entity_type,
            EmbeddingSyncStatus.entity_id == entity_id,
        )
        result = await self._db_session.execute(stmt)
        return result.scalar_one_or_none()

    async def _update_sync_status(
        self,
        entity_type: str,
        entity_id: str,
        status: str,
        error: Optional[str] = None,
        increment_retry: bool = False,
    ) -> None:
        """Create or update the sync status record for an entity.

        Uses PostgreSQL upsert (INSERT ... ON CONFLICT UPDATE) to handle
        both new and existing records atomically.

        Args:
            entity_type: The entity type.
            entity_id: The entity identifier.
            status: New status ('pending', 'synced', 'failed').
            error: Optional error message (set for 'failed' status).
            increment_retry: Whether to increment the retry_count.
        """
        now = datetime.now(timezone.utc)

        # Check if record exists
        existing = await self._get_sync_status(entity_type, entity_id)

        if existing is None:
            # Insert new record
            new_record = EmbeddingSyncStatus(
                entity_type=entity_type,
                entity_id=entity_id,
                status=status,
                last_synced_at=now if status == "synced" else None,
                last_error=error,
                retry_count=1 if increment_retry else 0,
                created_at=now,
                updated_at=now,
            )
            self._db_session.add(new_record)
        else:
            # Update existing record
            existing.status = status
            existing.updated_at = now
            if status == "synced":
                existing.last_synced_at = now
                existing.last_error = None
            if error is not None:
                existing.last_error = error
            if increment_retry:
                existing.retry_count = (existing.retry_count or 0) + 1

        await self._db_session.commit()
