"""SQLAlchemy models for the Sync Service.

Defines the embedding_sync_status table used to track the synchronization
state of entity embeddings between PostgreSQL and the vector database.
"""

from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.ai.db import Base


class EmbeddingSyncStatus(Base):
    """Tracks the sync status of entity embeddings.

    Each row represents a single entity (job_post, company, or candidate)
    and its current embedding sync state (pending, synced, or failed).
    """

    __tablename__ = "embedding_sync_status"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        UniqueConstraint("entity_type", "entity_id", name="uq_sync_entity_type_entity_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<EmbeddingSyncStatus(entity_type={self.entity_type!r}, "
            f"entity_id={self.entity_id!r}, status={self.status!r})>"
        )
