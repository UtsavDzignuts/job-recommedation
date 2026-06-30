"""Create embedding_sync_status table

Revision ID: 001
Revises:
Create Date: 2024-01-01 00:00:00.000000+00:00

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "embedding_sync_status",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("entity_type", sa.String(50), nullable=False),
        sa.Column("entity_id", sa.String(100), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("last_synced_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_embedding_sync_status"),
        sa.UniqueConstraint(
            "entity_type", "entity_id", name="uq_sync_entity_type_entity_id"
        ),
    )
    # Create index on entity_type + status for efficient querying during resync
    op.create_index(
        "ix_embedding_sync_status_type_status",
        "embedding_sync_status",
        ["entity_type", "status"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_embedding_sync_status_type_status",
        table_name="embedding_sync_status",
    )
    op.drop_table("embedding_sync_status")
