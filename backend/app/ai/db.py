"""Database setup for the AI Intelligence Layer.

Provides the SQLAlchemy async engine, session factory, and declarative base
for AI-related database models (e.g., embedding_sync_status).
"""

from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

# Naming conventions for consistent constraint naming in Alembic migrations
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base for all AI Intelligence Layer models."""

    metadata = MetaData(naming_convention=convention)


def create_session_factory(database_url: str) -> async_sessionmaker[AsyncSession]:
    """Create an async session factory for the given database URL.

    Args:
        database_url: PostgreSQL async connection string
            (e.g., 'postgresql+asyncpg://user:pass@host/db').

    Returns:
        An async session factory bound to the created engine.
    """
    engine = create_async_engine(database_url, echo=False)
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
