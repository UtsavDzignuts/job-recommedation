"""Factory for creating vector store instances based on configuration."""

import logging

from app.ai.config import AIConfig
from app.ai.vectorstore import VectorStoreInterface
from app.ai.vectorstore.chromadb_store import ChromaDBStore

logger = logging.getLogger(__name__)

# Supported provider identifiers
SUPPORTED_PROVIDERS = {"chromadb", "pgvector", "qdrant"}


def create_vector_store(config: AIConfig | None = None) -> VectorStoreInterface:
    """Create and return a vector store instance based on configuration.

    Reads the VECTOR_DB_PROVIDER setting from AIConfig to determine which
    implementation to instantiate.

    Args:
        config: Optional AIConfig instance. If not provided, a new one is created
                from environment variables.

    Returns:
        A VectorStoreInterface implementation for the configured provider.

    Raises:
        ValueError: If the configured provider is not supported.
    """
    if config is None:
        config = AIConfig()

    provider = config.VECTOR_DB_PROVIDER.lower().strip()

    if provider == "chromadb":
        logger.info("Initializing ChromaDB vector store at %s", config.VECTOR_DB_URL)
        return ChromaDBStore(
            url=config.VECTOR_DB_URL,
            api_key=config.VECTOR_DB_API_KEY,
        )
    elif provider == "pgvector":
        raise NotImplementedError(
            "PGVector provider is not yet implemented. "
            "Set VECTOR_DB_PROVIDER=chromadb to use ChromaDB."
        )
    elif provider == "qdrant":
        raise NotImplementedError(
            "Qdrant provider is not yet implemented. "
            "Set VECTOR_DB_PROVIDER=chromadb to use ChromaDB."
        )
    else:
        raise ValueError(
            f"Unsupported vector database provider: '{provider}'. "
            f"Supported providers: {', '.join(sorted(SUPPORTED_PROVIDERS))}"
        )
