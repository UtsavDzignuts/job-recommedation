"""Vector Database abstraction layer.

Provides a unified interface for vector operations regardless of the
underlying provider (ChromaDB, PGvector, Qdrant).
"""

from abc import ABC, abstractmethod
from typing import List

from app.ai.models import VectorDocument


class VectorStoreInterface(ABC):
    """Abstract base class defining the vector store contract.

    All vector store implementations must provide async methods for
    upserting, searching, deleting documents, and health checking.
    """

    @abstractmethod
    async def upsert(self, collection: str, documents: List[VectorDocument]) -> None:
        """Insert or update documents in a collection.

        Args:
            collection: Name of the collection (e.g., 'job_posts', 'companies', 'candidates').
            documents: List of VectorDocument instances to upsert.

        Raises:
            VectorDBUnavailableError: If the vector database is unreachable.
        """
        ...

    @abstractmethod
    async def search(
        self,
        collection: str,
        query_embedding: List[float],
        top_k: int = 10,
        min_score: float = 0.7,
    ) -> List[VectorDocument]:
        """Perform similarity search against a collection.

        Args:
            collection: Name of the collection to search.
            query_embedding: The query vector to compare against stored embeddings.
            top_k: Maximum number of results to return.
            min_score: Minimum similarity score threshold for results.

        Returns:
            List of VectorDocument instances sorted by descending relevance score,
            containing at most top_k results with score >= min_score.

        Raises:
            VectorDBUnavailableError: If the vector database is unreachable.
        """
        ...

    @abstractmethod
    async def delete(self, collection: str, ids: List[str]) -> None:
        """Delete documents from a collection by their IDs.

        Args:
            collection: Name of the collection.
            ids: List of document IDs to delete.

        Raises:
            VectorDBUnavailableError: If the vector database is unreachable.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the vector database is reachable and healthy.

        Returns:
            True if the database is healthy and responsive, False otherwise.
        """
        ...
