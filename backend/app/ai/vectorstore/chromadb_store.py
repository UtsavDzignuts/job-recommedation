"""ChromaDB implementation of the VectorStoreInterface."""

import logging
from typing import List
from urllib.parse import urlparse

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.ai.exceptions import VectorDBUnavailableError
from app.ai.models import VectorDocument
from app.ai.vectorstore import VectorStoreInterface

logger = logging.getLogger(__name__)


class ChromaDBStore(VectorStoreInterface):
    """Vector store implementation backed by ChromaDB.

    Supports both HTTP client (for remote ChromaDB server) and
    in-memory/persistent client for local development and testing.
    """

    def __init__(self, url: str = "http://localhost:8000", api_key: str | None = None) -> None:
        """Initialize ChromaDB client.

        Args:
            url: ChromaDB server URL (used for HTTP client mode).
            api_key: Optional API key for ChromaDB authentication.
        """
        self._url = url
        self._api_key = api_key
        self._client: chromadb.ClientAPI | None = None

    def _get_client(self) -> chromadb.ClientAPI:
        """Lazily initialize and return the ChromaDB client.

        Returns:
            A ChromaDB client instance.

        Raises:
            VectorDBUnavailableError: If the client cannot connect.
        """
        if self._client is None:
            try:
                parsed = urlparse(self._url)
                host = parsed.hostname or "localhost"
                port = parsed.port or 8000
                self._client = chromadb.HttpClient(
                    host=host,
                    port=port,
                    settings=ChromaSettings(
                        anonymized_telemetry=False,
                    ),
                )
            except Exception as e:
                logger.error("Failed to initialize ChromaDB client: %s", str(e))
                raise VectorDBUnavailableError(
                    f"Failed to connect to ChromaDB at {self._url}: {e}"
                ) from e
        return self._client

    def _get_collection(self, collection: str) -> chromadb.Collection:
        """Get or create a ChromaDB collection.

        Args:
            collection: Name of the collection.

        Returns:
            A ChromaDB Collection instance.

        Raises:
            VectorDBUnavailableError: If the operation fails due to connection issues.
        """
        try:
            client = self._get_client()
            return client.get_or_create_collection(
                name=collection,
                metadata={"hnsw:space": "cosine"},
            )
        except VectorDBUnavailableError:
            raise
        except Exception as e:
            logger.error("Failed to get/create collection '%s': %s", collection, str(e))
            raise VectorDBUnavailableError(
                f"Failed to access collection '{collection}': {e}"
            ) from e

    async def upsert(self, collection: str, documents: List[VectorDocument]) -> None:
        """Insert or update documents in a ChromaDB collection.

        Args:
            collection: Name of the collection.
            documents: List of VectorDocument instances to upsert.

        Raises:
            VectorDBUnavailableError: If ChromaDB is unreachable.
        """
        if not documents:
            return

        try:
            coll = self._get_collection(collection)
            coll.upsert(
                ids=[doc.id for doc in documents],
                embeddings=[doc.embedding for doc in documents],
                metadatas=[doc.metadata for doc in documents],
                documents=[doc.text_snippet for doc in documents],
            )
            logger.debug(
                "Upserted %d documents into collection '%s'",
                len(documents),
                collection,
            )
        except VectorDBUnavailableError:
            raise
        except Exception as e:
            logger.error(
                "Failed to upsert %d documents into collection '%s': %s",
                len(documents),
                collection,
                str(e),
            )
            raise VectorDBUnavailableError(
                f"Failed to upsert documents into '{collection}': {e}"
            ) from e

    async def search(
        self,
        collection: str,
        query_embedding: List[float],
        top_k: int = 10,
        min_score: float = 0.7,
    ) -> List[VectorDocument]:
        """Perform similarity search against a ChromaDB collection.

        ChromaDB uses cosine distance internally, so we convert distance to
        similarity score (1 - distance) for filtering and ranking.

        Args:
            collection: Name of the collection to search.
            query_embedding: The query vector.
            top_k: Maximum number of results to return.
            min_score: Minimum similarity score threshold.

        Returns:
            List of VectorDocument instances sorted by descending score.

        Raises:
            VectorDBUnavailableError: If ChromaDB is unreachable.
        """
        try:
            coll = self._get_collection(collection)
            results = coll.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["embeddings", "metadatas", "documents", "distances"],
            )
        except VectorDBUnavailableError:
            raise
        except Exception as e:
            logger.error(
                "Failed to search collection '%s': %s", collection, str(e)
            )
            raise VectorDBUnavailableError(
                f"Failed to search collection '{collection}': {e}"
            ) from e

        documents: List[VectorDocument] = []

        if not results or not results.get("ids") or not results["ids"][0]:
            return documents

        ids = results["ids"][0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        embeddings = results.get("embeddings", [[]])[0]
        doc_texts = results.get("documents", [[]])[0]

        for i, doc_id in enumerate(ids):
            # ChromaDB returns cosine distance; convert to similarity score
            distance = distances[i] if i < len(distances) else 1.0
            score = 1.0 - distance

            if score < min_score:
                continue

            embedding = embeddings[i] if embeddings is not None and i < len(embeddings) else []
            metadata = metadatas[i] if metadatas is not None and i < len(metadatas) else {}
            text_snippet = doc_texts[i] if doc_texts is not None and i < len(doc_texts) else ""

            documents.append(
                VectorDocument(
                    id=doc_id,
                    embedding=list(embedding) if embedding is not None else [],
                    metadata=metadata if metadata else {},
                    text_snippet=text_snippet if text_snippet else "",
                    score=score,
                )
            )

        # Sort by score descending
        documents.sort(key=lambda d: d.score or 0.0, reverse=True)
        return documents[:top_k]

    async def delete(self, collection: str, ids: List[str]) -> None:
        """Delete documents from a ChromaDB collection by their IDs.

        Args:
            collection: Name of the collection.
            ids: List of document IDs to delete.

        Raises:
            VectorDBUnavailableError: If ChromaDB is unreachable.
        """
        if not ids:
            return

        try:
            coll = self._get_collection(collection)
            coll.delete(ids=ids)
            logger.debug(
                "Deleted %d documents from collection '%s'",
                len(ids),
                collection,
            )
        except VectorDBUnavailableError:
            raise
        except Exception as e:
            logger.error(
                "Failed to delete %d documents from collection '%s': %s",
                len(ids),
                collection,
                str(e),
            )
            raise VectorDBUnavailableError(
                f"Failed to delete documents from '{collection}': {e}"
            ) from e

    async def health_check(self) -> bool:
        """Check if ChromaDB is reachable by calling heartbeat.

        Returns:
            True if ChromaDB responds to heartbeat, False otherwise.
        """
        try:
            client = self._get_client()
            client.heartbeat()
            return True
        except Exception as e:
            logger.warning("ChromaDB health check failed: %s", str(e))
            return False
