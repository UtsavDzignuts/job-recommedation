"""Unit tests for RAGEngine."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.ai.circuit_breaker import CircuitBreaker
from app.ai.config import AIConfig
from app.ai.embedding_service import EmbeddingService
from app.ai.exceptions import LLMServiceUnavailableError
from app.ai.models import AskAIResponse, VectorDocument
from app.ai.prompt_manager import PromptTemplateManager
from app.ai.rag_engine import RAGEngine
from app.ai.vectorstore import VectorStoreInterface


@pytest.fixture
def config():
    """Create an AIConfig with test values."""
    return AIConfig(
        OPENAI_API_KEY="test-api-key",
        OPENAI_CHAT_MODEL="gpt-4o-mini",
        OPENAI_EMBEDDING_MODEL="text-embedding-ada-002",
        RAG_TOP_K=10,
        RAG_MIN_RELEVANCE_THRESHOLD=0.7,
    )


@pytest.fixture
def embedding_service():
    """Create a mocked EmbeddingService."""
    service = AsyncMock(spec=EmbeddingService)
    service.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return service


@pytest.fixture
def vector_store():
    """Create a mocked VectorStoreInterface."""
    store = AsyncMock(spec=VectorStoreInterface)
    store.search = AsyncMock(return_value=[])
    return store


@pytest.fixture
def prompt_manager():
    """Create a mocked PromptTemplateManager."""
    manager = MagicMock(spec=PromptTemplateManager)
    manager.render = MagicMock(return_value="Rendered prompt content")
    return manager


@pytest.fixture
def circuit_breaker():
    """Create a CircuitBreaker that passes through calls."""
    cb = CircuitBreaker(
        failure_threshold=5,
        window_seconds=60.0,
        cooldown_seconds=30.0,
    )
    return cb


@pytest.fixture
def rag_engine(embedding_service, vector_store, prompt_manager, circuit_breaker, config):
    """Create a RAGEngine with mocked dependencies."""
    engine = RAGEngine(
        embedding_service=embedding_service,
        vector_store=vector_store,
        prompt_manager=prompt_manager,
        circuit_breaker=circuit_breaker,
        config=config,
    )
    # Replace _invoke_llm with an async mock for testing
    engine._invoke_llm = AsyncMock(return_value="Mocked LLM answer")
    return engine


def _make_vector_doc(
    doc_id: str,
    entity_type: str = "job_post",
    entity_id: str = "123",
    text_snippet: str = "Sample text",
    score: float = 0.85,
) -> VectorDocument:
    """Create a VectorDocument for testing."""
    return VectorDocument(
        id=doc_id,
        embedding=[0.1, 0.2, 0.3],
        metadata={"entity_type": entity_type, "entity_id": entity_id},
        text_snippet=text_snippet,
        score=score,
    )


class TestAnswerQuery:
    @pytest.mark.asyncio
    async def test_returns_no_info_when_no_documents_found(self, rag_engine, vector_store):
        """When no documents are above threshold, returns 'no relevant info' response."""
        vector_store.search = AsyncMock(return_value=[])

        result = await rag_engine.answer_query("What jobs are available?")

        assert isinstance(result, AskAIResponse)
        assert result.answer == "No relevant information found for your query."
        assert result.sources == []
        assert result.query == "What jobs are available?"

    @pytest.mark.asyncio
    async def test_returns_answer_with_sources_when_documents_found(
        self, rag_engine, vector_store, prompt_manager
    ):
        """When documents are found, returns LLM answer with sources."""
        job_docs = [
            _make_vector_doc("doc1", "job_post", "jp-1", "Senior Python Dev", 0.9),
        ]
        company_docs = [
            _make_vector_doc("doc2", "company", "co-1", "Tech Corp", 0.8),
        ]

        async def mock_search(collection, **kwargs):
            if collection == "job_posts":
                return job_docs
            elif collection == "companies":
                return company_docs
            return []

        vector_store.search = AsyncMock(side_effect=mock_search)
        rag_engine._invoke_llm = AsyncMock(
            return_value="There are Python developer positions available."
        )

        result = await rag_engine.answer_query("What Python jobs exist?")

        assert isinstance(result, AskAIResponse)
        assert result.answer == "There are Python developer positions available."
        assert result.query == "What Python jobs exist?"
        assert len(result.sources) == 2
        assert result.sources[0].entity_type == "job_post"
        assert result.sources[0].entity_id == "jp-1"
        assert result.sources[0].relevance_score == 0.9
        assert result.sources[1].entity_type == "company"
        assert result.sources[1].entity_id == "co-1"

    @pytest.mark.asyncio
    async def test_searches_all_collections(self, rag_engine, vector_store):
        """Verifies similarity search is performed across all 3 collections."""
        vector_store.search = AsyncMock(return_value=[])

        await rag_engine.answer_query("test query")

        assert vector_store.search.call_count == 3
        collections_searched = [
            call.kwargs["collection"] for call in vector_store.search.call_args_list
        ]
        assert "job_posts" in collections_searched
        assert "companies" in collections_searched
        assert "candidates" in collections_searched

    @pytest.mark.asyncio
    async def test_uses_config_top_k_and_min_score(self, rag_engine, vector_store):
        """Verifies config values are passed to similarity search."""
        vector_store.search = AsyncMock(return_value=[])

        await rag_engine.answer_query("test query")

        for call in vector_store.search.call_args_list:
            assert call.kwargs["top_k"] == 10
            assert call.kwargs["min_score"] == 0.7

    @pytest.mark.asyncio
    async def test_renders_rag_answer_template(
        self, rag_engine, vector_store, prompt_manager
    ):
        """Verifies the rag_answer template is rendered with query and context."""
        docs = [_make_vector_doc("doc1", "job_post", "jp-1", "Dev role", 0.85)]
        vector_store.search = AsyncMock(return_value=docs)

        await rag_engine.answer_query("Find dev jobs")

        prompt_manager.render.assert_called_once()
        call_args = prompt_manager.render.call_args
        assert call_args.args[0] == "rag_answer"
        assert call_args.kwargs["query"] == "Find dev jobs"
        assert "Dev role" in call_args.kwargs["context_documents"]

    @pytest.mark.asyncio
    async def test_raises_llm_unavailable_when_circuit_breaker_open(
        self, rag_engine, vector_store, circuit_breaker
    ):
        """When circuit breaker is open, raises LLMServiceUnavailableError."""
        import time
        from app.ai.circuit_breaker import CircuitState

        docs = [_make_vector_doc("doc1", "job_post", "jp-1", "Dev role", 0.85)]
        vector_store.search = AsyncMock(return_value=docs)

        # Force the circuit breaker to open state
        circuit_breaker._state = CircuitState.OPEN
        circuit_breaker._opened_at = time.monotonic()

        with pytest.raises(LLMServiceUnavailableError):
            await rag_engine.answer_query("test query")

    @pytest.mark.asyncio
    async def test_raises_llm_unavailable_on_llm_api_error(
        self, rag_engine, vector_store
    ):
        """When LLM call fails, raises LLMServiceUnavailableError."""
        docs = [_make_vector_doc("doc1", "job_post", "jp-1", "Dev role", 0.85)]
        vector_store.search = AsyncMock(return_value=docs)
        rag_engine._invoke_llm = AsyncMock(
            side_effect=Exception("API connection error")
        )

        with pytest.raises(LLMServiceUnavailableError):
            await rag_engine.answer_query("test query")

    @pytest.mark.asyncio
    async def test_results_sorted_by_score_descending(
        self, rag_engine, vector_store, prompt_manager
    ):
        """Documents from multiple collections are merged and sorted by score."""
        # Return docs from different collections with varying scores
        async def mock_search(collection, **kwargs):
            if collection == "job_posts":
                return [_make_vector_doc("jp1", "job_post", "jp-1", "Job A", 0.75)]
            elif collection == "companies":
                return [_make_vector_doc("co1", "company", "co-1", "Company B", 0.95)]
            else:
                return [_make_vector_doc("ca1", "candidate", "ca-1", "Candidate C", 0.80)]

        vector_store.search = AsyncMock(side_effect=mock_search)

        result = await rag_engine.answer_query("test query")

        # Sources should be sorted by descending score
        assert result.sources[0].relevance_score == 0.95
        assert result.sources[1].relevance_score == 0.80
        assert result.sources[2].relevance_score == 0.75

    @pytest.mark.asyncio
    async def test_limits_results_to_top_k(self, rag_engine, vector_store, config):
        """Total documents across collections are capped at top_k."""
        # Return 5 docs from each collection (15 total), but top_k is 10
        docs_per_collection = [
            _make_vector_doc(f"doc{i}", "job_post", f"id-{i}", f"text {i}", 0.9 - i * 0.01)
            for i in range(5)
        ]
        vector_store.search = AsyncMock(return_value=docs_per_collection)

        result = await rag_engine.answer_query("test query")

        assert len(result.sources) <= config.RAG_TOP_K

    @pytest.mark.asyncio
    async def test_generates_embedding_for_query(self, rag_engine, embedding_service):
        """Verifies the query is embedded before search."""
        await rag_engine.answer_query("What Python jobs exist?")

        embedding_service.generate_embedding.assert_called_once_with(
            "What Python jobs exist?"
        )
