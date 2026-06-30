"""Unit tests for RecommendationEngine."""

import json

import pytest
from unittest.mock import AsyncMock, patch

from app.ai.circuit_breaker import CircuitBreaker
from app.ai.config import AIConfig
from app.ai.embedding_service import EmbeddingService
from app.ai.exceptions import LLMServiceUnavailableError
from app.ai.models import VectorDocument
from app.ai.prompt_manager import PromptTemplateManager
from app.ai.recommendation_engine import RecommendationEngine
from app.ai.vectorstore import VectorStoreInterface


@pytest.fixture
def config():
    """Create an AIConfig with test values."""
    return AIConfig(
        OPENAI_API_KEY="test-api-key",
        OPENAI_CHAT_MODEL="gpt-4o-mini",
        OPENAI_EMBEDDING_MODEL="text-embedding-ada-002",
        PROMPT_TEMPLATES_DIR="app/prompts",
        RECOMMEND_TOP_K=10,
        RECOMMEND_MAX_RESULTS=5,
        RAG_MIN_RELEVANCE_THRESHOLD=0.7,
    )


@pytest.fixture
def prompt_manager():
    """Create a real PromptTemplateManager pointing to prompt templates."""
    return PromptTemplateManager(templates_dir="app/prompts")


@pytest.fixture
def circuit_breaker():
    """Create a CircuitBreaker for testing."""
    return CircuitBreaker(
        failure_threshold=5,
        window_seconds=60.0,
        cooldown_seconds=30.0,
    )


@pytest.fixture
def mock_embedding_service():
    """Create a mock EmbeddingService."""
    service = AsyncMock(spec=EmbeddingService)
    service.generate_embedding = AsyncMock(return_value=[0.1] * 1536)
    return service


@pytest.fixture
def mock_vector_store():
    """Create a mock VectorStoreInterface."""
    store = AsyncMock(spec=VectorStoreInterface)
    store.search = AsyncMock(return_value=[])
    return store


@pytest.fixture
def engine(config, mock_embedding_service, mock_vector_store, prompt_manager, circuit_breaker):
    """Create a RecommendationEngine with mocked dependencies."""
    return RecommendationEngine(
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
        prompt_manager=prompt_manager,
        circuit_breaker=circuit_breaker,
        config=config,
    )


def _make_vector_doc(doc_id: str, snippet: str, score: float) -> VectorDocument:
    """Helper to create a VectorDocument for tests."""
    return VectorDocument(
        id=doc_id,
        embedding=[0.1] * 1536,
        metadata={"title": f"Job {doc_id}", "entity_type": "job_post"},
        text_snippet=snippet,
        score=score,
    )


class TestRecommendJobsNoMatches:
    """Tests for when no job matches are found above threshold."""

    @pytest.mark.asyncio
    async def test_returns_empty_list_with_message_when_no_matches(
        self, engine, mock_vector_store
    ):
        """If similarity search returns no results, return empty recommendations with message."""
        mock_vector_store.search.return_value = []

        result = await engine.recommend_jobs("Software engineer with 5 years experience")

        assert result.recommendations == []
        assert result.message == "No relevant jobs found for the given profile."

    @pytest.mark.asyncio
    async def test_searches_job_posts_collection(
        self, engine, mock_vector_store, mock_embedding_service
    ):
        """Verify the search targets 'job_posts' collection with correct params."""
        mock_vector_store.search.return_value = []

        await engine.recommend_jobs("Data scientist resume text")

        mock_vector_store.search.assert_called_once_with(
            collection="job_posts",
            query_embedding=[0.1] * 1536,
            top_k=10,
            min_score=0.7,
        )


class TestRecommendJobsWithMatches:
    """Tests for when job matches are found and LLM produces valid JSON."""

    @pytest.mark.asyncio
    async def test_returns_parsed_recommendations_from_llm(
        self, engine, mock_vector_store
    ):
        """When LLM returns valid JSON, parse into JobRecommendation items."""
        mock_vector_store.search.return_value = [
            _make_vector_doc("job-1", "Senior Python Developer at TechCorp", 0.92),
            _make_vector_doc("job-2", "Backend Engineer at StartupInc", 0.85),
        ]

        llm_response = json.dumps([
            {
                "job_title": "Senior Python Developer",
                "job_id": "job-1",
                "match_reason": "Strong Python skills match",
                "confidence_score": 0.95,
            },
            {
                "job_title": "Backend Engineer",
                "job_id": "job-2",
                "match_reason": "Backend experience aligns well",
                "confidence_score": 0.82,
            },
        ])

        with patch.object(engine, "_invoke_llm", new=AsyncMock(return_value=llm_response)):
            result = await engine.recommend_jobs("Python developer with backend experience")

        assert len(result.recommendations) == 2
        assert result.recommendations[0].job_title == "Senior Python Developer"
        assert result.recommendations[0].job_id == "job-1"
        assert result.recommendations[0].confidence_score == 0.95
        assert result.recommendations[1].confidence_score == 0.82
        assert result.message is None

    @pytest.mark.asyncio
    async def test_caps_recommendations_at_max_results(
        self, engine, mock_vector_store, config
    ):
        """Recommendations are capped at RECOMMEND_MAX_RESULTS (5)."""
        mock_vector_store.search.return_value = [
            _make_vector_doc(f"job-{i}", f"Job {i} description", 0.9 - i * 0.02)
            for i in range(8)
        ]

        # LLM returns 8 recommendations
        llm_response = json.dumps([
            {
                "job_title": f"Job {i}",
                "job_id": f"job-{i}",
                "match_reason": f"Reason {i}",
                "confidence_score": 0.9 - i * 0.05,
            }
            for i in range(8)
        ])

        with patch.object(engine, "_invoke_llm", new=AsyncMock(return_value=llm_response)):
            result = await engine.recommend_jobs("Resume text here")

        assert len(result.recommendations) <= config.RECOMMEND_MAX_RESULTS

    @pytest.mark.asyncio
    async def test_clamps_confidence_score_to_valid_range(
        self, engine, mock_vector_store
    ):
        """Confidence scores outside [0.0, 1.0] are clamped."""
        mock_vector_store.search.return_value = [
            _make_vector_doc("job-1", "Job description", 0.9),
        ]

        llm_response = json.dumps([
            {
                "job_title": "Over-confident Job",
                "job_id": "job-1",
                "match_reason": "Great match",
                "confidence_score": 1.5,
            },
            {
                "job_title": "Under-confident Job",
                "job_id": "job-2",
                "match_reason": "Weak match",
                "confidence_score": -0.3,
            },
        ])

        with patch.object(engine, "_invoke_llm", new=AsyncMock(return_value=llm_response)):
            result = await engine.recommend_jobs("Resume text")

        assert result.recommendations[0].confidence_score == 1.0
        assert result.recommendations[1].confidence_score == 0.0


class TestRecommendJobsLLMFallback:
    """Tests for fallback behavior when LLM response is invalid."""

    @pytest.mark.asyncio
    async def test_fallback_on_invalid_json(self, engine, mock_vector_store):
        """When LLM returns non-JSON, fallback to raw match recommendations."""
        mock_vector_store.search.return_value = [
            _make_vector_doc("job-1", "Senior Developer role", 0.88),
            _make_vector_doc("job-2", "Junior Developer role", 0.75),
        ]

        with patch.object(
            engine, "_invoke_llm", new=AsyncMock(return_value="Not valid JSON at all")
        ):
            result = await engine.recommend_jobs("A developer resume")

        assert len(result.recommendations) == 2
        assert result.recommendations[0].job_id == "job-1"
        assert result.recommendations[0].match_reason == "Matched based on profile similarity."

    @pytest.mark.asyncio
    async def test_fallback_on_non_array_json(self, engine, mock_vector_store):
        """When LLM returns JSON object (not array), fallback to raw matches."""
        mock_vector_store.search.return_value = [
            _make_vector_doc("job-1", "A job posting", 0.80),
        ]

        with patch.object(
            engine, "_invoke_llm", new=AsyncMock(return_value='{"not": "an array"}')
        ):
            result = await engine.recommend_jobs("Resume text")

        assert len(result.recommendations) == 1
        assert result.recommendations[0].job_id == "job-1"


class TestRecommendJobsLLMErrors:
    """Tests for LLM service errors during recommendation."""

    @pytest.mark.asyncio
    async def test_raises_llm_unavailable_when_circuit_breaker_open(
        self, config, mock_embedding_service, mock_vector_store, prompt_manager
    ):
        """LLMServiceUnavailableError raised when circuit breaker is open."""
        cb = CircuitBreaker(failure_threshold=1, window_seconds=60.0, cooldown_seconds=30.0)
        engine = RecommendationEngine(
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
            prompt_manager=prompt_manager,
            circuit_breaker=cb,
            config=config,
        )

        mock_vector_store.search.return_value = [
            _make_vector_doc("job-1", "A job", 0.9),
        ]

        # Trip the circuit breaker
        async def failing_call():
            raise RuntimeError("LLM down")

        with pytest.raises(RuntimeError):
            await cb.call(failing_call)

        # Now the circuit should be open
        with pytest.raises(LLMServiceUnavailableError):
            await engine.recommend_jobs("Resume text")

    @pytest.mark.asyncio
    async def test_raises_llm_unavailable_on_llm_failure(
        self, engine, mock_vector_store
    ):
        """LLMServiceUnavailableError raised when LLM call itself fails."""
        mock_vector_store.search.return_value = [
            _make_vector_doc("job-1", "A job", 0.9),
        ]

        with patch.object(
            engine, "_invoke_llm", new=AsyncMock(side_effect=RuntimeError("API error"))
        ):
            with pytest.raises(LLMServiceUnavailableError):
                await engine.recommend_jobs("Resume text")


class TestBuildJobMatchesText:
    """Tests for the _build_job_matches_text helper method."""

    def test_formats_matches_with_id_score_and_snippet(self, engine):
        """Verify formatting includes job ID, score, and description."""
        matches = [
            _make_vector_doc("abc-123", "Full stack developer needed", 0.91),
            _make_vector_doc("def-456", "Data engineer position", 0.84),
        ]

        result = engine._build_job_matches_text(matches)

        assert "abc-123" in result
        assert "0.910" in result
        assert "Full stack developer needed" in result
        assert "def-456" in result
        assert "0.840" in result
        assert "Data engineer position" in result
