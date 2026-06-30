"""Unit tests for AI Agent tools."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.tools import BaseTool

from app.ai.agent.tools import (
    create_api_query_tool,
    create_vector_search_tool,
    create_llm_reasoning_tool,
    api_query_tool,
    vector_search_tool,
    llm_reasoning_tool,
    _format_search_results,
)
from app.ai.config import AIConfig
from app.ai.models import VectorDocument


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


# ---------------------------------------------------------------------------
# Module-level tool instances
# ---------------------------------------------------------------------------


class TestModuleLevelTools:
    """Tests for the module-level tool instances."""

    def test_api_query_tool_is_langchain_tool(self):
        assert isinstance(api_query_tool, BaseTool)
        assert api_query_tool.name == "api_query"

    def test_vector_search_tool_is_langchain_tool(self):
        assert isinstance(vector_search_tool, BaseTool)
        assert vector_search_tool.name == "vector_search"

    def test_llm_reasoning_tool_is_langchain_tool(self):
        assert isinstance(llm_reasoning_tool, BaseTool)
        assert llm_reasoning_tool.name == "llm_reasoning"

    def test_tools_have_descriptions(self):
        for t in [api_query_tool, vector_search_tool, llm_reasoning_tool]:
            assert t.description is not None
            assert len(t.description) > 10


# ---------------------------------------------------------------------------
# api_query_tool
# ---------------------------------------------------------------------------


class TestApiQueryTool:
    """Tests for the api_query tool."""

    @pytest.mark.asyncio
    async def test_returns_error_on_connection_failure(self, config):
        tool = create_api_query_tool(config=config, base_url="http://invalid:9999")
        result = await tool.ainvoke("get all jobs")
        assert "Error querying the platform API" in result

    @pytest.mark.asyncio
    async def test_unrecognized_query_returns_guidance(self, config):
        tool = create_api_query_tool(config=config, base_url="http://invalid:9999")
        result = await tool.ainvoke("fetch something random")
        assert "Please specify what to query" in result

    @pytest.mark.asyncio
    async def test_successful_job_query(self, config):
        """Test successful API query using mocked httpx."""
        tool = create_api_query_tool(config=config, base_url="http://testserver")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [
            {"id": 1, "title": "Software Engineer", "company": "Acme"},
            {"id": 2, "title": "Data Scientist", "company": "BigCo"},
        ]

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await tool.ainvoke("get all jobs")

        assert "Found 2 result(s)" in result
        assert "Software Engineer" in result


# ---------------------------------------------------------------------------
# vector_search_tool
# ---------------------------------------------------------------------------


class TestVectorSearchTool:
    """Tests for the vector_search tool."""

    @pytest.mark.asyncio
    async def test_returns_formatted_results(self, config):
        mock_embedding_service = AsyncMock()
        mock_embedding_service.generate_embedding = AsyncMock(
            return_value=[0.1] * 1536
        )

        mock_vector_store = AsyncMock()
        mock_vector_store.search = AsyncMock(
            return_value=[
                VectorDocument(
                    id="job-1",
                    embedding=[0.1] * 1536,
                    metadata={"entity_type": "job_post", "entity_id": "1"},
                    text_snippet="Senior Python Developer at TechCorp",
                    score=0.92,
                ),
            ]
        )

        tool = create_vector_search_tool(
            config=config,
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
        )

        result = await tool.ainvoke("python developer")

        assert "job_post" in result
        assert "Senior Python Developer" in result
        assert "0.92" in result

    @pytest.mark.asyncio
    async def test_returns_no_results_message(self, config):
        mock_embedding_service = AsyncMock()
        mock_embedding_service.generate_embedding = AsyncMock(
            return_value=[0.1] * 1536
        )

        mock_vector_store = AsyncMock()
        mock_vector_store.search = AsyncMock(return_value=[])

        tool = create_vector_search_tool(
            config=config,
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
        )

        result = await tool.ainvoke("nonexistent query")
        assert "No results found" in result

    @pytest.mark.asyncio
    async def test_handles_embedding_error_gracefully(self, config):
        mock_embedding_service = AsyncMock()
        mock_embedding_service.generate_embedding = AsyncMock(
            side_effect=Exception("Embedding API down")
        )

        tool = create_vector_search_tool(
            config=config,
            embedding_service=mock_embedding_service,
            vector_store=AsyncMock(),
        )

        result = await tool.ainvoke("test query")
        assert "Error performing vector search" in result

    @pytest.mark.asyncio
    async def test_handles_partial_collection_failure(self, config):
        """Tool continues searching other collections if one fails."""
        mock_embedding_service = AsyncMock()
        mock_embedding_service.generate_embedding = AsyncMock(
            return_value=[0.1] * 1536
        )

        call_count = 0

        async def mock_search(collection, query_embedding, top_k, min_score):
            nonlocal call_count
            call_count += 1
            if collection == "job_posts":
                raise Exception("Collection unavailable")
            return [
                VectorDocument(
                    id=f"{collection}-1",
                    embedding=[0.1] * 1536,
                    metadata={"entity_type": collection, "entity_id": "1"},
                    text_snippet=f"Result from {collection}",
                    score=0.85,
                )
            ]

        mock_vector_store = AsyncMock()
        mock_vector_store.search = mock_search

        tool = create_vector_search_tool(
            config=config,
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
        )

        result = await tool.ainvoke("test query")
        # Should still return results from the other collections
        assert "companies" in result or "candidates" in result


# ---------------------------------------------------------------------------
# llm_reasoning_tool
# ---------------------------------------------------------------------------


class TestLlmReasoningTool:
    """Tests for the llm_reasoning tool."""

    @pytest.mark.asyncio
    async def test_returns_llm_response(self, config):
        tool = create_llm_reasoning_tool(config=config)

        mock_response = MagicMock()
        mock_response.content = "The analysis shows that candidate A is a better fit."

        with patch("app.ai.agent.tools.ChatOpenAI") as mock_llm_cls:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm_cls.return_value = mock_llm

            result = await tool.ainvoke("Analyze this candidate profile")

        assert "candidate A is a better fit" in result

    @pytest.mark.asyncio
    async def test_handles_llm_error_gracefully(self, config):
        tool = create_llm_reasoning_tool(config=config)

        with patch("app.ai.agent.tools.ChatOpenAI") as mock_llm_cls:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM service down"))
            mock_llm_cls.return_value = mock_llm

            result = await tool.ainvoke("Analyze something")

        assert "Error performing LLM reasoning" in result


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestFormatSearchResults:
    """Tests for _format_search_results helper."""

    def test_empty_results(self):
        assert _format_search_results([]) == "No results found."

    def test_single_result(self):
        results = [
            VectorDocument(
                id="doc-1",
                embedding=[],
                metadata={"entity_type": "job_post", "entity_id": "42"},
                text_snippet="Python developer needed",
                score=0.95,
            )
        ]
        output = _format_search_results(results)
        assert "entity_type=job_post" in output
        assert "entity_id=42" in output
        assert "0.9500" in output
        assert "Python developer needed" in output

    def test_truncates_long_snippets(self):
        long_text = "x" * 300
        results = [
            VectorDocument(
                id="doc-1",
                embedding=[],
                metadata={"entity_type": "company", "entity_id": "1"},
                text_snippet=long_text,
                score=0.8,
            )
        ]
        output = _format_search_results(results)
        # Should be truncated to 200 chars
        assert len(long_text) > 200
        assert "x" * 200 in output
        assert "x" * 201 not in output
