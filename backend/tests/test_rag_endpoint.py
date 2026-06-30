"""Unit tests for the GET /ask-ai endpoint.

Tests the RAG API route using httpx AsyncClient with mocked RAGEngine dependency.
"""

import pytest
from unittest.mock import AsyncMock

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.ai.exceptions import LLMServiceUnavailableError
from app.ai.models import AskAIResponse, SourceReference
from app.ai.routes.rag import _get_rag_engine, router


@pytest.fixture
def app():
    """Create a FastAPI app with the RAG router mounted."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def mock_rag_engine():
    """Create a mocked RAGEngine that returns a valid AskAIResponse."""
    engine = AsyncMock()
    engine.answer_query = AsyncMock(
        return_value=AskAIResponse(
            answer="Python is a high-level programming language.",
            sources=[
                SourceReference(
                    entity_type="job_post",
                    entity_id="123",
                    text_snippet="Python developer needed...",
                    relevance_score=0.92,
                )
            ],
            query="What is Python?",
        )
    )
    return engine


@pytest.mark.asyncio
async def test_ask_ai_returns_successful_response(app, mock_rag_engine):
    """GET /ask-ai?query=... should return 200 with AskAIResponse."""
    app.dependency_overrides[_get_rag_engine] = lambda: mock_rag_engine

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/ask-ai", params={"query": "What is Python?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Python is a high-level programming language."
    assert data["query"] == "What is Python?"
    assert len(data["sources"]) == 1
    assert data["sources"][0]["entity_type"] == "job_post"
    assert data["sources"][0]["entity_id"] == "123"
    assert data["sources"][0]["relevance_score"] == 0.92


@pytest.mark.asyncio
async def test_ask_ai_returns_400_when_query_missing(app, mock_rag_engine):
    """GET /ask-ai without query parameter should return 400."""
    app.dependency_overrides[_get_rag_engine] = lambda: mock_rag_engine

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/ask-ai")

    assert response.status_code == 400
    data = response.json()
    assert data["detail"]["error"] == "invalid_query"


@pytest.mark.asyncio
async def test_ask_ai_returns_400_when_query_empty(app, mock_rag_engine):
    """GET /ask-ai?query= (empty string) should return 400."""
    app.dependency_overrides[_get_rag_engine] = lambda: mock_rag_engine

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/ask-ai", params={"query": ""})

    assert response.status_code == 400
    data = response.json()
    assert data["detail"]["error"] == "invalid_query"


@pytest.mark.asyncio
async def test_ask_ai_returns_400_when_query_whitespace_only(app, mock_rag_engine):
    """GET /ask-ai?query=   (whitespace only) should return 400."""
    app.dependency_overrides[_get_rag_engine] = lambda: mock_rag_engine

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/ask-ai", params={"query": "   "})

    assert response.status_code == 400
    data = response.json()
    assert data["detail"]["error"] == "invalid_query"


@pytest.mark.asyncio
async def test_ask_ai_returns_400_when_query_exceeds_1000_chars(app, mock_rag_engine):
    """GET /ask-ai with query > 1000 characters should return 400."""
    app.dependency_overrides[_get_rag_engine] = lambda: mock_rag_engine
    long_query = "a" * 1001

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/ask-ai", params={"query": long_query})

    assert response.status_code == 400
    data = response.json()
    assert data["detail"]["error"] == "query_too_long"
    assert data["detail"]["details"]["max_length"] == 1000
    assert data["detail"]["details"]["actual_length"] == 1001


@pytest.mark.asyncio
async def test_ask_ai_accepts_query_at_exactly_1000_chars(app, mock_rag_engine):
    """GET /ask-ai with query of exactly 1000 characters should succeed."""
    app.dependency_overrides[_get_rag_engine] = lambda: mock_rag_engine
    exact_query = "a" * 1000
    mock_rag_engine.answer_query = AsyncMock(
        return_value=AskAIResponse(
            answer="Response to long query.",
            sources=[],
            query=exact_query,
        )
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/ask-ai", params={"query": exact_query})

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_ask_ai_returns_503_when_llm_unavailable(app):
    """GET /ask-ai should return 503 when LLM service is unavailable."""
    failing_engine = AsyncMock()
    failing_engine.answer_query = AsyncMock(
        side_effect=LLMServiceUnavailableError("OpenAI API is down")
    )
    app.dependency_overrides[_get_rag_engine] = lambda: failing_engine

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/ask-ai", params={"query": "What jobs are available?"})

    assert response.status_code == 503
    data = response.json()
    assert data["detail"]["error"] == "llm_unavailable"
    assert "unavailable" in data["detail"]["message"].lower()
