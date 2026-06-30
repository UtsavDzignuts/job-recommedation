"""Unit tests for the POST /improve-description endpoint.

Tests the improve description API route using FastAPI's TestClient with
mocked dependencies. Covers success, validation errors, invalid mode, and
LLM unavailability scenarios.
"""

import pytest
from unittest.mock import AsyncMock

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.ai.exceptions import LLMServiceUnavailableError
from app.ai.models import ImprovementMode
from app.ai.routes.improve import _get_description_improver, router


@pytest.fixture
def app():
    """Create a FastAPI app with the improve router mounted."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def mock_improver():
    """Create a mocked DescriptionImprover that returns improved text."""
    improver = AsyncMock()
    improver.improve = AsyncMock(return_value="Improved job description text.")
    return improver


@pytest.mark.asyncio
async def test_improve_description_success(app, mock_improver):
    """POST /improve-description returns 200 with improved text on success."""
    app.dependency_overrides[_get_description_improver] = lambda: mock_improver

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/improve-description",
            json={
                "description": "A raw job description for a software engineer.",
                "mode": "short_and_crisp",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["improved_description"] == "Improved job description text."
    assert data["mode"] == "short_and_crisp"
    mock_improver.improve.assert_called_once_with(
        "A raw job description for a software engineer.",
        ImprovementMode.SHORT_AND_CRISP,
    )


@pytest.mark.asyncio
async def test_improve_description_detailed_mode(app, mock_improver):
    """POST /improve-description works with detailed_and_formal mode."""
    app.dependency_overrides[_get_description_improver] = lambda: mock_improver

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/improve-description",
            json={
                "description": "Some description text.",
                "mode": "detailed_and_formal",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "detailed_and_formal"


@pytest.mark.asyncio
async def test_improve_description_marketing_mode(app, mock_improver):
    """POST /improve-description works with marketing_oriented mode."""
    app.dependency_overrides[_get_description_improver] = lambda: mock_improver

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/improve-description",
            json={
                "description": "Some description text.",
                "mode": "marketing_oriented",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "marketing_oriented"


@pytest.mark.asyncio
async def test_improve_description_empty_description(app, mock_improver):
    """POST /improve-description returns 422 when description is empty."""
    app.dependency_overrides[_get_description_improver] = lambda: mock_improver

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/improve-description",
            json={
                "description": "",
                "mode": "short_and_crisp",
            },
        )

    assert response.status_code == 422
    mock_improver.improve.assert_not_called()


@pytest.mark.asyncio
async def test_improve_description_exceeds_max_length(app, mock_improver):
    """POST /improve-description returns 422 when description exceeds 50000 chars."""
    app.dependency_overrides[_get_description_improver] = lambda: mock_improver

    long_description = "x" * 50_001

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/improve-description",
            json={
                "description": long_description,
                "mode": "short_and_crisp",
            },
        )

    assert response.status_code == 422
    mock_improver.improve.assert_not_called()


@pytest.mark.asyncio
async def test_improve_description_missing_description(app, mock_improver):
    """POST /improve-description returns 422 when description field is missing."""
    app.dependency_overrides[_get_description_improver] = lambda: mock_improver

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/improve-description",
            json={
                "mode": "short_and_crisp",
            },
        )

    assert response.status_code == 422
    mock_improver.improve.assert_not_called()


@pytest.mark.asyncio
async def test_improve_description_invalid_mode(app, mock_improver):
    """POST /improve-description returns 422 when mode is not supported."""
    app.dependency_overrides[_get_description_improver] = lambda: mock_improver

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/improve-description",
            json={
                "description": "A valid description.",
                "mode": "invalid_mode",
            },
        )

    assert response.status_code == 422
    mock_improver.improve.assert_not_called()


@pytest.mark.asyncio
async def test_improve_description_llm_unavailable(app):
    """POST /improve-description returns 503 when LLM service is unavailable."""
    failing_improver = AsyncMock()
    failing_improver.improve = AsyncMock(
        side_effect=LLMServiceUnavailableError("LLM service is temporarily unavailable")
    )
    app.dependency_overrides[_get_description_improver] = lambda: failing_improver

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/improve-description",
            json={
                "description": "A valid job description.",
                "mode": "short_and_crisp",
            },
        )

    assert response.status_code == 503
    data = response.json()
    assert data["error"] == "llm_service_unavailable"
    assert "temporarily unavailable" in data["message"]


@pytest.mark.asyncio
async def test_improve_description_missing_mode(app, mock_improver):
    """POST /improve-description returns 422 when mode field is missing."""
    app.dependency_overrides[_get_description_improver] = lambda: mock_improver

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/improve-description",
            json={
                "description": "A valid description.",
            },
        )

    assert response.status_code == 422
    mock_improver.improve.assert_not_called()


@pytest.mark.asyncio
async def test_improve_description_at_max_length(app, mock_improver):
    """POST /improve-description returns 200 when description is exactly 50000 chars."""
    app.dependency_overrides[_get_description_improver] = lambda: mock_improver

    description_at_limit = "x" * 50_000

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/improve-description",
            json={
                "description": description_at_limit,
                "mode": "short_and_crisp",
            },
        )

    assert response.status_code == 200
