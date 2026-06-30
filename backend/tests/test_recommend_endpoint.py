"""Unit tests for the POST /recommend endpoint.

Tests the recommend API route using FastAPI's TestClient with mocked dependencies.
"""

import pytest
from unittest.mock import AsyncMock

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.ai.exceptions import LLMServiceUnavailableError
from app.ai.models import (
    JobRecommendation,
    RecommendationResponse,
)
from app.ai.routes.recommend import _get_recommendation_engine, router


@pytest.fixture
def app():
    """Create a FastAPI app with the recommend router mounted."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def mock_recommendation_engine():
    """Create a mocked RecommendationEngine that returns valid recommendations."""
    engine = AsyncMock()
    engine.recommend_jobs = AsyncMock(
        return_value=RecommendationResponse(
            recommendations=[
                JobRecommendation(
                    job_title="Senior Python Developer",
                    job_id="job-123",
                    match_reason="Strong match based on Python expertise and backend experience.",
                    confidence_score=0.92,
                ),
                JobRecommendation(
                    job_title="Data Engineer",
                    job_id="job-456",
                    match_reason="Experience with data pipelines aligns well.",
                    confidence_score=0.85,
                ),
            ],
            message=None,
        )
    )
    return engine


@pytest.mark.asyncio
async def test_recommend_returns_recommendations(app, mock_recommendation_engine):
    """POST /recommend with valid resume text should return recommendations."""
    app.dependency_overrides[_get_recommendation_engine] = (
        lambda: mock_recommendation_engine
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/recommend",
            json={"resume_text": "Experienced Python developer with 5 years of backend development."},
        )

    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert len(data["recommendations"]) == 2
    assert data["recommendations"][0]["job_title"] == "Senior Python Developer"
    assert data["recommendations"][0]["confidence_score"] == 0.92


@pytest.mark.asyncio
async def test_recommend_empty_resume_text_returns_422(app, mock_recommendation_engine):
    """POST /recommend with empty resume_text should return 422."""
    app.dependency_overrides[_get_recommendation_engine] = (
        lambda: mock_recommendation_engine
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/recommend",
            json={"resume_text": ""},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_recommend_missing_resume_text_returns_422(app, mock_recommendation_engine):
    """POST /recommend with missing resume_text field should return 422."""
    app.dependency_overrides[_get_recommendation_engine] = (
        lambda: mock_recommendation_engine
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/recommend",
            json={},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_recommend_exceeds_max_length_returns_422(app, mock_recommendation_engine):
    """POST /recommend with resume_text exceeding 10000 chars should return 422."""
    app.dependency_overrides[_get_recommendation_engine] = (
        lambda: mock_recommendation_engine
    )

    long_text = "x" * 10_001

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/recommend",
            json={"resume_text": long_text},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_recommend_exactly_max_length_returns_200(app, mock_recommendation_engine):
    """POST /recommend with resume_text at exactly 10000 chars should succeed."""
    app.dependency_overrides[_get_recommendation_engine] = (
        lambda: mock_recommendation_engine
    )

    max_text = "a" * 10_000

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/recommend",
            json={"resume_text": max_text},
        )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_recommend_single_char_returns_200(app, mock_recommendation_engine):
    """POST /recommend with a single character resume_text should succeed."""
    app.dependency_overrides[_get_recommendation_engine] = (
        lambda: mock_recommendation_engine
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/recommend",
            json={"resume_text": "x"},
        )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_recommend_llm_unavailable_returns_503(app):
    """POST /recommend should return 503 when LLM service is unavailable."""
    failing_engine = AsyncMock()
    failing_engine.recommend_jobs = AsyncMock(
        side_effect=LLMServiceUnavailableError("LLM service is temporarily unavailable")
    )
    app.dependency_overrides[_get_recommendation_engine] = lambda: failing_engine

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/recommend",
            json={"resume_text": "Experienced developer looking for new opportunities."},
        )

    assert response.status_code == 503
    data = response.json()
    assert data["error"] == "llm_service_unavailable"
    assert "unavailable" in data["message"].lower()


@pytest.mark.asyncio
async def test_recommend_no_matches_returns_empty_list(app):
    """POST /recommend should return empty recommendations when no matches found."""
    engine = AsyncMock()
    engine.recommend_jobs = AsyncMock(
        return_value=RecommendationResponse(
            recommendations=[],
            message="No relevant jobs found for the given profile.",
        )
    )
    app.dependency_overrides[_get_recommendation_engine] = lambda: engine

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/recommend",
            json={"resume_text": "Very niche skill set with no matching jobs."},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["recommendations"] == []
    assert data["message"] == "No relevant jobs found for the given profile."
