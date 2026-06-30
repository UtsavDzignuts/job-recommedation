"""Unit tests for the POST /sync/full endpoint.

Tests the sync API route using FastAPI's TestClient with mocked dependencies.
"""

import pytest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.ai.models import SyncReport
from app.ai.routes.sync import _get_sync_service, router


@pytest.fixture
def app():
    """Create a FastAPI app with the sync router mounted."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def mock_sync_service():
    """Create a mocked SyncService that returns a valid SyncReport."""
    service = AsyncMock()
    service.full_resync = AsyncMock(
        return_value=SyncReport(
            total_entities=10,
            created=5,
            updated=3,
            deleted=0,
            failed=2,
            duration_seconds=1.234,
        )
    )
    return service


@pytest.mark.asyncio
async def test_full_sync_returns_sync_report(app, mock_sync_service):
    """POST /sync/full should return a valid SyncReport JSON response."""
    app.dependency_overrides[_get_sync_service] = lambda: mock_sync_service

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/sync/full")

    assert response.status_code == 200
    data = response.json()
    assert data["total_entities"] == 10
    assert data["created"] == 5
    assert data["updated"] == 3
    assert data["deleted"] == 0
    assert data["failed"] == 2
    assert data["duration_seconds"] == 1.234


@pytest.mark.asyncio
async def test_full_sync_is_idempotent(app, mock_sync_service):
    """Calling POST /sync/full multiple times should produce the same result."""
    app.dependency_overrides[_get_sync_service] = lambda: mock_sync_service

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response1 = await client.post("/sync/full")
        response2 = await client.post("/sync/full")

    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response1.json() == response2.json()


@pytest.mark.asyncio
async def test_full_sync_handles_service_error(app):
    """POST /sync/full should return 503 when the sync service fails."""
    failing_service = AsyncMock()
    failing_service.full_resync = AsyncMock(
        side_effect=RuntimeError("Vector DB connection failed")
    )
    app.dependency_overrides[_get_sync_service] = lambda: failing_service

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/sync/full")

    assert response.status_code == 503
    assert "Sync operation failed" in response.json()["detail"]


@pytest.mark.asyncio
async def test_full_sync_empty_database(app):
    """POST /sync/full with no entities should return zero counts."""
    service = AsyncMock()
    service.full_resync = AsyncMock(
        return_value=SyncReport(
            total_entities=0,
            created=0,
            updated=0,
            deleted=0,
            failed=0,
            duration_seconds=0.001,
        )
    )
    app.dependency_overrides[_get_sync_service] = lambda: service

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/sync/full")

    assert response.status_code == 200
    data = response.json()
    assert data["total_entities"] == 0
    assert data["created"] == 0
    assert data["updated"] == 0
    assert data["deleted"] == 0
    assert data["failed"] == 0
