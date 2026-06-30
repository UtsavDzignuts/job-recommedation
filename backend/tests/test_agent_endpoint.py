"""Unit tests for the POST /agent/task endpoint.

Tests the agent task API route using FastAPI's TestClient with mocked dependencies.
"""

import pytest
from unittest.mock import AsyncMock

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.ai.models import AgentResponse, ToolInvocation
from app.ai.routes.agent import _get_agent_executor, router


@pytest.fixture
def app():
    """Create a FastAPI app with the agent router mounted."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def mock_agent_executor():
    """Create a mocked AIAgentExecutor that returns a completed response."""
    executor = AsyncMock()
    executor.execute_task = AsyncMock(
        return_value=AgentResponse(
            answer="Found 3 Python developer jobs posted in the last week.",
            steps=[
                ToolInvocation(
                    tool_name="vector_search",
                    input={"query": "Python developer jobs"},
                    output="Found 3 matching job posts.",
                    reasoning="Need to search for Python developer positions.",
                ),
                ToolInvocation(
                    tool_name="llm_reasoning",
                    input={"query": "Summarize the job results"},
                    output="3 Python developer jobs were posted recently.",
                    reasoning="Summarizing the search results for the user.",
                ),
            ],
            completed=True,
            message=None,
        )
    )
    return executor


@pytest.mark.asyncio
async def test_agent_task_returns_successful_response(app, mock_agent_executor):
    """POST /agent/task with valid task should return AgentResponse with 200."""
    app.dependency_overrides[_get_agent_executor] = lambda: mock_agent_executor

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/agent/task",
            json={"task": "Find Python developer jobs posted this week"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Found 3 Python developer jobs posted in the last week."
    assert data["completed"] is True
    assert len(data["steps"]) == 2
    assert data["steps"][0]["tool_name"] == "vector_search"
    assert data["steps"][1]["tool_name"] == "llm_reasoning"
    assert data["message"] is None


@pytest.mark.asyncio
async def test_agent_task_empty_task_returns_422(app, mock_agent_executor):
    """POST /agent/task with empty task string should return 422."""
    app.dependency_overrides[_get_agent_executor] = lambda: mock_agent_executor

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/agent/task",
            json={"task": ""},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_agent_task_missing_task_field_returns_422(app, mock_agent_executor):
    """POST /agent/task with missing task field should return 422."""
    app.dependency_overrides[_get_agent_executor] = lambda: mock_agent_executor

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/agent/task",
            json={},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_agent_task_missing_body_returns_422(app, mock_agent_executor):
    """POST /agent/task with no request body should return 422."""
    app.dependency_overrides[_get_agent_executor] = lambda: mock_agent_executor

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/agent/task",
            content="",
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_agent_task_partial_result_when_max_steps_exceeded(app):
    """POST /agent/task should return partial result when max steps exceeded."""
    executor = AsyncMock()
    executor.execute_task = AsyncMock(
        return_value=AgentResponse(
            answer="Partial result: found some jobs but could not complete analysis.",
            steps=[
                ToolInvocation(
                    tool_name="vector_search",
                    input={"query": "complex search"},
                    output="Some results found.",
                    reasoning="Starting the search.",
                )
            ],
            completed=False,
            message="The agent reached the maximum step limit of 10 without completing the task. A partial result has been provided.",
        )
    )
    app.dependency_overrides[_get_agent_executor] = lambda: executor

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/agent/task",
            json={"task": "Perform a very complex multi-step analysis"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["completed"] is False
    assert "maximum step limit" in data["message"]
    assert data["answer"] != ""
    assert len(data["steps"]) >= 1


@pytest.mark.asyncio
async def test_agent_task_single_char_task_returns_200(app, mock_agent_executor):
    """POST /agent/task with single character task should succeed."""
    app.dependency_overrides[_get_agent_executor] = lambda: mock_agent_executor

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/agent/task",
            json={"task": "x"},
        )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_agent_task_response_contains_steps_structure(app, mock_agent_executor):
    """POST /agent/task response steps should have correct structure."""
    app.dependency_overrides[_get_agent_executor] = lambda: mock_agent_executor

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/agent/task",
            json={"task": "Find jobs matching a profile"},
        )

    assert response.status_code == 200
    data = response.json()
    for step in data["steps"]:
        assert "tool_name" in step
        assert "input" in step
        assert "output" in step
        assert "reasoning" in step
