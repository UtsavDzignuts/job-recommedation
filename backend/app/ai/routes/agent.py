"""Agent task API endpoint for the AI Intelligence Layer.

Provides the POST /agent/task endpoint that executes complex multi-step
tasks using the AI Agent with ReAct-style reasoning.
"""

import logging

from fastapi import APIRouter, Depends, status

from app.ai.agent.executor import AIAgentExecutor
from app.ai.config import AIConfig
from app.ai.models import AgentResponse, AgentTaskRequest, ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agent"])


def _get_config() -> AIConfig:
    """Provide AIConfig instance."""
    return AIConfig()


def _get_agent_executor(
    config: AIConfig = Depends(_get_config),
) -> AIAgentExecutor:
    """Dependency that provides a configured AIAgentExecutor instance."""
    return AIAgentExecutor(config=config)


@router.post(
    "/agent/task",
    response_model=AgentResponse,
    status_code=status.HTTP_200_OK,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
    },
    summary="Execute a complex multi-step AI task",
    description=(
        "Accepts a natural language task description and uses an AI agent "
        "with ReAct-style reasoning to autonomously solve it. The task field "
        "must be a non-empty string."
    ),
)
async def agent_task(
    request: AgentTaskRequest,
    agent_executor: AIAgentExecutor = Depends(_get_agent_executor),
) -> AgentResponse:
    """Execute a complex task using the AI agent.

    Accepts an AgentTaskRequest body with a non-empty task string.
    Pydantic validation automatically returns HTTP 422 for empty or missing task.

    Returns:
        AgentResponse with answer, steps, completed status, and optional message.
    """
    response = await agent_executor.execute_task(request.task)
    return response
