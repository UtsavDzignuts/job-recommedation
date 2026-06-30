"""Recommendation API endpoint for the AI Intelligence Layer.

Provides the POST /recommend endpoint that generates AI-powered job
recommendations based on resume text input.
"""

import logging

from fastapi import APIRouter, Depends, status

from app.ai.circuit_breaker import CircuitBreaker
from app.ai.config import AIConfig
from app.ai.embedding_service import EmbeddingService
from app.ai.exceptions import LLMServiceUnavailableError
from app.ai.models import (
    ErrorResponse,
    RecommendationResponse,
    RecommendRequest,
)
from app.ai.prompt_manager import PromptTemplateManager
from app.ai.recommendation_engine import RecommendationEngine
from app.ai.vectorstore.factory import create_vector_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["recommendations"])


def _get_config() -> AIConfig:
    """Provide AIConfig instance."""
    return AIConfig()


def _get_recommendation_engine(
    config: AIConfig = Depends(_get_config),
) -> RecommendationEngine:
    """Dependency that provides a configured RecommendationEngine instance."""
    embedding_service = EmbeddingService(config=config)
    vector_store = create_vector_store(config=config)
    prompt_manager = PromptTemplateManager(
        templates_dir=config.PROMPT_TEMPLATES_DIR
    )
    circuit_breaker = CircuitBreaker()

    return RecommendationEngine(
        embedding_service=embedding_service,
        vector_store=vector_store,
        prompt_manager=prompt_manager,
        circuit_breaker=circuit_breaker,
        config=config,
    )


@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        503: {"model": ErrorResponse, "description": "LLM service unavailable"},
    },
    summary="Get AI job recommendations",
    description=(
        "Accepts resume text and returns up to 5 AI-powered job recommendations "
        "ranked by confidence score. The resume_text field must be between 1 and "
        "10,000 characters."
    ),
)
async def recommend(
    request: RecommendRequest,
    recommendation_engine: RecommendationEngine = Depends(_get_recommendation_engine),
) -> RecommendationResponse:
    """Generate job recommendations based on resume text.

    Accepts a RecommendRequest body with resume_text (1-10000 chars).
    Pydantic validation automatically returns HTTP 422 for invalid input.

    Returns:
        RecommendationResponse with ranked job recommendations.

    Raises:
        HTTPException 503: If the LLM service is unavailable.
    """
    try:
        response = await recommendation_engine.recommend_jobs(request.resume_text)
        return response
    except LLMServiceUnavailableError as exc:
        logger.error("LLM service unavailable during recommendation: %s", str(exc))
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                error="llm_service_unavailable",
                message=str(exc.message),
            ).model_dump(),
        )
