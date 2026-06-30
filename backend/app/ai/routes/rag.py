"""RAG API endpoint for the AI Intelligence Layer.

Provides the GET /ask-ai endpoint that accepts a natural language query
and returns an AI-generated answer with source references.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.ai.circuit_breaker import CircuitBreaker
from app.ai.config import AIConfig
from app.ai.embedding_service import EmbeddingService
from app.ai.exceptions import LLMServiceUnavailableError
from app.ai.models import AskAIResponse, ErrorResponse
from app.ai.prompt_manager import PromptTemplateManager
from app.ai.rag_engine import RAGEngine
from app.ai.vectorstore.factory import create_vector_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rag"])


def _get_config() -> AIConfig:
    """Provide AIConfig instance."""
    return AIConfig()


def _get_rag_engine(config: AIConfig = Depends(_get_config)) -> RAGEngine:
    """Dependency that provides a configured RAGEngine instance."""
    embedding_service = EmbeddingService(config=config)
    vector_store = create_vector_store(config=config)
    prompt_manager = PromptTemplateManager(templates_dir=config.PROMPT_TEMPLATES_DIR)
    circuit_breaker = CircuitBreaker()

    return RAGEngine(
        embedding_service=embedding_service,
        vector_store=vector_store,
        prompt_manager=prompt_manager,
        circuit_breaker=circuit_breaker,
        config=config,
    )


@router.get(
    "/ask-ai",
    response_model=AskAIResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid query parameter"},
        503: {"model": ErrorResponse, "description": "LLM service unavailable"},
    },
    summary="Ask AI a question",
    description=(
        "Accepts a natural language query and returns an AI-generated answer "
        "with source references from the knowledge base."
    ),
)
async def ask_ai(
    query: str = Query(default=None, description="Natural language question (max 1000 characters)"),
    rag_engine: RAGEngine = Depends(_get_rag_engine),
) -> AskAIResponse:
    """Process a natural language query through the RAG pipeline.

    Args:
        query: The question to answer (max 1000 characters).
        rag_engine: Injected RAGEngine instance.

    Returns:
        AskAIResponse with the generated answer and source references.

    Raises:
        HTTPException 400: If query is missing, empty, or exceeds 1000 chars.
        HTTPException 503: If the LLM service is unavailable.
    """
    # Validate query parameter
    if query is None or query.strip() == "":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorResponse(
                error="invalid_query",
                message="Query parameter is required and cannot be empty.",
            ).model_dump(),
        )

    if len(query) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorResponse(
                error="query_too_long",
                message="Query must not exceed 1000 characters.",
                details={"max_length": 1000, "actual_length": len(query)},
            ).model_dump(),
        )

    try:
        response = await rag_engine.answer_query(query)
        return response
    except LLMServiceUnavailableError as exc:
        logger.warning("LLM service unavailable: %s", str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorResponse(
                error="llm_unavailable",
                message="LLM service is temporarily unavailable. Please try again later.",
            ).model_dump(),
        ) from exc
