"""AI Intelligence Layer for the Job Board Platform.

This package provides AI-powered capabilities including:
- Vector-based semantic search
- RAG question answering
- AI-powered job recommendations
- Job description improvement
- Autonomous multi-step agent
"""

from app.ai.config import AIConfig
from app.ai.models import (
    AgentResponse,
    AgentTaskRequest,
    AskAIResponse,
    EmbeddingMetadata,
    ErrorResponse,
    ImproveDescriptionRequest,
    ImproveDescriptionResponse,
    ImprovementMode,
    JobRecommendation,
    RecommendationResponse,
    RecommendRequest,
    SourceReference,
    SyncReport,
    ToolInvocation,
    VectorDocument,
)

__all__ = [
    "AIConfig",
    "AgentResponse",
    "AgentTaskRequest",
    "AskAIResponse",
    "EmbeddingMetadata",
    "ErrorResponse",
    "ImproveDescriptionRequest",
    "ImproveDescriptionResponse",
    "ImprovementMode",
    "JobRecommendation",
    "RecommendationResponse",
    "RecommendRequest",
    "SourceReference",
    "SyncReport",
    "ToolInvocation",
    "VectorDocument",
]
