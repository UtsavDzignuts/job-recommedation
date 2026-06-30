"""Pydantic request/response models for the AI Intelligence Layer.

All models use Pydantic v2 syntax and follow the design document schemas.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# --- Vector Document (dataclass for internal use) ---


@dataclass
class VectorDocument:
    """Represents a document stored in or retrieved from the vector database."""

    id: str
    embedding: List[float]
    metadata: Dict[str, object]
    text_snippet: str
    score: Optional[float] = None


# --- RAG ---


class SourceReference(BaseModel):
    """A reference to a source document used in generating a RAG answer."""

    entity_type: str
    """Source entity type: 'job_post', 'company', or 'candidate'."""

    entity_id: str
    """Unique identifier of the source entity."""

    text_snippet: str
    """Short excerpt from the source document."""

    relevance_score: float
    """Similarity score indicating how relevant this source is to the query."""


class AskAIResponse(BaseModel):
    """Response from the /ask-ai RAG endpoint."""

    answer: str
    """LLM-generated natural language answer."""

    sources: List[SourceReference]
    """List of source references used to generate the answer."""

    query: str
    """The original query that was asked."""


# --- Recommendations ---


class RecommendRequest(BaseModel):
    """Request body for the /recommend endpoint."""

    resume_text: str = Field(..., min_length=1, max_length=10_000)
    """Resume or profile description text (1 to 10,000 characters)."""


class JobRecommendation(BaseModel):
    """A single job recommendation with match details."""

    job_title: str
    """Title of the recommended job."""

    job_id: str
    """Unique identifier of the recommended job."""

    match_reason: str
    """Explanation of why this job matches the candidate's profile."""

    confidence_score: float = Field(ge=0.0, le=1.0)
    """Confidence score between 0.0 and 1.0 indicating match strength."""


class RecommendationResponse(BaseModel):
    """Response from the /recommend endpoint."""

    recommendations: List[JobRecommendation] = Field(default_factory=list, max_length=5)
    """List of job recommendations, at most 5."""

    message: Optional[str] = None
    """Optional message, e.g. when no recommendations are found."""


# --- Description Improvement ---


class ImprovementMode(str, Enum):
    """Supported modes for job description improvement."""

    SHORT_AND_CRISP = "short_and_crisp"
    DETAILED_AND_FORMAL = "detailed_and_formal"
    MARKETING_ORIENTED = "marketing_oriented"


class ImproveDescriptionRequest(BaseModel):
    """Request body for the /improve-description endpoint."""

    description: str = Field(..., min_length=1, max_length=50_000)
    """Raw job description text (1 to 50,000 characters)."""

    mode: ImprovementMode
    """Improvement mode to apply."""


class ImproveDescriptionResponse(BaseModel):
    """Response from the /improve-description endpoint."""

    improved_description: str
    """The improved job description text."""

    mode: ImprovementMode
    """The improvement mode that was applied."""


# --- AI Agent ---


class AgentTaskRequest(BaseModel):
    """Request body for the /agent/task endpoint."""

    task: str = Field(..., min_length=1)
    """Natural language task description."""


class ToolInvocation(BaseModel):
    """Record of a single tool invocation during agent execution."""

    tool_name: str
    """Name of the tool that was invoked."""

    input: dict
    """Input parameters passed to the tool."""

    output: str
    """Output returned by the tool."""

    reasoning: str
    """Agent's reasoning for invoking this tool."""


class AgentResponse(BaseModel):
    """Response from the /agent/task endpoint."""

    answer: str
    """Final answer produced by the agent."""

    steps: List[ToolInvocation]
    """Sequence of tool invocations with inputs, outputs, and reasoning."""

    completed: bool
    """Whether the agent completed the task within the step limit."""

    message: Optional[str] = None
    """Optional message, e.g. when max steps were exceeded."""


# --- Sync ---


class SyncReport(BaseModel):
    """Report generated after a full re-sync operation."""

    total_entities: int
    """Total number of entities processed."""

    created: int
    """Number of new embeddings created."""

    updated: int
    """Number of existing embeddings updated."""

    deleted: int
    """Number of embeddings deleted."""

    failed: int
    """Number of entities that failed to sync."""

    duration_seconds: float
    """Duration of the sync operation in seconds."""


# --- Vector Document Metadata ---


class EmbeddingMetadata(BaseModel):
    """Metadata associated with a stored embedding in the vector database."""

    entity_type: str
    """Source entity type: 'job_post', 'company', or 'candidate'."""

    entity_id: str
    """Unique identifier of the source entity."""

    created_at: datetime
    """Timestamp when the embedding was created."""

    text_snippet: str
    """Short excerpt from the original content."""

    source_updated_at: Optional[datetime] = None
    """Timestamp when the source entity was last updated."""


# --- Error Response ---


class ErrorResponse(BaseModel):
    """Standard error response format for all AI endpoints."""

    error: str
    """Machine-readable error code."""

    message: str
    """Human-readable description of the error."""

    details: Optional[dict] = None
    """Additional context about the error."""
