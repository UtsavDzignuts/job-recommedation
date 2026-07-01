"""AI Intelligence Layer configuration.

Loads all AI-related settings from environment variables using pydantic-settings.
"""

from typing import Optional

from pydantic_settings import BaseSettings


class AIConfig(BaseSettings):
    """Configuration for the AI Intelligence Layer.

    All settings are loaded from environment variables with sensible defaults.
    Required settings (no default) must be set in the environment or .env file.
    """

    # Vector DB
    VECTOR_DB_PROVIDER: str = "chromadb"
    """Vector database provider: chromadb | pgvector | qdrant"""

    VECTOR_DB_URL: str = "http://localhost:8000"
    """Connection URL for the vector database."""

    VECTOR_DB_API_KEY: Optional[str] = None
    """Optional API key for vector database authentication (Chroma Cloud)."""

    CHROMA_TENANT: Optional[str] = None
    """Chroma Cloud tenant ID."""

    CHROMA_DATABASE: Optional[str] = None
    """Chroma Cloud database name."""

    # LLM Provider
    LLM_PROVIDER: str = "google"
    """LLM and embedding provider: 'openai' or 'google'."""

    # OpenAI
    OPENAI_API_KEY: str = ""
    """OpenAI API key for embeddings and chat completions."""

    OPENAI_EMBEDDING_MODEL: str = "text-embedding-ada-002"
    """OpenAI model used for generating embeddings."""

    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    """OpenAI model used for chat completions."""

    # Google Gemini
    GOOGLE_API_KEY: str = ""
    """Google API key for Gemini embeddings and chat completions."""

    GOOGLE_EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    """Google model used for generating embeddings."""

    GOOGLE_CHAT_MODEL: str = "gemini-2.0-flash"
    """Google Gemini model used for chat completions."""

    # RAG
    RAG_MIN_RELEVANCE_THRESHOLD: float = 0.5
    """Minimum similarity score for documents to be considered relevant."""

    RAG_TOP_K: int = 10
    """Maximum number of documents to retrieve in similarity search."""

    RAG_MAX_QUERY_LENGTH: int = 1000
    """Maximum allowed length for RAG query input (characters)."""

    # Recommendations
    RECOMMEND_TOP_K: int = 10
    """Number of candidate job matches to retrieve from vector DB."""

    RECOMMEND_MAX_RESULTS: int = 5
    """Maximum number of recommendations returned to the user."""

    # Agent
    AGENT_MAX_STEPS: int = 10
    """Maximum reasoning steps the AI agent can take per task."""

    # Sync
    SYNC_RETRY_MAX: int = 3
    """Maximum retry attempts for sync operations."""

    SYNC_RETRY_BASE_DELAY: float = 1.0
    """Base delay in seconds for exponential backoff in sync retries."""

    # Prompts
    PROMPT_TEMPLATES_DIR: str = "app/prompts"
    """Directory containing prompt template files."""

    model_config = {
        "env_prefix": "",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }
