"""LangChain-compatible tools for the AI Agent.

Provides three tools that the ReAct-style agent can invoke:
- api_query_tool: Queries the Job Board platform API for jobs, companies, candidates
- vector_search_tool: Performs semantic similarity search on the vector database
- llm_reasoning_tool: Calls the LLM for analysis, summaries, and decisions

Each tool accepts a string input, returns a string output, and handles errors
gracefully by returning error messages rather than raising exceptions.
"""

import logging
from typing import List, Optional

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from app.ai.config import AIConfig
from app.ai.embedding_service import EmbeddingService
from app.ai.models import VectorDocument
from app.ai.vectorstore import VectorStoreInterface

logger = logging.getLogger(__name__)

# Collections available in the vector database
_COLLECTIONS = ["job_posts", "companies", "candidates"]


def _get_config() -> AIConfig:
    """Load AI configuration from environment."""
    return AIConfig()


def _format_search_results(results: List[VectorDocument]) -> str:
    """Format vector search results as readable text.

    Args:
        results: List of VectorDocument instances from a similarity search.

    Returns:
        Formatted string with entity details and scores.
    """
    if not results:
        return "No results found."

    lines: List[str] = []
    for i, doc in enumerate(results, 1):
        entity_type = doc.metadata.get("entity_type", "unknown")
        entity_id = doc.metadata.get("entity_id", doc.id)
        score = f"{doc.score:.4f}" if doc.score is not None else "N/A"
        snippet = doc.text_snippet[:200] if doc.text_snippet else "(no text)"
        lines.append(
            f"{i}. [entity_type={entity_type}, entity_id={entity_id}, score={score}]\n"
            f"   {snippet}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool factory functions
#
# Each factory accepts optional dependencies (for testability) and returns a
# LangChain-compatible tool. The module-level tool instances use defaults.
# ---------------------------------------------------------------------------


def create_api_query_tool(
    config: Optional[AIConfig] = None,
    base_url: str = "http://localhost:8000",
):
    """Create the API query tool.

    This tool queries the Job Board platform's FastAPI endpoints for jobs,
    companies, or candidates using httpx. Falls back to a structured
    placeholder response when the API is unreachable.

    Args:
        config: Optional AIConfig instance. Loaded from env if not provided.
        base_url: Base URL of the FastAPI application.

    Returns:
        A LangChain-compatible tool function.
    """
    _config = config or _get_config()

    @tool
    async def api_query(query: str) -> str:
        """Query the Job Board platform API to fetch data about jobs, companies, or candidates.

        Input should specify what data you want to retrieve, for example:
        'get all jobs', 'get company details for ID 5', 'list candidates'.

        Returns structured data from the platform's database.
        """
        try:
            import httpx

            query_lower = query.lower()

            # Determine which endpoint to call based on query content
            if "job" in query_lower:
                endpoint = "/jobs"
            elif "company" in query_lower or "companies" in query_lower:
                endpoint = "/companies"
            elif "candidate" in query_lower:
                endpoint = "/candidates"
            else:
                return (
                    "Please specify what to query: 'jobs', 'companies', or "
                    f"'candidates'. Your query was: {query}"
                )

            async with httpx.AsyncClient(
                base_url=base_url, timeout=10.0
            ) as client:
                response = await client.get(endpoint)
                response.raise_for_status()
                data = response.json()

            if not data:
                return f"No results found for endpoint {endpoint}."

            # Format response as readable text
            if isinstance(data, list):
                lines = []
                for item in data[:20]:
                    parts = [f"{k}={v}" for k, v in item.items()]
                    lines.append(f"- {', '.join(parts)}")
                return (
                    f"Found {len(data)} result(s) from {endpoint}:\n"
                    + "\n".join(lines)
                )
            else:
                parts = [f"{k}={v}" for k, v in data.items()]
                return f"Result from {endpoint}: {', '.join(parts)}"

        except Exception as e:
            logger.error("api_query tool error: %s", str(e))
            return f"Error querying the platform API: {str(e)}"

    return api_query


def create_vector_search_tool(
    config: Optional[AIConfig] = None,
    embedding_service: Optional[EmbeddingService] = None,
    vector_store: Optional[VectorStoreInterface] = None,
):
    """Create the vector search tool.

    Uses EmbeddingService to generate a query embedding, then searches
    across all collections in the vector database for semantically similar
    content.

    Args:
        config: Optional AIConfig instance. Loaded from env if not provided.
        embedding_service: Optional EmbeddingService instance.
        vector_store: Optional VectorStoreInterface instance.

    Returns:
        A LangChain-compatible tool function.
    """
    _config = config or _get_config()

    @tool
    async def vector_search(query: str) -> str:
        """Search the vector database for semantically similar content.

        Input should be a search query describing what you're looking for.
        Results include job posts, company descriptions, or candidate profiles
        with relevance scores.
        """
        try:
            # Resolve dependencies
            _embedding_svc = embedding_service
            _vector_store = vector_store

            if _embedding_svc is None:
                _embedding_svc = EmbeddingService(_config)

            if _vector_store is None:
                from app.ai.vectorstore.factory import create_vector_store

                _vector_store = create_vector_store(_config)

            # Generate query embedding
            query_embedding = await _embedding_svc.generate_embedding(query)

            # Search across all collections
            all_results: List[VectorDocument] = []
            for collection_name in _COLLECTIONS:
                try:
                    results = await _vector_store.search(
                        collection=collection_name,
                        query_embedding=query_embedding,
                        top_k=_config.RAG_TOP_K,
                        min_score=_config.RAG_MIN_RELEVANCE_THRESHOLD,
                    )
                    all_results.extend(results)
                except Exception as coll_err:
                    logger.warning(
                        "Failed to search collection '%s': %s",
                        collection_name,
                        str(coll_err),
                    )

            # Sort all results by score descending and take top results
            all_results.sort(key=lambda d: d.score or 0.0, reverse=True)
            top_results = all_results[: _config.RAG_TOP_K]

            return _format_search_results(top_results)

        except Exception as e:
            logger.error("vector_search tool error: %s", str(e))
            return f"Error performing vector search: {str(e)}"

    return vector_search


def create_llm_reasoning_tool(config: Optional[AIConfig] = None):
    """Create the LLM reasoning tool.

    Sends input text to ChatOpenAI for analysis, summarization, or
    reasoning tasks.

    Args:
        config: Optional AIConfig instance. Loaded from env if not provided.

    Returns:
        A LangChain-compatible tool function.
    """
    _config = config or _get_config()

    @tool
    async def llm_reasoning(query: str) -> str:
        """Use LLM for analysis, summarization, or reasoning about data.

        Input should be the text you want to analyze along with instructions
        for what analysis to perform. For example:
        'Summarize the following job descriptions: ...' or
        'Compare these two candidates for a software engineer role: ...'
        """
        try:
            llm = ChatOpenAI(
                model=_config.OPENAI_CHAT_MODEL,
                openai_api_key=_config.OPENAI_API_KEY,
                temperature=0.3,
            )

            response = await llm.ainvoke(query)
            return response.content

        except Exception as e:
            logger.error("llm_reasoning tool error: %s", str(e))
            return f"Error performing LLM reasoning: {str(e)}"

    return llm_reasoning


# ---------------------------------------------------------------------------
# Module-level tool instances (default configuration)
# These are the tools used by the AIAgentExecutor.
# ---------------------------------------------------------------------------

api_query_tool = create_api_query_tool()
vector_search_tool = create_vector_search_tool()
llm_reasoning_tool = create_llm_reasoning_tool()
