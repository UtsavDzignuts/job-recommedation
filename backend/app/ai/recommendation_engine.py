"""Recommendation Engine for the AI Intelligence Layer.

Implements the job recommendation pipeline: embed resume → similarity search
on job_posts collection → LLM ranking → structured JSON output.
"""

import json
import logging
from typing import List

from app.ai.circuit_breaker import CircuitBreaker
from app.ai.config import AIConfig
from app.ai.embedding_service import EmbeddingService
from app.ai.exceptions import LLMServiceUnavailableError
from app.ai.llm_factory import create_chat_llm
from app.ai.models import JobRecommendation, RecommendationResponse
from app.ai.prompt_manager import PromptTemplateManager
from app.ai.vectorstore import VectorStoreInterface

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Engine for generating AI-powered job recommendations.

    Pipeline:
    1. Generate embedding for resume_text using EmbeddingService
    2. Perform similarity search on "job_posts" collection
    3. If no matches found above threshold, return empty list with message
    4. Build job_matches text from retrieved documents
    5. Render "job_recommendation" prompt template
    6. Send rendered prompt to LLM via ChatOpenAI (with circuit breaker)
    7. Parse LLM JSON response into JobRecommendation items
    8. Cap at max_results, ensure confidence_score in [0.0, 1.0]

    Args:
        embedding_service: Service for generating text embeddings.
        vector_store: Vector store implementation for similarity search.
        prompt_manager: Manager for loading and rendering prompt templates.
        circuit_breaker: Circuit breaker protecting LLM calls.
        config: AI configuration settings.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreInterface,
        prompt_manager: PromptTemplateManager,
        circuit_breaker: CircuitBreaker,
        config: AIConfig,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._prompt_manager = prompt_manager
        self._circuit_breaker = circuit_breaker
        self._config = config
        self._llm = create_chat_llm(config, temperature=0.0)

    async def recommend_jobs(self, resume_text: str) -> RecommendationResponse:
        """Generate job recommendations based on resume text.

        Args:
            resume_text: The candidate's resume or profile description.

        Returns:
            RecommendationResponse with ranked job recommendations or an
            empty list with an explanatory message if no matches are found.

        Raises:
            LLMServiceUnavailableError: If the circuit breaker is open or LLM fails.
            EmbeddingGenerationError: If embedding generation fails.
            VectorDBUnavailableError: If the vector database is unreachable.
        """
        # Step 1: Generate embedding for resume text
        resume_embedding = await self._embedding_service.generate_embedding(resume_text)

        # Step 2: Similarity search on job_posts collection
        top_k = self._config.RECOMMEND_TOP_K
        min_score = self._config.RAG_MIN_RELEVANCE_THRESHOLD
        matches = await self._vector_store.search(
            collection="job_posts",
            query_embedding=resume_embedding,
            top_k=top_k,
            min_score=min_score,
        )

        # Step 3: If no matches above threshold, return empty with message
        if not matches:
            return RecommendationResponse(
                recommendations=[],
                message="No relevant jobs found for the given profile.",
            )

        # Step 4: Build job_matches text from retrieved documents
        job_matches_text = self._build_job_matches_text(matches)

        # Step 5: Render the job_recommendation prompt template
        rendered_prompt = self._prompt_manager.render(
            "job_recommendation",
            resume_text=resume_text,
            job_matches=job_matches_text,
        )

        # Step 6: Call LLM via circuit breaker
        try:
            llm_response = await self._circuit_breaker.call(
                self._invoke_llm, rendered_prompt
            )
        except (LLMServiceUnavailableError, Exception) as exc:
            # Fallback: return recommendations from vector search directly
            logger.warning("LLM unavailable for recommendations, using fallback: %s", str(exc))
            return RecommendationResponse(
                recommendations=self._fallback_recommendations(matches)
            )

        # Step 7: Parse LLM response into JobRecommendation items
        recommendations = self._parse_llm_response(llm_response, matches)

        # Step 8: Cap at max_results, ensure scores are valid
        max_results = self._config.RECOMMEND_MAX_RESULTS
        recommendations = recommendations[:max_results]

        return RecommendationResponse(recommendations=recommendations)

    async def _invoke_llm(self, prompt: str) -> str:
        """Invoke the LLM with the given prompt.

        Args:
            prompt: The rendered prompt string.

        Returns:
            The LLM response content as a string.
        """
        import asyncio
        try:
            response = await asyncio.wait_for(
                self._llm.ainvoke(prompt), timeout=15.0
            )
            return response.content
        except asyncio.TimeoutError:
            from app.ai.exceptions import LLMServiceUnavailableError
            raise LLMServiceUnavailableError("LLM request timed out after 15s")

    def _build_job_matches_text(self, matches: list) -> str:
        """Build a formatted text block from vector search matches.

        Each match includes its ID, text snippet, and similarity score.

        Args:
            matches: List of VectorDocument results from similarity search.

        Returns:
            Formatted string with job match details.
        """
        lines: List[str] = []
        for i, doc in enumerate(matches, start=1):
            score = doc.score if doc.score is not None else 0.0
            lines.append(
                f"{i}. Job ID: {doc.id}\n"
                f"   Score: {score:.3f}\n"
                f"   Description: {doc.text_snippet}"
            )
        return "\n\n".join(lines)

    def _parse_llm_response(
        self, llm_response: str, matches: list
    ) -> List[JobRecommendation]:
        """Parse LLM JSON response into a list of JobRecommendation items.

        If the LLM response cannot be parsed as valid JSON, falls back to
        returning recommendations based on the raw vector search matches.

        Args:
            llm_response: Raw string response from the LLM.
            matches: Original vector search matches (used as fallback).

        Returns:
            List of JobRecommendation items with validated confidence scores.
        """
        try:
            parsed = json.loads(llm_response.strip())

            if not isinstance(parsed, list):
                logger.warning(
                    "LLM response is not a JSON array, falling back to raw matches."
                )
                return self._fallback_recommendations(matches)

            recommendations: List[JobRecommendation] = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue

                # Extract fields with defaults
                job_title = item.get("job_title", "Unknown")
                job_id = item.get("job_id", "unknown")
                match_reason = item.get("match_reason", "Matched based on profile similarity.")
                confidence_score = item.get("confidence_score", 0.5)

                # Clamp confidence_score to [0.0, 1.0]
                try:
                    confidence_score = float(confidence_score)
                except (TypeError, ValueError):
                    confidence_score = 0.5
                confidence_score = max(0.0, min(1.0, confidence_score))

                recommendations.append(
                    JobRecommendation(
                        job_title=job_title,
                        job_id=job_id,
                        match_reason=match_reason,
                        confidence_score=confidence_score,
                    )
                )

            return recommendations

        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "Failed to parse LLM recommendation response as JSON: %s. "
                "Falling back to raw matches.",
                str(exc),
            )
            return self._fallback_recommendations(matches)

    def _fallback_recommendations(self, matches: list) -> List[JobRecommendation]:
        """Generate fallback recommendations from raw vector search matches.

        Uses a gap-based filter: only includes results that are significantly
        more relevant than the rest (top cluster with a meaningful gap from
        the remaining results).

        Args:
            matches: Vector search match results.

        Returns:
            List of JobRecommendation items derived from match data.
        """
        if not matches:
            return []

        recommendations: List[JobRecommendation] = []
        max_results = min(3, self._config.RECOMMEND_MAX_RESULTS)

        # Get scores
        scores = [(doc, doc.score if doc.score is not None else 0.0) for doc in matches]
        scores.sort(key=lambda x: x[1], reverse=True)

        if not scores:
            return []

        # Find the natural gap: only include results within 95% of the top score
        top_score = scores[0][1]
        score_cutoff = top_score * 0.95

        for doc, score in scores:
            if score < score_cutoff:
                break
            if len(recommendations) >= max_results:
                break

            # Clamp score to [0.0, 1.0]
            score = max(0.0, min(1.0, score))

            # Extract job title from metadata
            job_title = doc.metadata.get("title", "") if doc.metadata else ""
            if not job_title:
                snippet = doc.text_snippet or "Unknown Job"
                job_title = snippet.split(" We ")[0].split(" Join ")[0].split(" Build ")[0][:80]

            # Build match reason from the snippet
            snippet = doc.text_snippet or ""
            # Get the description part (after title)
            desc = snippet[len(job_title):].strip() if job_title in snippet else snippet
            match_reason = desc[:300] if desc else "Matched based on profile similarity."

            recommendations.append(
                JobRecommendation(
                    job_title=job_title,
                    job_id=doc.metadata.get("entity_id", doc.id) if doc.metadata else doc.id,
                    match_reason=match_reason,
                    confidence_score=score,
                )
            )

        return recommendations
