"""RAG Engine for the AI Intelligence Layer.

Implements retrieval-augmented generation for the /ask-ai endpoint.
Pipeline: query → embedding → vector search → prompt rendering → LLM completion → response.
"""

import logging
from typing import List

from app.ai.circuit_breaker import CircuitBreaker
from app.ai.config import AIConfig
from app.ai.embedding_service import EmbeddingService
from app.ai.exceptions import LLMServiceUnavailableError
from app.ai.llm_factory import create_chat_llm
from app.ai.models import AskAIResponse, SourceReference, VectorDocument
from app.ai.prompt_manager import PromptTemplateManager
from app.ai.vectorstore import VectorStoreInterface

logger = logging.getLogger(__name__)

# Collections to search for RAG queries
_RAG_COLLECTIONS = ["job_posts", "companies", "candidates"]


class RAGEngine:
    """Retrieval-Augmented Generation engine for answering user queries.

    Uses embedding-based similarity search to find relevant documents,
    then passes them as context to an LLM for answer generation.

    Args:
        embedding_service: Service for generating query embeddings.
        vector_store: Vector database interface for similarity search.
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
        self._llm = create_chat_llm(config)

    async def answer_query(self, query: str) -> AskAIResponse:
        """Process a natural language query through the RAG pipeline.

        Pipeline steps:
        1. Generate embedding for the query text.
        2. Perform similarity search across all collections (job_posts,
           companies, candidates) with top_k and min_score from config.
        3. If no documents found above threshold, return a "no relevant
           information found" response.
        4. Build context string from retrieved documents.
        5. Render the "rag_answer" prompt template with query and context.
        6. Send rendered prompt to LLM via circuit breaker.
        7. Parse LLM response and build AskAIResponse with sources.

        Args:
            query: The natural language question to answer.

        Returns:
            AskAIResponse containing the answer, source references, and
            the original query.

        Raises:
            LLMServiceUnavailableError: If the LLM is unavailable (circuit
                breaker open or API error).
        """
        # Step 1: Generate embedding for the query
        query_embedding = await self._embedding_service.generate_embedding(query)

        # Step 2: Search across all collections
        all_documents: List[VectorDocument] = []
        for collection in _RAG_COLLECTIONS:
            results = await self._vector_store.search(
                collection=collection,
                query_embedding=query_embedding,
                top_k=self._config.RAG_TOP_K,
                min_score=self._config.RAG_MIN_RELEVANCE_THRESHOLD,
            )
            all_documents.extend(results)

        # Sort all results by score descending and take top_k overall
        all_documents.sort(key=lambda doc: doc.score or 0.0, reverse=True)
        top_documents = all_documents[: self._config.RAG_TOP_K]

        # Step 3: If no documents found above threshold, return "no info" response
        if not top_documents:
            return AskAIResponse(
                answer="No relevant information found for your query.",
                sources=[],
                query=query,
            )

        # Step 4: Build context string from retrieved documents
        context = self._build_context(top_documents)

        # Step 5: Render RAG prompt template
        rendered_prompt = self._prompt_manager.render(
            "rag_answer",
            query=query,
            context_documents=context,
        )

        # Step 6: Send to LLM via circuit breaker
        try:
            answer_text = await self._circuit_breaker.call(
                self._invoke_llm, rendered_prompt
            )
        except (LLMServiceUnavailableError, Exception) as exc:
            # Fallback: return search results as answer when LLM is unavailable
            logger.warning("LLM unavailable, returning search results directly.")
            sources = self._extract_sources(top_documents)
            fallback_answer = self._build_fallback_answer(query, top_documents)
            return AskAIResponse(
                answer=fallback_answer,
                sources=sources,
                query=query,
            )

        # Step 7: Build response with sources
        sources = self._extract_sources(top_documents)

        return AskAIResponse(
            answer=answer_text.strip(),
            sources=sources,
            query=query,
        )

    def _build_context(self, documents: List[VectorDocument]) -> str:
        """Build a context string from retrieved documents.

        Combines text snippets from all documents into a single
        formatted context string for the prompt template.

        Args:
            documents: List of retrieved VectorDocument instances.

        Returns:
            Formatted context string with document snippets.
        """
        context_parts: List[str] = []
        for i, doc in enumerate(documents, start=1):
            entity_type = doc.metadata.get("entity_type", "unknown")
            entity_id = doc.metadata.get("entity_id", "unknown")
            snippet = doc.text_snippet
            context_parts.append(
                f"[Document {i}] (Type: {entity_type}, ID: {entity_id})\n{snippet}"
            )
        return "\n\n".join(context_parts)

    def _extract_sources(self, documents: List[VectorDocument]) -> List[SourceReference]:
        """Extract source references from retrieved documents.

        Args:
            documents: List of retrieved VectorDocument instances.

        Returns:
            List of SourceReference models built from document metadata.
        """
        sources: List[SourceReference] = []
        for doc in documents:
            sources.append(
                SourceReference(
                    entity_type=doc.metadata.get("entity_type", "unknown"),
                    entity_id=doc.metadata.get("entity_id", "unknown"),
                    text_snippet=doc.text_snippet,
                    relevance_score=doc.score if doc.score is not None else 0.0,
                )
            )
        return sources

    def _build_fallback_answer(
        self, query: str, documents: List[VectorDocument]
    ) -> str:
        """Build a human-readable fallback answer from search results.

        Only includes results within 90% of the top score so users
        only see truly relevant items for their query.

        Args:
            query: The original user query.
            documents: Retrieved documents from vector search.

        Returns:
            A nicely formatted answer string.
        """
        if not documents:
            return "No relevant information found for your query."

        # Only keep results within 90% of the best score
        top_score = documents[0].score or 0.0
        score_cutoff = top_score * 0.90
        relevant_docs = [d for d in documents if (d.score or 0.0) >= score_cutoff]

        jobs = []
        companies = []
        candidates = []

        for doc in relevant_docs:
            entity_type = doc.metadata.get("entity_type", "unknown")
            snippet = doc.text_snippet or ""
            score = doc.score if doc.score is not None else 0.0

            if entity_type == "job_post":
                title = doc.metadata.get("title", "")
                if not title:
                    title = snippet.split(" We ")[0].split(" Join ")[0].split(" Build ")[0][:60]
                description = snippet[len(title):].strip() if title in snippet else snippet
                jobs.append({"title": title, "description": description[:120], "score": score})

            elif entity_type == "company":
                name = doc.metadata.get("name", "")
                industry = doc.metadata.get("industry", "")
                if not name:
                    name = snippet.split(" is ")[0].split(" specializes")[0].split(" provides")[0][:40]
                description = snippet[len(name):].strip() if name in snippet else snippet
                companies.append({"name": name, "industry": industry, "description": description[:120], "score": score})

            elif entity_type == "candidate":
                skills = doc.metadata.get("skills", "")
                if not skills:
                    skills = snippet[:80]
                experience = snippet[len(skills):].strip() if skills in snippet else snippet
                candidates.append({"skills": skills, "experience": experience[:120], "score": score})

        # Build the answer
        answer_parts = []
        answer_parts.append(f"Based on your query \"{query}\", here's what I found:\n")

        if jobs:
            answer_parts.append("📋 **Open Positions:**")
            for i, job in enumerate(jobs[:3], 1):
                match_pct = int(job['score'] * 100)
                answer_parts.append(f"  {i}. **{job['title']}** ({match_pct}% match) — {job['description'][:100]}")
            answer_parts.append("")

        if companies:
            answer_parts.append("🏢 **Companies:**")
            for i, comp in enumerate(companies[:3], 1):
                industry_tag = f" ({comp['industry']})" if comp['industry'] else ""
                match_pct = int(comp['score'] * 100)
                answer_parts.append(f"  {i}. **{comp['name']}**{industry_tag} ({match_pct}% match) —{comp['description'][:100]}")
            answer_parts.append("")

        if candidates:
            answer_parts.append("👤 **Relevant Candidates:**")
            for i, cand in enumerate(candidates[:2], 1):
                match_pct = int(cand['score'] * 100)
                answer_parts.append(f"  {i}. ({match_pct}% match) Skills: {cand['skills'][:80]} — {cand['experience'][:80]}")

        if not jobs and not companies and not candidates:
            answer_parts.append("No closely matching results found for your query.")

        return "\n".join(answer_parts)

    async def _invoke_llm(self, prompt: str) -> str:
        """Invoke the LLM with the given prompt.

        Args:
            prompt: The fully rendered prompt to send to the LLM.

        Returns:
            The LLM's response text.
        """
        import asyncio
        try:
            response = await asyncio.wait_for(
                self._llm.ainvoke(prompt), timeout=15.0
            )
            return response.content
        except asyncio.TimeoutError:
            raise LLMServiceUnavailableError("LLM request timed out after 15s")
