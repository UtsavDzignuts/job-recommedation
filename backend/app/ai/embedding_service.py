"""Embedding Service for the AI Intelligence Layer.

Handles text-to-vector conversion using either OpenAI or Google Gemini
embedding models with retry logic for transient failures.
"""

import logging
from typing import List

import openai

from app.ai.config import AIConfig
from app.ai.exceptions import EmbeddingGenerationError
from app.ai.retry import RetryConfig, async_retry

logger = logging.getLogger(__name__)

# Retry configuration: 3 retries with exponential backoff 1s, 2s, 4s
_EMBEDDING_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0,
)

# OpenAI status codes that are transient and should be retried
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_retryable_openai_error(exc: BaseException) -> bool:
    """Check if an OpenAI error is retryable based on status code."""
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in _RETRYABLE_STATUS_CODES
    return False


class _RetryableEmbeddingError(Exception):
    """Wrapper exception for retryable embedding errors."""

    def __init__(self, original: BaseException):
        self.original = original
        super().__init__(str(original))


class EmbeddingService:
    """Service for generating text embeddings using OpenAI or Google Gemini.

    Automatically selects the embedding provider based on config.LLM_PROVIDER.
    Supports retry on transient API errors.

    Args:
        config: AIConfig instance providing API keys and model settings.
    """

    def __init__(self, config: AIConfig) -> None:
        self._config = config
        self._provider = config.LLM_PROVIDER.lower().strip()

        if self._provider == "google":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=config.GOOGLE_EMBEDDING_MODEL,
                google_api_key=config.GOOGLE_API_KEY,
            )
        else:
            from langchain_openai import OpenAIEmbeddings

            self._embeddings = OpenAIEmbeddings(
                model=config.OPENAI_EMBEDDING_MODEL,
                openai_api_key=config.OPENAI_API_KEY,
            )

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate a single embedding vector for the given text.

        Retries up to 3 times with exponential backoff on transient errors.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            EmbeddingGenerationError: If all retries are exhausted.
        """

        @async_retry(
            retry_on=(_RetryableEmbeddingError,),
            config=_EMBEDDING_RETRY_CONFIG,
        )
        async def _embed() -> List[float]:
            try:
                return await self._embeddings.aembed_query(text)
            except (openai.RateLimitError, openai.APIStatusError) as exc:
                if _is_retryable_openai_error(exc):
                    raise _RetryableEmbeddingError(exc) from exc
                raise
            except Exception as exc:
                # For Google errors, check if retryable (429, 500, 503)
                err_str = str(exc).lower()
                if "429" in err_str or "quota" in err_str or "rate" in err_str:
                    raise _RetryableEmbeddingError(exc) from exc
                if "500" in err_str or "503" in err_str or "unavailable" in err_str:
                    raise _RetryableEmbeddingError(exc) from exc
                raise

        try:
            return await _embed()
        except _RetryableEmbeddingError as exc:
            logger.error(
                "Embedding generation failed after retries: %s", str(exc.original)
            )
            raise EmbeddingGenerationError(
                f"Embedding generation failed after retries: {exc.original}"
            ) from exc.original
        except (openai.RateLimitError, openai.APIStatusError) as exc:
            logger.error("Embedding generation failed (non-retryable): %s", str(exc))
            raise EmbeddingGenerationError(
                f"Embedding generation failed: {exc}"
            ) from exc
        except Exception as exc:
            logger.error("Unexpected embedding generation error: %s", str(exc))
            raise EmbeddingGenerationError(
                f"Embedding generation failed: {exc}"
            ) from exc

    async def generate_embeddings_batch(
        self, texts: List[str]
    ) -> List[List[float]]:
        """Generate embedding vectors for a batch of texts in one call.

        Args:
            texts: List of input texts to embed.

        Returns:
            A list of embedding vectors, one per input text.

        Raises:
            EmbeddingGenerationError: If all retries are exhausted.
        """

        @async_retry(
            retry_on=(_RetryableEmbeddingError,),
            config=_EMBEDDING_RETRY_CONFIG,
        )
        async def _embed_batch() -> List[List[float]]:
            try:
                return await self._embeddings.aembed_documents(texts)
            except (openai.RateLimitError, openai.APIStatusError) as exc:
                if _is_retryable_openai_error(exc):
                    raise _RetryableEmbeddingError(exc) from exc
                raise
            except Exception as exc:
                err_str = str(exc).lower()
                if "429" in err_str or "quota" in err_str or "rate" in err_str:
                    raise _RetryableEmbeddingError(exc) from exc
                if "500" in err_str or "503" in err_str or "unavailable" in err_str:
                    raise _RetryableEmbeddingError(exc) from exc
                raise

        try:
            return await _embed_batch()
        except _RetryableEmbeddingError as exc:
            logger.error(
                "Batch embedding generation failed after retries: %s",
                str(exc.original),
            )
            raise EmbeddingGenerationError(
                f"Batch embedding generation failed after retries: {exc.original}"
            ) from exc.original
        except (openai.RateLimitError, openai.APIStatusError) as exc:
            logger.error(
                "Batch embedding generation failed (non-retryable): %s", str(exc)
            )
            raise EmbeddingGenerationError(
                f"Batch embedding generation failed: {exc}"
            ) from exc
        except Exception as exc:
            logger.error(
                "Unexpected batch embedding generation error: %s", str(exc)
            )
            raise EmbeddingGenerationError(
                f"Batch embedding generation failed: {exc}"
            ) from exc
