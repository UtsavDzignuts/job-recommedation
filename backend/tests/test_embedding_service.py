"""Unit tests for EmbeddingService."""

import pytest
from unittest.mock import AsyncMock, patch

import openai
from httpx import Request, Response
from langchain_openai import OpenAIEmbeddings

from app.ai.config import AIConfig
from app.ai.embedding_service import EmbeddingService
from app.ai.exceptions import EmbeddingGenerationError


@pytest.fixture
def config():
    """Create an AIConfig with test values."""
    return AIConfig(
        OPENAI_API_KEY="test-api-key",
        OPENAI_EMBEDDING_MODEL="text-embedding-ada-002",
    )


@pytest.fixture
def service(config):
    """Create an EmbeddingService with mocked internals."""
    return EmbeddingService(config)


def _make_rate_limit_error():
    request = Request(method="POST", url="https://api.openai.com/v1/embeddings")
    response = Response(status_code=429, request=request)
    return openai.RateLimitError(
        message="Rate limit exceeded",
        response=response,
        body=None,
    )


def _make_api_status_error(status_code: int):
    request = Request(method="POST", url="https://api.openai.com/v1/embeddings")
    response = Response(status_code=status_code, request=request)
    return openai.APIStatusError(
        message=f"Error {status_code}",
        response=response,
        body=None,
    )


class TestGenerateEmbedding:
    @pytest.mark.asyncio
    async def test_returns_embedding_vector(self, service):
        """Test successful single embedding generation."""
        expected = [0.1, 0.2, 0.3, 0.4]
        with patch.object(
            OpenAIEmbeddings, "aembed_query", new=AsyncMock(return_value=expected)
        ) as mock_embed:
            result = await service.generate_embedding("hello world")

        assert result == expected
        mock_embed.assert_called_once_with("hello world")

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit_error(self, service):
        """Test that RateLimitError triggers retry."""
        rate_limit_error = _make_rate_limit_error()
        expected = [0.1, 0.2, 0.3]

        mock = AsyncMock(side_effect=[rate_limit_error, rate_limit_error, expected])
        with patch.object(OpenAIEmbeddings, "aembed_query", new=mock):
            with patch("app.ai.retry.asyncio.sleep", new_callable=AsyncMock):
                result = await service.generate_embedding("test")

        assert result == expected
        assert mock.call_count == 3

    @pytest.mark.asyncio
    async def test_retries_on_500_status_error(self, service):
        """Test that APIStatusError with 500 triggers retry."""
        server_error = _make_api_status_error(500)
        expected = [0.5, 0.6]

        mock = AsyncMock(side_effect=[server_error, expected])
        with patch.object(OpenAIEmbeddings, "aembed_query", new=mock):
            with patch("app.ai.retry.asyncio.sleep", new_callable=AsyncMock):
                result = await service.generate_embedding("test")

        assert result == expected
        assert mock.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_embedding_error_after_all_retries_exhausted(self, service):
        """Test that EmbeddingGenerationError is raised after 3 retries fail."""
        rate_limit_error = _make_rate_limit_error()

        # Initial call + 3 retries = 4 total calls
        mock = AsyncMock(side_effect=[rate_limit_error] * 4)
        with patch.object(OpenAIEmbeddings, "aembed_query", new=mock):
            with patch("app.ai.retry.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(EmbeddingGenerationError):
                    await service.generate_embedding("test")

        assert mock.call_count == 4

    @pytest.mark.asyncio
    async def test_does_not_retry_on_non_retryable_status(self, service):
        """Test that a 401 error is not retried."""
        request = Request(method="POST", url="https://api.openai.com/v1/embeddings")
        response = Response(status_code=401, request=request)
        auth_error = openai.AuthenticationError(
            message="Invalid API key",
            response=response,
            body=None,
        )

        mock = AsyncMock(side_effect=auth_error)
        with patch.object(OpenAIEmbeddings, "aembed_query", new=mock):
            with pytest.raises(EmbeddingGenerationError):
                await service.generate_embedding("test")

        # Should fail immediately without retry
        assert mock.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_502_503_504(self, service):
        """Test retry on gateway errors (502, 503, 504)."""
        errors = [_make_api_status_error(code) for code in [502, 503, 504]]
        expected = [1.0, 2.0, 3.0]

        mock = AsyncMock(side_effect=[*errors, expected])
        with patch.object(OpenAIEmbeddings, "aembed_query", new=mock):
            with patch("app.ai.retry.asyncio.sleep", new_callable=AsyncMock):
                result = await service.generate_embedding("test")

        assert result == expected
        assert mock.call_count == 4


class TestGenerateEmbeddingsBatch:
    @pytest.mark.asyncio
    async def test_returns_batch_embeddings(self, service):
        """Test successful batch embedding generation."""
        expected = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        texts = ["hello", "world", "test"]

        mock = AsyncMock(return_value=expected)
        with patch.object(OpenAIEmbeddings, "aembed_documents", new=mock):
            result = await service.generate_embeddings_batch(texts)

        assert result == expected
        mock.assert_called_once_with(texts)

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit_error(self, service):
        """Test that batch operation retries on RateLimitError."""
        rate_limit_error = _make_rate_limit_error()
        expected = [[0.1], [0.2]]

        mock = AsyncMock(side_effect=[rate_limit_error, expected])
        with patch.object(OpenAIEmbeddings, "aembed_documents", new=mock):
            with patch("app.ai.retry.asyncio.sleep", new_callable=AsyncMock):
                result = await service.generate_embeddings_batch(["a", "b"])

        assert result == expected
        assert mock.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_embedding_error_after_all_retries_exhausted(self, service):
        """Test EmbeddingGenerationError after exhausting retries in batch."""
        server_error = _make_api_status_error(500)

        mock = AsyncMock(side_effect=[server_error] * 4)
        with patch.object(OpenAIEmbeddings, "aembed_documents", new=mock):
            with patch("app.ai.retry.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(EmbeddingGenerationError):
                    await service.generate_embeddings_batch(["a", "b"])

        assert mock.call_count == 4

    @pytest.mark.asyncio
    async def test_handles_empty_list(self, service):
        """Test batch with empty list."""
        mock = AsyncMock(return_value=[])
        with patch.object(OpenAIEmbeddings, "aembed_documents", new=mock):
            result = await service.generate_embeddings_batch([])

        assert result == []

    @pytest.mark.asyncio
    async def test_does_not_retry_on_non_retryable_error(self, service):
        """Test that 403 is not retried in batch mode."""
        request = Request(method="POST", url="https://api.openai.com/v1/embeddings")
        response = Response(status_code=403, request=request)
        forbidden_error = openai.PermissionDeniedError(
            message="Forbidden",
            response=response,
            body=None,
        )

        mock = AsyncMock(side_effect=forbidden_error)
        with patch.object(OpenAIEmbeddings, "aembed_documents", new=mock):
            with pytest.raises(EmbeddingGenerationError):
                await service.generate_embeddings_batch(["test"])

        assert mock.call_count == 1
