"""Unit tests for DescriptionImprover."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.ai.circuit_breaker import CircuitBreaker
from app.ai.config import AIConfig
from app.ai.description_improver import DescriptionImprover
from app.ai.exceptions import LLMServiceUnavailableError
from app.ai.models import ImprovementMode
from app.ai.prompt_manager import PromptTemplateManager


@pytest.fixture
def config():
    """Create an AIConfig with test values."""
    return AIConfig(
        OPENAI_API_KEY="test-api-key",
        OPENAI_CHAT_MODEL="gpt-4o-mini",
        PROMPT_TEMPLATES_DIR="app/prompts",
    )


@pytest.fixture
def prompt_manager():
    """Create a real PromptTemplateManager pointing to test prompts."""
    return PromptTemplateManager(templates_dir="app/prompts")


@pytest.fixture
def circuit_breaker():
    """Create a CircuitBreaker for testing."""
    return CircuitBreaker(
        failure_threshold=5,
        window_seconds=60.0,
        cooldown_seconds=30.0,
    )


@pytest.fixture
def improver(config, prompt_manager, circuit_breaker):
    """Create a DescriptionImprover with real dependencies."""
    return DescriptionImprover(
        config=config,
        prompt_manager=prompt_manager,
        circuit_breaker=circuit_breaker,
    )


class TestImprove:
    """Tests for DescriptionImprover.improve()."""

    @pytest.mark.asyncio
    async def test_short_and_crisp_mode_uses_correct_template(self, improver):
        """Test that SHORT_AND_CRISP mode selects the improve_short_and_crisp template."""
        description = "We are hiring a software engineer."
        expected_output = "Hiring: Software Engineer"

        with patch.object(
            improver, "_invoke_llm", new=AsyncMock(return_value=expected_output)
        ) as mock_llm:
            result = await improver.improve(description, ImprovementMode.SHORT_AND_CRISP)

        assert result == expected_output
        # Verify the prompt sent to LLM contains the description
        call_args = mock_llm.call_args[0][0]
        assert description in call_args
        assert "short and crisp" in call_args.lower()

    @pytest.mark.asyncio
    async def test_detailed_and_formal_mode_uses_correct_template(self, improver):
        """Test that DETAILED_AND_FORMAL mode selects the improve_detailed_and_formal template."""
        description = "We need a developer for our team."
        expected_output = "Role Overview: We are seeking a Developer..."

        with patch.object(
            improver, "_invoke_llm", new=AsyncMock(return_value=expected_output)
        ) as mock_llm:
            result = await improver.improve(
                description, ImprovementMode.DETAILED_AND_FORMAL
            )

        assert result == expected_output
        call_args = mock_llm.call_args[0][0]
        assert description in call_args
        assert "formal" in call_args.lower()

    @pytest.mark.asyncio
    async def test_marketing_oriented_mode_uses_correct_template(self, improver):
        """Test that MARKETING_ORIENTED mode selects the improve_marketing_oriented template."""
        description = "Open position for a data analyst."
        expected_output = "Join our team as a Data Analyst!"

        with patch.object(
            improver, "_invoke_llm", new=AsyncMock(return_value=expected_output)
        ) as mock_llm:
            result = await improver.improve(
                description, ImprovementMode.MARKETING_ORIENTED
            )

        assert result == expected_output
        call_args = mock_llm.call_args[0][0]
        assert description in call_args
        assert "marketing" in call_args.lower()

    @pytest.mark.asyncio
    async def test_raises_llm_unavailable_when_circuit_breaker_open(
        self, config, prompt_manager
    ):
        """Test that LLMServiceUnavailableError is raised when circuit breaker is open."""
        cb = CircuitBreaker(failure_threshold=1, window_seconds=60.0, cooldown_seconds=30.0)
        improver = DescriptionImprover(
            config=config, prompt_manager=prompt_manager, circuit_breaker=cb
        )

        # Trip the circuit breaker by recording a failure
        async def failing_call():
            raise RuntimeError("LLM connection failed")

        with pytest.raises(RuntimeError):
            await cb.call(failing_call)

        # Now the circuit should be open, calling improve should raise
        with pytest.raises(LLMServiceUnavailableError):
            await improver.improve("Some description", ImprovementMode.SHORT_AND_CRISP)

    @pytest.mark.asyncio
    async def test_raises_llm_unavailable_on_llm_call_failure(self, improver):
        """Test that LLMServiceUnavailableError is raised when LLM call fails."""
        with patch.object(
            improver, "_invoke_llm", new=AsyncMock(side_effect=RuntimeError("API error"))
        ):
            with pytest.raises(LLMServiceUnavailableError):
                await improver.improve(
                    "A job description.", ImprovementMode.SHORT_AND_CRISP
                )

    @pytest.mark.asyncio
    async def test_returns_llm_response_content(self, improver):
        """Test that the LLM response content is returned as-is."""
        expected = "A beautifully improved description with many details."

        with patch.object(
            improver, "_invoke_llm", new=AsyncMock(return_value=expected)
        ):
            result = await improver.improve(
                "raw description", ImprovementMode.DETAILED_AND_FORMAL
            )

        assert result == expected
