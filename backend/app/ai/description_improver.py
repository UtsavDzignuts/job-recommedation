"""Description Improver for the AI Intelligence Layer.

Uses mode-specific prompt templates and LLM to rewrite job descriptions
in different styles (short_and_crisp, detailed_and_formal, marketing_oriented).
"""

import logging
from typing import Dict

from app.ai.circuit_breaker import CircuitBreaker
from app.ai.config import AIConfig
from app.ai.exceptions import LLMServiceUnavailableError
from app.ai.llm_factory import create_chat_llm
from app.ai.models import ImprovementMode
from app.ai.prompt_manager import PromptTemplateManager

logger = logging.getLogger(__name__)

# Mapping from ImprovementMode enum to prompt template names
_MODE_TEMPLATE_MAP: Dict[ImprovementMode, str] = {
    ImprovementMode.SHORT_AND_CRISP: "improve_short_and_crisp",
    ImprovementMode.DETAILED_AND_FORMAL: "improve_detailed_and_formal",
    ImprovementMode.MARKETING_ORIENTED: "improve_marketing_oriented",
}


class DescriptionImprover:
    """Rewrites job descriptions using mode-specific LLM prompts.

    Depends on:
        - PromptTemplateManager: for loading and rendering prompt templates
        - CircuitBreaker: for protecting LLM calls
        - AIConfig: for LLM model configuration

    Each improvement mode maps to a distinct prompt template that instructs the
    LLM to rewrite the description in a specific style.
    """

    def __init__(
        self,
        config: AIConfig,
        prompt_manager: PromptTemplateManager,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        """Initialize the DescriptionImprover.

        Args:
            config: AI configuration with OpenAI settings.
            prompt_manager: Manager for loading and rendering prompt templates.
            circuit_breaker: Circuit breaker to protect LLM calls.
        """
        self._config = config
        self._prompt_manager = prompt_manager
        self._circuit_breaker = circuit_breaker
        self._llm = create_chat_llm(config)

    async def improve(self, description: str, mode: ImprovementMode) -> str:
        """Improve a job description using the specified mode.

        Pipeline:
            1. Map mode enum to template name
            2. Render mode-specific prompt template with job_description variable
            3. Send rendered prompt to LLM via ChatOpenAI (using circuit breaker)
            4. Return the LLM response content as the improved description string

        Args:
            description: The raw job description text to improve.
            mode: The improvement mode determining the rewriting style.

        Returns:
            The improved job description text.

        Raises:
            LLMServiceUnavailableError: If the circuit breaker is open or
                the LLM call fails.
        """
        # Step 1: Map mode to template name
        template_name = _MODE_TEMPLATE_MAP[mode]

        # Step 2: Render the prompt template with the description
        rendered_prompt = self._prompt_manager.render(
            template_name, job_description=description
        )

        # Step 3: Send to LLM via circuit breaker
        try:
            response = await self._circuit_breaker.call(
                self._invoke_llm, rendered_prompt
            )
        except LLMServiceUnavailableError:
            raise
        except Exception as exc:
            logger.error(
                "LLM call failed during description improvement: %s", exc
            )
            raise LLMServiceUnavailableError(
                "LLM service is temporarily unavailable"
            ) from exc

        # Step 4: Return the improved text
        return response

    async def _invoke_llm(self, prompt: str) -> str:
        """Invoke the LLM with the given prompt and return the response content.

        Args:
            prompt: The rendered prompt string to send to the LLM.

        Returns:
            The LLM response content as a string.
        """
        response = await self._llm.ainvoke(prompt)
        return str(response.content)
