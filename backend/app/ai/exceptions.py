"""Custom exceptions for the AI Intelligence Layer.

All exceptions inherit from a base AIIntelligenceLayerError for easy
catch-all handling at the API boundary.
"""


class AIIntelligenceLayerError(Exception):
    """Base exception for all AI Intelligence Layer errors."""

    pass


class LLMServiceUnavailableError(AIIntelligenceLayerError):
    """Raised when the OpenAI API is unreachable or the circuit breaker is open."""

    def __init__(self, message: str = "LLM service is temporarily unavailable"):
        self.message = message
        super().__init__(self.message)


class VectorDBUnavailableError(AIIntelligenceLayerError):
    """Raised when the vector database connection fails."""

    def __init__(self, message: str = "Vector database is unavailable"):
        self.message = message
        super().__init__(self.message)


class MissingVariableError(AIIntelligenceLayerError):
    """Raised when a prompt template variable is missing or empty."""

    def __init__(self, template_name: str, variable_name: str):
        self.template_name = template_name
        self.variable_name = variable_name
        self.message = (
            f"Missing required variable '{variable_name}' "
            f"for template '{template_name}'"
        )
        super().__init__(self.message)


class EmbeddingGenerationError(AIIntelligenceLayerError):
    """Raised when embedding generation fails after all retries are exhausted."""

    def __init__(self, message: str = "Embedding generation failed after retries"):
        self.message = message
        super().__init__(self.message)
