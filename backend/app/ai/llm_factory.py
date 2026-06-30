"""LLM Factory for the AI Intelligence Layer.

Provides the correct LangChain chat model (OpenAI or Google Gemini)
based on configuration.
"""

from langchain_core.language_models.chat_models import BaseChatModel

from app.ai.config import AIConfig


def create_chat_llm(config: AIConfig, temperature: float = 0.0) -> BaseChatModel:
    """Create and return a chat LLM instance based on config.LLM_PROVIDER.

    Args:
        config: AIConfig instance with provider settings.
        temperature: Temperature for generation (0.0 = deterministic).

    Returns:
        A LangChain chat model instance (ChatOpenAI or ChatGoogleGenerativeAI).
    """
    provider = config.LLM_PROVIDER.lower().strip()

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.GOOGLE_CHAT_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=temperature,
        )
    else:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.OPENAI_CHAT_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=temperature,
        )
