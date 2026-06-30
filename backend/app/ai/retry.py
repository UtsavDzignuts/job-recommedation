"""Retry utility with exponential backoff for the AI Intelligence Layer.

Provides a configurable retry decorator for async functions that handles
transient failures in external services (OpenAI, Vector DB).
"""

import asyncio
import functools
import logging
from dataclasses import dataclass
from typing import Callable, Tuple, Type

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior with exponential backoff.

    Attributes:
        max_retries: Maximum number of retry attempts (not counting the initial call).
        base_delay: Base delay in seconds before the first retry.
        max_delay: Maximum delay cap in seconds.
        exponential_base: Multiplier for exponential backoff calculation.
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0

    def get_delay(self, attempt: int) -> float:
        """Calculate the delay for a given attempt number.

        Args:
            attempt: Zero-based attempt index (0 = first retry).

        Returns:
            Delay in seconds, capped at max_delay.
        """
        delay = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)


def async_retry(
    retry_on: Tuple[Type[BaseException], ...] = (Exception,),
    config: RetryConfig | None = None,
) -> Callable:
    """Async retry decorator with exponential backoff.

    Retries the decorated async function on specified exception types,
    using exponential backoff between attempts.

    Args:
        retry_on: Tuple of exception types that trigger a retry.
        config: RetryConfig instance. Uses defaults if not provided.

    Returns:
        Decorated async function with retry behavior.

    Example:
        @async_retry(retry_on=(ConnectionError, TimeoutError))
        async def call_external_api():
            ...
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception: BaseException | None = None

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retry_on as exc:
                    last_exception = exc

                    if attempt >= config.max_retries:
                        logger.error(
                            "Function '%s' failed after %d retries. "
                            "Last error: %s",
                            func.__name__,
                            config.max_retries,
                            str(exc),
                        )
                        raise

                    delay = config.get_delay(attempt)
                    logger.warning(
                        "Function '%s' failed (attempt %d/%d): %s. "
                        "Retrying in %.1fs...",
                        func.__name__,
                        attempt + 1,
                        config.max_retries + 1,
                        str(exc),
                        delay,
                    )
                    await asyncio.sleep(delay)

            # This should never be reached, but just in case
            if last_exception is not None:
                raise last_exception

        return wrapper

    return decorator
