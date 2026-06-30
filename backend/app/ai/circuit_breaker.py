"""Circuit breaker for LLM service calls.

Implements a simple circuit breaker pattern with three states:
- Closed (normal): Requests pass through to the LLM service.
- Open (tripped): After reaching the failure threshold within the time window,
  all requests are rejected immediately for the duration of the cooldown period.
- Half-open: After the cooldown expires, one request is allowed through to test
  if the service has recovered.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Callable

from app.ai.exceptions import LLMServiceUnavailableError

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Possible states of the circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for protecting LLM service calls.

    When the failure threshold is reached within the configured time window,
    the circuit opens and immediately rejects requests for the cooldown period.
    After the cooldown, a single request is allowed through to test recovery.

    Args:
        failure_threshold: Number of consecutive failures to trip the circuit.
        window_seconds: Time window in which failures are counted.
        cooldown_seconds: How long to stay open before transitioning to half-open.
        name: Descriptive name for this circuit breaker instance (for logging).
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        window_seconds: float = 60.0,
        cooldown_seconds: float = 30.0,
        name: str = "llm_circuit_breaker",
    ):
        self.failure_threshold = failure_threshold
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds
        self.name = name

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._first_failure_time: float | None = None
        self._opened_at: float | None = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Current state of the circuit breaker."""
        return self._state

    def _reset(self) -> None:
        """Reset failure tracking to initial state."""
        self._failure_count = 0
        self._first_failure_time = None
        self._opened_at = None
        self._state = CircuitState.CLOSED

    def _record_failure(self) -> None:
        """Record a failure and potentially trip the circuit."""
        now = time.monotonic()

        # If first failure or window expired, start a new window
        if (
            self._first_failure_time is None
            or (now - self._first_failure_time) > self.window_seconds
        ):
            self._failure_count = 1
            self._first_failure_time = now
        else:
            self._failure_count += 1

        # Check if threshold is reached
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = now
            logger.warning(
                "Circuit breaker '%s' OPENED after %d failures within %.0fs.",
                self.name,
                self._failure_count,
                self.window_seconds,
            )

    def _record_success(self) -> None:
        """Record a success, resetting the circuit breaker."""
        if self._state == CircuitState.HALF_OPEN:
            logger.info(
                "Circuit breaker '%s' recovered. Transitioning to CLOSED.",
                self.name,
            )
        self._reset()

    def _should_allow_request(self) -> bool:
        """Determine whether a request should be allowed through.

        Returns:
            True if the request can proceed, False if it should be rejected.
        """
        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            now = time.monotonic()
            # Check if cooldown has elapsed
            if self._opened_at and (now - self._opened_at) >= self.cooldown_seconds:
                self._state = CircuitState.HALF_OPEN
                logger.info(
                    "Circuit breaker '%s' transitioning to HALF_OPEN "
                    "after %.0fs cooldown.",
                    self.name,
                    self.cooldown_seconds,
                )
                return True
            return False

        if self._state == CircuitState.HALF_OPEN:
            # Only one request at a time in half-open state
            return True

        return False

    async def call(self, func: Callable, *args, **kwargs):
        """Execute a function through the circuit breaker.

        Args:
            func: Async callable to execute.
            *args: Positional arguments passed to func.
            **kwargs: Keyword arguments passed to func.

        Returns:
            The result of func(*args, **kwargs).

        Raises:
            LLMServiceUnavailableError: If the circuit is open.
        """
        async with self._lock:
            if not self._should_allow_request():
                raise LLMServiceUnavailableError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"LLM service is temporarily unavailable."
                )

        try:
            result = await func(*args, **kwargs)
            async with self._lock:
                self._record_success()
            return result
        except LLMServiceUnavailableError:
            # Don't double-count our own circuit breaker exceptions
            raise
        except Exception as exc:
            async with self._lock:
                self._record_failure()
            raise exc
