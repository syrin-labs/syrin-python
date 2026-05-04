"""Syrin exception hierarchy."""

from __future__ import annotations

from typing import Any


class SyrinError(Exception):
    """Base exception for all Syrin errors. Catch this for generic handling."""

    pass


class BudgetExceededError(SyrinError):
    """Raised when a budget limit is exceeded.

    Attributes:
        message: Human-readable error message.
        current_cost: Current cost or token count that exceeded the limit.
        limit: The limit that was exceeded.
        budget_type: Which limit was exceeded (str, one of BudgetLimitType values).
    """

    def __init__(  # type: ignore[explicit-any]
        self,
        message: str,
        current_cost: float = 0.0,
        limit: float = 0.0,
        budget_type: str | Any = "run",
    ) -> None:
        super().__init__(message)
        self.current_cost = current_cost
        self.limit = limit
        _bt = budget_type
        self.budget_type: str = _bt.value if hasattr(_bt, "value") else _bt


class BudgetThresholdError(SyrinError):
    """Raised when a budget threshold triggers a stop action.

    When a BudgetThreshold action raises to stop the run,
    this is raised. Use for graceful handling when budget is nearly exhausted.

    Attributes:
        message: Error message.
        threshold_percent: Threshold percentage that was crossed.
        action_taken: Action identifier (e.g. "stop").
    """

    def __init__(
        self,
        message: str,
        threshold_percent: float = 0.0,
        action_taken: str = "",
    ) -> None:
        super().__init__(message)
        self.threshold_percent = threshold_percent
        self.action_taken = action_taken


class ForecastAbortError(BudgetExceededError):
    """Raised when ``Budget(abort_on_forecast_exceeded=True)`` aborts a run.

    Fires when the real-time cost forecast predicts the budget will be exceeded
    by more than ``abort_forecast_multiplier`` × ``max_cost`` before the run
    completes.

    Attributes:
        forecast_p50: The p50 projected total cost at time of abort (USD).
        max_cost: The configured budget limit (USD).
        multiplier: The ``abort_forecast_multiplier`` that was applied.
    """

    def __init__(
        self,
        message: str,
        forecast_p50: float = 0.0,
        max_cost: float = 0.0,
        multiplier: float = 1.0,
    ) -> None:
        super().__init__(message, current_cost=forecast_p50, limit=max_cost)
        self.forecast_p50 = forecast_p50
        self.max_cost = max_cost
        self.multiplier = multiplier


class ModelNotFoundError(SyrinError):
    """Raised when a requested model is not found in the registry.

    Typically when ModelRegistry.resolve() cannot find a model by name.
    """

    pass


class ToolExecutionError(SyrinError):
    """Raised when a tool execution fails.

    Wraps the underlying exception. Check __cause__ for the original error.
    """

    pass


class ToolArgumentError(SyrinError):
    """Raised when LLM-generated tool arguments fail type validation.

    Validates arguments before execution so the tool function receives
    correctly typed inputs. Provides a clear diagnostic message with the
    tool name, parameter name, and expected type.

    Attributes:
        message: Human-readable error describing the mismatch.
        tool_name: Name of the tool whose argument was invalid.
        param_name: Name of the parameter that failed validation.
        expected_type: String representation of the expected type.
        received_type: String representation of the type actually received.
    """

    def __init__(
        self,
        message: str,
        *,
        tool_name: str = "",
        param_name: str = "",
        expected_type: str = "",
        received_type: str = "",
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.param_name = param_name
        self.expected_type = expected_type
        self.received_type = received_type


class TaskError(SyrinError):
    """Raised when a task execution fails.

    Use when AgentTask or similar task orchestration fails.
    """

    pass


class ProviderError(SyrinError):
    """Raised when an LLM provider returns an error.

    API errors, rate limits, auth failures, etc. Check message for details.
    """

    pass


class ProviderNotFoundError(SyrinError):
    """Raised when a requested provider name is not registered.

    Typically a typo in provider= or model_id prefix. Check ModelRegistry.
    """

    pass


class CodegenError(SyrinError):
    """Raised when .syrin DSL code generation fails.

    Parsing or translation errors from Syrin DSL to Python.
    """

    pass


class ValidationError(SyrinError):
    """Raised when structured output validation fails.

    This exception includes information about validation attempts for debugging.

    Attributes:
        message: Error message
        attempts: List of validation attempts made
        last_error: Last error that occurred
    """

    def __init__(
        self,
        message: str,
        attempts: list[str] | None = None,
        last_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts or []
        self.last_error = last_error


class HandoffBlockedError(SyrinError):
    """Raised when handoff is blocked by a before-handler or validation.

    Emitted as Hook.HANDOFF_BLOCKED when a handler blocks the transfer.

    Attributes:
        message: Reason for blocking
        source_agent: Source agent class or name
        target_agent: Target agent class or name
        task: Task that would have been passed
    """

    def __init__(
        self,
        message: str,
        source_agent: str = "",
        target_agent: str = "",
        task: str = "",
    ) -> None:
        super().__init__(message)
        self.source_agent = source_agent
        self.target_agent = target_agent
        self.task = task


class HandoffRetryRequested(SyrinError):
    """Target agent signals: data invalid, please retry with this hint.

    Raise this from the target (or a wrapper) to ask the caller to reformat
    and retry handoff. The caller implements the retry loop.

    Attributes:
        format_hint: Instructions for correct format (e.g. JSON schema, required fields)
    """

    def __init__(self, message: str, format_hint: str = "") -> None:
        super().__init__(message)
        self.format_hint = format_hint or message


class ModalityNotSupportedError(SyrinError):
    """Raised when Agent declares input/output modalities that no router profile supports.

    Use at construction time when input_modalities or output_modalities are provided
    and the router profiles do not cover them.

    Attributes:
        message: Human-readable error.
        required: Modalities that were required.
        supported: Modalities the router profiles actually support.
    """

    def __init__(
        self,
        message: str,
        *,
        required: set[str] | None = None,
        supported: set[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.required = required or set()
        self.supported = supported or set()


class NoMatchingProfileError(SyrinError):
    """Raised when no router profile matches the required task type and modality.

    Attributes:
        message: Human-readable error message.
        required_task_type: TaskType that was required (from classification or override).
        required_modalities: Modalities present in the input that must be supported.
        available_profiles: Profile names that were considered.
    """

    def __init__(
        self,
        message: str,
        *,
        required_task_type: object = None,
        required_modalities: object = None,
        available_profiles: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.required_task_type = required_task_type
        self.required_modalities = required_modalities or set()
        self.available_profiles = available_profiles or []


class CircuitBreakerOpenError(SyrinError):
    """Raised when circuit breaker is open and request is blocked."""

    def __init__(
        self,
        message: str,
        *,
        agent_name: str = "",
        circuit_state: object = None,
        recovery_at: float = 0.0,
        fallback_model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.agent_name = agent_name
        self.circuit_state = circuit_state
        self.recovery_at = recovery_at
        self.fallback_model = fallback_model


class TemplateParseError(SyrinError):
    """Raised when Template.from_file() encounters invalid YAML frontmatter.

    Attributes:
        message: Human-readable description of the parse failure.
        path: Path to the template file that failed to parse.
        line: Line number where the parse error occurred (1-based), or None.
    """

    def __init__(
        self,
        message: str,
        *,
        path: str = "",
        line: int | None = None,
    ) -> None:
        super().__init__(message)
        self.path = path
        self.line = line


class OutputValidationError(SyrinError):
    """Raised when structured output validation fails after all retries.

    Includes the raw LLM response and per-attempt error messages for debugging.

    Attributes:
        message: Human-readable summary.
        raw: Raw LLM response string that failed to validate.
        attempts: Number of validation attempts made.
        errors: Per-attempt error messages (most-recent last).
    """

    def __init__(
        self,
        message: str,
        *,
        raw: str = "",
        attempts: int = 0,
        errors: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.raw = raw
        self.attempts = attempts
        self.errors: list[str] = errors or []


class InputTooLargeError(SyrinError):
    """Raised when agent.run() input exceeds max_input_length.

    Enforces a size ceiling on raw user input to prevent accidental
    serialization of huge payloads to the LLM API.

    Attributes:
        message: Human-readable error.
        input_length: Byte/character length of the rejected input.
        max_length: The configured limit.
    """

    def __init__(
        self,
        message: str,
        *,
        input_length: int = 0,
        max_length: int = 0,
    ) -> None:
        super().__init__(message)
        self.input_length = input_length
        self.max_length = max_length


class ResourceExceededError(SyrinError):
    """Raised when any resource limit configured on a Resource is hit.

    Check the message for which limit was exceeded (max_tools, max_steps,
    max_context, or timeout).

    Attributes:
        message: Human-readable description of the exceeded limit.
        dimension: Which resource dimension was exceeded (optional).
        limit: The configured limit value (optional).
        used: The current usage at the time of the error (optional).
    """

    def __init__(
        self,
        message: str,
        *,
        dimension: str | None = None,
        limit: int | float | None = None,
        used: int | float | None = None,
    ) -> None:
        super().__init__(message)
        self.dimension = dimension
        self.limit = limit
        self.used = used


class ResourcePoolError(SyrinError):
    """Base for all ResourcePool errors."""

    pass


class ResourcePoolFullError(ResourcePoolError):
    """Raised when pool is full and overflow=REJECT.

    Attributes:
        pool_id: Pool identifier.
        dimension: Which limit was hit ('rpm', 'tpm', 'concurrency').
        requested: Amount requested.
        available: Amount available.
    """

    def __init__(
        self,
        msg: str,
        *,
        pool_id: str,
        dimension: str,
        requested: float,
        available: float,
    ) -> None:
        super().__init__(msg)
        self.pool_id = pool_id
        self.dimension = dimension
        self.requested = requested
        self.available = available


class ResourceAllocationError(ResourcePoolError):
    """Raised when an agent cannot be allocated in the pool.

    Attributes:
        agent_id: Agent identifier.
        reason: Why allocation failed.
    """

    def __init__(
        self,
        msg: str,
        *,
        agent_id: str,
        reason: str,
    ) -> None:
        super().__init__(msg)
        self.agent_id = agent_id
        self.reason = reason


class ResourceTimeoutError(ResourceExceededError):
    """Raised when the wall-clock timeout configured on Resource is exceeded.

    Subclass of :class:`ResourceExceededError`; always has
    ``dimension="timeout"``.

    Attributes:
        message: Human-readable description.
        timeout: The configured timeout in seconds.
        elapsed: Elapsed time when the timeout was hit.
    """

    def __init__(
        self,
        message: str,
        *,
        timeout: float | None = None,
        elapsed: float | None = None,
    ) -> None:
        super().__init__(
            message,
            dimension="timeout",
            limit=timeout,
            used=elapsed,
        )
        self.timeout = timeout
        self.elapsed = elapsed


class ConfigurationError(SyrinError):
    """Raised when an invalid configuration key or value is provided to :func:`~syrin.configure`.

    Attributes:
        message: Human-readable description of what went wrong.
        unknown_keys: Set of unrecognised key names that were passed.
        valid_keys: Set of valid key names for reference.

    Example::

        import syrin
        syrin.configure(trace=False)  # OK
        syrin.configure(trce=True)    # raises ConfigurationError: unknown key 'trce'
    """

    def __init__(
        self,
        message: str,
        *,
        unknown_keys: set[str] | None = None,
        valid_keys: frozenset[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.unknown_keys: set[str] = unknown_keys or set()
        self.valid_keys: frozenset[str] = valid_keys or frozenset()
