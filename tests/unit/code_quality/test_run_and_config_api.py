"""Tests for run() and config API: valid/invalid inputs, return types, edge cases."""

from __future__ import annotations

from unittest.mock import patch

import syrin
from syrin import get_config
from syrin.budget import Budget
from syrin.enums import StopReason
from syrin.response import Response


def _mock_agent_response(*args: object, **kwargs: object) -> Response:
    """Return a minimal Response for run() tests without calling the real API."""
    return Response(
        content="mocked",
        cost=0.0,
        tokens=0,
        stop_reason=StopReason.END_TURN,
    )


# -----------------------------------------------------------------------------
# run() — valid inputs
# -----------------------------------------------------------------------------


def test_run_returns_response_type() -> None:
    """run() must return Response (typed)."""
    with patch("syrin.agent.Agent") as MockAgent:
        MockAgent.return_value.run.return_value = _mock_agent_response()
        result = syrin.run("Hello", model="openai/gpt-4o-mini")
    assert isinstance(result, Response)
    assert result.content == "mocked"


def test_run_accepts_tools_none() -> None:
    """run() must accept tools=None (default)."""
    with patch("syrin.agent.Agent") as MockAgent:
        MockAgent.return_value.run.return_value = _mock_agent_response()
        result = syrin.run("Hi", model="openai/gpt-4o-mini", tools=None)
    assert isinstance(result, Response)


def test_run_accepts_tools_empty_list() -> None:
    """run() must accept tools=[]."""
    with patch("syrin.agent.Agent") as MockAgent:
        MockAgent.return_value.run.return_value = _mock_agent_response()
        result = syrin.run("Hi", model="openai/gpt-4o-mini", tools=[])
    assert isinstance(result, Response)


def test_run_accepts_list_of_tool_specs() -> None:
    """run() must accept tools as list of ToolSpec (e.g. from @tool)."""
    from syrin.tool import ToolSpec

    # Minimal ToolSpec-compatible: name, description, parameters_schema, func
    spec = ToolSpec(
        name="dummy",
        description="",
        parameters_schema={},
        func=lambda: "ok",
    )
    with patch("syrin.agent.Agent") as MockAgent:
        MockAgent.return_value.run.return_value = _mock_agent_response()
        result = syrin.run("Hi", model="openai/gpt-4o-mini", tools=[spec])
    assert isinstance(result, Response)


def test_run_accepts_budget_optional() -> None:
    """run() must accept budget=Budget(...) or None."""
    budget = Budget(per_run=1.0, on_exceeded=lambda _: None)
    with patch("syrin.agent.Agent") as MockAgent:
        MockAgent.return_value.run.return_value = _mock_agent_response()
        result = syrin.run("Hi", model="openai/gpt-4o-mini", budget=budget)
    assert isinstance(result, Response)


# -----------------------------------------------------------------------------
# run() — edge cases
# -----------------------------------------------------------------------------


def test_run_with_system_prompt() -> None:
    """run() must accept system_prompt kwarg."""
    with patch("syrin.agent.Agent") as MockAgent:
        MockAgent.return_value.run.return_value = _mock_agent_response()
        result = syrin.run(
            "Say OK",
            model="openai/gpt-4o-mini",
            system_prompt="You are a helper. Reply with one word.",
        )
    assert isinstance(result, Response)
    assert hasattr(result, "content")


def test_run_response_has_contract_attributes() -> None:
    """Response from run() must have content, cost, tokens, stop_reason."""
    with patch("syrin.agent.Agent") as MockAgent:
        MockAgent.return_value.run.return_value = _mock_agent_response()
        result = syrin.run("2+2", model="openai/gpt-4o-mini")
    assert hasattr(result, "content")
    assert hasattr(result, "cost")
    assert hasattr(result, "tokens")
    assert hasattr(result, "stop_reason")


# -----------------------------------------------------------------------------
# Config API — valid
# -----------------------------------------------------------------------------


def test_get_config_returns_same_instance() -> None:
    """get_config() must return the same global instance."""
    c1 = get_config()
    c2 = syrin.get_config()
    assert c1 is c2


def test_configure_accepts_valid_kwargs() -> None:
    """configure() must accept trace, default_model, default_api_key."""
    syrin.configure(trace=False)
    config = get_config()
    assert config.trace is False


def test_config_get_with_default() -> None:
    """config.get(key, default) must return default for unknown key."""
    config = get_config()
    value = config.get("nonexistent_key", "default_value")
    assert value == "default_value"


def test_config_set_updates_attributes() -> None:
    """config.set(**kwargs) must update only known attributes."""
    config = get_config()
    original_trace = config.trace
    config.set(trace=not original_trace)
    assert config.trace is not original_trace
    config.set(trace=original_trace)


# -----------------------------------------------------------------------------
# Config API — invalid / edge
# -----------------------------------------------------------------------------


def test_configure_with_unknown_key_raises() -> None:
    """configure() with unknown keys must raise ConfigurationError."""
    import pytest

    from syrin.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="unknown_key"):
        syrin.configure(unknown_key="ignored")  # type: ignore[call-arg]


def test_config_default_model_property() -> None:
    """GlobalConfig.default_model must be ModelConfig | None."""
    config = get_config()
    dm = config.default_model
    assert dm is None or hasattr(dm, "provider") and hasattr(dm, "model_id")
