"""Prompt/build use case: resolve system prompt, build messages, build output.

Agent delegates to functions here. Public API stays on Agent.
"""

from __future__ import annotations

import inspect
from contextlib import suppress
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from syrin.agent import Agent

from datetime import UTC

from syrin.agent._context_builder import build_messages as build_messages_for_llm
from syrin.enums import Hook
from syrin.events import EventContext
from syrin.prompt import Prompt, make_prompt_context
from syrin.response import StructuredOutput
from syrin.types import Message


def effective_template_variables(
    agent: Agent, call_vars: dict[str, object] | None = None
) -> dict[str, object]:
    """Return merged template_variables: class + instance + call."""
    class_tv = getattr(agent.__class__, "_syrin_default_template_vars", None) or {}
    merged = {**class_tv, **agent._template_vars}
    if call_vars:
        merged = {**merged, **call_vars}
    if agent._inject_template_vars:
        builtins = get_prompt_builtins(agent)
        for k, v in builtins.items():
            if k not in merged:
                merged[k] = v
    return merged


def get_prompt_builtins(agent: Agent) -> dict[str, object]:
    """Return built-in vars (date, agent_id, conversation_id)."""
    from datetime import datetime

    agent_id = getattr(agent, "_agent_name", None) or agent.__class__.__name__
    conversation_id = getattr(agent, "_conversation_id", None)
    return {
        "date": datetime.now(UTC),
        "agent_id": agent_id,
        "conversation_id": conversation_id,
    }


def resolve_system_prompt(agent: Agent, prompt_vars: dict[str, object], ctx: object) -> str:
    """Resolve system prompt from source (str, Prompt, callable, or @system_prompt method)."""
    source = getattr(agent.__class__, "_syrin_system_prompt_method", None)
    if source is None:
        source = agent._system_prompt_source
    if source is None or source == "":
        return ""
    if isinstance(source, str):
        return source
    if isinstance(source, Prompt):
        variables = source.variables
        var_names = [getattr(v, "name", "") for v in variables]
        filtered = {k: v for k, v in prompt_vars.items() if k in var_names}
        try:
            return str(source(**filtered))
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Prompt {getattr(source, 'name', 'unknown')} missing required params. "
                f"Pass via Agent(template_variables={{...}}) or response(..., template_variables={{...}}). {e}"
            ) from e
    if callable(source):
        sig = inspect.signature(source)
        params = list(sig.parameters.keys())
        bound = source
        first_param = params[0] if params else None
        if first_param == "self" and hasattr(source, "__get__"):
            with suppress(AttributeError, TypeError):
                bound = source.__get__(agent, type(agent))
        if len(params) >= 2 and params[1] == "ctx":
            result = bound(ctx)
        elif len(params) == 1:
            if params[0] == "ctx":
                result = bound(ctx)
            elif params[0] == "self":
                result = bound()
            else:
                filtered = {k: v for k, v in prompt_vars.items() if k in params}
                result = bound(**filtered)
        else:
            filtered = {k: v for k, v in prompt_vars.items() if k in params}
            result = bound(**filtered)
        if not isinstance(result, str):
            raise TypeError(f"System prompt callable must return str, got {type(result).__name__}")
        return result
    return ""


def build_messages(agent: Agent, user_input: str | list[dict[str, object]]) -> list[Message]:
    """Build message list for LLM from user input and agent state."""

    def get_capacity() -> object:
        model_for_context = agent._model if agent._model is not None else None
        call_ctx = getattr(agent, "_call_context", None)
        if call_ctx is not None:
            return call_ctx.get_capacity(model_for_context)
        if hasattr(agent._context, "context"):
            return agent._context.context.get_capacity(model_for_context)
        from syrin.context import Context

        return Context().get_capacity(model_for_context)

    call_tv = getattr(agent, "_call_template_vars", None) or {}
    effective_vars = effective_template_variables(agent, call_vars=call_tv)
    conversation_id = getattr(agent, "_conversation_id", None)
    ctx = make_prompt_context(
        agent, conversation_id=conversation_id, inject_template_vars=agent._inject_template_vars
    )
    emit = getattr(agent, "_emit_event", None)
    if emit:
        emit(
            Hook.SYSTEM_PROMPT_BEFORE_RESOLVE,
            EventContext(
                template_variables=effective_vars,
                source=getattr(agent.__class__, "_syrin_system_prompt_method", None)
                or agent._system_prompt_source,
            ),
        )
    resolved = resolve_system_prompt(agent, effective_vars, ctx)

    guardrail_instructions = ""
    guardrails = getattr(agent, "_guardrails", None)
    if guardrails is not None and hasattr(guardrails, "system_prompt_instructions"):
        guardrail_instructions = guardrails.system_prompt_instructions()
    if guardrail_instructions:
        resolved = f"{resolved}\n\n{guardrail_instructions}" if resolved else guardrail_instructions

    if emit:
        emit(Hook.SYSTEM_PROMPT_AFTER_RESOLVE, EventContext(resolved=resolved))

    call_ctx = getattr(agent, "_call_context", None)
    if call_ctx is None and hasattr(agent, "_context") and agent._context is not None:
        call_ctx = getattr(agent._context, "context", None)
    return build_messages_for_llm(
        user_input,
        system_prompt=resolved,
        tools=agent.tools,  # type: ignore[arg-type]
        conversation_memory=None,
        memory_backend=agent._memory_backend,
        persistent_memory=agent._persistent_memory,
        context_manager=agent._context,
        get_capacity=get_capacity,  # type: ignore[arg-type]
        call_context=call_ctx,
        tracer=agent._tracer,
        inject=getattr(agent, "_call_inject", None),
        inject_source_detail=getattr(agent, "_call_inject_source_detail", None),
    )


def build_output(
    agent: Agent,
    content: str,
    validation_retries: int = 3,
    validation_context: dict[str, object] | None = None,
    validator: object = None,
) -> StructuredOutput | None:
    """Build structured output from response content with validation."""
    from syrin.validation import ValidationPipeline

    output_type = getattr(agent._model_config, "output", None)
    if output_type is None:
        return None

    pydantic_model = None
    if hasattr(output_type, "_structured_pydantic"):
        pydantic_model = output_type._structured_pydantic
    if pydantic_model is None:
        try:
            from pydantic import BaseModel

            if isinstance(output_type, type) and issubclass(output_type, BaseModel):
                pydantic_model = output_type
        except Exception:
            pass

    if pydantic_model is None:
        import json

        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return StructuredOutput(raw=content, parsed=None, _data={})
        return StructuredOutput(
            raw=content, parsed=data, _data=data if isinstance(data, dict) else {}
        )

    context = validation_context or {}
    emit_fn = getattr(agent, "_emit_event", None)

    pipeline = ValidationPipeline(
        output_type=pydantic_model,
        max_retries=validation_retries,
        context=context,
        validator=validator,  # type: ignore[arg-type]
        emit_fn=emit_fn,
    )

    parsed, attempts, error = pipeline.validate(content)

    agent._run_report.output.validated = True
    agent._run_report.output.attempts = len(attempts)
    agent._run_report.output.is_valid = error is None and parsed is not None
    agent._run_report.output.final_error = str(error) if error else None

    return StructuredOutput(
        raw=content,
        parsed=parsed,
        _data=parsed.model_dump() if parsed else {},
        validation_attempts=attempts,
        final_error=error,
    )
