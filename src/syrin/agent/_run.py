"""Run orchestration and response building for Agent.

Single responsibility: run the loop (via AgentRunContext), handle guardrails,
checkpoints, and build Response from LoopResult. Agent delegates here from
response()/arun() so that agent/__init__.py stays focused on config and identity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from syrin.agent import Agent

from syrin.agent._context_builder import _user_input_to_search_str
from syrin.enums import GuardrailStage, MessageRole, StopReason
from syrin.loop import LoopResult
from syrin.output_format._citation import Citation
from syrin.response import Response
from syrin.types import Message, TokenUsage
from syrin.validation import get_retry_prompt


async def run_agent_loop_async(
    agent: Agent, user_input: str | list[dict[str, object]]
) -> Response[str]:
    """Run the configured loop with full observability and build Response.

    Performs: input guardrails → loop.run(ctx, user_input) → checkpoint →
    output guardrails → build_output → populate report → return Response.
    """
    from syrin.observability import SemanticAttributes, SpanKind

    agent._runtime.grounded_facts.clear()
    input_str = _user_input_to_search_str(user_input)
    with agent._tracer.span(
        f"{agent._agent_name}.response",
        kind=SpanKind.AGENT,
        attributes={
            SemanticAttributes.AGENT_NAME: agent._agent_name,
            SemanticAttributes.AGENT_CLASS: agent.__class__.__name__,
            "input": input_str if not agent._debug else input_str[:1000],
            SemanticAttributes.LLM_MODEL: agent._model_config.model_id,
            SemanticAttributes.LLM_PROVIDER: agent._model_config.provider,
        },
    ) as agent_span:
        # Input guardrails check (use text for guardrails)
        input_guardrail = agent._run_guardrails(input_str, GuardrailStage.INPUT)
        if not input_guardrail.passed:
            return agent._with_context_on_response(
                _guardrail_response(
                    agent,
                    0.0,
                    0.0,
                    TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
                    [],
                ),
            )

        result = await agent._loop.run(agent.run_context, user_input)

        # Auto-checkpoint after step completion
        agent._maybe_checkpoint("step")

        tokens = _tokens_from_result(result)
        agent_span.set_attribute(SemanticAttributes.LLM_TOKENS_TOTAL, tokens.total_tokens)
        agent_span.set_attribute("cost.usd", result.cost_usd)

        if agent._budget is not None:
            agent_span.set_attribute("budget.remaining", agent._budget.remaining)
            agent_span.set_attribute("budget.spent", agent._budget._spent)

        tool_calls_list = _tool_calls_from_result(result)
        if result.tool_calls:
            agent._maybe_checkpoint("tool")

        # Output guardrails check (only if no tool calls)
        if not result.tool_calls:
            output_guardrail = agent._run_guardrails(result.content or "", GuardrailStage.OUTPUT)
            if not output_guardrail.passed:
                agent._last_iteration = result.iterations
                return agent._with_context_on_response(
                    _guardrail_response(
                        agent,
                        result.cost_usd,
                        result.latency_ms / 1000,
                        tokens,
                        tool_calls_list,
                    ),
                )

        # Build structured output with validation; retry via LLM when validation fails
        structured, result_content = await _build_structured_with_llm_retry(
            agent,
            result.content or "",
            user_input,
        )

        # Raise OutputValidationError when all retries exhausted and output still invalid
        if structured is not None and structured.final_error is not None:
            from syrin.exceptions import OutputValidationError

            try:
                from syrin.enums import Hook
                from syrin.events import EventContext

                agent._emit_event(
                    Hook.OUTPUT_VALIDATION_ERROR,
                    EventContext(
                        content=result_content,
                        error=str(structured.final_error),
                    ),
                )
            except Exception as _hook_exc:
                _log.warning(
                    "Hook emission failed during %s event: %s",
                    Hook.OUTPUT_VALIDATION_ERROR,
                    _hook_exc,
                )
            raise OutputValidationError(
                f"Structured output validation failed after "
                f"{getattr(agent, '_validation_retries', 0)} retries. "
                f"Raw: {result_content[:200]!r}. Error: {structured.final_error}"
            )

        # Apply template if output_config has template and we have structured data
        content_for_response: str = result_content or ""
        template_data: dict[str, object] | None = None
        citations_list: list[Citation] = []
        output_config = getattr(agent, "_output_config", None)
        if (
            output_config is not None
            and output_config.template is not None
            and structured is not None
            and structured._data
        ):
            try:
                rendered = output_config.template.render(**structured._data)
                content_for_response = rendered
                template_data = dict(structured._data)
            except ValueError:
                pass  # Slot validation failed; use raw content

        # Apply citation parsing/styling when output_config.citation_style is set
        if output_config is not None and output_config.citation_style is not None:
            from syrin.output_format import apply_citation_to_content

            content_for_response, citations_list = apply_citation_to_content(
                content_for_response, output_config
            )

        # Generate file when output_config.format is set and content is non-empty
        file_path: Path | None = None
        file_bytes_val: bytes | None = None
        if output_config is not None and content_for_response.strip():
            try:
                from syrin.output_format import format_to_file

                file_path, file_bytes_val = format_to_file(
                    content_for_response,
                    output_config.format,
                    title=output_config.title,
                )
            except ImportError:
                pass  # Optional deps (WeasyPrint, python-docx) not installed

        # Populate report with final data
        agent._run_report.budget_remaining = agent._budget.remaining if agent._budget else None
        agent._run_report.budget_used = agent._budget._spent if agent._budget else 0.0
        agent._run_report.tokens.input_tokens = tokens.input_tokens
        agent._run_report.tokens.output_tokens = tokens.output_tokens
        agent._run_report.tokens.total_tokens = tokens.total_tokens
        agent._run_report.tokens.cost_usd = result.cost_usd

        facts = getattr(agent._runtime, "grounded_facts", None)
        if facts:
            from syrin.enums import VerificationStatus
            from syrin.response import GroundingReport

            verified_count = sum(1 for f in facts if f.verification == VerificationStatus.VERIFIED)
            sources = list(dict.fromkeys(f.source for f in facts if f.source))
            agent._run_report.grounding = GroundingReport(
                verified_count=verified_count,
                total_facts=len(facts),
                sources=sources,
            )
        else:
            agent._run_report.grounding = None

        agent._last_iteration = result.iterations

        # Auto-store user input and assistant response as episodic memories
        _auto_store_turn(agent, user_input, result.content)

        return agent._with_context_on_response(
            _response_from_loop_result(
                agent,
                result,
                tokens,
                tool_calls_list,
                structured,
                content_override=content_for_response,
                template_data=template_data,
                file_path=file_path,
                file_bytes=file_bytes_val,
                citations=citations_list,
            ),
        )


async def _build_structured_with_llm_retry(  # type: ignore[explicit-any]
    agent: Agent,
    initial_content: str,
    user_input: str | list[dict[str, object]],
) -> tuple[Any, str]:
    """Build structured output, retrying via LLM with validation error when validation fails.

    Returns:
        (structured_output, content_used) — content_used is the string that was validated
        (may be from a retry round).
    """
    from syrin.agent._prompt_build import build_output

    pydantic_model = _get_structured_output_model(agent)
    if pydantic_model is None:
        out = build_output(
            agent,
            initial_content,
            validation_retries=1,
            validation_context=getattr(agent, "_validation_context", None),
            validator=getattr(agent, "_output_validator", None),
        )
        return (out, initial_content or "")

    content = initial_content or ""
    max_retries = getattr(agent, "_validation_retries", 3)
    structured = None

    # Always do at least one build attempt regardless of retry count
    for attempt in range(max(max_retries, 1)):
        structured = build_output(
            agent,
            content,
            validation_retries=1,
            validation_context=getattr(agent, "_validation_context", None),
            validator=getattr(agent, "_output_validator", None),
        )
        if structured is None or structured.final_error is None:
            return (structured, content)
        if max_retries == 0 or attempt >= max_retries - 1:
            return (structured, content)
        # Emit OUTPUT_VALIDATION_RETRY for each LLM retry attempt
        try:
            from syrin.enums import Hook
            from syrin.events import EventContext

            agent._emit_event(
                Hook.OUTPUT_VALIDATION_RETRY,
                EventContext(
                    attempt=attempt + 1,
                    error=str(structured.final_error),
                    raw=content,
                ),
            )
        except Exception as _hook_exc:
            _log.warning(
                "Hook emission failed during %s event: %s", Hook.OUTPUT_VALIDATION_RETRY, _hook_exc
            )
        retry_prompt = get_retry_prompt(pydantic_model, str(structured.final_error))
        base_messages = agent._build_messages(user_input)
        messages = base_messages + [
            Message(role=MessageRole.ASSISTANT, content=content),
            Message(role=MessageRole.USER, content=retry_prompt),
        ]
        response = await agent.complete(messages, tools=[])
        content = (response.content or "").strip()
        if not content:
            return (structured, content)

    return (structured, content)


def _get_structured_output_model(agent: Agent) -> type | None:
    """Return the Pydantic model for structured output, or None if not set."""
    output_type = getattr(agent._model_config, "output", None)
    if output_type is None:
        return None
    if hasattr(output_type, "_structured_pydantic"):
        return cast("type", output_type._structured_pydantic)
    try:
        from pydantic import BaseModel

        if isinstance(output_type, type) and issubclass(output_type, BaseModel):
            return cast("type", output_type)
    except ImportError:
        pass
    return None


def _auto_store_turn(
    agent: Agent, user_input: str | list[dict[str, object]], assistant_content: str | None
) -> None:
    """Store user input and assistant response as episodic memories when auto_store is enabled."""
    pm = getattr(agent, "_persistent_memory", None)
    if pm is None or not getattr(pm, "auto_store", False):
        return
    if getattr(agent, "_memory_backend", None) is None:
        return
    try:
        from syrin.enums import MemoryType

        text = _user_input_to_search_str(user_input)
        if text and text.strip():
            agent.remember(
                f"User said: {text.strip()}",
                memory_type=MemoryType.HISTORY,
                importance=0.7,
            )
        if assistant_content and assistant_content.strip():
            agent.remember(
                f"Assistant replied: {assistant_content.strip()}",
                memory_type=MemoryType.HISTORY,
                importance=0.6,
            )

        # Emit MEMORY_EXTRACT hook at the auto-extraction entry point for observability.
        # This fires at the natural extraction entry point so tracing/logging tools can pick it up.
        store = getattr(pm, "_store", None)
        if store is not None and hasattr(store, "_emit_extract_hook"):
            combined = " ".join(
                filter(
                    None,
                    [
                        text.strip() if text else None,
                        assistant_content.strip() if assistant_content else None,
                    ],
                )
            )
            store._emit_extract_hook(combined)
    except Exception as exc:
        _log.warning("auto_store failed: %s", exc, exc_info=True)


def _tokens_from_result(result: LoopResult) -> TokenUsage:
    u = result.token_usage or {}
    return TokenUsage(
        input_tokens=u.get("input", 0),
        output_tokens=u.get("output", 0),
        total_tokens=u.get("total", 0),
    )


def _tool_calls_from_result(result: LoopResult) -> list[object]:
    from syrin.types import ToolCall

    out: list[object] = []
    for tc in result.tool_calls or []:
        out.append(
            ToolCall(
                id=tc.get("id", ""),  # type: ignore[arg-type]
                name=tc.get("name", ""),  # type: ignore[arg-type]
                arguments=tc.get("arguments", {}),  # type: ignore[arg-type]
            )
        )
    return out


def _guardrail_response(
    agent: Agent,
    cost_usd: float,
    duration_sec: float,
    tokens: TokenUsage,
    tool_calls: list[object],
) -> Response[str]:
    from syrin.response import Response as ResponseClass

    return ResponseClass(
        content="",
        raw="",
        cost=cost_usd,
        tokens=tokens,
        model=agent._model_config.model_id,
        duration=duration_sec,
        trace=[],
        tool_calls=tool_calls,
        stop_reason=StopReason.GUARDRAIL,
        budget_remaining=agent._budget.remaining if agent._budget else None,
        budget_used=agent._budget._spent if agent._budget else 0.0,
        structured=None,
        report=agent._run_report,
    )


def _response_from_loop_result(
    agent: Agent,
    result: LoopResult,
    tokens: TokenUsage,
    tool_calls_list: list[object],
    structured: object,
    *,
    content_override: str | None = None,
    template_data: dict[str, object] | None = None,
    file_path: Path | None = None,
    file_bytes: bytes | None = None,
    citations: list[Citation] | None = None,
) -> Response[str]:
    from syrin.response import MediaAttachment
    from syrin.response import Response as ResponseClass

    stop_reason = (
        StopReason(result.stop_reason)
        if isinstance(result.stop_reason, str)
        else result.stop_reason
    )
    effective_config = getattr(agent, "_active_model_config", None) or agent._model_config
    model_id = effective_config.model_id
    routing_reason = getattr(agent, "_last_routing_reason", None)
    task_type = routing_reason.task_type if routing_reason is not None else None
    gen_media = getattr(result, "generated_media", None)
    if not isinstance(gen_media, (list, tuple)):
        gen_media = []
    attachments = [
        MediaAttachment(
            type=m["type"],
            content_type=m.get(
                "content_type", "image/png" if m["type"] == "image" else "video/mp4"
            ),
            url=m.get("url"),
        )
        for m in gen_media
        if isinstance(m, dict) and m.get("url")
    ]
    content = content_override if content_override is not None else result.content
    return ResponseClass(
        content=content,
        raw=result.content,
        attachments=attachments,
        cost=result.cost_usd,
        tokens=tokens,
        model=model_id,
        duration=result.latency_ms / 1000,
        tool_calls=tool_calls_list,
        stop_reason=stop_reason,
        budget_remaining=agent._budget.remaining if agent._budget else None,
        budget_used=agent._budget._spent if agent._budget else 0.0,
        iterations=result.iterations,
        structured=structured,  # type: ignore[arg-type]
        report=agent._run_report,
        raw_response=result.raw_response,
        routing_reason=routing_reason,
        model_used=getattr(agent, "_last_model_used", None) or model_id,
        task_type=task_type,
        actual_cost=getattr(agent, "_last_actual_cost", None) or result.cost_usd,
        cost_estimated=getattr(agent, "_last_cost_estimated", None),
        cache_hit=bool(getattr(agent, "_last_cache_hit", False)),
        cache_savings=float(getattr(agent, "_last_cache_savings", 0.0)),
        template_data=template_data,
        file=file_path,
        file_bytes=file_bytes,
        citations=citations or [],
    )
