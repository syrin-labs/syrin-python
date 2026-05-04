"""HTTP serving — FastAPI routes for /chat, /stream, /health, /ready, /budget, /describe, /config, /.well-known/agent-card.json."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import suppress
from typing import TYPE_CHECKING

_log = logging.getLogger("syrin.serve")

if TYPE_CHECKING:
    from syrin.agent import Agent
    from syrin.agent.agent_router import AgentRouter
    from syrin.serve.config import ServeConfig


def _add_startup_endpoint_logging(app: object, config: ServeConfig | None = None) -> None:
    """Add startup event that prints endpoints with methods."""

    @app.on_event("startup")  # type: ignore[attr-defined, untyped-decorator]
    def _log_endpoints() -> None:
        lines: list[str] = ["Syrin endpoints:"]
        has_mcp = False
        for route in app.routes:  # type: ignore[attr-defined]
            if hasattr(route, "methods") and hasattr(route, "path"):
                methods = ", ".join(sorted(m for m in route.methods if m != "HEAD"))
                lines.append(f"  {methods:6} {route.path}")
                if "/mcp" in (route.path or ""):
                    has_mcp = True
        _log.info("\n".join(lines))
        if config is not None and config.protocol.value == "http":
            _log.warning(
                "Serving agent without authentication. Add auth middleware for production. "
                "See docs: https://syrin.dev/docs/serving#auth"
            )
        if has_mcp:
            import sys

            from syrin.mcp.stdio import _syrin_cli_message

            use_color = getattr(sys.stdout, "isatty", lambda: False)()
            _log.info(_syrin_cli_message(use_color=use_color))


def _to_json_serializable(obj: object) -> object:
    """Convert config values to JSON-serializable form to avoid Pydantic serializer warnings."""
    if obj is None or isinstance(obj, (int, float, str, bool)):
        return obj
    if hasattr(obj, "value") and not hasattr(obj, "model_dump"):
        return obj.value
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_serializable(x) for x in obj]
    return obj


def _ensure_serve_deps() -> None:
    """Ensure FastAPI and uvicorn are installed. Raise with install hint if not."""
    try:
        import fastapi  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "HTTP serving requires FastAPI. Install with: uv pip install syrin[serve]"
        ) from e


def build_router(
    agent: Agent | AgentRouter,
    config: ServeConfig,
) -> object:
    """Build a FastAPI APIRouter for the given agent or router.

    Accepts Agent or AgentRouter. AgentRouter is wrapped in an adapter that
    implements arun, astream, events, budget_state, etc.

    Routes: POST /chat, POST /stream, GET /health, GET /ready, GET /budget, GET /describe.
    With enable_playground: GET /playground, GET /stream (SSE).

    Args:
        agent: Agent, Pipeline, or DynamicPipeline to serve.
        config: ServeConfig (protocol, host, port, enable_playground, etc.).

    Returns:
        FastAPI APIRouter. Mount with app.include_router(router, prefix="/agent").

    Requires syrin[serve] (fastapi, uvicorn).
    """
    _ensure_serve_deps()
    from syrin.serve.adapter import to_serveable

    agent = to_serveable(agent)
    from fastapi import APIRouter, Body
    from fastapi.responses import JSONResponse, StreamingResponse

    from syrin.response import Response

    router = APIRouter()

    prefix = (config.route_prefix or "").strip().rstrip("/")
    if prefix and not prefix.startswith("/"):
        prefix = "/" + prefix

    def _route(path: str) -> str:
        return f"{prefix}{path}" if prefix else path

    def _message_length(msg: str | list[dict[str, object]]) -> int:
        """Total character length of message (str or sum of content parts)."""
        if isinstance(msg, str):
            return len(msg)
        total = 0
        for p in msg:
            if isinstance(p, dict):
                if "text" in p and isinstance(p["text"], str):
                    total += len(p["text"])
                if "image_url" in p and isinstance(p["image_url"], dict):
                    url = p["image_url"].get("url", "")
                    total += len(url) if isinstance(url, str) else 0
        return total

    def _chat_body(r: dict[str, object]) -> tuple[str | list[dict[str, object]], str | None]:
        """Extract message (str or multimodal content parts) and conversation_id."""
        message = r.get("message") or r.get("input") or r.get("content")
        if isinstance(message, str):
            msg = message.strip()
            return msg, r.get("conversation_id")  # type: ignore[return-value]
        if isinstance(message, list) and all(isinstance(p, dict) for p in message):
            return message, r.get("conversation_id")  # type: ignore[return-value]
        return "", None

    collect_debug = config.debug and config.enable_playground
    if collect_debug:
        from syrin.serve.playground import _attach_event_collector

        _attach_event_collector(agent)

    max_len = config.max_message_length

    @router.post(_route("/chat"))
    async def chat(body: dict[str, object] | None = Body(default=None)) -> object:  # noqa: B008
        """Run agent and return full response. POST body: {message: str}."""
        msg, conversation_id = _chat_body(body or {})
        if not msg:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing 'message', 'input', or 'content' in body"},
            )
        if _message_length(msg) > max_len:
            return JSONResponse(
                status_code=413,
                content={
                    "error": f"Message exceeds {max_len} characters. "
                    "Reduce message size or increase max_message_length in ServeConfig."
                },
            )
        start = time.perf_counter()
        try:
            if collect_debug:
                from syrin.serve.playground import _collect_events

                with _collect_events() as events_list:
                    r: Response[str] = await agent.arun(msg)
            else:
                r = await agent.arun(msg)
                events_list = []
            elapsed = time.perf_counter() - start
            out: dict[str, object] = {"content": str(r.content)}
            attachments = getattr(r, "attachments", None) or []
            if attachments:
                out["attachments"] = [
                    {"type": a.type, "url": a.url, "content_type": getattr(a, "content_type", "")}
                    for a in attachments
                    if getattr(a, "url", None)
                ]
            if config.include_metadata:
                out["cost"] = r.cost
                out["tokens"] = {
                    "input": r.tokens.input_tokens,
                    "output": r.tokens.output_tokens,
                    "total": r.tokens.total_tokens,
                }
                out["model"] = r.model
                out["stop_reason"] = str(r.stop_reason)
                out["duration"] = round(elapsed, 4)
            if collect_debug and events_list:
                out["events"] = [{"hook": h, "ctx": c} for h, c in events_list]
            return JSONResponse(content=out)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)},
            )

    @router.post(_route("/stream"))
    async def stream(body: dict[str, object] | None = Body(default=None)) -> object:  # noqa: B008
        """Stream response as SSE. POST body: {message: str}."""
        msg, _ = _chat_body(body or {})
        if not msg:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing 'message', 'input', or 'content' in body"},
            )
        if _message_length(msg) > max_len:
            return JSONResponse(
                status_code=413,
                content={
                    "error": f"Message exceeds {max_len} characters. "
                    "Reduce message size or increase max_message_length in ServeConfig."
                },
            )

        def _emit(d: dict[str, object]) -> str:
            return f"data: {json.dumps(d)}\n\n"

        def _emit_budget() -> object:
            """Emit budget event when metadata enabled and agent has budget."""
            if not config.include_metadata:
                return  # type: ignore[return-value]
            state = agent.budget_state
            if state is None:
                return  # type: ignore[return-value]
            return _emit(
                {
                    "type": "budget",
                    "limit": state.limit,
                    "remaining": state.remaining,
                    "spent": state.spent,
                    "percent_used": state.percent_used,
                }
            )

        async def sse_gen() -> object:
            import logging

            _log = logging.getLogger("syrin.serve")
            accumulated = ""
            events_list: list[tuple[str, dict[str, object]]] = []
            tokens_val: dict[str, object] | None = None
            attachments_from_result: list[dict[str, object]] = []
            progressive = collect_debug
            run_result: object = None

            # When agent has tools, run the full REACT loop so tool calls are executed and
            # the user gets the real reply instead of the "model chose to use a tool" fallback.
            # Use _tools (internal list) so we never skip the loop when tools exist but are filtered.
            agent_tools = getattr(agent, "_tools", None) or getattr(agent, "tools", None)
            if agent_tools and len(agent_tools) > 0:
                if progressive:
                    yield _emit({"type": "status", "message": "Thinking…"})
                from syrin.serve.playground import _collect_events

                try:
                    with _collect_events() as evts:
                        result = await agent.arun(msg)
                except Exception as run_err:
                    _log.exception("Agent run failed: %s", run_err)
                    if progressive:
                        yield _emit(
                            {
                                "type": "error",
                                "error": str(run_err),
                                "observability": "Error logged; check trace/debug output",
                            }
                        )
                    yield _emit({"type": "error", "error": str(run_err)})
                    return
                run_result = result
                events_list = list(evts)
                accumulated = result.content or ""
                atts = getattr(result, "attachments", None) or []
                attachments_from_result = [
                    {
                        "type": a.type,
                        "url": getattr(a, "url", None),
                        "content_type": getattr(a, "content_type", ""),
                    }
                    for a in atts
                    if getattr(a, "url", None)
                ]
                if accumulated:
                    if progressive:
                        yield _emit(
                            {"type": "text", "text": accumulated, "accumulated": accumulated}
                        )
                    else:
                        yield _emit({"text": accumulated, "accumulated": accumulated})
                    if (b := _emit_budget()) is not None:
                        yield b
                if hasattr(agent, "record_conversation_turn") and accumulated:
                    agent.record_conversation_turn(msg, accumulated)
                if config.include_metadata and getattr(result, "tokens", None):
                    t = result.tokens
                    tokens_val = (
                        {
                            "input_tokens": t.input_tokens,
                            "output_tokens": t.output_tokens,
                            "total_tokens": t.total_tokens,
                        }
                        if t
                        else None
                    )
            elif collect_debug:
                from syrin.serve.playground import _collect_events

                with _collect_events() as evts:
                    if progressive:
                        yield _emit({"type": "status", "message": "Thinking…"})
                    last_event_idx = 0
                    async for chunk in agent.astream(msg):
                        # Support _hook: chunks from DynamicPipeline adapter (real-time hooks)
                        hook_data = getattr(chunk, "_hook", None)
                        if hook_data is not None:
                            h, c = hook_data
                            # evts already updated by collector (adapter.events) in pipeline thread
                            if progressive:
                                yield _emit({"type": "hook", "hook": h, "ctx": c})
                            if (b := _emit_budget()) is not None:
                                yield b
                            continue
                        while last_event_idx < len(evts):
                            h, c = evts[last_event_idx]
                            last_event_idx += 1
                            if progressive:
                                yield _emit({"type": "hook", "hook": h, "ctx": c})
                        text = getattr(chunk, "text", "") or getattr(chunk, "content", "")
                        accumulated += text
                        if progressive:
                            yield _emit({"type": "text", "text": text, "accumulated": accumulated})
                        else:
                            yield _emit({"text": text, "accumulated": accumulated})
                        if (b := _emit_budget()) is not None:
                            yield b
                    while last_event_idx < len(evts):
                        h, c = evts[last_event_idx]
                        last_event_idx += 1
                        if progressive:
                            yield _emit({"type": "hook", "hook": h, "ctx": c})
                    if hasattr(agent, "record_conversation_turn") and accumulated:
                        agent.record_conversation_turn(msg, accumulated)
                    events_list = list(evts)
                    for _h, c in reversed(evts):
                        if isinstance(c, dict) and "tokens" in c:
                            t = c["tokens"]
                            tokens_val = (
                                t
                                if isinstance(t, dict)
                                else (
                                    {
                                        "input_tokens": getattr(t, "input_tokens", 0),
                                        "output_tokens": getattr(t, "output_tokens", 0),
                                        "total_tokens": getattr(t, "total_tokens", 0),
                                    }
                                )
                            )
                            break
            else:
                async for chunk in agent.astream(msg):
                    text = getattr(chunk, "text", "") or getattr(chunk, "content", "")
                    accumulated += text
                    yield _emit({"text": text, "accumulated": accumulated})
                    if (b := _emit_budget()) is not None:
                        yield b

            if hasattr(agent, "record_conversation_turn") and accumulated:
                agent.record_conversation_turn(msg, accumulated)

            done: dict[str, object] = {"done": True}
            if progressive:
                done["type"] = "done"
            if config.include_metadata:
                state = agent.budget_state
                if state is not None:
                    done["cost"] = state.spent
                    done["budget_remaining"] = state.remaining
                    done["budget"] = {
                        "limit": state.limit,
                        "remaining": state.remaining,
                        "spent": state.spent,
                        "percent_used": state.percent_used,
                    }
                elif run_result is not None and hasattr(run_result, "cost"):
                    done["cost"] = run_result.cost
                elif events_list:
                    for _h, c in reversed(events_list):
                        if isinstance(c, dict) and "cost" in c and c["cost"] is not None:
                            done["cost"] = c["cost"]
                            break
                if tokens_val:
                    done["tokens"] = tokens_val
            if collect_debug and events_list:
                done["events"] = [{"hook": h, "ctx": c} for h, c in events_list]
            if attachments_from_result:
                done["attachments"] = attachments_from_result
            yield _emit(done)

        return StreamingResponse(
            sse_gen(),  # type: ignore[arg-type]
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.get(_route("/health"))
    async def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}

    @router.get(_route("/ready"))
    async def ready() -> dict[str, bool]:
        """Readiness probe (agent initialized, model reachable). Minimal check for now."""
        return {"ready": True}

    @router.get(_route("/budget"))
    async def budget() -> object:
        """Budget state if configured."""
        state = agent.budget_state
        if state is None:
            return JSONResponse(status_code=404, content={"error": "No budget configured"})
        return JSONResponse(
            content={
                "limit": state.limit,
                "remaining": state.remaining,
                "spent": state.spent,
                "percent_used": state.percent_used,
            }
        )

    @router.get(_route("/describe"))
    async def describe() -> dict[str, object]:
        """Runtime introspection: name, tools (full specs), budget, internal_agents, setup_type."""
        tools_list: list[dict[str, object]] = [
            {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.parameters_schema or {},
            }
            for t in agent.tools
        ]
        budget_state = agent.budget_state
        out: dict[str, object] = {
            "name": agent.name,
            "description": agent.description,
            "tools": tools_list,
            "budget": (
                {
                    "limit": budget_state.limit,
                    "remaining": budget_state.remaining,
                    "spent": budget_state.spent,
                    "percent_used": budget_state.percent_used,
                }
                if budget_state is not None
                else None
            ),
        }
        internal = getattr(agent, "internal_agents", None)
        if internal:
            out["internal_agents"] = internal
            out["setup_type"] = "dynamic_pipeline"
        return out

    # Remote config: GET/PATCH /config, GET /config/stream (override store + baseline for scale/revert)
    from syrin.remote._registry import get_registry
    from syrin.remote._resolver import ConfigResolver
    from syrin.remote._schema import extract_agent_schema
    from syrin.remote._types import ConfigOverride, OverridePayload

    _config_stream_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    _config_last_version: list[int] = [0]  # Mutable so stream can read; updated on PATCH
    _resolver = ConfigResolver()

    def _get_baseline_overrides_current(
        a: object,
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        """Return (baseline, overrides, current) from agent override store. Freeze baseline on first use."""
        reg = get_registry()
        rt = getattr(a, "_runtime", None)
        baseline: dict[str, object]
        overrides: dict[str, object]
        if rt is None:
            baseline, overrides = {}, {}
        else:
            baseline = rt.remote_baseline
            overrides = rt.remote_overrides or {}
        if baseline is None:
            schema = reg.get_schema(reg.make_agent_id(a)) or extract_agent_schema(a)
            rt.remote_baseline = dict(schema.current_values)
            baseline = rt.remote_baseline
        current = {**baseline, **overrides}
        return baseline, overrides, current

    def _enrich_sections_with_values(
        sections: dict[str, object],
        baseline: dict[str, object],
        overrides: dict[str, object],
        current: dict[str, object],
    ) -> None:
        """Mutate section fields in-place to add baseline_value, current_value, overridden."""
        for sec in sections.values():
            for field in sec.get("fields") or []:  # type: ignore[attr-defined]
                path = field.get("path", "")
                field["baseline_value"] = baseline.get(path)
                field["current_value"] = current.get(path)
                field["overridden"] = path in overrides
                for child in field.get("children") or []:
                    p = child.get("path", "")
                    child["baseline_value"] = baseline.get(p)
                    child["current_value"] = current.get(p)
                    child["overridden"] = p in overrides

    @router.get(_route("/config"))
    async def get_config_route() -> object:
        """Return agent config schema, baseline (code) values, overrides, and current (baseline+overrides). Per-field baseline_value, current_value, overridden for dashboard."""
        reg = get_registry()
        agent_id = reg.make_agent_id(agent)
        schema = reg.get_schema(agent_id) or extract_agent_schema(agent)
        baseline, overrides, current = _get_baseline_overrides_current(agent)
        baseline_js = {k: _to_json_serializable(v) for k, v in baseline.items()}
        overrides_js = {k: _to_json_serializable(v) for k, v in overrides.items()}
        current_js = {k: _to_json_serializable(v) for k, v in current.items()}
        out = schema.model_copy(
            update={
                "agent_id": agent_id,
                "baseline_values": baseline_js,
                "overrides": overrides_js,
                "current_values": current_js,
            }
        ).model_dump(mode="json")
        _enrich_sections_with_values(out["sections"], baseline_js, overrides_js, current_js)
        return out

    @router.patch(_route("/config"))
    async def patch_config_route(body: dict[str, object] | None = Body(default=None)) -> object:  # noqa: B008
        """Apply config overrides. Body: OverridePayload (agent_id, version, overrides). value=null means revert to baseline."""
        if not body:
            return JSONResponse(status_code=400, content={"error": "Missing body"})
        try:
            payload = OverridePayload.model_validate(body)
        except Exception as e:
            return JSONResponse(status_code=422, content={"error": str(e)})
        reg = get_registry()
        agent_id = reg.make_agent_id(agent)
        if payload.agent_id != agent_id:
            return JSONResponse(
                status_code=400,
                content={"error": f"agent_id mismatch: expected {agent_id}"},
            )
        # Ensure baseline exists (e.g. PATCH before first GET)
        _get_baseline_overrides_current(agent)
        overrides_store = agent._runtime.remote_overrides
        # Update store: value is None -> revert (remove from overrides); else set override
        for ov in payload.overrides:
            if ov.value is None:
                overrides_store.pop(ov.path, None)
            else:
                overrides_store[ov.path] = ov.value
        # Sync agent state to baseline + overrides
        schema = reg.get_schema(agent_id) or extract_agent_schema(agent)
        baseline = agent._runtime.remote_baseline or {}
        current = {**baseline, **overrides_store}
        # Build sync payload: skip None and normalize invalid enum values so resolver never false-rejects
        from syrin.remote._resolver import _field_for_path
        from syrin.remote._resolver_helpers import _normalize_enum_value

        overrides_list: list[object] = []
        for p, v in current.items():
            if v is None:
                continue
            field = _field_for_path(schema, p)
            if (
                field
                and field.enum_values is not None
                and isinstance(v, str)
                and v not in field.enum_values
            ):
                v = _normalize_enum_value(p, v, field)
                if v is None:
                    continue
            overrides_list.append(ConfigOverride(path=p, value=v))
        sync_payload = OverridePayload(
            agent_id=agent_id,
            version=payload.version,
            overrides=overrides_list,  # type: ignore[arg-type]
        )
        result = _resolver.apply_overrides(agent, sync_payload, schema=schema)
        # Roll back rejected paths from the store so GET reflects actual agent state
        for path, _ in result.rejected:
            overrides_store.pop(path, None)
        # Notify stream subscribers and track version for heartbeat
        _config_last_version[0] = payload.version
        with suppress(asyncio.QueueFull):
            _config_stream_queue.put_nowait(
                {
                    "agent_id": agent_id,
                    "version": payload.version,
                    "overrides": [o.model_dump() for o in payload.overrides],
                }
            )
        return {
            "accepted": result.accepted,
            "rejected": result.rejected,
            "pending_restart": result.pending_restart,
        }

    @router.get(_route("/config/stream"))
    async def config_stream_route() -> object:
        """SSE stream for config updates (dashboard subscribes; events on PATCH)."""

        async def stream_events() -> object:
            # Send initial heartbeat so clients (and tests) get a first chunk immediately
            yield f'event: heartbeat\ndata: {{"version": {_config_last_version[0]}}}\n\n'
            while True:
                try:
                    msg = await asyncio.wait_for(_config_stream_queue.get(), timeout=30.0)
                    if msg is None:
                        break
                    yield f"event: override\ndata: {json.dumps(msg)}\n\n"
                except TimeoutError:
                    yield f'event: heartbeat\ndata: {{"version": {_config_last_version[0]}}}\n\n'

        return StreamingResponse(
            stream_events(),  # type: ignore[arg-type]
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # MCP co-location — when agent has MCP in tools, mount /mcp
    mcp_instances = getattr(agent, "_mcp_instances", []) or []
    if mcp_instances:
        from syrin.mcp.http import build_mcp_router

        mcp_router = build_mcp_router(mcp_instances[0])
        mcp_prefix = _route("/mcp").rstrip("/")
        router.include_router(mcp_router, prefix=mcp_prefix)  # type: ignore[arg-type]

    # A2A Agent Card (/.well-known/agent-card.json) — discovery for agent-to-agent
    from syrin.serve.discovery import (
        AGENT_CARD_PATH,
        build_agent_card_json,
        should_enable_discovery,
    )

    if should_enable_discovery(agent, config):
        host = getattr(config, "host", "0.0.0.0")
        port = getattr(config, "port", 8000)
        display_host = "localhost" if host == "0.0.0.0" else host
        base_url = f"http://{display_host}:{port}"
        if config.route_prefix:
            base_url = base_url.rstrip("/") + "/" + config.route_prefix.strip("/").lstrip("/")
        if prefix:
            base_url = (base_url.rstrip("/") + "/" + prefix.lstrip("/")).rstrip("/")

        @router.get(_route(AGENT_CARD_PATH))
        async def agent_card() -> dict[str, object]:
            """A2A Agent Card for discovery. GET /.well-known/agent-card.json."""
            emit = getattr(agent, "_emit_event", None)
            if emit is not None:
                from syrin.enums import Hook
                from syrin.events import EventContext

                emit(
                    Hook.DISCOVERY_REQUEST,
                    EventContext(
                        {
                            "agent_name": getattr(agent, "name", ""),
                            "path": AGENT_CARD_PATH,
                        }
                    ),
                )
            return build_agent_card_json(agent, base_url=base_url)

    # Playground — when enable_playground=True
    if config.enable_playground:
        from syrin.serve.playground import _playground_static_dir, get_playground_html

        # API base for playground fetch URLs
        api_base = prefix.rstrip("/") if prefix else ""
        agents_data = [{"name": agent.name, "description": agent.description}]

        @router.get(_route("/playground/config"))
        async def playground_config() -> dict[str, object]:
            """Playground config: apiBase, agents, debug, setup_type."""
            setup_type = "single"
            if len(agents_data) > 1:
                setup_type = "multi"
            elif len(agents_data) == 1 and agents_data[0].get("name") == "dynamic-pipeline":
                setup_type = "dynamic_pipeline"
            return {
                "apiBase": api_base or "/",
                "agents": agents_data,
                "debug": config.debug,
                "setup_type": setup_type,
            }

        # Static files must be mounted on the main app (router.mount is not transferred).
        # agent.serve() and AgentRouter.serve() call add_playground_static_mount().
        static_dir = _playground_static_dir()
        if static_dir is None:
            from fastapi.responses import HTMLResponse

            @router.get(_route("/playground"), response_class=HTMLResponse)
            async def playground() -> str:
                """Web playground (inline fallback when Next.js build not found)."""
                return get_playground_html(
                    base_path=_route("/playground"),
                    api_base=api_base,
                    agents=agents_data,  # type: ignore[arg-type]
                    debug=config.debug,
                )

    return router


def create_http_app(
    obj: Agent | AgentRouter,
    config: ServeConfig,
) -> object:
    """Create a FastAPI app for an agent or AgentRouter.

    Used internally by agent.serve() / router.serve(). Mounts the router
    and optionally the playground. Use for custom app setup.

    Args:
        obj: Agent or AgentRouter.
        config: ServeConfig.

    Returns:
        FastAPI app instance.
    """
    from fastapi import FastAPI

    from syrin.serve.adapter import to_serveable
    from syrin.serve.playground import add_playground_static_mount

    serveable = to_serveable(obj)
    app = FastAPI(
        title=f"Syrin: {serveable.name}",
        description=getattr(serveable, "description", "") or "",
    )
    router = build_router(obj, config)
    app.include_router(router)  # type: ignore[arg-type]
    if config.enable_playground:
        prefix = (config.route_prefix or "").strip().rstrip("/")
        mount_path = f"/{prefix}/playground" if prefix else "/playground"
        add_playground_static_mount(app, mount_path)
    _add_startup_endpoint_logging(app, config)
    return app
