"""Configuration for agent serving."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from syrin.enums import ServeAs, ServeProtocol


class ServeConfigKwargs(TypedDict, total=False):
    """Keyword arguments for building ServeConfig."""

    serve_as: ServeAs
    protocol: ServeProtocol
    host: str
    port: int
    route_prefix: str
    stream: bool
    include_metadata: bool
    debug: bool
    enable_playground: bool
    enable_discovery: bool | None


@dataclass
class ServeConfig:
    """Configuration for agent.serve() and the HTTP/CLI/STDIO/MCP serving layer.

    Use when calling ``agent.serve(**config)`` or ``agent.serve(config=ServeConfig(...))``.
    All fields have sensible defaults; override only what you need.

    MCP routes are driven by ``syrin.MCP`` in tools — no separate enable_mcp flag.
    Discovery (/.well-known/agent-card.json) is auto-detected when the agent has a name;
    set enable_discovery=False to force off.

    CORS and auth are not handled here. For custom middleware, mount
    ``agent.as_router()`` on your own FastAPI app and add CORSMiddleware,
    OAuth, etc. from Starlette or other libraries.

    Attributes:
        serve_as: What interface to expose. AGENT (default) serves Syrin REST/CLI/STDIO
            endpoints. MCP serves a Model Context Protocol server, making the agent
            callable from Claude Code, Claude Desktop, Cursor, Dify, n8n, and any
            MCP-compatible host. Transport is still controlled by ``protocol``.
        protocol: Which transport to use. HTTP (default) for REST + optional playground,
            CLI for interactive REPL, STDIO for JSON lines over stdin/stdout.
            When serve_as=MCP, STDIO is typical for Claude Code; HTTP for web platforms.
        host: Bind address for HTTP server. "0.0.0.0" allows external connections.
        port: HTTP port. Default 8000.
        route_prefix: Prefix for all routes. E.g. "/agent" → /agent/chat, /agent/stream.
            Empty string = no prefix.
        stream: If True (default), /stream endpoint is available for SSE streaming.
        include_metadata: If True (default), responses include cost, tokens, model, etc.
            Set False for minimal payloads.
        debug: Enable debug mode. When True with enable_playground, collects lifecycle
            events and shows them in the playground. Also enables verbose logging.
        enable_playground: Serve the web playground at /playground when True. Provides
            a simple chat UI for testing. Requires debug=True for event collection.
        enable_discovery: Serve /.well-known/agent-card.json for discovery. None = auto
            (on when agent has name). Set False to disable.
        max_message_length: Max characters for chat message body. Rejects with 413 if
            exceeded. Default 100_000 (~25K tokens). Prevents DoS via oversized payloads.
    """

    serve_as: ServeAs = ServeAs.AGENT
    protocol: ServeProtocol = ServeProtocol.HTTP
    host: str = "0.0.0.0"
    port: int = 8000
    route_prefix: str = ""
    stream: bool = True
    include_metadata: bool = True
    debug: bool = False
    enable_playground: bool = False
    enable_discovery: bool | None = None
    max_message_length: int = 100_000
