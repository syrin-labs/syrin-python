"""Servable base class — shared serve() logic for HTTP and CLI protocols.

Agent and AgentRouter inherit from Servable to get unified serve() behavior
that respects protocol (HTTP or CLI).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union, cast

if TYPE_CHECKING:
    from syrin.agent import Agent
    from syrin.agent.agent_router import AgentRouter
    from syrin.serve.config import ServeConfig

_ServableUnion = Union["Agent", "AgentRouter"]


class Servable:
    """Base class for objects that can be served via HTTP or CLI.

    Provides serve() that branches on protocol:
        - HTTP: FastAPI app + uvicorn (default). Exposes /chat, /stream, /playground, etc.
        - CLI: Interactive REPL (prompt, run, show cost per turn).
        - STDIO: JSON lines over stdin/stdout for programmatic use.

    Agent and AgentRouter inherit from Servable.

    Note:
        HTTP mode uses workers=1 to keep in-memory state (memory, budget) shared
        across requests. Multiple workers would each have separate state.
    """

    def serve(
        self,
        config: ServeConfig | None = None,
        *,
        stdin: object = None,
        stdout: object = None,
        **config_kwargs: object,
    ) -> None:
        """Serve this agent or pipeline via HTTP, CLI, STDIO, or as an MCP server.

        Starts a server (HTTP/MCP) or REPL (CLI) based on serve_as and protocol.
        For HTTP: visit the returned URL for /chat, /stream, or /playground.
        For MCP: the agent is exposed as an MCP tool callable from Claude Code,
        Claude Desktop, Cursor, Dify, n8n, and any MCP-compatible host.

        Args:
            config: Optional ServeConfig instance. If None, uses config_kwargs to build one.
                Pass ``ServeConfig(serve_as=ServeAs.MCP, protocol=ServeProtocol.STDIO)``
                for full control.
            stdin: Input stream for STDIO protocol only. Default: sys.stdin.
            stdout: Output stream for STDIO protocol only. Default: sys.stdout.
            **config_kwargs: Override any ServeConfig field. Common options:
                - serve_as: ServeAs — AGENT (default) or MCP.
                - protocol: ServeProtocol — HTTP (default), CLI, or STDIO.
                - host: str — Bind address for HTTP (default "0.0.0.0").
                - port: int — HTTP port (default 8000).
                - enable_playground: bool — Serve web UI at /playground when True.
                - debug: bool — Enable debug mode (verbose logs, event collection).
                - route_prefix: str — Prefix for routes (e.g. "/agent" → /agent/chat).
                - include_metadata: bool — Include cost, tokens in responses (default True).

        Raises:
            ImportError: If protocol is HTTP and uvicorn is not installed.

        Example:
            >>> agent.serve(port=8000)
            >>> agent.serve(port=8000, enable_playground=True, debug=True)
            >>> agent.serve(protocol=ServeProtocol.CLI)
            >>> agent.serve(protocol=ServeProtocol.STDIO, stdin=stream_in, stdout=stream_out)
            >>> agent.serve(serve_as=ServeAs.MCP)                          # MCP over STDIO
            >>> agent.serve(serve_as=ServeAs.MCP, protocol=ServeProtocol.HTTP, port=8080)
        """
        from syrin.enums import ServeAs, ServeProtocol
        from syrin.serve.config import ServeConfig

        cfg = config if isinstance(config, ServeConfig) else ServeConfig(**config_kwargs)  # type: ignore[arg-type]

        if cfg.serve_as == ServeAs.MCP:
            from syrin.mcp.server import SyrinMCPServer

            mcp = SyrinMCPServer(cast(_ServableUnion, self))
            mcp.serve(
                protocol=cfg.protocol,
                host=cfg.host,
                port=cfg.port,
                stdin=stdin,
                stdout=stdout,
            )
            return

        if cfg.protocol == ServeProtocol.HTTP:
            try:
                import uvicorn
            except ImportError as e:
                raise ImportError(
                    "HTTP serving requires uvicorn. Install with: uv pip install syrin[serve]"
                ) from e
            from syrin.serve.http import create_http_app

            app = create_http_app(cast(_ServableUnion, self), cfg)
            # workers=1: in-memory state (memory, budget) is per-process; multiple
            # workers would each have separate state, breaking memory and cost tracking
            uvicorn.run(app, host=cfg.host, port=cfg.port, workers=1)  # type: ignore[arg-type]

        elif cfg.protocol == ServeProtocol.CLI:
            from syrin.serve.adapter import to_serveable
            from syrin.serve.cli import run_cli_repl

            serveable = to_serveable(cast(_ServableUnion, self))
            run_cli_repl(serveable, cfg)

        elif cfg.protocol == ServeProtocol.STDIO:
            from syrin.serve.adapter import to_serveable
            from syrin.serve.stdio import run_stdio_protocol

            serveable = to_serveable(cast(_ServableUnion, self))
            run_stdio_protocol(serveable, cfg, stdin=stdin, stdout=stdout)  # type: ignore[arg-type]

        else:
            raise ValueError(f"Unknown protocol: {cfg.protocol}")
