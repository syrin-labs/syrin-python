"""Swarm — multi-agent swarm orchestration."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import uuid
from typing import TYPE_CHECKING

from syrin.enums import AgentRole, AgentStatus, Hook, PauseMode, SwarmTopology, WorkflowStatus
from syrin.events import EventContext, Events
from syrin.swarm._config import SwarmConfig
from syrin.swarm._result import AgentStatusEntry, SwarmResult
from syrin.workflow._lifecycle import RunHandle
from syrin.workflow.exceptions import WorkflowCancelledError

if TYPE_CHECKING:
    from syrin.agent._core import Agent
    from syrin.budget import Budget
    from syrin.resource._pool import ResourcePool
    from syrin.response import Response
    from syrin.swarm._agent_ref import AgentRef
    from syrin.swarm._control import SwarmController
    from syrin.swarm._memory_bus import MemoryBus
    from syrin.workflow._core import Workflow


class SwarmRunHandle(RunHandle[str]):
    """A :class:`~syrin.workflow._lifecycle.RunHandle` for swarm execution.

    Returned by :meth:`~syrin.swarm.Swarm.play`.  ``wait()`` returns the
    :class:`~syrin.swarm._result.SwarmResult` rather than a bare
    :class:`~syrin.response.Response`.

    Attributes:
        status: Current :class:`~syrin.enums.WorkflowStatus`.
    """

    def __init__(
        self,
        swarm_task: asyncio.Task[SwarmResult],
        run_id: str,
        pause_event: asyncio.Event,
        resume_event: asyncio.Event,
        cancel_event: asyncio.Event,
        pause_mode_ref: list[PauseMode],
    ) -> None:
        """Initialise SwarmRunHandle.

        Args:
            swarm_task: Background task running the swarm executor.
            run_id: Unique run identifier.
            pause_event: Set when a pause is requested.
            resume_event: Set when resume is requested.
            cancel_event: Set when cancel is requested.
            pause_mode_ref: Single-element list holding the current PauseMode.
        """

        # We pass a dummy placeholder task to the base-class; our override of
        # wait() uses _swarm_task directly so the base task is never awaited.
        # The placeholder silently absorbs the swarm task's result/exception so
        # asyncio doesn't log "Task exception was never retrieved".
        async def _placeholder() -> Response[str]:
            try:
                result = await swarm_task
                from syrin.response import Response as _Resp

                return _Resp(content=getattr(result, "content", ""), cost=0.0)
            except Exception:
                from syrin.response import Response as _Resp

                return _Resp(content="", cost=0.0)

        placeholder: asyncio.Task[Response[str]] = asyncio.create_task(_placeholder())
        super().__init__(
            placeholder, run_id, pause_event, resume_event, cancel_event, pause_mode_ref
        )
        self._swarm_task = swarm_task
        self._swarm_ref: Swarm | None = None

    async def wait(self) -> SwarmResult:  # type: ignore[override]
        """Await swarm completion and return the :class:`~syrin.swarm._result.SwarmResult`.

        Raises:
            WorkflowCancelledError: If the swarm was cancelled.
            Exception: Re-raises any unhandled failure from the executor.
        """
        try:
            result = await self._swarm_task
        except asyncio.CancelledError as err:
            self._mark_cancelled()
            raise WorkflowCancelledError("Swarm was cancelled.") from err
        except Exception:
            self._mark_failed()
            raise
        if self._status == WorkflowStatus.CANCELLED:
            raise WorkflowCancelledError("Swarm was cancelled.")
        self._mark_completed()
        return result

    @property
    def controller(self) -> SwarmController:
        """Return a :class:`~syrin.swarm._control.SwarmController` bound to this swarm's live state.

        Use this to pause, resume, or inspect individual agents without
        constructing a controller manually.

        Returns:
            :class:`~syrin.swarm._control.SwarmController` acting as the first
            agent in the swarm (or the first ORCHESTRATOR-role agent).

        Example::

            handle = swarm.play()
            ctrl = handle.controller
            await ctrl.pause_agent(worker)
        """
        if self._swarm_ref is None:
            raise RuntimeError(
                "SwarmRunHandle.controller requires a swarm reference. "
                "Use swarm.play() to obtain this handle."
            )
        return self._swarm_ref._make_controller()

    def controller_for(self, actor: AgentRef | str) -> SwarmController:
        """Return a :class:`~syrin.swarm._control.SwarmController` acting on behalf of *actor*.

        Args:
            actor: The agent instance or ID that will be the actor for all
                control actions issued through the returned controller.

        Returns:
            :class:`~syrin.swarm._control.SwarmController`.
        """
        if self._swarm_ref is None:
            raise RuntimeError("SwarmRunHandle.controller_for() requires a swarm reference.")
        return self._swarm_ref._make_controller(actor=actor)


class Swarm:
    """Multi-agent Swarm for concurrent AI workloads.

    A :class:`Swarm` groups one or more agents under a shared goal and runs
    them according to a :class:`~syrin.swarm._config.SwarmConfig` topology.
    All five topologies are fully supported:

    - **PARALLEL** — agents run concurrently, results merged.
    - **CONSENSUS** — agents vote; majority (or configured fraction) wins.
    - **REFLECTION** — iterative producer-critic refinement loop.
    - **ORCHESTRATOR** — first agent decomposes the goal; workers execute.
    - **WORKFLOW** — wraps a :class:`~syrin.workflow.Workflow` inside a swarm.

    Topology-specific configuration lives inside
    :class:`~syrin.swarm._config.SwarmConfig` via the ``consensus`` and
    ``reflection`` attributes.  This keeps all configuration co-located.

    Attributes:
        goal: Shared goal string passed to every agent.
        budget: Optional shared :class:`~syrin.budget.Budget`.
        config: Swarm configuration including topology and topology-specific
            settings.

    Examples::

        # PARALLEL — all agents run concurrently
        from syrin.swarm import Swarm, SwarmConfig
        from syrin.enums import SwarmTopology

        swarm = Swarm(
            agents=[ResearchAgent(), AnalysisAgent()],
            goal="Summarise AI trends for Q1 2025",
            budget=Budget(max_cost=5.00),
            config=SwarmConfig(topology=SwarmTopology.PARALLEL),
        )
        result = await swarm.run()
        print(result.content)

        # CONSENSUS — agents vote, ≥67% must agree
        from syrin.swarm import ConsensusConfig
        swarm = Swarm(
            agents=[Agent1(), Agent2(), Agent3()],
            goal="Is this clause enforceable?",
            config=SwarmConfig(
                topology=SwarmTopology.CONSENSUS,
                consensus=ConsensusConfig(min_agreement=0.67),
            ),
        )

        # REFLECTION — writer + critic loop
        from syrin.swarm import ReflectionConfig
        swarm = Swarm(
            agents=[WriterAgent(), EditorAgent()],
            goal="Write a technical explanation of vector embeddings",
            config=SwarmConfig(
                topology=SwarmTopology.REFLECTION,
                reflection=ReflectionConfig(
                    producer=WriterAgent,
                    critic=EditorAgent,
                    max_rounds=3,
                    score_threshold=0.85,
                ),
            ),
        )
    """

    def __init__(
        self,
        agents: list[Agent | type[Agent]],
        goal: str,
        budget: Budget | None = None,
        memory: MemoryBus | None = None,
        config: SwarmConfig | None = None,
        pry: bool = False,
        workflow: Workflow | None = None,
        pool: ResourcePool | None = None,
    ) -> None:
        """Initialise a Swarm.

        Args:
            agents: Non-empty list of agent instances or agent classes.  Agent
                classes are instantiated automatically with no arguments.
            goal: Shared goal for all agents.  Must be non-empty.
            budget: Optional budget.  When ``max_cost`` is set, a shared
                :class:`~syrin.budget.BudgetPool` is created automatically.
            memory: Optional :class:`~syrin.swarm._memory_bus.MemoryBus`.
            config: Optional :class:`~syrin.swarm._config.SwarmConfig`.
                Topology-specific configuration (``consensus``, ``reflection``)
                lives here.  A default ORCHESTRATOR config is used if omitted.
            pry: Enable Pry multi-agent debugger (future feature).
            workflow: Optional :class:`~syrin.workflow.Workflow` for
                :attr:`~syrin.enums.SwarmTopology.WORKFLOW` topology runs.
            pool: Optional :class:`~syrin.resource.ResourcePool` for
                swarm-level rate and concurrency coordination.

        Raises:
            ValueError: If ``agents`` is empty or ``goal`` is blank.
        """
        if not agents:
            raise ValueError("Swarm requires at least one agent")
        # Normalize: instantiate any class references
        resolved: list[Agent] = []
        for a in agents:
            if inspect.isclass(a):
                resolved.append(a())
            else:
                resolved.append(a)  # type: ignore[arg-type]
        goal = goal.strip()
        if not goal:
            raise ValueError("Swarm goal must be a non-empty string")

        self.goal: str = goal
        self._agents: list[Agent] = resolved
        self.budget: Budget | None = budget
        self.memory: MemoryBus | None = memory
        self.config: SwarmConfig = config or SwarmConfig()
        self.pry: bool = pry
        self._workflow: Workflow | None = workflow
        self.pool: ResourcePool | None = pool

        # Expand team members from Agent.team ClassVar
        self._expand_team_agents()

        # Lifecycle primitives shared between Swarm and the handle
        self._pause_event: asyncio.Event = asyncio.Event()
        self._resume_event: asyncio.Event = asyncio.Event()
        self._cancel_event: asyncio.Event = asyncio.Event()
        self._resume_event.set()  # start in RUNNING state

        # Per-run state (reset on each play())
        self._handle: SwarmRunHandle | None = None
        self._agent_tasks: dict[str, asyncio.Task[object]] = {}
        self._agent_status: dict[str, AgentStatus] = {
            a.agent_id: AgentStatus.IDLE for a in self._agents
        }
        self._run_id: str = str(uuid.uuid4())

        # Events system — emit_fn is a no-op placeholder; swarm triggers via _trigger directly
        def _noop_emit(hook: Hook, ctx: EventContext) -> None:
            pass

        self.events: Events = Events(_noop_emit)

    # ──────────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def agent_count(self) -> int:
        """Number of agents in the swarm."""
        return len(self._agents)

    @property
    def topology(self) -> SwarmTopology:
        """Execution topology (from :attr:`config`)."""
        return self.config.topology

    @property
    def estimated_cost(self) -> object | None:
        """Pre-flight cost estimate for all agents in the swarm.

        Returns ``None`` when ``estimation=False`` on the budget, or when no
        budget is set.  Access is always synchronous.

        Example::

            swarm = Swarm(
                agents=[agent1, agent2],
                goal="...",
                budget=Budget(max_cost=10.0, estimation=True),
            )
            est = swarm.estimated_cost
            if est is not None:
                print(f"p50=${est.p50:.4f}  p95=${est.p95:.4f}")
        """
        import logging

        from syrin.budget._estimate import CostEstimate
        from syrin.budget._preflight import InsufficientBudgetError
        from syrin.enums import EstimationPolicy

        budget = self.budget
        if budget is None or not budget.estimation:
            return None

        estimator = budget._effective_estimator()
        agent_classes = [type(a) for a in self._agents]
        result: CostEstimate = estimator.estimate_many(agent_classes, budget)

        policy = budget.estimation_policy
        if not result.sufficient:
            if policy == EstimationPolicy.RAISE:
                max_cost = budget.max_cost or 0.0
                raise InsufficientBudgetError(
                    total_p50=result.p50,
                    total_p95=result.p95,
                    budget_configured=max_cost,
                    policy=budget.estimation_policy,
                )
            elif policy == EstimationPolicy.WARN_ONLY:
                logging.getLogger(__name__).warning(
                    "Swarm pre-flight estimation: budget $%.4f may be insufficient "
                    "(p50=$%.4f, p95=$%.4f). Run may exceed budget.",
                    budget.max_cost or 0.0,
                    result.p50,
                    result.p95,
                )

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Run interface
    # ──────────────────────────────────────────────────────────────────────────

    async def run(self) -> SwarmResult:
        """Run the swarm to completion and return the :class:`~syrin.swarm._result.SwarmResult`.

        Convenience wrapper around :meth:`play` + :meth:`~SwarmRunHandle.wait`.

        Returns:
            :class:`~syrin.swarm._result.SwarmResult` with merged content and
            budget information.
        """
        handle = self.play()
        return await handle.wait()

    def run_sync(self) -> SwarmResult:
        """Run the swarm synchronously. Use in scripts and non-async contexts.

        Calls ``asyncio.run(self.run())`` internally. Do NOT call this from inside
        an existing event loop — use ``await swarm.run()`` there instead.

        Returns:
            :class:`~syrin.swarm._result.SwarmResult` with merged content and
            budget information.

        Example:
            swarm = Swarm(agents=[ResearchAgent(), WriterAgent()], goal="summarise AI trends")
            result = swarm.run_sync()
            print(result.content)
        """
        import asyncio as _asyncio

        return _asyncio.run(self.run())

    def play(self) -> SwarmRunHandle:
        """Start swarm execution in the background and return a handle immediately.

        Returns:
            :class:`SwarmRunHandle` for lifecycle control and result retrieval.

        Example::

            handle = swarm.play()
            await swarm.pause()
            ...
            await swarm.resume()
            result = await handle.wait()
        """
        # Reset per-run state
        self._run_id = str(uuid.uuid4())
        self._pause_event.clear()
        self._resume_event.set()
        self._cancel_event.clear()
        self._agent_tasks = {}
        self._agent_status = {a.agent_id: AgentStatus.IDLE for a in self._agents}

        exec_task: asyncio.Task[SwarmResult] = asyncio.create_task(self._execute())
        handle = SwarmRunHandle(
            swarm_task=exec_task,
            run_id=self._run_id,
            pause_event=self._pause_event,
            resume_event=self._resume_event,
            cancel_event=self._cancel_event,
            pause_mode_ref=[PauseMode.AFTER_CURRENT_STEP],
        )
        handle._swarm_ref = self
        self._handle = handle
        return handle

    async def _execute(self) -> SwarmResult:
        """Internal executor dispatched to the correct topology."""
        topology = self.config.topology

        if topology == SwarmTopology.CONSENSUS:
            from syrin.swarm.topologies._consensus import run_consensus

            return await run_consensus(self, self.config.consensus)

        if topology == SwarmTopology.REFLECTION:
            from syrin.swarm.topologies._reflection import run_reflection

            if self.config.reflection is None:
                raise ValueError(
                    "REFLECTION topology requires ReflectionConfig. "
                    "Pass it via SwarmConfig(topology=SwarmTopology.REFLECTION, "
                    "reflection=ReflectionConfig(producer=..., critic=...))"
                )
            return await run_reflection(self, self.config.reflection)

        if topology == SwarmTopology.ORCHESTRATOR:
            from syrin.swarm.topologies._orchestrator import run_orchestrator

            return await run_orchestrator(self, self._agents, self.goal, self.goal)

        if topology == SwarmTopology.WORKFLOW:
            from syrin.swarm.topologies._workflow_topology import run_workflow_topology

            if self._workflow is None:
                raise ValueError("WORKFLOW topology requires a workflow= argument in Swarm()")
            return await run_workflow_topology(self, self._workflow)

        from syrin.swarm.topologies._parallel import run_parallel

        return await run_parallel(self)

    # ──────────────────────────────────────────────────────────────────────────
    # Lifecycle controls
    # ──────────────────────────────────────────────────────────────────────────

    async def pause(self) -> None:
        """Request a pause of the swarm.

        Sets the status to PAUSED.  Agents that are mid-execution will
        complete their current step before pausing takes observable effect.

        When the swarm uses :attr:`~syrin.enums.SwarmTopology.WORKFLOW`
        topology and a :class:`~syrin.workflow.Workflow` was provided, the
        pause is cascaded to the inner workflow as well.
        """
        self._pause_event.set()
        self._resume_event.clear()
        if self._handle is not None:
            self._handle._mark_paused()
        # Cascade to inner workflow if present
        if self._workflow is not None:
            await self._workflow.pause()

    async def resume(self) -> None:
        """Resume a paused swarm.

        Raises:
            WorkflowCancelledError: If the swarm was previously cancelled.
        """
        if self._cancel_event.is_set():
            raise WorkflowCancelledError("Swarm has been cancelled and cannot be resumed.")
        self._pause_event.clear()
        self._resume_event.set()
        if self._handle is not None:
            self._handle._mark_running()

    async def cancel(self) -> None:
        """Cancel the entire swarm, stopping all running agents."""
        self._cancel_event.set()
        if self._handle is not None:
            self._handle._mark_cancelled()
        # Cancel all individual agent tasks
        for task in list(self._agent_tasks.values()):
            task.cancel()
        # Cancel the top-level execution task
        if self._handle is not None:
            self._handle._swarm_task.cancel()

    async def cancel_agent(self, target: AgentRef | str) -> None:
        """Cancel a single agent without stopping the swarm.

        Args:
            target: Agent instance or agent ID string.

        Raises:
            ValueError: If the agent is not registered in this swarm.
        """
        from syrin.swarm._agent_ref import _aid

        target_id = _aid(target)
        known_ids = {a.agent_id for a in self._agents}
        if target_id not in known_ids:
            raise ValueError(
                f"No agent with ID {target_id!r} is registered in this swarm. "
                f"Known agent IDs: {sorted(known_ids)}"
            )
        task = self._agent_tasks.get(target_id)
        if task is not None and not task.done():
            task.cancel()

    def serve(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        **kwargs: object,
    ) -> None:
        """Serve this swarm as an HTTP endpoint. Blocks until stopped.

        Exposes:

        - ``POST /chat`` — accepts ``{"message": "..."}`` and returns the
          swarm result as JSON.
        - ``GET /graph`` — returns the Mermaid graph string (placeholder for
          swarms that embed a workflow).

        Args:
            host: Bind address (default ``"0.0.0.0"``).
            port: HTTP port (default ``8000``).
            **kwargs: Reserved for future options (ignored).

        Raises:
            ImportError: If ``uvicorn`` or ``fastapi`` is not installed.

        Example::

            swarm.serve(port=8080)
        """
        try:
            import uvicorn
            from fastapi import FastAPI
            from fastapi.responses import JSONResponse
        except ImportError as exc:
            raise ImportError(
                "Swarm.serve() requires fastapi and uvicorn. "
                "Install with: uv pip install syrin[serve]"
            ) from exc

        swarm_ref = self
        app = FastAPI(title=f"Syrin Swarm: {self.goal[:40]}")

        @app.post("/chat")
        async def _chat(body: dict[str, object]) -> JSONResponse:
            message = str(body.get("message", swarm_ref.goal))
            swarm_ref.goal = message  # update goal for this request
            result = await swarm_ref.run()
            total_cost = sum(result.cost_breakdown.values())
            return JSONResponse({"content": result.content, "cost": total_cost})

        @app.get("/graph")
        async def _graph() -> JSONResponse:
            if swarm_ref._workflow is not None:
                return JSONResponse({"graph": swarm_ref._workflow.to_mermaid()})
            return JSONResponse({"graph": None, "note": "Graph available for WORKFLOW topology"})

        uvicorn.run(app, host=host, port=port, workers=1)

    # ──────────────────────────────────────────────────────────────────────────
    # Status inspection
    # ──────────────────────────────────────────────────────────────────────────

    def agent_statuses(self) -> list[AgentStatusEntry]:
        """Return a status snapshot for every agent registered in the swarm."""
        return [
            AgentStatusEntry(
                agent_name=type(a).__name__,
                state=self._agent_status.get(a.agent_id, AgentStatus.IDLE),
            )
            for a in self._agents
        ]

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _set_agent_status(self, name: str, status: AgentStatus) -> None:
        """Update the tracked status for *name*."""
        self._agent_status[name] = status

    def visualize(self) -> None:
        """Print a rich summary of the swarm to stdout.

        Shows topology, agent pool (name + model), shared budget, and
        MemoryBus configuration if present.

        Example::

            swarm.visualize()
            # Swarm — PARALLEL topology
            # Agents (2):
            #   ResearchAgent  [gpt-4o-mini]
            #   WriterAgent    [gpt-4o-mini]
            # Budget: $5.00 shared
        """
        try:
            from rich import print as rprint  # noqa: PLC0415
            from rich.tree import Tree  # noqa: PLC0415

            topology_name = self.config.topology.value.upper()
            tree = Tree(f"[bold cyan]Swarm[/bold cyan] — {topology_name} topology")

            # Agent pool
            agents_branch = tree.add(f"[bold]Agents ({len(self._agents)}):[/bold]")
            for agent in self._agents:
                agent_name = type(agent).__name__
                model_id = ""
                cls_model = getattr(type(agent), "model", None)
                if cls_model is not None:
                    model_id = getattr(cls_model, "model_id", str(cls_model))
                if model_id:
                    agents_branch.add(f"[green]{agent_name}[/green] [dim]\\[{model_id}][/dim]")
                else:
                    agents_branch.add(f"[green]{agent_name}[/green]")

            # Budget
            if self.budget is not None:
                max_cost = self.budget.max_cost
                budget_str = f"${max_cost:.2f}" if max_cost is not None else "unlimited"
                tree.add(f"[bold]Budget:[/bold] {budget_str}")

            # MemoryBus
            if self.memory is not None:
                filter_val = getattr(self.memory, "filter", None)
                backend_obj = getattr(self.memory, "backend", None)
                backend_name = type(backend_obj).__name__ if backend_obj is not None else "default"
                filter_str = f"  filter={filter_val}" if filter_val is not None else ""
                tree.add(f"[bold]MemoryBus:[/bold]{filter_str}  backend={backend_name}")

            rprint(tree)

        except ImportError:
            topology_name = self.config.topology.value.upper()
            print(f"Swarm — {topology_name} topology")
            print(f"Agents ({len(self._agents)}):")
            for agent in self._agents:
                print(f"  {type(agent).__name__}")

    def _fire_event(self, hook: Hook, data: dict[str, object]) -> None:
        """Dispatch *hook* to all registered handlers via the Events system."""
        ctx = EventContext(data)
        ctx.scrub()
        self.events._trigger(hook, ctx)

    def _make_controller(self, actor: AgentRef | str | None = None) -> SwarmController:
        """Build a SwarmController bound to this swarm's live registries.

        Args:
            actor: Agent instance or ID acting as the controller's authority
                source. Defaults to the first ORCHESTRATOR or ADMIN role agent,
                falling back to the first agent.

        Returns:
            :class:`~syrin.swarm._control.SwarmController`.
        """
        from syrin.swarm._authority import build_guard_from_agents
        from syrin.swarm._control import AgentStateSnapshot, SwarmController

        if actor is None:
            actor_agent: Agent | None = None
            for a in self._agents:
                r: AgentRole = getattr(type(a), "role", AgentRole.WORKER)
                if r in (AgentRole.ORCHESTRATOR, AgentRole.ADMIN):
                    actor_agent = a
                    break
            if actor_agent is None:
                actor_agent = self._agents[0]
            actor = actor_agent

        # Build authority guard from agent class metadata
        guard = build_guard_from_agents(self._agents)

        # Build state registry from current status
        state_registry: dict[str, AgentStateSnapshot] = {}
        for a in self._agents:
            aid = a.agent_id
            status = self._agent_status.get(aid, AgentStatus.IDLE)
            role: AgentRole = getattr(type(a), "role", AgentRole.WORKER)
            state_registry[aid] = AgentStateSnapshot(
                agent_id=aid,
                status=status,
                role=role,
                last_output_summary="",
                cost_spent=0.0,
                task="",
                context_override=None,
                supervisor_id=getattr(a, "_supervisor_id", None),
            )

        return SwarmController(
            actor_id=actor,
            guard=guard,
            state_registry=state_registry,
            task_registry=self._agent_tasks,
        )

    def _expand_team_agents(self) -> None:
        """Expand ``Agent.team`` class variables into the swarm's agent pool.

        Recursively processes all agents: for each agent with a non-empty
        ``team`` ClassVar, instantiates each team-member class, adds it to
        the pool, records the parent–child relationship for authority
        checking, and sets ``_supervisor_id`` on the member.

        Supports arbitrarily deep hierarchies (e.g., CEO → CTO → Engineer).
        """
        from syrin.agent._core import Agent as _Agent

        # Mapping: parent_agent_id -> [team_member_agent_id, ...]
        self._team_map: dict[str, list[str]] = {}

        # Process a queue so nested teams are also expanded
        queue: list[_Agent] = list(self._agents)
        processed_ids: set[str] = set()

        while queue:
            parent = queue.pop(0)
            parent_id: str = getattr(parent, "agent_id", type(parent).__name__)
            if parent_id in processed_ids:
                continue
            processed_ids.add(parent_id)

            team_classes: list[type[_Agent]] | None = getattr(type(parent), "team", None)
            if not team_classes:
                continue

            children: list[str] = []
            for member_cls in team_classes:
                member: _Agent = member_cls()
                member_id: str = getattr(member, "agent_id", type(member).__name__)
                # Record supervisor relationship
                with contextlib.suppress(AttributeError, TypeError):
                    object.__setattr__(member, "_supervisor_id", parent_id)
                children.append(member_id)
                self._agents.append(member)
                # Queue for further expansion (handles nested teams)
                queue.append(member)
            self._team_map[parent_id] = children
