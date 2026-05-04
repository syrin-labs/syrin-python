<p align="center">
  <img src="https://raw.githubusercontent.com/syrin-labs/syrin-python/main/assets/logo/syrin-logo.png" alt="Syrin" width="200">
</p>

<h1 align="center">The AI Harness for Python Developers</h1>

<p align="center">
  Wrap any LLM with budget control, memory, observability, sandboxed execution,<br>
  multi-agent orchestration, and guardrails — in one Python class.
</p>

<p align="center">
  <a href="https://pypi.org/project/syrin/"><img src="https://img.shields.io/pypi/v/syrin.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/syrin/"><img src="https://img.shields.io/pypi/pyversions/syrin.svg" alt="Python"></a>
  <a href="https://github.com/syrin-labs/syrin-python/blob/main/LICENSE"><img src="https://img.shields.io/github/license/syrin-labs/syrin-python.svg" alt="License"></a>
  <a href="https://github.com/syrin-labs/syrin-python/stargazers"><img src="https://img.shields.io/github/stars/syrin-labs/syrin-python.svg?style=social" alt="Stars"></a>
  <a href="https://discord.gg/BJUdNBh4TC"><img src="https://img.shields.io/badge/Discord-Join%20Community-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
</p>

<p align="center">
  <a href="https://syrin.ai">Website</a> ·
  <a href="https://docs.syrin.dev/agent-kit/">Docs</a> ·
  <a href="https://discord.gg/BJUdNBh4TC">Discord</a> ·
  <a href="https://reddit.com/r/syrin_ai">Reddit</a> ·
  <a href="https://www.youtube.com/@syrin_dev">YouTube</a>
</p>

---

## What is an AI Harness?

A harness gives you control over something powerful. It does not limit what the horse can do — it channels that power precisely where you need it, safely, predictably, every time.

LLMs are extraordinarily capable. They are also unpredictable, expensive, opaque, and stateless out of the box. A raw LLM call is a power source with no circuit breaker, no meter, and no safety rail.

**Syrin is the harness.** It wraps any LLM — OpenAI, Anthropic, Google, Ollama, or your own — with everything a Python developer needs to build production AI systems:

- **Hard cost limits** that actually stop execution before the bill becomes a problem
- **Persistent memory** across sessions with four distinct types and pluggable backends
- **Isolated code execution** so LLM-generated code never runs in your process
- **72+ lifecycle hooks** so nothing in your agent is hidden from you
- **Multi-agent orchestration** with recursive decomposition and shared budget pools
- **Guardrails** for PII, content safety, output validation, and prompt injection defense

You write Python classes. Syrin handles the production concerns. Ship faster, safer, and with full visibility.

---

## Install

```bash
pip install syrin
```

---

## Five Minutes to a Production Agent

```python
from syrin import Agent, Budget, Model
from syrin.enums import ExceedPolicy

class Analyst(Agent):
    model  = Model.OpenAI("gpt-4o-mini", api_key="sk-...")
    budget = Budget(max_cost=0.10, exceed_policy=ExceedPolicy.STOP)
    system_prompt = "You are a precise financial analyst."

result = Analyst().run("Summarise Q3 revenue trends")

print(result.content)
print(f"Cost:      \${result.cost:.6f}")
print(f"Tokens:    {result.tokens.total_tokens}")
print(f"Remaining: \${result.budget_remaining:.4f}")
```

```
Q3 revenue grew 14% YoY, driven by enterprise deals (+22%) offsetting
consumer softness (-3%). Gross margin held at 71%...

Cost:      \$0.000312
Tokens:    284
Remaining: \$0.0997
```

The agent hard-stops at \$0.10. No surprise invoices. No extra code.

---

## What the Harness Gives You

### Budget Control — Runtime Enforcement, Not Just Monitoring

Every other library treats cost as a logging concern. Syrin treats it as a **runtime constraint**. The agent checks its budget before every LLM call.

```python
from syrin import Agent, Budget, Model, RateLimit
from syrin.enums import ExceedPolicy
from syrin.budget import BudgetThreshold

class ProductionAgent(Agent):
    model  = Model.OpenAI("gpt-4o", api_key="sk-...")
    budget = Budget(
        max_cost=1.00,                        # Hard cap per run
        reserve=0.10,                         # Hold back for the final reply
        exceed_policy=ExceedPolicy.STOP,      # STOP | WARN | IGNORE | SWITCH
        rate_limits=RateLimit(
            hour=10.00,                       # \$10/hour ceiling
            day=100.00,                       # \$100/day ceiling
            month=2000.00,                    # \$2,000/month ceiling
        ),
        thresholds=[
            BudgetThreshold(at=80, action=lambda ctx: alert_ops(ctx)),
        ],
    )
```

Pre-call estimation, post-call actuals, threshold callbacks, rate-window enforcement — all declarative, zero boilerplate. The \$47K runaway-agent incident? A `Budget(max_cost=50)` would have been a \$50 error.

---

### Sandboxed Execution — LLM Code Never Runs in Your Process

When your agent generates and runs code, it should not run in the same process as your application. `Sandbox` spawns a fresh subprocess for every execution — isolated memory, hard timeout, no shared state.

```python
import asyncio
from syrin.sandbox import Sandbox

async def main():
    async with Sandbox(python=True, bash=True, timeout=30, memory_mb=256) as sb:

        # Execute Python — in a fresh isolated subprocess
        result = await sb.exec_python("""
import statistics
data = [12, 45, 7, 89, 23, 56, 34, 78]
print(f"mean={statistics.mean(data):.1f}  p95={sorted(data)[int(len(data)*0.95)]}")
""")
        print(result.stdout)        # mean=43.0  p95=78
        print(result.exit_code)     # 0
        print(f"{result.duration_ms:.0f}ms")

        # Execute bash — same isolation, shell tooling available
        result = await sb.exec_bash("""
echo "Disk usage:" && du -sh /tmp
find /tmp -name "*.log" -newer /tmp -mmin -60 | wc -l
""")
        print(result.stdout)

        # Write a file, let Python read it back
        await sb.write("data.csv", "name,score\nalice,95\nbob,87\ncarol,92\n")
        result = await sb.exec_python("""
import csv, os
with open(os.environ["SANDBOX_WORKSPACE"] + "/data.csv") as f:
    rows = list(csv.DictReader(f))
print(f"{len(rows)} rows, avg score={sum(int(r['score']) for r in rows)/len(rows):.1f}")
""")
        print(result.stdout)        # 3 rows, avg score=91.3

        # Read a file the agent wrote back out
        summary = await sb.read("data.csv")
        print(summary.decode()[:40])

asyncio.run(main())
```

```
mean=43.0  p95=78
Disk usage:
/tmp  12K
0
3 rows, avg score=91.3
name,score
alice,95
bob,87
```

Pre-install packages once, used across all exec calls:

```python
sb = Sandbox(packages=["pandas", "matplotlib"], timeout=60)
result = await sb.exec_python("import pandas; print(pandas.__version__)")
# pandas is installed once before the first exec call, then cached
```

No external dependencies — `PROCESS` backend uses only the Python standard library.

---

### Persistent Memory — Four Types, Extensible Backends

```python
from syrin import Agent, Model
from syrin.enums import MemoryType

agent = Agent(model=Model.OpenAI("gpt-4o-mini", api_key="sk-..."))

# Persist across sessions
agent.remember("User is a TypeScript engineer at a fintech startup", memory_type=MemoryType.FACTS)
agent.remember("Prefers concise bullet-point answers", memory_type=MemoryType.FACTS)

# Semantic recall — top-k by relevance
memories = agent.recall("user preferences", limit=5)

# Forget outdated facts
agent.forget("previous role title")
```

| Type | What it stores |
|---|---|
| `FACTS` | Identity, preferences, persistent user facts |
| `HISTORY` | Past events and conversation summaries |
| `KNOWLEDGE` | General knowledge — ideal for vector/semantic search |
| `INSTRUCTIONS` | Skills, workflows, how-to procedures |

**Swap backends with one line:**

```python
from syrin.memory import Memory, QdrantConfig
from syrin.enums import MemoryBackend

# SQLite — zero config, single-process production
Memory(backend=MemoryBackend.SQLITE, path="~/.syrin/memory.db")

# Qdrant — semantic search at scale
Memory(backend=MemoryBackend.QDRANT, qdrant=QdrantConfig(url="...", api_key="..."))

# PostgreSQL — multi-agent shared store, pgvector
Memory(backend=MemoryBackend.POSTGRES, postgres=PostgresConfig(...))
```

---

### Multi-Agent Orchestration — Swarms and Recursive Decomposition

**Five swarm topologies — one class:**

```python
from syrin.swarm import Swarm, SwarmConfig, BudgetPool
from syrin.enums import SwarmTopology

pool = BudgetPool(total=5.00)  # $5 shared; no agent exceeds its slice

swarm = Swarm(
    agents=[Researcher, FactChecker, Writer],
    config=SwarmConfig(
        topology=SwarmTopology.ORCHESTRATOR,
        budget_pool=pool,
    ),
)
result = swarm.run("Research and write a report on battery technology trends")
print(result.cost_breakdown)   # per-agent cost
print(result.budget_report)    # pool utilisation
```

| Topology | Behaviour |
|---|---|
| `ORCHESTRATOR` | First agent routes tasks to the rest dynamically |
| `PARALLEL` | All agents run concurrently; results merged |
| `CONSENSUS` | Multiple agents vote; winner selected by strategy |
| `REFLECTION` | Producer–critic loop until quality threshold met |
| `WORKFLOW` | Sequential, parallel, branch, and dynamic fan-out steps |

**Recursive decomposition — agents that spawn agents:**

```python
from syrin import Agent, Budget, BudgetSplit, Model, Spawn
from syrin.sandbox import Sandbox

class DataAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    sandbox = Sandbox(python=True, bash=True, timeout=30)
    system_prompt = "Analyse data using code. Return concise summaries only."

class SummaryAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = "Summarise findings into bullet points."

class Orchestrator(Agent):
    model  = Model.OpenAI("gpt-4o")
    budget = Budget(max_cost=0.50)
    agents = [DataAgent, SummaryAgent]   # RLMLoop auto-wired
    spawn  = Spawn(
        max_depth=2,
        budget_split=BudgetSplit.EQUAL,  # divide budget evenly across children
        child_timeout=60.0,
    )
    sandbox = Sandbox(python=True, bash=True)  # propagated to all children

result = await Orchestrator().arun("Analyse the sales CSV and write an executive summary")
```

The orchestrator spawns specialist sub-agents at runtime. No raw file bytes enter the LLM context — all heavy processing happens inside the sandbox.

---

### Observability — 72+ Typed Lifecycle Hooks

Every LLM call, tool invocation, budget event, memory operation, sandbox execution, and agent spawn fires a typed hook. No patching. No monkey-patching. No log parsing.

```python
from syrin.enums import Hook

agent.events.on(Hook.AGENT_RUN_END,       lambda ctx: metrics.record(ctx.cost, ctx.tokens))
agent.events.on(Hook.BUDGET_THRESHOLD,    lambda ctx: pagerduty.alert(f"Budget at {ctx.percentage}%"))
agent.events.on(Hook.TOOL_CALL_END,       lambda ctx: logger.info(f"Tool {ctx.name} → {ctx.duration_ms}ms"))
agent.events.on(Hook.MEMORY_RECALL,       lambda ctx: trace.span("recall", memories=ctx.count))
agent.events.on(Hook.SANDBOX_EXEC_END,    lambda ctx: print(f"Sandbox {ctx.language} exit={ctx.exit_code} {ctx.duration_ms:.0f}ms"))
agent.events.on(Hook.RLM_SPAWN,           lambda ctx: print(f"Spawned {ctx.agent} at depth {ctx.depth}"))
agent.events.on(Hook.RLM_BUDGET_SPLIT,    lambda ctx: print(f"Child budget: \${ctx.child_budget:.4f}"))
```

Or enable full tracing with one flag — no code changes:

```bash
python my_agent.py --trace
```

---

### Guardrails — Safety Without Ceremony

```python
from syrin import Agent, Model
from syrin.guardrails import PIIGuardrail, LengthGuardrail, GuardrailChain
from syrin.enums import GuardrailMode

class SafeAgent(Agent):
    model      = Model.OpenAI("gpt-4o-mini", api_key="sk-...")
    guardrails = GuardrailChain([
        PIIGuardrail(redact=True, mode=GuardrailMode.PRE_CALL),   # scrub input
        LengthGuardrail(max_length=4000),                          # cap output
    ])

result = SafeAgent().run("Process: call me at 555-123-4567")
print(result.content)
# "call me at ***-***-****"
```

---

### Resource Limits — Per-Agent Guardrails on Execution

```python
from syrin import Agent, Model
from syrin.resource import Resource, ResourceThreshold, DegradePolicy
from syrin.enums import OnExceed, RestoreWhen

class ManagedAgent(Agent):
    model    = Model.OpenAI("gpt-4o-mini")
    resource = Resource(
        timeout=120,               # seconds per run
        max_steps=20,              # max LLM iterations
        max_context=50_000,        # token context ceiling
        on_exceed=OnExceed.STOP,
        thresholds=[
            ResourceThreshold(dimension="steps", at=80,
                              action=lambda ctx: logger.warning("Step limit near")),
        ],
        degrade=DegradePolicy(
            tool_to_disable="web_search",   # disable expensive tool at limit
            restore_when=RestoreWhen.NEVER,
        ),
    )
```

---

### Production Serving — One Line

```python
agent.serve(port=8000, enable_playground=True)
# → POST /chat   POST /stream   GET /playground   GET /health
```

**Crash-proof checkpoints:**

```python
from syrin.checkpoint import CheckpointConfig

agent = Agent(
    model=model,
    checkpoint_config=CheckpointConfig(dir="/tmp/checkpoints", auto_save=True),
)
agent.run("Begin long analysis...")
# Crash? Resume exactly where it left off:
agent.load_checkpoint("analysis-run-1")
```

**Event-driven triggers:**

```python
from syrin.watch import CronProtocol, WebhookProtocol

agent.watch(CronProtocol(cron="0 9 * * *"), task="Send morning briefing")
agent.watch(WebhookProtocol(path="/events"),  task="Process incoming event")
```

---

## Real-World Patterns

### Data Pipeline — Bash + Python in Isolation

Process large files without loading raw bytes into the LLM context:

```python
import asyncio
from syrin import Agent, Budget, BudgetSplit, Model, Spawn
from syrin.sandbox import Sandbox
from syrin.tool import tool

class IngestionAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    sandbox = Sandbox(bash=True, python=True, timeout=30)
    system_prompt = "Generate and validate data using bash and Python."

    @tool
    async def generate_dataset(self) -> str:
        """Generate 10 000-line synthetic log file."""
        result = await self.sandbox.exec_bash("""
python3 -c "
import random, datetime
for _ in range(10000):
    ip = '.'.join(str(random.randint(1,254)) for _ in range(4))
    code = random.choice([200,200,200,404,500])
    print(f'{ip} GET /api/{random.choice([\"users\",\"orders\"])} {code}')
" > \$SANDBOX_WORKSPACE/access.log
wc -l \$SANDBOX_WORKSPACE/access.log
""")
        return result.stdout.strip()

    @tool
    async def analyze_log(self) -> str:
        """Parse the log and compute error rate."""
        result = await self.sandbox.exec_python("""
import collections, os, pathlib
log = pathlib.Path(os.environ["SANDBOX_WORKSPACE"]) / "access.log"
codes = collections.Counter(line.split()[2] for line in log.open())
total = sum(codes.values())
print(f"Total: {total:,}  Errors: {codes['500']:,}  Rate: {codes['500']/total:.1%}")
""")
        return result.stdout.strip()

class Orchestrator(Agent):
    model   = Model.OpenAI("gpt-4o")
    budget  = Budget(max_cost=0.10)
    agents  = [IngestionAgent]
    spawn   = Spawn(max_depth=2, budget_split=BudgetSplit.EQUAL)
    sandbox = Sandbox(bash=True, python=True)
```

```
Total: 10,000  Errors: 1,423  Rate: 14.2%
```

---

### Customer Support — Memory + Handoff + Guardrails

```python
from syrin import Agent, Budget, Model
from syrin.guardrails import PIIGuardrail
from syrin.enums import MemoryType

class SupportAgent(Agent):
    model      = Model.OpenAI("gpt-4o-mini", api_key="sk-...")
    budget     = Budget(max_cost=0.05)
    guardrails = [PIIGuardrail(redact=True)]
    system_prompt = "You are a helpful customer support agent."

class EscalationAgent(Agent):
    model = Model.OpenAI("gpt-4o", api_key="sk-...")
    system_prompt = "Handle escalated cases requiring senior judgment."

agent = SupportAgent()
agent.remember(f"Customer {user_id}: premium plan, joined 2023", memory_type=MemoryType.FACTS)

result = agent.run(user_message)
if result.confidence < 0.6:
    result = agent.handoff(EscalationAgent, context=result.content)
```

---

### Document Intelligence — RAG + Structured Output

```python
from syrin import Agent, Model
from syrin.knowledge import Knowledge
from syrin.enums import KnowledgeBackend
from pydantic import BaseModel

class ContractRisk(BaseModel):
    risk_level: str            # low | medium | high | critical
    key_clauses: list[str]
    recommended_action: str

kb = Knowledge(
    sources=["contracts/"],
    backend=KnowledgeBackend.QDRANT,
    embedding_provider="openai",
)

class ContractReviewer(Agent):
    model       = Model.OpenAI("gpt-4o", api_key="sk-...")
    budget      = Budget(max_cost=0.25)
    knowledge   = kb
    output_type = ContractRisk

result = ContractReviewer().run("Review the indemnification clause in contract-2024-07.pdf")
risk: ContractRisk = result.output       # guaranteed typed
print(f"{risk.risk_level} — {risk.recommended_action}")
```

---

## The Harness at a Glance

| Capability | Syrin | DIY / Other Frameworks |
|---|---|---|
| **Hard budget enforcement** | Declarative, pre-call + post-call | Not available or manual |
| **Rate windows** | hour / day / month built-in | Build and persist yourself |
| **Threshold callbacks** | `BudgetThreshold(at=80, ...)` | Write from scratch |
| **Shared budget pools** | Thread-safe `BudgetPool` | Implement locking |
| **Sandboxed code execution** | `Sandbox(python=True, bash=True)` — zero deps | Manual subprocess plumbing |
| **Memory (4 types)** | Built-in, auto-managed, backend-agnostic | Manual setup |
| **Multi-agent (5 topologies + RLM)** | Single `Swarm` or `agents=` | Complex orchestration code |
| **Lifecycle hooks** | 72+ typed events | Logging + parsing |
| **Live debugger** | Rich TUI (Pry) | Parse log files |
| **Guardrails** | PII, length, content, output validation | Per-project code |
| **Checkpoints** | Auto-save, crash recovery | DIY |
| **RAG / Knowledge** | GitHub, docs, PDFs, websites | Manual indexing pipeline |
| **Structured output** | Guaranteed Pydantic / JSON | Parse + validate manually |
| **Resource limits** | `Resource(timeout, max_steps, degrade=...)` | Manual counter logic |
| **Type safety** | StrEnum everywhere, `mypy --strict` passes | String literals |

---

## Installation

```bash
# Minimal — no LLM providers
pip install syrin

# With OpenAI
pip install syrin[openai]

# With Anthropic
pip install syrin[anthropic]

# Multi-modal — voice, documents, vector stores
pip install syrin[voice,pdf,vector]

# Full install
pip install syrin[openai,anthropic,serve,vector,postgres,pdf,voice]
```

---

## Documentation

| Guide | Description |
|---|---|
| [Introduction](https://docs.syrin.dev/agent-kit/introduction) | Why Syrin exists — the AI Harness concept |
| [Quick Start](https://docs.syrin.dev/agent-kit/quick-start) | First agent in 10 minutes |
| [Budget Control](https://docs.syrin.dev/agent-kit/core/budget) | Caps, rate limits, thresholds, shared pools |
| [Sandbox](https://docs.syrin.dev/agent-kit/production/sandbox) | Isolated Python + bash execution |
| [Memory](https://docs.syrin.dev/agent-kit/core/memory) | 4 types, backends, decay curves |
| [Sub-agent Spawning](https://docs.syrin.dev/agent-kit/advanced/rlm-loop) | Recursive decomposition, budget split |
| [Multi-Agent Swarms](https://docs.syrin.dev/agent-kit/multi-agent/swarm) | 5 topologies, A2A messaging, budget delegation |
| [Resource Limits](https://docs.syrin.dev/agent-kit/production/resource-limits) | Per-agent timeouts, step caps, degrade policies |
| [Observability & Hooks](https://docs.syrin.dev/agent-kit/debugging/hooks) | 72+ events, tracing, Pry debugger |
| [Guardrails](https://docs.syrin.dev/agent-kit/agent/guardrails) | PII, length, content filtering, output validation |
| [Serving](https://docs.syrin.dev/agent-kit/production/serving) | HTTP API, streaming, playground |
| [Checkpoints](https://docs.syrin.dev/agent-kit/production/checkpointing) | State persistence, crash recovery |
| [Migration v0.11 → v0.12](https://docs.syrin.dev/agent-kit/migration/v0.11-to-v0.12) | Breaking changes and upgrade guide |
| [Examples](examples/) | Runnable code for every use case |

---

## Join the Community

<p align="center">
  <a href="https://discord.gg/BJUdNBh4TC">
    <img src="https://img.shields.io/badge/Discord-Join%20the%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord">
  </a>
  &nbsp;&nbsp;
  <a href="https://www.youtube.com/@syrin_dev">
    <img src="https://img.shields.io/badge/YouTube-Watch%20tutorials-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="YouTube">
  </a>
  &nbsp;&nbsp;
  <a href="https://reddit.com/r/syrin_ai">
    <img src="https://img.shields.io/badge/Reddit-r%2Fsyrin__ai-FF4500?style=for-the-badge&logo=reddit&logoColor=white" alt="Reddit">
  </a>
</p>

| Channel | What's there |
|---|---|
| [Discord](https://discord.gg/BJUdNBh4TC) | Real-time help, showcase your agents, roadmap discussion |
| [Reddit — r/syrin_ai](https://reddit.com/r/syrin_ai) | Longer posts, tutorials, use-case deep dives |
| [YouTube — @syrin_dev](https://www.youtube.com/@syrin_dev) | Walkthroughs, feature demos, production patterns |
| [GitHub Discussions](https://github.com/syrin-labs/syrin-python/discussions) | RFCs, architecture questions, feature requests |
| [Website](https://syrin.ai) | Product overview, roadmap, changelog |

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on setting up the dev environment, running tests, and submitting pull requests.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Give developers control over AI. That's the harness.
</p>
