"""Data-pipeline orchestrator: RLMLoop + Sandbox (bash + Python) end-to-end.

Shows every RLM and Sandbox capability in one example:

  Sandbox features
  ─────────────────
  • exec_bash()   — generate data, validate, compress, format reports
  • exec_python() — parse and aggregate with stdlib (no extra packages)
  • write() / read() — pass datasets between tools without touching LLM context
  • packages=     — auto-install before first exec (double-checked locking)
  • async context manager — workspace cleaned up on exit
  • Lifecycle hooks: SANDBOX_SESSION_CREATED/DESTROYED, EXEC_START/END, TIMEOUT

  RLMLoop features
  ─────────────────
  • max_depth=2, child_timeout=60.0
  • BudgetSplit.EQUAL — parent budget divided evenly across spawned children
  • sandbox= auto-propagated from orchestrator to every child loop
  • Lifecycle hooks: RLM_SPAWN, RLM_COMPLETE, RLM_BUDGET_SPLIT, RLM_DEPTH_EXCEEDED

Pipeline
────────
  LogOrchestrator (RLMLoop, Budget=$0.10, EQUAL split, max_depth=2)
    ├── LogIngestionAgent  — bash: generates 10 000-line access log → workspace
    ├── LogAnalysisAgent   — Python: parses log, writes JSON summary → workspace
    └── ReportBuilderAgent — bash: formats report, creates .tar.gz archive

Each agent accesses the shared Sandbox via self.sandbox (class attribute propagated
by RLMLoop into every child). No raw log bytes flow through the LLM context.
"""

from __future__ import annotations

import asyncio

from syrin import Agent, Budget, BudgetSplit, Hook, Model, Spawn
from syrin.sandbox import Sandbox
from syrin.tool import tool

# ---------------------------------------------------------------------------
# Shared sandbox — set on the orchestrator; auto-propagated to all children
# ---------------------------------------------------------------------------
_SHARED_SANDBOX = Sandbox(
    python=True,
    bash=True,
    timeout=30.0,
    memory_mb=256,
    packages=[],  # no extra packages — stdlib only for this example
)


# ---------------------------------------------------------------------------
# Sub-agent 1: ingest — generates synthetic access-log data with bash
# ---------------------------------------------------------------------------
class LogIngestionAgent(Agent):
    """Generates synthetic access-log data using bash and writes it to the sandbox workspace."""

    model = Model.Almock(lorem_length=80)
    system_prompt = (
        "You are a log ingestion specialist. "
        "Call generate_log to produce the dataset, then validate_log to confirm it."
    )

    @tool
    async def generate_log(self, lines: int = 10_000) -> str:
        """Generate a synthetic nginx-style access log in the sandbox workspace.

        Args:
            lines: Number of log lines to generate (default 10 000).

        Returns:
            Status message with line count and file size.
        """
        script = f"""
set -euo pipefail
OUTFILE="$SANDBOX_WORKSPACE/access.log"
python3 -c "
import random, datetime
methods = ['GET', 'POST', 'PUT', 'DELETE']
paths = ['/api/users', '/api/orders', '/api/products', '/health', '/metrics']
codes = [200, 200, 200, 201, 400, 404, 500]
for _ in range({lines}):
    ip  = '.'.join(str(random.randint(1, 254)) for _ in range(4))
    ts  = datetime.datetime.utcnow().strftime('%d/%b/%Y:%H:%M:%S +0000')
    m   = random.choice(methods)
    p   = random.choice(paths)
    c   = random.choice(codes)
    ms  = random.randint(5, 2000)
    print(f'{{ip}} - - [{{ts}}] \\"{{m}} {{p}} HTTP/1.1\\" {{c}} {{ms}}')
" > "$OUTFILE"
wc -l "$OUTFILE"
du -sh "$OUTFILE"
"""
        result = await self.sandbox.exec_bash(script)  # type: ignore[attr-defined]
        if result.exit_code != 0:
            return f"ERROR: {result.stderr.strip()}"
        return f"Generated access.log — {result.stdout.strip()}"

    @tool
    async def validate_log(self) -> str:
        """Validate that the log file exists and has expected structure.

        Returns:
            Validation summary (line count, sample lines, error-code distribution).
        """
        script = r"""
set -euo pipefail
LOG="$SANDBOX_WORKSPACE/access.log"
[ -f "$LOG" ] || { echo "ERROR: access.log not found"; exit 1; }

total=$(wc -l < "$LOG")
errors=$(awk '$9 >= 500' "$LOG" | wc -l)
client_err=$(awk '$9 >= 400 && $9 < 500' "$LOG" | wc -l)
ok=$(awk '$9 >= 200 && $9 < 300' "$LOG" | wc -l)

echo "Total lines : $total"
echo "2xx responses: $ok"
echo "4xx responses: $client_err"
echo "5xx responses: $errors"
echo "---"
echo "Sample (first 3 lines):"
head -3 "$LOG"
"""
        result = await self.sandbox.exec_bash(script)  # type: ignore[attr-defined]
        return result.stdout.strip() if result.exit_code == 0 else f"ERROR: {result.stderr.strip()}"


# ---------------------------------------------------------------------------
# Sub-agent 2: analysis — Python stdlib parsing → JSON summary in workspace
# ---------------------------------------------------------------------------
class LogAnalysisAgent(Agent):
    """Parses the access log with Python and writes a JSON summary to the workspace."""

    model = Model.Almock(lorem_length=120)
    system_prompt = (
        "You are a log analysis specialist. "
        "Call analyze_log to parse the dataset and produce a JSON summary."
    )

    @tool
    async def analyze_log(self) -> str:
        """Parse access.log and write summary.json to the workspace.

        Computes per-endpoint hit counts, status-code distribution, and
        average response time. Writes summary.json to the workspace so the
        report agent can read it without touching raw bytes.

        Returns:
            Top-5 endpoints and overall error rate.
        """
        code = r"""
import json, re, statistics, collections, pathlib, os

log_path = pathlib.Path(os.environ["SANDBOX_WORKSPACE"]) / "access.log"
out_path  = pathlib.Path(os.environ["SANDBOX_WORKSPACE"]) / "summary.json"

pattern = re.compile(
    r'(?P<ip>\S+) .+ \[.+\] "(?P<method>\S+) (?P<path>\S+) \S+" '
    r'(?P<status>\d{3}) (?P<ms>\d+)'
)

endpoints: dict[str, int]      = collections.Counter()
statuses:  dict[str, int]      = collections.Counter()
latencies: list[float]         = []

with log_path.open() as f:
    for line in f:
        m = pattern.match(line)
        if not m:
            continue
        endpoints[m.group("path")] += 1
        statuses[m.group("status")] += 1
        latencies.append(float(m.group("ms")))

total   = sum(statuses.values())
errors  = sum(v for k, v in statuses.items() if k.startswith("5"))
p95_ms  = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

summary = {
    "total_requests": total,
    "error_rate_pct": round(errors / total * 100, 2) if total else 0,
    "p95_latency_ms": round(p95_ms, 1),
    "avg_latency_ms": round(statistics.mean(latencies), 1) if latencies else 0,
    "status_dist":    dict(sorted(statuses.items())),
    "top_endpoints":  dict(endpoints.most_common(5)),
}

out_path.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
"""
        result = await self.sandbox.exec_python(code)  # type: ignore[attr-defined]
        if result.exit_code != 0:
            return f"ERROR: {result.stderr.strip()}"
        return f"summary.json written.\n{result.stdout.strip()}"


# ---------------------------------------------------------------------------
# Sub-agent 3: report — bash formatting + tar.gz archive
# ---------------------------------------------------------------------------
class ReportBuilderAgent(Agent):
    """Reads summary.json from the workspace, formats a human-readable report, and archives it."""

    model = Model.Almock(lorem_length=80)
    system_prompt = (
        "You are a report builder. Call build_report to format and archive the analysis results."
    )

    @tool
    async def build_report(self) -> str:
        """Format summary.json into a text report and create a .tar.gz archive.

        Reads summary.json (written by LogAnalysisAgent), formats a
        human-readable report.txt, then packages both files into
        analysis_report.tar.gz. Returns the archive size and checksum.

        Returns:
            Archive path, size, and sha256 checksum.
        """
        script = r"""
set -euo pipefail
WS="$SANDBOX_WORKSPACE"
SUMMARY="$WS/summary.json"
REPORT="$WS/report.txt"
ARCHIVE="$WS/analysis_report.tar.gz"

[ -f "$SUMMARY" ] || { echo "ERROR: summary.json not found — run analysis first"; exit 1; }

# Format the JSON into a readable report
python3 - <<'PY'
import json, pathlib, os, datetime
ws = pathlib.Path(os.environ["SANDBOX_WORKSPACE"])
s  = json.loads((ws / "summary.json").read_text())

lines = [
    "=" * 60,
    f"  ACCESS LOG ANALYSIS REPORT",
    f"  Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
    "=" * 60,
    "",
    f"  Total requests   : {s['total_requests']:,}",
    f"  Error rate       : {s['error_rate_pct']}%",
    f"  Avg latency      : {s['avg_latency_ms']} ms",
    f"  P95 latency      : {s['p95_latency_ms']} ms",
    "",
    "  Status Distribution",
    "  -------------------",
]
for code, count in sorted(s["status_dist"].items()):
    lines.append(f"    {code}: {count:,}")

lines += [
    "",
    "  Top 5 Endpoints",
    "  ---------------",
]
for path, hits in s["top_endpoints"].items():
    lines.append(f"    {hits:>6,}  {path}")

lines += ["", "=" * 60]
(ws / "report.txt").write_text("\n".join(lines))
PY

# Print the report
cat "$REPORT"
echo ""

# Archive both files
tar -czf "$ARCHIVE" -C "$WS" summary.json report.txt

# Output archive info
echo "Archive : $ARCHIVE"
du -sh "$ARCHIVE"
sha256sum "$ARCHIVE" 2>/dev/null || shasum -a 256 "$ARCHIVE"
"""
        result = await self.sandbox.exec_bash(script)  # type: ignore[attr-defined]
        if result.exit_code != 0:
            return f"ERROR: {result.stderr.strip()}"
        return result.stdout.strip()

    @tool
    async def read_report(self) -> str:
        """Read the final report.txt from the sandbox workspace.

        Returns:
            Full report content as a string (read via sandbox.read()).
        """
        data = await self.sandbox.read("report.txt")  # type: ignore[attr-defined]
        return data.decode()


# ---------------------------------------------------------------------------
# Orchestrator — wires everything together
# ---------------------------------------------------------------------------
class LogOrchestrator(Agent):
    """Orchestrates a full log-analysis pipeline using bash, Python, and file ops.

    No raw log bytes pass through the LLM context — all heavy work runs
    inside the shared sandbox and only concise summaries come back.
    """

    model = Model.Almock(lorem_length=200)
    budget = Budget(max_cost=0.10)
    sandbox = _SHARED_SANDBOX  # auto-propagated to all children by RLMLoop
    agents = [LogIngestionAgent, LogAnalysisAgent, ReportBuilderAgent]
    spawn = Spawn(
        max_depth=2,
        budget_split=BudgetSplit.EQUAL,  # $0.10 / 3 children ≈ $0.033 each
        child_timeout=60.0,
    )
    system_prompt = (
        "You are a log pipeline orchestrator. "
        "Delegate work in order: "
        "1) spawn LogIngestionAgent to generate and validate the dataset; "
        "2) spawn LogAnalysisAgent to parse and summarize it; "
        "3) spawn ReportBuilderAgent to format and archive the results. "
        "Never load raw file contents into your own response."
    )


# ---------------------------------------------------------------------------
# Hook observers — print every lifecycle event to stdout
# ---------------------------------------------------------------------------
def _attach_hooks(agent: LogOrchestrator) -> None:
    def on_rlm_spawn(ctx: object) -> None:
        d = dict(ctx)  # type: ignore[call-overload]
        print(f"  [RLM_SPAWN]        agent={d.get('agent')}  depth={d.get('depth')}")

    def on_rlm_complete(ctx: object) -> None:
        d = dict(ctx)  # type: ignore[call-overload]
        print(f"  [RLM_COMPLETE]     agent={d.get('agent')}  depth={d.get('depth')}")

    def on_budget_split(ctx: object) -> None:
        d = dict(ctx)  # type: ignore[call-overload]
        print(
            f"  [RLM_BUDGET_SPLIT] agent={d.get('agent')}  "
            f"parent=${d.get('parent_budget', 0):.4f}  "
            f"child=${d.get('child_budget', 0):.4f}  "
            f"split={d.get('split')}"
        )

    def on_exec_start(ctx: object) -> None:
        d = dict(ctx)  # type: ignore[call-overload]
        print(f"  [SANDBOX_EXEC_START] lang={d.get('language')}  timeout={d.get('timeout')}s")

    def on_exec_end(ctx: object) -> None:
        d = dict(ctx)  # type: ignore[call-overload]
        print(
            f"  [SANDBOX_EXEC_END]   lang={d.get('language')}  "
            f"exit={d.get('exit_code')}  "
            f"dur={d.get('duration_ms', 0):.0f}ms"
        )

    agent.events.on(Hook.RLM_SPAWN, on_rlm_spawn)
    agent.events.on(Hook.RLM_COMPLETE, on_rlm_complete)
    agent.events.on(Hook.RLM_BUDGET_SPLIT, on_budget_split)
    agent.events.on(Hook.SANDBOX_EXEC_START, on_exec_start)
    agent.events.on(Hook.SANDBOX_EXEC_END, on_exec_end)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main() -> None:
    """Run the log-analysis pipeline and print the orchestrator's final synthesis."""
    async with _SHARED_SANDBOX:  # ensures workspace cleanup even on exception
        orchestrator = LogOrchestrator()
        _attach_hooks(orchestrator)

        print("Starting log-analysis pipeline...")
        print("─" * 60)
        result = await orchestrator.arun(
            "Generate a 10 000-line access log, analyze it for error rates and "
            "top endpoints, then produce a formatted report and compressed archive."
        )
        print("─" * 60)
        print("\nOrchestrator synthesis:")
        print(result.content)
        print(f"\nTotal cost: ${result.cost.total_cost:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
