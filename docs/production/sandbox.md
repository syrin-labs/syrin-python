---
title: Sandbox — Isolated Code Execution
description: Run shell scripts and Python code safely with the PROCESS sandbox. Zero dependencies, production-ready.
weight: 40
---

# Sandbox — Isolated Code Execution

`syrin.Sandbox` provides isolated subprocess execution for AI-generated code. Instead of running LLM output with bare `exec()` inside your process, each exec call spawns a fresh child process with a hard timeout, optional memory limit, and a clean environment.

**Zero external dependencies.** The default `PROCESS` backend uses only the Python standard library — no Docker daemon, no cloud accounts, no optional installs. It works on macOS, Linux, and Windows out of the box.

**Production-ready since v0.12.0.** Shell script execution (`exec_bash`), packages auto-install, async context manager, and lifecycle hooks are all included.

## Why sandboxing matters

Without a sandbox, LLM-generated code runs in the same process as your application, with the same permissions, file-system access, and memory. A single `os.system("rm -rf /")` or `import secrets; print(secrets.token_hex(32))` is all it takes to cause damage or leak sensitive data.

With `Sandbox`:

- Code runs in a **fresh subprocess** — no shared state with the parent.
- A **hard timeout** kills the process if it exceeds the limit.
- An optional **memory cap** prevents runaway allocations (POSIX only, via `resource.RLIMIT_AS`).
- **Temp files are cleaned up** after every call — no residual code or scripts on disk.
- The workspace is an isolated directory. Parent packages are not importable unless you list them in `packages=` or call `install()`.
- The subprocess receives `SANDBOX_WORKSPACE` in its environment so it knows where to find shared files.

## Quick start

```python
import asyncio
from syrin.sandbox import Sandbox

async def main():
    sandbox = Sandbox(timeout=10.0, memory_mb=128)
    result = await sandbox.exec_python("print(sum(range(100)))")
    print(result.stdout)     # "4950\n"
    print(result.exit_code)  # 0

asyncio.run(main())
```

## exec_python

```python
result = await sandbox.exec_python(code, timeout=None)
```

- `code` — Python source string.
- `timeout` — Override the sandbox's default timeout for this call (seconds).
- Returns `SandboxExecResult(stdout, stderr, exit_code, duration_ms)`.
- Raises `SandboxTimeoutError` if execution exceeds the timeout.

```python
result = await sandbox.exec_python("raise ValueError('boom')")
# result.exit_code == 1
# result.stderr contains "ValueError: boom"
print(result.exit_code)   # 1
print(result.stderr)      # "Traceback...\nValueError: boom\n"
```

## exec_bash

New in v0.12.0. Runs a shell script via `bash` (falling back to `sh` if bash is not on PATH). The script is written to a `.sh` temp file with `chmod 0o700`, executed in a fresh subprocess, then deleted.

**Requires `bash=True` in the Sandbox config** — shell execution is disabled by default.

```python
sandbox = Sandbox(bash=True, timeout=30.0)

result = await sandbox.exec_bash("""
#!/usr/bin/env bash
set -euo pipefail

echo "Generating data..."
seq 1 100 > "$SANDBOX_WORKSPACE/numbers.txt"
wc -l "$SANDBOX_WORKSPACE/numbers.txt"
echo "Exit status: $?"
""")

print(result.stdout)
# "Generating data...\n100 /tmp/syrin-sandbox-abc123/numbers.txt\nExit status: 0\n"
print(result.exit_code)  # 0
print(result.stderr)     # ""
```

### Per-call timeout override

```python
result = await sandbox.exec_bash("sleep 5", timeout=2.0)
# Raises SandboxTimeoutError after 2 seconds
```

### Environment variables in bash scripts

All variables from `env=` (and the auto-injected `SANDBOX_WORKSPACE`) are available as normal environment variables:

```python
sandbox = Sandbox(
    bash=True,
    env={"API_KEY": "test-key-123", "OUTPUT_DIR": "/tmp/out"},
)

result = await sandbox.exec_bash("""
echo "API_KEY is: $API_KEY"
echo "Workspace: $SANDBOX_WORKSPACE"
mkdir -p "$OUTPUT_DIR"
echo "done" > "$OUTPUT_DIR/result.txt"
""")
# result.stdout: "API_KEY is: test-key-123\nWorkspace: /tmp/syrin-sandbox-xyz\n"
```

## exec_js

```python
sandbox = Sandbox(js=True, timeout=10.0)
result = await sandbox.exec_js("console.log(1 + 1)")
```

- Requires `node` to be on `PATH`.
- Raises `SandboxError` if `js=False` (the default).
- Raises `SandboxError` if `node` is not installed.

## SANDBOX_WORKSPACE environment variable

Every subprocess started by the sandbox receives `SANDBOX_WORKSPACE` set to the workspace directory path. Code running inside the subprocess can use this to locate shared files:

```python
# Parent process
sandbox = Sandbox(bash=True, python=True)
await sandbox.exec_bash("echo 'hello from bash' > $SANDBOX_WORKSPACE/message.txt")

# Now Python in the same sandbox can read it
result = await sandbox.exec_python("""
import os
workspace = os.environ["SANDBOX_WORKSPACE"]
with open(os.path.join(workspace, "message.txt")) as f:
    print(f.read().strip())
""")
print(result.stdout)  # "hello from bash"
```

The value is always an absolute path. Do not assume its location — always read `SANDBOX_WORKSPACE` from the environment.

## Packages auto-install

List packages at construction time and they are installed before the first `exec_python` call, using a double-checked `asyncio.Lock` so concurrent calls are safe. Installation only happens once per `Sandbox` instance.

```python
sandbox = Sandbox(
    packages=["requests", "numpy"],
    timeout=60.0,
)

# No manual install() call needed — packages are ready automatically
result = await sandbox.exec_python("""
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(arr.mean())
""")
print(result.stdout)  # "3.0\n"
```

Packages are installed into `<workspace>/.packages` and added to `PYTHONPATH` automatically.

### Manual install

If you need to install packages after construction:

```python
await sandbox.install(["requests", "numpy"])
result = await sandbox.exec_python("import numpy; print(numpy.__version__)")
```

`install()` raises `SandboxError` if pip exits non-zero.

## Cleanup and context manager

### async with (recommended)

Use `async with Sandbox() as sb:` to guarantee cleanup even if an exception occurs:

```python
async with Sandbox(bash=True, packages=["httpx"]) as sb:
    result = await sb.exec_bash("echo 'workspace is ready'")
    print(result.stdout)
# Workspace is deleted here — sb.cleanup() is called automatically
```

This is the recommended pattern for scripts and one-off tasks. The context manager calls `cleanup()` in `__aexit__`.

### Manual cleanup

For long-lived sandboxes (e.g., held across multiple requests), call `cleanup()` explicitly:

```python
sandbox = Sandbox()
try:
    result = await sandbox.exec_python("print('hello')")
    # ... more work ...
finally:
    sandbox.cleanup()
```

`cleanup()` removes the workspace directory only when it was auto-created (i.e., `workspace=None` was passed to the constructor). Workspaces you provide explicitly are never deleted. It is safe to call multiple times — subsequent calls are no-ops.

A `__del__` fallback also calls `cleanup()` when the object is garbage-collected, but relying on `__del__` in asyncio programs is not recommended.

## write / read

Persist files in the workspace across exec calls:

```python
await sandbox.write("data.csv", "a,b\n1,2\n3,4\n")
result = await sandbox.exec_python("""
import os, csv
workspace = os.environ["SANDBOX_WORKSPACE"]
with open(os.path.join(workspace, "data.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row)
""")
```

```python
await sandbox.write("result.json", '{"answer": 42}')
raw = await sandbox.read("result.json")
import json
print(json.loads(raw))  # {'answer': 42}
```

## End-to-end example: generate data with bash, analyze with Python

This example shows the full cross-language workflow: a bash script generates structured data, Python reads and analyzes it, and the parent collects the final result.

```python
import asyncio
import json
from syrin.sandbox import Sandbox

async def analyze_sales():
    async with Sandbox(bash=True, python=True, packages=["statistics"]) as sb:

        # Step 1: bash generates CSV data
        await sb.exec_bash("""
        cat > $SANDBOX_WORKSPACE/sales.csv << 'EOF'
        month,revenue
        Jan,12000
        Feb,15500
        Mar,9800
        Apr,18200
        May,21000
        Jun,17400
        EOF
        echo "Generated sales.csv"
        """)

        # Step 2: Python analyzes it
        result = await sb.exec_python("""
        import os, csv, statistics, json

        workspace = os.environ["SANDBOX_WORKSPACE"]
        with open(os.path.join(workspace, "sales.csv")) as f:
            rows = list(csv.DictReader(f))

        revenues = [int(r["revenue"].strip()) for r in rows]

        summary = {
            "months": len(revenues),
            "total": sum(revenues),
            "mean": statistics.mean(revenues),
            "median": statistics.median(revenues),
            "best_month": rows[revenues.index(max(revenues))]["month"].strip(),
        }

        output_path = os.path.join(workspace, "summary.json")
        with open(output_path, "w") as f:
            json.dump(summary, f)

        print(json.dumps(summary))
        """)

        print(result.stdout)
        # {"months": 6, "total": 93900, "mean": 15650.0, "median": 16450.0, "best_month": "May"}

        # Step 3: parent reads the file directly
        raw = await sb.read("summary.json")
        return json.loads(raw)

summary = asyncio.run(analyze_sales())
print(f"Best month: {summary['best_month']}, Total: ${summary['total']:,}")
# Best month: May, Total: $93,900
```

## CodeActionLoop with sandbox

Pass a `Sandbox` to `CodeActionLoop` to sandbox all LLM-generated code:

```python
from syrin import Agent, Model
from syrin.loop import CodeActionLoop
from syrin.sandbox import Sandbox

class MathAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    loop = CodeActionLoop(
        max_iterations=5,
        timeout_seconds=30,
        sandbox=Sandbox(bash=True, timeout=10.0, memory_mb=256),
    )

agent = MathAgent()
result = agent.run("Calculate the sum of primes below 1000")
print(result.content)
```

Without `sandbox=`, `CodeActionLoop` falls back to in-process `exec()`. Only use that for trusted, internal inputs.

## Configuration reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backend` | `SandboxBackendType \| SandboxBackendProtocol` | `PROCESS` | Backend to use. |
| `python` | `bool` | `True` | Enable Python execution via `exec_python`. |
| `js` | `bool` | `False` | Enable JavaScript execution via `exec_js` (requires node on PATH). |
| `bash` | `bool` | `False` | Enable shell script execution via `exec_bash`. New in v0.12.0. |
| `memory_mb` | `int \| None` | `None` | Memory limit in MB per exec call (POSIX only). |
| `timeout` | `float` | `30.0` | Default wall-clock timeout in seconds. |
| `packages` | `list[str]` | `[]` | Packages installed before the first exec call. |
| `env` | `dict[str, str]` | `{}` | Extra environment variables injected into every subprocess. |
| `workspace` | `str \| None` | `None` | Auto-generates `/tmp/syrin-sandbox-<uuid>` when `None`. |
| `max_code_length` | `int` | `10_000_000` | Maximum bytes allowed in a code/script string. |

## Backends

| Backend | Requires | Notes |
|---------|----------|-------|
| `PROCESS` | Nothing (stdlib only) | Default. Fresh subprocess per call. Works on macOS, Linux, Windows. |
| `DOCKER` | `docker` package + Docker daemon | Container isolation. Not yet implemented. |
| `E2B` | `e2b` package + API key | Managed cloud sandbox. Not yet implemented. |
| `NSJAIL` | `nsjail` binary on PATH, Linux | Namespace isolation. Not yet implemented. |

## Custom backend

Implement `SandboxBackendProtocol` to plug in any execution environment. The protocol now includes `exec_bash` as of v0.12.0:

```python
from syrin.sandbox import Sandbox, SandboxBackendProtocol, SandboxExecResult

class MyDockerBackend:
    async def exec_python(
        self,
        code: str,
        workspace: str,
        timeout: float,
        memory_mb: int | None,
        env: dict[str, str],
    ) -> SandboxExecResult:
        # ... run Python in Docker container ...
        return SandboxExecResult(stdout="...", stderr="", exit_code=0, duration_ms=0.0)

    async def exec_js(
        self,
        code: str,
        workspace: str,
        timeout: float,
        env: dict[str, str],
    ) -> SandboxExecResult:
        # ... run node in Docker container ...
        return SandboxExecResult(stdout="...", stderr="", exit_code=0, duration_ms=0.0)

    async def exec_bash(
        self,
        script: str,
        workspace: str,
        timeout: float,
        env: dict[str, str],
    ) -> SandboxExecResult:
        # ... run bash in Docker container ...
        return SandboxExecResult(stdout="...", stderr="", exit_code=0, duration_ms=0.0)

    async def install(self, packages: list[str], workspace: str) -> None:
        # ... pip install inside the container ...
        pass

sandbox = Sandbox(backend=MyDockerBackend())
result = await sandbox.exec_python("print('hello from Docker')")
```

All four methods are required by the protocol. If your backend does not support one of them (e.g., no JavaScript), raise `SandboxError` with a clear message.

## Lifecycle hooks

`syrin.Hook` includes sandbox events for observability. `SANDBOX_EXEC_START` fires for all exec types — Python, JavaScript, and bash.

| Hook | When | Key fields |
|------|------|------------|
| `Hook.SANDBOX_EXEC_START` | Before each `exec_python` / `exec_js` / `exec_bash` call | `language`, `timeout` |
| `Hook.SANDBOX_EXEC_END` | After successful execution | `language`, `exit_code`, `duration_ms` |
| `Hook.SANDBOX_TIMEOUT` | Execution timed out | `language`, `timeout` |
| `Hook.SANDBOX_OOM` | Execution exceeded memory limit | — |
| `Hook.SANDBOX_SESSION_CREATED` | Workspace directory created | `workspace` |
| `Hook.SANDBOX_SESSION_DESTROYED` | Workspace directory destroyed | `workspace` |

Hooks only fire when the sandbox is used inside an agent or loop that has wired up the `_emit_fn`. When using `Sandbox` standalone, hooks are silently skipped.

```python
agent.events.on(
    Hook.SANDBOX_EXEC_START,
    lambda ctx: print(f"[sandbox] {ctx.language} starting (timeout={ctx.timeout}s)"),
)
agent.events.on(
    Hook.SANDBOX_EXEC_END,
    lambda ctx: print(f"[sandbox] done in {ctx.duration_ms:.1f}ms, exit={ctx.exit_code}"),
)
```

## Exception hierarchy

```
SyrinError
└── SandboxError              # base for all sandbox errors
    ├── SandboxTimeoutError   # execution exceeded timeout
    └── SandboxMemoryError    # execution exceeded memory_mb
```

```python
from syrin import SandboxError, SandboxTimeoutError

try:
    result = await sandbox.exec_python(slow_code, timeout=5.0)
except SandboxTimeoutError:
    print("code took too long")
except SandboxError as e:
    print(f"sandbox error: {e}")
```

Common causes of `SandboxError` (non-timeout):

- `bash=False` and you called `exec_bash()`
- `js=False` and you called `exec_js()`
- `python=False` and you called `exec_python()`
- `node` not on PATH when `js=True`
- Neither `bash` nor `sh` on PATH when `bash=True`
- `pip install` failed during auto-install or manual `install()`
- Code string exceeds `max_code_length`
- File not found when calling `read()`
