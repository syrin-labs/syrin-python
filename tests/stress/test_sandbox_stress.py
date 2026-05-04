"""Sandbox stress tests — resource isolation, concurrency, failure modes, robustness."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

# Prime the import cache for syrin.loop (resolves circular import)
from syrin.agent._run_context import DefaultAgentRunContext as _  # noqa: F401

# ──────────────────────────────────────────────────────────────────────────────
# Section 1 — Concurrent execution isolation
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.slow
async def test_10_sandboxes_concurrent_no_interference() -> None:
    """10 Sandbox instances run concurrently; each writes to its own workspace.

    No sandbox reads another sandbox's file.  All 10 complete without error.
    """
    from syrin.sandbox import Sandbox

    sandboxes = [Sandbox() for _ in range(10)]

    async def _run_one(sb: Sandbox, idx: int) -> str:
        workspace = sb.workspace
        assert workspace is not None
        code = (
            f"import os\n"
            f"p = os.path.join({workspace!r}, 'output.txt')\n"
            f"open(p, 'w').write('sandbox_{idx}')\n"
        )
        result = await sb.exec_python(code)
        assert result.exit_code == 0, f"Sandbox {idx} failed: {result.stderr}"
        data = await sb.read("output.txt")
        return data.decode()

    results = await asyncio.gather(*[_run_one(sb, i) for i, sb in enumerate(sandboxes)])

    for i, content in enumerate(results):
        assert content == f"sandbox_{i}", (
            f"Sandbox {i} got wrong content: {content!r} (cross-contamination?)"
        )


@pytest.mark.asyncio
async def test_same_sandbox_concurrent_exec_serialized_or_isolated() -> None:
    """Same Sandbox instance, 5 concurrent exec_python calls.

    Each prints a unique object id.  All 5 complete without error and
    outputs don't mix (each result is a single line).
    """
    from syrin.sandbox import Sandbox

    sb = Sandbox()

    async def _unique_id() -> str:
        result = await sb.exec_python("print(id(object()))")
        assert result.exit_code == 0
        return result.stdout.strip()

    outputs = await asyncio.gather(*[_unique_id() for _ in range(5)])

    assert len(outputs) == 5
    for out in outputs:
        # Each output should be a single numeric string (no mixing)
        assert out.isdigit(), f"Expected single numeric id, got: {out!r}"


# ──────────────────────────────────────────────────────────────────────────────
# Section 2 — Timeout enforcement
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_timeout_fires_and_child_process_is_reaped() -> None:
    """Sandbox(timeout=0.05). Sleeping code must raise SandboxTimeoutError within ~0.2s.

    After the error, no zombie process should linger — the process is awaited
    in the finally block of ProcessSandbox.exec_python.
    """
    from syrin.sandbox import Sandbox, SandboxTimeoutError

    sb = Sandbox(timeout=0.05)

    with pytest.raises(SandboxTimeoutError):
        await asyncio.wait_for(
            sb.exec_python("import time; time.sleep(10)"),
            timeout=2.0,  # fail-safe
        )

    # Subsequent exec on a fresh sandbox must not hang (processes were reaped)
    sb2 = Sandbox()
    result = await sb2.exec_python("print('alive')")
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_timeout_does_not_affect_subsequent_exec() -> None:
    """After a timeout, the same Sandbox instance can still execute code normally."""
    from syrin.sandbox import Sandbox, SandboxTimeoutError

    sb = Sandbox(timeout=10.0)  # long default timeout

    # First call: force an early timeout via per-call override
    with pytest.raises(SandboxTimeoutError):
        await sb.exec_python("import time; time.sleep(10)", timeout=0.05)

    # Second call on the same sandbox must succeed
    result = await sb.exec_python("print('still works')")
    assert result.exit_code == 0
    assert "still works" in result.stdout


@pytest.mark.asyncio
async def test_per_call_timeout_overrides_sandbox_timeout() -> None:
    """Sandbox(timeout=10). exec_python(code, timeout=0.01) times out at 0.01s, not 10s."""
    import time

    from syrin.sandbox import Sandbox, SandboxTimeoutError

    sb = Sandbox(timeout=10.0)

    start = time.perf_counter()
    with pytest.raises(SandboxTimeoutError):
        await sb.exec_python("import time; time.sleep(10)", timeout=0.05)
    elapsed = time.perf_counter() - start

    # Must have timed out well before the 10s sandbox-level timeout
    assert elapsed < 2.0, f"Per-call timeout not respected — elapsed {elapsed:.2f}s"


# ──────────────────────────────────────────────────────────────────────────────
# Section 3 — Exit codes and stderr
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_nonzero_exit_code_returned_not_raised() -> None:
    """exec_python("import sys; sys.exit(42)") → result.exit_code == 42, no exception."""
    from syrin.sandbox import Sandbox

    sb = Sandbox()
    result = await sb.exec_python("import sys; sys.exit(42)")
    assert result.exit_code == 42, f"Expected exit_code 42, got {result.exit_code}"
    # No exception raised — just captured


@pytest.mark.asyncio
async def test_syntax_error_nonzero_exit_and_stderr() -> None:
    """Syntax error in code → exit_code != 0, stderr contains 'SyntaxError'."""
    from syrin.sandbox import Sandbox

    sb = Sandbox()
    result = await sb.exec_python("def foo(: pass")
    assert result.exit_code != 0, "Expected non-zero exit for syntax error"
    assert "SyntaxError" in result.stderr, f"Expected SyntaxError in stderr, got: {result.stderr!r}"


@pytest.mark.asyncio
async def test_import_error_nonzero_exit_stderr_captured() -> None:
    """Importing a nonexistent package → exit_code != 0, stderr has ModuleNotFoundError."""
    from syrin.sandbox import Sandbox

    sb = Sandbox()
    result = await sb.exec_python("import nonexistent_package_xyz_stress_test")
    assert result.exit_code != 0, "Expected non-zero exit for import error"
    assert "ModuleNotFoundError" in result.stderr or "No module named" in result.stderr, (
        f"Expected ModuleNotFoundError in stderr, got: {result.stderr!r}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Section 4 — Workspace file I/O under load
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.slow
async def test_write_read_large_file() -> None:
    """Write 1MB of data to sandbox workspace. Read it back. Content matches exactly."""
    from syrin.sandbox import Sandbox

    sb = Sandbox()
    data = b"x" * (1024 * 1024)  # 1MB

    await sb.write("large.bin", data)
    read_back = await sb.read("large.bin")
    assert read_back == data, f"Content mismatch: got {len(read_back)} bytes"


@pytest.mark.asyncio
async def test_exec_python_can_read_written_file() -> None:
    """sandbox.write('data.txt', 'hello world'); exec_python reads it; stdout == expected."""
    from syrin.sandbox import Sandbox

    sb = Sandbox()
    await sb.write("data.txt", "hello world")

    # The workspace path is injected via PYTHONPATH env, but we need to reference the absolute path
    workspace = sb.workspace
    assert workspace is not None

    code = f"""
import os
with open(os.path.join({workspace!r}, 'data.txt')) as f:
    print(f.read(), end='')
"""
    result = await sb.exec_python(code)
    assert result.exit_code == 0, f"exec failed: {result.stderr}"
    assert result.stdout == "hello world", f"Unexpected stdout: {result.stdout!r}"


@pytest.mark.asyncio
async def test_workspace_isolation_between_sandbox_instances() -> None:
    """Sandbox A writes 'secret' to data.txt.

    Sandbox B reads data.txt from ITS OWN workspace — file not found (SandboxError).
    Workspace directories must be separate.
    """
    from syrin.sandbox import Sandbox, SandboxError

    sb_a = Sandbox()
    sb_b = Sandbox()

    # Confirm workspaces are different directories
    assert sb_a.workspace != sb_b.workspace, "Sandbox instances share a workspace directory!"

    await sb_a.write("data.txt", "secret")

    # sb_b should not find the file written by sb_a
    with pytest.raises(SandboxError):
        await sb_b.read("data.txt")


# ──────────────────────────────────────────────────────────────────────────────
# Section 5 — Code length guard
# ──────────────────────────────────────────────────────────────────────────────


def test_code_length_limit_enforced() -> None:
    """Sandbox(max_code_length=100). exec_python with 200-char code raises SandboxError."""
    from syrin.sandbox import Sandbox, SandboxError

    sb = Sandbox(max_code_length=100)

    async def _run() -> None:
        await sb.exec_python("x" * 200)

    with pytest.raises(SandboxError, match="max_code_length"):
        asyncio.run(_run())


def test_code_length_just_under_limit_passes() -> None:
    """Sandbox(max_code_length=100). exec_python with 93-char code (x=1 + 90 spaces) succeeds."""
    from syrin.sandbox import Sandbox

    sb = Sandbox(max_code_length=100)

    code = "x=1" + " " * 90  # 93 chars — under 100
    assert len(code) == 93

    async def _run() -> None:
        # Does not raise at the length-check stage
        await sb.exec_python(code)

    # Should not raise SandboxError for length
    asyncio.run(_run())


# ──────────────────────────────────────────────────────────────────────────────
# Section 6 — Resource limits (memory)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.skipif(
    sys.platform != "linux",
    reason="RLIMIT_AS memory enforcement only reliably works on Linux",
)
@pytest.mark.slow
async def test_memory_limit_kills_process() -> None:
    """Sandbox(memory_mb=50). Trying to allocate 500MB triggers an error on Linux.

    On Linux: process is killed (MemoryError, SandboxMemoryError, or non-zero exit).
    We accept: SandboxMemoryError raised OR result.exit_code != 0.
    macOS/Windows are skipped because RLIMIT_AS behaves differently there.
    """
    from syrin.sandbox import Sandbox, SandboxError

    sb = Sandbox(memory_mb=50, timeout=5.0)

    code = "x = bytearray(500 * 1024 * 1024)"

    try:
        result = await sb.exec_python(code)
        # No exception: must at least have non-zero exit code
        assert result.exit_code != 0, (
            f"Expected non-zero exit for OOM, got exit_code={result.exit_code}"
        )
    except SandboxError:
        # SandboxMemoryError or SandboxTimeoutError — acceptable
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Section 7 — Robustness
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_code_exits_cleanly() -> None:
    """exec_python('') → exit_code == 0, stdout == '', stderr == ''."""
    from syrin.sandbox import Sandbox

    sb = Sandbox()
    result = await sb.exec_python("")
    assert result.exit_code == 0
    assert result.stdout == ""
    assert result.stderr == ""


@pytest.mark.asyncio
async def test_print_unicode_stdout() -> None:
    """exec_python with unicode → stdout contains the unicode characters."""
    from syrin.sandbox import Sandbox

    sb = Sandbox()
    result = await sb.exec_python('print("héllo wörld")')
    assert result.exit_code == 0
    assert "héllo" in result.stdout
    assert "wörld" in result.stdout


@pytest.mark.asyncio
@pytest.mark.slow
async def test_100_sequential_execs_same_sandbox() -> None:
    """Same Sandbox instance, 100 sequential exec_python calls.

    All succeed.  Verify temp files don't accumulate by checking workspace .py count.
    """
    from syrin.sandbox import Sandbox

    sb = Sandbox()
    workspace = sb.workspace
    assert workspace is not None

    for i in range(100):
        result = await sb.exec_python(f"print({i})")
        assert result.exit_code == 0, f"exec {i} failed: {result.stderr}"
        assert str(i) in result.stdout

    # After 100 execs, no leftover .py files should remain in the workspace
    py_files = list(Path(workspace).glob("*.py"))
    assert len(py_files) == 0, f"Found leftover .py files after 100 execs: {py_files}"


@pytest.mark.asyncio
@pytest.mark.slow
async def test_concurrent_timeouts_dont_leave_zombies() -> None:
    """5 concurrent exec_python calls each sleeping 10s with 0.05s timeout.

    All raise SandboxTimeoutError.  Subsequent exec calls must not hang —
    the processes were properly killed and waited.
    """
    from syrin.sandbox import Sandbox, SandboxTimeoutError

    sb = Sandbox(timeout=0.05)

    async def _timeout_call() -> None:
        with pytest.raises(SandboxTimeoutError):
            await sb.exec_python("import time; time.sleep(10)")

    await asyncio.gather(*[_timeout_call() for _ in range(5)])

    # Verify no hang by running a quick exec with a strict time limit
    sb2 = Sandbox()
    result = await asyncio.wait_for(sb2.exec_python("print('ok')"), timeout=5.0)
    assert result.exit_code == 0
    assert "ok" in result.stdout


# ──────────────────────────────────────────────────────────────────────────────
# Section 8 — Additional edge cases
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stdout_and_stderr_both_captured() -> None:
    """Code writes to both stdout and stderr — both are captured correctly."""
    from syrin.sandbox import Sandbox

    sb = Sandbox()
    code = "import sys; print('out'); print('err', file=sys.stderr)"
    result = await sb.exec_python(code)
    assert result.exit_code == 0
    assert "out" in result.stdout
    assert "err" in result.stderr


@pytest.mark.asyncio
async def test_environment_variable_injection() -> None:
    """env={'MY_VAR': 'stress_test'} is visible inside the sandbox."""
    from syrin.sandbox import Sandbox

    sb = Sandbox(env={"MY_VAR": "stress_test"})
    result = await sb.exec_python("import os; print(os.environ.get('MY_VAR', 'MISSING'))")
    assert result.exit_code == 0
    assert "stress_test" in result.stdout


@pytest.mark.asyncio
async def test_exec_duration_ms_positive_for_all_calls() -> None:
    """duration_ms is always positive, including for very fast executions."""
    from syrin.sandbox import Sandbox

    sb = Sandbox()
    for _i in range(5):
        result = await sb.exec_python("pass")
        assert result.duration_ms > 0, f"Expected positive duration_ms, got {result.duration_ms}"


@pytest.mark.asyncio
async def test_write_nested_path_creates_dirs() -> None:
    """sandbox.write('a/b/c.txt', 'data') creates intermediate directories."""
    from syrin.sandbox import Sandbox

    sb = Sandbox()
    await sb.write("a/b/c.txt", "nested")
    data = await sb.read("a/b/c.txt")
    assert data == b"nested"


@pytest.mark.asyncio
async def test_multiple_writes_same_path_overwrites() -> None:
    """Writing to the same path twice results in the second content being readable."""
    from syrin.sandbox import Sandbox

    sb = Sandbox()
    await sb.write("file.txt", "first")
    await sb.write("file.txt", "second")
    data = await sb.read("file.txt")
    assert data == b"second"


@pytest.mark.asyncio
@pytest.mark.slow
async def test_20_parallel_different_workspaces_no_cross_read() -> None:
    """20 sandboxes in parallel each write a unique sentinel; reads must match."""
    from syrin.sandbox import Sandbox

    sandboxes = [Sandbox() for _ in range(20)]

    async def _write_and_read(sb: Sandbox, sentinel: str) -> str:
        await sb.write("sentinel.txt", sentinel)
        data = await sb.read("sentinel.txt")
        return data.decode()

    sentinels = [f"sentinel_{i}" for i in range(20)]
    results = await asyncio.gather(
        *[_write_and_read(sb, s) for sb, s in zip(sandboxes, sentinels, strict=True)]
    )

    for expected, actual in zip(sentinels, results, strict=True):
        assert actual == expected, f"Expected {expected!r}, got {actual!r}"
