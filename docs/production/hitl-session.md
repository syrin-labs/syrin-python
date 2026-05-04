---
title: HITL Session Persistence
description: Persist approval requests across process restarts using SQLiteApprovalSession so human reviewers can approve tool calls from a separate process or after a crash.
weight: 145
---

## Why Session Persistence?

Without persistence, every approval request lives only in process memory. If the agent process crashes or restarts while waiting for a human to approve a sensitive tool call, the request is lost — no record of what was pending, no way to resume.

`SQLiteApprovalSession` writes each request to a local SQLite file before the gate blocks. The session survives crashes, restarts, and deployments. An external reviewer can inspect pending sessions from a different process, approve or reject them, and the agent picks up the updated state on its next check.

## Quick Start

```python
from syrin.enums import ApprovalState, HITLTimeout
from syrin.hitl import ApprovalGate, SQLiteApprovalSession
from syrin.loop import HumanInTheLoop

store = SQLiteApprovalSession(path="./hitl.db")
gate = ApprovalGate(callback=my_approval_callback)

loop = HumanInTheLoop(
    approval_gate=gate,
    timeout=3600,                    # 1-hour window
    session=store,                   # enables persistence
    on_timeout=HITLTimeout.REJECT,   # default
)
```

`session=None` (the default) disables persistence and is fully backward-compatible with existing code.

## How it works

When `HumanInTheLoop` encounters a tool that needs approval, the sequence is:

1. A session is created in the SQLite file with state `PENDING` and an `expires_at` deadline.
2. The `HITL_REQUESTED` hook fires — wire this to Slack, email, or a dashboard.
3. The agent blocks on the gate callback.
4. A reviewer calls `session.resolve(sid, state=ApprovalState.APPROVED)` from any process.
5. The gate returns `True`; the agent executes the tool and marks the session `APPROVED`.

## SQLiteApprovalSession

```python
from syrin.hitl import SQLiteApprovalSession
from syrin.enums import ApprovalState

store = SQLiteApprovalSession(path="./hitl.db")

# Create a pending session — returns a UUID string
sid = store.create(
    tool_name="delete_user_data",
    arguments={"user_id": 42, "confirm": True},
    timeout=3600,
    message="Production deletion — DevOps approval required",
)

# Inspect
record = store.get(sid)
print(store.is_pending(sid))    # True

# List outstanding requests
for p in store.pending_sessions():
    print(p["tool_name"], p["state"])

# Resolve from any process
store.resolve(sid, state=ApprovalState.APPROVED)

# Sweep timed-out sessions (run on startup or periodically)
count = store.expire_stale()    # returns number marked TIMED_OUT
```

### Session record fields

| Field | Type | Description |
|---|---|---|
| `id` | `str` | UUID of the session |
| `tool_name` | `str` | Tool that triggered the approval request |
| `arguments` | `dict` | Deserialized tool arguments |
| `state` | `ApprovalState` | `PENDING`, `APPROVED`, `REJECTED`, or `TIMED_OUT` |
| `created_at` | `float` | Unix timestamp of creation |
| `expires_at` | `float` | Unix timestamp of expiry (`created_at + timeout`) |
| `message` | `str` | Human-readable context shown to the reviewer |

## State machine

Sessions flow in one direction only. Calling `resolve()` on an already-resolved session raises `HITLSessionError`.

```
              ┌─────────┐
   create()   │         │
   ──────────►│ PENDING │
              │         │
              └────┬────┘
                   │
         ┌─────────┼─────────┐
         │         │         │
      approve    reject    timeout
         │         │         │
         ▼         ▼         ▼
    APPROVED   REJECTED  TIMED_OUT
```

## Timeout behaviors

What happens when the approval window expires is controlled by `on_timeout`:

| Value | Behavior |
|---|---|
| `HITLTimeout.REJECT` | Treats timeout as rejection. Session set to `TIMED_OUT`. `HITL_TIMEOUT` hook fires. **(default)** |
| `HITLTimeout.APPROVE` | Auto-approves on timeout. Session set to `TIMED_OUT`. `HITL_TIMEOUT` hook fires. |
| `HITLTimeout.RAISE` | Raises `HITLTimeoutError`. Session set to `TIMED_OUT`. `HITL_TIMEOUT` hook fires. |

The `RAISE` pattern lets you handle timeouts explicitly in your application:

```python
from syrin.hitl import HITLTimeoutError

loop = HumanInTheLoop(
    approval_gate=gate,
    timeout=300,
    session=store,
    on_timeout=HITLTimeout.RAISE,
)

try:
    result = await loop.run(ctx, user_input)
except HITLTimeoutError as e:
    print(f"Approval timed out after {e.timeout}s")
    print(f"Session: {e.session_id}, tool: {e.tool_name}")
    # Alert on-call, escalate, log, etc.
```

## Resuming after a restart

Because sessions are written to disk before the gate blocks, they survive process restarts. On startup, recreate the store from the same file path and sweep for anything that expired during downtime:

```python
# On application startup
store = SQLiteApprovalSession(path="./hitl.db")

# Mark expired sessions as TIMED_OUT
store.expire_stale()

# Re-notify approvers for any still-live requests
for p in store.pending_sessions():
    notify_reviewer(session_id=p["id"], tool=p["tool_name"], message=p["message"])

# Wire the store into the loop for all new runs
loop = HumanInTheLoop(approval_gate=gate, session=store)
```

Recommended startup sequence:

1. Create `SQLiteApprovalSession` from the same path as before.
2. Call `store.expire_stale()` to close out sessions whose timeout window passed during downtime.
3. Iterate `store.pending_sessions()` and re-notify approvers for any that are still live.
4. Pass the store into `HumanInTheLoop(session=store)` for all new agent runs.

## Lifecycle hooks

Wire hooks to notify reviewers the moment a session is created, and to escalate when one times out:

| Hook | When it fires |
|---|---|
| `Hook.HITL_PENDING` | Before the gate blocks. Always fires, even without `session=`. |
| `Hook.HITL_REQUESTED` | After the session is persisted. Only fires when `session=` is set. Context includes `session_id`. |
| `Hook.HITL_APPROVED` | After the gate returns `True`. |
| `Hook.HITL_REJECTED` | After the gate returns `False` or on timeout-reject. |
| `Hook.HITL_TIMEOUT` | On any timeout path (REJECT, APPROVE, or RAISE). |

```python
agent.events.on(
    Hook.HITL_REQUESTED,
    lambda ctx: notify_slack(
        f"Approval needed for `{ctx['tool_name']}` — session {ctx['session_id'][:8]}..."
    ),
)
agent.events.on(
    Hook.HITL_TIMEOUT,
    lambda ctx: escalate_to_oncall(ctx["session_id"]),
)
```

## Thread safety

`SQLiteApprovalSession` is thread-safe. All public methods acquire a `threading.Lock` before opening the SQLite connection, so multiple threads in the same process can call `create()`, `resolve()`, and `pending_sessions()` concurrently.

Cross-process access is safe for `resolve()` and `get()` — SQLite handles the read-modify-write at this granularity without WAL mode. For very high throughput multi-process use, implement a Postgres-backed custom backend (see below).

## Custom backends

Any class that satisfies `ApprovalSessionProtocol` can be passed as `session=`. This lets you store approval state in Postgres, Redis, or any other store:

```python
from syrin.hitl._session import ApprovalSessionProtocol
from syrin.enums import ApprovalState

class RedisApprovalSession:
    def create(self, *, tool_name, arguments, timeout, message="") -> str: ...
    def resolve(self, session_id: str, *, state: ApprovalState) -> None: ...
    def get(self, session_id: str) -> dict | None: ...
    def is_pending(self, session_id: str) -> bool: ...
    def expire_stale(self) -> int: ...
    def pending_sessions(self) -> list[dict]: ...

# Verify at startup
assert isinstance(RedisApprovalSession(), ApprovalSessionProtocol)
```

## End-to-end example

```python
import asyncio
from unittest.mock import AsyncMock

from syrin.enums import ApprovalState, HITLTimeout
from syrin.hitl import ApprovalGate, SQLiteApprovalSession
from syrin.loop import HumanInTheLoop

store = SQLiteApprovalSession(path="/tmp/hitl_demo.db")

# Simulate an approver reviewing from a separate process:
# store.resolve(sid, state=ApprovalState.APPROVED)

# Gate backed by a real callback — e.g. POST to your approval webhook
gate = ApprovalGate(callback=AsyncMock(return_value=True))

loop = HumanInTheLoop(
    approval_gate=gate,
    timeout=1800,
    max_iterations=5,
    session=store,
    on_timeout=HITLTimeout.REJECT,
)

# Pending sessions survive a restart
store.expire_stale()
for p in store.pending_sessions():
    print(f"Resuming: {p['id'][:8]}... {p['tool_name']} — {p['state']}")
```

See [`examples/26_hitl/hitl_session.py`](../../examples/26_hitl/hitl_session.py) for a complete runnable walkthrough covering all five patterns: basic CRUD, `expire_stale()`, session resumption, `HITLTimeout` behaviors, and protocol conformance.
