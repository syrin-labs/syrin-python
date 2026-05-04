# HITL Session Persistence

> **Phase 6 — v0.12.0**
>
> `SQLiteApprovalSession`, `HITLTimeout`, `ApprovalState`, `HITLTimeoutError`

## Why persistence matters

Without session persistence, every approval request lives only in process memory.
If the agent process restarts while waiting for a human to approve, the request is
lost — no record of what was pending, no way to resume.

`SQLiteApprovalSession` writes each request to a local SQLite file before blocking
on the gate. The session survives crashes, restarts, and deployments. An external
reviewer can inspect pending sessions in another process, approve or reject them,
and the agent will see the updated state on its next check.

Typical scenario:

1. Agent calls `HumanInTheLoop.run()`, encounters a tool that needs approval.
2. A session is persisted with state `PENDING` and an `expires_at` deadline.
3. The `HITL_REQUESTED` hook fires — you can wire this to Slack, email, or a dashboard.
4. The agent blocks on its gate (your callback or webhook).
5. A reviewer approves via a separate CLI or web app that calls `session.resolve(sid, state=ApprovalState.APPROVED)`.
6. The gate returns `True`; the agent continues and sets the session to `APPROVED`.

---

## ApprovalState state machine

```
              ┌─────────┐
   create()   │         │
   ──────────►│ PENDING │
              │         │
              └────┬────┘
                   │
         ┌─────────┼─────────┐
         │         │         │
   approve       reject    timeout
         │         │         │
         ▼         ▼         ▼
    APPROVED   REJECTED  TIMED_OUT
```

Only `PENDING` → terminal transitions are allowed.
Calling `resolve()` on an already-resolved session raises `HITLSessionError`.

---

## SQLiteApprovalSession API

```python
from syrin.hitl import SQLiteApprovalSession
from syrin.enums import ApprovalState

# Create a session store (file created on first use)
store = SQLiteApprovalSession(path="./hitl.db")

# Create a pending session (returns UUID string)
sid = store.create(
    tool_name="delete_user_data",
    arguments={"user_id": 42, "confirm": True},
    timeout=3600,                          # 1-hour window
    message="Production deletion — DevOps approval required",
)

# Inspect
record = store.get(sid)         # dict with id, tool_name, arguments, state, …
print(store.is_pending(sid))    # True

# List all outstanding requests
for p in store.pending_sessions():
    print(p["tool_name"], p["state"])

# External approver resolves it
store.resolve(sid, state=ApprovalState.APPROVED)

# Sweep expired sessions (run on startup or periodically)
count = store.expire_stale()   # returns int — number marked TIMED_OUT
```

### Session dict keys

| Key | Type | Description |
|-----|------|-------------|
| `id` | `str` | UUID of the session |
| `tool_name` | `str` | Tool that requested approval |
| `arguments` | `dict` | Deserialized tool arguments |
| `state` | `ApprovalState` | `PENDING`, `APPROVED`, `REJECTED`, or `TIMED_OUT` |
| `created_at` | `float` | Unix timestamp of creation |
| `expires_at` | `float` | Unix timestamp of expiry (`created_at + timeout`) |
| `message` | `str` | Human-readable context for the approver |

---

## HumanInTheLoop configuration

```python
from syrin.enums import HITLTimeout
from syrin.hitl import ApprovalGate, SQLiteApprovalSession
from syrin.loop import HumanInTheLoop

gate = ApprovalGate(callback=my_slack_approval_fn)
store = SQLiteApprovalSession(path="./hitl.db")

loop = HumanInTheLoop(
    approval_gate=gate,
    timeout=1800,                   # 30-minute window
    max_iterations=5,
    session=store,                  # enable persistence
    on_timeout=HITLTimeout.REJECT,  # default
)
```

`session=None` (the default) disables persistence — fully backward-compatible
with existing code.

---

## HITLTimeout behaviors

| Value | Behavior on timeout |
|-------|---------------------|
| `HITLTimeout.REJECT` | Treats timeout as rejection (default). Session set to `TIMED_OUT`. `HITL_TIMEOUT` hook fires. |
| `HITLTimeout.APPROVE` | Auto-approves on timeout. Session set to `TIMED_OUT`. `HITL_TIMEOUT` hook fires. |
| `HITLTimeout.RAISE` | Raises `HITLTimeoutError`. Session set to `TIMED_OUT`. `HITL_TIMEOUT` hook fires. |

```python
# RAISE pattern — catch and handle explicitly
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
    # Alert on-call, escalate, etc.
```

---

## Session resumption on restart

```python
# Process 1 (before crash)
store = SQLiteApprovalSession(path="./hitl.db")
# … agent runs, creates session …

# Process 2 (after restart)
store = SQLiteApprovalSession(path="./hitl.db")
for p in store.pending_sessions():
    print(f"Pending: {p['id']} — {p['tool_name']}")
    # Reconnect to your gate, re-queue the approval request…

# Always expire stale sessions on startup
store.expire_stale()
```

Recommended startup sequence:

1. Create `SQLiteApprovalSession` from the same file path.
2. Call `store.expire_stale()` to clean up sessions whose timeout window passed.
3. Iterate `store.pending_sessions()` and re-notify approvers for any live requests.
4. Pass the store into `HumanInTheLoop(session=store)` for new runs.

---

## Lifecycle hooks

| Hook | When |
|------|------|
| `Hook.HITL_PENDING` | Immediately before the gate blocks (always fires, even without session). |
| `Hook.HITL_REQUESTED` | After the session is persisted; context includes `session_id`. Only fires when `session` is set. |
| `Hook.HITL_APPROVED` | After gate returns `True`. |
| `Hook.HITL_REJECTED` | After gate returns `False` or timeout-reject. |
| `Hook.HITL_TIMEOUT` | When timeout occurs (REJECT, APPROVE, or RAISE paths). |

```python
agent.events.on(Hook.HITL_REQUESTED, lambda ctx: notify_slack(ctx["session_id"]))
agent.events.on(Hook.HITL_TIMEOUT, lambda ctx: escalate(ctx["session_id"]))
```

---

## Thread safety

`SQLiteApprovalSession` is thread-safe via `threading.Lock`. All public methods
acquire the lock before opening the SQLite connection. Multiple threads in the
same process can safely call `create()`, `resolve()`, and `pending_sessions()`
concurrently.

Cross-process access is safe for `resolve()` and `get()` (SQLite WAL mode is not
required for read-modify-write at this granularity). For very high throughput
multi-process use, consider a Postgres-backed custom backend.

---

## Custom backends

Implement `ApprovalSessionProtocol` to plug in any storage backend:

```python
from syrin.hitl._session import ApprovalSessionProtocol
from syrin.enums import ApprovalState
from typing import runtime_checkable

class RedisApprovalSession:
    def create(self, *, tool_name, arguments, timeout, message=""):
        ...
    def resolve(self, session_id, *, state):
        ...
    def get(self, session_id):
        ...
    def is_pending(self, session_id):
        ...
    def expire_stale(self):
        ...
    def pending_sessions(self):
        ...

assert isinstance(RedisApprovalSession(), ApprovalSessionProtocol)
```

See [`ApprovalGateProtocol`](../../src/syrin/hitl/_gate.py) for the corresponding
gate interface.

---

## Example

See [`examples/26_hitl/hitl_session.py`](../../examples/26_hitl/hitl_session.py)
for a complete runnable example covering all five patterns above.
