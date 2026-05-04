"""Resource Limits — Per-agent runtime resource controls.

Demonstrates how to configure Resource limits on an agent:
  - max_steps: cap LLM iterations
  - max_tools: cap total tool calls
  - max_context: cap context tokens
  - timeout: wall-clock time limit
  - ResourceThreshold: fire callbacks automatically when a dimension hits a %
  - DegradePolicy.tool_to_disable: remove a tool when max_tools is hit (DEGRADE mode)
  - OnExceed.WARN: continue with a warning instead of raising

Run:
    python examples/17_resource/resource_limits.py
"""

from syrin import Agent, Model
from syrin.enums import OnExceed, ResourceDimension
from syrin.exceptions import ResourceExceededError, ResourceTimeoutError
from syrin.resource import DegradePolicy, Resource, ResourceThreshold, ResourceTracker

model = Model.mock()


# ============================================================
# 1. Basic step and tool limits
# ============================================================
class StepLimitedAgent(Agent):
    """Agent capped at 3 LLM iterations and 5 tool calls per run."""

    model = Model.mock()
    resource = Resource(max_steps=3, max_tools=5)


agent1 = StepLimitedAgent()
result1 = agent1.run("What is recursion?")

print("=== Step + Tool Limited Agent ===")
print(f"Answer:     {result1.content[:60]}...")
print(f"Iterations: {agent1.iteration}")
print(f"Loop max:   {agent1._loop.max_iterations}")  # should be 3
print()


# ============================================================
# 2. Timeout limit
# ============================================================
agent2 = Agent(
    model=model,
    resource=Resource(timeout=30.0),  # 30-second hard stop
)

print("=== Timeout-Limited Agent ===")
try:
    result2 = agent2.run("Summarize the history of computing.")
    print(f"Answer: {result2.content[:60]}...")
except ResourceTimeoutError as e:
    print(f"Timed out: {e}")
print()


# ============================================================
# 3. OnExceed.WARN — continue with warning instead of raising
# ============================================================
warn_agent = Agent(
    model=model,
    resource=Resource(max_steps=10, on_exceed=OnExceed.WARN),
)
result3 = warn_agent.run("Describe three programming paradigms briefly.")
print("=== WARN Policy ===")
print(f"Answer: {result3.content[:60]}...")
print()


# ============================================================
# 4. ResourceThreshold — auto-fires callback at 80% usage
# ============================================================
# The callback is invoked automatically by ReactLoop via check_thresholds()
# after each LLM iteration. Here we demonstrate it directly on the tracker.
alerts: list[str] = []


def on_near_limit(state: object) -> None:
    tools_used = getattr(state, "tools_used", 0)
    alerts.append(f"Threshold fired — tools used: {tools_used}")


threshold = ResourceThreshold(
    at=80,
    dimension=ResourceDimension.TOOLS,
    action=on_near_limit,
)

tracker_demo = ResourceTracker(Resource(max_tools=10, thresholds=(threshold,)))
tracker_demo.start()

for _ in range(8):
    tracker_demo.record_tool_call()

# check_thresholds fires the callback once when usage >= 80%
state_demo = tracker_demo.state(steps=1, context_tokens=0)
tracker_demo.check_thresholds(state_demo)  # fires — 8/10 = 80%
tracker_demo.check_thresholds(state_demo)  # no-op — already fired

print("=== ResourceThreshold ===")
print(f"Alerts: {alerts}")  # ['Threshold fired — tools used: 8']
print()


# ============================================================
# 5. DegradePolicy — disable tool when max_tools hit
# ============================================================
# When max_tools is reached, ReactLoop removes the named tool from subsequent
# LLM calls and emits Hook.RESOURCE_DEGRADED. The run continues.
degrade = DegradePolicy(
    tool_to_disable="web_search",
)

resource_degrade = Resource(
    on_exceed=OnExceed.DEGRADE,
    degrade_policy=degrade,
    max_tools=20,
)

agent3 = Agent(model=model, resource=resource_degrade)
result4 = agent3.run("What is a microservice?")
print("=== DegradePolicy ===")
print(f"Answer: {result4.content[:60]}...")
print()


# ============================================================
# 6. ResourceTracker — standalone usage
# ============================================================
resource = Resource(max_tools=3)
tracker = ResourceTracker(resource)
tracker.start()

print("=== Standalone ResourceTracker ===")
for i in range(3):
    tracker.record_tool_call()
    state = tracker.state(steps=i + 1, context_tokens=(i + 1) * 1000)
    print(f"  Tool {i + 1}: pct={state.pct('tools'):.0%}")

try:
    tracker.record_tool_call()  # 4th call — exceeds max_tools=3
except ResourceExceededError as e:
    print(f"  Caught: {e}")

print()
print("Done — all resource limit examples completed.")
