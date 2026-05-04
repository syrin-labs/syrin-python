---
title: Testing
description: Comprehensive testing patterns for agents, tools, and models
weight: 270
---

## The Problem: Agents Are Hard to Test

Testing AI agents is challenging:

- **External dependencies** — LLM APIs are slow, expensive, and flaky
- **Non-deterministic output** — Same input → different output
- **Complex state** — Memory, context, budget tracking
- **Side effects** — Tool calls modify external systems

**The challenge:**
- You need fast, repeatable tests
- You can't hit real APIs in CI/CD
- You need to test agent logic, not the LLM
- State management complicates isolation

**The solution:** Syrin provides everything you need for comprehensive testing.

## Testing Philosophy

Testing follows a pyramid structure:

**E2E Tests (Top)**
- Full agent with real API
- Slowest, most expensive
- Fewest tests

**Integration Tests (Middle)**
- Agent with mock tools
- Moderate speed/cost
- More tests

**Unit Tests (Base)**
- Pure logic, no I/O
- Fastest, cheapest
- Most tests

Balance: Many fast unit tests, fewer integration tests, minimal E2E tests.

## Mocking Models: Model.mock()

The fastest way to test: replace real LLMs with mocks.

### Basic Usage

```python
from syrin import Agent, Model

# No API calls, no latency
mock_model = Model.mock()

agent = Agent(model=mock_model)
result = agent.run("Hello!")

assert result.content is not None
```

### Configure Response

```python
# Fixed response
mock = Model.mock(
    custom_response="The weather is sunny with a high of 72°F.",
)

# Lorem ipsum
mock = Model.mock(
    response_mode="lorem",
    lorem_length=100,  # 100 characters
)

# Latency simulation
mock = Model.mock(
    latency_min=0.5,   # Min 0.5 seconds
    latency_max=1.0,    # Max 1 second
)
```

### Pricing Tiers for Budget Testing

```python
# Test with different pricing
cheap_model = Model.mock(pricing_tier="low")      # ~$0.10/1M tokens
medium_model = Model.mock(pricing_tier="medium")  # ~$1.00/1M tokens
expensive_model = Model.mock(pricing_tier="high")  # ~$10.00/1M tokens
ultra_expensive = Model.mock(pricing_tier="ultra_high")
```

## Mocking Tools

### Option 1: Mock Function

```python
from syrin import Agent, tool
from unittest.mock import AsyncMock

class TestAgent(Agent):
    @tool
    def get_weather(self, city: str) -> str:
        """Get weather for a city."""
        # This will be mocked in tests
        return ""


def test_weather_tool(monkeypatch):
    agent = TestAgent(model=Model.mock())
    
    # Monkeypatch the tool function
    original = agent.get_weather.func
    agent.get_weather.func = lambda city: f"Weather in {city}: 72°F"
    
    result = agent.run("What's the weather in Tokyo?")
    assert "Tokyo" in result.content
    assert "72°F" in result.content
    
    # Restore
    agent.get_weather.func = original
```

### Option 2: Subclass and Override

```python
class MockAgent(TestAgent):
    @tool
    def get_weather(self, city: str) -> str:
        """Mock implementation."""
        return f"Weather in {city}: 72°F"


def test_with_mock():
    agent = MockAgent(model=Model.mock())
    result = agent.run("Weather in Paris?")
    assert "Paris" in result.content
```

### Option 3: Dependency Injection (Recommended)

```python
from dataclasses import dataclass
from syrin import Agent, tool, RunContext

@dataclass
class TestDeps:
    weather_service: WeatherService


class MyAgent(Agent):
    @tool
    def get_weather(self, ctx: RunContext[TestDeps], city: str) -> str:
        # Uses injected service
        return ctx.deps.weather_service.get(city)


# Test with fake
class FakeWeatherService:
    def get(self, city: str) -> str:
        return f"Weather in {city}: 72°F"


def test():
    agent = MyAgent(deps=TestDeps(weather_service=FakeWeatherService()))
    result = agent.run("Weather in London?")
```

## Unit Testing Tools

```python
import pytest
from syrin.tool import tool, ToolSpec


def test_tool_decorator():
    """Test @tool decorator creates correct spec."""
    
    @tool
    def get_user(user_id: str) -> str:
        """Get a user by ID."""
        return f"User {user_id}"
    
    assert isinstance(get_user, ToolSpec)
    assert get_user.name == "get_user"
    assert get_user.description == "Get a user by ID."
    assert "user_id" in get_user.parameters_schema["properties"]
    assert "required" in get_user.parameters_schema


def test_tool_with_defaults():
    """Test tool with optional parameters."""
    
    @tool
    def search(query: str, limit: int = 10, offset: int = 0) -> str:
        """Search with pagination."""
        return f"Results {offset}-{offset + limit} for {query}"
    
    assert "query" in search.parameters_schema["required"]
    assert "limit" not in search.parameters_schema["required"]


def test_tool_execution():
    """Test tool function is callable."""
    
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    
    result = add.func(2, 3)
    assert result == 5
```

## Integration Testing

### Full Agent Tests

```python
import pytest
from syrin import Agent, Model, Memory
from syrin.enums import MemoryType


@pytest.fixture
def agent():
    """Create a test agent."""
    return Agent(
        model=Model.mock(),
        memory=Memory(),
    )


def test_simple_response(agent):
    """Test basic response."""
    result = agent.run("Hello!")
    assert result.content is not None
    assert len(result.content) > 0


def test_with_memory(agent):
    """Test memory integration."""
    agent.remember("Python is great", memory_type=MemoryType.FACTS)
    
    result = agent.run("What programming language do I like?")
    assert "python" in result.content.lower()


def test_cost_tracking(agent):
    """Test cost is tracked."""
    agent.model = Model.mock(pricing_tier="high")
    
    result = agent.run("Tell me a story")
    assert result.cost >= 0  # Cost should be tracked
```

### Testing with Real-ish Models

Use LiteLLM for more realistic mock responses:

```python
from syrin import Model

# Litellm can proxy to real APIs or mocks
# Configure a test endpoint that returns structured responses

test_model = Model.LiteLLM(
    model_name="test/local",
    api_base="http://localhost:4000",  # Your mock server
    api_key="test",
)
```

## Fixture Patterns

### pytest Fixtures

```python
import pytest
from dataclasses import dataclass

from syrin import Agent, Model, Budget
from syrin.memory import Memory


@dataclass
class TestDeps:
    db: MockDatabase
    cache: MockCache


@pytest.fixture
def mock_deps():
    return TestDeps(
        db=MockDatabase(),
        cache=MockCache(),
    )


@pytest.fixture
def memory():
    return Memory()


@pytest.fixture
def agent(mock_deps, memory):
    return Agent(
        model=Model.mock(),
        deps=mock_deps,
        memory=memory,
        budget=Budget(max_cost=10.00),
    )


# Use in tests
def test_with_fixtures(agent, memory):
    agent.remember("Test fact", memory_type=MemoryType.FACTS)
    result = agent.run("What fact do you remember?")
    assert "Test fact" in result.content
```

### Session-Scoped Fixtures

```python
@pytest.fixture(scope="session")
def db_connection():
    """One DB connection for entire test session."""
    conn = create_test_connection()
    yield conn
    conn.close()


@pytest.fixture
def clean_memory():
    """Fresh memory for each test."""
    return Memory()
```

## Testing with Events

Verify events fire correctly:

```python
def test_events_fire(agent):
    """Test that hooks fire as expected."""
    events = []
    
    def capture(hook, ctx):
        events.append(hook)
    
    agent.events.on_all(capture)
    agent.run("Hello")
    
    assert any(Hook.AGENT_RUN_START in str(e) for e in events)
    assert any(Hook.AGENT_RUN_END in str(e) for e in events)


def test_cost_event(agent):
    """Test cost tracking via events."""
    costs = []
    
    def track_cost(ctx):
        costs.append(ctx.get("cost", 0))
    
    agent.events.on(Hook.AGENT_RUN_END, track_cost)
    
    agent.run("Hello")
    agent.run("Hello")
    
    assert len(costs) == 2
```

## Testing Budget Enforcement

```python
def test_budget_limit():
    """Test that budget limits are enforced."""
    agent = Agent(
        model=Model.mock(pricing_tier="ultra_high"),
        budget=Budget(max_cost=0.01),  # Tiny budget
    )
    
    # Should hit budget limit
    with pytest.raises(BudgetExceededError):
        for _ in range(100):
            agent.run("Hello")


def test_budget_callback():
    """Test budget callbacks."""
    callback_called = []
    
    def on_exceeded(ctx):
        callback_called.append(ctx)
    
    agent = Agent(
        model=Model.mock(pricing_tier="high"),
        budget=Budget(
            max_cost=1.00,
            on_exceeded=on_exceeded,
        ),
    )
    
    # Exhaust budget
    for _ in range(50):
        try:
            agent.run("Hi")
        except BudgetExceeded:
            break
    
    assert len(callback_called) > 0
```

## Testing Multi-Agent

```python
def test_spawn():
    """Test agent spawn (task delegation)."""
    from syrin import Agent
    from syrin.enums import AgentType
    
    agent1 = Agent(model=Model.mock(), agent_type=AgentType.ROUTER)
    agent2 = Agent(model=Model.mock(), agent_type=AgentType.WORKER)
    
    # Setup spawn target
    agent1._register_spawn_target(agent2)
    
    result = agent1.run("Transfer to worker")
    
    # Verify spawn occurred
    assert "worker" in result.content.lower() or result.spawn_completed


def test_spawn():
    """Test spawning sub-agents."""
    parent = Agent(model=Model.mock())
    
    # Spawn a child agent
    child = parent.spawn(
        agent_type=AgentType.WORKER,
        context={"task": "analysis"},
    )
    
    assert child is not None
    assert child.parent == parent
```

## Testing Error Handling

```python
def test_tool_error_handling():
    """Test graceful handling of tool errors."""
    agent = TestAgent(model=Model.mock())
    
    # Mock tool that raises
    def failing_tool():
        raise ConnectionError("Database unavailable")
    
    agent._tools["failing_tool"] = ToolSpec(
        name="failing_tool",
        description="A failing tool",
        func=failing_tool,
    )
    
    # Should handle gracefully
    result = agent.run("Use failing_tool")
    
    # Agent should recover or explain error
    assert result is not None


def test_api_error_recovery():
    """Test fallback on API errors."""
    # Create model with fallback
    primary = Model.mock()  # Will fail
    fallback = Model.mock()
    
    model = primary.with_fallback(fallback)
    agent = Agent(model=model)
    
    # Should fall back gracefully
    result = agent.run("Hello")
    assert result.content is not None
```

## Property-Based Testing

```python
import hypothesis
from hypothesis import given, strategies as st


@given(
    name=st.text(min_size=1, max_size=100),
    email=st.emails(),
)
@hypothesis.settings(max_examples=100)
def test_user_tool(name, email):
    """Property-based test for user creation."""
    
    @tool
    def create_user(name: str, email: str) -> dict:
        return {"name": name, "email": email}
    
    result = create_user.func(name, email)
    
    assert result["name"] == name
    assert result["email"] == email
```

## CI/CD Considerations

### Environment Variables

```bash
# In CI, don't set real API keys
export SYRIN_TESTING=1
# syrin will use mocks automatically

# Or explicitly use mock model
AGENT_MODEL=almock
```

### Parallel Execution

```python
import pytest

@pytest.mark.parametrize("prompt", [
    "Hello",
    "How are you?",
    "What's the weather?",
])
def test_prompts(prompt):
    agent = Agent(model=Model.mock())
    result = agent.run(prompt)
    assert result.content is not None
```

### Coverage

```bash
# Run with coverage
pytest --cov=syrin --cov-report=term-missing tests/
```

## Test Organization

```
tests/
├── conftest.py              # Shared fixtures
├── unit/
│   ├── test_tools.py        # Tool tests
│   ├── test_memory.py      # Memory tests
│   └── test_budget.py       # Budget tests
├── integration/
│   ├── test_agents.py      # Agent integration
│   ├── test_multi_agent.py # Multi-agent
│   └── test_mcp.py         # MCP integration
└── e2e/
    └── test_full_flows.py  # End-to-end tests
```

### conftest.py Example

```python
import pytest
import os

# Set testing mode
os.environ["SYRIN_TESTING"] = "1"


@pytest.fixture(autouse=True)
def reset_state():
    """Reset global state between tests."""
    # Clear registries
    from syrin.model import ModelRegistry
    ModelRegistry().clear()
    
    yield
    
    # Cleanup
    ModelRegistry().clear()
```

## Test Data Management

```python
# fixtures/test_data.py
TEST_USERS = [
    {"id": "1", "name": "Alice", "email": "alice@test.com"},
    {"id": "2", "name": "Bob", "email": "bob@test.com"},
]

TEST_DOCUMENTS = [
    {"id": "doc1", "content": "Important document content..."},
]


# In tests
def test_with_test_data():
    mock_db = MockDatabase(users=TEST_USERS)
    agent = MyAgent(deps=TestDeps(db=mock_db))
```

## Performance Testing

```python
import time

def test_response_time():
    """Test agent responds within acceptable time."""
    agent = Agent(model=Model.mock(latency_seconds=0.1))
    
    start = time.time()
    agent.run("Hello")
    duration = time.time() - start
    
    assert duration < 1.0  # Should be fast with mock


def test_concurrent_requests():
    """Test handling concurrent requests."""
    import concurrent.futures
    
    agent = Agent(model=Model.mock())
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(agent.run, f"Request {i}")
            for i in range(10)
        ]
        
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    assert len(results) == 10
```

## Public Testing-Friendly Types

The shared public types that commonly show up in tests include `MultimodalInput`, `ValidationAttempt`, and `CostInfo`. They are useful when asserting on media payloads, output-validation retries, and pricing/cost calculations directly.

## What's Next?

- [Custom Model](/agent-kit/advanced/custom-model) — Test custom providers
- [Dependency Injection](/agent-kit/advanced/dependency-injection) — Testable architectures
- [Event Bus](/agent-kit/advanced/event-bus) — Test event handling

## See Also

- [Error Handling](/agent-kit/agent/error-handling) — Graceful failure
- [Guardrails](/agent-kit/agent/guardrails) — Safety testing
- [Checkpointing](/agent-kit/production/checkpointing) — State testing
