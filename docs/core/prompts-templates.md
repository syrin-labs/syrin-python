---
title: Prompt Templates
description: Create dynamic, parameterized prompts with runtime variable injection, composition, and the powerful @prompt decorator. Make your prompts adaptable.
weight: 41
---

# The Problem: Static Prompts Have Limits

You've written the perfect system prompt for your customer support agent. But now you need:
- A Spanish version for users in Mexico
- A formal version for enterprise clients
- A casual version for younger demographics

Do you copy-paste the entire prompt and modify it? That leads to:
- **Duplicated code** — Three copies of the same logic
- **Maintenance nightmares** — Fix a bug? Update three places
- **Inconsistency** — Small differences creep in over time

Or imagine you're building a multi-tenant SaaS product where each customer wants their AI assistant customized with their brand voice, policies, and knowledge.

**Static prompts can't scale.**

**The solution?** Prompt templates—parameterized prompts that generate the right system prompt for the right situation.

---

## The Core Idea: Prompts as Functions

Think of prompt templates like functions:

```python
# A static prompt is like a hardcoded value
system_prompt = "You are a helpful assistant."

# A prompt template is like a function
@prompt
def support_agent(language: str, company_name: str) -> str:
    return f"You are a {company_name} support agent. Respond in {language}."
```

Just like functions, templates:
- **Take parameters** — Customize behavior
- **Encapsulate logic** — Single source of truth
- **Are reusable** — Use the same template for different contexts
- **Are testable** — Verify output matches expectations

---

## Quick Start: Your First Prompt Template

```python
from syrin import Agent, Model, prompt

# Define a parameterized prompt
@prompt
def persona_prompt(name: str, tone: str = "professional") -> str:
    return f"You are {name}, a helpful assistant. Be {tone}."

# Use it in an agent
agent = Agent(
    model=Model.OpenAI("gpt-4o", api_key="your-api-key"),
    system_prompt=persona_prompt,  # Pass the template
    template_variables={"name": "Alice", "tone": "friendly"},
)

response = agent.run("Hello!")
print(response.content)
```

**What just happened?**
- You defined a prompt template with parameters
- You passed it to the agent
- At runtime, the template resolved with your variables
- The agent got: "You are Alice, a helpful assistant. Be friendly."

---

## The @prompt Decorator: Native Python f-Strings

Syrin's `@prompt` decorator turns any function into a parameterized prompt. It uses native Python f-string syntax—no special template language to learn.

### Basic Usage

```python
from syrin import prompt

@prompt
def greeting_prompt(user_name: str, time_of_day: str) -> str:
    return f"Good {time_of_day}, {user_name}! How can I help you today?"
```

**That's it!** The function becomes a prompt template.

### Calling Templates

```python
# Direct call
result = greeting_prompt(user_name="Alice", time_of_day="morning")
# Returns: "Good morning, Alice! How can I help you today?"

# With defaults
result = greeting_prompt(user_name="Bob")  # time_of_day defaults to its default value
```

### Enterprise Features

The `@prompt` decorator adds powerful features:

```python
@prompt
def expert_prompt(domain: str, complexity: str = "intermediate") -> str:
    """Generate expert-level system prompts."""
    return f"You are an expert in {domain}. Explain at {complexity} level."

# Access metadata
print(expert_prompt.name)          # "expert_prompt"
print(expert_prompt.variables)     # List of parameters with types
print(expert_prompt.version)       # Version tracking
print(expert_prompt.template_hash) # Hash for change detection

# Validation
expert_prompt.validate(domain="Python", complexity="expert")  # Returns True or raises

# Testing
result = expert_prompt.test_render(domain="AI", complexity="beginner")
print(f"Output: {result['output']}")
print(f"Est. tokens: {result['estimated_tokens']}")
```

---

## Using Templates with Agents

### Method 1: Constructor Variables

```python
@prompt
def support_agent(language: str, company_name: str) -> str:
    return f"You are the {company_name} support bot. Respond in {language}."

# Class-level templates
class EnglishSupport(Agent):
    model = Model.OpenAI("gpt-4o", api_key="your-key")
    system_prompt = support_agent
    template_variables = {"language": "English", "company_name": "AcmeCorp"}

# Instance-level override
agent = EnglishSupport(
    template_variables={"company_name": "NewClient Inc."}  # Merges with class vars
)
```

### Method 2: Per-Call Variables

```python
@prompt
def personalized_prompt(user_name: str, tier: str) -> str:
    return f"You are helping {user_name}, a {tier} customer."

agent = Agent(
    model=Model.OpenAI("gpt-4o", api_key="your-key"),
    system_prompt=personalized_prompt,
)

# Different users get different prompts
agent.run("Help me", template_variables={"user_name": "Alice", "tier": "premium"})
agent.run("Help me", template_variables={"user_name": "Bob", "tier": "free"})
```

### Variable Merge Order

Variables are merged in this order (later overrides earlier):

1. **Class-level** `template_variables`
2. **Instance-level** `template_variables`
3. **Per-call** `template_variables`

```python
class BaseAgent(Agent):
    system_prompt = my_prompt
    template_variables = {"tone": "professional", "company": "Acme"}  # Layer 1

agent = BaseAgent(
    template_variables={"company": "NewCo"}  # Layer 2: overrides company
)

agent.run("Hello", template_variables={"tone": "casual"})  # Layer 3: overrides tone
# Result: {"tone": "casual", "company": "NewCo"}
```

---

## Built-in Variables

Syrin automatically injects useful variables:

```python
@prompt
def contextual_prompt(date: str, agent_id: str, conversation_id: str) -> str:
    return f"""You are a helpful assistant.
    
Today's date is: {date}
Your agent ID is: {agent_id}
This conversation ID is: {conversation_id}
"""
```

These are available automatically unless you disable them:

```python
agent = Agent(
    model=Model.OpenAI("gpt-4o", api_key="your-key"),
    system_prompt=my_prompt,
    inject_builtins=False,  # Disable automatic injection
)
```

---

## @system_prompt: In-Class Prompt Definition

For better encapsulation, define prompts directly in your agent class:

```python
from syrin import Agent, Model, system_prompt, PromptContext

class PersonaAgent(Agent):
    model = Model.OpenAI("gpt-4o", api_key="your-key")

    @system_prompt
    def my_prompt(self, user_name: str = "", tone: str = "professional") -> str:
        """Personalized system prompt with user context."""
        return f"You assist {user_name or 'the user'}. Be {tone}."

# Use it
agent = PersonaAgent(template_variables={"user_name": "Alice", "tone": "friendly"})
```

**Supported method signatures:**

```python
class MyAgent(Agent):
    model = Model.OpenAI("gpt-4o", api_key="your-key")

    # Option 1: No parameters
    @system_prompt
    def basic_prompt(self) -> str:
        return "You are a helpful assistant."

    # Option 2: Template variables
    @system_prompt
    def dynamic_prompt(self, user_name: str = "") -> str:
        return f"You help {user_name}."

    # Option 3: Full context access
    @system_prompt
    def context_aware_prompt(self, ctx: PromptContext) -> str:
        user = ctx.memory.recall("user_profile") if ctx.memory else None
        return f"You help {user.name if user else 'anyone'}."
```

---

## Dynamic Prompts with PromptContext

For complex scenarios, access full agent state at runtime:

```python
from syrin import Agent, Model
from syrin.prompt import PromptContext

def build_dynamic_prompt(ctx: PromptContext) -> str:
    """Build prompt based on agent state at runtime."""
    
    # Access agent
    agent_name = ctx.agent.name
    budget_state = ctx.budget_state
    
    # Access memory
    memories = []
    if ctx.memory:
        memories = ctx.memory.recall("preferences", limit=3)
    
    # Access time
    current_date = ctx.date.strftime("%Y-%m-%d")
    
    # Build dynamic prompt
    prompt_parts = [
        f"You are {agent_name}, a helpful AI assistant.",
        f"Today's date is {current_date}.",
    ]
    
    if memories:
        prompt_parts.append(f"User preferences: {memories}")
    
    if budget_state and budget_state.get("remaining", 0) < 0.10:
        prompt_parts.append("Budget is low. Be extra concise.")
    
    return "\n".join(prompt_parts)


agent = Agent(
    model=Model.OpenAI("gpt-4o", api_key="your-key"),
    system_prompt=build_dynamic_prompt,
)
```

### PromptContext Fields

`PromptContext` exposes seven fields for dynamic prompt construction. `agent` provides the `Agent` instance itself. `agent_id` is a `str` containing the agent's name or class name. `conversation_id` is a `str | None` with the current conversation ID, if one exists. `memory` provides access to the agent's memory backend. `budget_state` reflects the current budget state. `date` is the current UTC `datetime`. `builtins` is a `dict` of automatically injected built-in variables such as the date and agent ID.

---

## Template Composition

Combine multiple prompts into one:

### Composing with `.compose()`

```python
@prompt
def base_prompt() -> str:
    return "You are a helpful AI assistant."

@prompt
def code_prompt(language: str) -> str:
    return f"Provide code examples in {language} when relevant."

@prompt
def style_prompt(style: str) -> str:
    return f"Use a {style} communication style."

# Combine prompts
full_prompt = base_prompt.compose(code_prompt, style_prompt)

# Use
result = full_prompt(language="Python", style="concise")
# Returns: "You are a helpful AI assistant.\n\nProvide code examples in Python when relevant.\n\nUse a concise communication style."
```

### Partial Application

Fix some variables, leave others flexible:

```python
@prompt
def support_prompt(company: str, language: str) -> str:
    return f"You are {company} support. Respond in {language}."

# Create company-specific template
company_prompt = support_prompt.partial(company="AcmeCorp")

# Use with different languages
result1 = company_prompt(language="English")
result2 = company_prompt(language="Spanish")
result3 = company_prompt(language="French")
```

---

## Validation and Testing

### Parameter Validation

```python
@prompt
def validated_prompt(
    domain: str,  # Required
    tone: str = "professional",  # Optional with default
) -> str:
    return f"You are a {domain} expert. Be {tone}."

# Validate before using
try:
    validated_prompt.validate(domain="Python", tone="friendly")
    print("Valid!")
except ValueError as e:
    print(f"Invalid: {e}")

# Validate with missing required parameter
try:
    validated_prompt.validate(tone="friendly")  # Missing domain
except ValueError as e:
    print(f"Invalid: {e}")
    # Output: Invalid: Required parameter 'domain' is missing
```

### Test Rendering

```python
result = validated_prompt.test_render(domain="AI", tone="concise")

print(f"Output: {result['output']}")
print(f"Length: {result['length']} chars")
print(f"Est. tokens: {result['estimated_tokens']}")
print(f"Version: {result['version']}")
print(f"Hash: {result['hash']}")
```

### Caching

Templates cache results by default:

```python
@prompt
def expensive_prompt(config: str) -> str:
    # Complex logic here
    return f"Generated with {config}"

# First call - executes
result1 = expensive_prompt(config="A")

# Second call with same params - uses cache
result2 = expensive_prompt(config="A")

# Different params - executes and caches
result3 = expensive_prompt(config="B")

# Clear cache
expensive_prompt.clear_cache()

# Check cache stats
stats = expensive_prompt.get_cache_stats()
print(f"Cache size: {stats['size']}")
```

---

## Real-World Examples

### Example 1: Multi-Language Support

```python
@prompt
def multilingual_support(
    company_name: str,
    language: str,
    region: str,
) -> str:
    return f"""You are {company_name} customer support.
    
LANGUAGE: Respond entirely in {language}.
REGION CONTEXT: {region}

GREETING:
- English: "Hello! How can I help?"
- Spanish: "¡Hola! ¿Cómo puedo ayudar?"
- French: "Bonjour! Comment puis-je aider?"
- German: "Guten Tag! Wie kann ich helfen?"

Always respond in the requested {language}."""

# Create agents for different markets
english_agent = Agent(
    model=Model.OpenAI("gpt-4o", api_key="your-key"),
    system_prompt=multilingual_support,
    template_variables={
        "company_name": "TechCorp",
        "language": "English",
        "region": "United States",
    },
)

spanish_agent = Agent(
    model=Model.OpenAI("gpt-4o", api_key="your-key"),
    system_prompt=multilingual_support,
    template_variables={
        "company_name": "TechCorp",
        "language": "Spanish",
        "region": "Latin America",
    },
)
```

### Example 2: Tiered Customer Service

```python
@prompt
def tiered_support(tier: str, features: str) -> str:
    base = f"You are a {tier} support specialist."
    
    if tier == "free":
        return f"{base}\n\nYou can help with common questions. For advanced issues, suggest upgrading."
    
    elif tier == "pro":
        return f"""{base}
        
FEATURES: {features}

You have access to:
- Priority response queue
- Extended support hours
- Direct escalation path
"""
    
    elif tier == "enterprise":
        return f"""{base}
        
FEATURES: {features}

You have access to:
- Dedicated support line
- 24/7 availability
- Custom integrations
- Account manager liaison
"""
    
    return base


# Usage
agent = Agent(
    model=Model.OpenAI("gpt-4o", api_key="your-key"),
    system_prompt=tiered_support,
)

# Per-user templates
agent.run(query, template_variables={"tier": "pro", "features": "API access, analytics"})
```

### Example 3: Domain-Specific Assistants

```python
@prompt
def domain_expert(domain: str, expertise_level: str, use_cases: str) -> str:
    return f"""You are an expert in {domain}.

EXPERTISE LEVEL: {expertise_level}
- beginner: Explain concepts simply, avoid jargon
- intermediate: Assume basic knowledge
- advanced: Technical depth welcome

PRIMARY USE CASES:
{use_cases}

RESPONSE STYLE:
- Lead with the most important information
- Provide concrete examples
- Anticipate follow-up questions
- Include practical next steps
"""


class PythonExpert(Agent):
    model = Model.OpenAI("gpt-4o", api_key="your-key")
    system_prompt = domain_expert
    template_variables = {
        "domain": "Python Programming",
        "expertise_level": "intermediate",
        "use_cases": "- Web development (Django, Flask)\n- Data analysis (pandas, numpy)\n- Automation scripts",
    }


class FinanceAnalyst(Agent):
    model = Model.OpenAI("gpt-4o", api_key="your-key")
    system_prompt = domain_expert
    template_variables = {
        "domain": "Financial Analysis",
        "expertise_level": "advanced",
        "use_cases": "- Investment analysis\n- Risk assessment\n- Portfolio optimization",
    }
```

### Example 4: Context-Aware Memory Integration

```python
@prompt
def memory_aware_prompt(ctx: PromptContext) -> str:
    """Build prompt with user's memory context."""
    
    user_info = ""
    if ctx.memory:
        # Get user's core facts
        facts = ctx.memory.recall("core_facts", memory_type="facts", limit=5)
        if facts:
            user_info = f"User profile: {facts}"
        
        # Get conversation history
        history = ctx.memory.recall("recent_topics", memory_type="history", limit=3)
        if history:
            user_info += f"\nRecent topics: {history}"
    
    return f"""You are a personalized assistant.

{user_info if user_info else "No prior context available."}

Remember to:
- Acknowledge previous conversations
- Build on established preferences
- Maintain continuity
"""


agent = Agent(
    model=Model.OpenAI("gpt-4o", api_key="your-key"),
    system_prompt=memory_aware_prompt,
    memory=Memory(),  # Enable memory
)
```

---

## Advanced: Custom Validators

```python
from syrin.prompt import Prompt, PromptValidation, validated

@prompt
@validated(
    min_length=3,
    max_length=50,
)
def name_prompt(name: str) -> str:
    return f"Hello, {name}!"

# This will fail
try:
    name_prompt(name="A")  # Too short
except ValueError as e:
    print(f"Error: {e}")
    # Output: Error: Parameter 'name' must be at least 3 characters

# Custom validator
@prompt
@validated(custom=lambda x: x.startswith("support_"))
def role_prompt(role: str) -> str:
    return f"You are the {role} bot."

try:
    role_prompt(role="admin")  # Doesn't start with support_
except ValueError as e:
    print(f"Error: {e}")
```

---

## Best Practices

### 1. Keep Templates Focused

```python
# Good: Single responsibility
@prompt
def persona_prompt(personality: str, expertise: str) -> str:
    return f"You are {personality}. Expert in {expertise}."

# Avoid: Too many parameters
@prompt
def everything_prompt(a, b, c, d, e, f, g, h) -> str:
    # Hard to maintain and use
    return "..."
```

### 2. Use Sensible Defaults

```python
@prompt
def helpful_prompt(tone: str = "friendly", verbosity: str = "medium") -> str:
    return f"You are a {tone} assistant. Be {verbosity} verbose."

# Easy to use with defaults
agent = Agent(system_prompt=helpful_prompt)

# Easy to customize
agent = Agent(system_prompt=helpful_prompt, template_variables={"tone": "formal"})
```

### 3. Document Your Templates

```python
@prompt
def api_support_prompt(
    api_version: str,
    language: str,
) -> str:
    """
    Generate API support prompts for different versions and languages.
    
    Args:
        api_version: API version (e.g., "v1", "v2")
        language: Response language (e.g., "en", "es", "fr")
    
    Returns:
        System prompt for API support agent
    """
    return f"You are an API expert for version {api_version}. Respond in {language}."
```

### 4. Test Template Outputs

```python
@prompt
def my_template(param1: str, param2: str) -> str:
    return f"{param1} and {param2}"

# Test all parameter combinations
test_cases = [
    {"param1": "A", "param2": "B"},
    {"param1": "", "param2": "B"},
    {"param1": "A", "param2": ""},
]

for case in test_cases:
    result = my_template.test_render(**case)
    assert result['length'] > 0, f"Empty output for {case}"
    print(f"✓ {case}: {result['length']} chars")
```

### 5. Version Your Templates

```python
from syrin.prompt import Prompt, PromptVersion

prompt_v1 = Prompt(
    my_function,
    version=PromptVersion(1, 0, 0),
)

# When you update
prompt_v2 = Prompt(
    updated_function,
    version=PromptVersion(1, 1, 0),  # Minor bump for additions
)

# Compare versions
print(f"v{prompt_v1.version} -> v{prompt_v2.version}")
print(f"Changed: {prompt_v1.template_hash != prompt_v2.template_hash}")
```

---

## Troubleshooting

### "Missing required parameter"

```python
# Error: Required parameter 'domain' is missing
# Fix: Provide the parameter
agent = Agent(
    system_prompt=my_template,
    template_variables={"domain": "Python"},  # Add missing param
)
```

### "Prompt not resolving"

```python
# Check effective variables
agent = Agent(system_prompt=my_template, template_variables={"a": "A"})
print(agent.effective_template_variables())  # See what will be used

# Check resolved prompt
print(agent._system_prompt)  # After init
```

### "Variables not being injected"

```python
# Ensure variable names match template parameters
@prompt
def my_prompt(user_name: str) -> str:  # Note: user_name, not userName
    return f"Hello {user_name}!"

# Wrong
agent = Agent(system_prompt=my_prompt, template_variables={"userName": "Alice"})

# Correct
agent = Agent(system_prompt=my_prompt, template_variables={"user_name": "Alice"})
```

---

## Event Hooks for Debugging

Syrin emits events for prompt resolution:

```python
agent.events.on("system_prompt.before_resolve", lambda e: 
    print(f"Resolving with vars: {e['template_variables']}")
)

agent.events.on("system_prompt.after_resolve", lambda e:
    print(f"Resolved to: {e['resolved'][:100]}...")
)
```

---

## Lower-Level Prompt Exports

The prompt package also exposes `PromptVariable` for describing individual template variables and `make_prompt_context()` for constructing `PromptContext` objects programmatically when you need prompt resolution outside the normal agent flow.

## What's Next?

- **[Memory Integration](/agent-kit/core/memory)** — How memory works with dynamic prompts
- **[Agent Teams](/agent-kit/multi-agent/overview)** — Using templates with multiple agents
- **[Production Patterns](/agent-kit/production/serving)** — Template patterns for production systems

## See Also

- [System Prompts](/agent-kit/core/prompts) — Writing effective base prompts
- [Creating Agents](/agent-kit/agent/creating-agents) — Putting templates into practice
- [Memory System](/agent-kit/core/memory) — Dynamic content from memory
