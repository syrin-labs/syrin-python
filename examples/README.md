# Syrin examples

This directory contains example scripts showing how to use the Syrin library.

## Get started quickly — one notebook

**[`getting_started.ipynb`](getting_started.ipynb)** — A single Jupyter notebook with **all topics**: markdown docs, runnable code, and output. Uses Almock (no API key). Sections: Minimal → Tasks → Budget (with thresholds) → Memory → Tools → Loops → Multi-agent → Streaming → Guardrails → Observability (trace/debug) → Context → Checkpoints → Models → Prompts → Advanced, plus a **real-world** section (budget + threshold + `debug=True`). Share this file with anyone who wants to get started quickly.

## Example Categories

### 01_minimal/
Basic agent creation and simple responses.
- **hello_model.py** — Create a `Model` and inspect config (no API key).
- **hello_agent.py** — Agent with budget, prompt template (`@prompt`), and Response. Requires API key.
- **hello_memory.py** — Agent with `BufferMemory`; multi-turn conversation with context.

### 02_tasks/
Task-based execution patterns.
- **single_task.py** — Basic task execution.
- **multiple_tasks.py** — Running multiple tasks.
- **task_with_output_type.py** — Tasks with structured output.

### 03_budget/
Budget management and cost tracking.
- Examples showing budget limits, thresholds, and cost tracking.

### 04_memory/
Memory systems and conversation history.
- **basic_memory.py** — Buffer memory for multi-turn conversations.
- **async_memory.py** — Async memory operations.
- **memory_types_and_decay.py** — Different memory types with decay strategies.
- **chroma_memory.py**, **qdrant_memory.py**, **postgres_memory.py** — Vector store backends.

### 05_tools/
Tool definitions and execution.
- Tool creation, registration, and execution patterns.

### 06_loops/
Agent loop strategies (REACT, HITL, etc.).
- **react_loop.py** — REACT pattern execution.
- **human_in_the_loop.py** — Human approval workflows.

### 07_multi_agent/
Multi-agent patterns: handoff, spawn, pipeline, dynamic pipeline.
- **pipeline.py** — Sequential/parallel pipeline execution.
- **dynamic_pipeline_basic.py** — LLM-orchestrated multi-agent systems.
- **dynamic_pipeline_5agents.py** — Complex multi-agent orchestration.

### 08_streaming/
Streaming responses and async patterns.
- **stream_sync.py**, **stream_async.py** — Token-by-token output.

### 09_guardrails/
Content filtering, validation, and safety.
- See `09_guardrails/README.md` for details.

### 10_observability/
Tracing, debugging, and metrics.
- **comprehensive_tracing.py** — Full observability setup.
- **audit_logging.py** — Audit log recording.

### 11_context/
Context management, token limits, compaction.
- **context_management.py** — Context configuration and thresholds.
- **context_snapshot_demo.py** — Inspect context contents.
- **context_thresholds_compaction_demo.py** — Automatic compaction.
- **context_runtime_injection_demo.py** — RAG injection patterns.

### 11_mcp/
Model Context Protocol integration.
- **mcp_client.py** — MCP client usage.

### 12_checkpoints/
Agent state persistence and recovery.
- **long_running_agent.py** — Checkpointing for long sessions.

### 12_remote_config/
Remote configuration management.
- **init_and_serve.py** — Remote config server.

### 13_models/
Model configuration and providers.
- Model selection, configuration, and pricing.

### 14_prompts/
Prompt engineering and templates.
- System prompts, user prompts, and validation.

### 15_advanced/
Advanced patterns: dependency injection, circuit breaker, agent inheritance.
- **dependency_injection.py** — Passing dependencies to tools.
- **circuit_breaker.py** — Circuit breaker pattern for LLM failures.
- **agent_inheritance.py** — Extending Agent classes.
- **config.py** — Configuration management.
- **hitl_approval.py** — Human-in-the-loop approval.

### 16_serving/
HTTP serving and playground UI.
- **playground_dynamic_pipeline.py** — Serve multi-agent pipelines.
- See `16_serving/README.md` for details.

### 17_routing/
Model routing and cost optimization.
- **model_router.py** — Route to different models based on cost/quality.
- See `17_routing/README.md` for details.

### 18_multimodal/
Image, video, and voice generation.
- **image_generation.py** — Generate images with DALL-E.
- **voice_generation.py** — Generate speech with TTS providers.
- See `18_multimodal/README.md` for details.

### 19_knowledge/
Knowledge management and RAG (Retrieval-Augmented Generation).
- **loaders_and_document.py** — Document loading from files, URLs, GitHub.
- **chunking.py** — Text chunking strategies.
- **vector_store.py** — Vector storage backends.
- **agentic_rag.py** — Agentic RAG patterns.
- **full_rag_lifecycle.py** — Complete RAG pipeline.
- **knowledge_agent.py** — Agent with knowledge base.
- **postgres_backend.py** — PostgreSQL vector store.
- **serve_agentic_postgres.py** — Serve RAG agent.

### 20_template/
Template engine for slot-based generation.
- **template_standalone.py** — Template with slots, render(), slot_schema(), from_file/from_string
- **template_with_agent.py** — Agent with output_config.template; response.content = rendered, response.template_data = slot values
- **output_file_generation.py** — output_config format (TEXT/MARKDOWN/PDF/DOCX); citation parsing (`CitationConfig`); response.file, response.file_bytes, response.citations; save_as_pdf, save_as_docx

### ipo_drafting_agent/
Real-world example: DRHP Drafting Agent for Capital Structure & Shareholding Pattern section.
- **run.py** — Single command to draft a DRHP section from ROC filings (PAS-3, SH-7, MOA, etc.)
- Uses Knowledge (RAG), agentic retrieval, structured output, automation transparency
- Outputs: intermediate JSON, draft Markdown/DOCX, sources used, items requiring review

### resume_agent/
Real-world example: Voice AI agent for recruiter calls using Syrin + Pipecat.

## Running examples

From the project root (with `OPENAI_API_KEY` or other provider keys in `examples/.env`):

```bash
# Run a specific example
PYTHONPATH=. python examples/01_minimal/hello_agent.py

# Run a module
PYTHONPATH=. python -m examples.01_minimal.hello_agent
```

## Setup

1. From project root with virtualenv activated:
    ```bash
    source .venv/bin/activate
    uv pip install -e ".[dev,anthropic]"
    ```

2. Add a `.env` file in this directory (`examples/.env`) with your API key:
    ```
    ANTHROPIC_API_KEY=sk-ant-...
    ```
    Optional: `ANTHROPIC_MODEL_ID=anthropic/claude-3-7-sonnet-latest` (default; use any valid id from Anthropic's API).

Examples load `.env` via `python-dotenv`. Run: `python examples/01_minimal/hello_agent.py`

**CLI (transport):** `syrin serve -a examples.hello_agent:agent` (interactive REPL), `syrin cost`, `syrin version`, `syrin run <script.py>`.

## Remote config

- **12_remote_config/** — `syrin.init()`, config routes (GET/PATCH /config, /config/stream) when serving. See `12_remote_config/README.md`.
