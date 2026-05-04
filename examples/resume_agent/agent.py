"""ResumeAgent — Syrin voice agent for recruiter calls.

Knowledge-backed conversational agent. Answers questions about resume, skills,
projects. Can schedule meetings.

Run standalone (text mode):
    python agent.py

Used by voice_server.py as the brain behind the Pipecat pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from syrin import Agent, tool
from syrin.embedding import Embedding
from syrin.enums import KnowledgeBackend
from syrin.knowledge import Knowledge
from syrin.model import Model

# ── Scheduling tools (Calendly integration) ──


@tool
def get_available_slots(date_range: str) -> str:
    """Check calendar availability for scheduling a meeting.
    Use when the recruiter wants to schedule a call or meeting.

    Example: get_available_slots("next week")
    """
    return (
        "I have availability. How about March 18, 2025 at 2pm, March 19 at 10am, "
        "or March 20 at 3pm? Which works for you? I'll send a calendar invite once confirmed."
    )


@tool
def book_appointment(date_time: str, recruiter_name: str, recruiter_email: str) -> str:
    """Book a meeting after the recruiter confirms a time.
    Use ONLY after they explicitly pick a slot.

    Example: book_appointment("Tuesday at 2pm", "Jane Smith", "jane@company.com")
    """
    return (
        f"Done! I've booked {date_time} for {recruiter_name}. "
        "Confirmed for March 13, 2025 at 2:00 PM. A calendar invite will be sent to the email provided."
    )


# ── Knowledge + Agent ──


def _data_path(*parts: str) -> Path:
    return Path(__file__).resolve().parent / "data" / Path(*parts)


def create_resume_agent() -> Agent:
    """Create the ResumeAgent with Knowledge and tools."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    db_url = os.getenv("DATABASE_URL", "").strip()

    if not api_key:
        raise ValueError("Set OPENAI_API_KEY in examples/.env")

    # Use MEMORY by default for simple setup. Set DATABASE_URL + uv sync --extra knowledge-postgres for Postgres.
    try:
        if db_url:
            import asyncpg  # noqa: F401

            backend = KnowledgeBackend.POSTGRES
            conn = db_url
        else:
            backend = KnowledgeBackend.MEMORY
            conn = None
    except ImportError:
        backend = KnowledgeBackend.MEMORY
        conn = None

    knowledge = Knowledge(
        sources=[
            Knowledge.Markdown(_data_path("about.md")),
            Knowledge.YAML(_data_path("skills.yaml")),
            Knowledge.Directory(_data_path("projects"), glob="*.md"),
        ],
        embedding=Embedding.OpenAI("text-embedding-3-small", api_key=api_key),
        backend=backend,
        connection_url=conn,
        chunk_size=400,
        top_k=5,
    )

    agent = Agent(
        model=Model.OpenAI("gpt-4o-mini", api_key=api_key),
        system_prompt=(
            "You are Divyanshu Shekhar's AI voice assistant. You represent Divyanshu on a phone call with a recruiter. "
            "Your job is to answer questions about Divyanshu—his background, Syrin (the company he founded), skills, projects, and experience. "
            "\n\n"
            "RULES:\n"
            "1. ALWAYS search the knowledge base before answering questions about skills or projects.\n"
            "2. Keep responses SHORT and CONVERSATIONAL (1-3 sentences). This is a phone call.\n"
            "3. No bullet points, markdown, or lists. Speak naturally.\n"
            "4. If you don't have info, say so and offer to schedule a direct call.\n"
            "5. When asked to schedule: check availability, offer options, confirm name/email, then book.\n"
            "6. Be warm and professional."
        ),
        knowledge=knowledge,
        tools=[get_available_slots, book_appointment],
    )
    return agent


# Singleton for voice server
_resume_agent: Agent | None = None


def get_resume_agent() -> Agent:
    """Get or create the ResumeAgent singleton."""
    global _resume_agent
    if _resume_agent is None:
        _resume_agent = create_resume_agent()
    return _resume_agent


async def _main() -> None:
    agent = create_resume_agent()
    print("ResumeAgent (text mode). Type 'quit' to exit.\n")
    while True:
        try:
            user = input("You: ").strip()
        except EOFError:
            break
        if not user or user.lower() == "quit":
            break
        response = await agent.arun(user)
        print(f"Agent: {response.content}\n")


if __name__ == "__main__":
    import asyncio

    asyncio.run(_main())
