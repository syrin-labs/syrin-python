"""Trace replay tests: replay_trace() API and trace saving via Pry."""

from __future__ import annotations

import json


class TestReplayTraceAPI:
    def test_replay_trace_importable(self) -> None:
        """syrin.replay_trace is importable."""
        import syrin

        assert hasattr(syrin, "replay_trace")
        assert callable(syrin.replay_trace)

    def test_replay_trace_from_syrin_direct(self) -> None:
        """replay_trace can be imported directly."""
        from syrin import replay_trace

        assert callable(replay_trace)

    def test_replay_trace_with_empty_trace(self, tmp_path) -> None:
        """replay_trace() with empty trace file returns empty list."""
        from syrin import replay_trace

        trace_file = tmp_path / "trace.jsonl"
        trace_file.write_text("")  # Empty file

        result = replay_trace(str(trace_file))
        assert isinstance(result, list)
        assert len(result) == 0

    def test_replay_trace_with_trace_file(self, tmp_path) -> None:
        """replay_trace() reads JSONL trace and returns list of events."""
        from syrin import replay_trace

        events = [
            {"event": "agent.run.start", "input": "Hello", "ts": "2026-01-01T00:00:00Z"},
            {
                "event": "llm.request.start",
                "model": "claude-sonnet-4-6",
                "ts": "2026-01-01T00:00:01Z",
            },
            {"event": "llm.request.end", "tokens": 42, "ts": "2026-01-01T00:00:02Z"},
            {"event": "agent.run.end", "text": "Hi there!", "ts": "2026-01-01T00:00:02Z"},
        ]

        trace_file = tmp_path / "trace.jsonl"
        trace_file.write_text("\n".join(json.dumps(e) for e in events))

        result = replay_trace(str(trace_file))
        assert len(result) == 4
        assert result[0]["event"] == "agent.run.start"
        assert result[-1]["event"] == "agent.run.end"

    def test_replay_trace_from_step(self, tmp_path) -> None:
        """replay_trace(from_step=N) returns events starting at step N."""
        from syrin import replay_trace

        events = [
            {"event": "agent.run.start", "step": 0},
            {"event": "llm.request.start", "step": 1},
            {"event": "llm.request.end", "step": 2},
            {"event": "agent.run.end", "step": 3},
        ]

        trace_file = tmp_path / "trace.jsonl"
        trace_file.write_text("\n".join(json.dumps(e) for e in events))

        result = replay_trace(str(trace_file), from_step=2)
        assert len(result) == 2
        assert result[0]["event"] == "llm.request.end"

    def test_replay_trace_skips_invalid_lines(self, tmp_path) -> None:
        """replay_trace() skips non-JSON lines gracefully."""
        from syrin import replay_trace

        content = "\n".join(
            [
                '{"event": "agent.run.start"}',
                "NOT JSON",
                '{"event": "agent.run.end"}',
                "",
            ]
        )

        trace_file = tmp_path / "trace.jsonl"
        trace_file.write_text(content)

        result = replay_trace(str(trace_file))
        assert len(result) == 2
        assert result[0]["event"] == "agent.run.start"
        assert result[1]["event"] == "agent.run.end"

    def test_replay_trace_nonexistent_file_raises(self) -> None:
        """replay_trace() raises FileNotFoundError for missing file."""
        from syrin import replay_trace

        try:
            replay_trace("/nonexistent/path/trace.jsonl")
            raise AssertionError("Expected FileNotFoundError")
        except FileNotFoundError:
            pass


class TestSaveTrace:
    def test_save_trace_via_debug_ui(self, tmp_path) -> None:
        """Pry can emit JSON lines to a file (trace saving)."""
        from syrin.debug import Pry
        from syrin.enums import Hook
        from syrin.events import EventContext

        captured: list[str] = []
        ui = Pry(json_fallback=True, stream_override=captured)

        ui._handle_event(str(Hook.AGENT_RUN_START), EventContext(input="test"))
        ui._handle_event(str(Hook.AGENT_RUN_END), EventContext())

        assert len(captured) == 2
        for line in captured:
            data = json.loads(line)
            assert "event" in data

    def test_trace_format_is_valid_jsonl(self, tmp_path) -> None:
        """Each captured trace line is valid JSON with 'event' key."""
        from syrin.debug import Pry
        from syrin.enums import Hook
        from syrin.events import EventContext

        captured: list[str] = []
        ui = Pry(json_fallback=True, stream_override=captured)

        hooks = [
            Hook.AGENT_RUN_START,
            Hook.LLM_REQUEST_START,
            Hook.LLM_REQUEST_END,
            Hook.AGENT_RUN_END,
        ]
        for hook in hooks:
            ui._handle_event(str(hook), EventContext())

        assert len(captured) == 4
        for line in captured:
            data = json.loads(line)
            assert isinstance(data.get("event"), str)
