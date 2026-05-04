"""Tests for @syrin.tool decorator (tool.py)."""

from __future__ import annotations

import pytest

from syrin.tool import ToolSpec, tool


def test_tool_decorator_without_args() -> None:
    @tool
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"weather in {city}"

    assert isinstance(get_weather, ToolSpec)
    assert get_weather.name == "get_weather"
    assert "city" in get_weather.parameters_schema.get("properties", {})
    assert get_weather.parameters_schema.get("required") == ["city"]
    assert get_weather.func is not None
    assert get_weather.func("Paris") == "weather in Paris"


def test_tool_decorator_with_name_and_description() -> None:
    @tool(name="fetch_weather", description="Fetch current weather by city")
    def get_weather(city: str) -> str:  # noqa: ARG001
        return "sunny"

    assert get_weather.name == "fetch_weather"
    assert get_weather.description == "Fetch current weather by city"


def test_tool_schema_types() -> None:
    @tool
    def mixed(a: str, b: int, c: float, d: bool) -> None:
        """Mixed params."""
        pass

    props = mixed.parameters_schema.get("properties", {})
    assert props.get("a") == {"type": "string"}
    assert props.get("b") == {"type": "integer"}
    assert props.get("c") == {"type": "number"}
    assert props.get("d") == {"type": "boolean"}
    assert set(mixed.parameters_schema.get("required", [])) == {"a", "b", "c", "d"}


def test_tool_optional_param() -> None:
    @tool
    def with_optional(x: str, y: int | None = None) -> str:  # noqa: ARG001
        """Optional y."""
        return x

    props = with_optional.parameters_schema.get("properties", {})
    assert "x" in props
    assert "y" in props
    assert with_optional.parameters_schema.get("required") == ["x"]


# =============================================================================
# TOOL EDGE CASES - TRY TO BREAK FUNCTIONALITY
# =============================================================================


def test_tool_with_no_parameters() -> None:
    """Tool with no parameters."""

    @tool
    def no_params() -> str:
        """No parameters."""
        return "done"

    assert no_params.name == "no_params"
    assert no_params.func() == "done"
    assert no_params.parameters_schema.get("required") == []


def test_tool_with_many_parameters() -> None:
    """Tool with many parameters."""

    @tool
    def many_params(a: str, b: str, c: str, d: str, e: str) -> str:
        """Many params."""
        return f"{a}{b}{c}{d}{e}"

    props = many_params.parameters_schema.get("properties", {})
    assert len(props) == 5
    assert many_params.func("1", "2", "3", "4", "5") == "12345"


def test_tool_with_default_values() -> None:
    """Tool with default values."""

    @tool
    def with_defaults(a: str, b: int = 10, c: float = 1.5) -> str:
        """Defaults."""
        return f"{a}-{b}-{c}"

    assert with_defaults.func("x") == "x-10-1.5"
    assert with_defaults.func("x", 20) == "x-20-1.5"
    assert with_defaults.func("x", 20, 3.0) == "x-20-3.0"


def test_tool_with_list_type() -> None:
    """Tool with list type."""

    @tool
    def list_param(items: list[str]) -> int:
        """List param."""
        return len(items)

    props = list_param.parameters_schema.get("properties", {})
    assert props.get("items", {}).get("type") == "array"


def test_tool_with_dict_type() -> None:
    """Tool with dict type."""

    @tool
    def dict_param(data: dict) -> str:
        """Dict param."""
        return str(data)

    props = dict_param.parameters_schema.get("properties", {})
    assert props.get("data", {}).get("type") == "object"


def test_tool_returns_none() -> None:
    """Tool that returns None."""

    @tool
    def returns_none(_x: str) -> None:
        """Returns None."""
        return None

    assert returns_none.func("test") is None


def test_tool_with_complex_return() -> None:
    """Tool that returns complex data."""

    @tool
    def complex_return() -> dict:
        """Returns complex."""
        return {"key": "value", "nested": {"a": 1}}

    result = complex_return.func()
    assert result["key"] == "value"
    assert result["nested"]["a"] == 1


def test_tool_preserves_function_metadata() -> None:
    """Tool preserves function metadata."""

    @tool
    def documented() -> str:
        """This is the docstring."""
        return "result"

    assert "docstring" in documented.description.lower() or documented.description != ""


def test_tool_with_special_characters_in_name() -> None:
    """Tool with underscores and numbers."""

    @tool
    def tool_123_test(x: int) -> int:
        """Test tool."""
        return x * 2

    assert tool_123_test.name == "tool_123_test"
    assert tool_123_test.func(5) == 10


def test_tool_execution_error_handling() -> None:
    """Tool that raises error during execution."""

    @tool
    def failing_tool() -> str:
        raise ValueError("Intentional error")

    # Tool execution should raise
    with pytest.raises(ValueError):
        failing_tool.func()


def test_tool_with_union_types() -> None:
    """Tool with union types."""

    @tool
    def union_param(x: int | str) -> str:
        """Union type."""
        return str(x)

    result = union_param.func(5)
    assert result == "5"
    result = union_param.func("hello")
    assert result == "hello"


# =============================================================================
# Docstring parsing — Args: and Returns:
# =============================================================================


def test_tool_parses_args_section_from_docstring() -> None:
    """@tool reads Args: section and injects descriptions into JSON schema."""
    from syrin.enums import DocFormat

    @tool
    def search(query: str, max_results: int = 5) -> str:
        """Search the web.

        Args:
            query: The search query string
            max_results: Maximum number of results to return
        """
        return ""

    schema = search.to_format(DocFormat.JSON)["function"]  # type: ignore[index]
    props = schema["parameters"]["properties"]  # type: ignore[index]
    assert props["query"].get("description") == "The search query string"  # type: ignore[union-attr]
    assert props["max_results"].get("description") == "Maximum number of results to return"  # type: ignore[union-attr]


def test_tool_parses_returns_section_from_docstring() -> None:
    """@tool reads Returns: section and injects it into the description."""
    from syrin.enums import DocFormat

    @tool
    def lookup(key: str) -> str:
        """Look up a value.

        Returns:
            The value associated with the key.
        """
        return ""

    schema = lookup.to_format(DocFormat.JSON)["function"]  # type: ignore[index]
    assert "The value associated with the key." in schema.get("description", "")  # type: ignore[operator]


def test_tool_param_descriptions_override_docstring() -> None:
    """Explicit param_descriptions= wins over the Args: docstring section."""
    from syrin.enums import DocFormat

    @tool(param_descriptions={"city": "Explicit city description"})
    def get_weather(city: str) -> str:
        """Get weather.

        Args:
            city: Docstring city description
        """
        return ""

    schema = get_weather.to_format(DocFormat.JSON)["function"]  # type: ignore[index]
    props = schema["parameters"]["properties"]  # type: ignore[index]
    assert props["city"].get("description") == "Explicit city description"  # type: ignore[union-attr]


def test_tool_returns_override_docstring() -> None:
    """Explicit returns= wins over the Returns: docstring section."""
    from syrin.enums import DocFormat

    @tool(returns="Explicit return description")
    def do_thing() -> str:
        """Do a thing.

        Returns:
            Docstring return description.
        """
        return ""

    schema = do_thing.to_format(DocFormat.JSON)["function"]  # type: ignore[index]
    desc = schema.get("description", "")
    assert "Explicit return description" in desc  # type: ignore[operator]
    assert "Docstring return description" not in desc  # type: ignore[operator]


def test_tool_no_args_section_has_no_param_descriptions() -> None:
    """@tool without Args: section produces schema with no descriptions."""

    @tool
    def plain(x: int) -> str:
        """A plain tool."""
        return str(x)

    props = plain.parameters_schema.get("properties", {})
    assert "description" not in props.get("x", {})  # type: ignore[call-overload]


def test_tool_multiline_param_description_parsed_correctly() -> None:
    """Multi-line param descriptions in docstrings are joined."""

    @tool
    def search(query: str) -> str:
        """Search.

        Args:
            query: The search query. Can be a simple keyword
                or a full natural language question.
        """
        return ""

    props = search.parameters_schema.get("properties", {})
    desc = props.get("query", {}).get("description", "")  # type: ignore[union-attr]
    assert "keyword" in desc
    assert "natural language question" in desc


# =============================================================================
# @tool(examples=..., depends_on=...) params
# =============================================================================


def test_tool_examples_stored_on_spec() -> None:
    """@tool(examples=[...]) stores examples on ToolSpec."""

    @tool(examples=["search('cats')", "search('dogs')"])
    def search(query: str) -> str:
        return f"results for {query}"

    assert search.examples == ["search('cats')", "search('dogs')"]


def test_tool_depends_on_stored_on_spec() -> None:
    """@tool(depends_on=[...]) stores depends_on on ToolSpec."""

    @tool
    def read_file(path: str) -> str:
        """Read a file."""
        return ""

    @tool(depends_on=[read_file])
    def write_file(path: str, content: str) -> str:
        """Write a file."""
        return "ok"

    assert write_file.depends_on == ["read_file"]


def test_tool_examples_appear_in_toon_description() -> None:
    """Examples are injected into the TOON-format description."""
    from syrin.enums import DocFormat

    @tool(examples=["add(1, 2)"])
    def add(x: int, y: int) -> int:
        return x + y

    schema = add.to_format(DocFormat.TOON)
    # TOON schema nests content under 'function'
    description = schema.get("function", {}).get("description", "")  # type: ignore[union-attr]
    assert "add(1, 2)" in description


def test_tool_depends_on_appears_in_toon_description() -> None:
    """depends_on is injected into the TOON-format description."""
    from syrin.enums import DocFormat

    @tool
    def open_file(path: str) -> str:
        """Open a file."""
        return ""

    @tool
    def read_file(path: str) -> str:
        """Read a file."""
        return ""

    @tool(depends_on=[open_file, read_file])
    def close_file(handle: str) -> str:
        """Close a file."""
        return "closed"

    schema = close_file.to_format(DocFormat.TOON)
    description = schema.get("function", {}).get("description", "")  # type: ignore[union-attr]
    assert "open_file" in description
    assert "read_file" in description


def test_tool_no_examples_or_depends_on_defaults_to_empty() -> None:
    """ToolSpec.examples and .depends_on default to empty lists."""

    @tool
    def simple() -> str:
        return "ok"

    assert simple.examples == []
    assert simple.depends_on == []


def test_tool_returns_stored_on_spec() -> None:
    """@tool(returns=...) stores returns on ToolSpec."""

    @tool(returns="The computed result as a string.")
    def compute(x: int) -> str:
        return str(x)

    assert compute.returns == "The computed result as a string."


def test_tool_returns_appears_in_toon_description() -> None:
    """returns= is injected into the TOON-format description."""
    from syrin.enums import DocFormat

    @tool(returns="JSON with keys: title, url")
    def fetch(url: str) -> str:
        return ""

    schema = fetch.to_format(DocFormat.TOON)
    description = schema.get("function", {}).get("description", "")  # type: ignore[union-attr]
    assert "JSON with keys: title, url" in description


# =============================================================================
# ToolErrorMode integration
# =============================================================================


def test_tool_error_mode_propagate_reraises_original() -> None:
    """ToolErrorMode.PROPAGATE (default) re-raises the original exception."""
    from syrin import Agent, Model
    from syrin.enums import ToolErrorMode

    @tool
    def explode() -> str:
        raise ValueError("boom")

    agent = Agent(
        model=Model.mock(),
        tools=[explode],
        tool_error_mode=ToolErrorMode.PROPAGATE,
    )
    with pytest.raises(ValueError, match="boom"):
        agent._execute_tool("explode", {})


def test_tool_error_mode_stop_wraps_in_tool_execution_error() -> None:
    """ToolErrorMode.STOP wraps the error in ToolExecutionError."""
    from syrin import Agent, Model
    from syrin.enums import ToolErrorMode
    from syrin.exceptions import ToolExecutionError

    @tool
    def explode() -> str:
        raise ValueError("original error")

    agent = Agent(
        model=Model.mock(),
        tools=[explode],
        tool_error_mode=ToolErrorMode.STOP,
    )
    with pytest.raises(ToolExecutionError, match="original error"):
        agent._execute_tool("explode", {})


def test_tool_error_mode_return_as_string_returns_error_message() -> None:
    """ToolErrorMode.RETURN_AS_STRING returns error as a string instead of raising."""
    from syrin import Agent, Model
    from syrin.enums import ToolErrorMode

    @tool
    def explode() -> str:
        raise ValueError("string error")

    agent = Agent(
        model=Model.mock(),
        tools=[explode],
        tool_error_mode=ToolErrorMode.RETURN_AS_STRING,
    )
    result = agent._execute_tool("explode", {})
    assert isinstance(result, str)
    assert "string error" in result
