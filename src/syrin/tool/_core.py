"""Tool decorator and schema generation for agent tools."""

from __future__ import annotations

import inspect
import re
import warnings
from collections.abc import Callable
from typing import Union, get_type_hints

from pydantic import BaseModel, Field, PrivateAttr

from syrin.enums import DocFormat
from syrin.tool._schema import schema_to_toon as _schema_to_toon
from syrin.tool._schema import tool_schema_to_format_dict

_TYPE_TO_JSON: dict[type[object], str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

# Section headers that end a Google-style docstring block we care about.
_SECTION_HEADERS: frozenset[str] = frozenset(
    {
        "args",
        "arguments",
        "parameters",
        "returns",
        "return",
        "raises",
        "raise",
        "note",
        "notes",
        "example",
        "examples",
        "yields",
        "yield",
        "todo",
        "attributes",
        "see also",
        "references",
        "warnings",
    }
)


def _parse_google_docstring(docstring: str) -> tuple[str, dict[str, str], str]:
    """Parse a Google-style docstring into its useful parts.

    Handles ``Args:``, ``Returns:`` (and common aliases), plus
    multi-line parameter descriptions.

    Args:
        docstring: The raw docstring to parse.

    Returns:
        A 3-tuple of ``(summary, param_descriptions, returns_description)``
        where *summary* is the first non-empty line, *param_descriptions*
        maps each parameter name to its description string, and
        *returns_description* is the text following the ``Returns:`` header.
    """
    if not docstring:
        return "", {}, ""

    lines = docstring.strip().splitlines()
    if not lines:
        return "", {}, ""

    summary = lines[0].strip()

    # ── Identify section boundaries ───────────────────────────────────────────
    in_section: str | None = None
    section_lines: dict[str, list[str]] = {}

    for line in lines[1:]:
        bare = line.strip()
        if bare.endswith(":"):
            key = bare[:-1].lower()
            if key in _SECTION_HEADERS:
                in_section = key
                section_lines.setdefault(in_section, [])
                continue
        if in_section is not None:
            section_lines[in_section].append(line)

    # ── Parse Args section ────────────────────────────────────────────────────
    args_lines: list[str] = (
        section_lines.get("args")
        or section_lines.get("arguments")
        or section_lines.get("parameters")
        or []
    )

    param_descs: dict[str, str] = {}
    current_param: str | None = None
    current_indent: int = 0

    for line in args_lines:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        bare = line.strip()

        # A new param line looks like: "    city: description" or "    city (str): desc"
        if ":" in bare:
            before_colon, _, after_colon = bare.partition(":")
            # Strip optional "(type)" annotation: "city (str)" → "city"
            name_candidate = re.sub(r"\s*\(.*?\)\s*$", "", before_colon).strip()
            if name_candidate.isidentifier():
                current_param = name_candidate
                current_indent = indent
                param_descs[current_param] = after_colon.strip()
                continue

        # Continuation line — indented more than its param header
        if current_param is not None and indent > current_indent:
            param_descs[current_param] = (param_descs[current_param] + " " + bare).strip()

    # ── Parse Returns section ─────────────────────────────────────────────────
    returns_lines: list[str] = section_lines.get("returns") or section_lines.get("return") or []
    returns_desc = " ".join(ln.strip() for ln in returns_lines if ln.strip())

    return summary, param_descs, returns_desc


class ToolSpec(BaseModel):  # type: ignore[explicit-any]
    """Spec for a tool the model can call. Usually built via @tool decorator or syrin.tool().

    Attributes:
        name: Tool name. Model uses this in tool_calls.name.
        description: Description for the model. Be specific about when and why to call.
        parameters_schema: JSON schema for parameters. Model uses this to generate args.
        func: Python function to run. Receives parsed arguments from the model.
        requires_approval: If True, block execution until human approval via ApprovalGate.
        inject_run_context: If True, first param is ctx: RunContext; agent injects at runtime.
        examples: Call-string examples shown to the LLM.
        depends_on: Names of other tools this tool is typically used after.
        returns: Description of the tool's return value shown to the LLM.
    """

    name: str = Field(..., description="Tool name (used in tool_calls.name)")
    description: str = Field(
        default="",
        description=(
            "Description for the model. The LLM uses this to decide when to call this tool—"
            "be specific about when and why. Example: 'Search web for current facts. "
            "Use when the user asks about news, events, or recent information.'"
        ),
    )
    parameters_schema: dict[str, object] = Field(
        default_factory=dict,
        description="JSON schema for parameters. Model uses this to generate args.",
    )
    func: Callable[..., object] = Field(  # type: ignore[explicit-any]
        ...,
        description="Python function to run. Receives parsed arguments from model.",
    )
    requires_approval: bool = Field(
        default=False,
        description="If True, block execution until human approval via ApprovalGate.",
    )
    inject_run_context: bool = Field(
        default=False,
        description="If True, first param is ctx: RunContext[Deps]; agent injects it at runtime.",
    )
    examples: list[str] = Field(
        default_factory=list,
        description=(
            "Usage examples shown to the LLM. Each is a short call string like "
            "'search(\"python tutorials\")' or 'calculate(a=5, b=3, operation=\"add\")'. "
            "Helps the LLM understand when and how to call this tool."
        ),
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description=(
            "Names of other tools this tool typically uses after. "
            "E.g. 'search' → 'summarise'. Helps the LLM understand tool chaining."
        ),
    )
    returns: str = Field(
        default="",
        description=(
            "Description of what this tool returns, shown to the LLM. "
            "E.g. 'JSON string with keys: title, url, snippet'. "
            "Helps the LLM interpret the tool output correctly."
        ),
    )

    model_config = {"arbitrary_types_allowed": True}

    _format_cache: dict[DocFormat, dict[str, object]] = PrivateAttr(default_factory=dict)

    def schema_to_toon(self, indent: int = 0) -> str:
        """Return this tool's parameters schema as TOON (token-efficient) string."""
        return _schema_to_toon(self.parameters_schema or {}, indent)

    def to_format(self, format: DocFormat = DocFormat.TOON) -> dict[str, object]:
        """Return this tool as a provider-ready schema dict (TOON, JSON, or YAML).

        P6: Result is cached per DocFormat so repeated LLM context builds don't
        re-serialize the same static schema on every call.

        Note: Cached results become stale if ``examples``, ``depends_on``, or
        ``returns`` change after construction — that is intentional since
        ToolSpec is constructed once.
        """
        if format not in self._format_cache:
            # Build enriched description: base + returns + examples + depends_on
            enriched_desc = self.description or ""
            if self.returns:
                enriched_desc += "\nReturns: " + self.returns
            if self.examples:
                enriched_desc += "\nExamples: " + " | ".join(self.examples)
            if self.depends_on:
                enriched_desc += "\nOften followed by: " + ", ".join(self.depends_on)
            self._format_cache[format] = tool_schema_to_format_dict(
                self.name,
                enriched_desc,
                self.parameters_schema or {},
                format,
            )
        return self._format_cache[format]


def _annotation_to_json_schema(annotation: type[object] | object) -> dict[str, object]:
    """Convert a type annotation to a JSON schema fragment."""
    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", ())

    if annotation is type(None):
        return {"type": "null"}
    if origin is Union or (hasattr(annotation, "__args__") and type(None) in (args or ())):
        non_none = [a for a in (args or ()) if a is not type(None)]
        if len(non_none) == 1:
            return {"oneOf": [_annotation_to_json_schema(non_none[0]), {"type": "null"}]}
        if non_none:
            return {"oneOf": [_annotation_to_json_schema(a) for a in non_none]}
    if origin is list and args:
        return {"type": "array", "items": _annotation_to_json_schema(args[0])}
    if origin is dict and len(args) >= 2:
        value_ann = args[1]  # type: ignore[misc]  # safe after len check; mypy tuple narrow
        if value_ann is not type(None):
            return {
                "type": "object",
                "additionalProperties": _annotation_to_json_schema(value_ann),
            }
    if isinstance(annotation, type) and annotation in _TYPE_TO_JSON:
        return {"type": _TYPE_TO_JSON[annotation]}
    return {"type": "string"}


def _parameters_schema_from_function(  # type: ignore[explicit-any]
    func: Callable[..., object],
    param_descriptions: dict[str, str] | None = None,
) -> tuple[dict[str, object], bool]:
    """Build a JSON schema for the function's parameters from type hints.

    If *param_descriptions* is provided, each parameter entry in the schema
    gets a ``"description"`` field, which the LLM uses to understand what
    value to pass.

    Excludes the parameter named ``ctx`` (RunContext for DI).

    Args:
        func: The function to inspect.
        param_descriptions: Optional mapping of parameter name → description
            string. These are injected into the JSON schema ``properties`` as
            ``"description"`` fields.

    Returns:
        A 2-tuple of ``(schema_dict, inject_run_context)`` where
        *inject_run_context* is True when the function has a ``ctx`` parameter.
    """
    hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
    sig = inspect.signature(func)
    properties: dict[str, object] = {}
    required: list[str] = []
    inject_run_context = False
    _descs = param_descriptions or {}

    for name, param in sig.parameters.items():
        if name == "self" or name == "cls":
            continue
        if name == "ctx":
            inject_run_context = True
            continue
        ann = hints.get(name, param.annotation)
        if ann is inspect.Parameter.empty:
            ann = object
        prop: dict[str, object] = _annotation_to_json_schema(ann)
        if name in _descs:
            prop = {**prop, "description": _descs[name]}
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema: dict[str, object] = {"type": "object", "properties": properties, "required": required}
    return (schema, inject_run_context)


def tool(  # type: ignore[explicit-any]
    func: Callable[..., object] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    param_descriptions: dict[str, str] | None = None,
    returns: str | None = None,
    examples: list[str] | None = None,
    depends_on: list[str | object] | None = None,
    requires_approval: bool = False,
) -> Callable[..., object] | ToolSpec:
    """Decorator to register a Python function as a Syrin tool.

    The LLM sees: the tool name, its description, each parameter's type and
    description, and (optionally) example call strings and a description of
    what it returns.

    By default, everything is inferred from the function automatically:

    - **name** → function name
    - **description** → first line of the docstring
    - **parameter descriptions** → ``Args:`` section of the docstring
    - **returns description** → ``Returns:`` section of the docstring

    Use the decorator keyword arguments to override any of these explicitly
    (no docstring required).

    Args:
        func: Function to decorate. Pass ``None`` when using keyword args
            (``@tool(name="...")``).
        name: Override the tool name shown to the LLM. Default: function name.
        description: Override the main description. Default: first docstring line.
        param_descriptions: Per-parameter descriptions injected into the JSON
            schema so the LLM knows what each argument means. Overrides the
            ``Args:`` section of the docstring. Example::

                param_descriptions={
                    "query": "The search query string",
                    "max_results": "Maximum number of results (default 5)",
                }

        returns: Description of what the tool returns. Shown to the LLM so it
            can interpret the result correctly. Overrides the ``Returns:``
            section of the docstring. Example::

                returns="JSON string with keys: title, url, snippet"

        examples: Call-string examples shown to the LLM. Provide 1–3
            representative patterns. Example::

                examples=[
                    "search('latest AI news')",
                    "search('Python tutorials', max_results=3)",
                ]

        depends_on: Tool names (or ToolSpec objects) that this tool is
            typically called after. Signals tool chaining to the LLM. Example::

                depends_on=[search_tool]   # or depends_on=["search"]

        requires_approval: If ``True``, block execution until a human approves
            the call via the HITL approval gate.

    Returns:
        A :class:`ToolSpec` when used as ``@tool`` or ``@tool()``;
        a callable decorator when used with keyword arguments.

    Examples:
        Minimal — everything inferred from the docstring::

            @tool
            def get_weather(city: str) -> str:
                \"\"\"Get current weather for a city.

                Args:
                    city: The city name, e.g. 'Tokyo' or 'London'

                Returns:
                    A human-readable weather summary string.
                \"\"\"
                ...

        Fully explicit — no docstring needed::

            @tool(
                description="Get current weather for a city.",
                param_descriptions={"city": "City name, e.g. 'Tokyo'"},
                returns="Human-readable weather summary string.",
                examples=["get_weather('Tokyo')", "get_weather('London')"],
            )
            def get_weather(city: str) -> str:
                ...
    """

    def decorator(f: Callable[..., object]) -> ToolSpec:  # type: ignore[explicit-any]
        tool_name = name or f.__name__

        # Parse docstring once; explicit kwargs win over parsed values.
        raw_doc = inspect.getdoc(f) or ""
        doc_summary, doc_param_descs, doc_returns = _parse_google_docstring(raw_doc)

        final_desc = description or doc_summary or ""
        final_param_descs = (
            param_descriptions if param_descriptions is not None else doc_param_descs
        )
        final_returns = returns if returns is not None else doc_returns

        # Warn if the tool has no description — the LLM needs this to decide when to call it.
        if not final_desc:
            warnings.warn(
                f"@tool '{tool_name}' has no description. "
                "The LLM uses the description to decide when to call the tool. "
                "Add a docstring or pass description= to @tool(description=...).",
                UserWarning,
                stacklevel=3,
            )

        params_schema, inject_run_context = _parameters_schema_from_function(
            f, param_descriptions=final_param_descs or None
        )

        _depends: list[str] = []
        for d in depends_on or []:
            if hasattr(d, "name"):
                _depends.append(d.name)

        return ToolSpec(
            name=tool_name,
            description=final_desc,
            parameters_schema=params_schema,
            func=f,
            requires_approval=requires_approval,
            inject_run_context=inject_run_context,
            examples=examples or [],
            depends_on=_depends,
            returns=final_returns,
        )

    if func is not None:
        return decorator(func)
    return decorator


__all__ = ["tool", "ToolSpec"]
