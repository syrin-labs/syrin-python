"""Data models for remote config: schema, overrides, and registration handshake.

Used by schema extraction, registry, resolver, and transports. All models are
Pydantic BaseModel for validation and JSON serialization.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FieldSchema(BaseModel):  # type: ignore[explicit-any]
    """One configurable field: name, dotted path, type, default, constraints, enum values, nested children.

    Used by schema extraction and by the dashboard to render and validate overrides.
    Fields that are callables or internal are marked remote_excluded (read-only in dashboard).

    Attributes:
        name: Field name (e.g. "max_cost", "top_k").
        path: Dotted path for overrides (e.g. "budget.max_cost", "memory.decay.rate").
        type: Type name as string for UI/validation: "float", "str", "int", "bool", "object", etc.
        default: Default value; None if not set. JSON-serializable.
        description: Optional human-readable description.
        constraints: Validation constraints: ge, le, gt, lt, pattern, min_length, max_length, etc.
        enum_values: For StrEnum fields, list of allowed string values.
        children: Nested fields when this field is an object (e.g. budget.rate_limits → hour, day).
        remote_excluded: If True, field is not writable via remote overrides (e.g. callables).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, description="Field name")
    path: str = Field(
        ..., min_length=1, description="Dotted path for overrides (e.g. budget.max_cost)"
    )
    type: str = Field(..., description="Type name: float, str, int, bool, object, etc.")
    default: object | None = Field(default=None, description="Default value; JSON-serializable")
    description: str | None = Field(default=None, description="Human-readable description")
    constraints: dict[str, float | int | str] = Field(
        default_factory=dict,
        description="Validation constraints: ge, le, gt, lt, pattern, min_length, max_length",
    )
    enum_values: list[str] | None = Field(
        default=None,
        description="For StrEnum fields, allowed string values",
    )
    children: list[FieldSchema] | None = Field(
        default=None,
        description="Nested fields when this field is an object",
    )
    remote_excluded: bool = Field(
        default=False,
        description="If True, not writable via remote overrides (e.g. callables)",
    )
    # Populated in GET /config response for dashboard UX (per-field baseline, current, overridden)
    baseline_value: object | None = Field(default=None, description="Value from code (baseline)")
    current_value: object | None = Field(
        default=None, description="Effective value (baseline + overrides)"
    )
    overridden: bool = Field(default=False, description="True if this path has a remote override")


class ConfigSchema(BaseModel):  # type: ignore[explicit-any]
    """All fields for one config object (e.g. Budget, Memory).

    Attributes:
        section: Section key used in API and paths (e.g. "budget", "memory").
        class_name: Python type name for debugging (e.g. "Budget", "Memory").
        fields: List of field schemas for this section.
    """

    model_config = ConfigDict(extra="forbid")

    section: str = Field(..., min_length=1, description="Section key (e.g. budget, memory)")
    class_name: str = Field(..., description="Python type name (e.g. Budget, Memory)")
    fields: list[FieldSchema] = Field(
        default_factory=list,
        description="Field schemas for this config section",
    )


class AgentSchema(BaseModel):  # type: ignore[explicit-any]
    """Full schema for a registered agent: sections, baseline (code) values, overrides, and current values.

    Sent to the backend on registration. GET /config returns this with baseline_values (frozen at
    first read), overrides (path → value for user-applied changes), and current_values = baseline + overrides.
    Revert = remove path from overrides; current falls back to baseline.

    Attributes:
        agent_id: Unique agent identifier (e.g. "my_agent:MyAgent").
        agent_name: Human-readable name (e.g. "my_agent").
        class_name: Python class name (e.g. "MyAgent").
        sections: Map of section key to ConfigSchema (fields may include baseline_value, current_value, overridden in response).
        baseline_values: Values from code (frozen at first GET). Used for revert.
        overrides: User-applied overrides (path → value). Only overridden paths; revert = remove path.
        current_values: Effective values (baseline + overrides). Same as baseline until overrides applied.
    """

    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(..., min_length=1, description="Unique agent identifier")
    agent_name: str = Field(..., description="Human-readable agent name")
    class_name: str = Field(..., description="Python class name")
    sections: dict[str, ConfigSchema] = Field(
        default_factory=dict,
        description="Map of section key to config schema",
    )
    baseline_values: dict[str, object] = Field(
        default_factory=dict,
        description="Values from code (frozen at first GET); used for revert",
    )
    overrides: dict[str, object] = Field(
        default_factory=dict,
        description="User-applied overrides (path → value); revert = remove path",
    )
    current_values: dict[str, object] = Field(
        default_factory=dict,
        description="Effective values (baseline + overrides)",
    )


class ConfigOverride(BaseModel):  # type: ignore[explicit-any]
    """Single override: path and value.

    Applied by ConfigResolver to the live agent. Value is validated against FieldSchema.
    Use value=null to revert that path to baseline (remove from override store).
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(..., min_length=1, description="Dotted path (e.g. budget.max_cost)")
    value: object | None = Field(
        ...,
        description="New value; type must match schema. Use null to revert path to baseline.",
    )


class OverridePayload(BaseModel):  # type: ignore[explicit-any]
    """List of overrides from backend, with monotonic version.

    Used by SSE, polling, and PATCH responses. version is used for ordering and since_version in polling.
    extra="ignore" so that extra backend fields (e.g. ok) are silently discarded.
    """

    model_config = ConfigDict(extra="ignore")

    agent_id: str = Field(..., min_length=1, description="Target agent identifier")
    version: int = Field(..., ge=0, description="Monotonic version number")
    overrides: list[ConfigOverride] = Field(
        default_factory=list,
        description="List of path/value overrides to apply",
    )


class SyncRequest(BaseModel):  # type: ignore[explicit-any]
    """Registration handshake request: agent sends schema and library version to backend."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    agent_id: str = Field(..., min_length=1, description="Agent identifier")
    agent_schema: AgentSchema = Field(
        ...,
        description="Full agent schema",
        validation_alias="schema",
        serialization_alias="schema",
    )
    library_version: str = Field(..., description="Syrin library version (e.g. 0.6.0)")


class SyncResponse(BaseModel):  # type: ignore[explicit-any]
    """Registration handshake response: ok, optional initial overrides, optional error.

    When ok is True, initial_overrides may contain overrides to apply immediately.
    When ok is False (e.g. backend down), agent continues with local config.
    extra="ignore" so that extra backend fields (e.g. agentId, configDelta) are silently discarded.
    """

    model_config = ConfigDict(extra="ignore")

    ok: bool = Field(..., description="True if registration succeeded")
    initial_overrides: list[ConfigOverride] | None = Field(
        default=None,
        description="Overrides to apply immediately after registration",
    )
    error: str | None = Field(
        default=None,
        description="Error message when ok is False",
    )
