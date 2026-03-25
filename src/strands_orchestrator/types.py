"""Shared data types for strands-orchestrator.

These Pydantic models define the canonical data structures used
throughout the orchestrator. Config sources produce these types,
and the factory/pool/hooks consume them.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MCPServerDefinition(BaseModel):
    """Configuration for connecting to an MCP server."""

    name: str
    transport: str = "sse"  # "sse", "stdio", "streamable-http"
    url: str | None = None  # For SSE/HTTP transports
    command: str | None = None  # For stdio transport
    args: list[str] = Field(default_factory=list)  # For stdio transport
    env: dict[str, str] = Field(default_factory=dict)  # Environment variables
    read_timeout_seconds: int = 120


class ModeDefinition(BaseModel):
    """Configuration for an agent mode.

    A mode controls which tools are visible to the agent.
    Servers map server names to tool allowlists ("*" means all tools).
    """

    name: str
    description: str = ""
    instructions: str = ""
    servers: dict[str, list[str]] = Field(default_factory=dict)
    # servers example: {"fusion": ["analyze_spectrum", "run_model"], "brave": ["*"]}


class AgentDefinition(BaseModel):
    """Configuration for a single agent."""

    name: str
    system_prompt: str = ""
    description: str = ""
    model: str = "sonnet"  # Model alias, resolved by ModelFactory
    max_iterations: int = 25
    max_tokens: int = 16384
    modes: list[ModeDefinition] = Field(default_factory=list)
    default_mode: str | None = None


class AgentState(BaseModel):
    """Serializable state of a single agent.

    Used for persisting and restoring agent conversation state
    between requests (via SessionPersistenceProtocol).
    """

    messages: list[dict] = Field(default_factory=list)
    state: dict = Field(default_factory=dict)
    current_mode: str | None = None
    usage: AgentUsage = Field(default_factory=lambda: AgentUsage())
    active_workflows: list[str] = Field(default_factory=list)


class AgentUsage(BaseModel):
    """Token usage and cost tracking for an agent."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    turn_input_tokens: int = 0
    turn_output_tokens: int = 0
    total_cost: float = 0.0
