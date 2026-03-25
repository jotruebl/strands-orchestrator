"""OrchestratorConfig — declarative configuration for strands-orchestrator.

Declares what services are available. The config is stored on each
AgentContainer at pool creation time. When prepare_for_request() is
called per-request, the container reads the config and registers
hooks automatically. Callers only provide request-scoped context
(chat_id, user, interrupt_event).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strands_orchestrator.protocols import (
        AgentConfigSourceProtocol,
        BackgroundTaskInboxProtocol,
        ConsentServiceProtocol,
        EventBusProtocol,
        SessionPersistenceProtocol,
        StreamEventFactoryProtocol,
    )


@dataclass
class OrchestratorConfig:
    """Configuration for the Strands orchestrator.

    Features are enabled by providing protocol implementations:
    - event_bus + event_factory → EventBridgeHook (SSE streaming)
    - background_inbox → InboxHook (background task injection)
    - consent_service → ConsentHook (tool consent gating)
    - session_persistence → automatic state save/restore

    Example:
        config = OrchestratorConfig(
            agent_config_source=MongoDBAgentConfigSource(db_name="mydb"),
            event_bus=my_event_bus,
            event_factory=MyStreamEventFactory(),
            pool_size=3,
            default_model="sonnet",
        )
    """

    # Required — where to load agent definitions from
    agent_config_source: AgentConfigSourceProtocol

    # Optional integrations — provide to enable features
    event_bus: EventBusProtocol | None = None
    event_factory: StreamEventFactoryProtocol | None = None
    background_inbox: BackgroundTaskInboxProtocol | None = None
    consent_service: ConsentServiceProtocol | None = None
    session_persistence: SessionPersistenceProtocol | None = None

    # MCP server configuration
    mcp_config_path: str | None = None
    mcp_servers: list[dict] | None = None

    # Agent pool settings
    pool_size: int = 1
    pool_max_burst: int = 20

    # Model defaults
    default_model: str = "sonnet"
    model_aliases: dict[str, str] = field(default_factory=dict)

    # Tool consent
    auto_approve_tools: set[str] = field(default_factory=set)

    # Behavioral flags
    enable_mode_filtering: bool = True
    enable_consent: bool = False
    enable_background_tasks: bool = False
    enable_interrupts: bool = True

    def validate(self) -> list[str]:
        """Validate configuration and return list of warnings."""
        warnings = []

        if self.enable_consent and self.consent_service is None:
            warnings.append(
                "enable_consent=True but no consent_service provided. "
                "Consent checking will be disabled."
            )

        if self.enable_background_tasks and self.background_inbox is None:
            warnings.append(
                "enable_background_tasks=True but no background_inbox provided. "
                "Background task integration will be disabled."
            )

        if self.event_bus is not None and self.event_factory is None:
            warnings.append(
                "event_bus provided without event_factory. "
                "Event bridge will not be able to create typed events."
            )

        return warnings
