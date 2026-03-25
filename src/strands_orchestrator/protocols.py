"""Protocol definitions (contracts) for the strands-orchestrator DI system.

Each protocol defines an interface that consumers implement for their specific
infrastructure. The orchestrator enables features based on which protocols
are provided in OrchestratorConfig.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EventBusProtocol(Protocol):
    """Contract for publishing SSE/streaming events to a frontend.

    Implementations typically manage per-conversation event queues
    and route events to connected SSE/WebSocket clients.
    """

    async def publish(self, event: Any, user: Any = None) -> None:
        """Publish an event to subscribers.

        Args:
            event: The event object to publish.
            user: Optional user context for role-based filtering.
        """
        ...


@runtime_checkable
class BackgroundTaskInboxProtocol(Protocol):
    """Contract for background task result delivery.

    Implementations must support:
    - Watching containers for completion (e.g., long-running compute jobs)
    - Watching LLM tasks for completion (e.g., subagent delegations)
    - Storing results in an inbox until the agent loop picks them up
    - Popping results from the inbox for injection into conversation
    """

    async def pop_inbox(self, conversation_id: str) -> list[dict]:
        """Get and clear all pending results for a conversation.

        Returns:
            List of result dicts. Empty list if no pending results.
        """
        ...

    async def watch_container(
        self,
        container_group_uuid: str,
        conversation_id: str,
        tool_name: str,
        tenant_id: int | None = None,
    ) -> None:
        """Register a container/job for completion tracking.

        When the container finishes, a result should be added
        to the conversation's inbox automatically.
        """
        ...

    async def watch_llm_task(
        self,
        task_id: str,
        conversation_id: str,
        tool_name: str,
        tenant_id: int | None = None,
        ensure_subscription: bool = True,
    ) -> None:
        """Register an LLM background task for completion tracking."""
        ...

    async def ensure_subscribed(
        self,
        conversation_id: str,
        timeout: float = 5.0,
    ) -> bool:
        """Ensure event subscriptions are established for a conversation.

        Returns:
            True if subscriptions confirmed, False if timeout.
        """
        ...


@runtime_checkable
class ConsentServiceProtocol(Protocol):
    """Contract for tool execution consent checking.

    Implementations gate tool execution behind user approval,
    typically via a UI prompt delivered through SSE events.
    """

    async def check_consent(self, tool_name: str, session_id: str) -> bool:
        """Check if a tool has been pre-approved for execution.

        Returns:
            True if tool is approved, False if consent needed.
        """
        ...

    async def request_consent(
        self,
        tool_name: str,
        tool_input: dict,
        session_id: str,
    ) -> bool:
        """Request user consent for a tool execution and wait for response.

        Returns:
            True if user approved, False if denied.
        """
        ...


@runtime_checkable
class SessionPersistenceProtocol(Protocol):
    """Contract for persisting agent state between requests.

    Implementations handle the storage backend (Redis, MongoDB,
    filesystem, etc.) for serialized agent state.
    """

    async def load_state(self, session_id: str) -> dict | None:
        """Load persisted agent state for a session.

        Returns:
            Dict of agent states keyed by agent name, or None if no state exists.
        """
        ...

    async def save_state(self, session_id: str, state: dict) -> None:
        """Save agent state for a session.

        Args:
            session_id: Unique session identifier.
            state: Dict of agent states keyed by agent name.
        """
        ...


@runtime_checkable
class UserContextProtocol(Protocol):
    """Contract for user identity and auth context.

    Provides the identity and authentication details needed
    for tool calls and billing.
    """

    @property
    def user_id(self) -> str:
        """Unique user identifier."""
        ...

    @property
    def tenant_id(self) -> int | None:
        """Tenant/organization identifier, if applicable."""
        ...

    @property
    def auth_token(self) -> str:
        """Authentication token for MCP tool calls."""
        ...


@runtime_checkable
class StreamEventFactoryProtocol(Protocol):
    """Contract for creating typed stream events.

    Implementations create event objects in whatever format
    the consumer's EventBus expects.
    """

    def create_tool_start_event(
        self,
        chat_id: str,
        tool_name: str,
        tool_input: dict,
    ) -> Any:
        """Create an event for when a tool call starts."""
        ...

    def create_tool_end_event(
        self,
        chat_id: str,
        tool_name: str,
        tool_result: Any,
    ) -> Any:
        """Create an event for when a tool call completes."""
        ...

    def create_turn_start_event(self, chat_id: str, agent_name: str) -> Any:
        """Create an event for when an agent turn begins."""
        ...

    def create_turn_end_event(self, chat_id: str, agent_name: str) -> Any:
        """Create an event for when an agent turn ends."""
        ...

    def create_error_event(self, chat_id: str, error: Exception) -> Any:
        """Create an event for an error during agent execution."""
        ...

    def create_background_task_completed_event(
        self,
        chat_id: str,
        task_id: str,
        tool_name: str,
        status: str,
        result: str | None,
    ) -> Any:
        """Create an event for background task completion."""
        ...


@runtime_checkable
class AgentConfigSourceProtocol(Protocol):
    """Contract for loading agent configurations.

    Could be MongoDB (kubernagents), YAML files, environment
    variables, or any other config source.
    """

    async def get_agent_configs(self) -> list[dict]:
        """Load all agent definitions.

        Returns:
            List of agent config dicts with keys: name, system_prompt,
            model, description, modes (list of mode names).
        """
        ...

    async def get_mode_configs(self, agent_name: str) -> dict[str, dict]:
        """Load mode configurations for a specific agent.

        Returns:
            Dict mapping mode_name -> mode config dict with keys:
            description, instructions, servers (dict of server_name -> tool list).
        """
        ...

    async def get_mcp_server_configs(self) -> list[dict]:
        """Load MCP server connection configurations.

        Returns:
            List of server config dicts with keys: name, transport,
            url (for SSE/HTTP), command/args (for stdio), env (optional).
        """
        ...
