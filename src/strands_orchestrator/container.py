"""AgentContainer — Dict-like wrapper around multiple Strands agents.

Provides attribute and dictionary access to agents by name,
plus lifecycle methods for state management and hook registration.

Note: Strands Agent.state is a JSONSerializableDict with a limited API:
  - state.set(key, value)  — set a value
  - state.get(key)         — get a value (returns None if missing)
  - state.get()            — get full dict copy
  - state.delete(key)      — delete a key
Standard dict operations (__setitem__, __getitem__, .keys(), .clear(), etc.)
are NOT supported.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, TYPE_CHECKING

from strands import Agent
from strands.handlers.callback_handler import null_callback_handler

from strands_orchestrator.hooks import (
    AuthTokenInjectorHook,
    ConsentHook,
    EventBridgeHook,
    InterruptHook,
)
from strands_orchestrator.mode_manager import ModeManager
from strands_orchestrator.protocols import UserContextProtocol

if TYPE_CHECKING:
    from strands_orchestrator.config import OrchestratorConfig

logger = logging.getLogger(__name__)


def _state_get(agent: Agent, key: str, default: Any = None) -> Any:
    """Safe get from agent.state with default value support.

    Strands JSONSerializableDict.get() doesn't accept a default parameter.
    """
    val = agent.state.get(key)
    return val if val is not None else default


def _state_set(agent: Agent, key: str, value: Any) -> None:
    """Set a value on agent.state using the JSONSerializableDict API."""
    agent.state.set(key, value)


def _state_clear(agent: Agent) -> None:
    """Clear all keys from agent.state.

    JSONSerializableDict has no .clear() or .keys(), so we get
    the full dict copy, then delete each key.
    """
    all_data = agent.state.get()  # Returns full dict copy
    if all_data:
        for key in list(all_data.keys()):
            agent.state.delete(key)


class AgentContainer:
    """Container for a collection of named Strands agents.

    Provides dict-like access: container["agent-name"] or container.agent_name.
    Manages shared state like user context and mode managers.

    Usage:
        container = AgentContainer(agents={"chat": agent1, "namer": agent2})
        agent = container["chat"]
        result = agent("Hello!")
    """

    def __init__(
        self,
        agents: dict[str, Agent],
        mode_managers: dict[str, ModeManager] | None = None,
        config: OrchestratorConfig | None = None,
    ):
        self._agents = agents
        self._mode_managers = mode_managers or {}
        self._config = config

    def __getitem__(self, name: str) -> Agent:
        try:
            return self._agents[name]
        except KeyError:
            raise KeyError(
                f"Agent '{name}' not found. Available: {list(self._agents.keys())}"
            )

    def __getattr__(self, name: str) -> Agent:
        if name.startswith("_"):
            raise AttributeError(name)
        # Convert underscores to hyphens for kebab-case agent names
        kebab_name = name.replace("_", "-")
        if kebab_name in self._agents:
            return self._agents[kebab_name]
        if name in self._agents:
            return self._agents[name]
        raise AttributeError(
            f"Agent '{name}' not found. Available: {list(self._agents.keys())}"
        )

    def __contains__(self, name: str) -> bool:
        return name in self._agents

    def __iter__(self):
        return iter(self._agents)

    def __len__(self) -> int:
        return len(self._agents)

    @property
    def agents(self) -> dict[str, Agent]:
        """Direct access to the agents dict."""
        return self._agents

    @property
    def agent_names(self) -> list[str]:
        """List of all agent names in this container."""
        return list(self._agents.keys())

    def get_mode_manager(self, agent_name: str) -> ModeManager | None:
        """Get the mode manager for a specific agent."""
        return self._mode_managers.get(agent_name)

    def set_user_context(self, user: UserContextProtocol) -> None:
        """Inject user context into all agents' state."""
        for agent in self._agents.values():
            _state_set(agent, "user_id", user.user_id)
            _state_set(agent, "tenant_id", user.tenant_id)
            _state_set(agent, "mcp_custom_auth_token", user.auth_token)

    def set_state_value(self, key: str, value: Any) -> None:
        """Set a value on all agents' state."""
        for agent in self._agents.values():
            _state_set(agent, key, value)

    def prepare_for_request(
        self,
        agent_name: str,
        *,
        chat_id: str,
        user: object | None = None,
        interrupt_event: asyncio.Event | None = None,
        main_loop: asyncio.AbstractEventLoop | None = None,
    ) -> Agent:
        """Prepare a named agent for a single request.

        Registers hooks based on the container's OrchestratorConfig:
        - EventBridgeHook for SSE events (if event_bus + event_factory in config)
        - AuthTokenInjectorHook for MCP tool auth (always)
        - ConsentHook for tool consent gating (if enable_consent in config)
        - InterruptHook for user cancellation (if interrupt_event provided)

        Suppresses the default PrintingCallbackHandler.

        Args:
            agent_name: Name of the agent to prepare (e.g., "mode-based").
            chat_id: Conversation ID for SSE event routing.
            user: User context for role-based event filtering.
            interrupt_event: asyncio.Event checked before each tool call.
            main_loop: The main asyncio event loop (e.g., FastAPI's). Required
                for cross-thread event publishing since Strands runs agent()
                in a ThreadPoolExecutor.

        Returns:
            The configured Agent, ready to be called with a prompt.
        """
        agent = self[agent_name]
        config = self._config

        # SSE event publishing (TURN_START/END, TOOL_CALL_START/END, REASONING_STEP)
        if config and config.event_bus and config.event_factory:
            hook = EventBridgeHook(
                event_bus=config.event_bus,
                event_factory=config.event_factory,
                chat_id=chat_id,
                agent_name=agent_name,
                user=user,
            )
            if main_loop:
                hook.set_main_loop(main_loop)
            hook.register_hooks(agent.hooks)

        # Auth token injection into tool call arguments
        AuthTokenInjectorHook(agent=agent).register_hooks(agent.hooks)

        # Tool consent gating
        if config and config.enable_consent and config.consent_service:
            ConsentHook(
                consent_service=config.consent_service,
                auto_approve_tools=config.auto_approve_tools,
                session_id=chat_id,
            ).register_hooks(agent.hooks)

        # Interrupt checking before each tool call
        if interrupt_event:
            InterruptHook(interrupt_event=interrupt_event).register_hooks(agent.hooks)

        # Suppress default PrintingCallbackHandler
        agent.callback_handler = null_callback_handler

        return agent

    async def reset_state(self) -> None:
        """Clear conversation history, state, and hooks for all agents.

        Called between requests to prevent state leakage from the agent pool.
        Hooks are cleared here because prepare_for_request() re-registers them
        each time; without clearing, callbacks accumulate and fire N times
        after N reuses.
        """
        for name, agent in self._agents.items():
            agent.messages.clear()
            _state_clear(agent)
            # Clear hook callbacks to prevent accumulation across pool reuses.
            # prepare_for_request() will re-register the necessary hooks.
            agent.hooks._registered_callbacks.clear()
            logger.debug("Reset state for agent '%s'", name)

        # Reset mode managers to default mode
        for name, manager in self._mode_managers.items():
            if manager.available_modes:
                manager.current_mode = manager.available_modes[0]
