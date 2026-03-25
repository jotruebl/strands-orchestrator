"""AgentContainer — Dict-like wrapper around multiple Strands agents.

Provides attribute and dictionary access to agents by name,
plus lifecycle methods for state management and hook registration.
"""

from __future__ import annotations

import logging
from typing import Any

from strands import Agent

from strands_orchestrator.mode_manager import ModeManager
from strands_orchestrator.protocols import UserContextProtocol

logger = logging.getLogger(__name__)


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
    ):
        self._agents = agents
        self._mode_managers = mode_managers or {}

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
        """Inject user context into all agents' state dicts.

        Sets user_id, tenant_id, and auth_token on each agent's state
        so tools can access them via agent_state parameter.
        """
        for agent in self._agents.values():
            agent.state["user_id"] = user.user_id
            agent.state["tenant_id"] = user.tenant_id
            agent.state["mcp_custom_auth_token"] = user.auth_token

    def set_state_value(self, key: str, value: Any) -> None:
        """Set a value on all agents' state dicts."""
        for agent in self._agents.values():
            agent.state[key] = value

    async def reset_state(self) -> None:
        """Clear conversation history and state for all agents.

        Called between requests to prevent state leakage from the agent pool.
        """
        for name, agent in self._agents.items():
            agent.messages.clear()
            # agent.state may be a JSONSerializableDict that lacks .clear()
            for key in list(agent.state.keys()):
                del agent.state[key]
            logger.debug("Reset state for agent '%s'", name)

        # Reset mode managers to default mode
        for name, manager in self._mode_managers.items():
            if manager.available_modes:
                # Reset to first mode (default)
                manager.current_mode = manager.available_modes[0]
