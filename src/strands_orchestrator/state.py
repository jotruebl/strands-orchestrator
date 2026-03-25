"""State management — serialization and restoration of agent state.

Handles converting between live Strands agent state and
serializable Pydantic models for persistence.

Note: Strands Agent.state is a JSONSerializableDict with limited API.
Use agent.state.get() for full dict copy, agent.state.set(k, v) to set.
"""

from __future__ import annotations

import logging

from strands import Agent

from strands_orchestrator.container import AgentContainer, _state_set
from strands_orchestrator.types import AgentState

logger = logging.getLogger(__name__)


class StateAdapter:
    """Bridges between AgentContainer runtime state and serializable AgentState.

    Extract: Read live agent state → AgentState Pydantic models
    Restore: AgentState Pydantic models → hydrate live agents
    """

    @staticmethod
    def extract(container: AgentContainer) -> dict[str, dict]:
        """Extract state from all agents in a container.

        Returns:
            Dict mapping agent_name → serialized AgentState dict.
        """
        states = {}
        for name, agent in container.agents.items():
            mode_manager = container.get_mode_manager(name)
            # agent.state.get() with no args returns a full dict copy
            state_dict = agent.state.get() or {}
            state = AgentState(
                messages=list(agent.messages),
                state=state_dict,
                current_mode=mode_manager.current_mode if mode_manager else None,
            )
            states[name] = state.model_dump(mode="json")

        return states

    @staticmethod
    async def restore(
        container: AgentContainer,
        states: dict[str, dict],
    ) -> None:
        """Restore state into agents from serialized AgentState dicts.

        Always resets all agents first to prevent state leakage,
        then restores from the provided states.

        Args:
            container: The agent container to hydrate.
            states: Dict mapping agent_name → AgentState dict.
        """
        # CRITICAL: Always reset first to prevent state leakage
        await container.reset_state()

        for agent_name, state_data in states.items():
            if agent_name not in container:
                logger.warning(
                    "Agent '%s' from saved state not found in container. Skipping.",
                    agent_name,
                )
                continue

            agent = container[agent_name]
            state = AgentState.model_validate(state_data)

            # Restore conversation history
            if state.messages:
                agent.messages.extend(state.messages)

            # Restore custom state via JSONSerializableDict API
            if state.state:
                for key, value in state.state.items():
                    _state_set(agent, key, value)

            # Restore mode
            mode_manager = container.get_mode_manager(agent_name)
            if mode_manager and state.current_mode:
                try:
                    mode_manager.current_mode = state.current_mode
                except ValueError:
                    logger.warning(
                        "Saved mode '%s' for agent '%s' no longer exists. "
                        "Using default.",
                        state.current_mode,
                        agent_name,
                    )

            logger.debug(
                "Restored state for agent '%s': %d messages, mode=%s",
                agent_name,
                len(state.messages),
                state.current_mode,
            )
