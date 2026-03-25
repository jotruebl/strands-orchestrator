"""AgentFactory — builds AgentContainer from config source + MCP + modes.

Takes an AgentConfigSourceProtocol, connects to MCP servers,
sets up mode filtering, and creates Strands Agent instances.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from strands import Agent

from strands_orchestrator.container import AgentContainer
from strands_orchestrator.mcp_connector import MCPConnector
from strands_orchestrator.mode_manager import ModeManager
from strands_orchestrator.model_factory import ModelFactory
from strands_orchestrator.types import AgentDefinition, MCPServerDefinition, ModeDefinition

if TYPE_CHECKING:
    from strands_orchestrator.protocols import AgentConfigSourceProtocol

logger = logging.getLogger(__name__)


class AgentFactory:
    """Builds an AgentContainer from configuration.

    Orchestrates the creation pipeline:
    1. Load agent configs from source (MongoDB, YAML, etc.)
    2. Connect to MCP servers
    3. Set up mode filtering per agent
    4. Create Strands Agent instances
    5. Return an AgentContainer
    """

    def __init__(
        self,
        config_source: AgentConfigSourceProtocol,
        model_factory: ModelFactory,
        mcp_connector: MCPConnector | None = None,
        enable_mode_filtering: bool = True,
    ):
        self._config_source = config_source
        self._model_factory = model_factory
        self._mcp_connector = mcp_connector
        self._enable_mode_filtering = enable_mode_filtering

    async def build(self) -> AgentContainer:
        """Build an AgentContainer from the configured sources.

        Returns:
            AgentContainer with all agents initialized and ready.
        """
        # 1. Load agent definitions
        raw_configs = await self._config_source.get_agent_configs()
        agent_defs = [AgentDefinition.model_validate(c) for c in raw_configs]

        logger.info(
            "Building %d agents: %s",
            len(agent_defs),
            [a.name for a in agent_defs],
        )

        # 2. Get all available tools from MCP
        all_tools = []
        if self._mcp_connector and self._mcp_connector.is_connected:
            all_tools = self._mcp_connector.get_all_tools()
            logger.info("Loaded %d tools from MCP servers", len(all_tools))

        # 3. Build each agent
        agents: dict[str, Agent] = {}
        mode_managers: dict[str, ModeManager] = {}

        for agent_def in agent_defs:
            agent, mode_manager = await self._build_agent(agent_def, all_tools)
            agents[agent_def.name] = agent
            if mode_manager:
                mode_managers[agent_def.name] = mode_manager

        container = AgentContainer(agents=agents, mode_managers=mode_managers)
        logger.info("AgentContainer built with %d agents", len(agents))
        return container

    async def _build_agent(
        self,
        agent_def: AgentDefinition,
        all_tools: list,
    ) -> tuple[Agent, ModeManager | None]:
        """Build a single agent from its definition."""
        # Create model
        model = self._model_factory.create(agent_def.model, max_tokens=agent_def.max_tokens)

        # Set up mode filtering
        mode_manager = None
        tools = list(all_tools)

        if self._enable_mode_filtering and agent_def.modes:
            # Load mode configs
            mode_configs = await self._config_source.get_mode_configs(agent_def.name)
            mode_defs = []
            for mode_name, mode_data in mode_configs.items():
                mode_defs.append(ModeDefinition.model_validate({"name": mode_name, **mode_data}))

            # Also include modes from agent_def if not in config source
            for mode in agent_def.modes:
                if mode.name not in mode_configs:
                    mode_defs.append(mode)

            if mode_defs:
                mode_manager = ModeManager(
                    modes=mode_defs,
                    default_mode=agent_def.default_mode,
                )

                # Build tool-server mapping
                tool_server_map = self._build_tool_server_map(all_tools)
                mode_manager.set_tools(all_tools, tool_server_map)

                # Use filtered tools for this agent
                tools = mode_manager.get_filtered_tools()

                # Add switch_mode tool if multiple modes
                switch_tool = mode_manager.create_switch_mode_tool()
                if switch_tool:
                    tools.append(switch_tool)

        # Build system prompt
        system_prompt = agent_def.system_prompt

        # Add mode instructions if applicable
        if mode_manager and mode_manager.current_mode:
            mode = mode_manager._modes.get(mode_manager.current_mode)
            if mode and mode.instructions:
                system_prompt = f"{system_prompt}\n\n{mode.instructions}"

        # Create the agent
        agent = Agent(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            name=agent_def.name,
        )

        logger.info(
            "Created agent '%s' (model=%s, tools=%d, modes=%d)",
            agent_def.name,
            agent_def.model,
            len(tools),
            len(agent_def.modes),
        )

        return agent, mode_manager

    def _build_tool_server_map(self, tools: list) -> dict[str, str]:
        """Build a mapping of tool_name → server_name from MCP connector."""
        tool_server_map = {}
        if self._mcp_connector:
            for server_name in self._mcp_connector.get_server_names():
                for tool in self._mcp_connector.get_tools_by_server(server_name):
                    name = tool.tool_name if hasattr(tool, "tool_name") else str(tool)
                    tool_server_map[name] = server_name
        return tool_server_map
