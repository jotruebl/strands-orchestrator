"""ModeManager — mode-aware tool filtering with dynamic mode switching.

Provides mode-based tool visibility control. Each mode defines which
servers and tools are accessible. A virtual 'switch_mode' tool allows
the LLM to change modes during a conversation.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from strands import tool as strands_tool

from strands_orchestrator.types import ModeDefinition

logger = logging.getLogger(__name__)


class ModeManager:
    """Manages mode-based tool filtering for an agent.

    Each mode defines a set of allowed servers and tools. The manager
    filters the full tool list based on the current mode, and provides
    a switch_mode tool that the LLM can call to change modes.
    """

    def __init__(
        self,
        modes: list[ModeDefinition],
        default_mode: str | None = None,
    ):
        self._modes: dict[str, ModeDefinition] = {m.name: m for m in modes}
        self._current_mode: str | None = default_mode or (
            modes[0].name if modes else None
        )
        self._all_tools: list = []
        self._tool_server_map: dict[str, str] = {}

    @property
    def current_mode(self) -> str | None:
        return self._current_mode

    @current_mode.setter
    def current_mode(self, mode_name: str) -> None:
        if mode_name not in self._modes:
            raise ValueError(
                f"Unknown mode '{mode_name}'. Available: {list(self._modes.keys())}"
            )
        self._current_mode = mode_name

    @property
    def available_modes(self) -> list[str]:
        return list(self._modes.keys())

    def set_tools(self, tools: list, tool_server_map: dict[str, str] | None = None) -> None:
        """Set the full tool list and optional server mapping.

        Args:
            tools: All available tools from MCP servers.
            tool_server_map: Mapping of tool_name -> server_name.
                If not provided, tools are assumed to be allowed in all modes.
        """
        self._all_tools = tools
        self._tool_server_map = tool_server_map or {}

    def get_filtered_tools(self) -> list:
        """Get tools filtered by the current mode.

        Returns all tools if no mode is set or mode has no restrictions.
        """
        if not self._current_mode or self._current_mode not in self._modes:
            return list(self._all_tools)

        mode = self._modes[self._current_mode]
        if not mode.servers:
            return list(self._all_tools)

        filtered = []
        for tool in self._all_tools:
            tool_name = self._get_tool_name(tool)
            if self._is_tool_allowed(tool_name, mode):
                filtered.append(tool)

        return filtered

    def is_tool_allowed_in_current_mode(self, tool_name: str) -> bool:
        """Check if a specific tool is allowed in the current mode."""
        if not self._current_mode or self._current_mode not in self._modes:
            return True

        mode = self._modes[self._current_mode]
        if not mode.servers:
            return True

        return self._is_tool_allowed(tool_name, mode)

    def _is_tool_allowed(self, tool_name: str, mode: ModeDefinition) -> bool:
        """Check if a tool is in the mode's allowlist."""
        server_name = self._tool_server_map.get(tool_name)

        if server_name and server_name in mode.servers:
            allowed_tools = mode.servers[server_name]
            return "*" in allowed_tools or tool_name in allowed_tools

        # If tool has no server mapping, check all server allowlists
        for allowed_tools in mode.servers.values():
            if "*" in allowed_tools or tool_name in allowed_tools:
                return True

        return False

    def _get_tool_name(self, tool: Any) -> str:
        """Extract tool name from a tool object."""
        if hasattr(tool, "tool_name"):
            return tool.tool_name
        if hasattr(tool, "name"):
            return tool.name
        if isinstance(tool, dict):
            return tool.get("name", "")
        return str(tool)

    def create_switch_mode_tool(self) -> Any:
        """Create a Strands tool for switching modes.

        Returns None if there's only one mode (no switching needed).
        """
        if len(self._modes) <= 1:
            return None

        mode_descriptions = {
            name: mode.description for name, mode in self._modes.items()
        }
        modes_json = json.dumps(mode_descriptions, indent=2)
        available = list(self._modes.keys())
        manager = self  # capture for closure

        @strands_tool
        def switch_mode(mode_name: str, reason: str) -> dict[str, Any]:
            """Switch the agent to a different mode, changing which tools are available.

            Args:
                mode_name: The name of the mode to switch to.
                reason: Why you are switching modes.

            Returns:
                Confirmation of the mode switch.
            """
            if mode_name not in available:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": f"Unknown mode '{mode_name}'. "
                            f"Available modes: {available}"
                        }
                    ],
                }

            old_mode = manager.current_mode
            manager.current_mode = mode_name
            logger.info(
                "Mode switched: %s → %s (reason: %s)", old_mode, mode_name, reason
            )
            return {
                "status": "success",
                "content": [
                    {
                        "text": f"Switched from '{old_mode}' to '{mode_name}'. "
                        f"Tools have been updated for the new mode."
                    }
                ],
            }

        # Update the tool's docstring with available modes
        switch_mode.__doc__ = (
            f"Switch the agent to a different mode.\n\n"
            f"Available modes:\n{modes_json}\n\n"
            f"Args:\n"
            f"    mode_name: The name of the mode to switch to.\n"
            f"    reason: Why you are switching modes.\n"
        )

        return switch_mode
