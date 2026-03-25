"""MCPConnector — manages connections to multiple MCP servers.

Aggregates tools from all connected servers and manages
connection lifecycle via context manager.
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp import stdio_client, StdioServerParameters
from mcp.client.sse import sse_client
from strands.tools.mcp import MCPClient

from strands_orchestrator.types import MCPServerDefinition

logger = logging.getLogger(__name__)


class MCPConnector:
    """Manages multiple MCP server connections and aggregates their tools.

    Usage:
        connector = MCPConnector(server_configs)
        async with connector:
            tools = connector.get_all_tools()
            agent = Agent(tools=tools)
    """

    def __init__(self, servers: list[MCPServerDefinition]):
        self._server_defs = servers
        self._clients: dict[str, MCPClient] = {}
        self._tools_by_server: dict[str, list] = {}
        self._exit_stack: AsyncExitStack | None = None

    async def __aenter__(self) -> MCPConnector:
        """Open connections to all MCP servers."""
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        for server_def in self._server_defs:
            try:
                client = self._create_client(server_def)
                self._clients[server_def.name] = client
                await self._exit_stack.enter_async_context(client)
                tools = client.list_tools_sync()
                self._tools_by_server[server_def.name] = tools
                logger.info(
                    "Connected to MCP server '%s' (%s transport, %d tools)",
                    server_def.name,
                    server_def.transport,
                    len(tools),
                )
            except Exception:
                logger.exception(
                    "Failed to connect to MCP server '%s'", server_def.name
                )

        return self

    async def __aexit__(self, *exc: Any) -> None:
        """Close all MCP server connections."""
        if self._exit_stack:
            await self._exit_stack.__aexit__(*exc)
        self._clients.clear()
        self._tools_by_server.clear()

    def _create_client(self, server_def: MCPServerDefinition) -> MCPClient:
        """Create an MCPClient for a server definition."""
        if server_def.transport == "sse":
            if not server_def.url:
                raise ValueError(
                    f"SSE transport requires 'url' for server '{server_def.name}'"
                )
            return MCPClient(lambda url=server_def.url: sse_client(url))

        elif server_def.transport == "stdio":
            if not server_def.command:
                raise ValueError(
                    f"stdio transport requires 'command' for server '{server_def.name}'"
                )
            params = StdioServerParameters(
                command=server_def.command,
                args=server_def.args,
                env=server_def.env or None,
            )
            return MCPClient(lambda p=params: stdio_client(p))

        elif server_def.transport == "streamable-http":
            if not server_def.url:
                raise ValueError(
                    f"streamable-http transport requires 'url' for server '{server_def.name}'"
                )
            from mcp.client.streamable_http import streamablehttp_client

            return MCPClient(
                lambda url=server_def.url: streamablehttp_client(url)
            )

        else:
            raise ValueError(
                f"Unknown transport '{server_def.transport}' for server '{server_def.name}'"
            )

    def get_all_tools(self) -> list:
        """Get aggregated tools from all connected servers."""
        all_tools = []
        for tools in self._tools_by_server.values():
            all_tools.extend(tools)
        return all_tools

    def get_tools_by_server(self, server_name: str) -> list:
        """Get tools from a specific server."""
        return self._tools_by_server.get(server_name, [])

    def get_server_names(self) -> list[str]:
        """Get names of all connected servers."""
        return list(self._clients.keys())

    @property
    def is_connected(self) -> bool:
        """Whether any servers are connected."""
        return len(self._clients) > 0
