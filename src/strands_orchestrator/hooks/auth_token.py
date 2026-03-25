"""AuthTokenInjectorHook — injects auth tokens into tool call arguments.

Reads a token from agent.state and transparently adds it to every
tool call's input dict before execution. This allows MCP tools to
authenticate with backend APIs without the LLM needing to know
about the token.
"""

from __future__ import annotations

import logging

from strands import Agent
from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import BeforeToolCallEvent

logger = logging.getLogger(__name__)


class AuthTokenInjectorHook(HookProvider):
    """Injects an auth token into tool call arguments.

    Reads the token from ``agent.state.get("mcp_custom_auth_token")``
    and adds it to ``tool_use["input"]["mcp_custom_auth_token"]`` before
    each tool call. If no token is set, the hook is a no-op.

    Args:
        agent: The Strands Agent whose state contains the auth token.
    """

    def __init__(self, agent: Agent):
        self._agent = agent

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeToolCallEvent, self._inject_token)

    def _inject_token(self, event: BeforeToolCallEvent) -> None:
        token = self._agent.state.get("mcp_custom_auth_token")
        if not token:
            return

        tool_input = event.tool_use["input"]
        if isinstance(tool_input, dict):
            tool_input["mcp_custom_auth_token"] = token
        else:
            # input is not a dict (e.g., None or string) — replace with dict
            event.tool_use["input"] = {"mcp_custom_auth_token": token}
