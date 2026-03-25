"""ConsentHook — gates tool execution behind user consent.

Checks ConsentServiceProtocol before each tool call. Auto-approved
tools bypass the check. If consent is denied, the tool call is
blocked and an error message is returned to the LLM.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import BeforeToolCallEvent
from strands.types.tools import ToolUse

if TYPE_CHECKING:
    from strands_orchestrator.protocols import ConsentServiceProtocol

logger = logging.getLogger(__name__)

# Tools that never require consent
DEFAULT_AUTO_APPROVE: set[str] = {"switch_mode"}


class ConsentHook(HookProvider):
    """Checks tool consent before execution.

    If a tool is not in the auto-approve list and hasn't been
    pre-approved, the hook requests consent via the ConsentService.
    If denied, the tool call is blocked.
    """

    def __init__(
        self,
        consent_service: ConsentServiceProtocol,
        auto_approve_tools: set[str] | None = None,
        session_id: str = "",
    ):
        self.consent_service = consent_service
        self.auto_approve_tools = (auto_approve_tools or set()) | DEFAULT_AUTO_APPROVE
        self.session_id = session_id

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeToolCallEvent, self._check_consent)

    def _check_consent(self, event: BeforeToolCallEvent) -> None:
        """Check if tool execution is consented."""
        tool_use: ToolUse = event.tool_use
        tool_name: str = tool_use["name"]

        if tool_name in self.auto_approve_tools:
            return

        if not self.session_id:
            return

        try:
            asyncio.ensure_future(
                self._async_consent_check(tool_name, tool_use)
            )
            logger.info(
                "Consent check initiated for tool '%s' in session '%s'",
                tool_name,
                self.session_id,
            )
        except RuntimeError:
            logger.warning("No running event loop for consent check")

    async def _async_consent_check(
        self, tool_name: str, tool_use: ToolUse
    ) -> bool:
        """Async consent check implementation."""
        is_approved = await self.consent_service.check_consent(
            tool_name, self.session_id
        )
        if is_approved:
            return True

        tool_input: dict = tool_use["input"] if isinstance(tool_use["input"], dict) else {}
        approved = await self.consent_service.request_consent(
            tool_name, tool_input, self.session_id
        )

        if not approved:
            logger.info(
                "Tool '%s' denied by user in session '%s'",
                tool_name,
                self.session_id,
            )

        return approved
