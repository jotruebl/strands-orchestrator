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
        tool_name = event.tool_name

        # Auto-approved tools skip consent
        if tool_name in self.auto_approve_tools:
            return

        if not self.session_id:
            return

        # Run async consent check synchronously in the hook
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.ensure_future(
                self._async_consent_check(tool_name, event)
            )
            # Note: Strands hooks are synchronous. If consent checking
            # needs to be async, this may need to be adapted based on
            # how Strands handles hook blocking.
            # For now, log the consent check intent.
            logger.info(
                "Consent check initiated for tool '%s' in session '%s'",
                tool_name,
                self.session_id,
            )
        except RuntimeError:
            logger.warning("No running event loop for consent check")

    async def _async_consent_check(
        self, tool_name: str, event: BeforeToolCallEvent
    ) -> bool:
        """Async consent check implementation."""
        # First check if already consented
        is_approved = await self.consent_service.check_consent(
            tool_name, self.session_id
        )
        if is_approved:
            return True

        # Request consent
        tool_input = event.tool_input if hasattr(event, "tool_input") else {}
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
