"""InterruptHook — enables user-initiated cancellation of agent execution.

Checks an asyncio.Event before each tool call. If the event is set,
raises CancelledError to stop the agent loop.
"""

from __future__ import annotations

import asyncio
import logging

from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import BeforeToolCallEvent

logger = logging.getLogger(__name__)


class InterruptHook(HookProvider):
    """Checks for user-initiated interruption before each tool call.

    Usage:
        interrupt_event = asyncio.Event()
        hook = InterruptHook(interrupt_event)
        # To interrupt: interrupt_event.set()
    """

    def __init__(self, interrupt_event: asyncio.Event | None = None):
        self.interrupt_event = interrupt_event or asyncio.Event()

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeToolCallEvent, self._check_interrupt)

    def _check_interrupt(self, event: BeforeToolCallEvent) -> None:
        """Raise CancelledError if interrupt has been requested."""
        if self.interrupt_event.is_set():
            logger.info(
                "Interrupt detected before tool call '%s'. Cancelling.",
                event.tool_name,
            )
            raise asyncio.CancelledError("User requested interruption")

    def request_interrupt(self) -> None:
        """Signal that execution should be interrupted."""
        self.interrupt_event.set()

    def clear_interrupt(self) -> None:
        """Clear the interrupt flag for the next invocation."""
        self.interrupt_event.clear()
