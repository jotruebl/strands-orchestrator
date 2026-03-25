"""EventBridgeHook — bridges Strands agent events to an EventBus.

Converts Strands hook events (BeforeToolCall, AfterToolCall, etc.)
into consumer-defined events via StreamEventFactoryProtocol,
then publishes them through EventBusProtocol.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import (
    AfterInvocationEvent,
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeToolCallEvent,
)

if TYPE_CHECKING:
    from strands_orchestrator.protocols import EventBusProtocol, StreamEventFactoryProtocol

logger = logging.getLogger(__name__)


class EventBridgeHook(HookProvider):
    """Publishes agent execution events to an external EventBus.

    Translates Strands lifecycle events into consumer-defined event
    objects via StreamEventFactoryProtocol, then publishes them
    asynchronously via EventBusProtocol.
    """

    def __init__(
        self,
        event_bus: EventBusProtocol,
        event_factory: StreamEventFactoryProtocol,
        chat_id: str = "",
        agent_name: str = "",
        user: object | None = None,
    ):
        self.event_bus = event_bus
        self.event_factory = event_factory
        self.chat_id = chat_id
        self.agent_name = agent_name
        self.user = user

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeInvocationEvent, self._on_turn_start)
        registry.add_callback(AfterInvocationEvent, self._on_turn_end)
        registry.add_callback(BeforeToolCallEvent, self._on_tool_start)
        registry.add_callback(AfterToolCallEvent, self._on_tool_end)

    def _on_turn_start(self, event: BeforeInvocationEvent) -> None:
        """Publish agent turn start event."""
        if not self.chat_id:
            return
        stream_event = self.event_factory.create_turn_start_event(
            chat_id=self.chat_id,
            agent_name=self.agent_name,
        )
        self._publish_async(stream_event)

    def _on_turn_end(self, event: AfterInvocationEvent) -> None:
        """Publish agent turn end event."""
        if not self.chat_id:
            return
        stream_event = self.event_factory.create_turn_end_event(
            chat_id=self.chat_id,
            agent_name=self.agent_name,
        )
        self._publish_async(stream_event)

    def _on_tool_start(self, event: BeforeToolCallEvent) -> None:
        """Publish tool call start event."""
        if not self.chat_id:
            return
        stream_event = self.event_factory.create_tool_start_event(
            chat_id=self.chat_id,
            tool_name=event.tool_name,
            tool_input=event.tool_input if hasattr(event, "tool_input") else {},
        )
        self._publish_async(stream_event)

    def _on_tool_end(self, event: AfterToolCallEvent) -> None:
        """Publish tool call end event."""
        if not self.chat_id:
            return
        stream_event = self.event_factory.create_tool_end_event(
            chat_id=self.chat_id,
            tool_name=event.tool_name,
            tool_result=event.tool_result if hasattr(event, "tool_result") else None,
        )
        self._publish_async(stream_event)

    def _publish_async(self, event: object) -> None:
        """Fire-and-forget async publish."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.event_bus.publish(event, user=self.user))
        except RuntimeError:
            logger.debug("No running event loop, skipping event publish")
