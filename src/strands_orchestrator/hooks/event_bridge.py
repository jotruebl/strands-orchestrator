"""EventBridgeHook — bridges Strands agent events to an EventBus.

Converts Strands hook events into consumer-defined events via
StreamEventFactoryProtocol, then publishes them through EventBusProtocol.

Handles ALL agent SSE events through a single hook system:
- AGENT_TURN_START (BeforeInvocationEvent)
- AGENT_REASONING_STEP (AfterModelCallEvent)
- TOOL_CALL_START (BeforeToolCallEvent)
- TOOL_CALL_END (AfterToolCallEvent)
- AGENT_TURN_END (AfterInvocationEvent)
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import (
    AfterInvocationEvent,
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeToolCallEvent,
)

if TYPE_CHECKING:
    from strands_orchestrator.protocols import EventBusProtocol, StreamEventFactoryProtocol

logger = logging.getLogger(__name__)


class EventBridgeHook(HookProvider):
    """Publishes agent execution events to an external EventBus.

    Single hook provider that handles all SSE events from the agent loop.
    Uses run_coroutine_threadsafe to cross the thread boundary (Strands
    runs agent() in a ThreadPoolExecutor with its own event loop).
    Call set_main_loop() before invoking the agent.
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
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._iteration = 0

    def set_main_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the main event loop for cross-thread event publishing."""
        self._main_loop = loop

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeInvocationEvent, self._on_turn_start)
        registry.add_callback(AfterModelCallEvent, self._on_model_response)
        registry.add_callback(BeforeToolCallEvent, self._on_tool_start)
        registry.add_callback(AfterToolCallEvent, self._on_tool_end)
        registry.add_callback(AfterInvocationEvent, self._on_turn_end)

    def _on_turn_start(self, event: BeforeInvocationEvent) -> None:
        if not self.chat_id:
            return
        self._iteration = 0
        stream_event = self.event_factory.create_turn_start_event(
            chat_id=self.chat_id,
            agent_name=self.agent_name,
        )
        self._publish_async(stream_event)

    def _on_model_response(self, event: AfterModelCallEvent) -> None:
        """Publish AGENT_REASONING_STEP — fires after each model call."""
        if not self.chat_id:
            return
        if not event.stop_response:
            return

        message = event.stop_response.message
        stop_reason = event.stop_response.stop_reason

        # Extract text content from the model's response
        content = []
        if isinstance(message, dict):
            for block in message.get("content", []):
                if isinstance(block, dict) and "text" in block:
                    content.append({"type": "text", "text": block["text"]})

        self._iteration += 1
        stream_event = self.event_factory.create_reasoning_step_event(
            chat_id=self.chat_id,
            agent_name=self.agent_name,
            iteration=self._iteration,
            content=content,
            stop_reason=str(stop_reason) if stop_reason else "end_turn",
        )
        self._publish_async(stream_event)

    def _on_tool_start(self, event: BeforeToolCallEvent) -> None:
        if not self.chat_id:
            return
        tool_use = event.tool_use
        stream_event = self.event_factory.create_tool_start_event(
            chat_id=self.chat_id,
            tool_name=tool_use.get("name", ""),
            tool_input=tool_use.get("input", {}),
        )
        self._publish_async(stream_event)

    def _on_tool_end(self, event: AfterToolCallEvent) -> None:
        if not self.chat_id:
            return
        tool_use = event.tool_use
        stream_event = self.event_factory.create_tool_end_event(
            chat_id=self.chat_id,
            tool_name=tool_use.get("name", ""),
            tool_result=event.result if hasattr(event, "result") else None,
        )
        self._publish_async(stream_event)

    def _on_turn_end(self, event: AfterInvocationEvent) -> None:
        if not self.chat_id:
            return
        response_content = None
        if event.result and hasattr(event.result, "message") and event.result.message:
            content = event.result.message.get("content", [])
            response_content = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    response_content.append({"type": "text", "text": block["text"]})

        stream_event = self.event_factory.create_turn_end_event(
            chat_id=self.chat_id,
            agent_name=self.agent_name,
            response_content=response_content,
        )
        self._publish_async(stream_event)

    def _publish_async(self, event: object) -> None:
        """Fire-and-forget publish on the main event loop."""
        loop = self._main_loop
        if not loop or loop.is_closed():
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                logger.debug("No event loop available, skipping event publish")
                return

        try:
            asyncio.run_coroutine_threadsafe(
                self.event_bus.publish(event, user=self.user), loop
            )
        except RuntimeError:
            logger.debug("Failed to schedule event publish on main loop")
