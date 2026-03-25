"""InboxHook — integrates background task results into agent conversations.

Handles three responsibilities:
1. Before each invocation: pop completed results from inbox and inject as messages
2. After tool calls: auto-register containers for completion tracking
3. After tool calls: auto-register LLM tasks for completion tracking
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING, Any

from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import (
    AfterToolCallEvent,
    BeforeInvocationEvent,
)

if TYPE_CHECKING:
    from strands_orchestrator.protocols import BackgroundTaskInboxProtocol

logger = logging.getLogger(__name__)

# Tools that produce LLM background tasks
LLM_TASK_TOOLS = {"prompt_subagent"}

# Pattern for extracting container_group_uuid from tool results
CONTAINER_UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)

# Pattern for extracting task IDs from prompt_subagent results
TASK_ID_PATTERN = re.compile(r"\*\*Task ID:\*\*\s*(\d+)")


class InboxHook(HookProvider):
    """Integrates background task results into agent conversations.

    Before invocation: Pops completed results from the inbox and
    prepends them to the conversation as user messages.

    After tool calls: Inspects tool results for container UUIDs
    and task IDs, auto-registering them for completion tracking.
    """

    def __init__(
        self,
        inbox_service: BackgroundTaskInboxProtocol,
        conversation_id: str = "",
        user_tenant_id: int | None = None,
    ):
        self.inbox_service = inbox_service
        self.conversation_id = conversation_id
        self.user_tenant_id = user_tenant_id

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeInvocationEvent, self._inject_inbox_results)
        registry.add_callback(AfterToolCallEvent, self._auto_register_tasks)

    def _inject_inbox_results(self, event: BeforeInvocationEvent) -> None:
        """Pop inbox and inject background task results into conversation."""
        if not self.conversation_id:
            return

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._async_inject(event))
        except RuntimeError:
            pass

    async def _async_inject(self, event: BeforeInvocationEvent) -> None:
        """Async inbox injection."""
        results = await self.inbox_service.pop_inbox(self.conversation_id)
        if not results:
            return

        logger.info(
            "Injecting %d background task results into conversation %s",
            len(results),
            self.conversation_id,
        )

        # Inject results as context into the agent
        # The consuming application is responsible for how these get
        # injected into the conversation (e.g., as synthetic user messages)
        # We store them on the event for the application layer to handle
        if hasattr(event, "metadata"):
            event.metadata["inbox_results"] = results

    def _auto_register_tasks(self, event: AfterToolCallEvent) -> None:
        """Inspect tool results and auto-register background tasks."""
        if not self.conversation_id:
            return

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._async_auto_register(event)
            )
        except RuntimeError:
            pass

    async def _async_auto_register(self, event: AfterToolCallEvent) -> None:
        """Async auto-registration of containers and LLM tasks."""
        tool_name: str = event.tool_use["name"]
        tool_result = event.result

        if tool_result is None:
            return

        result_text = self._extract_text(tool_result)
        if not result_text:
            return

        # Check for container_group_uuid in results
        await self._check_container_registration(tool_name, result_text)

        # Check for LLM task registration (prompt_subagent)
        if tool_name in LLM_TASK_TOOLS:
            await self._check_llm_task_registration(tool_name, result_text)

    async def _check_container_registration(
        self, tool_name: str, result_text: str
    ) -> None:
        """Auto-register containers for completion tracking."""
        # Look for container_group_uuid in result
        try:
            data = json.loads(result_text)
            container_uuid = data.get("container_group_uuid")
        except (json.JSONDecodeError, AttributeError):
            # Try regex extraction as fallback
            match = CONTAINER_UUID_PATTERN.search(result_text)
            container_uuid = match.group(0) if match else None

        if container_uuid:
            await self.inbox_service.watch_container(
                container_group_uuid=container_uuid,
                conversation_id=self.conversation_id,
                tool_name=tool_name,
                tenant_id=self.user_tenant_id,
            )
            await self.inbox_service.ensure_subscribed(
                self.conversation_id, timeout=5.0
            )
            logger.info(
                "Auto-registered container %s for tracking (tool=%s)",
                container_uuid[:8],
                tool_name,
            )

    async def _check_llm_task_registration(
        self, tool_name: str, result_text: str
    ) -> None:
        """Auto-register LLM tasks for completion tracking."""
        match = TASK_ID_PATTERN.search(result_text)
        if not match:
            return

        task_id = match.group(1).strip()
        await self.inbox_service.watch_llm_task(
            task_id=task_id,
            conversation_id=self.conversation_id,
            tool_name=tool_name,
            tenant_id=self.user_tenant_id,
        )
        logger.info(
            "Auto-registered LLM task %s for tracking (tool=%s)",
            task_id,
            tool_name,
        )

    @staticmethod
    def _extract_text(tool_result: Any) -> str | None:
        """Extract text content from a tool result."""
        if isinstance(tool_result, str):
            return tool_result
        if isinstance(tool_result, dict):
            content = tool_result.get("content", [])
            if isinstance(content, list):
                texts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        texts.append(item["text"])
                return "\n".join(texts) if texts else None
            return str(content)
        return str(tool_result)
