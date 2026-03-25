"""Shared test fixtures with mock protocol implementations."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest


class MockEventBus:
    """Mock EventBusProtocol implementation."""

    def __init__(self):
        self.published_events: list[tuple[Any, Any]] = []

    async def publish(self, event: Any, user: Any = None) -> None:
        self.published_events.append((event, user))


class MockStreamEventFactory:
    """Mock StreamEventFactoryProtocol implementation."""

    def create_tool_start_event(self, chat_id, tool_name, tool_input):
        return {"type": "tool_start", "chat_id": chat_id, "tool_name": tool_name}

    def create_tool_end_event(self, chat_id, tool_name, tool_result):
        return {"type": "tool_end", "chat_id": chat_id, "tool_name": tool_name}

    def create_turn_start_event(self, chat_id, agent_name):
        return {"type": "turn_start", "chat_id": chat_id, "agent_name": agent_name}

    def create_turn_end_event(self, chat_id, agent_name):
        return {"type": "turn_end", "chat_id": chat_id, "agent_name": agent_name}

    def create_error_event(self, chat_id, error):
        return {"type": "error", "chat_id": chat_id, "error": str(error)}

    def create_background_task_completed_event(
        self, chat_id, task_id, tool_name, status, result
    ):
        return {
            "type": "bg_task_completed",
            "chat_id": chat_id,
            "task_id": task_id,
        }


class MockBackgroundInbox:
    """Mock BackgroundTaskInboxProtocol implementation."""

    def __init__(self):
        self.inbox: dict[str, list[dict]] = {}
        self.watched_containers: list[dict] = []
        self.watched_tasks: list[dict] = []

    async def pop_inbox(self, conversation_id: str) -> list[dict]:
        return self.inbox.pop(conversation_id, [])

    async def watch_container(
        self, container_group_uuid, conversation_id, tool_name, tenant_id=None
    ):
        self.watched_containers.append(
            {
                "uuid": container_group_uuid,
                "conversation_id": conversation_id,
                "tool_name": tool_name,
            }
        )

    async def watch_llm_task(
        self, task_id, conversation_id, tool_name, tenant_id=None, ensure_subscription=True
    ):
        self.watched_tasks.append(
            {
                "task_id": task_id,
                "conversation_id": conversation_id,
                "tool_name": tool_name,
            }
        )

    async def ensure_subscribed(self, conversation_id, timeout=5.0):
        return True


class MockConsentService:
    """Mock ConsentServiceProtocol implementation."""

    def __init__(self, auto_approve: bool = True):
        self._auto_approve = auto_approve
        self.checked_tools: list[str] = []

    async def check_consent(self, tool_name: str, session_id: str) -> bool:
        self.checked_tools.append(tool_name)
        return self._auto_approve

    async def request_consent(
        self, tool_name: str, tool_input: dict, session_id: str
    ) -> bool:
        return self._auto_approve


class MockUserContext:
    """Mock UserContextProtocol implementation."""

    def __init__(
        self, user_id: str = "test-user", tenant_id: int = 1, auth_token: str = "test-token"
    ):
        self._user_id = user_id
        self._tenant_id = tenant_id
        self._auth_token = auth_token

    @property
    def user_id(self) -> str:
        return self._user_id

    @property
    def tenant_id(self) -> int | None:
        return self._tenant_id

    @property
    def auth_token(self) -> str:
        return self._auth_token


class MockAgentConfigSource:
    """Mock AgentConfigSourceProtocol implementation."""

    def __init__(
        self,
        agents: list[dict] | None = None,
        modes: dict[str, dict[str, dict]] | None = None,
        servers: list[dict] | None = None,
    ):
        self._agents = agents or []
        self._modes = modes or {}
        self._servers = servers or []

    async def get_agent_configs(self) -> list[dict]:
        return self._agents

    async def get_mode_configs(self, agent_name: str) -> dict[str, dict]:
        return self._modes.get(agent_name, {})

    async def get_mcp_server_configs(self) -> list[dict]:
        return self._servers


@pytest.fixture
def mock_event_bus():
    return MockEventBus()


@pytest.fixture
def mock_event_factory():
    return MockStreamEventFactory()


@pytest.fixture
def mock_inbox():
    return MockBackgroundInbox()


@pytest.fixture
def mock_consent():
    return MockConsentService()


@pytest.fixture
def mock_user():
    return MockUserContext()


@pytest.fixture
def mock_config_source():
    return MockAgentConfigSource(
        agents=[
            {
                "name": "test-agent",
                "system_prompt": "You are a test agent.",
                "model": "sonnet",
                "description": "A test agent",
            }
        ]
    )
