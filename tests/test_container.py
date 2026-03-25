"""Tests for AgentContainer."""

import pytest
from unittest.mock import MagicMock

from strands_orchestrator.container import AgentContainer
from tests.conftest import MockUserContext


def _make_mock_agent(name="test"):
    """Create a mock Strands Agent."""
    agent = MagicMock()
    agent.name = name
    agent.messages = []
    agent.state = {}
    return agent


class TestAgentContainer:
    def test_getitem(self):
        agent = _make_mock_agent()
        container = AgentContainer(agents={"my-agent": agent})
        assert container["my-agent"] is agent

    def test_getitem_missing(self):
        container = AgentContainer(agents={})
        with pytest.raises(KeyError, match="not found"):
            container["missing"]

    def test_getattr(self):
        agent = _make_mock_agent()
        container = AgentContainer(agents={"my-agent": agent})
        # Underscores map to hyphens
        assert container.my_agent is agent

    def test_getattr_missing(self):
        container = AgentContainer(agents={})
        with pytest.raises(AttributeError):
            _ = container.missing

    def test_contains(self):
        agent = _make_mock_agent()
        container = AgentContainer(agents={"chat": agent})
        assert "chat" in container
        assert "other" not in container

    def test_len(self):
        container = AgentContainer(
            agents={"a": _make_mock_agent(), "b": _make_mock_agent()}
        )
        assert len(container) == 2

    def test_agent_names(self):
        container = AgentContainer(
            agents={"alpha": _make_mock_agent(), "beta": _make_mock_agent()}
        )
        assert container.agent_names == ["alpha", "beta"]

    def test_set_user_context(self):
        agent = _make_mock_agent()
        container = AgentContainer(agents={"agent": agent})
        user = MockUserContext(user_id="u1", tenant_id=5, auth_token="tok")
        container.set_user_context(user)
        assert agent.state["user_id"] == "u1"
        assert agent.state["tenant_id"] == 5
        assert agent.state["mcp_custom_auth_token"] == "tok"

    @pytest.mark.asyncio
    async def test_reset_state(self):
        agent = _make_mock_agent()
        agent.messages = [{"role": "user", "content": "hi"}]
        agent.state = {"key": "val"}
        container = AgentContainer(agents={"agent": agent})
        await container.reset_state()
        assert agent.messages == []
        assert agent.state == {}
