"""Tests for StateAdapter."""

import pytest
from unittest.mock import MagicMock

from strands_orchestrator.container import AgentContainer
from strands_orchestrator.mode_manager import ModeManager
from strands_orchestrator.state import StateAdapter
from strands_orchestrator.types import ModeDefinition


def _make_mock_agent(name="test"):
    agent = MagicMock()
    agent.name = name
    agent.messages = []
    agent.state = {}
    return agent


class TestStateAdapter:
    def test_extract_empty(self):
        agent = _make_mock_agent()
        container = AgentContainer(agents={"a": agent})
        states = StateAdapter.extract(container)
        assert "a" in states
        assert states["a"]["messages"] == []
        assert states["a"]["current_mode"] is None

    def test_extract_with_messages(self):
        agent = _make_mock_agent()
        agent.messages = [{"role": "user", "content": "hi"}]
        agent.state = {"key": "val"}
        container = AgentContainer(agents={"a": agent})
        states = StateAdapter.extract(container)
        assert len(states["a"]["messages"]) == 1
        assert states["a"]["state"]["key"] == "val"

    def test_extract_with_mode(self):
        agent = _make_mock_agent()
        mode_mgr = ModeManager(
            [ModeDefinition(name="analysis"), ModeDefinition(name="writing")],
            default_mode="analysis",
        )
        container = AgentContainer(
            agents={"a": agent}, mode_managers={"a": mode_mgr}
        )
        states = StateAdapter.extract(container)
        assert states["a"]["current_mode"] == "analysis"

    @pytest.mark.asyncio
    async def test_restore(self):
        agent = _make_mock_agent()
        container = AgentContainer(agents={"a": agent})

        saved_states = {
            "a": {
                "messages": [{"role": "user", "content": "hello"}],
                "state": {"restored": True},
                "current_mode": None,
            }
        }

        await StateAdapter.restore(container, saved_states)
        assert len(agent.messages) == 1
        assert agent.state["restored"] is True

    @pytest.mark.asyncio
    async def test_restore_resets_first(self):
        agent = _make_mock_agent()
        agent.messages = [{"role": "user", "content": "old"}]
        agent.state = {"old_key": True}
        container = AgentContainer(agents={"a": agent})

        await StateAdapter.restore(container, {"a": {"messages": [], "state": {}}})
        # Old state should be cleared
        assert agent.messages == []
        assert "old_key" not in agent.state

    @pytest.mark.asyncio
    async def test_restore_unknown_agent(self):
        agent = _make_mock_agent()
        container = AgentContainer(agents={"a": agent})

        # Should not raise, just warn
        await StateAdapter.restore(container, {"unknown-agent": {"messages": []}})

    @pytest.mark.asyncio
    async def test_roundtrip(self):
        agent = _make_mock_agent()
        agent.messages = [{"role": "user", "content": "test"}]
        agent.state = {"counter": 42}

        container = AgentContainer(agents={"a": agent})

        # Extract
        states = StateAdapter.extract(container)

        # Reset
        await container.reset_state()
        assert agent.messages == []

        # Restore
        await StateAdapter.restore(container, states)
        assert len(agent.messages) == 1
        assert agent.state["counter"] == 42
