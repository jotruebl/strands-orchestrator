"""Tests for ModeManager."""

import pytest

from strands_orchestrator.mode_manager import ModeManager
from strands_orchestrator.types import ModeDefinition


def _make_mode(name, servers=None):
    return ModeDefinition(name=name, description=f"{name} mode", servers=servers or {})


def _make_tool(name):
    """Create a mock tool object."""
    from unittest.mock import MagicMock
    tool = MagicMock()
    tool.tool_name = name
    tool.name = name
    return tool


class TestModeManager:
    def test_initial_mode(self):
        modes = [_make_mode("default"), _make_mode("advanced")]
        mgr = ModeManager(modes, default_mode="default")
        assert mgr.current_mode == "default"

    def test_initial_mode_defaults_to_first(self):
        modes = [_make_mode("alpha"), _make_mode("beta")]
        mgr = ModeManager(modes)
        assert mgr.current_mode == "alpha"

    def test_switch_mode(self):
        modes = [_make_mode("a"), _make_mode("b")]
        mgr = ModeManager(modes)
        mgr.current_mode = "b"
        assert mgr.current_mode == "b"

    def test_switch_invalid_mode(self):
        modes = [_make_mode("a")]
        mgr = ModeManager(modes)
        with pytest.raises(ValueError, match="Unknown mode"):
            mgr.current_mode = "nonexistent"

    def test_available_modes(self):
        modes = [_make_mode("x"), _make_mode("y"), _make_mode("z")]
        mgr = ModeManager(modes)
        assert mgr.available_modes == ["x", "y", "z"]

    def test_filter_tools_by_mode(self):
        modes = [
            _make_mode("analysis", servers={"fusion": ["analyze", "scan"]}),
            _make_mode("writing", servers={"fusion": ["write", "export"]}),
        ]
        mgr = ModeManager(modes, default_mode="analysis")

        tools = [_make_tool("analyze"), _make_tool("scan"), _make_tool("write")]
        mgr.set_tools(tools, {"analyze": "fusion", "scan": "fusion", "write": "fusion"})

        filtered = mgr.get_filtered_tools()
        names = [t.tool_name for t in filtered]
        assert "analyze" in names
        assert "scan" in names
        assert "write" not in names

    def test_wildcard_server(self):
        modes = [_make_mode("all", servers={"fusion": ["*"]})]
        mgr = ModeManager(modes, default_mode="all")

        tools = [_make_tool("a"), _make_tool("b"), _make_tool("c")]
        mgr.set_tools(tools, {"a": "fusion", "b": "fusion", "c": "fusion"})

        assert len(mgr.get_filtered_tools()) == 3

    def test_no_modes_returns_all_tools(self):
        mgr = ModeManager([])
        tools = [_make_tool("x"), _make_tool("y")]
        mgr.set_tools(tools)
        assert len(mgr.get_filtered_tools()) == 2

    def test_is_tool_allowed(self):
        modes = [_make_mode("m1", servers={"s1": ["tool_a"]})]
        mgr = ModeManager(modes, default_mode="m1")
        mgr.set_tools([], {"tool_a": "s1", "tool_b": "s1"})

        assert mgr.is_tool_allowed_in_current_mode("tool_a") is True
        assert mgr.is_tool_allowed_in_current_mode("tool_b") is False

    def test_create_switch_mode_tool_single_mode(self):
        mgr = ModeManager([_make_mode("only")])
        assert mgr.create_switch_mode_tool() is None

    def test_create_switch_mode_tool_multiple_modes(self):
        mgr = ModeManager([_make_mode("a"), _make_mode("b")])
        tool = mgr.create_switch_mode_tool()
        assert tool is not None
