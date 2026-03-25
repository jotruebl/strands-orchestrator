"""strands-orchestrator — Protocol-based DI framework for Strands Agents SDK.

Provides agent pooling, MCP server management, mode-aware tool filtering,
and extensible hooks for event streaming, consent, and background tasks.

Usage:
    from strands_orchestrator import AgentPoolService, OrchestratorConfig
    from strands_orchestrator.sources import YAMLAgentConfigSource

    config = OrchestratorConfig(
        agent_config_source=YAMLAgentConfigSource("./config"),
        pool_size=2,
    )
    pool = await AgentPoolService.create(config)

    async with pool.get_container(chat_id="abc") as container:
        agent = container["my-agent"]
        result = agent("Hello!")
"""

from strands_orchestrator.config import OrchestratorConfig
from strands_orchestrator.container import AgentContainer
from strands_orchestrator.factory import AgentFactory
from strands_orchestrator.mcp_connector import MCPConnector
from strands_orchestrator.mode_manager import ModeManager
from strands_orchestrator.model_factory import ModelFactory
from strands_orchestrator.pool import AgentPoolService
from strands_orchestrator.state import StateAdapter

__all__ = [
    "AgentContainer",
    "AgentFactory",
    "AgentPoolService",
    "MCPConnector",
    "ModeManager",
    "ModelFactory",
    "OrchestratorConfig",
    "StateAdapter",
]

__version__ = "0.1.0"
