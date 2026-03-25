"""AgentPoolService — manages a pool of pre-warmed AgentContainers.

Pre-warms agent instances with MCP connections at startup.
Acquire/release pattern with automatic state reset between requests.
Auto-registers hooks based on OrchestratorConfig.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from strands_orchestrator.config import OrchestratorConfig
from strands_orchestrator.container import AgentContainer
from strands_orchestrator.factory import AgentFactory
from strands_orchestrator.hooks import ConsentHook, EventBridgeHook, InboxHook, InterruptHook
from strands_orchestrator.mcp_connector import MCPConnector
from strands_orchestrator.model_factory import ModelFactory
from strands_orchestrator.state import StateAdapter
from strands_orchestrator.types import MCPServerDefinition

logger = logging.getLogger(__name__)


class AgentPoolService:
    """Singleton service managing a pool of pre-warmed AgentContainers.

    Creates multiple AgentContainer instances at startup with live
    MCP connections. Requests acquire a container from the pool,
    use it, then release it back with state reset.
    """

    _instance: AgentPoolService | None = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self, config: OrchestratorConfig):
        self._config = config
        self._pool: asyncio.Queue[AgentContainer] | None = None
        self._pool_lock = asyncio.Lock()
        self._in_use: dict[int, AgentContainer] = {}
        self._mcp_connector: MCPConnector | None = None
        self._is_initialized = False
        self._is_shutting_down = False

    @property
    def mcp_connector(self) -> MCPConnector | None:
        """Access the underlying MCP connector for prompt/resource calls."""
        return self._mcp_connector

    @classmethod
    async def create(cls, config: OrchestratorConfig) -> AgentPoolService:
        """Create and initialize an AgentPoolService.

        This is the primary entry point. Validates config, connects
        to MCP servers, and warms the agent pool.
        """
        warnings = config.validate()
        for w in warnings:
            logger.warning("Config warning: %s", w)

        service = cls(config)
        await service.initialize()
        return service

    async def initialize(self) -> None:
        """Initialize MCP connections and warm the agent pool."""
        if self._is_initialized:
            return

        # 1. Set up MCP connections
        mcp_servers = await self._load_mcp_servers()
        if mcp_servers:
            self._mcp_connector = MCPConnector(mcp_servers)
            await self._mcp_connector.__aenter__()

        # 2. Create model factory
        model_factory = ModelFactory(
            custom_aliases=self._config.model_aliases or None
        )

        # 3. Create agent factory
        factory = AgentFactory(
            config_source=self._config.agent_config_source,
            model_factory=model_factory,
            mcp_connector=self._mcp_connector,
            enable_mode_filtering=self._config.enable_mode_filtering,
        )

        # 4. Warm the pool
        pool_size = self._config.pool_size
        self._pool = asyncio.Queue(maxsize=pool_size)

        logger.info("Warming agent pool (size=%d)...", pool_size)
        tasks = [self._create_and_enqueue(factory, i) for i in range(pool_size)]
        await asyncio.gather(*tasks)

        self._is_initialized = True
        logger.info("Agent pool initialized with %d containers", pool_size)

    async def _create_and_enqueue(
        self, factory: AgentFactory, index: int
    ) -> None:
        """Create a single AgentContainer and add to pool."""
        try:
            container = await factory.build()
            await self._pool.put(container)
            logger.info("Created pool container %d/%d", index + 1, self._config.pool_size)
        except Exception:
            logger.exception("Failed to create pool container %d", index + 1)

    async def _load_mcp_servers(self) -> list[MCPServerDefinition]:
        """Load MCP server definitions from config source or YAML."""
        servers = []

        # From inline config
        if self._config.mcp_servers:
            for s in self._config.mcp_servers:
                servers.append(MCPServerDefinition.model_validate(s))

        # From config source
        try:
            raw_servers = await self._config.agent_config_source.get_mcp_server_configs()
            for s in raw_servers:
                servers.append(MCPServerDefinition.model_validate(s))
        except Exception:
            logger.debug("Config source did not provide MCP server configs")

        # From YAML file
        if self._config.mcp_config_path:
            servers.extend(self._load_mcp_from_yaml(self._config.mcp_config_path))

        return servers

    @staticmethod
    def _load_mcp_from_yaml(path: str) -> list[MCPServerDefinition]:
        """Load MCP server configs from a YAML file."""
        import yaml

        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("MCP config file not found: %s", path)
            return []

        servers = []
        mcp_section = data.get("mcp", data).get("servers", data.get("servers", {}))
        for name, cfg in mcp_section.items():
            cfg["name"] = name
            # Infer transport
            if "url" in cfg and "transport" not in cfg:
                cfg["transport"] = "sse"
            elif "command" in cfg and "transport" not in cfg:
                cfg["transport"] = "stdio"
            servers.append(MCPServerDefinition.model_validate(cfg))

        return servers

    async def acquire(self) -> AgentContainer:
        """Acquire an AgentContainer from the pool.

        Blocks until a container is available.
        Caller MUST call release() when done.
        """
        if self._is_shutting_down or not self._pool:
            raise RuntimeError("Pool is shutting down or not initialized")

        container = await self._pool.get()
        async with self._pool_lock:
            self._in_use[id(container)] = container
        return container

    async def release(self, container: AgentContainer) -> None:
        """Release an AgentContainer back to the pool after state reset."""
        if self._is_shutting_down or not self._pool:
            return

        async with self._pool_lock:
            self._in_use.pop(id(container), None)

        await container.reset_state()
        await self._pool.put(container)

    @asynccontextmanager
    async def get_container(
        self,
        chat_id: str = "",
        agent_name: str = "",
        user: object | None = None,
        session_id: str = "",
        conversation_id: str = "",
        interrupt_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[AgentContainer, None]:
        """Context manager for acquiring a container with hooks auto-wired.

        Registers appropriate hooks based on OrchestratorConfig,
        then releases and resets the container on exit.
        """
        container = await self.acquire()

        # Register hooks based on config
        self._register_hooks(
            container,
            chat_id=chat_id,
            agent_name=agent_name,
            user=user,
            session_id=session_id,
            conversation_id=conversation_id,
            interrupt_event=interrupt_event,
        )

        try:
            yield container
        finally:
            await self.release(container)

    def _register_hooks(
        self,
        container: AgentContainer,
        chat_id: str = "",
        agent_name: str = "",
        user: object | None = None,
        session_id: str = "",
        conversation_id: str = "",
        interrupt_event: asyncio.Event | None = None,
    ) -> None:
        """Register hooks on all agents based on config."""
        for agent in container.agents.values():
            # Event bridge
            if self._config.event_bus and self._config.event_factory:
                hook = EventBridgeHook(
                    event_bus=self._config.event_bus,
                    event_factory=self._config.event_factory,
                    chat_id=chat_id,
                    agent_name=agent_name,
                    user=user,
                )
                hook.register_hooks(agent.hooks)

            # Consent
            if self._config.enable_consent and self._config.consent_service:
                hook = ConsentHook(
                    consent_service=self._config.consent_service,
                    auto_approve_tools=self._config.auto_approve_tools,
                    session_id=session_id,
                )
                hook.register_hooks(agent.hooks)

            # Background inbox
            if self._config.enable_background_tasks and self._config.background_inbox:
                user_tenant_id = None
                if user and hasattr(user, "tenant_id"):
                    user_tenant_id = user.tenant_id
                hook = InboxHook(
                    inbox_service=self._config.background_inbox,
                    conversation_id=conversation_id,
                    user_tenant_id=user_tenant_id,
                )
                hook.register_hooks(agent.hooks)

            # Interrupt
            if self._config.enable_interrupts:
                hook = InterruptHook(interrupt_event=interrupt_event)
                hook.register_hooks(agent.hooks)

    async def shutdown(self) -> None:
        """Shut down the pool and close MCP connections."""
        self._is_shutting_down = True

        # Drain pool
        if self._pool:
            while not self._pool.empty():
                try:
                    self._pool.get_nowait()
                except asyncio.QueueEmpty:
                    break

        # Close MCP connections
        if self._mcp_connector:
            await self._mcp_connector.__aexit__(None, None, None)

        self._is_initialized = False
        logger.info("Agent pool shut down")

    async def diagnostics(self) -> dict:
        """Return pool state snapshot."""
        if not self._pool:
            return {"status": "not_initialized"}

        async with self._pool_lock:
            in_use_count = len(self._in_use)

        return {
            "max_size": self._pool.maxsize,
            "available_count": self._pool.qsize(),
            "in_use_count": in_use_count,
            "is_initialized": self._is_initialized,
        }
