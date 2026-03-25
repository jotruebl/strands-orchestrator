"""MongoDBAgentConfigSource — loads agent configs from MongoDB.

Compatible with the kubernagents resource format used by
Fusion Platform's agent configuration system.

Requires the 'mongodb' extra: pip install strands-orchestrator[mongodb]
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MongoDBAgentConfigSource:
    """Loads agent configurations from MongoDB (kubernagents format).

    Reads from MongoDB collections following the kubernagents resource
    hierarchy: agent_collections → agents → agent_modes → mcp_servers → tools.

    Usage:
        source = MongoDBAgentConfigSource(
            mongo_url="mongodb://localhost:27017",
            db_name="fusion_ai",
            collection_name="Multi-Agent System",
        )
        configs = await source.get_agent_configs()
    """

    def __init__(
        self,
        mongo_url: str = "mongodb://localhost:27017",
        db_name: str = "fusion_ai",
        collection_name: str = "Multi-Agent System",
    ):
        self._mongo_url = mongo_url
        self._db_name = db_name
        self._collection_name = collection_name
        self._client: Any = None
        self._db: Any = None

    async def _ensure_connection(self) -> None:
        """Lazily connect to MongoDB."""
        if self._client is None:
            try:
                from motor.motor_asyncio import AsyncIOMotorClient
            except ImportError:
                raise ImportError(
                    "motor is required for MongoDB support. "
                    "Install with: pip install strands-orchestrator[mongodb]"
                )
            self._client = AsyncIOMotorClient(self._mongo_url)
            self._db = self._client[self._db_name]

    async def get_agent_configs(self) -> list[dict]:
        """Load all agent definitions from the active agent collection."""
        await self._ensure_connection()

        # Find the active agent collection
        collection_doc = await self._db.agent_collections.find_one(
            {
                "metadata.name": self._collection_name,
                "metadata.active": True,
            },
            sort=[("metadata.version", -1)],
        )

        if not collection_doc:
            logger.warning(
                "Agent collection '%s' not found in MongoDB", self._collection_name
            )
            return []

        # Resolve agent references
        agent_refs = collection_doc.get("spec", {}).get("agents", [])
        configs = []

        for ref in agent_refs:
            agent_name = ref.get("name", "")
            agent_doc = await self._db.agents.find_one(
                {"metadata.name": agent_name, "metadata.active": True},
                sort=[("metadata.version", -1)],
            )

            if agent_doc:
                config = await self._parse_agent_doc(agent_doc)
                configs.append(config)
            else:
                logger.warning("Agent '%s' not found in MongoDB", agent_name)

        return configs

    async def get_mode_configs(self, agent_name: str) -> dict[str, dict]:
        """Load mode configurations for a specific agent."""
        await self._ensure_connection()

        # Get the agent doc to find mode references
        agent_doc = await self._db.agents.find_one(
            {"metadata.name": agent_name, "metadata.active": True},
            sort=[("metadata.version", -1)],
        )

        if not agent_doc:
            return {}

        mode_refs = agent_doc.get("spec", {}).get("modes", [])
        modes = {}

        for ref in mode_refs:
            mode_name = ref.get("name", "")
            mode_doc = await self._db.agent_modes.find_one(
                {"metadata.name": mode_name, "metadata.active": True},
                sort=[("metadata.version", -1)],
            )

            if mode_doc:
                spec = mode_doc.get("spec", {})
                modes[mode_name] = {
                    "description": spec.get("description", ""),
                    "instructions": spec.get("instruction", ""),
                    "servers": self._parse_server_refs(spec.get("servers", [])),
                }

        return modes

    async def get_mcp_server_configs(self) -> list[dict]:
        """Load MCP server connection configurations."""
        await self._ensure_connection()

        cursor = self._db.mcp_servers.find(
            {"metadata.active": True},
            sort=[("metadata.version", -1)],
        )

        configs = []
        seen_names = set()

        async for doc in cursor:
            name = doc.get("metadata", {}).get("name", "")
            if name in seen_names:
                continue
            seen_names.add(name)

            spec = doc.get("spec", {})
            config = {
                "name": name,
                "transport": spec.get("transport", "sse"),
                "url": spec.get("url"),
                "command": spec.get("command"),
                "args": spec.get("args", []),
                "env": spec.get("env", {}),
            }
            configs.append(config)

        return configs

    async def _parse_agent_doc(self, doc: dict) -> dict:
        """Parse a MongoDB agent document into AgentDefinition format."""
        metadata = doc.get("metadata", {})
        spec = doc.get("spec", {})

        # Resolve content references for description/instruction
        description = await self._resolve_content(spec.get("description", ""))
        instruction = await self._resolve_content(spec.get("instruction", ""))

        llm_config = spec.get("llm", {})

        return {
            "name": metadata.get("name", ""),
            "description": description,
            "system_prompt": instruction,
            "model": llm_config.get("model", "sonnet"),
            "max_iterations": llm_config.get("max_iterations", 25),
            "max_tokens": llm_config.get("max_tokens", 16384),
            "modes": [
                {"name": m.get("name", ""), "default": m.get("default", False)}
                for m in spec.get("modes", [])
            ],
            "default_mode": next(
                (m.get("name") for m in spec.get("modes", []) if m.get("default")),
                None,
            ),
        }

    async def _resolve_content(self, content_ref: Any) -> str:
        """Resolve a content reference (string or list of refs) to text."""
        if isinstance(content_ref, str):
            return content_ref

        if isinstance(content_ref, list):
            parts = []
            for ref in content_ref:
                if isinstance(ref, str):
                    parts.append(ref)
                elif isinstance(ref, dict):
                    name = ref.get("name", "")
                    version = ref.get("version")
                    content_doc = await self._db.contents.find_one(
                        {"metadata.name": name, "metadata.active": True},
                        sort=[("metadata.version", -1)],
                    )
                    if content_doc:
                        parts.append(content_doc.get("spec", {}).get("text", ""))
            return "\n\n".join(parts)

        return str(content_ref) if content_ref else ""

    @staticmethod
    def _parse_server_refs(server_refs: list) -> dict[str, list[str]]:
        """Parse server references from mode spec into {server: [tools]} format."""
        servers = {}
        for ref in server_refs:
            if isinstance(ref, dict):
                name = ref.get("name", "")
                tools = ref.get("tools", ["*"])
                servers[name] = tools
            elif isinstance(ref, str):
                servers[ref] = ["*"]
        return servers
