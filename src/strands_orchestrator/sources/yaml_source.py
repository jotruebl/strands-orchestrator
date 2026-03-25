"""YAMLAgentConfigSource — loads agent configs from YAML files.

Expected directory structure:
    config_dir/
    ├── agents/
    │   ├── agent-name.yaml
    │   └── another-agent.yaml
    ├── modes/
    │   ├── mode-name.yaml
    │   └── another-mode.yaml
    └── servers/
        ├── server-name.yaml
        └── another-server.yaml

Or a single agents.yaml file with all configs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class YAMLAgentConfigSource:
    """Loads agent configurations from YAML files on disk.

    Supports two layouts:
    1. Directory mode: separate agents/, modes/, servers/ directories
    2. Single file mode: one YAML file with agents, modes, servers sections
    """

    def __init__(self, config_path: str):
        """Initialize with a path to a config directory or single YAML file."""
        self._path = Path(config_path)

    async def get_agent_configs(self) -> list[dict]:
        """Load all agent definitions."""
        if self._path.is_file():
            return self._load_from_single_file("agents")

        agents_dir = self._path / "agents"
        if not agents_dir.exists():
            logger.warning("No agents/ directory found at %s", self._path)
            return []

        configs = []
        for yaml_file in sorted(agents_dir.glob("*.yaml")):
            data = self._load_yaml(yaml_file)
            if data:
                # Support kubernagents format (kind: Agent, spec: ...)
                if "kind" in data and data.get("kind") == "Agent":
                    config = self._parse_kubernagents_agent(data)
                else:
                    config = data
                configs.append(config)

        return configs

    async def get_mode_configs(self, agent_name: str) -> dict[str, dict]:
        """Load mode configurations for a specific agent."""
        if self._path.is_file():
            all_modes = self._load_from_single_file("modes")
            # Filter modes for this agent
            return {
                m.get("name", ""): m
                for m in all_modes
                if m.get("agent") == agent_name or "agent" not in m
            }

        modes_dir = self._path / "modes"
        if not modes_dir.exists():
            return {}

        modes = {}
        for yaml_file in sorted(modes_dir.glob("*.yaml")):
            data = self._load_yaml(yaml_file)
            if data:
                name = data.get("name") or data.get("metadata", {}).get("name", yaml_file.stem)
                modes[name] = data

        return modes

    async def get_mcp_server_configs(self) -> list[dict]:
        """Load MCP server connection configurations."""
        if self._path.is_file():
            return self._load_from_single_file("servers")

        servers_dir = self._path / "servers"
        if not servers_dir.exists():
            return []

        configs = []
        for yaml_file in sorted(servers_dir.glob("*.yaml")):
            data = self._load_yaml(yaml_file)
            if data:
                if "name" not in data:
                    data["name"] = yaml_file.stem
                configs.append(data)

        return configs

    def _load_from_single_file(self, section: str) -> list[dict]:
        """Load a section from a single YAML file."""
        data = self._load_yaml(self._path)
        if not data:
            return []
        items = data.get(section, [])
        return items if isinstance(items, list) else [items]

    @staticmethod
    def _load_yaml(path: Path) -> dict | None:
        """Load and parse a YAML file."""
        try:
            with open(path) as f:
                return yaml.safe_load(f) or {}
        except Exception:
            logger.exception("Failed to load YAML: %s", path)
            return None

    @staticmethod
    def _parse_kubernagents_agent(data: dict) -> dict:
        """Parse kubernagents Agent resource format into AgentDefinition format."""
        metadata = data.get("metadata", {})
        spec = data.get("spec", {})

        return {
            "name": metadata.get("name", ""),
            "description": spec.get("description", ""),
            "system_prompt": spec.get("instruction", ""),
            "model": spec.get("llm", {}).get("model", "sonnet"),
            "max_iterations": spec.get("llm", {}).get("max_iterations", 25),
            "max_tokens": spec.get("llm", {}).get("max_tokens", 16384),
            "modes": [
                {"name": m.get("name", ""), "default": m.get("default", False)}
                for m in spec.get("modes", [])
            ],
            "default_mode": next(
                (m.get("name") for m in spec.get("modes", []) if m.get("default")),
                None,
            ),
        }
