# strands-orchestrator

Protocol-based DI framework for [Strands Agents SDK](https://strandsagents.com/). Adds agent pooling, MCP server management, mode-aware tool filtering, session state persistence, and extensible hooks for event streaming, consent, and background tasks.

## Install

```bash
uv add strands-orchestrator
# or with MongoDB support:
uv add "strands-orchestrator[mongodb]"
```

## Quick Start

```python
from strands_orchestrator import AgentPoolService, OrchestratorConfig
from strands_orchestrator.sources import YAMLAgentConfigSource

config = OrchestratorConfig(
    agent_config_source=YAMLAgentConfigSource("./config"),
    pool_size=2,
    default_model="sonnet",
)

pool = await AgentPoolService.create(config)

async with pool.get_container() as container:
    result = container["my-agent"]("What is 2+2?")
    print(result)
```

## FastAPI Integration

Full example with Redis (session cache + distributed locking) and MongoDB (durable persistence). Agent pool warms at startup, sessions are locked per-request to prevent concurrent writes, state is restored from cache/DB, and agents are reset and returned to pool on exit.

```python
import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from redis import asyncio as aioredis

from strands_orchestrator import AgentPoolService, OrchestratorConfig, StateAdapter
from strands_orchestrator.sources.mongodb import MongoDBAgentConfigSource


# --- Session persistence: Redis cache + MongoDB durable store ---

class SessionManager:
    """Two-tier session persistence with distributed locking.

    Redis: fast cache for active sessions + distributed lock per session.
    MongoDB: durable store for sessions that survive Redis eviction.
    """

    LOCK_TTL = 30  # seconds
    CACHE_TTL = 3600  # 1 hour

    def __init__(self, redis: aioredis.Redis, mongo_db):
        self.redis = redis
        self.sessions = mongo_db.sessions  # MongoDB collection

    async def acquire_lock(self, session_id: str) -> str:
        """Acquire a distributed lock for a session. Returns lock token."""
        token = str(uuid.uuid4())
        acquired = await self.redis.set(
            f"lock:{session_id}", token, nx=True, ex=self.LOCK_TTL
        )
        if not acquired:
            raise RuntimeError(f"Session {session_id} is locked by another request")
        return token

    async def release_lock(self, session_id: str, token: str) -> None:
        """Release lock only if we still own it (compare token)."""
        current = await self.redis.get(f"lock:{session_id}")
        if current and current.decode() == token:
            await self.redis.delete(f"lock:{session_id}")

    async def load_state(self, session_id: str) -> dict | None:
        """Load from Redis cache first, fall back to MongoDB."""
        # Try Redis cache
        cached = await self.redis.get(f"session:{session_id}")
        if cached:
            return json.loads(cached)

        # Fall back to MongoDB
        doc = await self.sessions.find_one({"_id": session_id})
        if doc:
            state = doc["state"]
            # Re-populate cache
            await self.redis.setex(
                f"session:{session_id}", self.CACHE_TTL, json.dumps(state)
            )
            return state

        return None

    async def save_state(self, session_id: str, state: dict) -> None:
        """Write to both Redis cache and MongoDB."""
        state_json = json.dumps(state)

        # Redis cache
        await self.redis.setex(f"session:{session_id}", self.CACHE_TTL, state_json)

        # MongoDB durable store (upsert)
        await self.sessions.update_one(
            {"_id": session_id},
            {"$set": {"state": state}},
            upsert=True,
        )


# --- App setup ---

pool_service: AgentPoolService | None = None
session_manager: SessionManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool_service, session_manager

    # 1. Connect to Redis and MongoDB
    redis = await aioredis.from_url("redis://localhost:6379")
    mongo = AsyncIOMotorClient("mongodb://localhost:27017")
    db = mongo["my_app"]

    session_manager = SessionManager(redis, db)

    # 2. Initialize agent pool (agents loaded from MongoDB)
    config = OrchestratorConfig(
        agent_config_source=MongoDBAgentConfigSource(
            mongo_url="mongodb://localhost:27017",
            db_name="my_app",
            collection_name="Multi-Agent System",
        ),
        pool_size=3,
        default_model="sonnet",
    )
    pool_service = await AgentPoolService.create(config)

    yield

    # 3. Shutdown
    await pool_service.shutdown()
    await redis.close()
    mongo.close()


app = FastAPI(lifespan=lifespan)


# --- Chat endpoint ---

class ChatRequest(BaseModel):
    message: str
    session_id: str
    agent_name: str = "assistant"


@app.post("/chat")
async def chat(req: ChatRequest):
    # 1. Acquire distributed lock for this session
    lock_token = await session_manager.acquire_lock(req.session_id)

    try:
        # 2. Acquire agent container from pool
        async with pool_service.get_container() as container:
            agent = container[req.agent_name]

            # 3. Load previous conversation (Redis cache → MongoDB fallback)
            saved_state = await session_manager.load_state(req.session_id)
            if saved_state:
                await StateAdapter.restore(container, saved_state)

            # 4. Run the agent (conversation history is already loaded)
            result = agent(req.message)

            # 5. Extract state and persist to Redis + MongoDB
            state = StateAdapter.extract(container)
            await session_manager.save_state(req.session_id, state)

            return {"response": str(result)}

        # Container is automatically reset and returned to pool on context exit

    finally:
        # 6. Always release the session lock
        await session_manager.release_lock(req.session_id, lock_token)
```

**What happens on each request:**

1. **Lock** -- Redis `SET NX` prevents concurrent writes to the same session
2. **Acquire** -- an `AgentContainer` is pulled from the pre-warmed pool (blocks if all in use)
3. **Restore** -- `StateAdapter.restore()` resets the agent first (clearing any prior state from the pool), then loads conversation history from Redis/MongoDB
4. **Execute** -- the agent runs with full conversation context
5. **Persist** -- `StateAdapter.extract()` captures messages + state, writes to both Redis (cache) and MongoDB (durable)
6. **Release** -- `get_container()` exit calls `reset_state()` on all agents, returns container to pool. Lock is released in `finally`

## Config Sources

**YAML** -- load from a directory of YAML files:

```
config/
  agents/
    assistant.yaml
    summarizer.yaml
  modes/
    default-mode.yaml
    research-mode.yaml
  servers/
    fusion.yaml
```

```python
source = YAMLAgentConfigSource("./config")
```

**MongoDB** -- load from kubernagents-format collections:

```python
from strands_orchestrator.sources.mongodb import MongoDBAgentConfigSource

source = MongoDBAgentConfigSource(
    mongo_url="mongodb://localhost:27017",
    db_name="fusion_ai",
    collection_name="Multi-Agent System",
)
```

## Hooks

Features activate automatically based on which protocols you provide in `OrchestratorConfig`:

| Protocol provided | Hook activated | What it does |
|---|---|---|
| `event_bus` + `event_factory` | `EventBridgeHook` | Publishes tool call / turn start / turn end events to your event bus |
| `consent_service` + `enable_consent` | `ConsentHook` | Gates tool execution behind user approval |
| `background_inbox` + `enable_background_tasks` | `InboxHook` | Injects background task results into conversations, auto-registers containers/tasks |
| `enable_interrupts` (default: True) | `InterruptHook` | Checks for user cancellation before each tool call |

## Protocols

Implement these interfaces to plug in your infrastructure:

```python
from strands_orchestrator.protocols import (
    EventBusProtocol,            # async publish(event, user)
    BackgroundTaskInboxProtocol, # pop_inbox, watch_container, watch_llm_task
    ConsentServiceProtocol,      # check_consent, request_consent
    SessionPersistenceProtocol,  # load_state, save_state
    UserContextProtocol,         # user_id, tenant_id, auth_token
    StreamEventFactoryProtocol,  # create_tool_start_event, create_turn_end_event, ...
    AgentConfigSourceProtocol,   # get_agent_configs, get_mode_configs, get_mcp_server_configs
)
```

All protocols are `@runtime_checkable` -- the framework validates at startup that your implementations satisfy the interface.

## Mode-Aware Tool Filtering

Agents can have multiple modes that control which tools are visible. The LLM gets a `switch_mode` tool and can change modes autonomously:

```yaml
# config/agents/assistant.yaml
name: assistant
system_prompt: "You are a helpful assistant."
model: sonnet
modes:
  - name: analysis
    description: "Spectral analysis and data processing"
    servers:
      fusion: ["analyze_spectrum", "run_model", "get_results"]
  - name: writing
    description: "Report writing and export"
    servers:
      fusion: ["create_report", "export_pdf"]
default_mode: analysis
```

## License

MIT
