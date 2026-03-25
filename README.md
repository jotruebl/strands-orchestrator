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
container = await pool.acquire()

agent = container.prepare_for_request(
    agent_name="my-agent",
    chat_id="conversation-123",
)
result = agent("What is 2+2?")
print(result)

await pool.release(container)
```

## How It Works

**Startup:** Declare your infrastructure in `OrchestratorConfig`. The pool creates pre-warmed `AgentContainer` instances, each carrying a reference to the config.

**Per request:** Call `container.prepare_for_request()` with request-scoped context (chat_id, user, interrupt_event). The container reads its config and automatically registers the right hooks — SSE events, auth token injection, consent gating, interrupts.

**You provide:** Only what changes per request. Everything else comes from config.

```
OrchestratorConfig (startup, once)
    ├── event_bus + event_factory  →  EventBridgeHook (auto)
    ├── consent_service            →  ConsentHook (auto)
    └── enable_interrupts          →  InterruptHook (auto)

prepare_for_request (per request)
    ├── chat_id        →  routes SSE events to the right conversation
    ├── user           →  role-based event filtering
    ├── interrupt_event →  user cancellation signal
    └── main_loop      →  cross-thread event publishing
```

## FastAPI Integration

Full example with Redis (session cache + distributed locking) and MongoDB (durable persistence).

```python
import asyncio
import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from redis import asyncio as aioredis

from strands_orchestrator import AgentPoolService, OrchestratorConfig, StateAdapter
from strands_orchestrator.sources.mongodb import MongoDBAgentConfigSource


# --- Session persistence ---

class SessionManager:
    LOCK_TTL = 30
    CACHE_TTL = 3600

    def __init__(self, redis: aioredis.Redis, mongo_db):
        self.redis = redis
        self.sessions = mongo_db.sessions

    async def acquire_lock(self, session_id: str) -> str:
        token = str(uuid.uuid4())
        acquired = await self.redis.set(
            f"lock:{session_id}", token, nx=True, ex=self.LOCK_TTL
        )
        if not acquired:
            raise RuntimeError(f"Session {session_id} is locked")
        return token

    async def release_lock(self, session_id: str, token: str) -> None:
        current = await self.redis.get(f"lock:{session_id}")
        if current and current.decode() == token:
            await self.redis.delete(f"lock:{session_id}")

    async def load_state(self, session_id: str) -> dict | None:
        cached = await self.redis.get(f"session:{session_id}")
        if cached:
            return json.loads(cached)
        doc = await self.sessions.find_one({"_id": session_id})
        if doc:
            state = doc["state"]
            await self.redis.setex(
                f"session:{session_id}", self.CACHE_TTL, json.dumps(state)
            )
            return state
        return None

    async def save_state(self, session_id: str, state: dict) -> None:
        state_json = json.dumps(state)
        await self.redis.setex(f"session:{session_id}", self.CACHE_TTL, state_json)
        await self.sessions.update_one(
            {"_id": session_id}, {"$set": {"state": state}}, upsert=True
        )


# --- App setup ---

pool_service: AgentPoolService | None = None
session_manager: SessionManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool_service, session_manager

    redis = await aioredis.from_url("redis://localhost:6379")
    mongo = AsyncIOMotorClient("mongodb://localhost:27017")
    db = mongo["my_app"]
    session_manager = SessionManager(redis, db)

    # Declare infrastructure once — config flows to every container
    config = OrchestratorConfig(
        agent_config_source=MongoDBAgentConfigSource(
            mongo_url="mongodb://localhost:27017",
            db_name="my_app",
            collection_name="Multi-Agent System",
        ),
        event_bus=my_event_bus,          # your EventBusProtocol impl
        event_factory=my_event_factory,  # your StreamEventFactoryProtocol impl
        consent_service=my_consent_svc,  # your ConsentServiceProtocol impl
        enable_consent=True,
        pool_size=3,
        default_model="sonnet",
    )
    pool_service = await AgentPoolService.create(config)

    yield

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
    lock_token = await session_manager.acquire_lock(req.session_id)

    try:
        # 1. Acquire container from pool
        container = await pool_service.acquire()

        try:
            # 2. Restore conversation history
            saved_state = await session_manager.load_state(req.session_id)
            if saved_state:
                await StateAdapter.restore(container, saved_state)

            # 3. Prepare agent — hooks registered automatically from config
            agent = container.prepare_for_request(
                agent_name=req.agent_name,
                chat_id=req.session_id,
                user=current_user,
                main_loop=asyncio.get_running_loop(),
            )

            # 4. Set request-scoped state
            agent.state.set("mcp_custom_auth_token", current_user.auth_token)

            # 5. Run agent in thread (main loop stays free for SSE events)
            result = await asyncio.to_thread(agent, req.message)

            # 6. Persist state
            state = StateAdapter.extract(container)
            await session_manager.save_state(req.session_id, state)

            return {"response": str(result)}

        finally:
            await pool_service.release(container)

    finally:
        await session_manager.release_lock(req.session_id, lock_token)
```

**What happens on each request:**

1. **Lock** — Redis `SET NX` prevents concurrent writes to the same session
2. **Acquire** — an `AgentContainer` is pulled from the pre-warmed pool
3. **Restore** — `StateAdapter.restore()` loads conversation history
4. **Prepare** — `prepare_for_request()` reads config and registers hooks (EventBridge, AuthToken, Consent, Interrupt)
5. **Execute** — agent runs in a thread; hooks publish SSE events via `run_coroutine_threadsafe` back to the main loop
6. **Persist** — `StateAdapter.extract()` captures messages + state
7. **Release** — container is reset and returned to pool

## Config Sources

**YAML** — load from a directory of YAML files:

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

**MongoDB** — load from kubernagents-format collections:

```python
from strands_orchestrator.sources.mongodb import MongoDBAgentConfigSource

source = MongoDBAgentConfigSource(
    mongo_url="mongodb://localhost:27017",
    db_name="fusion_ai",
    collection_name="Multi-Agent System",
)
```

## Hooks

Hooks are registered automatically by `prepare_for_request()` based on what's in your config:

| Config field | Hook | What it does |
|---|---|---|
| `event_bus` + `event_factory` | `EventBridgeHook` | Publishes AGENT_TURN_START/END, TOOL_CALL_START/END, AGENT_REASONING_STEP via SSE |
| `consent_service` + `enable_consent` | `ConsentHook` | Gates tool execution behind user approval |
| (always) | `AuthTokenInjectorHook` | Injects auth token from agent.state into tool call args |
| `interrupt_event` (per-request) | `InterruptHook` | Checks for user cancellation before each tool call |

### EventBridgeHook Events

| Strands Hook Event | SSE Event Published |
|---|---|
| `BeforeInvocationEvent` | `AGENT_TURN_START` |
| `AfterModelCallEvent` | `AGENT_REASONING_STEP` (per-iteration model response) |
| `BeforeToolCallEvent` | `TOOL_CALL_START` |
| `AfterToolCallEvent` | `TOOL_CALL_END` |
| `AfterInvocationEvent` | `AGENT_TURN_END` (includes full response content) |

### Cross-Thread Publishing

Strands runs `agent()` in a `ThreadPoolExecutor`. Hooks fire in that thread, not on the main asyncio loop. `EventBridgeHook` uses `asyncio.run_coroutine_threadsafe(coro, main_loop)` to bridge events back. Pass `main_loop=asyncio.get_running_loop()` to `prepare_for_request()`.

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

All protocols are `@runtime_checkable` — the framework validates at startup that your implementations satisfy the interface.

## MCP Prompts

`MCPConnector` provides access to MCP server prompts in addition to tools:

```python
connector = pool_service.mcp_connector
result = connector.get_prompt(
    server_name="fusion",
    prompt_name="home_context",
    arguments={"context_string": "..."},
)
prompt_text = result.messages[0].content.text
```

Synchronous (`MCPClient.get_prompt_sync()`). Wrap in `asyncio.to_thread()` from async code.

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
