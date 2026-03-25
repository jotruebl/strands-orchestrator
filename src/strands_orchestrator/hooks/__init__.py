"""Hook implementations for strands-orchestrator.

Hooks integrate with the Strands HookProvider system to add
cross-cutting behavior: event streaming, consent checking,
background task management, and interruption.
"""

from strands_orchestrator.hooks.auth_token import AuthTokenInjectorHook
from strands_orchestrator.hooks.consent_hook import ConsentHook
from strands_orchestrator.hooks.event_bridge import EventBridgeHook
from strands_orchestrator.hooks.inbox_hook import InboxHook
from strands_orchestrator.hooks.interrupt_hook import InterruptHook

__all__ = [
    "AuthTokenInjectorHook",
    "ConsentHook",
    "EventBridgeHook",
    "InboxHook",
    "InterruptHook",
]
