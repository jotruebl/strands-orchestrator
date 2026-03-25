"""Tests for protocol compliance — verify mock implementations satisfy protocols."""

from strands_orchestrator.protocols import (
    AgentConfigSourceProtocol,
    BackgroundTaskInboxProtocol,
    ConsentServiceProtocol,
    EventBusProtocol,
    StreamEventFactoryProtocol,
    UserContextProtocol,
)
from tests.conftest import (
    MockAgentConfigSource,
    MockBackgroundInbox,
    MockConsentService,
    MockEventBus,
    MockStreamEventFactory,
    MockUserContext,
)


def test_event_bus_protocol():
    bus = MockEventBus()
    assert isinstance(bus, EventBusProtocol)


def test_background_inbox_protocol():
    inbox = MockBackgroundInbox()
    assert isinstance(inbox, BackgroundTaskInboxProtocol)


def test_consent_service_protocol():
    consent = MockConsentService()
    assert isinstance(consent, ConsentServiceProtocol)


def test_user_context_protocol():
    user = MockUserContext()
    assert isinstance(user, UserContextProtocol)
    assert user.user_id == "test-user"
    assert user.tenant_id == 1
    assert user.auth_token == "test-token"


def test_stream_event_factory_protocol():
    factory = MockStreamEventFactory()
    assert isinstance(factory, StreamEventFactoryProtocol)


def test_agent_config_source_protocol():
    source = MockAgentConfigSource()
    assert isinstance(source, AgentConfigSourceProtocol)
