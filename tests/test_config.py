"""Tests for OrchestratorConfig."""

from strands_orchestrator.config import OrchestratorConfig
from tests.conftest import (
    MockAgentConfigSource,
    MockConsentService,
    MockEventBus,
    MockStreamEventFactory,
    MockBackgroundInbox,
)


class TestOrchestratorConfig:
    def test_minimal_config(self):
        config = OrchestratorConfig(
            agent_config_source=MockAgentConfigSource()
        )
        assert config.pool_size == 1
        assert config.default_model == "sonnet"

    def test_validate_consent_warning(self):
        config = OrchestratorConfig(
            agent_config_source=MockAgentConfigSource(),
            enable_consent=True,
        )
        warnings = config.validate()
        assert any("consent_service" in w for w in warnings)

    def test_validate_background_warning(self):
        config = OrchestratorConfig(
            agent_config_source=MockAgentConfigSource(),
            enable_background_tasks=True,
        )
        warnings = config.validate()
        assert any("background_inbox" in w for w in warnings)

    def test_validate_event_bus_without_factory(self):
        config = OrchestratorConfig(
            agent_config_source=MockAgentConfigSource(),
            event_bus=MockEventBus(),
        )
        warnings = config.validate()
        assert any("event_factory" in w for w in warnings)

    def test_validate_no_warnings(self):
        config = OrchestratorConfig(
            agent_config_source=MockAgentConfigSource(),
            event_bus=MockEventBus(),
            event_factory=MockStreamEventFactory(),
            consent_service=MockConsentService(),
            enable_consent=True,
            background_inbox=MockBackgroundInbox(),
            enable_background_tasks=True,
        )
        warnings = config.validate()
        assert len(warnings) == 0
