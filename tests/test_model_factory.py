"""Tests for ModelFactory."""

from strands_orchestrator.model_factory import ModelFactory


class TestModelFactory:
    def test_sonnet_alias(self):
        factory = ModelFactory()
        model = factory.create("sonnet")
        assert model.config["model_id"] == "claude-sonnet-4-20250514"

    def test_gpt4o_alias(self):
        factory = ModelFactory()
        model = factory.create("gpt-4o")
        assert model.config["model_id"] == "gpt-4o"

    def test_custom_alias(self):
        factory = ModelFactory(custom_aliases={"my-model": "claude-3-haiku-20240307"})
        model = factory.create("my-model")
        assert model.config["model_id"] == "claude-3-haiku-20240307"

    def test_literal_model_id(self):
        factory = ModelFactory()
        model = factory.create("claude-sonnet-4-20250514")
        assert model.config["model_id"] == "claude-sonnet-4-20250514"

    def test_infer_openai_provider(self):
        assert ModelFactory._infer_provider("gpt-4o") == "openai"
        assert ModelFactory._infer_provider("o3-mini") == "openai"

    def test_infer_anthropic_provider(self):
        assert ModelFactory._infer_provider("claude-3-haiku") == "anthropic"
        assert ModelFactory._infer_provider("some-unknown") == "anthropic"
