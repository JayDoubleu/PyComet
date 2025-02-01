import re
from typing import Any, Dict, Tuple

import pytest
from pytest import FixtureRequest

from pycomet.ai import AIClient
from pycomet.models import ModelUsage
from tests.utils import UsageMetricsCollector as TestMetrics


@pytest.mark.ai_models
class TestAIModels:
    """Test suite for different AI models"""

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "model_name,config_fixture",
        [
            ("gemini", "gemini_config"),
            ("anthropic", "anthropic_config"),
            ("openai", "openai_config"),
            ("groq", "groq_config"),
            ("azure", "azure_openai_config"),
            ("xai", "xai_config"),
            ("github", "github_config"),
            ("openrouter", "openrouter_config"),
        ],
    )
    @pytest.mark.parametrize(
        "diff_name,diff_fixture",
        [("logging", "sample_git_diff"), ("new_file", "hello_world_diff")],
    )
    def test_basic_response(
        self,
        model_name: str,
        config_fixture: str,
        diff_name: str,
        diff_fixture: str,
        request: FixtureRequest,
        test_metrics: TestMetrics,
    ) -> None:
        """Integration test for model's ability to generate commit messages."""
        config = request.getfixturevalue(config_fixture)
        git_diff = request.getfixturevalue(diff_fixture)
        message, _ = self._run_test(config, git_diff, test_metrics)

        assert message is not None, "Message should not be None"
        assert len(message) > 0, "Message should not be empty"
        assert isinstance(message, str), "Message should be a string"

        # Verify commit message format
        self._verify_commit_format(message)

    @pytest.mark.integration
    @pytest.mark.verbose_only
    @pytest.mark.parametrize(
        "model_name,config_fixture",
        [
            ("gemini", "gemini_config"),
            ("anthropic", "anthropic_config"),
            ("openai", "openai_config"),
            ("groq", "groq_config"),
            ("azure", "azure_openai_config"),
            ("xai", "xai_config"),
            ("github", "github_config"),
            ("openrouter", "openrouter_config"),
        ],
    )
    @pytest.mark.parametrize(
        "diff_name,diff_fixture",
        [("logging", "sample_git_diff"), ("new_file", "hello_world_diff")],
    )
    def test_verbose_output(
        self,
        model_name: str,
        config_fixture: str,
        diff_name: str,
        diff_fixture: str,
        request: FixtureRequest,
        test_metrics: TestMetrics,
        capsys: Any,
    ) -> None:
        """Integration test for model with verbose output."""
        config = request.getfixturevalue(config_fixture)
        git_diff = request.getfixturevalue(diff_fixture)

        print("\n" + "=" * 80)
        print(f"🤖 Test Case: {diff_name}")
        print(f"🔧 Model: {model_name}")
        print("-" * 80)

        message, usage = self._run_test(config, git_diff, test_metrics, verbose=True)

        # Capture and display verbose output
        captured = capsys.readouterr()

        print("\n📝 Generated Commit Message:")
        print("=" * 40)
        print(f"\n{message}")
        print("\n" + "=" * 40)
        print("\n🔍 Model Details:")
        print(captured.out.strip())

        # Update usage display to handle None values more explicitly
        input_tokens = usage.input_tokens or 0
        output_tokens = usage.output_tokens or 0
        cost = usage.total_cost if usage.total_cost is not None else 0.0

        print("\n💰 Usage:")
        print("=" * 40)
        print(f"Input Tokens: {input_tokens}")
        print(f"Output Tokens: {output_tokens}")
        print(f"Total Tokens: {input_tokens + output_tokens}")
        if cost > 0:
            print(f"Cost: ${cost:.4f}")
        else:
            print("Cost: Not available")
        print("\n" + "=" * 40)
        print("\n" + "=" * 80 + "\n")

        assert message is not None and len(message) > 0

    def _run_test(
        self,
        config: Dict[str, Any],
        git_diff: str,
        test_metrics: TestMetrics,
        verbose: bool = False,
    ) -> Tuple[str, ModelUsage]:
        """Helper method to run tests and collect metrics"""
        client = AIClient(config, verbose=verbose)
        message = client.generate_commit_message(git_diff)

        usage = client.last_usage if hasattr(client, "last_usage") else ModelUsage()
        if usage is None:
            usage = ModelUsage()

        test_metrics.add_usage(
            model=config["ai"]["model"],
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cost=usage.total_cost if usage.total_cost is not None else 0.0,
            char_count=len(message),
        )

        return message, usage

    def _verify_commit_format(self, message: str) -> None:
        """Helper method to verify commit message format"""
        commit_types = ["feat", "fix", "docs", "style", "refactor", "test", "chore"]
        pattern = r"\b(?:" + "|".join(commit_types) + r")\b"
        match = re.search(pattern, message, re.IGNORECASE)
        found_type = match.group(0).lower() if match else ""

        assert (
            found_type in commit_types
        ), f"Expected commit type not found. Found '{found_type}' instead."


class TestAIClientUnit:
    """Unit tests for AIClient that don't require API access"""

    def test_client_initialization(self) -> None:
        """Test that client initializes with valid config"""
        config = {
            "ai": {
                "provider": "test",
                "model": "test-model",
                "api_key": "dummy-key",
            },
            "commit": {"include_emoji": False},
        }
        client = AIClient(config)
        assert client is not None
        # The client stores only the "ai" portion of the config
        assert client.config == config["ai"]

    def test_invalid_config(self) -> None:
        """Test that client raises error with invalid config"""
        # Test completely empty config
        with pytest.raises(KeyError):
            AIClient({})

        # Test config with empty ai section
        with pytest.raises(KeyError):
            AIClient({"ai": {}})

    def test_missing_api_key(self) -> None:
        """Test that client raises error with missing API key"""
        config = {
            "ai": {
                "provider": "test",
                "model": "test-model",
            }
        }
        with pytest.raises(KeyError):
            AIClient(config)
