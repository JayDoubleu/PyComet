from typing import Any, Dict, Tuple

import pytest
from pytest import FixtureRequest

from pycomet.ai import AIClient
from pycomet.models import ModelUsage
from tests.utils import UsageMetricsCollector as TestMetrics


@pytest.mark.ai_models
class TestAIModels:
    """Test suite for different AI models"""

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

        # Get usage from the client's response and handle None values
        usage = client.last_usage if hasattr(client, "last_usage") else ModelUsage()
        if usage is None:
            usage = ModelUsage()

        # Track metrics - ensure all values are numbers, not None
        test_metrics.add_usage(
            model=config["ai"]["model"],
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cost=usage.total_cost if usage.total_cost is not None else 0.0,
            char_count=len(message),
        )

        return message, usage

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
        """Test model's ability to generate commit messages."""
        config = request.getfixturevalue(config_fixture)
        git_diff = request.getfixturevalue(diff_fixture)
        message, _ = self._run_test(config, git_diff, test_metrics)

        assert message is not None, "Message should not be None"
        assert len(message) > 0, "Message should not be empty"
        assert isinstance(message, str), "Message should be a string"
        assert any(
            t in message
            for t in ["feat", "fix", "docs", "style", "refactor", "test", "chore"]
        ), "Message should use conventional commit format"

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
        """Test model with verbose output."""
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
