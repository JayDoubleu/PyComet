import os
from typing import Any, Dict, Generator, List

import pytest
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.nodes import Item
from dotenv import load_dotenv

from tests.utils import UsageMetricsCollector as TestMetrics


def pytest_addoption(parser: Parser) -> None:
    """Add verbose option to pytest"""
    parser.addoption(
        "--show-llm-output",
        action="store_true",
        default=False,
        help="show detailed LLM responses and processing details",
    )


def pytest_configure(config: Config) -> None:
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "verbose_only: mark test that shows detailed LLM output"
    )
    config.addinivalue_line(
        "markers", "ai_models: mark tests that interact with AI model APIs"
    )
    config.addinivalue_line(
        "markers", "integration: mark tests that require API access"
    )


def pytest_collection_modifyitems(config: Config, items: List[Item]) -> None:
    """Skip tests based on available API keys and --show-llm-output flag"""
    # Skip verbose tests if flag not set
    if not config.getoption("--show-llm-output"):
        skip_verbose = pytest.mark.skip(reason="need --show-llm-output option to run")
        for item in items:
            if "verbose_only" in item.keywords:
                item.add_marker(skip_verbose)

    # Skip tests if API keys not available
    provider_env_map = {
        "test_anthropic": "TEST_ANTHROPIC_API_KEY",
        "test_openai": "TEST_OPENAI_API_KEY",
        "test_gemini": "TEST_GEMINI_API_KEY",
        "test_azure": "TEST_AZURE_API_KEY",
        "test_groq": "TEST_GROQ_API_KEY",
        "test_xai": "TEST_XAI_API_KEY",
        "test_github": "TEST_GITHUB_API_KEY",
        "test_openrouter": "TEST_OPENROUTER_API_KEY",
    }

    for item in items:
        for provider, env_var in provider_env_map.items():
            if provider in item.name and not os.getenv(env_var):
                item.add_marker(pytest.mark.skip(reason=f"no {env_var} available"))


@pytest.fixture(scope="session", autouse=True)
def load_env() -> None:
    """Load environment variables from .env.test file"""
    load_dotenv(".env.test")


@pytest.fixture
def gemini_config() -> Dict[str, Any]:
    """Test configuration for Gemini"""
    return {
        "ai": {
            "provider": "gemini",
            "model": "gemini/gemini-1.5-pro",
            "api_key": os.getenv("TEST_GEMINI_API_KEY"),
        },
        "commit": {"include_emoji": False},
    }


@pytest.fixture
def anthropic_config() -> Dict[str, Any]:
    """Test configuration for Anthropic"""
    return {
        "ai": {
            "provider": "anthropic",
            "model": "claude-3-sonnet-20240229",
            "api_key": os.getenv("TEST_ANTHROPIC_API_KEY"),
        },
        "commit": {"include_emoji": False},
    }


@pytest.fixture
def openai_config() -> Dict[str, Any]:
    """Test configuration for OpenAI"""
    return {
        "ai": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": os.getenv("TEST_OPENAI_API_KEY"),
        },
        "commit": {"include_emoji": False},
    }


@pytest.fixture
def groq_config() -> Dict[str, Any]:
    """Test configuration for Groq"""
    return {
        "ai": {
            "provider": "groq",
            "model": "groq/llama-3.3-70b-versatile",
            "api_key": os.getenv("TEST_GROQ_API_KEY"),
        },
        "commit": {"include_emoji": False},
    }


@pytest.fixture
def azure_openai_config() -> Dict[str, Any]:
    """Test configuration for Azure OpenAI"""
    return {
        "ai": {
            "provider": "azure",
            "model": "azure/gpt-4o-mini",
            "api_key": os.getenv("TEST_AZURE_API_KEY"),
            "api_base": os.getenv("TEST_AZURE_API_BASE"),
            "api_version": os.getenv("TEST_AZURE_API_VERSION", "2024-02-15-preview"),
            "deployment_name": os.getenv("TEST_AZURE_DEPLOYMENT_NAME"),
        },
        "commit": {"include_emoji": False},
    }


@pytest.fixture
def xai_config() -> Dict[str, Any]:
    """Test configuration for XAI (Grok-2)"""
    return {
        "ai": {
            "provider": "xai",
            "model": "xai/grok-2",
            "api_key": os.getenv("TEST_XAI_API_KEY"),
        },
        "commit": {"include_emoji": False},
    }


@pytest.fixture
def github_config() -> Dict[str, Any]:
    """Test configuration for GitHub (Phi-3.5)"""
    return {
        "ai": {
            "provider": "github",
            "model": "github/phi-3.5-mini-instruct",
            "api_key": os.getenv("TEST_GITHUB_API_KEY"),
        },
        "commit": {"include_emoji": False},
    }


@pytest.fixture
def openrouter_config() -> Dict[str, Any]:
    """Test configuration for OpenRouter"""
    return {
        "ai": {
            "provider": "openrouter",
            "model": "openrouter/mistralai/mistral-small-24b-instruct-2501",
            "api_key": os.getenv("TEST_OPENROUTER_API_KEY"),
            "api_base": os.getenv(
                "TEST_OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"
            ),
        },
        "commit": {"include_emoji": False},
    }


@pytest.fixture
def sample_git_diff() -> str:
    """Sample git diff for testing logging change"""
    return """diff --git a/src/main.py b/src/main.py
+import logging
-print("Hello")
+logging.info("Hello")
"""


@pytest.fixture
def hello_world_diff() -> str:
    """Sample git diff for testing new file addition"""
    return """diff --git a/hello_world.md b/hello_world.md
new file mode 100644
index 0000000..a042389
--- /dev/null
+++ b/hello_world.md
@@ -0,0 +1 @@
+hello world!
"""


@pytest.fixture(scope="session")
def test_metrics() -> Generator[TestMetrics, None, None]:
    """Fixture to track token usage and costs across all tests"""
    metrics = TestMetrics()
    yield metrics
    metrics.display_summary()
