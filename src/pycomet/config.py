import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel


class AIConfig(BaseModel):
    provider: str
    model: str
    api_key: str
    api_base: Optional[str] = None
    api_version: Optional[str] = None


class CommitConfig(BaseModel):
    editor: Optional[str] = None
    include_emoji: bool = True
    custom_prompt: Optional[str] = None
    detailed: bool = False


class PyCometConfig(BaseModel):
    ai: AIConfig
    commit: Optional[CommitConfig] = None


# Use yaml.safe_dump to create a literal block style string
DEFAULT_SYSTEM_PROMPT = """You are a Git commit message generator. Your task is to \
create clear, concise, and meaningful commit messages.

Guidelines:
1. Keep messages under 72 characters
2. Use the Conventional Commits format: type(scope): description
3. Include a relevant emoji at the start
4. Be direct and descriptive
5. Focus on the "what" and "why" of the changes
6. Do not include any explanatory text or metadata
7. Return ONLY the commit message, nothing else

Common types:
- feat: new features
- fix: bug fixes
- docs: documentation changes
- style: formatting, missing semicolons, etc.
- refactor: code restructuring
- test: adding tests
- chore: maintenance tasks

Example good messages:
✨ feat(auth): add OAuth2 authentication
🐛 fix(api): handle null response from server
📚 docs(readme): update installation steps"""


class literal_str(str):
    pass


def literal_presenter(dumper: yaml.Dumper, data: literal_str) -> yaml.Node:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(literal_str, literal_presenter)

DEFAULT_CONFIG = {
    "ai": {
        "provider": "anthropic",
        "model": "claude-3-sonnet-20240229",
        "api_key": "your-api-key",
    },
    "commit": {
        "editor": os.environ.get("EDITOR", "nano"),
        "include_emoji": True,
        "custom_prompt": None,
        "detailed": False,
    },
}


def get_config_dir() -> Path:
    """Get the config directory path, creating it if it doesn't exist."""
    config_dir = Path.home() / ".config" / "pycomet"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_path() -> Path:
    """Get the full path to the config file."""
    return get_config_dir() / "config.yaml"


def load_config() -> Dict[str, Any]:
    """Load configuration, creating default if it doesn't exist."""
    config_path = get_config_path()

    if not config_path.exists():
        save_config(DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG.copy())  # Explicitly cast to Dict[str, Any]

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
        # Validate config with Pydantic
        PyCometConfig(**config_data)
        return dict(config_data)  # Explicitly cast to Dict[str, Any]


def save_config(config: Dict[str, Any]) -> None:
    """Save configuration to file."""
    config_path = get_config_path()

    # Validate config before saving
    PyCometConfig(**config)

    # Convert system_prompt to literal style if it exists
    if "system_prompt" in config["ai"]:
        config["ai"]["system_prompt"] = literal_str(config["ai"]["system_prompt"])

    with open(config_path, "w") as f:
        yaml.dump(
            config,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            indent=2,
        )
