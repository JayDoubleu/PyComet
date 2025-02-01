# PyComet 🚀

PyComet is an AI-powered Git commit message generator that helps you create meaningful and consistent commit messages using advanced AI models.

## Installation & Usage Options

### Recommended: Global Installation with UV
Install PyComet globally with uv (recommended for better dependency management):
```bash
uv tool install pycomet-cli
```

### Alternative: Standard Installation (pip)
```bash
# Install from PyPI
pip install pycomet-cli

# Or with specific version
pip install pycomet-cli==0.1.2
```

### Quick Run with UV (No Installation)
Use `uvx` to run PyComet directly without installing:
```bash
uvx pycomet-cli
```
This is equivalent to `uv tool run pycomet-cli`

After installation (via any method), you can use these commands:
```bash
pycomet commit    # Generate and create a commit
pycomet-cli commit    # Alternative command name
```

> **Note**: Both command names (`pycomet` and `pycomet-cli`) are available and function identically.

## Quick Start

### Option 1: Install and Run (Recommended)
```bash
# Install globally (recommended)
uv tool install pycomet-cli
# Or via pip
pip install pycomet-cli

# Configure your AI provider
pycomet config

# Use in any git repository
git add .
pycomet preview  # Preview the message
pycomet commit   # Create commit
```

### Option 2: Run from Source
```bash
# Clone the repository
git clone https://github.com/jaydoubleu/pycomet.git
cd pycomet

# Install dependencies
uv sync

# Configure your AI provider
uv run pycomet config

# Use PyComet
git add .
uv run pycomet preview  # Preview the message
uv run pycomet commit   # Create commit
```

### Option 3: Direct Execution (No Clone)
```bash
# Run directly without installing
uvx pycomet-cli config
uvx pycomet-cli preview
uvx pycomet-cli commit
```

After configuration, edit `~/.config/pycomet/config.yaml` to add your API key and preferred settings.

## Features

- 🤖 **Smart Analysis**: Analyzes code changes to generate contextual commit messages
- 📝 **Conventional Commits**: Follows standard commit message format
- 😊 **Emoji Support**: Automatic emoji inclusion based on change type
- ⚙️ **Multiple AI Providers**: 
  - Anthropic Claude
  - OpenAI GPT-4
  - Google Gemini
  - Azure OpenAI
  - And more via litellm
- ✏️ **Interactive Editing**: Review and modify messages before committing
- 🔧 **Customizable**: Configure prompts, formats, and preferences
- 📊 **Usage Tracking**: Monitor token usage and costs
- 🚀 **Rate Limiting**: Automatic handling of API rate limits

## Basic Commands

```bash
# Create a commit with AI-generated message
uv run pycomet commit

# Preview message without committing
uv run pycomet preview

# Configure settings
uv run pycomet config
```

## Command Options

```bash
# Show detailed execution info
uv run pycomet commit --verbose
uv run pycomet preview --verbose

# Control emoji inclusion
uv run pycomet commit --emoji     # Force emoji
uv run pycomet commit --no-emoji  # Disable emoji

# Control message format
uv run pycomet commit --detailed    # Multi-line format
uv run pycomet commit --no-detailed # Single-line format

# Use custom prompt
uv run pycomet commit --prompt "$(cat my-prompt.txt)"

# Specify editor
uv run pycomet commit --editor vim
```

## Configuration

PyComet uses a YAML config file at `~/.config/pycomet/config.yaml`. For detailed configuration options and examples for all supported AI providers, see [CONFIGS.md](CONFIGS.md).

Basic configuration example:
```yaml
ai:
  provider: anthropic
  model: claude-3-sonnet-20240229
  api_key: your-api-key
commit:
  editor: nvim  # Your preferred editor
  include_emoji: true
  detailed: false  # Single-line or multi-line format
```

### Supported AI Providers

Here are some common provider configurations. For a complete list and detailed options, see [CONFIGS.md](CONFIGS.md).

#### Anthropic Claude (Default)
```yaml
ai:
  provider: anthropic
  model: claude-3-sonnet-20240229
  api_key: your-api-key
```

#### OpenAI
```yaml
ai:
  provider: openai
  model: gpt-4
  api_key: your-openai-key
```

#### Google Gemini
```yaml
ai:
  provider: gemini
  model: gemini-pro
  api_key: your-google-key
```

#### Azure OpenAI
```yaml
ai:
  provider: azure
  model: gpt-4
  api_key: your-azure-key
  api_base: your-azure-endpoint
  api_version: 2024-02-15-preview
```

## Message Formats

### Single-line (Default)
```
✨ feat(auth): add OAuth2 authentication
```

### Detailed (Multi-line)
```
✨ feat(auth): implement OAuth2 authentication

Add support for OAuth2 authentication flow

- Add OAuth2 middleware and handlers
- Implement token refresh logic
- Add user session management
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Make your changes
4. Run tests and quality checks
5. Submit a pull request

See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed guidelines.

## License

PyComet is licensed under the GNU General Public License v3.0 (GPLv3). See [LICENSE](LICENSE) for details.

## Support

- [GitHub Issues](https://github.com/jaydoubleu/pycomet/issues)
- [Security Advisories](https://github.com/jaydoubleu/pycomet/security/advisories/new)

## Acknowledgments

- Built with [litellm](https://github.com/BerriAI/litellm) for AI integration
- Uses [Click](https://click.palletsprojects.com/) for CLI interface
- Inspired by the [opencommit](https://github.com/di-sukharev/opencommit) project 
