# PyComet Configuration Guide

PyComet uses [litellm](https://github.com/BerriAI/litellm) for AI model integration, supporting a wide range of providers. The configuration file is located at `~/.config/pycomet/config.yaml`.

## Configuration Schema

```yaml
ai:
  provider: str           # AI provider name (e.g., "anthropic", "openai")
  model: str             # Model identifier
  api_key: str           # Your API key
  api_base: str | null   # Optional: Custom API endpoint
  api_version: str | null # Optional: API version (mainly for Azure)

commit:
  editor: str | null     # Editor for commit message review (default: $EDITOR or nano)
  include_emoji: bool    # Whether to include emoji in commit messages (default: true)
  custom_prompt: str | null # Optional: Custom system prompt
  detailed: bool        # Generate detailed multi-line commit messages (default: false)
```

## Supported Providers

PyComet supports all providers available through litellm. Here are some common configurations:

## Anthropic (Claude)
```yaml
ai:
  provider: anthropic
  model: claude-3-sonnet-20240229
  api_key: YOUR_ANTHROPIC_API_KEY
commit:
  include_emoji: true
```

## OpenAI
```yaml
ai:
  provider: openai
  model: gpt-4o-mini
  api_key: YOUR_OPENAI_API_KEY
commit:
  include_emoji: true
```

## Gemini
```yaml
ai:
  provider: gemini
  model: gemini/gemini-1.5-pro
  api_key: YOUR_GEMINI_API_KEY
commit:
  include_emoji: true
```

## Azure OpenAI
```yaml
ai:
  provider: azure
  model: azure/gpt-4o-mini
  api_key: YOUR_AZURE_API_KEY
  api_base: YOUR_AZURE_API_BASE
  api_version: 2024-02-15-preview
  deployment_name: YOUR_AZURE_DEPLOYMENT_NAME
commit:
  include_emoji: true
```

## Groq
```yaml
ai:
  provider: groq
  model: groq/deepseek-r1-distill-llama-70b
  api_key: YOUR_GROQ_API_KEY
commit:
  include_emoji: true
```

## XAI (Grok-2)
```yaml
ai:
  provider: xai
  model: xai/grok-2
  api_key: YOUR_XAI_API_KEY
commit:
  include_emoji: true
```

## GitHub (Phi-3.5)
```yaml
ai:
  provider: github
  model: github/phi-3.5-mini-instruct
  api_key: YOUR_GITHUB_API_KEY
commit:
  include_emoji: true
```

## OpenRouter
```yaml
ai:
  provider: openrouter
  model: openrouter/mistralai/mistral-small-24b-instruct-2501
  api_key: YOUR_OPENROUTER_API_KEY
  # Optional: specify the API base if not the default
  api_base: https://openrouter.ai/api/v1
commit:
  include_emoji: true
```

## Additional Configuration Options

### Custom Editor
```yaml
commit:
  editor: nvim  # Or any other editor command
```

### Disable Emojis
```yaml
commit:
  include_emoji: false
```

### Detailed Commit Messages
```yaml
commit:
  detailed: true  # Enables multi-line commit messages with bullet points
```

### Custom System Prompt
```yaml
commit:
  custom_prompt: |
    You are a Git commit message expert...
    Follow these rules:
    1. Use format: type(scope): description
    2. Keep messages under 72 characters
    ...
```

## Provider Support

PyComet leverages litellm's provider support, which means you can use any model supported by litellm. For the most up-to-date list of supported providers and models, refer to the [litellm documentation](https://docs.litellm.ai/docs/providers).