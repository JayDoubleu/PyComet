# Test Suite Documentation

This directory contains automated tests for the PyComet commit message generation system, testing various AI model providers and configurations.

## Prerequisites

1. Environment Setup
   - Copy `.env.example` to `.env.test`:
     ```bash
     cp .env.example .env.test
     ```
   - Add your API keys to `.env.test` (never commit this file)
   - Ensure Python version matches `.python-version`
   - Install test dependencies:
     ```bash
     uv add --dev pytest pytest-cov pytest-dotenv
     ```

## Running Tests

### Basic Test Execution
```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_ai.py

# Run tests for specific model
uv run pytest tests/ -k "gemini"
```

### Test Output Options
```bash
# Show LLM responses and detailed model information
uv run pytest tests/ --show-llm-output

# Run specific model test with output
uv run pytest tests/ -k "gemini" --show-llm-output
```

### Test Selection
```bash
# Run only basic response tests
uv run pytest tests/ -k "test_basic_response"
```

## Test Structure

### Main Test Files
- `test_ai.py`: Tests for AI model integrations
- `conftest.py`: Test fixtures and configurations
- `utils.py`: Test utilities and metrics collection

### Test Coverage
- AI Provider Integration Tests
  - Gemini
  - Anthropic (Claude)
  - OpenAI
  - Groq
  - Azure OpenAI
  - XAI (Grok-2)
  - GitHub (Phi-3.5)
  - OpenRouter

## Test Categories

### Unit Tests
Run tests that don't require API access:
```bash
uv run pytest tests/ -m "not integration"
```

### Integration Tests
Run tests that require API access:
```bash
uv run pytest tests/ -m "integration"
```

### Verbose Tests
Run tests with detailed output:
```bash
uv run pytest tests/ -m "verbose_only" --show-llm-output
```

## Development

### Code Quality

Format code:
```bash
uv run black tests/
```

Lint code:
```bash
uv run ruff check tests/
```

Check type hints:
```bash
uv run mypy tests/
```

### Adding New Tests
1. Add new test fixtures in `conftest.py`
2. Create test functions in `test_ai.py`
3. Follow existing patterns for consistency
4. Include both basic and verbose test variants

## Metrics and Reporting

The test suite automatically tracks and reports:
- Token usage per model
- Cost analysis
- Character count statistics
- API call frequency

Example output:
```
========================================================================================================================
📊 Model Usage Summary
========================================================================================================================
Model                                               Calls  Avg Chars   Input Tokens  Output Tokens   Total Tokens Cost ($)
------------------------------------------------------------------------------------------------------------------------
azure/gpt-4o-mini                                       2       66.2            720             50            770   0.0002
claude-3-sonnet-20240229                                2       48.2            810             67            877   0.0034
gemini/gemini-1.5-pro                                   2       50.5            752             41            793   0.0031
github/phi-3.5-mini-instruct                            2       49.8            864             54            918   0.0000
gpt-4o-mini                                             2       62.8            720             51            771   0.0001
groq/deepseek-r1-distill-llama-70b                      2      458.8            690            400           1090   0.0009
openrouter/mistralai/mistral-small-24b-instruct         2       51.5            734             47            781   0.0000
xai/grok-2                                              2       63.2            734             49            783   0.0020
------------------------------------------------------------------------------------------------------------------------
Total Cost: $0.0097
========================================================================================================================
```

## Required Environment Variables

The following API keys should be set in your `.env.test` file:

```bash
TEST_GEMINI_API_KEY=your_key_here
TEST_ANTHROPIC_API_KEY=your_key_here
TEST_OPENAI_API_KEY=your_key_here
TEST_GROQ_API_KEY=your_key_here
TEST_XAI_API_KEY=your_key_here
TEST_GITHUB_API_KEY=your_key_here
TEST_OPENROUTER_API_KEY=your_key_here

# Azure OpenAI specific variables
TEST_AZURE_API_KEY=your_key_here
TEST_AZURE_API_BASE=your_endpoint_here
TEST_AZURE_API_VERSION=2024-02-15-preview
TEST_AZURE_DEPLOYMENT_NAME=your_deployment_here
```

## Security Note
⚠️ Never commit real API keys to version control. Use environment variables or `.env.test` file (added to `.gitignore`).

## Support
If you encounter test-related issues:
1. Check existing [test-related issues](https://github.com/jaydoubleu/pycomet/issues?q=is%3Aissue+is%3Aopen+label%3Atesting)
2. Create a new issue with the "testing" label
3. Include test output and environment details

## Contributing
Please see the main [Development Guide](../DEVELOPMENT.md) for contribution guidelines and setup instructions. 