# Development Guide

This guide contains detailed information for developers contributing to PyComet.

## Development Setup

1. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone and install:
```bash
git clone https://github.com/jaydoubleu/pycomet.git
cd pycomet
uv sync
```

3. Install development dependencies:
```bash
uv add --dev black ruff pytest pytest-cov mypy types-PyYAML pre-commit
```

4. Set up test environment:
```bash
# Copy example test environment file
cp .env.example .env.test
# Edit .env.test with your test API keys
```

## Testing

PyComet includes a comprehensive test suite that verifies functionality across multiple AI providers.

### Running Tests

Basic test execution:
```bash
# Run all tests
uv run pytest tests/

# Run tests for specific model
uv run pytest tests/ -k "gemini"

# Run tests with detailed LLM output
uv run pytest tests/ --show-llm-output

# Run tests for specific model with LLM output
uv run pytest tests/ -k "gemini" --show-llm-output
```

For detailed information about testing options and configurations, see [tests/README.md](tests/README.md).

## Code Quality

### Formatting and Linting
```bash
# Format code
uv run black .

# Run linter
uv run ruff check .

# Type checking
uv run mypy .
```

### Pre-commit Checks
Before submitting a PR, ensure:
1. All tests pass
2. Code is formatted with black
3. No linting errors from ruff
4. Type hints are valid (mypy)
5. Test coverage is maintained
6. LLM tests pass with `--show-llm-output`

To automate these checks, we use pre-commit hooks:

1. Install pre-commit:
```bash
uv add --dev pre-commit
```

2. Install the pre-commit hooks:
```bash
uv run pre-commit install
```

3. Run the hooks manually (if needed):
```bash
uv run pre-commit run --all-files
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Make your changes
4. Run the tests and quality checks
5. Submit a pull request

### Pull Request Guidelines
- Include tests for new functionality
- Update documentation as needed
- Follow the existing code style
- Keep changes focused and atomic
- Add meaningful commit messages
