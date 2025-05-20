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

#### Local Test Execution

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

#### Docker Test Execution

You can also run tests in an isolated Docker environment:

```bash
# Make the script executable (if not already)
chmod +x docker-test.sh

# Run all tests (excluding integration tests)
./docker-test.sh all

# Run only git hooks tests
./docker-test.sh hooks

# Run specific tests with custom arguments
./docker-test.sh tests/test_git.py -v
```

The Docker testing environment ensures consistent test execution across different development setups.

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

## Git Hooks Integration

PyComet can integrate with Git's `prepare-commit-msg` hook to automatically generate AI-powered commit messages when you run `git commit`.

### Setting up Git Hooks

To install the Git hooks integration:

```bash
# Install the prepare-commit-msg hook
pycomet hooks install

# To uninstall the hook
pycomet hooks uninstall
```

When the hook is installed, running `git commit` (without a `-m` message) will:
1. Generate an AI-powered commit message based on staged changes
2. Pre-fill the commit message in your editor
3. Allow you to edit the message before finalizing the commit

The hook won't run when you provide a message with `git commit -m "message"` or during 
merge/rebase operations.

To temporarily disable the hook for a specific commit, you can run:
```bash
git -c core.hooksPath=/dev/null commit
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
