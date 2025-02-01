import re
import subprocess
import time
from random import uniform
from typing import Dict, List, Optional

import click
from litellm import (
    completion,
    token_counter,
)

from .models import ModelUsage

"""AI-powered Git commit message generator.

Configuration Options:
    ai:
        provider: anthropic
        model: claude-3-sonnet-20240229
        api_key: your-api-key
    
    commit:
        editor: nvim
        include_emoji: false
        custom_prompt: |
            # Optional: Override the default system prompt
            You are an expert at writing Git commit messages...
            
            1. Format Rules:
            - Use format: type(scope): description
            ...
"""


class AIClient:
    # Rate limit handling settings
    MAX_RETRIES = 3
    INITIAL_RETRY_DELAY = 1  # seconds
    MAX_RETRY_DELAY = 15  # seconds

    # List of models with large context windows
    LARGE_CONTEXT_MODELS = {
        "gemini/gemini-1.5-pro": "2M tokens",
        "gemini/gemini-exp-1206": "2M tokens",
        "gemini/gemini-1.5-flash": "1M tokens",
        "gemini/gemini-2.0-flash-exp": "1M tokens",
        "gemini/gemini-2.0-flash-thinking-exp": "1M tokens",
    }

    BASE_PROMPT = """You are an expert at analyzing Git changes and creating precise 
commit messages. You must follow these rules exactly:

1. Analysis Rules:
   - First review the list of changed files to understand the scope
   - CRITICAL: Always analyze modified files FIRST:
     * Modified files contain the primary changes that should drive the commit message
     * Code changes in modified files take precedence over any file deletions
     * Look for structural changes, improvements, and fixes in modified code
   
   - Secondary considerations:
     * Deleted files: While important to note, file deletions (especially of cache/temp 
       files) should NEVER be the main focus if there are code changes
     * Created files: New files support the main changes, consider their purpose
   
   - Priority Order:
     1. Code changes in modified files (MOST IMPORTANT)
     2. New feature/functionality additions
     3. File cleanup operations (LEAST IMPORTANT)

2. Message Format:
   {format_rules}

3. Writing Style Rules:
   - ALWAYS use imperative mood ("add", not "added" or "adds")
   - Be specific and focused
   - Explain WHAT and WHY, not HOW
   - Keep atomic - one main change per commit
   - Never use vague descriptions like "update stuff" or "fix things"
   - Avoid mentioning file operations unless they're the only change

4. Type Selection Rules:
   - feat: new features or capabilities
   - fix: bug corrections or error handling
   - refactor: code restructuring without behavior change
   - docs: documentation only changes
   - style: formatting, semicolons, etc.
   - test: adding/updating tests
   - chore: maintenance (ONLY when no code changes)
   - perf: performance improvements

5. Scope Selection:
   - Use the main component being changed
   - For multiple components, use the most significant one
   - Keep scope short and lowercase
   - For widespread changes, use broader scope (e.g., 'core', 'all')

6. Description Guidelines:
   - Start with action verb in imperative form
   - Include ALL significant changes
   - Focus on the purpose of the change
   - Be specific about code changes
   - No periods at end
   - Maximum length: 72 characters

7. Real Examples:
   Given changes:
   - Deleted: multiple __pycache__ files
   - Modified: src/core/parser.py (improved error handling)
   Good: "{example_format}"
   Bad: "chore: remove pycache files" (WRONG: ignores important code changes)
   Bad: "update error handling" (WRONG: too vague, missing type/scope)
   Bad: "parser: added error checks" (WRONG: past tense, missing type)

Return only the commit message, exactly as it should appear in git log."""

    USER_PROMPT = """Changes to be committed:

{file_summary}

Detailed changes:
{diff}

Remember: Return ONLY the commit message, no explanations."""

    # Add these constants
    COMMIT_TYPES = frozenset(
        ["feat", "fix", "docs", "style", "refactor", "test", "chore", "perf"]
    )

    def __init__(self, config: Dict, verbose: bool = False):
        self.config = config["ai"]
        self.commit_config = config["commit"]
        self.verbose = verbose
        self.last_usage: Optional[ModelUsage] = None

    def _get_system_prompt(self) -> str:
        """Get the appropriate system prompt based on configuration."""
        # Check for custom prompt in commit config
        if custom_prompt := self.commit_config.get("custom_prompt"):
            if self.verbose:
                click.echo("Using custom prompt from config")
            return str(custom_prompt)

        # Use default prompt with emoji configuration
        if self.verbose:
            click.echo("Using default system prompt")

        use_emoji = self.commit_config.get("include_emoji", True)
        detailed = self.commit_config.get("detailed", False)

        if self.verbose:
            click.echo(f"Commit config: {self.commit_config}")
            click.echo(
                f"Include emoji setting: {self.commit_config.get('include_emoji')}"
            )

        # Base format rules for each mode
        if detailed:
            base_format = """IMPORTANT: Return your response exactly as shown below, 
with literal newlines.
                
Line 1: {prefix}type(scope): summary title (max 72 chars)
(blank line)
Line 2: One-line summary of changes
(blank line)
Then bullet points, each on its own line, starting with "- ".

Example (use real newlines, not "\\n"):
{example}

CRITICAL:
1. Provide exactly 2 blank lines in the message: one after the title, one after the 
   summary.
2. Do not add or remove lines. The final result must be exactly 3 sections: title, 
   summary, bullets.
3. No code fences, no markdown backticks, no "```python" or comments. Just the lines 
   of text."""
        else:
            base_format = """IMPORTANT: Focus on the core change:
   - Format: {prefix}type(scope): description
   - Maximum length: 72 characters
   - Describe the main purpose only
   - {type_list}"""

        # Add emoji-specific elements if enabled
        if use_emoji:
            prefix = "<emoji> "
            type_list = """Include appropriate emoji:
     ✨ feat: new features
     🐛 fix: bug fixes
     📚 docs: documentation
     💄 style: formatting
     ♻️ refactor: restructuring
     ✅ test: testing
     🔧 chore: maintenance
     🚀 perf: performance"""
            example = """✨ feat(auth): implement OAuth2 authentication

Add support for OAuth2 authentication flow

- Add OAuth2 middleware and handlers
- Implement token refresh logic
- Add user session management"""
            example_short = "✨ feat(auth): add OAuth2 authentication support"
        else:
            prefix = ""
            type_list = """Available types:
     feat: new features
     fix: bug fixes
     docs: documentation
     style: formatting
     refactor: restructuring
     test: testing
     chore: maintenance
     perf: performance"""
            example = """feat(auth): implement OAuth2 authentication

Add support for OAuth2 authentication flow

- Add OAuth2 middleware and handlers
- Implement token refresh logic
- Add user session management"""
            example_short = "feat(auth): add OAuth2 authentication support"

        # Format the rules with the appropriate prefix and examples
        format_rules = base_format.format(
            prefix=prefix,
            type_list=type_list,
            example=example if detailed else example_short,  # This was backwards!
        )

        format_vars = {
            "format_rules": format_rules,
            "example_format": example_short if not detailed else example,
        }

        return self.BASE_PROMPT.format(**format_vars)

    def _get_file_summary(self, diff: str) -> str:
        """Extract and format the list of changed files from git status."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            )

            lines = []
            for line in result.stdout.splitlines():
                if line.strip():
                    status = line[:2]
                    file_path = line[3:]
                    if status == "D ":
                        lines.append(f"        deleted:    {file_path}")
                    elif status == "M ":
                        lines.append(f"        modified:   {file_path}")
                    elif status == "A ":
                        lines.append(f"        added:      {file_path}")
                    elif status == "R ":
                        lines.append(f"        renamed:    {file_path}")

            return "\n".join(lines) if lines else "No files changed"

        except subprocess.CalledProcessError:
            return "Error getting file status"

    def _get_messages(self, diff: str) -> List[Dict]:
        """Prepare messages for the AI model."""
        system_prompt = self._get_system_prompt()

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Generate a commit message for these changes:\n\n{diff}",
            },
        ]

        if self.verbose:
            click.echo("\nSystem Prompt:")
            click.echo("------------------------")
            click.echo(system_prompt)
            click.echo("------------------------\n")

        return messages

    def _make_completion_request(self, messages: List[Dict], model: str) -> str:
        """Make a completion request with retry logic for rate limits."""
        retry_count = 0
        delay = self.INITIAL_RETRY_DELAY

        while retry_count < self.MAX_RETRIES:
            try:
                if self.verbose:
                    click.echo(f"Making request to {model}")
                    click.echo("\nSystem Prompt:")
                    click.echo("-" * 40)
                    click.echo(messages[0]["content"])
                    click.echo("\nUser Prompt:")
                    click.echo("-" * 40)
                    click.echo(messages[1]["content"])
                    click.echo("-" * 40)

                response = completion(
                    model=model,
                    messages=messages,
                    api_key=self.config["api_key"],
                    max_tokens=100,  # Standardize max response length
                    temperature=0.7,  # Standardize temperature
                    **self._get_provider_specific_args(),
                )

                if self.verbose:
                    # Get token usage
                    if hasattr(response, "usage"):
                        usage = response.usage
                        click.echo("\nToken Usage:")
                        click.echo(
                            f"  Input tokens: {usage.get('prompt_tokens', 'N/A')}"
                        )
                        click.echo(
                            f"  Output tokens: {usage.get('completion_tokens', 'N/A')}"
                        )
                        click.echo(
                            f"  Total tokens: {usage.get('total_tokens', 'N/A')}"
                        )

                    # Get cost from response
                    if (
                        hasattr(response, "_hidden_params")
                        and "response_cost" in response._hidden_params
                    ):
                        cost = response._hidden_params["response_cost"]
                        if cost is not None:
                            click.echo(f"\nRequest cost: ${float(cost):.10f}")
                        else:
                            click.echo("\nRequest cost: Not available")
                    else:
                        click.echo("\nRequest cost: Not available")

                # Track usage after successful completion
                try:
                    cost = (
                        float(response._hidden_params.get("response_cost", 0))
                        if hasattr(response, "_hidden_params")
                        else 0.0
                    )
                except (TypeError, AttributeError):
                    cost = 0.0

                self.last_usage = ModelUsage(
                    input_tokens=response.usage.get("prompt_tokens", 0),
                    output_tokens=response.usage.get("completion_tokens", 0),
                    total_cost=cost,
                    calls=1,
                )

                message = str(response.choices[0].message.content.strip())

                if self.verbose:
                    click.echo("\nRaw AI Response:")
                    click.echo("-" * 40)
                    click.echo(message)
                    click.echo("-" * 40)

                # Remove any "thinking" tags or stray whitespace
                message = re.sub(r"<think>.*?</think>", "", message, flags=re.DOTALL)
                message = message.strip()

                if self.verbose:
                    click.echo("\nProcessed Message:")
                    click.echo("-" * 40)
                    click.echo(message)
                    click.echo("-" * 40)

                return message

            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = (
                    "rate_limit" in error_str or "too many requests" in error_str
                )

                if not is_rate_limit or retry_count == self.MAX_RETRIES - 1:
                    raise

                retry_count += 1
                if self.verbose:
                    click.echo(
                        f"Rate limit hit. Retrying in {delay} seconds... "
                        f"(Attempt {retry_count}/{self.MAX_RETRIES})"
                    )

                actual_delay = delay * (1 + uniform(-0.1, 0.1))
                time.sleep(actual_delay)
                delay = min(delay * 2, self.MAX_RETRY_DELAY)

        raise click.ClickException("Failed to complete request after max retries")

    def generate_commit_message(self, diff: str) -> str:
        """Generate a commit message using the configured AI service."""
        model = self.config.get("model", "claude-3-sonnet-20240229")
        provider = self.config.get("provider", "anthropic")

        if self.verbose:
            click.echo(f"Using AI provider: {provider}")
            click.echo(f"Using model: {model}")

            try:
                estimated_tokens = token_counter(model=model, text=diff)
                click.echo(f"\nEstimated diff tokens: {estimated_tokens}")
                click.echo(
                    "Note: Actual token count may differ due to model-specific encoding"
                )
            except Exception as e:
                click.echo(f"Could not calculate tokens: {str(e)}")

        try:
            messages = self._get_messages(diff)
            return self._make_completion_request(messages, model)
        except Exception as e:
            # Reset usage on error
            self.last_usage = ModelUsage()
            if self.verbose:
                click.echo(f"Error generating commit message: {str(e)}")
            raise

    def _get_provider_specific_args(self) -> Dict:
        """Get provider-specific arguments for the API call."""
        args = {}
        if self.config.get("api_base"):
            args["api_base"] = self.config["api_base"]
        if self.config.get("api_version"):
            args["api_version"] = self.config["api_version"]
        return args
