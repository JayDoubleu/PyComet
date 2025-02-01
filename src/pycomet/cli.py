import os
from typing import Dict, Optional

import click

from .ai import AIClient
from .config import get_config_path, load_config
from .git import GitRepo


@click.group()
def cli() -> None:
    """PyComet - AI-powered Git commit messages"""
    pass


def get_editor(config: Dict, verbose: bool = False) -> str:
    """Get the editor to use, with verbose logging about selection."""
    # Safely get editor from config
    editor = config.get("commit", {}).get("editor")
    env_editor = os.environ.get("EDITOR")

    if editor and verbose:
        click.echo(f"Using editor from config: {editor}")
    elif env_editor and verbose:
        click.echo(f"Using editor from EDITOR environment variable: {env_editor}")
    elif verbose:
        click.echo("No editor configured, using default: nano")

    return editor or env_editor or "nano"


@cli.command()
def config() -> None:
    """Configure PyComet settings"""
    config_path = get_config_path()
    config = load_config()
    editor = get_editor(config, verbose=True)
    os.system(f'{editor} "{config_path}"')


def _warn_if_emoji_with_custom_prompt(
    config: Dict, emoji_flag_used: bool, custom_prompt: Optional[str], verbose: bool
) -> None:
    """Warn if emoji flags are used with a custom prompt."""
    has_custom_prompt = (
        config.get("commit", {}).get("custom_prompt") or custom_prompt is not None
    )
    if emoji_flag_used and has_custom_prompt:
        click.echo(
            "\nWarning: Emoji flags (--emoji/--no-emoji) are ignored when using a "
            "custom prompt. Emoji handling must be configured in your custom prompt.",
            err=True,
        )


def _get_detailed_mode(
    config: Dict, detailed_flag: Optional[bool], verbose: bool = False
) -> bool:
    """Get the detailed mode setting, with proper fallback behavior."""
    if detailed_flag is not None:
        # CLI flag takes precedence
        if verbose:
            click.echo(
                f"Detailed mode {'enabled' if detailed_flag else 'disabled'} from CLI"
            )
        return bool(detailed_flag)  # Explicitly cast to bool

    # Check config
    detailed = bool(
        config.get("commit", {}).get("detailed", False)
    )  # Explicitly cast to bool
    if verbose:
        click.echo(f"Using detailed mode from config: {detailed}")
    return detailed


@cli.command()
@click.option(
    "--verbose", "-v", is_flag=True, help="Show detailed execution information"
)
@click.option(
    "--emoji/--no-emoji",
    default=None,
    help="Override emoji inclusion in commit messages",
)
@click.option("--prompt", "-p", help="Override system prompt for this commit")
@click.option("--editor", "-e", help="Override editor for this commit")
@click.option(
    "--detailed/--no-detailed",
    "-d/-D",
    default=None,
    help="Enable/disable detailed multi-line commit message (default: disabled)",
)
def commit(
    verbose: bool,
    emoji: Optional[bool],
    prompt: Optional[str],
    editor: Optional[str],
    detailed: Optional[bool],
) -> None:
    """Generate and create an AI-powered commit"""
    try:
        if verbose:
            click.echo("Loading configuration...")
        config = load_config(verbose=verbose)

        # Warn about emoji flags with custom prompt (check before applying overrides)
        _warn_if_emoji_with_custom_prompt(config, emoji is not None, prompt, verbose)

        # Apply command line overrides
        if any(x is not None for x in [emoji, prompt]) or detailed is not None:
            if "commit" not in config:
                config["commit"] = {}
            if emoji is not None:
                config["commit"]["include_emoji"] = emoji
                if verbose:
                    click.echo(f"Emoji override from CLI: {emoji}")
            if prompt:
                config["commit"]["custom_prompt"] = prompt
                if verbose:
                    click.echo("Using custom prompt from CLI")
            if editor:
                config["commit"]["editor"] = editor
                if verbose:
                    click.echo(f"Using editor from CLI: {editor}")
            config["commit"]["detailed"] = _get_detailed_mode(config, detailed, verbose)

        git = GitRepo()

        # Check for staged changes
        if verbose:
            click.echo("Checking for staged changes...")
        if not git.has_staged_changes():
            click.echo(
                "\nNo staged changes found. Stage your changes first with 'git add'"
            )
            return

        # Get diff and generate message
        if verbose:
            click.echo("Initializing AI client...")
        diff = git.get_staged_diff()
        ai_client = AIClient(config, verbose=verbose)
        if verbose:
            click.echo("Generating commit message...")
        initial_message = ai_client.generate_commit_message(diff)

        # Always open editor for review
        if verbose:
            click.echo("Opening editor for message review...")
        editor = get_editor(config, verbose=verbose)
        # Preserve newlines in editor
        edited_message: str = click.edit(initial_message + "\n", editor=editor) or ""
        if not edited_message:
            click.echo("Commit aborted")
            return

        # Create commit with multiline message
        if verbose:
            click.echo("Creating commit...")
        git.create_commit(edited_message.rstrip())
        # Use write to preserve newlines in output
        click.echo("Created commit:")
        click.echo("------------------------")
        click.get_text_stream("stdout").write(edited_message)
        click.echo("------------------------")

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)


@cli.command()
@click.option(
    "--verbose", "-v", is_flag=True, help="Show detailed execution information"
)
@click.option(
    "--emoji/--no-emoji",
    default=None,
    help="Override emoji inclusion in commit messages",
)
@click.option("--prompt", "-p", help="Override system prompt for this commit")
@click.option(
    "--detailed/--no-detailed",
    "-d/-D",
    default=None,
    help="Enable/disable detailed multi-line commit message (default: disabled)",
)
def preview(
    verbose: bool,
    emoji: Optional[bool],
    prompt: Optional[str],
    detailed: Optional[bool],
) -> None:
    """Preview the commit message without creating a commit"""
    try:
        if verbose:
            click.echo("Loading configuration...")
        config = load_config(verbose=verbose)

        # Warn about emoji flags with custom prompt (check before applying overrides)
        _warn_if_emoji_with_custom_prompt(config, emoji is not None, prompt, verbose)

        # Apply command line overrides
        if any(x is not None for x in [emoji, prompt]) or detailed is not None:
            if "commit" not in config:
                config["commit"] = {}
            if emoji is not None:
                config["commit"]["include_emoji"] = emoji
                if verbose:
                    click.echo(f"Emoji override from CLI: {emoji}")
            if prompt:
                config["commit"]["custom_prompt"] = prompt
                if verbose:
                    click.echo("Using custom prompt from CLI")
            config["commit"]["detailed"] = _get_detailed_mode(config, detailed, verbose)

        git = GitRepo()

        # Check for staged changes
        if verbose:
            click.echo("Checking for staged changes...")
        if not git.has_staged_changes():
            click.echo(
                "\nNo staged changes found. Stage your changes first with 'git add'"
            )
            return

        # Get diff and generate message
        if verbose:
            click.echo("Initializing AI client...")
        diff = git.get_staged_diff()
        ai_client = AIClient(config, verbose=verbose)
        if verbose:
            click.echo("Generating commit message...")
        initial_message = ai_client.generate_commit_message(diff)

        # Show the preview
        click.echo("\nPreview of commit message:")
        click.echo("------------------------")
        # Use write to preserve newlines exactly
        click.get_text_stream("stdout").write(initial_message + "\n")
        click.echo("------------------------")
        click.echo("\nTo create this commit, run: pycomet commit")

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)


if __name__ == "__main__":
    cli()
