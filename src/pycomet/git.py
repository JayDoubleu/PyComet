import os
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple


class GitRepo:
    @staticmethod
    def _run_git_command(
        args: List[str], check: bool = True, capture_output: bool = True
    ) -> Optional[str]:
        """Run a git command and return its output."""
        try:
            result = subprocess.run(
                ["git"] + args, capture_output=capture_output, text=True, check=check
            )
            return result.stdout if capture_output else None
        except subprocess.CalledProcessError as e:
            if "diff" in args:
                raise Exception(
                    "Failed to get changes. Are you in a git repository?"
                ) from e
            raise Exception(f"Git command failed: {e.stderr}") from e

    @staticmethod
    def get_staged_diff() -> str:
        """Get the diff of staged changes."""
        return GitRepo._run_git_command(["diff", "--cached"]) or ""

    @staticmethod
    def get_unstaged_diff() -> str:
        """Get the diff of unstaged changes."""
        return GitRepo._run_git_command(["diff"]) or ""

    @staticmethod
    def create_commit(message: str) -> None:
        """Create a commit with the given message."""
        # Use -F flag to read message from file to preserve multiline format
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(message)
            f.flush()
            try:
                GitRepo._run_git_command(["commit", "-F", f.name], capture_output=False)
            finally:
                os.unlink(f.name)

    @staticmethod
    def has_staged_changes() -> bool:
        """Check if there are staged changes.
        Returns True if there are staged changes, False otherwise.
        """
        try:
            subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                capture_output=True,
                check=True,  # This will raise CalledProcessError when there are changes
            )
            return False  # No changes (command succeeded)
        except subprocess.CalledProcessError:
            return True  # Has changes (command failed with exit code 1)
        except Exception as e:
            # Log unexpected errors but assume no changes for safety
            print(f"Error checking staged changes: {str(e)}")
            return False

    @staticmethod
    def get_git_root() -> Optional[str]:
        """Get the git repository root directory.
        Returns the path to the repository root or None if not in a git repo.
        """
        try:
            return GitRepo._run_git_command(["rev-parse", "--show-toplevel"]).strip()
        except Exception:
            return None

    @staticmethod
    def install_prepare_commit_msg_hook() -> Tuple[bool, str]:
        """Install the prepare-commit-msg git hook.
        Returns (success, message) tuple.
        """
        git_root = GitRepo.get_git_root()
        if not git_root:
            return False, "Not in a git repository"

        hooks_dir = Path(git_root) / ".git" / "hooks"
        hook_path = hooks_dir / "prepare-commit-msg"

        # Create the hook script
        hook_content = """#!/bin/sh
# PyComet AI-powered commit message hook
# https://github.com/JayDoubleu/PyComet

# Get the commit message file path from Git
COMMIT_MSG_FILE=$1
COMMIT_SOURCE=$2

# Skip if not an interactive commit (e.g., merge commit, commit with -m)
if [ "$COMMIT_SOURCE" = "message" ] || [ "$COMMIT_SOURCE" = "template" ] || \\
   [ "$COMMIT_SOURCE" = "merge" ] || [ "$COMMIT_SOURCE" = "squash" ]; then
    exit 0
fi

# Generate commit message with PyComet and save to commit message file
pycomet preview --no-detailed 2>/dev/null | \\
    awk 'BEGIN{f=0} /^-+$/{f=!f; next} f{print}' > "$COMMIT_MSG_FILE.pycomet"
if [ -s "$COMMIT_MSG_FILE.pycomet" ]; then
    cat "$COMMIT_MSG_FILE.pycomet" > "$COMMIT_MSG_FILE"
    rm "$COMMIT_MSG_FILE.pycomet"
    echo "# PyComet: Generated AI commit message. Edit as needed." >> "$COMMIT_MSG_FILE"
    echo "# To disable the PyComet hook: git config --local core.hooksPath /dev/null" \\
        >> "$COMMIT_MSG_FILE"
fi
"""
        try:
            # Ensure hooks directory exists
            hooks_dir.mkdir(exist_ok=True, parents=True)

            # Check if hook already exists
            if hook_path.exists():
                with open(hook_path, "r") as f:
                    existing_content = f.read()
                    if "PyComet" in existing_content:
                        return True, "Prepare-commit-msg hook is already installed"
                    else:
                        # Backup existing hook
                        backup_path = Path(str(hook_path) + ".backup")
                        hook_path.rename(backup_path)
                        backup_msg = f" (existing hook backed up to {backup_path})"
            else:
                backup_msg = ""

            # Write the hook script
            with open(hook_path, "w") as f:
                f.write(hook_content)

            # Make the hook executable
            hook_mode = os.stat(hook_path).st_mode
            executable_mode = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
            os.chmod(hook_path, hook_mode | executable_mode)

            return True, f"Prepare-commit-msg hook installed successfully{backup_msg}"

        except Exception as e:
            return False, f"Failed to install hook: {str(e)}"

    @staticmethod
    def uninstall_prepare_commit_msg_hook() -> Tuple[bool, str]:
        """Uninstall the prepare-commit-msg git hook.
        Returns (success, message) tuple.
        """
        git_root = GitRepo.get_git_root()
        if not git_root:
            return False, "Not in a git repository"

        hook_path = Path(git_root) / ".git" / "hooks" / "prepare-commit-msg"
        backup_path = Path(str(hook_path) + ".backup")

        if not hook_path.exists():
            return True, "Prepare-commit-msg hook is not installed"

        try:
            # Check if it's the PyComet hook
            with open(hook_path, "r") as f:
                content = f.read()
                is_pycomet_hook = "PyComet" in content

            # Remove the hook
            if is_pycomet_hook:
                # Restore backup if it exists
                if backup_path.exists():
                    backup_path.rename(hook_path)
                    return True, "PyComet hook removed and original hook restored"
                else:
                    hook_path.unlink()
                    return True, "PyComet hook removed"
            else:
                return False, "The prepare-commit-msg hook is not a PyComet hook"

        except Exception as e:
            return False, f"Failed to uninstall hook: {str(e)}"
