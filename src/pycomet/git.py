import os
import subprocess
import tempfile
from typing import List, Optional


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
