from unittest import mock

import pytest
from click.testing import CliRunner

from pycomet.cli import hooks, hooks_install, hooks_uninstall
from pycomet.git import GitRepo


class TestCliHooks:
    """Tests for CLI hooks commands."""

    @pytest.fixture
    def runner(self):
        """Return a Click CLI test runner."""
        return CliRunner()

    def test_hooks_group(self, runner):
        """Test the hooks command group exists."""
        result = runner.invoke(hooks)
        assert result.exit_code == 0
        assert "Manage git hooks" in result.output

    def test_hooks_install_success(self, runner):
        """Test the hooks install command when successful."""
        with mock.patch.object(
            GitRepo, "install_prepare_commit_msg_hook", 
            return_value=(True, "Prepare-commit-msg hook installed successfully")
        ):
            result = runner.invoke(hooks_install)
            
            assert result.exit_code == 0
            assert "✅" in result.output
            assert "installed successfully" in result.output
            assert "git commit" in result.output  # Instructions in output

    def test_hooks_install_verbose(self, runner):
        """Test the hooks install command with verbose flag."""
        with mock.patch.object(
            GitRepo, "install_prepare_commit_msg_hook", 
            return_value=(True, "Prepare-commit-msg hook installed successfully")
        ):
            result = runner.invoke(hooks_install, ["--verbose"])
            
            assert result.exit_code == 0
            assert "Installing prepare-commit-msg hook..." in result.output
            assert "✅" in result.output

    def test_hooks_install_failure(self, runner):
        """Test the hooks install command when it fails."""
        with mock.patch.object(
            GitRepo, "install_prepare_commit_msg_hook", 
            return_value=(False, "Not in a git repository")
        ):
            result = runner.invoke(hooks_install)
            
            assert result.exit_code == 0  # CLI doesn't return error code
            assert "❌" in result.output
            assert "Not in a git repository" in result.output

    def test_hooks_uninstall_success(self, runner):
        """Test the hooks uninstall command when successful."""
        with mock.patch.object(
            GitRepo, "uninstall_prepare_commit_msg_hook", 
            return_value=(True, "PyComet hook removed")
        ):
            result = runner.invoke(hooks_uninstall)
            
            assert result.exit_code == 0
            assert "✅" in result.output
            assert "hook removed" in result.output

    def test_hooks_uninstall_verbose(self, runner):
        """Test the hooks uninstall command with verbose flag."""
        with mock.patch.object(
            GitRepo, "uninstall_prepare_commit_msg_hook", 
            return_value=(True, "PyComet hook removed")
        ):
            result = runner.invoke(hooks_uninstall, ["--verbose"])
            
            assert result.exit_code == 0
            assert "Uninstalling prepare-commit-msg hook..." in result.output
            assert "✅" in result.output

    def test_hooks_uninstall_failure(self, runner):
        """Test the hooks uninstall command when it fails."""
        with mock.patch.object(
            GitRepo, "uninstall_prepare_commit_msg_hook", 
            return_value=(False, "Not in a git repository")
        ):
            result = runner.invoke(hooks_uninstall)
            
            assert result.exit_code == 0  # CLI doesn't return error code
            assert "❌" in result.output
            assert "Not in a git repository" in result.output