import os
import stat
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from pycomet.git import GitRepo


class TestGitHooks:
    """Tests for git hooks functionality in GitRepo class."""

    @pytest.fixture
    def mock_git_repo(self):
        """Create a temporary directory with a fake .git structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create .git/hooks directory structure
            git_dir = Path(temp_dir) / ".git"
            hooks_dir = git_dir / "hooks"
            hooks_dir.mkdir(parents=True)

            # Mock get_git_root to return our temp directory
            with mock.patch.object(
                GitRepo, "get_git_root", return_value=temp_dir
            ) as _:
                yield temp_dir

    def test_install_hook_creates_file(self, mock_git_repo):
        """Test that install_prepare_commit_msg_hook creates the hook file."""
        # Setup
        git_repo = GitRepo()
        hook_path = Path(mock_git_repo) / ".git" / "hooks" / "prepare-commit-msg"

        # Execute
        success, message = git_repo.install_prepare_commit_msg_hook()

        # Assert
        assert success is True
        assert "successfully" in message
        assert hook_path.exists()
        assert "PyComet" in hook_path.read_text()
        
        # Check file is executable
        assert bool(os.stat(hook_path).st_mode & stat.S_IXUSR)

    def test_install_hook_when_already_installed(self, mock_git_repo):
        """Test that installation is idempotent."""
        # Setup - install hook first
        git_repo = GitRepo()
        hook_path = Path(mock_git_repo) / ".git" / "hooks" / "prepare-commit-msg"
        hook_path.write_text("#!/bin/sh\n# PyComet AI-powered commit message hook")
        
        # Execute
        success, message = git_repo.install_prepare_commit_msg_hook()
        
        # Assert
        assert success is True
        assert "already installed" in message

    def test_install_hook_with_existing_hook(self, mock_git_repo):
        """Test that existing hooks are backed up."""
        # Setup - create an existing hook
        git_repo = GitRepo()
        hook_path = Path(mock_git_repo) / ".git" / "hooks" / "prepare-commit-msg"
        hook_path.write_text("#!/bin/sh\n# Some existing hook")
        
        # Execute
        success, message = git_repo.install_prepare_commit_msg_hook()
        
        # Assert
        assert success is True
        assert "backed up" in message
        backup_path = Path(str(hook_path) + ".backup")
        assert backup_path.exists()
        assert "Some existing hook" in backup_path.read_text()

    def test_uninstall_hook(self, mock_git_repo):
        """Test that uninstall_prepare_commit_msg_hook removes the hook."""
        # Setup - install hook first
        git_repo = GitRepo()
        hook_path = Path(mock_git_repo) / ".git" / "hooks" / "prepare-commit-msg"
        hook_path.write_text("#!/bin/sh\n# PyComet AI-powered commit message hook")
        
        # Execute
        success, message = git_repo.uninstall_prepare_commit_msg_hook()
        
        # Assert
        assert success is True
        assert "hook removed" in message
        assert not hook_path.exists()

    def test_uninstall_hook_with_backup(self, mock_git_repo):
        """Test that uninstallation restores backups."""
        # Setup - create hook and backup
        git_repo = GitRepo()
        hook_path = Path(mock_git_repo) / ".git" / "hooks" / "prepare-commit-msg"
        backup_path = Path(str(hook_path) + ".backup")
        
        hook_path.write_text("#!/bin/sh\n# PyComet AI-powered commit message hook")
        backup_path.write_text("#!/bin/sh\n# Original hook")
        
        # Execute
        success, message = git_repo.uninstall_prepare_commit_msg_hook()
        
        # Assert
        assert success is True
        assert "restored" in message
        assert hook_path.exists()
        assert "Original hook" in hook_path.read_text()
        assert not backup_path.exists()

    def test_uninstall_non_pycomet_hook(self, mock_git_repo):
        """Test that uninstall won't remove non-PyComet hooks."""
        # Setup - create non-PyComet hook
        git_repo = GitRepo()
        hook_path = Path(mock_git_repo) / ".git" / "hooks" / "prepare-commit-msg"
        hook_path.write_text("#!/bin/sh\n# Some other hook")
        
        # Execute
        success, message = git_repo.uninstall_prepare_commit_msg_hook()
        
        # Assert
        assert success is False
        assert "not a PyComet hook" in message
        assert hook_path.exists()

    def test_uninstall_when_not_installed(self, mock_git_repo):
        """Test uninstallation when hook is not installed."""
        # Setup - no hook present
        git_repo = GitRepo()
        
        # Execute
        success, message = git_repo.uninstall_prepare_commit_msg_hook()
        
        # Assert
        assert success is True
        assert "not installed" in message

    def test_install_not_in_git_repo(self):
        """Test installation failure when not in a git repository."""
        # Setup - mock get_git_root to return None
        with mock.patch.object(GitRepo, "get_git_root", return_value=None):
            git_repo = GitRepo()
            
            # Execute
            success, message = git_repo.install_prepare_commit_msg_hook()
            
            # Assert
            assert success is False
            assert "Not in a git repository" in message

    def test_uninstall_not_in_git_repo(self):
        """Test uninstallation failure when not in a git repository."""
        # Setup - mock get_git_root to return None
        with mock.patch.object(GitRepo, "get_git_root", return_value=None):
            git_repo = GitRepo()
            
            # Execute
            success, message = git_repo.uninstall_prepare_commit_msg_hook()
            
            # Assert
            assert success is False
            assert "Not in a git repository" in message