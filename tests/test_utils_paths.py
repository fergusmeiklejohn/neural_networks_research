"""
Tests for utils.paths module.

Tests path resolution, directory management, and cross-platform compatibility.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from utils.paths import (
    get_project_root,
    resolve_path,
    get_data_path,
    get_output_path,
    get_model_path,
    get_cache_path,
    get_experiment_path,
    get_checkpoint_path,
    get_results_path,
    get_log_path,
    find_files,
    ensure_directory,
    clean_directory,
    get_relative_path,
    validate_path,
    get_platform_paths,
)


class TestProjectRoot:
    """Test project root detection."""
    
    def test_get_project_root_returns_path(self):
        """Test that get_project_root returns a Path object."""
        root = get_project_root()
        assert isinstance(root, Path)
        assert root.exists()
    
    def test_get_project_root_finds_markers(self):
        """Test that project root contains expected marker files."""
        root = get_project_root()
        
        # Should find at least one marker file
        markers = ["pyproject.toml", "setup.py", "CLAUDE.md", ".git", "requirements.txt", "README.md"]
        found_markers = [marker for marker in markers if (root / marker).exists()]
        assert len(found_markers) > 0
    
    def test_get_project_root_consistent(self):
        """Test that project root detection is consistent."""
        root1 = get_project_root()
        root2 = get_project_root()
        assert root1 == root2


class TestPathResolution:
    """Test path resolution functionality."""
    
    def test_resolve_path_absolute(self):
        """Test resolving absolute paths."""
        abs_path = Path("/tmp/test")
        result = resolve_path(abs_path)
        assert result == abs_path
    
    def test_resolve_path_relative_to_project_root(self):
        """Test resolving relative paths to project root."""
        result = resolve_path("data/test")
        project_root = get_project_root()
        expected = project_root / "data" / "test"
        assert result == expected.resolve()
    
    def test_resolve_path_relative_to_base(self, temp_dir):
        """Test resolving relative paths to custom base."""
        base_path = temp_dir
        result = resolve_path("subdir/file.txt", base_path)
        expected = base_path / "subdir" / "file.txt"
        assert result == expected.resolve()
    
    def test_resolve_path_string_input(self):
        """Test resolving paths from string input."""
        result = resolve_path("data/test")
        assert isinstance(result, Path)
        assert result.name == "test"
    
    def test_resolve_path_path_input(self):
        """Test resolving paths from Path input."""
        input_path = Path("data/test")
        result = resolve_path(input_path)
        assert isinstance(result, Path)
        assert result.name == "test"


class TestDirectoryPaths:
    """Test directory path functions."""
    
    @patch("utils.paths.get_config")
    def test_get_data_path_default(self, mock_get_config):
        """Test get_data_path with default configuration."""
        mock_get_config.return_value = {"data_dir": "./data"}
        
        result = get_data_path(create=False)
        assert isinstance(result, Path)
        assert result.name == "data"
    
    @patch("utils.paths.get_config")
    def test_get_data_path_with_subpath(self, mock_get_config):
        """Test get_data_path with subdirectory."""
        mock_get_config.return_value = {"data_dir": "./data"}
        
        result = get_data_path("processed", create=False)
        assert isinstance(result, Path)
        assert result.name == "processed"
        assert result.parent.name == "data"
    
    @patch("utils.paths.get_config")
    def test_get_data_path_creates_directory(self, mock_get_config, temp_dir):
        """Test get_data_path creates directory when requested."""
        data_dir = temp_dir / "data"
        mock_get_config.return_value = {"data_dir": str(data_dir)}
        
        result = get_data_path("subdir", create=True)
        assert result.exists()
        assert result.is_dir()
    
    @patch("utils.paths.get_config")
    def test_get_output_path_default(self, mock_get_config):
        """Test get_output_path with default configuration."""
        mock_get_config.return_value = {"output_dir": "./outputs"}
        
        result = get_output_path(create=False)
        assert isinstance(result, Path)
        assert result.name == "outputs"
    
    @patch("utils.paths.get_config")
    def test_get_model_path_default(self, mock_get_config):
        """Test get_model_path with default configuration."""
        mock_get_config.return_value = {"model_dir": "./models"}
        
        result = get_model_path(create=False)
        assert isinstance(result, Path)
        assert result.name == "models"
    
    @patch("utils.paths.get_config")
    def test_get_cache_path_default(self, mock_get_config):
        """Test get_cache_path with default configuration."""
        mock_get_config.return_value = {"cache_dir": "./cache"}
        
        result = get_cache_path(create=False)
        assert isinstance(result, Path)
        assert result.name == "cache"
    
    @patch("utils.paths.get_config")
    def test_get_cache_path_absolute(self, mock_get_config, temp_dir):
        """Test get_cache_path with absolute path."""
        cache_dir = temp_dir / "cache"
        mock_get_config.return_value = {"cache_dir": str(cache_dir)}
        
        result = get_cache_path(create=True)
        assert result == cache_dir
        assert result.exists()


class TestExperimentPaths:
    """Test experiment-specific path functions."""
    
    def test_get_experiment_path_default(self):
        """Test get_experiment_path with default settings."""
        result = get_experiment_path("test_experiment", create=False)
        assert isinstance(result, Path)
        assert result.name == "test_experiment"
        assert "experiments" in result.parts
    
    def test_get_experiment_path_with_subpath(self):
        """Test get_experiment_path with subdirectory."""
        result = get_experiment_path("test_experiment", "outputs", create=False)
        assert isinstance(result, Path)
        assert result.name == "outputs"
        assert result.parent.name == "test_experiment"
    
    def test_get_experiment_path_creates_directory(self, temp_dir):
        """Test get_experiment_path creates directory when requested."""
        with patch("utils.paths.get_project_root", return_value=temp_dir):
            result = get_experiment_path("test_experiment", create=True)
            assert result.exists()
            assert result.is_dir()
    
    def test_get_checkpoint_path(self):
        """Test get_checkpoint_path functionality."""
        result = get_checkpoint_path("test_experiment", "test_model", create=False)
        assert isinstance(result, Path)
        assert result.name == "test_model.keras"
        assert "checkpoints" in result.parts
        assert "test_experiment" in result.parts
    
    def test_get_results_path(self):
        """Test get_results_path functionality."""
        result = get_results_path("test_experiment", "test_results", create=False)
        assert isinstance(result, Path)
        assert result.name == "test_results.json"
        assert "results" in result.parts
        assert "test_experiment" in result.parts
    
    def test_get_log_path(self):
        """Test get_log_path functionality."""
        result = get_log_path("test_experiment", "test_log", create=False)
        assert isinstance(result, Path)
        assert result.name == "test_log.log"
        assert "logs" in result.parts
        assert "test_experiment" in result.parts


class TestFileOperations:
    """Test file operation utilities."""
    
    def test_find_files_recursive(self, temp_dir):
        """Test find_files with recursive search."""
        # Create test files
        (temp_dir / "file1.txt").touch()
        (temp_dir / "subdir").mkdir()
        (temp_dir / "subdir" / "file2.txt").touch()
        (temp_dir / "subdir" / "file3.py").touch()
        
        # Find all .txt files
        result = find_files("*.txt", temp_dir, recursive=True)
        assert len(result) == 2
        assert all(f.suffix == ".txt" for f in result)
    
    def test_find_files_non_recursive(self, temp_dir):
        """Test find_files with non-recursive search."""
        # Create test files
        (temp_dir / "file1.txt").touch()
        (temp_dir / "subdir").mkdir()
        (temp_dir / "subdir" / "file2.txt").touch()
        
        # Find .txt files (non-recursive)
        result = find_files("*.txt", temp_dir, recursive=False)
        assert len(result) == 1
        assert result[0].name == "file1.txt"
    
    def test_find_files_no_matches(self, temp_dir):
        """Test find_files with no matches."""
        result = find_files("*.nonexistent", temp_dir)
        assert len(result) == 0
    
    def test_ensure_directory_creates_directory(self, temp_dir):
        """Test ensure_directory creates directory."""
        new_dir = temp_dir / "new_directory"
        assert not new_dir.exists()
        
        result = ensure_directory(new_dir)
        assert result == new_dir
        assert new_dir.exists()
        assert new_dir.is_dir()
    
    def test_ensure_directory_existing_directory(self, temp_dir):
        """Test ensure_directory with existing directory."""
        existing_dir = temp_dir / "existing"
        existing_dir.mkdir()
        
        result = ensure_directory(existing_dir)
        assert result == existing_dir
        assert existing_dir.exists()
    
    def test_ensure_directory_nested_path(self, temp_dir):
        """Test ensure_directory with nested path."""
        nested_dir = temp_dir / "level1" / "level2" / "level3"
        assert not nested_dir.exists()
        
        result = ensure_directory(nested_dir)
        assert result == nested_dir
        assert nested_dir.exists()
        assert nested_dir.is_dir()
    
    def test_clean_directory_removes_files(self, temp_dir):
        """Test clean_directory removes matching files."""
        # Create test files
        (temp_dir / "file1.txt").touch()
        (temp_dir / "file2.txt").touch()
        (temp_dir / "file3.py").touch()
        
        result = clean_directory(temp_dir, "*.txt")
        assert result == 2
        assert not (temp_dir / "file1.txt").exists()
        assert not (temp_dir / "file2.txt").exists()
        assert (temp_dir / "file3.py").exists()
    
    def test_clean_directory_nonexistent(self, temp_dir):
        """Test clean_directory with nonexistent directory."""
        nonexistent = temp_dir / "nonexistent"
        result = clean_directory(nonexistent)
        assert result == 0
    
    def test_clean_directory_no_matches(self, temp_dir):
        """Test clean_directory with no matching files."""
        (temp_dir / "file1.txt").touch()
        
        result = clean_directory(temp_dir, "*.py")
        assert result == 0
        assert (temp_dir / "file1.txt").exists()


class TestPathUtilities:
    """Test path utility functions."""
    
    def test_get_relative_path_to_project_root(self):
        """Test get_relative_path to project root."""
        project_root = get_project_root()
        test_path = project_root / "data" / "test.txt"
        
        result = get_relative_path(test_path)
        assert result == "data/test.txt" or result == "data\\test.txt"  # Handle Windows
    
    def test_get_relative_path_to_custom_base(self, temp_dir):
        """Test get_relative_path to custom base."""
        base_path = temp_dir
        test_path = temp_dir / "subdir" / "file.txt"
        
        result = get_relative_path(test_path, base_path)
        assert result == "subdir/file.txt" or result == "subdir\\file.txt"
    
    def test_get_relative_path_not_relative(self, temp_dir):
        """Test get_relative_path with non-relative path."""
        base_path = temp_dir / "base"
        test_path = temp_dir / "other" / "file.txt"
        
        result = get_relative_path(test_path, base_path)
        assert result == str(test_path)
    
    def test_validate_path_exists(self, temp_dir):
        """Test validate_path with existing path."""
        test_file = temp_dir / "test.txt"
        test_file.touch()
        
        assert validate_path(test_file, must_exist=True) is True
        assert validate_path(test_file, must_exist=False) is True
    
    def test_validate_path_not_exists(self, temp_dir):
        """Test validate_path with non-existing path."""
        test_file = temp_dir / "nonexistent.txt"
        
        assert validate_path(test_file, must_exist=False) is True
        assert validate_path(test_file, must_exist=True) is False
    
    def test_validate_path_file_type(self, temp_dir):
        """Test validate_path with file type validation."""
        test_file = temp_dir / "test.txt"
        test_file.touch()
        test_dir = temp_dir / "testdir"
        test_dir.mkdir()
        
        assert validate_path(test_file, must_be_file=True) is True
        assert validate_path(test_file, must_be_dir=True) is False
        assert validate_path(test_dir, must_be_file=True) is False
        assert validate_path(test_dir, must_be_dir=True) is True
    
    def test_validate_path_combined_constraints(self, temp_dir):
        """Test validate_path with multiple constraints."""
        test_file = temp_dir / "test.txt"
        test_file.touch()
        
        assert validate_path(test_file, must_exist=True, must_be_file=True) is True
        assert validate_path(test_file, must_exist=True, must_be_dir=True) is False


class TestPlatformPaths:
    """Test platform-specific path functionality."""
    
    @patch("utils.paths.detect_platform")
    def test_get_platform_paths_local(self, mock_detect):
        """Test get_platform_paths for local platform."""
        mock_detect.return_value = "local"
        
        paths = get_platform_paths()
        assert isinstance(paths, dict)
        assert "data_root" in paths
        assert "output_root" in paths
        assert "cache_root" in paths
        assert "temp_root" in paths
        assert paths["data_root"] == "./data"
    
    @patch("utils.paths.detect_platform")
    def test_get_platform_paths_colab(self, mock_detect):
        """Test get_platform_paths for Colab platform."""
        mock_detect.return_value = "colab"
        
        paths = get_platform_paths()
        assert paths["data_root"] == "/content/data"
        assert paths["output_root"] == "/content/outputs"
        assert paths["cache_root"] == "/content/cache"
    
    @patch("utils.paths.detect_platform")
    def test_get_platform_paths_paperspace(self, mock_detect):
        """Test get_platform_paths for Paperspace platform."""
        mock_detect.return_value = "paperspace"
        
        paths = get_platform_paths()
        assert paths["data_root"] == "/notebooks/data"
        assert paths["output_root"] == "/storage/outputs"
        assert paths["cache_root"] == "/storage/cache"
    
    @patch("utils.paths.detect_platform")
    def test_get_platform_paths_kaggle(self, mock_detect):
        """Test get_platform_paths for Kaggle platform."""
        mock_detect.return_value = "kaggle"
        
        paths = get_platform_paths()
        assert paths["data_root"] == "/kaggle/input"
        assert paths["output_root"] == "/kaggle/working"
        assert paths["cache_root"] == "/kaggle/working/cache"
    
    @patch("utils.paths.detect_platform")
    def test_get_platform_paths_unknown(self, mock_detect):
        """Test get_platform_paths for unknown platform."""
        mock_detect.return_value = "unknown_platform"
        
        paths = get_platform_paths()
        # Should fall back to local paths
        assert paths["data_root"] == "./data"
        assert paths["output_root"] == "./outputs"


@pytest.mark.integration
class TestIntegration:
    """Integration tests for path utilities."""
    
    def test_full_path_workflow(self, temp_dir):
        """Test complete path workflow."""
        # Mock project root to use temp directory
        with patch("utils.paths.get_project_root", return_value=temp_dir):
            # Create experiment directory
            exp_path = get_experiment_path("test_experiment", create=True)
            assert exp_path.exists()
            
            # Create checkpoint path
            checkpoint_path = get_checkpoint_path("test_experiment", "test_model", create=True)
            assert checkpoint_path.parent.exists()
            
            # Create results path
            results_path = get_results_path("test_experiment", "test_results", create=True)
            assert results_path.parent.exists()
            
            # Create log path
            log_path = get_log_path("test_experiment", "test_log", create=True)
            assert log_path.parent.exists()
            
            # All should be under the experiment directory
            assert exp_path in checkpoint_path.parents
            assert exp_path in results_path.parents
            assert exp_path in log_path.parents
    
    def test_path_resolution_consistency(self, temp_dir):
        """Test that path resolution is consistent across calls."""
        test_path = "data/test/file.txt"
        
        result1 = resolve_path(test_path)
        result2 = resolve_path(test_path)
        
        assert result1 == result2
        assert isinstance(result1, Path)
        assert isinstance(result2, Path)
    
    def test_directory_creation_and_cleanup(self, temp_dir):
        """Test directory creation and cleanup workflow."""
        # Create directory
        test_dir = temp_dir / "test_workflow"
        ensure_directory(test_dir)
        assert test_dir.exists()
        
        # Add some files
        (test_dir / "file1.txt").touch()
        (test_dir / "file2.txt").touch()
        (test_dir / "file3.py").touch()
        
        # Clean specific files
        removed_count = clean_directory(test_dir, "*.txt")
        assert removed_count == 2
        assert not (test_dir / "file1.txt").exists()
        assert not (test_dir / "file2.txt").exists()
        assert (test_dir / "file3.py").exists()
        
        # Clean all files
        removed_count = clean_directory(test_dir, "*")
        assert removed_count == 1
        assert not (test_dir / "file3.py").exists()
        assert test_dir.exists()  # Directory should still exist