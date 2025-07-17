"""
Tests for utils.imports module.

Tests import management, path setup, and module loading functionality.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from utils.imports import (
    get_project_root,
    setup_project_paths,
    safe_import,
    safe_import_from,
    import_keras_backend,
    import_torch,
    import_jax,
    import_scientific_stack,
    check_environment,
    validate_imports,
    get_available_backends,
    standard_ml_imports,
)


class TestProjectRoot:
    """Test project root detection."""
    
    def test_get_project_root_finds_markers(self):
        """Test that project root is found using marker files."""
        root = get_project_root()
        assert isinstance(root, Path)
        assert root.exists()
        
        # Should find at least one marker file
        markers = ["pyproject.toml", "setup.py", "CLAUDE.md", ".git"]
        assert any((root / marker).exists() for marker in markers)
    
    def test_get_project_root_consistent(self):
        """Test that project root detection is consistent."""
        root1 = get_project_root()
        root2 = get_project_root()
        assert root1 == root2


class TestPathSetup:
    """Test Python path setup functionality."""
    
    def test_setup_project_paths_adds_to_sys_path(self):
        """Test that setup_project_paths adds directories to sys.path."""
        original_path = sys.path.copy()
        
        try:
            setup_project_paths()
            
            # Check that project root was added
            project_root = get_project_root()
            assert str(project_root) in sys.path
            
            # Check that subdirectories were added if they exist
            subdirs = ["models", "utils", "scripts", "experiments"]
            for subdir in subdirs:
                subdir_path = project_root / subdir
                if subdir_path.exists():
                    assert str(subdir_path) in sys.path
        
        finally:
            sys.path = original_path
    
    def test_setup_project_paths_no_duplicates(self):
        """Test that setup_project_paths doesn't add duplicates."""
        original_path = sys.path.copy()
        
        try:
            setup_project_paths()
            first_length = len(sys.path)
            
            setup_project_paths()
            second_length = len(sys.path)
            
            assert first_length == second_length
        
        finally:
            sys.path = original_path


class TestSafeImport:
    """Test safe import functionality."""
    
    def test_safe_import_existing_module(self):
        """Test importing an existing module."""
        os_module = safe_import("os")
        assert os_module is not None
        assert os_module.__name__ == "os"
    
    def test_safe_import_nonexistent_module(self):
        """Test importing a nonexistent module."""
        result = safe_import("nonexistent_module_12345")
        assert result is None
    
    def test_safe_import_from_existing(self):
        """Test importing from an existing module."""
        path_class = safe_import_from("pathlib", "Path")
        assert path_class is not None
        assert path_class.__name__ == "Path"
    
    def test_safe_import_from_nonexistent_module(self):
        """Test importing from a nonexistent module."""
        result = safe_import_from("nonexistent_module", "SomeClass")
        assert result is None
    
    def test_safe_import_from_nonexistent_item(self):
        """Test importing nonexistent item from existing module."""
        result = safe_import_from("os", "NonexistentFunction")
        assert result is None


class TestBackendImports:
    """Test ML backend import functionality."""
    
    @patch("utils.imports.safe_import")
    def test_import_keras_backend_success(self, mock_safe_import):
        """Test successful Keras import."""
        mock_keras = MagicMock()
        mock_keras.backend.backend.return_value = "jax"
        mock_safe_import.return_value = mock_keras
        
        with patch("utils.imports.importlib.import_module", return_value=mock_keras):
            result = import_keras_backend()
            assert result is not None
    
    @patch("utils.imports.safe_import")
    def test_import_keras_backend_failure(self, mock_safe_import):
        """Test failed Keras import."""
        mock_safe_import.side_effect = ImportError("Keras not available")
        
        with patch("utils.imports.importlib.import_module", side_effect=ImportError()):
            result = import_keras_backend()
            assert result is None
    
    @patch("utils.imports.safe_import")
    def test_import_torch_success(self, mock_safe_import):
        """Test successful PyTorch import."""
        mock_torch = MagicMock()
        mock_torch.__version__ = "2.0.0"
        mock_safe_import.return_value = mock_torch
        
        result = import_torch()
        assert result is not None
    
    @patch("utils.imports.safe_import")
    def test_import_torch_failure(self, mock_safe_import):
        """Test failed PyTorch import."""
        mock_safe_import.return_value = None
        
        result = import_torch()
        assert result is None
    
    @patch("utils.imports.safe_import")
    def test_import_jax_success(self, mock_safe_import):
        """Test successful JAX import."""
        mock_jax = MagicMock()
        mock_jax.__version__ = "0.4.0"
        mock_safe_import.return_value = mock_jax
        
        result = import_jax()
        assert result is not None
    
    @patch("utils.imports.safe_import")
    def test_import_jax_failure(self, mock_safe_import):
        """Test failed JAX import."""
        mock_safe_import.return_value = None
        
        result = import_jax()
        assert result is None


class TestScientificStack:
    """Test scientific computing stack imports."""
    
    @patch("utils.imports.safe_import")
    def test_import_scientific_stack_success(self, mock_safe_import):
        """Test successful scientific stack import."""
        mock_modules = []
        for i in range(5):  # numpy, scipy, matplotlib, pandas, sklearn
            mock_module = MagicMock()
            mock_module.__version__ = "1.0.0"
            mock_modules.append(mock_module)
        
        mock_safe_import.side_effect = mock_modules
        
        result = import_scientific_stack()
        assert len(result) == 5
        assert all(module is not None for module in result)
    
    @patch("utils.imports.safe_import")
    def test_import_scientific_stack_partial_failure(self, mock_safe_import):
        """Test scientific stack import with some failures."""
        # Some modules succeed, some fail
        mock_safe_import.side_effect = [
            MagicMock(__version__="1.0.0"),  # numpy
            None,  # scipy fails
            MagicMock(__version__="3.0.0"),  # matplotlib
            None,  # pandas fails
            MagicMock(__version__="1.0.0"),  # sklearn
        ]
        
        result = import_scientific_stack()
        assert len(result) == 5
        assert result[0] is not None  # numpy
        assert result[1] is None      # scipy
        assert result[2] is not None  # matplotlib
        assert result[3] is None      # pandas
        assert result[4] is not None  # sklearn


class TestEnvironmentChecking:
    """Test environment checking functionality."""
    
    @patch("utils.imports.safe_import")
    def test_check_environment_structure(self, mock_safe_import):
        """Test check_environment returns correct structure."""
        mock_safe_import.return_value = MagicMock(__version__="1.0.0")
        
        env_info = check_environment()
        
        assert isinstance(env_info, dict)
        assert "python_version" in env_info
        assert "project_root" in env_info
        assert "python_path" in env_info
        assert "available_modules" in env_info
        
        assert isinstance(env_info["available_modules"], dict)
    
    @patch("utils.imports.safe_import")
    def test_check_environment_module_availability(self, mock_safe_import):
        """Test check_environment correctly reports module availability."""
        def mock_import_side_effect(module_name):
            if module_name in ["numpy", "scipy"]:
                mock_module = MagicMock()
                mock_module.__version__ = "1.0.0"
                return mock_module
            return None
        
        mock_safe_import.side_effect = mock_import_side_effect
        
        env_info = check_environment()
        
        # numpy and scipy should be available
        assert env_info["available_modules"]["numpy"]["available"] is True
        assert env_info["available_modules"]["scipy"]["available"] is True
        
        # Others should not be available
        assert env_info["available_modules"]["keras"]["available"] is False
        assert env_info["available_modules"]["torch"]["available"] is False
    
    @patch("utils.imports.safe_import")
    def test_validate_imports_success(self, mock_safe_import):
        """Test validate_imports with all required modules available."""
        mock_safe_import.return_value = MagicMock(__version__="1.0.0")
        
        result = validate_imports()
        assert result is True
    
    @patch("utils.imports.safe_import")
    def test_validate_imports_failure(self, mock_safe_import):
        """Test validate_imports with missing required modules."""
        def mock_import_side_effect(module_name):
            if module_name == "numpy":
                return None  # numpy missing
            return MagicMock(__version__="1.0.0")
        
        mock_safe_import.side_effect = mock_import_side_effect
        
        result = validate_imports()
        assert result is False
    
    @patch("utils.imports.safe_import")
    def test_get_available_backends(self, mock_safe_import):
        """Test get_available_backends functionality."""
        def mock_import_side_effect(module_name):
            if module_name in ["keras", "torch"]:
                return MagicMock()
            return None
        
        mock_safe_import.side_effect = mock_import_side_effect
        
        backends = get_available_backends()
        
        assert "keras" in backends
        assert "torch" in backends
        assert "jax" not in backends
        assert "tensorflow" not in backends


class TestStandardMLImports:
    """Test standard ML imports functionality."""
    
    @patch("utils.imports.safe_import")
    @patch("utils.imports.safe_import_from")
    @patch("utils.imports.import_keras_backend")
    @patch("utils.imports.import_torch")
    @patch("utils.imports.import_jax")
    def test_standard_ml_imports_success(self, mock_jax, mock_torch, mock_keras, mock_from, mock_import):
        """Test successful standard ML imports."""
        # Mock successful imports
        mock_numpy = MagicMock()
        mock_pandas = MagicMock()
        mock_pyplot = MagicMock()
        mock_seaborn = MagicMock()
        mock_keras_mod = MagicMock()
        mock_torch_mod = MagicMock()
        mock_jax_mod = MagicMock()
        
        mock_import.side_effect = lambda name: {
            "numpy": mock_numpy,
            "pandas": mock_pandas,
            "seaborn": mock_seaborn,
        }.get(name)
        
        mock_from.return_value = mock_pyplot
        mock_keras.return_value = mock_keras_mod
        mock_torch.return_value = mock_torch_mod
        mock_jax.return_value = mock_jax_mod
        
        imports = standard_ml_imports()
        
        assert "np" in imports
        assert "pd" in imports
        assert "plt" in imports
        assert "sns" in imports
        assert "keras" in imports
        assert "torch" in imports
        assert "jax" in imports
        assert "project_root" in imports
        
        # None values should be filtered out
        assert all(v is not None for v in imports.values())
    
    @patch("utils.imports.safe_import")
    @patch("utils.imports.safe_import_from")
    @patch("utils.imports.import_keras_backend")
    @patch("utils.imports.import_torch")
    @patch("utils.imports.import_jax")
    def test_standard_ml_imports_partial_failure(self, mock_jax, mock_torch, mock_keras, mock_from, mock_import):
        """Test standard ML imports with some failures."""
        # Some imports succeed, some fail
        mock_numpy = MagicMock()
        
        mock_import.side_effect = lambda name: {
            "numpy": mock_numpy,
            "pandas": None,
            "seaborn": None,
        }.get(name)
        
        mock_from.return_value = None  # matplotlib fails
        mock_keras.return_value = MagicMock()
        mock_torch.return_value = None  # torch fails
        mock_jax.return_value = None    # jax fails
        
        imports = standard_ml_imports()
        
        # Only successful imports should be included
        assert "np" in imports
        assert "keras" in imports
        assert "project_root" in imports
        
        # Failed imports should not be included
        assert "pd" not in imports
        assert "plt" not in imports
        assert "sns" not in imports
        assert "torch" not in imports
        assert "jax" not in imports


@pytest.mark.integration
class TestIntegration:
    """Integration tests for import management."""
    
    def test_full_import_workflow(self):
        """Test complete import workflow."""
        # Set up project paths
        setup_project_paths()
        
        # Check environment
        env_info = check_environment()
        assert isinstance(env_info, dict)
        
        # Validate imports
        # Note: This might fail in CI environments without all dependencies
        try:
            validate_imports()
        except Exception:
            pytest.skip("Required modules not available in test environment")
        
        # Get standard imports
        imports = standard_ml_imports()
        assert isinstance(imports, dict)
        assert "project_root" in imports
    
    def test_project_paths_persistence(self):
        """Test that project paths persist across import calls."""
        original_path = sys.path.copy()
        
        try:
            setup_project_paths()
            first_length = len(sys.path)
            
            # Import some modules
            safe_import("os")
            safe_import_from("pathlib", "Path")
            
            # Path should remain the same
            assert len(sys.path) == first_length
            
            # Call setup again
            setup_project_paths()
            assert len(sys.path) == first_length
        
        finally:
            sys.path = original_path