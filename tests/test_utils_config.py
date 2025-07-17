"""
Tests for utils.config module.

Tests configuration management, environment detection, and backend setup.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

from utils.config import (
    detect_platform,
    get_default_config,
    load_config,
    setup_keras_backend,
    validate_keras_backend,
    setup_logging,
    setup_random_seeds,
    setup_environment,
    ConfigurationError,
)


class TestPlatformDetection:
    """Test platform detection functionality."""
    
    def test_detect_local_platform(self):
        """Test detection of local platform."""
        platform = detect_platform()
        assert platform == "local"
    
    @patch("sys.modules")
    def test_detect_colab_platform(self, mock_modules):
        """Test detection of Google Colab platform."""
        mock_modules.__contains__ = lambda x: x == "google.colab"
        platform = detect_platform()
        assert platform == "colab"
    
    @patch("os.path.exists")
    def test_detect_paperspace_platform(self, mock_exists):
        """Test detection of Paperspace platform."""
        mock_exists.side_effect = lambda path: path in ["/notebooks", "/storage"]
        platform = detect_platform()
        assert platform == "paperspace"
    
    @patch("os.path.exists")
    def test_detect_kaggle_platform(self, mock_exists):
        """Test detection of Kaggle platform."""
        mock_exists.side_effect = lambda path: path == "/kaggle"
        platform = detect_platform()
        assert platform == "kaggle"


class TestDefaultConfig:
    """Test default configuration generation."""
    
    def test_default_config_structure(self):
        """Test that default config has required keys."""
        config = get_default_config()
        
        required_keys = [
            "platform",
            "keras_backend",
            "data_dir",
            "output_dir",
            "model_dir",
            "cache_dir",
            "log_level",
            "random_seed",
            "mixed_precision",
            "memory_growth",
        ]
        
        for key in required_keys:
            assert key in config
    
    def test_default_config_values(self):
        """Test default configuration values."""
        config = get_default_config()
        
        assert config["keras_backend"] == "jax"
        assert config["log_level"] == "INFO"
        assert config["random_seed"] == 42
        assert isinstance(config["mixed_precision"], bool)
        assert isinstance(config["memory_growth"], bool)
    
    @patch("utils.config.detect_platform")
    def test_platform_specific_config(self, mock_detect):
        """Test platform-specific configuration overrides."""
        mock_detect.return_value = "colab"
        config = get_default_config()
        
        assert config["platform"] == "colab"
        assert config["data_dir"] == "/content/data"
        assert config["output_dir"] == "/content/outputs"


class TestConfigLoading:
    """Test configuration loading from files and environment."""
    
    def test_load_config_default(self):
        """Test loading default configuration."""
        config = load_config()
        assert isinstance(config, dict)
        assert "platform" in config
    
    def test_load_config_from_file(self, temp_dir):
        """Test loading configuration from file."""
        config_file = temp_dir / "config.json"
        test_config = {
            "keras_backend": "tensorflow",
            "log_level": "DEBUG",
            "custom_key": "custom_value"
        }
        
        with open(config_file, "w") as f:
            json.dump(test_config, f)
        
        config = load_config(config_file)
        assert config["keras_backend"] == "tensorflow"
        assert config["log_level"] == "DEBUG"
        assert config["custom_key"] == "custom_value"
    
    def test_load_config_nonexistent_file(self):
        """Test loading config with nonexistent file."""
        config = load_config("/nonexistent/path/config.json")
        # Should still return default config
        assert isinstance(config, dict)
        assert "platform" in config
    
    def test_load_config_invalid_json(self, temp_dir):
        """Test loading config with invalid JSON."""
        config_file = temp_dir / "invalid.json"
        with open(config_file, "w") as f:
            f.write("invalid json content")
        
        config = load_config(config_file)
        # Should still return default config
        assert isinstance(config, dict)
        assert "platform" in config
    
    def test_environment_variable_overrides(self):
        """Test environment variable overrides."""
        with patch.dict(os.environ, {
            "KERAS_BACKEND": "torch",
            "LOG_LEVEL": "DEBUG",
            "RANDOM_SEED": "123",
            "MIXED_PRECISION": "true"
        }):
            config = load_config()
            assert config["keras_backend"] == "torch"
            assert config["log_level"] == "DEBUG"
            assert config["random_seed"] == 123
            assert config["mixed_precision"] is True


class TestKerasBackend:
    """Test Keras backend configuration."""
    
    def test_setup_keras_backend_valid(self):
        """Test setting up valid Keras backend."""
        assert setup_keras_backend("jax") is True
        assert os.environ.get("KERAS_BACKEND") == "jax"
    
    def test_setup_keras_backend_invalid(self):
        """Test setting up invalid Keras backend."""
        assert setup_keras_backend("invalid") is False
    
    def test_setup_keras_backend_default(self):
        """Test setting up default Keras backend."""
        assert setup_keras_backend() is True
        assert os.environ.get("KERAS_BACKEND") in ["jax", "tensorflow", "torch"]
    
    @patch("utils.config.sys.platform", "darwin")
    def test_jax_metal_on_mac(self):
        """Test JAX Metal configuration on macOS."""
        setup_keras_backend("jax")
        assert os.environ.get("JAX_PLATFORMS") == "metal,cpu"
    
    def test_tensorflow_configuration(self):
        """Test TensorFlow backend configuration."""
        setup_keras_backend("tensorflow")
        assert os.environ.get("TF_CPP_MIN_LOG_LEVEL") == "2"
    
    @patch("utils.config.detect_platform")
    def test_platform_specific_backend_config(self, mock_detect):
        """Test platform-specific backend configuration."""
        mock_detect.return_value = "local"
        setup_keras_backend("tensorflow")
        assert os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") == "true"


class TestValidation:
    """Test configuration validation."""
    
    @patch("utils.config.safe_import")
    def test_validate_keras_backend_success(self, mock_import):
        """Test successful Keras backend validation."""
        mock_keras = MagicMock()
        mock_keras.backend.backend.return_value = "jax"
        mock_import.return_value = mock_keras
        
        with patch("utils.config.importlib.import_module", return_value=mock_keras):
            assert validate_keras_backend() is True
    
    @patch("utils.config.safe_import")
    def test_validate_keras_backend_failure(self, mock_import):
        """Test failed Keras backend validation."""
        mock_import.side_effect = ImportError("Keras not available")
        
        with patch("utils.config.importlib.import_module", side_effect=ImportError()):
            assert validate_keras_backend() is False


class TestLogging:
    """Test logging configuration."""
    
    def test_setup_logging_info(self):
        """Test setting up INFO level logging."""
        setup_logging("INFO")
        # Just test that it doesn't raise an exception
        assert True
    
    def test_setup_logging_debug(self):
        """Test setting up DEBUG level logging."""
        setup_logging("DEBUG")
        assert True
    
    def test_setup_logging_invalid_level(self):
        """Test setting up logging with invalid level."""
        with pytest.raises(ValueError):
            setup_logging("INVALID")


class TestRandomSeeds:
    """Test random seed configuration."""
    
    def test_setup_random_seeds(self):
        """Test setting up random seeds."""
        setup_random_seeds(42)
        
        # Test that seeds are set consistently
        import random
        import numpy as np
        
        r1 = random.random()
        n1 = np.random.random()
        
        setup_random_seeds(42)
        
        r2 = random.random()
        n2 = np.random.random()
        
        assert r1 == r2
        assert n1 == n2
    
    def test_setup_random_seeds_different_values(self):
        """Test setting up different random seeds."""
        setup_random_seeds(42)
        import random
        r1 = random.random()
        
        setup_random_seeds(123)
        r2 = random.random()
        
        assert r1 != r2


class TestEnvironmentSetup:
    """Test complete environment setup."""
    
    @patch("utils.config.validate_keras_backend")
    @patch("utils.config.setup_memory_growth")
    @patch("utils.config.setup_keras_backend")
    @patch("utils.config.setup_random_seeds")
    @patch("utils.config.setup_logging")
    def test_setup_environment_success(self, mock_logging, mock_seeds, mock_keras, mock_memory, mock_validate):
        """Test successful environment setup."""
        mock_validate.return_value = True
        
        config = setup_environment()
        
        assert isinstance(config, dict)
        mock_logging.assert_called_once()
        mock_seeds.assert_called_once()
        mock_keras.assert_called_once()
        mock_memory.assert_called_once()
        mock_validate.assert_called_once()
    
    @patch("utils.config.validate_keras_backend")
    def test_setup_environment_validation_failure(self, mock_validate):
        """Test environment setup with validation failure."""
        mock_validate.return_value = False
        
        with pytest.raises(ConfigurationError):
            setup_environment()
    
    def test_setup_environment_creates_directories(self, temp_dir):
        """Test that environment setup creates required directories."""
        config = {
            "data_dir": str(temp_dir / "data"),
            "output_dir": str(temp_dir / "outputs"),
            "model_dir": str(temp_dir / "models"),
            "cache_dir": str(temp_dir / "cache"),
            "log_level": "WARNING",
            "random_seed": 42,
            "keras_backend": "jax",
            "memory_growth": False,
        }
        
        with patch("utils.config.validate_keras_backend", return_value=True):
            result_config = setup_environment(config)
        
        assert Path(result_config["data_dir"]).exists()
        assert Path(result_config["output_dir"]).exists()
        assert Path(result_config["model_dir"]).exists()
        assert Path(result_config["cache_dir"]).exists()


@pytest.mark.integration
class TestIntegration:
    """Integration tests for configuration system."""
    
    def test_full_configuration_workflow(self, temp_dir):
        """Test complete configuration workflow."""
        # Create a config file
        config_file = temp_dir / "config.json"
        test_config = {
            "keras_backend": "jax",
            "log_level": "WARNING",
            "data_dir": str(temp_dir / "data"),
            "output_dir": str(temp_dir / "outputs"),
            "model_dir": str(temp_dir / "models"),
            "cache_dir": str(temp_dir / "cache"),
        }
        
        with open(config_file, "w") as f:
            json.dump(test_config, f)
        
        # Load and set up environment
        config = load_config(config_file)
        
        with patch("utils.config.validate_keras_backend", return_value=True):
            final_config = setup_environment(config)
        
        # Verify everything is set up correctly
        assert final_config["keras_backend"] == "jax"
        assert Path(final_config["data_dir"]).exists()
        assert Path(final_config["output_dir"]).exists()
        assert os.environ.get("KERAS_BACKEND") == "jax"