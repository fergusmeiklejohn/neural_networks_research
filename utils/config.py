"""
Environment configuration utilities for neural networks research project.

This module provides centralized configuration management for:
- Keras backend selection
- Environment variables
- Cloud platform detection
- Development settings

Author: Fergus Meiklejohn
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Set up logging
logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""


def detect_platform() -> str:
    """
    Detect the current platform (local, colab, paperspace, etc.).

    Returns:
        Platform name as string
    """
    # Check for Google Colab
    if "google.colab" in sys.modules:
        return "colab"

    # Check for Paperspace
    if os.path.exists("/notebooks") and os.path.exists("/storage"):
        return "paperspace"

    # Check for Kaggle
    if os.path.exists("/kaggle"):
        return "kaggle"

    # Check for other cloud platforms
    if "COLAB_GPU" in os.environ:
        return "colab"

    if "PAPERSPACE_MACHINE_NAME" in os.environ:
        return "paperspace"

    if "KAGGLE_USER_SECRETS_TOKEN" in os.environ:
        return "kaggle"

    # Default to local
    return "local"


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration based on platform.

    Returns:
        Dictionary with default configuration
    """
    platform = detect_platform()

    config = {
        "platform": platform,
        "keras_backend": "jax",  # Default to JAX
        "data_dir": "./data",
        "output_dir": "./outputs",
        "model_dir": "./models",
        "cache_dir": "./cache",
        "log_level": "INFO",
        "random_seed": 42,
        "mixed_precision": True,
        "memory_growth": True,
    }

    # Platform-specific overrides
    if platform == "colab":
        config.update(
            {
                "data_dir": "/content/data",
                "output_dir": "/content/outputs",
                "cache_dir": "/content/cache",
            }
        )
    elif platform == "paperspace":
        config.update(
            {
                "data_dir": "/notebooks/data",
                "output_dir": "/storage/outputs",
                "cache_dir": "/storage/cache",
            }
        )
    elif platform == "kaggle":
        config.update(
            {
                "data_dir": "/kaggle/input",
                "output_dir": "/kaggle/working",
                "cache_dir": "/kaggle/working/cache",
            }
        )

    return config


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load configuration from file or environment variables.

    Args:
        config_path: Path to configuration file (optional)

    Returns:
        Configuration dictionary
    """
    config = get_default_config()

    # Load from file if provided
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    file_config = json.load(f)
                config.update(file_config)
                logger.info(f"Configuration loaded from {config_path}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

    # Override with environment variables
    env_overrides = {
        "KERAS_BACKEND": "keras_backend",
        "DATA_DIR": "data_dir",
        "OUTPUT_DIR": "output_dir",
        "MODEL_DIR": "model_dir",
        "CACHE_DIR": "cache_dir",
        "LOG_LEVEL": "log_level",
        "RANDOM_SEED": "random_seed",
        "MIXED_PRECISION": "mixed_precision",
        "MEMORY_GROWTH": "memory_growth",
    }

    for env_var, config_key in env_overrides.items():
        if env_var in os.environ:
            value = os.environ[env_var]

            # Type conversion for boolean values
            if config_key in ["mixed_precision", "memory_growth"]:
                value = value.lower() in ["true", "1", "yes", "on"]
            # Type conversion for integer values
            elif config_key == "random_seed":
                try:
                    value = int(value)
                except ValueError:
                    logger.warning(f"Invalid integer value for {env_var}: {value}")
                    continue

            config[config_key] = value
            logger.info(f"Config override from environment: {config_key} = {value}")

    return config


def setup_keras_backend(backend: Optional[str] = None) -> bool:
    """
    Set up Keras backend configuration.

    Args:
        backend: Backend name ('jax', 'tensorflow', 'torch') or None for auto-detection

    Returns:
        True if setup was successful, False otherwise
    """
    if backend is None:
        config = get_default_config()
        backend = config["keras_backend"]

    # Validate backend
    valid_backends = ["jax", "tensorflow", "torch"]
    if backend not in valid_backends:
        logger.error(f"Invalid backend: {backend}. Must be one of {valid_backends}")
        return False

    # Set environment variable before importing Keras
    os.environ["KERAS_BACKEND"] = backend

    # Platform-specific configuration
    platform = detect_platform()

    if backend == "jax":
        # JAX configuration
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

        # Enable Metal acceleration on Mac
        if sys.platform == "darwin":
            os.environ["JAX_PLATFORMS"] = "metal,cpu"

        # Memory growth for JAX
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    elif backend == "tensorflow":
        # TensorFlow configuration
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TF logging

        # Memory growth for TensorFlow
        if platform != "colab":  # Colab handles this automatically
            os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    elif backend == "torch":
        # PyTorch configuration
        if platform == "local" and sys.platform == "darwin":
            # Enable Metal Performance Shaders on Mac
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    logger.info(f"Keras backend configured: {backend}")
    return True


def validate_keras_backend() -> bool:
    """
    Validate that Keras backend is properly configured.

    Returns:
        True if backend is valid, False otherwise
    """
    try:
        import keras

        backend = keras.backend.backend()
        logger.info(f"Keras backend validation: {backend}")
        return True
    except ImportError as e:
        logger.error(f"Failed to import Keras: {e}")
        return False
    except Exception as e:
        logger.error(f"Keras backend validation failed: {e}")
        return False


def setup_logging(level: str = "INFO") -> None:
    """
    Set up logging configuration.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    # Convert string to logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Reduce third-party logging noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("wandb").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    logger.info(f"Logging configured with level: {level}")


def setup_random_seeds(seed: int = 42) -> None:
    """
    Set up random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    import random

    import numpy as np

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Backend-specific seeds
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass

    logger.info(f"Random seeds set to: {seed}")


def setup_memory_growth() -> None:
    """
    Configure memory growth for GPU usage.
    """
    backend = os.environ.get("KERAS_BACKEND", "jax")

    if backend == "tensorflow":
        try:
            import tensorflow as tf

            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("TensorFlow GPU memory growth enabled")
        except ImportError:
            pass

    elif backend == "jax":
        # JAX handles memory growth through environment variables
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        logger.info("JAX memory growth configured")

    elif backend == "torch":
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("PyTorch CUDA cache cleared")
        except ImportError:
            pass


def setup_environment(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Set up the complete environment configuration.

    Args:
        config: Optional configuration dictionary

    Returns:
        Final configuration dictionary
    """
    if config is None:
        config = load_config()

    # Setup logging first
    setup_logging(config["log_level"])

    # Setup random seeds
    setup_random_seeds(config["random_seed"])

    # Setup Keras backend
    setup_keras_backend(config["keras_backend"])

    # Setup memory growth
    if config["memory_growth"]:
        setup_memory_growth()

    # Create directories
    for dir_key in ["data_dir", "output_dir", "model_dir", "cache_dir"]:
        dir_path = Path(config[dir_key])
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ensured: {dir_path}")

    # Validate configuration
    if not validate_keras_backend():
        raise ConfigurationError("Keras backend validation failed")

    logger.info("Environment setup complete")
    return config


def get_config() -> Dict[str, Any]:
    """
    Get the current configuration.

    Returns:
        Current configuration dictionary
    """
    return load_config()


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Configuration saved to {config_path}")


def print_environment_info() -> None:
    """
    Print detailed environment information.
    """
    config = get_config()

    print("=== Environment Configuration ===")
    print(f"Platform: {config['platform']}")
    print(f"Keras Backend: {config['keras_backend']}")
    print(f"Data Directory: {config['data_dir']}")
    print(f"Output Directory: {config['output_dir']}")
    print(f"Python Version: {sys.version}")
    print(f"Python Path: {sys.path}")

    # Print GPU information if available
    try:
        import keras

        print(f"Keras Backend: {keras.backend.backend()}")
    except ImportError:
        print("Keras not available")

    # Print available devices
    backend = config["keras_backend"]
    if backend == "jax":
        try:
            import jax

            print(f"JAX Devices: {jax.devices()}")
        except ImportError:
            print("JAX not available")
    elif backend == "tensorflow":
        try:
            import tensorflow as tf

            print(f"TensorFlow Devices: {tf.config.list_physical_devices()}")
        except ImportError:
            print("TensorFlow not available")
    elif backend == "torch":
        try:
            import torch

            print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"PyTorch CUDA Device: {torch.cuda.get_device_name()}")
        except ImportError:
            print("PyTorch not available")

    print("=" * 35)


if __name__ == "__main__":
    # Test configuration
    print_environment_info()
