"""
Pytest configuration and fixtures for the neural networks research project.

This file contains shared fixtures and configuration for all tests.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import get_default_config
from utils.paths import get_project_root


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get the project root directory."""
    return get_project_root()


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Get test configuration."""
    config = get_default_config()
    config.update(
        {
            "log_level": "WARNING",  # Reduce logging during tests
            "random_seed": 42,
            "mixed_precision": False,  # Disable for consistent testing
            "memory_growth": False,
        }
    )
    return config


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def test_data_dir(temp_dir: Path) -> Path:
    """Create a test data directory."""
    data_dir = temp_dir / "test_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture
def test_output_dir(temp_dir: Path) -> Path:
    """Create a test output directory."""
    output_dir = temp_dir / "test_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def sample_2d_trajectory() -> np.ndarray:
    """
    Create a sample 2D trajectory for testing.

    Returns:
        Array of shape (timesteps, 2) with x, y coordinates
    """
    timesteps = 100
    t = np.linspace(0, 10, timesteps)

    # Simple parabolic trajectory
    x = t * 2
    y = -0.5 * 9.8 * t**2 + 10 * t + 5

    return np.column_stack([x, y])


@pytest.fixture
def sample_physics_data() -> Dict[str, Any]:
    """
    Create sample physics data for testing.

    Returns:
        Dictionary with physics parameters and trajectory
    """
    return {
        "gravity": 9.8,
        "friction": 0.1,
        "mass": 1.0,
        "initial_velocity": [5.0, 10.0],
        "initial_position": [0.0, 5.0],
        "timestep": 0.01,
        "duration": 10.0,
    }


@pytest.fixture
def sample_model_config() -> Dict[str, Any]:
    """
    Create sample model configuration for testing.

    Returns:
        Dictionary with model parameters
    """
    return {
        "hidden_dim": 64,
        "num_layers": 3,
        "activation": "relu",
        "dropout_rate": 0.1,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
    }


@pytest.fixture
def mock_keras_model():
    """Create a mock Keras model for testing."""
    try:
        from unittest.mock import Mock

        model = Mock()
        model.compile = Mock()
        model.fit = Mock()
        model.predict = Mock(return_value=np.array([[0.5, 0.5]]))
        model.evaluate = Mock(return_value=[0.1, 0.95])  # [loss, accuracy]
        model.save = Mock()
        model.load_weights = Mock()

        return model
    except ImportError:
        pytest.skip("Mock not available")


@pytest.fixture(autouse=True)
def setup_test_environment(test_config: Dict[str, Any], temp_dir: Path):
    """
    Set up test environment before each test.

    This fixture runs automatically before each test and ensures:
    - Consistent random seeds
    - Proper logging configuration
    - Isolated temporary directories
    """
    # Set random seeds
    np.random.seed(test_config["random_seed"])

    # Set up environment variables for testing
    original_env = os.environ.copy()

    # Override environment variables for testing
    os.environ.update(
        {
            "KERAS_BACKEND": "jax",  # Use JAX for testing
            "DATA_DIR": str(temp_dir / "data"),
            "OUTPUT_DIR": str(temp_dir / "outputs"),
            "LOG_LEVEL": "WARNING",
            "MIXED_PRECISION": "false",
            "MEMORY_GROWTH": "false",
        }
    )

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""

    def _create_dataset(
        num_samples: int = 100, input_dim: int = 2, output_dim: int = 1
    ):
        """
        Create a sample dataset.

        Args:
            num_samples: Number of samples
            input_dim: Input dimension
            output_dim: Output dimension

        Returns:
            Tuple of (X, y) arrays
        """
        X = np.random.randn(num_samples, input_dim)
        y = np.random.randn(num_samples, output_dim)
        return X, y

    return _create_dataset


@pytest.fixture
def physics_worlds_data():
    """Create sample physics worlds data for testing."""

    def _create_physics_data(num_trajectories: int = 10, timesteps: int = 100):
        """
        Create sample physics worlds data.

        Args:
            num_trajectories: Number of trajectories
            timesteps: Number of timesteps per trajectory

        Returns:
            Dictionary with trajectories and metadata
        """
        trajectories = []
        metadata = []

        for i in range(num_trajectories):
            # Create a simple trajectory
            t = np.linspace(0, 10, timesteps)
            gravity = 9.8 + np.random.normal(0, 0.1)  # Add some variation

            x = t * 2
            y = -0.5 * gravity * t**2 + 10 * t + 5

            trajectory = np.column_stack([x, y])
            trajectories.append(trajectory)

            metadata.append(
                {
                    "gravity": gravity,
                    "friction": 0.1,
                    "mass": 1.0,
                    "initial_velocity": [5.0, 10.0],
                    "initial_position": [0.0, 5.0],
                }
            )

        return {
            "trajectories": np.array(trajectories),
            "metadata": metadata,
        }

    return _create_physics_data


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line(
        "markers", "cloud: marks tests that require cloud resources"
    )


# Skip GPU tests if no GPU available
def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip GPU tests if no GPU available."""
    gpu_available = False

    try:
        # Check for GPU availability
        import keras

        if keras.backend.backend() == "jax":
            import jax

            # Check for available devices - on Mac it's "metal", not "gpu"
            devices = jax.devices()
            gpu_available = any(d.platform != "cpu" for d in devices)
        elif keras.backend.backend() == "tensorflow":
            import tensorflow as tf

            gpu_available = len(tf.config.list_physical_devices("GPU")) > 0
        elif keras.backend.backend() == "torch":
            import torch

            gpu_available = (
                torch.cuda.is_available() or torch.backends.mps.is_available()
            )
    except (ImportError, RuntimeError):
        # If we can't check, assume no GPU
        pass

    if not gpu_available:
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


# Performance monitoring
@pytest.fixture
def benchmark_timer():
    """Simple benchmark timer for performance tests."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def __enter__(self):
            self.start_time = time.time()
            return self

        def __exit__(self, *args):
            self.end_time = time.time()

        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None

    return Timer
