"""
Centralized import management for neural networks research project.

This module provides utilities to handle imports consistently across the project,
eliminating the need for scattered sys.path.append calls and ensuring proper
module resolution in different environments (local, cloud, etc.).

Author: Fergus Meiklejohn
"""

import importlib
import logging
import sys
from pathlib import Path
from typing import Any, List, Optional

# Set up logging
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path: The project root directory path.
    """
    # Try to find the project root by looking for key files
    current_path = Path(__file__).parent

    # Search upward for project markers
    markers = ["pyproject.toml", "setup.py", "CLAUDE.md", ".git"]

    for parent in [current_path] + list(current_path.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent

    # Fallback to current directory
    return Path.cwd()


def setup_project_paths() -> None:
    """
    Set up Python path to include all necessary project directories.

    This should be called at the beginning of scripts to ensure proper imports.
    """
    project_root = get_project_root()

    # Add project root to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Add key subdirectories to path
    subdirs = ["models", "utils", "scripts", "experiments"]

    for subdir in subdirs:
        subdir_path = project_root / subdir
        if subdir_path.exists() and str(subdir_path) not in sys.path:
            sys.path.insert(0, str(subdir_path))

    logger.info(f"Project paths set up with root: {project_root}")


def safe_import(module_name: str, package: Optional[str] = None) -> Optional[Any]:
    """
    Safely import a module with proper error handling.

    Args:
        module_name: Name of the module to import
        package: Package name for relative imports

    Returns:
        The imported module or None if import failed
    """
    try:
        return importlib.import_module(module_name, package)
    except ImportError as e:
        logger.warning(f"Failed to import {module_name}: {e}")
        return None


def safe_import_from(
    module_name: str, item_name: str, package: Optional[str] = None
) -> Optional[Any]:
    """
    Safely import a specific item from a module.

    Args:
        module_name: Name of the module to import from
        item_name: Name of the item to import
        package: Package name for relative imports

    Returns:
        The imported item or None if import failed
    """
    try:
        module = importlib.import_module(module_name, package)
        return getattr(module, item_name)
    except (ImportError, AttributeError) as e:
        logger.warning(f"Failed to import {item_name} from {module_name}: {e}")
        return None


def import_keras_backend() -> Optional[Any]:
    """
    Import Keras with proper backend configuration.

    Returns:
        keras module or None if import failed
    """
    # Set up Keras backend before importing
    from .config import setup_keras_backend

    setup_keras_backend()

    try:
        import keras

        logger.info(
            f"Keras imported successfully with backend: {keras.backend.backend()}"
        )
        return keras
    except ImportError as e:
        logger.error(f"Failed to import Keras: {e}")
        return None


def import_torch() -> Optional[Any]:
    """
    Import PyTorch with proper error handling.

    Returns:
        torch module or None if import failed
    """
    try:
        import torch

        logger.info(f"PyTorch imported successfully, version: {torch.__version__}")
        return torch
    except ImportError as e:
        logger.warning(f"Failed to import PyTorch: {e}")
        return None


def import_jax() -> Optional[Any]:
    """
    Import JAX with proper error handling.

    Returns:
        jax module or None if import failed
    """
    try:
        import jax

        logger.info(f"JAX imported successfully, version: {jax.__version__}")
        return jax
    except ImportError as e:
        logger.warning(f"Failed to import JAX: {e}")
        return None


def import_scientific_stack() -> tuple:
    """
    Import common scientific computing libraries.

    Returns:
        Tuple of (numpy, scipy, matplotlib, pandas, sklearn) modules
        None for any that failed to import
    """
    modules = []

    for module_name in ["numpy", "scipy", "matplotlib", "pandas", "sklearn"]:
        module = safe_import(module_name)
        modules.append(module)

        if module is not None:
            version = getattr(module, "__version__", "unknown")
            logger.info(f"{module_name} imported successfully, version: {version}")

    return tuple(modules)


def check_environment() -> dict:
    """
    Check the current environment and available imports.

    Returns:
        Dictionary with information about available modules and environment
    """
    setup_project_paths()

    env_info = {
        "python_version": sys.version,
        "project_root": str(get_project_root()),
        "python_path": sys.path.copy(),
        "available_modules": {},
    }

    # Check key modules
    key_modules = [
        "keras",
        "torch",
        "jax",
        "transformers",
        "wandb",
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "sklearn",
    ]

    for module_name in key_modules:
        module = safe_import(module_name)
        env_info["available_modules"][module_name] = {
            "available": module is not None,
            "version": getattr(module, "__version__", "unknown") if module else None,
        }

    return env_info


def validate_imports() -> bool:
    """
    Validate that all required imports are available.

    Returns:
        True if all required imports are available, False otherwise
    """
    required_modules = ["keras", "numpy", "scipy", "matplotlib"]

    for module_name in required_modules:
        if safe_import(module_name) is None:
            logger.error(f"Required module {module_name} not available")
            return False

    logger.info("All required modules available")
    return True


def get_available_backends() -> List[str]:
    """
    Get list of available ML backends.

    Returns:
        List of available backend names
    """
    backends = []

    if safe_import("keras"):
        backends.append("keras")

    if safe_import("torch"):
        backends.append("torch")

    if safe_import("jax"):
        backends.append("jax")

    if safe_import("tensorflow"):
        backends.append("tensorflow")

    return backends


# Convenience imports that are commonly used
def standard_ml_imports():
    """
    Perform standard ML imports and return commonly used modules.

    Returns:
        Dictionary with commonly used modules
    """
    setup_project_paths()

    imports = {}

    # Core scientific computing
    imports["np"] = safe_import("numpy")
    imports["pd"] = safe_import("pandas")
    imports["plt"] = safe_import_from("matplotlib", "pyplot")
    imports["sns"] = safe_import("seaborn")

    # ML libraries
    imports["keras"] = import_keras_backend()
    imports["torch"] = import_torch()
    imports["jax"] = import_jax()

    # Project-specific imports
    imports["project_root"] = get_project_root()

    # Filter out None values
    imports = {k: v for k, v in imports.items() if v is not None}

    logger.info(f"Standard ML imports loaded: {list(imports.keys())}")
    return imports


# Auto-setup when module is imported
if __name__ != "__main__":
    setup_project_paths()
