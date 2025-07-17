"""
Utilities package for neural networks research project.

This package provides centralized utilities for:
- Import management
- Configuration handling
- Path resolution
- Common functionality across experiments

Author: Fergus Meiklejohn
"""

from .config import get_config, setup_environment, validate_keras_backend
from .imports import safe_import, setup_project_paths
from .paths import get_project_root, resolve_path, get_data_path, get_output_path

__all__ = [
    "get_config",
    "setup_environment", 
    "validate_keras_backend",
    "safe_import",
    "setup_project_paths",
    "get_project_root",
    "resolve_path",
    "get_data_path",
    "get_output_path",
]