"""
Path resolution utilities for neural networks research project.

This module provides consistent path handling across different environments
(local, cloud, different operating systems) and eliminates hardcoded paths.

Author: Fergus Meiklejohn
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Union

# Set up logging
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """
    Get the project root directory by looking for key marker files.

    Returns:
        Path object pointing to the project root
    """
    # Start from the current file's directory
    current_path = Path(__file__).parent

    # Look for project markers (in order of preference)
    markers = [
        "pyproject.toml",
        "setup.py",
        "CLAUDE.md",
        ".git",
        "requirements.txt",
        "README.md",
    ]

    # Search upward through parent directories
    for parent in [current_path] + list(current_path.parents):
        if any((parent / marker).exists() for marker in markers):
            logger.debug(f"Project root found at: {parent}")
            return parent

    # Fallback to current working directory
    fallback = Path.cwd()
    logger.warning(f"Project root not found, using current directory: {fallback}")
    return fallback


def resolve_path(path: Union[str, Path], base_path: Optional[Path] = None) -> Path:
    """
    Resolve a path relative to the project root or a specified base path.

    Args:
        path: Path to resolve (can be relative or absolute)
        base_path: Base path to resolve relative to (default: project root)

    Returns:
        Resolved Path object
    """
    path = Path(path)

    # If path is already absolute, return as-is
    if path.is_absolute():
        return path

    # Use project root as base if not specified
    if base_path is None:
        base_path = get_project_root()

    # Resolve relative to base path
    resolved = (base_path / path).resolve()
    logger.debug(f"Path resolved: {path} -> {resolved}")
    return resolved


def get_data_path(subpath: str = "", create: bool = True) -> Path:
    """
    Get path to data directory with optional subdirectory.

    Args:
        subpath: Subdirectory within data directory
        create: Whether to create the directory if it doesn't exist

    Returns:
        Path to data directory
    """
    from utils.config import get_config

    config = get_config()
    data_dir = Path(config["data_dir"])

    # Handle absolute vs relative paths
    if not data_dir.is_absolute():
        data_dir = resolve_path(data_dir)

    # Add subdirectory if specified
    if subpath:
        data_dir = data_dir / subpath

    # Create directory if requested
    if create:
        data_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Data directory ensured: {data_dir}")

    return data_dir


def get_output_path(subpath: str = "", create: bool = True) -> Path:
    """
    Get path to output directory with optional subdirectory.

    Args:
        subpath: Subdirectory within output directory
        create: Whether to create the directory if it doesn't exist

    Returns:
        Path to output directory
    """
    from utils.config import get_config

    config = get_config()
    output_dir = Path(config["output_dir"])

    # Handle absolute vs relative paths
    if not output_dir.is_absolute():
        output_dir = resolve_path(output_dir)

    # Add subdirectory if specified
    if subpath:
        output_dir = output_dir / subpath

    # Create directory if requested
    if create:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Output directory ensured: {output_dir}")

    return output_dir


def get_model_path(subpath: str = "", create: bool = True) -> Path:
    """
    Get path to model directory with optional subdirectory.

    Args:
        subpath: Subdirectory within model directory
        create: Whether to create the directory if it doesn't exist

    Returns:
        Path to model directory
    """
    from utils.config import get_config

    config = get_config()
    model_dir = Path(config["model_dir"])

    # Handle absolute vs relative paths
    if not model_dir.is_absolute():
        model_dir = resolve_path(model_dir)

    # Add subdirectory if specified
    if subpath:
        model_dir = model_dir / subpath

    # Create directory if requested
    if create:
        model_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Model directory ensured: {model_dir}")

    return model_dir


def get_cache_path(subpath: str = "", create: bool = True) -> Path:
    """
    Get path to cache directory with optional subdirectory.

    Args:
        subpath: Subdirectory within cache directory
        create: Whether to create the directory if it doesn't exist

    Returns:
        Path to cache directory
    """
    from utils.config import get_config

    config = get_config()
    cache_dir = Path(config["cache_dir"])

    # Handle absolute vs relative paths
    if not cache_dir.is_absolute():
        cache_dir = resolve_path(cache_dir)

    # Add subdirectory if specified
    if subpath:
        cache_dir = cache_dir / subpath

    # Create directory if requested
    if create:
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Cache directory ensured: {cache_dir}")

    return cache_dir


def get_experiment_path(
    experiment_name: str, subpath: str = "", create: bool = True
) -> Path:
    """
    Get path to a specific experiment directory.

    Args:
        experiment_name: Name of the experiment
        subpath: Subdirectory within experiment directory
        create: Whether to create the directory if it doesn't exist

    Returns:
        Path to experiment directory
    """
    project_root = get_project_root()
    experiment_dir = project_root / "experiments" / experiment_name

    # Add subdirectory if specified
    if subpath:
        experiment_dir = experiment_dir / subpath

    # Create directory if requested
    if create:
        experiment_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Experiment directory ensured: {experiment_dir}")

    return experiment_dir


def get_checkpoint_path(
    experiment_name: str, model_name: str, create: bool = True
) -> Path:
    """
    Get path for saving model checkpoints.

    Args:
        experiment_name: Name of the experiment
        model_name: Name of the model
        create: Whether to create the directory if it doesn't exist

    Returns:
        Path to checkpoint directory
    """
    checkpoint_dir = get_experiment_path(experiment_name, "checkpoints", create)
    return checkpoint_dir / f"{model_name}.keras"


def get_results_path(
    experiment_name: str, results_name: str, create: bool = True
) -> Path:
    """
    Get path for saving experiment results.

    Args:
        experiment_name: Name of the experiment
        results_name: Name of the results file
        create: Whether to create the directory if it doesn't exist

    Returns:
        Path to results file
    """
    results_dir = get_experiment_path(experiment_name, "results", create)
    return results_dir / f"{results_name}.json"


def get_log_path(experiment_name: str, log_name: str, create: bool = True) -> Path:
    """
    Get path for saving experiment logs.

    Args:
        experiment_name: Name of the experiment
        log_name: Name of the log file
        create: Whether to create the directory if it doesn't exist

    Returns:
        Path to log file
    """
    log_dir = get_experiment_path(experiment_name, "logs", create)
    return log_dir / f"{log_name}.log"


def find_files(
    pattern: str, search_path: Optional[Path] = None, recursive: bool = True
) -> List[Path]:
    """
    Find files matching a pattern.

    Args:
        pattern: Glob pattern to search for
        search_path: Directory to search in (default: project root)
        recursive: Whether to search recursively

    Returns:
        List of Path objects matching the pattern
    """
    if search_path is None:
        search_path = get_project_root()

    search_path = Path(search_path)

    if recursive:
        matches = list(search_path.rglob(pattern))
    else:
        matches = list(search_path.glob(pattern))

    logger.debug(
        f"Found {len(matches)} files matching pattern '{pattern}' in {search_path}"
    )
    return matches


def ensure_directory(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to directory

    Returns:
        Path object
    """
    path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Directory ensured: {path}")
    return path


def clean_directory(path: Path, pattern: str = "*") -> int:
    """
    Clean files from a directory matching a pattern.

    Args:
        path: Directory to clean
        pattern: Glob pattern for files to remove

    Returns:
        Number of files removed
    """
    if not path.exists():
        return 0

    files_removed = 0
    for file_path in path.glob(pattern):
        if file_path.is_file():
            file_path.unlink()
            files_removed += 1

    logger.info(f"Cleaned {files_removed} files from {path}")
    return files_removed


def get_relative_path(path: Path, base_path: Optional[Path] = None) -> str:
    """
    Get a path relative to the project root or specified base.

    Args:
        path: Path to make relative
        base_path: Base path (default: project root)

    Returns:
        Relative path as string
    """
    if base_path is None:
        base_path = get_project_root()

    try:
        relative = path.relative_to(base_path)
        return str(relative)
    except ValueError:
        # Path is not relative to base_path
        return str(path)


def validate_path(
    path: Path,
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
) -> bool:
    """
    Validate a path meets certain criteria.

    Args:
        path: Path to validate
        must_exist: Whether path must exist
        must_be_file: Whether path must be a file
        must_be_dir: Whether path must be a directory

    Returns:
        True if path is valid, False otherwise
    """
    if must_exist and not path.exists():
        logger.error(f"Path does not exist: {path}")
        return False

    if must_be_file and not path.is_file():
        logger.error(f"Path is not a file: {path}")
        return False

    if must_be_dir and not path.is_dir():
        logger.error(f"Path is not a directory: {path}")
        return False

    return True


def get_platform_paths() -> dict:
    """
    Get platform-specific paths for different environments.

    Returns:
        Dictionary with platform-specific path information
    """
    from utils.config import detect_platform

    platform = detect_platform()

    platform_paths = {
        "local": {
            "data_root": "./data",
            "output_root": "./outputs",
            "cache_root": "./cache",
            "temp_root": "/tmp" if os.name != "nt" else "C:\\temp",
        },
        "colab": {
            "data_root": "/content/data",
            "output_root": "/content/outputs",
            "cache_root": "/content/cache",
            "temp_root": "/tmp",
        },
        "paperspace": {
            "data_root": "/notebooks/data",
            "output_root": "/storage/outputs",
            "cache_root": "/storage/cache",
            "temp_root": "/tmp",
        },
        "kaggle": {
            "data_root": "/kaggle/input",
            "output_root": "/kaggle/working",
            "cache_root": "/kaggle/working/cache",
            "temp_root": "/tmp",
        },
    }

    return platform_paths.get(platform, platform_paths["local"])


def print_path_info() -> None:
    """
    Print information about current path configuration.
    """
    print("=== Path Configuration ===")
    print(f"Project Root: {get_project_root()}")
    print(f"Data Path: {get_data_path()}")
    print(f"Output Path: {get_output_path()}")
    print(f"Model Path: {get_model_path()}")
    print(f"Cache Path: {get_cache_path()}")

    # Platform-specific paths
    platform_paths = get_platform_paths()
    print(f"Platform Paths: {platform_paths}")

    print("=" * 27)


if __name__ == "__main__":
    # Test path utilities
    print_path_info()
