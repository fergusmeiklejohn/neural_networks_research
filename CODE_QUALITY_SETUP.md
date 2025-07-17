# Python Code Quality Infrastructure

## Overview

This project now includes a comprehensive code quality infrastructure designed to reduce Python debugging cycles from 5-10 iterations to 1-2 iterations. The setup catches errors at development time rather than runtime.

## Key Components

### 1. Configuration Files

- **`pyproject.toml`**: Central configuration for all Python tools (mypy, black, isort, pytest)
- **`.pre-commit-config.yaml`**: Automated quality checks before each commit
- **`.vscode/settings.json`**: Real-time error detection in Cursor/VS Code with Pylance

### 2. Centralized Utilities (`utils/`)

- **`imports.py`**: Eliminates scattered `sys.path.append` calls
- **`config.py`**: Handles environment configuration consistently
- **`paths.py`**: Resolves paths correctly across local/cloud environments

### 3. Testing Infrastructure (`tests/`)

- **`conftest.py`**: Shared pytest fixtures and configuration
- **`test_utils_*.py`**: Comprehensive tests for utility modules

## Installation

```bash
# Install development dependencies
pip install -e ".[dev]"

# Or just the essentials
pip install black isort flake8 mypy pytest pre-commit

# Set up pre-commit hooks
pre-commit install
```

## Usage

### Using Centralized Imports

Instead of:
```python
import sys
sys.path.append('../..')
from models.baseline_models import BaselineModel
```

Use:
```python
from utils.imports import setup_project_paths, safe_import
setup_project_paths()

from models.baseline_models import BaselineModel
```

### Using Configuration Management

```python
from utils.config import setup_environment, get_config

# Set up environment (logging, random seeds, Keras backend, etc.)
config = setup_environment()

# Access configuration values
data_dir = config["data_dir"]
output_dir = config["output_dir"]
```

### Using Path Resolution

```python
from utils.paths import get_data_path, get_output_path, get_experiment_path

# Get paths (automatically creates directories)
data_path = get_data_path("processed/physics_worlds")
output_path = get_output_path("results")
checkpoint_path = get_checkpoint_path("01_physics_worlds", "baseline_model")
```

## Real-Time Error Detection

With the Cursor/VS Code configuration:

1. **Type errors** are caught as you type
2. **Import errors** are highlighted immediately
3. **Undefined variables** are flagged before running
4. **Type hints** provide better autocomplete

### Key Features Enabled:

- **Pylance strict mode**: Maximum type checking
- **Auto-formatting**: Black on save
- **Import sorting**: isort on save
- **Real-time linting**: flake8 and mypy
- **Inlay hints**: Shows inferred types inline

## Pre-Commit Hooks

Before each commit, the following checks run automatically:

1. **Code formatting**: Black
2. **Import sorting**: isort
3. **Linting**: flake8 with plugins
4. **Type checking**: mypy on changed files
5. **Security**: bandit
6. **Documentation**: pydocstyle
7. **Notebook cleaning**: Removes outputs

### Manual Pre-Commit Run

```bash
# Run on all files
pre-commit run --all-files

# Run on specific hook
pre-commit run black --all-files
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=models --cov=utils

# Run specific test file
pytest tests/test_utils_config.py

# Run specific test
pytest tests/test_utils_config.py::TestPlatformDetection::test_detect_local_platform
```

## Common Workflows

### Starting a New Script

```python
#!/usr/bin/env python3
"""
Script description here.
"""

from typing import Dict, List, Optional
import logging

# Set up project paths first
from utils.imports import setup_project_paths
setup_project_paths()

# Import project modules
from utils.config import setup_environment
from utils.paths import get_data_path, get_output_path
from models.baseline_models import BaselineModel

# Set up logging
logger = logging.getLogger(__name__)


def main() -> None:
    """Main function."""
    # Set up environment
    config = setup_environment()
    
    # Your code here
    data_path = get_data_path("processed")
    output_path = get_output_path("results")
    
    logger.info(f"Processing data from {data_path}")
    logger.info(f"Saving results to {output_path}")


if __name__ == "__main__":
    main()
```

### Adding Type Hints

```python
# Before
def process_data(data, config):
    results = []
    for item in data:
        result = transform(item, config["param"])
        results.append(result)
    return results

# After
from typing import List, Dict, Any

def process_data(data: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process data according to configuration.
    
    Args:
        data: List of data items to process
        config: Configuration dictionary with 'param' key
        
    Returns:
        List of processed results
    """
    results: List[Dict[str, Any]] = []
    for item in data:
        result = transform(item, config["param"])
        results.append(result)
    return results
```

## Troubleshooting

### Import Errors

If you get import errors:
1. Ensure you call `setup_project_paths()` at the beginning of your script
2. Check that the module exists in the expected location
3. Verify the module has an `__init__.py` file

### Type Checking Errors

If mypy complains:
1. Add type hints to function signatures
2. Use `# type: ignore` sparingly for third-party libraries
3. Check `pyproject.toml` for mypy configuration

### Pre-Commit Failures

If pre-commit fails:
1. Run `black .` to format code
2. Run `isort .` to sort imports
3. Fix any flake8 errors manually
4. Add type hints for mypy errors

## Benefits

1. **Reduced Debugging Time**: Catch errors before running code
2. **Consistent Code Style**: Automatic formatting
3. **Better IDE Support**: Improved autocomplete and navigation
4. **Cross-Platform Compatibility**: Works on local/cloud environments
5. **Improved Code Quality**: Type safety and better documentation

## Next Steps

To add type hints to existing modules:
1. Start with the most-used modules
2. Add hints to function signatures first
3. Use `reveal_type()` to check inferred types
4. Run `mypy` to verify correctness

Remember: The goal is to catch errors early and make development smoother!