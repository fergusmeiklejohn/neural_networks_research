#!/usr/bin/env python
"""
Check if we're in the correct conda environment with all required packages.
"""

import subprocess
import sys


def check_environment():
    """Check current environment and packages"""
    print("=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)

    # Check Python version and path
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")

    # Check if we're in a conda environment
    try:
        result = subprocess.run(
            ["conda", "info", "--envs"], capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"\nConda environments:")
            print(result.stdout)
        else:
            print(f"\nConda not available or error: {result.stderr}")
    except FileNotFoundError:
        print("\nConda command not found")

    # Check current environment
    import os

    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "Not set")
    print(f"\nCurrent conda environment: {conda_env}")

    # Check required packages
    required_packages = [
        "keras",
        "torch",
        "jax",
        "numpy",
        "scipy",
        "matplotlib",
        "pymunk",
        "pygame",
        "transformers",
        "wandb",
        "tqdm",
    ]

    print(f"\nChecking required packages:")
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ❌ {package} - MISSING")
            missing_packages.append(package)

    # Summary
    print(f"\n" + "=" * 60)
    if missing_packages:
        print(f"❌ Environment check FAILED")
        print(f"Missing packages: {', '.join(missing_packages)}")
        print(f"\nTo fix this:")
        print(f"1. Activate the conda environment: conda activate dist-invention")
        print(f"2. Install missing packages or run the setup script")
        return False
    else:
        print(f"✅ Environment check PASSED")
        print(f"All required packages are available!")
        if conda_env == "dist-invention":
            print(f"You're in the correct conda environment!")
        else:
            print(
                f"Warning: Expected 'dist-invention' environment, but you're in '{conda_env}'"
            )
        return True


if __name__ == "__main__":
    check_environment()
