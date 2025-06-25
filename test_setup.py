#!/usr/bin/env python
"""Test script to verify the development environment setup."""

import sys
print(f"Python version: {sys.version}")

# Test Keras backend
try:
    import keras
    print(f"Keras version: {keras.__version__}")
    print(f"Keras backend: {keras.backend.backend()}")
except ImportError as e:
    print(f"Error importing Keras: {e}")

# Test JAX
try:
    import jax
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
except ImportError as e:
    print(f"Error importing JAX: {e}")

# Test PyTorch
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
except ImportError as e:
    print(f"Error importing PyTorch: {e}")

# Test TensorFlow
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    print(f"TF devices: {tf.config.list_physical_devices()}")
except ImportError as e:
    print(f"Error importing TensorFlow: {e}")

# Test other key libraries
libraries = [
    'numpy', 'scipy', 'pandas', 'matplotlib', 'sklearn',
    'transformers', 'wandb', 'pymunk', 'sympy'
]

for lib in libraries:
    try:
        module = __import__(lib)
        version = getattr(module, '__version__', 'unknown')
        print(f"{lib}: {version}")
    except ImportError:
        print(f"{lib}: NOT INSTALLED")

print("\nSetup verification complete!")