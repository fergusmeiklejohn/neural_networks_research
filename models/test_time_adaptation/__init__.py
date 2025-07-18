"""Test-Time Adaptation implementations for neural networks.

This module provides various test-time adaptation techniques including:
- TENT (Test-time Entropy Minimization)
- TTT (Test-Time Training) for physics
- Physics-aware adaptation methods
"""

import keras

# Auto-select base class based on backend
if keras.backend.backend() == 'jax':
    from .base_tta_jax import BaseTTAJax as BaseTTA
else:
    from .base_tta import BaseTTA

from .tent import TENT, PhysicsTENT
from .ttt_physics import PhysicsTTT

__all__ = ['BaseTTA', 'TENT', 'PhysicsTENT', 'PhysicsTTT']