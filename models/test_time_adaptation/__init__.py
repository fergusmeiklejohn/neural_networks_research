"""Test-Time Adaptation implementations for neural networks.

This module provides various test-time adaptation techniques including:
- TENT (Test-time Entropy Minimization)
- TTT (Test-Time Training) for physics
- Physics-aware adaptation methods
"""

from .base_tta import BaseTTA
from .tent import TENT, PhysicsTENT
from .ttt_physics import PhysicsTTT

__all__ = ['BaseTTA', 'TENT', 'PhysicsTENT', 'PhysicsTTT']