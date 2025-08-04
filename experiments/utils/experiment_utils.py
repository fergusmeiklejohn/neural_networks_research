"""Common utilities for experiments"""

import os
import random

import numpy as np
import tensorflow as tf


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
