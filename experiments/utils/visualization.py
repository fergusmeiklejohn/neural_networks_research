"""Visualization utilities for experiments"""

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectories(predicted, actual, title="Trajectory Comparison"):
    """Plot predicted vs actual trajectories"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot predicted
    axes[0].set_title("Predicted")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    
    # Plot actual
    axes[1].set_title("Actual")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    
    fig.suptitle(title)
    plt.tight_layout()
    return fig