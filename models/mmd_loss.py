"""
Maximum Mean Discrepancy (MMD) Loss for Distribution Matching

Based on CGNN paper insights for comparing generated and target distributions.
Used to ensure distribution invention maintains statistical properties.
"""

import keras
import keras.ops as ops
import numpy as np
from typing import List, Optional, Union


class MMDLoss(keras.losses.Loss):
    """
    Maximum Mean Discrepancy loss for comparing distributions.
    
    Uses multi-bandwidth RBF kernel for robustness across scales.
    """
    
    def __init__(self,
                 kernel: str = 'rbf',
                 bandwidths: Optional[List[float]] = None,
                 name: str = 'mmd_loss'):
        """
        Args:
            kernel: Kernel type ('rbf' or 'linear')
            bandwidths: List of bandwidths for RBF kernel
            name: Loss name
        """
        super().__init__(name=name)
        self.kernel = kernel
        self.bandwidths = bandwidths or [0.01, 0.1, 1.0, 10.0, 100.0]
        
    def compute_kernel_matrix(self, X: keras.KerasTensor, Y: keras.KerasTensor) -> keras.KerasTensor:
        """
        Compute kernel matrix between samples.
        
        Args:
            X: First set of samples (batch_size, features)
            Y: Second set of samples (batch_size, features)
            
        Returns:
            Kernel matrix (batch_size, batch_size)
        """
        if self.kernel == 'rbf':
            # Compute pairwise squared distances
            X_sqnorms = ops.sum(ops.square(X), axis=1, keepdims=True)
            Y_sqnorms = ops.sum(ops.square(Y), axis=1, keepdims=True)
            XY = ops.matmul(X, ops.transpose(Y))
            
            # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
            dists = X_sqnorms + ops.transpose(Y_sqnorms) - 2 * XY
            
            # Multi-bandwidth RBF kernel
            kernel_val = ops.zeros_like(dists)
            for bandwidth in self.bandwidths:
                kernel_val += ops.exp(-dists / (2 * bandwidth ** 2))
                
            return kernel_val / len(self.bandwidths)
            
        elif self.kernel == 'linear':
            return ops.matmul(X, ops.transpose(Y))
            
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """
        Compute MMD loss between true and predicted distributions.
        
        Args:
            y_true: True samples (batch_size, features)
            y_pred: Generated samples (batch_size, features)
            
        Returns:
            MMD loss value
        """
        # Compute kernel matrices
        XX = self.compute_kernel_matrix(y_true, y_true)
        YY = self.compute_kernel_matrix(y_pred, y_pred)
        XY = self.compute_kernel_matrix(y_true, y_pred)
        
        # Remove diagonal elements (kernel of sample with itself)
        batch_size = ops.shape(XX)[0]
        mask = 1.0 - ops.eye(batch_size)
        
        XX = XX * mask
        YY = YY * mask
        
        # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
        # Normalize by number of pairs
        n_pairs = ops.cast(batch_size * (batch_size - 1), dtype='float32')
        
        mmd = (ops.sum(XX) / n_pairs + 
               ops.sum(YY) / n_pairs - 
               2 * ops.sum(XY) / ops.cast(batch_size ** 2, dtype='float32'))
        
        # Return square root for MMD (not MMD^2)
        return ops.sqrt(ops.maximum(mmd, 0.0))


class ConditionalMMDLoss(keras.losses.Loss):
    """
    Conditional MMD loss for comparing distributions given context.
    
    Useful for physics models where we want to match distributions
    conditioned on initial conditions or parameters.
    """
    
    def __init__(self,
                 context_weight: float = 0.5,
                 kernel: str = 'rbf',
                 bandwidths: Optional[List[float]] = None,
                 name: str = 'conditional_mmd_loss'):
        """
        Args:
            context_weight: Weight for context similarity (0-1)
            kernel: Kernel type
            bandwidths: RBF bandwidths
            name: Loss name
        """
        super().__init__(name=name)
        self.context_weight = context_weight
        self.base_mmd = MMDLoss(kernel=kernel, bandwidths=bandwidths)
        
    def call(self,
             y_true: keras.KerasTensor,
             y_pred: keras.KerasTensor,
             context: Optional[keras.KerasTensor] = None) -> keras.KerasTensor:
        """
        Compute conditional MMD loss.
        
        Args:
            y_true: True samples
            y_pred: Generated samples
            context: Context/condition variables
            
        Returns:
            Conditional MMD loss
        """
        # Base MMD on outputs
        output_mmd = self.base_mmd(y_true, y_pred)
        
        if context is None:
            return output_mmd
            
        # Concatenate context with outputs for conditional comparison
        y_true_with_context = ops.concatenate([y_true, context], axis=-1)
        y_pred_with_context = ops.concatenate([y_pred, context], axis=-1)
        
        # Conditional MMD
        conditional_mmd = self.base_mmd(y_true_with_context, y_pred_with_context)
        
        # Weighted combination
        return (1 - self.context_weight) * output_mmd + self.context_weight * conditional_mmd


def create_physics_loss_with_mmd(base_loss: keras.losses.Loss,
                                mmd_weight: float = 0.1,
                                bandwidths: Optional[List[float]] = None) -> keras.losses.Loss:
    """
    Create composite loss combining physics loss with MMD regularization.
    
    Args:
        base_loss: Base physics loss (e.g., MSE for trajectories)
        mmd_weight: Weight for MMD term
        bandwidths: RBF kernel bandwidths
        
    Returns:
        Composite loss function
    """
    mmd_loss = MMDLoss(bandwidths=bandwidths)
    
    def composite_loss(y_true, y_pred):
        # Base physics loss
        physics_loss = base_loss(y_true, y_pred)
        
        # MMD regularization
        mmd_term = mmd_loss(y_true, y_pred)
        
        # Weighted combination
        return physics_loss + mmd_weight * mmd_term
        
    return composite_loss


# Example usage for physics trajectory matching
if __name__ == "__main__":
    # Create sample data
    batch_size = 32
    trajectory_length = 10
    features = 4  # x, y, vx, vy
    
    # Simulate true and predicted trajectories
    true_trajectories = np.random.randn(batch_size, trajectory_length * features)
    pred_trajectories = true_trajectories + 0.1 * np.random.randn(batch_size, trajectory_length * features)
    
    # Create MMD loss
    mmd_loss = MMDLoss()
    
    # Compute loss
    loss_value = mmd_loss(true_trajectories, pred_trajectories)
    print(f"MMD Loss: {loss_value}")
    
    # Create composite physics + MMD loss
    composite_loss = create_physics_loss_with_mmd(
        keras.losses.MeanSquaredError(),
        mmd_weight=0.1
    )
    
    # Test composite loss
    total_loss = composite_loss(true_trajectories, pred_trajectories)
    print(f"Composite Loss: {total_loss}")