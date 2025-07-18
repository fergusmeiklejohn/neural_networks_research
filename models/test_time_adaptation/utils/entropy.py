"""Entropy-based utilities for test-time adaptation."""

import keras
from keras import ops


def entropy_loss(probs: keras.ops.Tensor, eps: float = 1e-7) -> keras.ops.Tensor:
    """Compute entropy of probability distributions.
    
    H(p) = -sum(p * log(p))
    
    Args:
        probs: Probability tensor (batch_size, num_classes)
        eps: Small constant for numerical stability
        
    Returns:
        Entropy tensor (batch_size,)
    """
    # Clip probabilities for numerical stability
    probs = ops.clip(probs, eps, 1.0 - eps)
    
    # Compute entropy
    entropy = -ops.sum(probs * ops.log(probs), axis=-1)
    
    return entropy


def confidence_selection(
    probs: keras.ops.Tensor, 
    threshold: float = 0.9
) -> keras.ops.Tensor:
    """Select samples based on prediction confidence.
    
    Args:
        probs: Probability tensor (batch_size, num_classes)
        threshold: Confidence threshold
        
    Returns:
        Binary mask tensor (batch_size,) where 1 indicates confident samples
    """
    # Get maximum probability (confidence)
    max_probs = ops.max(probs, axis=-1)
    
    # Create mask for confident predictions
    mask = ops.cast(max_probs >= threshold, dtype=probs.dtype)
    
    return mask


def symmetric_entropy_loss(
    probs: keras.ops.Tensor,
    eps: float = 1e-7
) -> keras.ops.Tensor:
    """Compute symmetric entropy loss.
    
    This variant is more robust to label noise.
    L = H(p) + H(1-p)
    
    Args:
        probs: Probability tensor
        eps: Small constant for stability
        
    Returns:
        Symmetric entropy loss
    """
    ent_pos = entropy_loss(probs, eps)
    ent_neg = entropy_loss(1.0 - probs, eps)
    
    return ent_pos + ent_neg


def diversity_loss(
    probs: keras.ops.Tensor,
    target_distribution: keras.ops.Tensor = None
) -> keras.ops.Tensor:
    """Encourage diversity in predictions.
    
    Prevents collapse to uniform predictions.
    
    Args:
        probs: Probability tensor (batch_size, num_classes)
        target_distribution: Target class distribution (num_classes,)
        
    Returns:
        Diversity loss
    """
    # Average predictions across batch
    avg_probs = ops.mean(probs, axis=0)
    
    if target_distribution is None:
        # Uniform distribution
        num_classes = ops.shape(probs)[-1]
        target_distribution = ops.ones(num_classes) / num_classes
    
    # KL divergence from target distribution
    kl_div = ops.sum(
        avg_probs * ops.log(avg_probs / (target_distribution + 1e-7))
    )
    
    return kl_div