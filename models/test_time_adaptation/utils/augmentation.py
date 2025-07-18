"""Data augmentation utilities for physics test-time adaptation."""

from typing import List, Tuple, Callable, Any
import numpy as np
import keras
from keras import ops


def create_physics_augmentations() -> List[Callable]:
    """Create a list of physics-preserving augmentations.
    
    Returns:
        List of augmentation functions
    """
    return [
        flip_trajectory_horizontal,
        rotate_trajectory,
        add_gaussian_noise,
        time_reversal,
        interpolate_trajectory
    ]


def flip_trajectory_horizontal(
    trajectory: Any
) -> Any:
    """Flip trajectory horizontally (mirror in y-axis).
    
    Args:
        trajectory: Shape (batch, time, features) where features = [x1,y1,vx1,vy1,x2,y2,vx2,vy2]
        
    Returns:
        Flipped trajectory
    """
    flipped = ops.copy(trajectory)
    
    # Flip x positions
    flipped[..., 0] = -flipped[..., 0]  # x1
    flipped[..., 4] = -flipped[..., 4]  # x2
    
    # Flip x velocities
    flipped[..., 2] = -flipped[..., 2]  # vx1
    flipped[..., 6] = -flipped[..., 6]  # vx2
    
    return flipped


def rotate_trajectory(
    trajectory: Any,
    angle: float = None
) -> Any:
    """Rotate trajectory by given angle.
    
    Args:
        trajectory: Input trajectory
        angle: Rotation angle in radians (random if None)
        
    Returns:
        Rotated trajectory
    """
    if angle is None:
        # Random angle between -pi/4 and pi/4
        angle = np.random.uniform(-np.pi/4, np.pi/4)
    
    cos_a = ops.cos(angle)
    sin_a = ops.sin(angle)
    
    rotated = ops.copy(trajectory)
    
    # Rotate positions and velocities for ball 1
    x1, y1 = trajectory[..., 0], trajectory[..., 1]
    vx1, vy1 = trajectory[..., 2], trajectory[..., 3]
    
    rotated[..., 0] = cos_a * x1 - sin_a * y1
    rotated[..., 1] = sin_a * x1 + cos_a * y1
    rotated[..., 2] = cos_a * vx1 - sin_a * vy1
    rotated[..., 3] = sin_a * vx1 + cos_a * vy1
    
    # Rotate positions and velocities for ball 2
    x2, y2 = trajectory[..., 4], trajectory[..., 5]
    vx2, vy2 = trajectory[..., 6], trajectory[..., 7]
    
    rotated[..., 4] = cos_a * x2 - sin_a * y2
    rotated[..., 5] = sin_a * x2 + cos_a * y2
    rotated[..., 6] = cos_a * vx2 - sin_a * vy2
    rotated[..., 7] = sin_a * vx2 + cos_a * vy2
    
    return rotated


def add_gaussian_noise(
    trajectory: Any,
    noise_scale: float = 0.01
) -> Any:
    """Add small Gaussian noise to trajectory.
    
    Args:
        trajectory: Input trajectory
        noise_scale: Standard deviation of noise
        
    Returns:
        Noisy trajectory
    """
    noise = ops.random.normal(
        shape=ops.shape(trajectory),
        mean=0.0,
        stddev=noise_scale
    )
    
    return trajectory + noise


def time_reversal(
    trajectory: Any
) -> Any:
    """Reverse trajectory in time.
    
    Args:
        trajectory: Input trajectory
        
    Returns:
        Time-reversed trajectory
    """
    # Reverse time axis
    reversed_traj = trajectory[:, ::-1]
    
    # Negate velocities
    reversed_traj[..., 2] = -reversed_traj[..., 2]  # vx1
    reversed_traj[..., 3] = -reversed_traj[..., 3]  # vy1
    reversed_traj[..., 6] = -reversed_traj[..., 6]  # vx2
    reversed_traj[..., 7] = -reversed_traj[..., 7]  # vy2
    
    return reversed_traj


def interpolate_trajectory(
    trajectory: Any,
    factor: int = 2
) -> Any:
    """Interpolate trajectory to higher time resolution.
    
    Args:
        trajectory: Input trajectory (batch, time, features)
        factor: Interpolation factor
        
    Returns:
        Interpolated trajectory
    """
    batch_size, time_steps, features = ops.shape(trajectory)
    
    # Create new time points
    new_time_steps = (time_steps - 1) * factor + 1
    
    # Linear interpolation
    # Note: This is a simplified version. In practice, you might want
    # to use more sophisticated interpolation
    interpolated = ops.zeros((batch_size, new_time_steps, features))
    
    # Fill in original points
    interpolated[:, ::factor] = trajectory
    
    # Interpolate between points
    for i in range(1, factor):
        alpha = i / factor
        interpolated[:, i::factor] = (
            (1 - alpha) * trajectory[:, :-1] + 
            alpha * trajectory[:, 1:]
        )
    
    return interpolated


def create_augmented_batch(
    trajectory: Any,
    augmentations: List[Callable],
    num_augmentations: int = 2
) -> Any:
    """Create batch of augmented trajectories.
    
    Args:
        trajectory: Original trajectory (batch, time, features)
        augmentations: List of augmentation functions
        num_augmentations: Number of augmentations to apply
        
    Returns:
        Augmented batch (batch * (1 + num_augmentations), time, features)
    """
    batch_list = [trajectory]
    
    for _ in range(num_augmentations):
        # Randomly select an augmentation
        aug_fn = np.random.choice(augmentations)
        augmented = aug_fn(trajectory)
        batch_list.append(augmented)
    
    return ops.concatenate(batch_list, axis=0)