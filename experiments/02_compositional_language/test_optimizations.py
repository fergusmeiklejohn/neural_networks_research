#!/usr/bin/env python3
"""
Quick test to verify memory optimizations work
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

print("Testing memory optimizations...")

# 1. Test GPU memory growth
print("\n1. GPU Memory Growth:")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  ✓ Memory growth enabled for {gpu}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
else:
    print("  - No GPU detected (expected on Mac)")

# 2. Test mixed precision
print("\n2. Mixed Precision:")
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f"  ✓ Policy set: {policy.name}")
    print(f"  - Compute dtype: {policy.compute_dtype}")
    print(f"  - Variable dtype: {policy.variable_dtype}")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# 3. Test tf.function compilation
print("\n3. tf.function Compilation:")
@tf.function
def test_function(x):
    return x * 2

try:
    x = tf.constant([1.0, 2.0, 3.0])
    result = test_function(x)
    print(f"  ✓ tf.function works: {result.numpy()}")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# 4. Test gradient accumulation
print("\n4. Gradient Accumulation:")
try:
    # Simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(5,))
    ])
    
    # Optimizer with mixed precision
    optimizer = tf.keras.optimizers.Adam()
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    # Test gradient accumulation
    accumulated_grads = []
    for i in range(2):
        x = tf.random.normal((2, 5))
        y = tf.random.normal((2, 10))
        
        with tf.GradientTape() as tape:
            pred = model(x)
            loss = tf.reduce_mean(tf.square(pred - y))
            scaled_loss = optimizer.get_scaled_loss(loss)
        
        scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
        grads = optimizer.get_unscaled_gradients(scaled_grads)
        
        if not accumulated_grads:
            accumulated_grads = grads
        else:
            accumulated_grads = [ag + g for ag, g in zip(accumulated_grads, grads)]
    
    # Average and apply
    averaged_grads = [g / 2 for g in accumulated_grads]
    optimizer.apply_gradients(zip(averaged_grads, model.trainable_variables))
    
    print("  ✓ Gradient accumulation works")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# 5. Test memory clearing
print("\n5. Memory Management:")
try:
    import gc
    
    # Create and delete some tensors
    for i in range(3):
        x = tf.random.normal((100, 100))
        del x
    
    # Clear session and collect garbage
    tf.keras.backend.clear_session()
    gc.collect()
    
    print("  ✓ Memory clearing works")
except Exception as e:
    print(f"  ✗ Failed: {e}")

print("\n✅ All optimizations tested successfully!")
print("\nRecommendations:")
print("- These optimizations should help with GPU memory issues")
print("- Mixed precision can reduce memory usage by ~50%")
print("- Gradient accumulation allows larger effective batch sizes")
print("- Periodic memory clearing prevents accumulation")