#!/usr/bin/env python3
"""
Quick memory test to find optimal batch size for A4000 (16GB)
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gc

import tensorflow as tf

print("Testing GPU memory limits...")
print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")

# Test different configurations
configs_to_test = [
    {"d_model": 256, "batch_size": 16},
    {"d_model": 256, "batch_size": 8},
    {"d_model": 128, "batch_size": 32},
    {"d_model": 128, "batch_size": 16},
    {"d_model": 128, "batch_size": 8},
]

for config in configs_to_test:
    print(
        f"\nTesting d_model={config['d_model']}, batch_size={config['batch_size']}..."
    )

    try:
        # Clear memory
        tf.keras.backend.clear_session()
        gc.collect()

        # Create dummy tensors similar to our model
        batch_size = config["batch_size"]
        d_model = config["d_model"]
        seq_len = 99

        # Simulate transformer computations
        x = tf.random.normal((batch_size, seq_len, d_model))

        # Multi-head attention (8 heads)
        attn = tf.random.normal((batch_size, 8, seq_len, seq_len))
        attn_reduced = tf.reduce_mean(attn, axis=1)  # (batch_size, seq_len, seq_len)
        attn_reduced = tf.reduce_mean(attn_reduced, axis=-1)  # (batch_size, seq_len)
        attn_reduced = tf.expand_dims(attn_reduced, -1)  # (batch_size, seq_len, 1)
        attn_reduced = tf.tile(
            attn_reduced, [1, 1, d_model]
        )  # (batch_size, seq_len, d_model)

        # FFN with 4x expansion
        ffn = tf.random.normal((batch_size, seq_len, d_model * 4))
        ffn_reduced = tf.reduce_mean(
            ffn, axis=-1, keepdims=True
        )  # (batch_size, seq_len, 1)
        ffn_reduced = tf.tile(
            ffn_reduced, [1, 1, d_model]
        )  # (batch_size, seq_len, d_model)

        # Force computation - all shapes are now (batch_size, seq_len, d_model)
        result = x + attn_reduced + ffn_reduced
        _ = tf.reduce_sum(result)

        print(f"  ✓ Configuration works!")

    except tf.errors.ResourceExhaustedError:
        print(f"  ✗ OOM - Configuration too large")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    finally:
        # Clean up
        del x, attn, ffn
        gc.collect()

print("\nRecommendation: Use the largest working configuration")
