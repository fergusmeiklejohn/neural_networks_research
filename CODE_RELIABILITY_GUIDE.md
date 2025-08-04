# Code Reliability Guide

This guide captures hard-won lessons from implementing neural network experiments. Reference this BEFORE writing code to avoid common pitfalls.

## 🆕 NEW: Use Centralized Utilities (Added July 2025)

**IMPORTANT**: We now have centralized utilities that eliminate many common issues. Always use these patterns:

```python
# Start every script with:
from utils.imports import setup_project_paths
setup_project_paths()

# Use centralized configuration
from utils.config import setup_environment
config = setup_environment()

# Use centralized path resolution
from utils.paths import get_data_path, get_output_path
data_path = get_data_path("processed/physics_worlds")
output_path = get_output_path("results")
```

This eliminates:
- Import errors from `sys.path.append`
- Path issues between local/cloud
- Inconsistent Keras backend settings
- Missing directories

See `CODE_QUALITY_SETUP.md` for complete details.

## Data Loading Best Practices

### Always Verify Data Format First
```python
# ALWAYS start with data inspection
print(f"Data shape: {data.shape}")
print(f"First sample: {data[0]}")
print(f"Column meanings: {column_names}")  # Document what each column represents
print(f"Value ranges: min={data.min(axis=0)}, max={data.max(axis=0)}")
```

### Critical Data Format Issues

#### Physics Experiment Specific
- **Units**: Data uses PIXELS, not meters! (40 pixels = 1 meter)
- **Gravity values**: ~400-1200 pixels/s² in data (not -9.8 m/s²)
- **Conversion**: Use `physics_value_SI = pixel_value / 40.0`
- **Trajectory format**: `[time, x1, y1, vx1, vy1, mass1, radius1, x2, y2, vx2, vy2, mass2, radius2]`

#### Common Data Pitfalls
1. **Wrong column extraction**: Always use named indices, not magic numbers
   ```python
   # BAD
   positions = data[:, 1:3]  # What columns are these?

   # GOOD
   X1_COL, Y1_COL = 1, 2  # Document at top of file
   positions = data[:, [X1_COL, Y1_COL]]
   ```

2. **Silent unit mismatches**: Always document units in variable names
   ```python
   gravity_pixels_per_s2 = 400.0  # Clear unit indication
   gravity_m_per_s2 = gravity_pixels_per_s2 / 40.0
   ```

## JAX/TensorFlow/Keras Specific Issues

### JAX Immutability
```python
# WRONG - JAX arrays are immutable
array[index] = value  # Will fail!

# CORRECT
array = array.at[index].set(value)
```

### TensorFlow GPU Memory
```python
# At start of script
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce log spam

# For TensorFlow GPU memory growth
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

### Keras Backend Issues
```python
# Always verify backend at start
import keras
print(f"Keras backend: {keras.backend.backend()}")

# Common backend-specific fixes
if keras.backend.backend() == 'jax':
    # JAX-specific settings
    import jax
    # Disable JIT for debugging
    jax.config.update('jax_disable_jit', True)
```

## Training Reliability

### Save Early, Save Often
```python
# After EACH training stage
def save_checkpoint(model, stage, metrics, storage_path='/storage'):
    """Save model and metrics with automatic fallback"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Try persistent storage first
    if os.path.exists(storage_path):
        save_path = f"{storage_path}/stage_{stage}_{timestamp}"
    else:
        save_path = f"./outputs/stage_{stage}_{timestamp}"
        print(f"WARNING: No persistent storage, saving locally to {save_path}")

    # Save model
    model.save_weights(f"{save_path}_model.h5")

    # Save metrics
    with open(f"{save_path}_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"✓ Checkpoint saved: {save_path}")
    return save_path
```

### Numerical Stability Checklist
1. **Start conservative**: Learning rate 1e-4 or smaller
2. **Add epsilon**: To all denominators to prevent division by zero
3. **Clip gradients**: If seeing NaN losses
4. **Reduce batch size**: If seeing GPU OOM errors
5. **Monitor losses**: Print every 10 batches to catch instability early

```python
# Example stable training loop
for epoch in range(epochs):
    for batch_idx, (x, y) in enumerate(train_loader):
        loss = train_step(x, y)

        # Catch numerical issues early
        if tf.math.is_nan(loss):
            print(f"ERROR: NaN loss at epoch {epoch}, batch {batch_idx}")
            print(f"Learning rate: {optimizer.learning_rate}")
            print(f"Batch stats: x mean={x.mean()}, std={x.std()}")
            break

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}: loss={loss:.6f}")
```

## Testing Before Full Runs

### Minimal Test Protocol
```python
# ALWAYS test with tiny data first
TEST_SAMPLES = 100
TEST_EPOCHS = 2

print("Running minimal test...")
test_data = full_data[:TEST_SAMPLES]
model = create_model()
history = model.fit(test_data, epochs=TEST_EPOCHS)
print("✓ Minimal test passed")

# Only then run full training
```

### GPU Utilization Check
```bash
# In separate terminal during training
watch -n 1 nvidia-smi

# Good: GPU-Util > 80%
# Bad: GPU-Util < 20% (increase batch size)
```

## Common Error Patterns and Solutions

### 1. "AttributeError: 'Sequential' object has no attribute 'layers'"
**Cause**: Model not built yet
**Solution**: Call model.build() or pass input_shape to first layer

### 2. "No gradients provided for any variable"
**Cause**: tf.function compilation issues or disconnected graph
**Solution**:
- Remove @tf.function decorator for debugging
- Ensure loss is connected to trainable variables
- Check for any operations that break gradient flow

### 3. "CUDA_ERROR_OUT_OF_MEMORY"
**Progressive solutions**:
1. Reduce batch size by half
2. Use gradient accumulation
3. Reduce model size
4. Enable mixed precision training

### 4. "ValueError: Dimensions must be equal"
**Common causes**:
- Mismatched input/output shapes
- Wrong axis for operations
- Batch dimension handling

**Debug approach**:
```python
print(f"Input shape: {input_tensor.shape}")
print(f"Weight shape: {layer.weights[0].shape}")
print(f"Expected output shape: {expected_shape}")
```

### 5. Physics-Specific: "MSE is 1000x too high"
**Always check**:
- Unit conversion (pixels vs meters)
- Data normalization
- Are you comparing positions in same coordinate system?

## Performance Optimization

### Profiling First
```python
# Simple profiling
import time
start = time.time()
# ... operation ...
print(f"Operation took: {time.time() - start:.2f}s")

# For detailed profiling
import cProfile
cProfile.run('train_model()', sort='cumulative')
```

### Common Bottlenecks
1. **Data loading**: Use tf.data.Dataset with prefetch
2. **Small batch size**: Increase until GPU memory is ~90% used
3. **CPU preprocessing**: Move to GPU with tf.function
4. **Excessive logging**: Log every N batches, not every batch

## Debugging Workflow

### 1. Start Simple
- Tiny model (2 layers, 10 units)
- Tiny data (100 samples)
- No fancy features (no callbacks, mixed precision, etc.)
- Verify this works first

### 2. Add Complexity Gradually
- Increase model size
- Add more data
- Enable optimizations
- Add callbacks/logging

### 3. When Things Break
1. Git commit current state (even if broken)
2. Revert to last working version
3. Binary search for the breaking change
4. Document the issue and solution

## Environment-Specific Notes

### Paperspace
- Always check `/storage` exists before training
- Use `/notebooks` path prefix (not `/workspace`)
- Save results before 6-hour auto-shutdown
- GPU instances may have different CUDA versions

### Local Mac Development
- JAX works well with Metal acceleration
- TensorFlow may have GPU issues - use JAX backend
- Smaller batch sizes due to memory constraints

### Colab
- Sessions timeout after ~90 min idle
- Use Google Drive for persistence
- Free tier has usage limits - plan accordingly

## Quick Reference Commands

```bash
# Check GPU
nvidia-smi

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check disk space
df -h

# Find large files
du -h . | sort -rh | head -20

# Quick performance test
python -m timeit -n 100 "import numpy as np; np.random.randn(1000, 1000) @ np.random.randn(1000, 1000)"
```

## Final Checklist Before Training

- [ ] Data format verified and documented
- [ ] Units clearly specified in variable names
- [ ] Test run with 100 samples completed successfully
- [ ] GPU utilization checked (>80%)
- [ ] Checkpoint saving tested
- [ ] Numerical stability verified (no NaN/Inf)
- [ ] Results path exists and is writable
- [ ] Git commit created before long training run

---

Remember: Every debugging session is an opportunity to update this guide. The 2 hours you spend debugging today can save 2 days next month.
