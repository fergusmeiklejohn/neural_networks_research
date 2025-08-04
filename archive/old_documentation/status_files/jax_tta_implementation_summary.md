# JAX TTA Implementation Summary

## What We've Implemented

### 1. **Full JAX Gradient Support** (`base_tta_jax_v2.py`)

- **Stateless Operations**: Uses `model.stateless_call()` for JAX compatibility
- **Complete State Management**: Tracks model trainable/non-trainable vars + optimizer state
- **Proper Gradient Computation**: Uses `jax.value_and_grad()` with auxiliary outputs
- **Flexible Parameter Selection**: Can adapt all params or just specific ones (e.g., BatchNorm)

Key improvements:
```python
# Old: Only BatchNorm stats updated
y_pred = model(x, training=True)

# New: Full gradient-based updates
grad_fn = jax.value_and_grad(compute_loss_and_updates, has_aux=True)
(loss, (y_pred, updated_non_trainable)), grads = grad_fn(...)
trainable_vars, optimizer_vars = optimizer.stateless_apply(...)
```

### 2. **Regression-Specific TTA** (`regression_tta_v2.py`)

- **RegressionTTAV2**: Base regression TTA with consistency and smoothness losses
- **PhysicsRegressionTTAV2**: Adds physics constraints (momentum, energy, collisions)
- **Configurable Updates**: Can update all parameters or just BatchNorm

Physics constraints include:
- Momentum conservation
- Energy conservation/dissipation
- Collision avoidance
- Position bounds

### 3. **Testing Infrastructure** (`test_jax_tta_v2.py`)

Comprehensive tests for:
- Gradient computation verification
- BatchNorm-only vs full parameter updates
- Complete state restoration
- Performance on physics OOD data

## How to Use the New Implementation

### 1. **Update Existing Code**

Replace imports in your scripts:
```python
# Old
from models.test_time_adaptation.regression_tta import RegressionTTA

# New
from models.test_time_adaptation.regression_tta_v2 import RegressionTTAV2
```

### 2. **Create TTA Wrapper**

```python
# For general regression
tta = RegressionTTAV2(
    model,
    adaptation_steps=5,
    learning_rate=5e-4,
    update_bn_only=False,  # Full parameter updates
    consistency_weight=1.0,
    smoothness_weight=0.1
)

# For physics-specific tasks
tta = PhysicsRegressionTTAV2(
    model,
    adaptation_steps=5,
    learning_rate=5e-4,
    physics_weight=0.5
)
```

### 3. **Adapt and Predict**

```python
# Single batch adaptation
y_pred = tta.adapt(X_test)
tta.reset()  # Reset to original state

# Or use predict_and_adapt for automatic reset
y_pred = tta.predict_and_adapt(X_test, batch_size=32)
```

## Expected Benefits

1. **Full Gradient Updates**: Not limited to BatchNorm statistics
2. **Better Adaptation**: Can learn from test data more effectively
3. **Physics Awareness**: Respects physical constraints during adaptation
4. **JAX Performance**: Can leverage JAX's JIT compilation (future work)

## Next Steps

### 1. **Integration** (HIGH PRIORITY)
- Update `tta_wrappers.py` to include V2 implementations
- Modify existing experiments to use new implementation
- Run full evaluation suite

### 2. **Performance Optimization**
- Add `@jax.jit` decorators for speed
- Implement batched adaptation for efficiency
- Profile and optimize bottlenecks

### 3. **Extended Testing**
- Test on rotating frames (Coriolis forces)
- Test on spring-coupled systems
- Compare with TensorFlow backend performance

### 4. **Hyperparameter Tuning**
With full gradient support, we need to re-tune:
- Learning rates (likely need lower values)
- Adaptation steps (fewer might suffice)
- Loss weights (physics vs consistency)

## Running the Test

To verify everything works:

```bash
# Test the new implementation
/Users/fergusmeiklejohn/miniconda3/envs/dist-invention/bin/python experiments/01_physics_worlds/test_jax_tta_v2.py
```

## Important Notes

1. **JAX Backend Required**: Ensure `KERAS_BACKEND=jax` is set
2. **Memory Usage**: JAX may use more memory due to gradient tracking
3. **First Run Slower**: JAX compiles functions on first use
4. **Debugging**: JAX errors can be cryptic - test incrementally

## Comparison with Original

| Feature | Original TTA | New JAX TTA V2 |
|---------|--------------|----------------|
| Gradient computation | Limited/None | Full support |
| Parameter updates | BN stats only | All parameters |
| State restoration | Partial | Complete |
| Physics constraints | Basic | Comprehensive |
| JAX compatibility | Minimal | Native |

## Conclusion

The new implementation provides full gradient-based Test-Time Adaptation for JAX backend, enabling proper parameter updates beyond just BatchNorm statistics. This should significantly improve adaptation performance on true OOD scenarios like time-varying gravity.
