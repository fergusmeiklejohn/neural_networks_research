# Training Guide for Variable Binding Model

## Summary of Issues

The main issue is that `GradientTape` is not available in `keras.ops` - it's a TensorFlow-specific API. In Keras 3, gradient computation varies by backend:

- **TensorFlow**: Uses `tf.GradientTape`
- **JAX**: Uses `jax.value_and_grad` or `jax.grad`
- **PyTorch**: Uses `torch.autograd`

## Available Training Scripts

### 1. train_binding_final.py (SIMPLEST)
```bash
python train_binding_final.py
```
- Avoids Keras compilation complexities
- Uses simple numpy-based training loop
- Falls back to optimizer momentum for gradient updates
- Works but may train slowly

### 2. train_binding_jax.py (RECOMMENDED FOR JAX)
```bash
python train_binding_jax.py
```
- Handles multi-output model correctly
- Uses `model.train_on_batch()` 
- Requires fixing the metrics specification for multiple outputs
- Most "proper" Keras approach

### 3. Manual Gradient Approach
If you need full control, implement backend-specific gradients:

```python
# For JAX
import jax
loss_value, grads = jax.value_and_grad(loss_fn)(params)

# For TensorFlow  
import tensorflow as tf
with tf.GradientTape() as tape:
    loss = loss_fn()
grads = tape.gradient(loss, vars)

# For PyTorch
loss.backward()
grads = [p.grad for p in model.parameters()]
```

## Quick Start

1. **Easiest option** (may be slow):
   ```bash
   python train_binding_final.py
   ```

2. **For proper training** (needs multi-output fix):
   ```bash
   python train_binding_jax.py
   ```

## Model Architecture Note

The model returns a dictionary with three outputs:
- `action_logits`: The main output we train on
- `bindings`: Variable binding assignments (for analysis)
- `binding_scores`: Attention scores (for visualization)

When using Keras compile/fit, you must specify losses and metrics for each output or use dict-based specification.

## Success Criteria

The model succeeds if it achieves >50% accuracy on modification tests, demonstrating true variable binding capability rather than memorization.