# Next Steps for Variable Binding Architecture

## Current Status
✅ Proper architecture implemented in MLX
✅ Training runs without errors
✅ Good inference performance (97k samples/sec)
❌ Model not learning binding (0% modification success)
❌ Predictions are constant regardless of input

## Immediate Fixes Needed

### 1. Fix Gradient Flow for Binding
The hard argmax in BindingAttention blocks gradients:
```python
bindings = mx.argmax(scores, axis=-1)  # No gradients!
```

**Solution**: Use Gumbel-Softmax or straight-through estimator:
```python
# Forward: hard selection, Backward: soft gradients
bindings_hard = mx.argmax(scores, axis=-1)
bindings_soft = mx.softmax(scores / temperature, axis=-1)
# Use soft for gradient, hard for indexing
```

### 2. Debug Slot Retrieval
Current retrieval may not be differentiable:
```python
retrieved = slot_values[slot_indices]  # Indexing blocks gradients
```

**Solution**: Use soft attention for retrieval:
```python
# Soft retrieval using attention weights
retrieved = mx.sum(bindings_soft[:, :, :, None] * slot_values[None, None, :, :], axis=2)
```

### 3. Add Positional Encoding
Model needs to understand word positions for proper binding:
```python
# Add positional embeddings
pos_encoding = self.positional_encoding(positions)
word_embeds = word_embeds + pos_encoding
```

### 4. Simplify Initial Task
Start with single-variable binding only:
- Train on: "X means jump. Do X." → "JUMP"
- Once working, add multiple variables
- Then add rebinding

## Architecture Improvements

### 1. Implement Gumbel-Softmax Binding
```python
def gumbel_softmax(logits, temperature=1.0, hard=False):
    # Add Gumbel noise
    gumbel_noise = -mx.log(-mx.log(mx.random.uniform(logits.shape)))
    y_soft = mx.softmax((logits + gumbel_noise) / temperature, axis=-1)

    if hard:
        # Straight-through estimator
        y_hard = mx.one_hot(mx.argmax(y_soft, axis=-1), logits.shape[-1])
        y = y_hard - mx.stop_gradient(y_soft) + y_soft
    else:
        y = y_soft

    return y
```

### 2. Add Diagnostic Logging
```python
# Log binding patterns
print(f"Binding distribution: {mx.mean(bindings_soft, axis=(0,1))}")
print(f"Slot usage: {mx.sum(mx.argmax(bindings_soft, axis=-1) == slot_id) for slot_id in range(num_slots)}")
```

### 3. Implement Copying Baseline
First verify the model can learn simple copying:
```python
# Task: "jump walk turn" → "JUMP WALK TURN"
# No variables, just word-to-action mapping
```

## Training Improvements

### 1. Curriculum Learning
- Stage 1: Direct word-to-action mapping (no variables)
- Stage 2: Single variable binding
- Stage 3: Multiple variables
- Stage 4: Rebinding

### 2. Better Loss Function
```python
# Add auxiliary losses
binding_loss = mx.mean(mx.sum(bindings_soft * mx.log(bindings_soft + 1e-8), axis=-1))
diversity_loss = -mx.mean(mx.log(mx.mean(bindings_soft, axis=(0,1)) + 1e-8))
total_loss = action_loss + 0.1 * binding_loss + 0.1 * diversity_loss
```

### 3. Learning Rate Schedule
```python
# Warmup then decay
lr = min(1.0, step / warmup_steps) * base_lr
lr = lr * (0.95 ** (step // decay_steps))
```

## Testing Protocol

### 1. Unit Tests for Components
```bash
# Test each component separately
python test_variable_memory.py  # Verify slot storage/retrieval
python test_binding_attention.py  # Check attention patterns
python test_executor.py  # Ensure action generation works
```

### 2. Gradient Flow Analysis
```python
# Check if gradients reach all parameters
for name, param in model.parameters().items():
    grad_norm = mx.norm(grads[name])
    print(f"{name}: grad_norm = {grad_norm}")
```

### 3. Visualization
- Plot attention maps
- Show binding assignments over time
- Track slot usage statistics

## Commands to Run

```bash
# Quick debugging run
python train_binding_mlx_proper.py --quick --lr 0.01

# Test simplified tasks
python train_binding_mlx_proper.py --task simple --epochs 10

# Full training with logging
python train_binding_mlx_proper.py --epochs 50 --log_interval 10

# Run test suite
python test_binding_capabilities.py --model proper_binding_model.npz
```

## Expected Timeline

- **Today**: Fix gradient flow, implement Gumbel-Softmax
- **Tomorrow**: Add curriculum learning, test on simple tasks
- **Day 3**: Full binding working, start rebinding tests
- **Day 4**: Integrate with SCAN, compare with baselines
- **Day 5**: Write up results, prepare figures

## Success Criteria

1. **Immediate** (Today):
   - Model learns simple copying task (>90% accuracy)
   - Gradients flow to all parameters

2. **Short-term** (This week):
   - Basic binding works (>80% on "X means jump. Do X.")
   - Some rebinding success (>50% on modifications)

3. **Medium-term** (Next week):
   - Full test suite passes (>70% overall)
   - Outperforms baselines on binding tasks
   - Ready for paper submission

## Key Insight

The current model treats binding as a classification problem, but it's really about creating persistent associations. We need to ensure:
1. Bindings are learnable (gradient flow)
2. Bindings are persistent (memory)
3. Bindings are modifiable (updates)

Without these three properties, we're just doing pattern matching, not true variable binding.
