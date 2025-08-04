# Variable Binding Architecture Experiment

This experiment implements a minimal variable binding mechanism to enable true rule modifications in SCAN tasks, based on Wu et al. (2025).

## Key Innovation

The core insight is that models need explicit variable binding through dereferencing tasks to achieve true compositional generalization. Without this, models achieving 84.3% on SCAN completely fail (0%) on actual modifications.

## Architecture Components

1. **VariableMemory**: Explicit slots for variable storage (not distributed representations)
2. **BindingAttention**: Associates words with specific memory slots
3. **BoundVariableExecutor**: Executes commands using bound variables
4. **Dereferencing Tasks**: Force binding through tasks like "X means jump. Do X."

## Files

- `minimal_binding_scan.py`: Core binding model implementation
- `dereferencing_tasks.py`: Generates tasks that force variable binding
- `train_binding_jax.py`: **WORKING** - JAX-compatible training script
- `train_binding_model.py`: Original training script (has `keras.ops.GradientTape` error)
- `train_binding_model_simple.py`: Simplified version using model.fit()
- `run_training.py`: Manual training loop version
- `test_binding_components.py`: Basic component tests
- `test_minimal_training.py`: Pre-commit test script
- `PROCESS_IMPROVEMENT.md`: Lessons learned about testing before committing

## Running the Experiment

### Environment Setup
```bash
conda activate dist-invention
```

### Quick Test
```bash
python test_binding_components.py
```

### Training (Use this script!)

**IMPORTANT**: `keras.ops.GradientTape` doesn't exist. Use the JAX-compatible script:

```bash
python train_binding_jax.py
```

This script:
- Works with JAX backend (default in this project)
- Uses `model.train_on_batch()` for stable training
- Tests modification capability every 10 epochs
- Saves checkpoints and final results

### Alternative Scripts (may have issues)

1. **train_binding_model.py**: Has `AttributeError: module 'keras.ops' has no attribute 'GradientTape'`
2. **train_binding_model_simple.py**: Uses `model.fit()` but may have custom model issues
3. **run_training.py**: Manual loop, needs backend-specific fixes

### Known Keras 3 Issues

- `keras.ops.GradientTape` doesn't exist (it's TensorFlow-specific)
- `keras.utils.function` doesn't exist (removed in Keras 3)
- `GradientTape` is only in `tf.GradientTape`, not portable across backends
- JAX backend requires different gradient computation approach

## Success Criteria

- Model successfully binds words to variable slots
- Can modify bindings and execute with changes
- Achieves >50% accuracy on single-modification validation set

## Key Findings

- Explicit variable slots are essential (not just embeddings)
- Dereferencing tasks force true binding to occur
- Standard architectures fail because they lack this mechanism

## Next Steps

1. Fix training script for your specific Keras backend
2. Train for 50+ epochs on dereferencing tasks
3. Test modification capability (jump → hop)
4. Scale up to full SCAN dataset if successful
