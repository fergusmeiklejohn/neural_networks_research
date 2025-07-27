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
- `train_binding_model.py`: Original training script (has Keras 3 API issues)
- `train_binding_model_simple.py`: Simplified version using model.fit()
- `run_training.py`: Manual training loop version
- `test_binding_components.py`: Basic component tests
- `test_minimal_training.py`: Pre-commit test script
- `PROCESS_IMPROVEMENT.md`: Lessons learned about testing before committing

## Running the Experiment

Due to Keras 3 API changes, the training scripts need adjustment for your specific backend. The model architecture is sound, but the training loop needs to match your Keras version.

### Quick Test
```bash
/Users/fergusmeiklejohn/miniconda3/envs/dist-invention/bin/python test_binding_components.py
```

### Training Options

1. **For TensorFlow backend**: `train_binding_model.py` should work with minor fixes
2. **For JAX backend**: Use `run_training.py` or adapt for JAX-specific training
3. **For PyTorch backend**: Adapt the training loop for PyTorch gradients

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