# Compositional Language Experiment

This experiment tests whether neural networks can modify linguistic compositional rules and generate consistent novel command combinations using the SCAN dataset.

## Quick Start

### Local Testing
```bash
# Test the setup
python quick_test.py

# Run minimal training test
python test_training.py
```

### Full Training (Paperspace)
```bash
# On Paperspace GPU with memory optimizations
python paperspace_generate_and_train.py

# This now uses the optimized version with:
# - GPU memory growth (prevents OOM)
# - Mixed precision training (50% less memory)
# - tf.function compilation (faster)
# - Periodic memory clearing
```

## Components

1. **scan_data_loader.py**: Downloads and processes SCAN dataset with proper train/test isolation
2. **modification_generator.py**: Creates systematic rule modifications (e.g., "jump" → "walk")
3. **models.py**: Transformer-based architecture for compositional rule learning and modification
4. **train_progressive_curriculum.py**: 4-stage progressive training pipeline
5. **train_paperspace.py**: Full training configuration for GPU

## Expected Results

Based on our physics experiment success (83.51% extrapolation):
- Stage 1: >95% accuracy on standard SCAN
- Stage 2: >80% on simple modifications
- Stage 3: >70% on complex modifications  
- Stage 4: >60% on novel combinations

## Data

The SCAN dataset tests compositional generalization:
- Commands: "jump twice and walk left"
- Actions: "I_JUMP I_JUMP I_TURN_LEFT I_WALK"

We test modifications like:
- Word swaps: "jump" means "walk"
- Direction reversal: "left" ↔ "right"
- New actions: "jump" means "turn around"

## Model Architecture

- **Compositional Rule Extractor**: Identifies primitives, modifiers, and patterns
- **Rule Modification Component**: Applies requested rule changes
- **Sequence Generator**: Produces consistent action sequences
- Total: ~50M parameters

## Training Time

- Local test: ~5 minutes
- Full training: 4-6 hours on A4000 GPU (with optimizations)

## Troubleshooting

### GPU Out of Memory (OOM)
The optimized version includes several memory-saving features:
- **Mixed precision**: Reduces memory by ~50%
- **Memory growth**: Prevents TensorFlow from allocating all GPU memory
- **Periodic clearing**: Prevents gradual memory accumulation
- **Gradient accumulation**: Allows smaller batches with same effective size

If you still get OOM errors:
1. Reduce `batch_size` to 4 in the config
2. Reduce `d_model` to 64
3. Use a larger GPU (A5000 24GB or A6000 48GB)