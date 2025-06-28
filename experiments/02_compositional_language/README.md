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
# On Paperspace A4000 GPU
python train_paperspace.py
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
- Full training: 6-8 hours on A4000 GPU