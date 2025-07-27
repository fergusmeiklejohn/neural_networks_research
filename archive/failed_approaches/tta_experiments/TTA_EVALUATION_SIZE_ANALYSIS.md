# TTA Evaluation Size Analysis

## Dataset Sizes
- **True OOD data**: ~10MB total
  - Constant gravity: 100 trajectories × 50 timesteps × 8 features
  - Time-varying gravity: 100 trajectories × 50 timesteps × 8 features
  - Additional physics types: 5 datasets × ~500 samples each

## Model Sizes
- **Baseline models**: ~5MB total
  - GFlowNet: 1.8MB
  - MAML: 682KB
  - PINN: ~2MB (estimated)
  - Graph Extrapolation: ~2MB (if trained)

## Evaluation Components

### 1. Baseline Performance (No TTA)
- 4 models × 100 test trajectories = 400 predictions
- Time: ~1-2 minutes on CPU

### 2. TTA Methods per Model
- **TENT**: 1 adaptation step per sample
- **PhysicsTENT**: 5 adaptation steps per sample
- **TTT**: 10 adaptation steps per sample

### 3. Total Computations
- 4 models × 3 TTA methods × 100 samples = 1,200 TTA runs
- Average 5 adaptation steps = 6,000 gradient updates
- Estimated time: 10-30 minutes on CPU

### 4. Memory Requirements
- Model in memory: ~5MB
- Batch of trajectories: ~1MB
- Gradient computations: ~10MB peak
- **Total RAM needed**: < 500MB

## Recommendation: **Run Locally**

### Reasons:
1. **Small scale**: Only 100-500 test samples
2. **Quick iterations**: 10-30 minute runs
3. **JAX optimized**: Good CPU performance on Mac
4. **Debugging ease**: Better to debug TTA locally
5. **No GPU needed**: TTA is memory-bound, not compute-bound

### When to use Paperspace:
- Training new models from scratch (50+ epochs)
- Scaling to 10,000+ test samples
- Running comprehensive hyperparameter search
- Training larger models (>10M parameters)

## Suggested Local Evaluation Script

```python
# Quick local evaluation
python evaluate_tta_on_true_ood.py \
    --models gflownet,maml \
    --tta_methods tent,physics_tent \
    --test_samples 100 \
    --adaptation_steps 5
```

## Estimated Runtime
- **Local (Mac M1/M2)**: 15-30 minutes
- **Paperspace P4000**: 5-10 minutes
- **Overhead**: Paperspace setup ~10 minutes

**Verdict**: The evaluation is small enough to run locally for rapid iteration.