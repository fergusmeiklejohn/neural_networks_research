# Research Diary - June 27, 2025

## Physics Worlds Experiment: PINN Implementation

### Morning Session: Assessment and Planning

**Context**: Started with 0% extrapolation accuracy after fixing data leakage. Need architectural innovations.

**Key Decision**: Implement Physics-Informed Neural Networks (PINNs) based on research showing 70-85% extrapolation possible with physics constraints.

**Planning**: Created timestamped plan in `claude-plans/` with 5 implementation steps.

### Afternoon Session: PINN Component Implementation

**What Worked**:
1. **Soft Collision Models**: Using `softplus(penetration/ε)` solved discontinuity issues perfectly
2. **ReLoBRaLo Loss Balancing**: Automatic weight adjustment prevents any loss dominating
3. **Hybrid Architecture**: Transformer features + HNN physics constraints complement each other

**Challenges & Solutions**:
1. **JAX/Keras 3 Compatibility**:
   - Issue: `keras.utils.autograph_mode` doesn't exist
   - Solution: Removed decorators, used simplified training

2. **FourierFeatures Tracer Leak**:
   - Issue: JAX traced intermediate values escaping
   - Solution: Used `add_weight()` for proper variable initialization

3. **Batching Errors**:
   - Issue: Shape mismatches in physics losses (masses broadcasting)
   - Solution: Proper `expand_dims()` for time dimension

4. **Collision Model Interface**:
   - Issue: Keras expects dictionary inputs, not positional args
   - Solution: Refactored all collision models to use dict inputs

**Key Implementation Details**:
- HamiltonianNN: Energy conservation by construction (H = KE + PE + V)
- 4-stage progressive curriculum: in-dist → physics → randomization → extrapolation
- Smooth wall boundaries using sigmoid activation
- Physics-guided attention mechanism for feature fusion

### Results

✅ All components tested successfully:
- HamiltonianNN computes energy and (placeholder) dynamics
- Soft collisions produce smooth potentials
- ReLoBRaLo balances losses dynamically
- Full transformer runs with physics features

### Decisions Made

1. **Training Strategy**: User will run full training separately (hours/days), I'll test with small subsets
2. **Research Documentation**: Daily diary entries with frequent notes throughout the day
3. **Soft vs Hard Constraints**: Chose soft models for differentiability despite less physical accuracy

### Next Steps

1. Create minimal test training script (100 samples, 2 epochs)
2. User runs full progressive training
3. Evaluate energy conservation and extrapolation metrics
4. Compare with baseline transformer

### Insights

The key insight today was that **physics knowledge must be differentiable** to be useful in neural networks. Hard constraints (exact bounces) break gradients, but soft approximations maintain physical intuition while enabling learning. This is why PINNs succeed where pure data-driven approaches fail at extrapolation.

### Evening Session: Bug Fix

**Issue**: Training script looking for data in wrong location
- Expected: `data/improved_train_data.pkl`
- Actual: `data/processed/physics_worlds_v2/train_data.pkl`

**Fix**: Updated `load_physics_world_data()` to use correct paths with mapping for all 6 data splits.

**Issue 2**: GradientTape not available in Keras 3 with JAX backend
- `keras.ops.GradientTape` doesn't exist
- Need to use JAX-specific gradient computation or Keras model.fit()

**Fix**: Created `train_pinn_keras3.py` with simplified training that computes losses but doesn't update weights. Full implementation would need:
- `jax.grad()` for gradient computation
- `optax` for optimizer state
- Custom training loop with JAX primitives

**Note**: The architecture is sound, but Keras 3 + JAX requires different training approach than TensorFlow-style GradientTape.

### Decision: TensorFlow Backend for Training

After analyzing options, decided to use TensorFlow backend for training because:
1. Existing code works immediately (GradientTape, optimizers, callbacks)
2. Better cloud GPU support and tooling
3. Can focus on science vs implementation details
4. Mature Keras 3 integration

**Plan for Tomorrow**:
- Set up cloud GPU (Paperspace/Colab)
- Switch to TensorFlow backend
- Run full 4-stage progressive training
- Monitor energy conservation and extrapolation metrics
- Success = >70% extrapolation accuracy (from current 0%)

This tests our core hypothesis: encoding physics enables true extrapolation beyond training data.
