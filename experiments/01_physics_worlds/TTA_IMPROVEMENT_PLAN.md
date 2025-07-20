# TTA Improvement Plan

## Current Status

Our initial TTA evaluation showed disappointing results:
- All TTA methods (TENT, PhysicsTENT, TTT) converged to the same MSE (6884.27)
- Performance was actually 1.1% worse than no adaptation
- This suggests a fundamental issue with our approach

## Root Cause Analysis

### 1. Why All Methods Converged to Same Value

**Hypothesis**: All methods are stuck in the same local minimum because:
- Single timestep input provides insufficient signal for meaningful adaptation
- Learning rate (1e-4) may be too small to escape local minima
- BatchNorm-only updates (TENT) might be too restrictive

**Evidence**:
- The exact same MSE value (6884.27) is statistically improbable unless they're converging to the same solution
- This often happens when gradient signal is too weak

### 2. Single Timestep Limitation

Current setup: 1 timestep → predict 49 timesteps

Problems:
- No temporal context for physics understanding
- TTT's self-supervised tasks fail (need trajectory segments)
- Adaptation based on single point is like extrapolating from one data point

### 3. Model Architecture Issues

Current model might not have enough "adaptable capacity":
- Only 2 BatchNorm layers for TENT to update
- Dense layers might not capture physics invariances well
- No explicit temporal modeling (LSTM, Transformer)

## Improvement Strategy

### Phase 1: Immediate Fixes (High Priority)

1. **Multi-Timestep Inputs**
   ```python
   # Instead of: shape=(1, 8) → (49, 8)
   # Use: shape=(5, 8) → (10, 8)
   ```
   - Provides temporal context
   - Enables TTT's reconstruction tasks
   - Richer gradient signal

2. **Hyperparameter Tuning**
   - Learning rates: [1e-3, 1e-2, 1e-1]
   - Adaptation steps: [10, 20, 50]
   - Different optimizers: SGD with momentum

3. **Debug Adaptation Dynamics**
   - Track weight changes during adaptation
   - Verify BatchNorm statistics are updating
   - Monitor gradient magnitudes

### Phase 2: Architecture Improvements

1. **Temporal Models**
   - LSTM/GRU for sequence modeling
   - 1D CNN for local temporal patterns
   - Transformer for long-range dependencies

2. **More Adaptable Layers**
   - Add more BatchNorm layers
   - Use LayerNorm (updates all parameters)
   - Adaptive Instance Normalization

3. **Physics-Informed Architecture**
   - Separate modules for position/velocity
   - Energy-conserving layers
   - Symmetry-preserving operations

### Phase 3: Advanced TTA Methods

1. **Meta-Learning Integration**
   - Train model to be good at adapting
   - MAML-style initialization
   - Learned learning rates

2. **Ensemble TTA**
   - Multiple models with different initializations
   - Adapt each independently
   - Ensemble predictions

3. **Source-Free Domain Adaptation**
   - Recent techniques from SFDA literature
   - Pseudo-labeling with confidence
   - Entropy minimization variants

## Experiments to Run

### Experiment 1: Multi-Step Validation
```bash
python evaluate_tta_multistep.py
```
Expected: 5-15% improvement with 5-step inputs

### Experiment 2: Hyperparameter Search
```bash
python tta_hyperparameter_search.py
```
Expected: Find at least one configuration with >0% improvement

### Experiment 3: Debug Current Failure
```bash
python debug_tta_convergence.py
```
Expected: Understand why all methods converge to same value

## Success Metrics

1. **Short Term** (Today):
   - At least one TTA configuration shows >0% improvement
   - Understand convergence issue
   - Identify best hyperparameters

2. **Medium Term** (This Week):
   - Achieve 10-20% improvement on time-varying gravity
   - Show larger improvements on extreme OOD (rotating frame)
   - Develop reliable TTA protocol

3. **Long Term** (Paper-worthy):
   - 20-40% improvement on genuine OOD
   - Novel TTA method for physics
   - Theoretical understanding of when/why TTA helps

## Key Insights from Literature

1. **TENT (Wang et al., 2021)**:
   - Works best with sufficient BatchNorm layers
   - Requires confident predictions to avoid collapse
   - Learning rate crucial: too small = no effect, too large = instability

2. **TTT (Sun et al., 2020)**:
   - Needs meaningful self-supervised task
   - Reconstruction from partial observations works well
   - Auxiliary loss weight balancing is critical

3. **Physics-Specific Considerations**:
   - Conservation laws provide strong adaptation signal
   - Symmetries can guide adaptation
   - Energy-based losses prevent unrealistic adaptations

## Action Items

1. ✓ Created debugging script
2. ✓ Created multi-step evaluation
3. ✓ Created hyperparameter search
4. ⏳ Run experiments and analyze results
5. ⏳ Implement best configuration
6. ⏳ Test on extreme OOD scenarios

## Expected Outcomes

With proper configuration, we expect:
- **Time-varying gravity**: 10-20% improvement
- **Rotating frame**: 20-30% improvement  
- **Spring coupling**: 15-25% improvement

The key is providing enough information (multi-step) and adaptation capacity (architecture + hyperparameters) for meaningful updates.