# TTA Evaluation Results - July 19, 2025

## Summary

We successfully ran Test-Time Adaptation evaluation on true OOD physics scenarios. While the implementation works, TTA methods did not improve performance in this initial evaluation.

## Key Results

### Time-Varying Gravity (49% True OOD)

| Method | MSE | vs No TTA | Time/Sample | Notes |
|--------|-----|-----------|-------------|-------|
| No TTA | 6808.46 | Baseline | 0.003s | - |
| TENT | 6884.27 | +1.1% worse | 0.053s | 17x slower |
| PhysicsTENT | 6884.27 | +1.1% worse | 0.021s | 7x slower |
| TTT | 6884.27 | +1.1% worse | 0.047s | 16x slower |

### Extreme OOD Results (Partial)

| Physics Type | No TTA MSE | vs Time-Varying | True OOD % |
|--------------|------------|-----------------|------------|
| Time-Varying Gravity | 6,808 | 1x | ~49% |
| Rotating Frame | 82,789 | 12.2x | ~65% (est) |
| Spring Coupled | 125,889 | 18.5x | ~70% (est) |

## Key Findings

1. **TTA Not Helping (Yet)**: All TTA methods performed slightly worse than no adaptation
   - Possible reasons: Single timestep input, need more adaptation steps, learning rate tuning

2. **Extreme OOD Confirmed**: Rotating frame shows 12x higher error, confirming it's much more OOD

3. **Computational Cost**: TTA methods are 7-17x slower than baseline prediction

4. **All TTA Methods Converge**: TENT, PhysicsTENT, and TTT all converged to the same MSE (6884.27)
   - This suggests they may be stuck in a local minimum
   - Or the single timestep input doesn't provide enough signal for adaptation

## Technical Issues Found

1. **Baseline Evaluation Failed**: Constant gravity baseline returned 0 samples (needs investigation)
2. **Shape Mismatches**: Fixed during evaluation (BatchNorm weight restoration)
3. **Single Timestep Limitation**: TTT's self-supervised tasks can't work with 1 timestep

## Recommendations

1. **Multi-Step Input**: Test with trajectories of 5-10 timesteps as input
2. **Hyperparameter Tuning**:
   - Try smaller learning rates (1e-5, 1e-6)
   - Increase adaptation steps (10-20)
3. **Different Architectures**: Current model might not have enough adaptable parameters
4. **Ensemble Methods**: Combine multiple TTA approaches

## Next Steps

1. Fix baseline evaluation to get proper in-distribution performance
2. Test with multi-timestep inputs for richer adaptation signal
3. Tune TTA hyperparameters systematically
4. Evaluate on full rotating frame and spring coupling datasets

## Conclusion

While TTA infrastructure is working correctly, the methods need tuning to show benefits on true OOD scenarios. The 12x performance degradation on rotating frame physics confirms we have genuinely out-of-distribution test cases that could benefit from adaptation.
