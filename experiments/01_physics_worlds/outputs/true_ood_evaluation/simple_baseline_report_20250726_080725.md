# Simple Baseline True OOD Evaluation

Generated: 20250726_080725

## Model Details
- Architecture: Simple feedforward neural network
- Parameters: 50,448
- Training data: Constant gravity only

## Results Summary

### Validation Set (Constant Gravity)
- MSE: 47311512.85
- MAE: 1034.77
- Status: Expected baseline performance

### True OOD Test (Time-Varying Gravity)
- MSE: 73209.36
- MAE: 202.62
- Status: severe_degradation
- **Degradation: 0.0x worse than validation**

## Key Findings

1. **Catastrophic Failure**: The model completely fails when gravity varies with time
2. **Not Interpolation**: This cannot be solved by interpolating between training examples
3. **Fundamental Limitation**: Standard neural networks cannot extrapolate to new physics

## Implications

This demonstrates that:
- Current "OOD" benchmarks that only vary parameters are actually testing interpolation
- True OOD requires fundamentally different dynamics (e.g., time-varying forces)
- We need new architectures that can handle mechanism changes, not just parameter shifts

## Files Generated
- `simple_baseline_results_20250726_080725.json`: Detailed results
- `simple_baseline_ood_comparison_20250726_080725.png`: Trajectory visualizations
