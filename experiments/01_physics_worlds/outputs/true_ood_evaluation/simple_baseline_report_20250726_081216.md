# Simple Baseline True OOD Evaluation

Generated: 20250726_081216

## Model Details
- Architecture: Simple feedforward neural network
- Parameters: 50,448
- Training data: Constant gravity only

## Results Summary

### Validation Set (Constant Gravity)
- MSE: 3315.34
- MAE: 20.24
- Status: Expected baseline performance

### True OOD Test (Time-Varying Gravity)
- MSE: 1657392775.62
- MAE: 5596.83
- Status: catastrophic_failure
- **Degradation: 499916.3x worse than validation**

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
- `simple_baseline_results_20250726_081216.json`: Detailed results
- `simple_baseline_ood_comparison_20250726_081216.png`: Trajectory visualizations
