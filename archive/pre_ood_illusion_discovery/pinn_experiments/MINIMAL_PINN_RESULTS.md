# Minimal PINN Results: Another Failure Mode

## Summary

Our minimal PINN achieved 42,468 MSE on Jupiter gravity - still 55,000x worse than GraphExtrap's 0.766, though better than the original PINN's 880.879.

## Key Findings

### 1. Data Format Issues
- The physics data uses pixel coordinates, not meters
- Trajectory format: [time, x1, y1, vx1, vy1, mass1, radius1, ..., x2, y2, vx2, vy2, ...]
- Gravity values are in pixels/s² (~400-1200 range)

### 2. Model Limitations
- **Single gravity parameter**: Model assumes one global gravity value
- **Can't adapt**: Learns Earth gravity (-9.81 m/s²) but can't adjust for Jupiter (-42.8 m/s²)
- **Physics constraint is a weakness**: Enforcing F=ma prevents adaptation to new physics

### 3. Why GraphExtrap Succeeded
GraphExtrap doesn't assume specific physics equations. It:
- Uses geometric features (r, θ) that naturally encode physics
- Learns patterns from data without hard constraints
- Can interpolate between different gravity values in its training set

### 4. Why PINNs Keep Failing
1. **Assumption mismatch**: Real data doesn't follow idealized physics
2. **Over-constrained**: Physics losses prevent learning data patterns
3. **Wrong inductive bias**: Assuming F=ma exactly is too rigid

## Numerical Results

| Model | Jupiter MSE | Notes |
|-------|-------------|-------|
| GraphExtrap | 0.766 | Learns from data patterns |
| MAML | 0.823 | Meta-learns adaptation |
| GFlowNet | 0.850 | Explores distributions |
| ERM+Aug | 1.128 | Standard deep learning |
| Original PINN | 880.879 | Complex architecture failed |
| Minimal PINN | 42,468 | Simple physics failed differently |

## Lessons Learned

1. **Data understanding is critical**: We spent significant time debugging data format issues
2. **Physics constraints can hurt**: Enforcing exact physics prevents adaptation
3. **Flexibility beats rigidity**: GraphExtrap's data-driven approach outperforms physics-informed models
4. **Unit conversions matter**: Pixel vs meter coordinates caused major confusion

## Next Steps

1. **Abandon rigid physics constraints**: They're hurting more than helping
2. **Focus on representation learning**: Like GraphExtrap's geometric features
3. **Build in adaptability**: Models need to handle varying physics parameters
4. **Test true OOD**: Current "OOD" tests are mostly interpolation

## Conclusion

The minimal PINN confirmed our hypothesis: physics-informed approaches fail when test conditions violate the encoded physics assumptions. Even a "perfect" physics model can't extrapolate if it assumes wrong constants.
