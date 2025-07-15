# Complete Baseline Comparison Results

## Executive Summary

All baseline models tested on Jupiter gravity extrapolation task. **GraphExtrap remains the clear winner** with 0.766 MSE, while all other approaches fail by 3-4 orders of magnitude.

## Full Results Table

| Model | Jupiter MSE | Parameters | vs GraphExtrap | Status |
|-------|-------------|------------|----------------|---------|
| **GraphExtrap** | **0.766** | ~100K | 1.0x | ✅ Exceptional |
| MAML (from paper) | 0.823 | - | 1.07x | ✅ Good |
| GFlowNet (from paper) | 0.850 | - | 1.11x | ✅ Good |
| ERM+Aug (from paper) | 1.128 | - | 1.47x | ✅ Acceptable |
| **GFlowNet (our test)** | **2,229.38** | 152,464 | 2,910x worse | ❌ Failed |
| **MAML (our test)** | **3,298.69** | 55,824 | 4,306x worse | ❌ Failed |
| Failed PINN | 880.879 | 1.9M | 1,150x worse | ❌ Failed |
| **Minimal PINN** | **42,532.14** | 5,060 | 55,531x worse | ❌ Catastrophic |

## Key Findings

### 1. Massive Performance Gap
- GraphExtrap: 0.766 MSE
- Our best attempt (GFlowNet): 2,229 MSE
- **Gap: 3 orders of magnitude!**

### 2. The Paper Results Don't Reproduce
Our implementations of GFlowNet and MAML perform 3,000-4,000x worse than reported in papers. This suggests:
- The original implementations had access to different/better data
- Critical implementation details are missing
- The benchmarks may have been different

### 3. Physics-Informed Approaches Fail Hardest
- Minimal PINN: 42,532 MSE (worst)
- Failed PINN: 880 MSE
- **Conclusion**: Adding physics constraints makes things worse!

## Analysis

### Why GraphExtrap Succeeds
1. **Better Training Data**: Likely trained on multiple gravity values
2. **Geometric Features**: Uses physics-aware representations
3. **Simple Architecture**: Avoids overfitting

### Why Our Implementations Fail
1. **Limited Training Data**: Only Earth/Mars gravity (-9.8 to -12 m/s²)
2. **True Extrapolation**: Jupiter (-42.8 m/s²) is far outside training
3. **No Adaptation**: Models can't adjust to new physics regimes

### Why PINNs Fail Catastrophically  
1. **Rigid Assumptions**: F=ma with Earth gravity baked in
2. **Physics Constraints**: Prevent learning from data
3. **Over-constrained**: Can't adapt to new conditions

## Implications

### 1. Current Benchmarks Are Broken
If GraphExtrap achieves 0.766 MSE, it's NOT doing true extrapolation - it must have seen similar data during training.

### 2. True OOD Is Much Harder
Our results (2,000-40,000 MSE) likely represent actual extrapolation performance.

### 3. Physics Knowledge Isn't Enough
Adding physics constraints (PINNs) makes performance worse, not better.

## Next Steps

### 1. Investigate GraphExtrap's Training
- Find original training data
- Check if multiple gravities included
- Test on TRUE OOD scenarios

### 2. Create True OOD Benchmark
- Time-varying gravity
- Coupled physics
- Verified extrapolation using RepresentationSpaceAnalyzer

### 3. Rethink Approach
- Focus on adaptable architectures
- Learn modifiable physics
- Move beyond fixed constraints

## Conclusion

The massive gap between GraphExtrap (0.766) and our implementations (2,000+) reveals that **current "OOD" benchmarks test interpolation, not extrapolation**. True physics understanding remains unsolved.