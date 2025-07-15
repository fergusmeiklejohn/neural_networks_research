# True OOD Benchmark Summary

## What Makes This True OOD?

### Standard "OOD" (Actually Interpolation)
- Training: Earth gravity (-9.8 m/s²), Mars gravity (-3.7 m/s²)
- Testing: Jupiter gravity (-24.8 m/s²)
- **This is interpolation**: -24.8 can be reached by extrapolating the pattern

### True OOD (Genuine Extrapolation)
- Training: Constant gravity values
- Testing: Time-varying gravity g(t) = -9.8 * (1 + 0.3*sin(2πft))
- **This is true OOD**: No interpolation of constants produces time variation

## Why All Models Fail

1. **Fundamental Assumption Violation**
   - All models assume physics parameters are constant
   - Time variation breaks this core assumption

2. **No Temporal Physics Understanding**
   - Models learn spatial patterns, not temporal dynamics
   - Cannot discover oscillatory behavior from static examples

3. **Causal Structure is Different**
   - Standard physics: acceleration = constant * mass
   - True OOD physics: acceleration = time_function(t) * mass
   - Different causal graph that models haven't seen

## Implications

This demonstrates that current "physics understanding" in ML is actually:
- Pattern matching within seen distributions
- Interpolation between known parameter values
- NOT true understanding of physical laws

True physics understanding would require:
- Learning modifiable causal structures
- Discovering temporal dependencies
- Generalizing to genuinely new physics regimes
