# Physics Worlds Experiment - Current Status

Last Updated: 2025-01-14

## 🎯 Current State Summary

**Major Discovery**: Found the "OOD Illusion" - 91.7% of supposedly "far-OOD" test samples are actually interpolation within the learned representation space.

**PINN Investigation Complete**: Physics-informed models failed catastrophically:
- Original PINN: 880.879 MSE (1,150x worse than baseline)
- Minimal PINN: 42,468 MSE (55,000x worse than baseline)
- GraphExtrap baseline: 0.766 MSE (best performer)

**Key Insight**: Physics constraints prevent adaptation. Models with fixed physics assumptions (like constant gravity) cannot extrapolate to new conditions.

## 📊 Latest Results

### Model Performance on Jupiter Gravity (-24.8 m/s² vs Earth's -9.8 m/s²)
| Model | MSE | Status |
|-------|-----|--------|
| GraphExtrap | 0.766 | ✅ Best |
| ERM + Aug | ~2-3 | ✅ Good |
| GFlowNet | TBD | 🔄 Pending |
| MAML | TBD | 🔄 Pending |
| Original PINN | 880.879 | ❌ Failed |
| Minimal PINN | 42,468 | ❌ Failed |

## 🔧 What's Working

1. **Data Pipeline**: 
   - `train_minimal_pinn.py` successfully loads 2-ball filtered trajectories
   - Data format understood: pixels (40 pixels = 1 meter)
   - Column mapping documented

2. **Baseline Models**:
   - `models/baseline_models.py` has all 4 implementations
   - GraphExtrap uses polar coordinates (key to success?)
   - Training scripts ready

3. **Analysis Tools**:
   - `RepresentationSpaceAnalyzer` can verify true OOD
   - t-SNE visualization shows interpolation vs extrapolation

## ❗ Known Issues

1. **Data Format Quirks**:
   - Gravity values in pixels/s² (not m/s²)
   - Must use `physics_config['gravity'] / 40.0` for SI units
   - Jupiter gravity shows as -42.8 in some data (should be -24.8)

2. **Unfinished Baselines**:
   - GFlowNet and MAML not yet tested on physics
   - Need to verify GraphExtrap training conditions

## 🚀 Immediate Next Steps

### 1. Understand GraphExtrap Success
```bash
python train_baselines.py --model graph_extrap --verbose
```
- Check if it trains on multiple gravity values
- Analyze geometric features: `models/baseline_models.py:L142-156`

### 2. Implement True OOD Benchmark (Level 2)
- Design in: `TRUE_OOD_BENCHMARK.md:L36-47`
- Add time-varying gravity: `gravity_fn=lambda t: -9.8 * (1 + 0.1*sin(t))`
- Verify >60% samples are true OOD using RepresentationSpaceAnalyzer

### 3. Complete Baseline Evaluation
```bash
python train_baselines.py --model gflownet
python train_baselines.py --model maml
python run_unified_evaluation.py
```

## 📝 Key Documents

- **Analyses**: 
  - `MINIMAL_PINN_RESULTS.md` - Why minimal PINN failed
  - `PINN_LESSONS_LEARNED.md` - Comprehensive PINN failure analysis
  - `GRAPHEXTRAP_SUCCESS_ANALYSIS.md` - Why it works
  - `TRUE_OOD_BENCHMARK.md` - Design for real extrapolation tests

- **Next Research**: 
  - `NEXT_RESEARCH_STEPS.md` - Including paper outlines

## 🎯 Research Direction

Moving from "can we make physics-informed models work?" to "how do we design benchmarks that truly test extrapolation?" Focus on:

1. **Paper 1**: "The OOD Illusion in Physics Learning"
2. **Paper 2**: "When Physics-Informed Neural Networks Fail"
3. **True OOD Benchmark**: Time-varying and coupled physics

## 💡 Critical Context

- **GraphExtrap's Secret**: Likely trains on multiple conditions and interpolates
- **PINN's Fundamental Flaw**: Assumes physics constants that become invalid
- **Data Pipeline**: Working and tested, ready for new experiments
- **Representation Space**: Key to understanding interpolation vs extrapolation