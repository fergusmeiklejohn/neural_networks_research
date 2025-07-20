# Physics Worlds Experiment - Current Status

Last Updated: 2025-07-19

## 🎯 Current State Summary

**Test-Time Adaptation V2 Implemented**: Successfully created JAX-compatible TTA infrastructure with full gradient support. Fixed critical BatchNorm weight restoration bug. Implemented regression-specific TTA methods (`regression_tta_v2.py`). Initial tuning shows TTA degrading performance - needs investigation.

**True OOD Data Generated**: Created multiple genuine out-of-distribution scenarios:
- Time-varying gravity: ~49% true OOD (5.68x distance from training)
- Rotating frame physics: ~65% true OOD (Coriolis forces)
- Spring coupled balls: ~70% true OOD (new interaction type)

**Paper Revision Complete**: Successfully revised OOD evaluation paper with k-NN analysis showing 96-97% interpolation rate. Second reviewer provided very positive assessment - paper ready for submission to top-tier venues.

**Major Discovery Confirmed**: The "OOD Illusion" is real - current benchmarks test interpolation, not extrapolation. Our baseline tests reveal a 3,000x performance gap between reported results and true OOD.

**Complete Baseline Testing Done**: 
- GraphExtrap (paper): 0.766 MSE ✅
- GFlowNet (our test): 2,229.38 MSE (2,910x worse) ❌
- MAML (our test): 3,298.69 MSE (4,306x worse) ❌
- Minimal PINN: 42,532.14 MSE (55,531x worse) ❌

**Critical Insight**: NO current method achieves true extrapolation. GraphExtrap's success comes from seeing diverse training data, not understanding physics.

## 📊 Latest Results

### Complete Baseline Comparison (January 15, 2025)
| Model | Jupiter MSE | Parameters | vs Best | Status |
|-------|-------------|------------|---------|--------|
| GraphExtrap (paper) | 0.766 | ~100K | 1x | ✅ Best |
| MAML (paper) | 0.823 | - | 1.07x | ✅ Good |
| GFlowNet (paper) | 0.850 | - | 1.11x | ✅ Good |
| ERM+Aug (paper) | 1.128 | - | 1.47x | ✅ Acceptable |
| Original PINN | 880.879 | 1.9M | 1,150x | ❌ Failed |
| **GFlowNet (our)** | **2,229.38** | 152K | 2,910x | ❌ Failed |
| **MAML (our)** | **3,298.69** | 56K | 4,306x | ❌ Failed |
| **Minimal PINN** | **42,532.14** | 5K | 55,531x | ❌ Catastrophic |

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
   
4. **Test-Time Adaptation (TTA)**:
   - Fixed critical weight restoration bug in `base_tta.py` and `base_tta_jax.py`
   - TTA now properly saves/restores ALL variables including BatchNorm statistics
   - Ready for hyperparameter tuning and full evaluation

## ❗ Known Issues

1. **TTA Performance** (UPDATED July 19 - JAX Implementation Complete):
   - ✅ Full JAX gradient computation implemented (`base_tta_jax_v2.py`)
   - ✅ Regression-specific TTA created (`regression_tta_v2.py`)
   - ✅ Complete state restoration working (0.0 error)
   - ⚠️ Performance still negative - needs hyperparameter re-tuning
   - 📝 Can now update all parameters, not just BatchNorm stats

2. **Data Format Quirks**:
   - Gravity values in pixels/s² (not m/s²)
   - Must use `physics_config['gravity'] / 40.0` for SI units
   - Jupiter gravity shows as -42.8 in some data (should be -24.8)

3. **Unfinished Baselines**:
   - GFlowNet and MAML not yet tested on physics
   - Need to verify GraphExtrap training conditions

## 📝 Paper Revision Complete (July 17, 2025)

### Major Revisions Implemented:
1. **k-NN Analysis**: Replaced convex hull with k-NN distance metric (k=10)
   - Results: 96-97% of "far-OOD" within 99th percentile
   - More robust to high dimensionality
   - Created `analyze_representations_knn.py`

2. **Full MSE Values**: Updated Table 5 with actual degradation
   - GFlowNet: 487,293 MSE (219x degradation)
   - MAML: 652,471 MSE (198x degradation)
   - GraphExtrap: 1,247,856 MSE (1.6M× degradation)
   - Minimal PINN: 8,934,672 MSE (210x degradation)

3. **Language & References**: 
   - Softened universal claims throughout
   - Added 2025 references (Fesser, Kim, Wang)
   - Maintained scholarly tone

4. **Second Reviewer Assessment**: "High-quality paper suitable for NeurIPS/ICML/ICLR"

**Paper Location**: `papers/ood_evaluation_analysis/ood_evaluation_analysis_complete.md`

## 🚀 Immediate Next Steps

### 1. Re-tune Hyperparameters for Gradient-Based TTA (HIGHEST PRIORITY)
```bash
# Test the new JAX TTA implementation
/Users/fergusmeiklejohn/miniconda3/envs/dist-invention/bin/python experiments/01_physics_worlds/test_jax_tta_v2.py

# Create new hyperparameter tuning script for V2
/Users/fergusmeiklejohn/miniconda3/envs/dist-invention/bin/python experiments/01_physics_worlds/tune_tta_hyperparameters_v2.py
```
- ✅ JAX gradient computation implemented and working
- Now need to find optimal hyperparameters for gradient-based updates
- Try much lower learning rates (1e-5 to 1e-6) for full parameter updates

### 2. Run Full TTA Evaluation ✅
```bash
conda activate dist-invention
python experiments/01_physics_worlds/evaluate_tta_comprehensive.py
```
- Test all three TTA methods on time-varying gravity
- Quantify improvement percentages
- Generate performance comparison table

### 3. Verify Extreme OOD Status
```bash
python experiments/01_physics_worlds/verify_true_ood_simple.py
```
- Confirm rotating frame is >65% true OOD
- Confirm spring coupling is >70% true OOD
- Use k-NN analysis with physics features

### 4. Test TTA on Extreme OOD
```bash
python experiments/01_physics_worlds/evaluate_tta_simple.py
```
- Evaluate on rotating frame physics
- Evaluate on spring coupled physics
- Compare improvement vs time-varying gravity

### 5. Implement True OOD Benchmark (Level 2)
- Design in: `TRUE_OOD_BENCHMARK.md:L36-47`
- Add time-varying gravity: `gravity_fn=lambda t: -9.8 * (1 + 0.1*sin(t))`
- Verify >60% samples are true OOD using RepresentationSpaceAnalyzer

### 6. Complete Baseline Evaluation
```bash
python train_baselines.py --model graph_extrap --verbose
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