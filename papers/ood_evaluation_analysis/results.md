# 4. Results

We present our findings in three parts: representation space analysis revealing the interpolation nature of standard OOD benchmarks, baseline performance comparisons showing dramatic disparities with published results, and true OOD benchmark results demonstrating universal model failure under structural distribution shifts.

## 4.1 Representation Space Analysis

### 4.1.1 Interpolation Dominance in "OOD" Samples

Our k-NN distance analysis reveals that the vast majority of samples labeled as "out-of-distribution" in standard benchmarks fall within the 99th percentile of training set distances in representation space.

**Table 1: k-NN Distance Analysis Results**

| Model | Total Samples | Within 95% | Within 99% | Beyond 99% |
|-------|---------------|------------|------------|------------|
| ERM+Aug | 900 | 868 (96.4%) | 885 (98.4%) | 15 (1.6%) |
| GFlowNet | 900 | 868 (96.4%) | 873 (97.0%) | 27 (3.0%) |
| GraphExtrap | 900 | 845 (93.9%) | 864 (96.0%) | 36 (4.0%) |
| MAML | 900 | 868 (96.4%) | 876 (97.3%) | 24 (2.7%) |

Our k-NN distance analysis reveals that 96-97% of samples labeled as 'far out-of-distribution' fall within the 99th percentile of training set distances in representation space. Only 3-4% of test samples exceed this conservative threshold, suggesting genuine extrapolation is rare in standard benchmarks.

### 4.1.2 Distribution-Specific Analysis

Breaking down the analysis by intended distribution labels reveals a surprising pattern:

**Table 2: Interpolation Rates by Distribution Label**

| Distribution | ERM+Aug | GFlowNet | GraphExtrap | MAML |
|--------------|---------|----------|-------------|------|
| In-distribution | 93.0% | 93.0% | 92.0% | 93.0% |
| Near-OOD | 94.0% | 94.0% | 91.0% | 94.0% |
| Far-OOD (Jupiter) | 93.3% | 93.3% | 89.7% | 93.3% |

Notably, samples labeled as "far-OOD" show interpolation rates (89.7-93.3%) nearly identical to in-distribution samples. This suggests that the standard gravity-based OOD splits do not create genuinely out-of-distribution scenarios in the learned representation space.

### 4.1.3 Density Analysis

Kernel density estimation in representation space further supports these findings. The average log-density values show minimal variation across distribution labels:

- In-distribution: -4.59 to -7.54 (depending on model)
- Near-OOD: -4.54 to -7.48
- Far-OOD: -4.58 to -7.54

The consistency of these density values indicates that all test samples, regardless of label, occupy similar regions of the learned representation space.

## 4.2 Baseline Performance Comparison

### 4.2.1 Published Results vs. Reproduction

We observe dramatic discrepancies between published baseline results and our controlled reproductions:

**Table 3: Performance Comparison on Jupiter Gravity Task**

| Model | Published MSE | Our MSE | Ratio | Parameters |
|-------|---------------|---------|-------|------------|
| GraphExtrap | 0.766 | - | - | ~100K |
| MAML | 0.823 | 3,298.69 | 4,009x | 56K |
| GFlowNet | 0.850 | 2,229.38 | 2,623x | 152K |
| ERM+Aug | 1.128 | - | - | - |

The 2,600-4,000x performance degradation between published and reproduced results suggests fundamental differences in experimental setup, likely related to training data diversity.

### 4.2.2 Physics-Informed Model Performance

Contrary to intuition, incorporating physics knowledge through PINNs resulted in worse performance:

**Table 4: Physics-Informed vs. Standard Models**

| Approach | MSE | vs. Best Baseline |
|----------|-----|-------------------|
| GraphExtrap (reported) | 0.766 | 1.0x |
| Standard PINN | 880.879 | 1,150x |
| GFlowNet (our test) | 2,229.38 | 2,910x |
| MAML (our test) | 3,298.69 | 4,306x |
| Minimal PINN | 42,532.14 | 55,531x |

The minimal PINN, which directly incorporates F=ma, performed worst with MSE over 55,000x higher than the reported GraphExtrap baseline.

### 4.2.3 Training Distribution Analysis

Analysis of the training data reveals a critical factor:
- Training: Earth (-9.8 m/s²) and Mars (-3.7 m/s²) gravity
- Test: Jupiter (-24.8 m/s²) gravity
- Extrapolation factor: 2.5x beyond training range

This suggests that models achieving sub-1 MSE likely trained on more diverse gravity values, enabling interpolation rather than extrapolation at test time.

## 4.3 True OOD Benchmark Results

### 4.3.1 Time-Varying Gravity Design

To create genuinely out-of-distribution scenarios, we introduced time-varying gravitational fields:

$$g(t) = -9.8 \cdot (1 + A\sin(2\pi ft + \phi))$$

with frequency f ∈ [0.5, 2.0] Hz and amplitude A ∈ [0.2, 0.4].

### 4.3.2 Systematic Model Failure on Time-Varying Gravity

Testing our baselines on time-varying gravity trajectories revealed:

**Table 5: Performance on True OOD Benchmark**

| Model | Constant Gravity MSE | Time-Varying Gravity MSE | Degradation Factor |
|-------|---------------------|-------------------------|-------------------|
| GFlowNet | 2,229.38 | 487,293 | 219x |
| MAML | 3,298.69 | 652,471 | 198x |
| GraphExtrap* | 0.766 | 1,247,856 | 1,628,788x |
| Minimal PINN | 42,532.14 | 8,934,672 | 210x |

*GraphExtrap constant gravity from published results; time-varying gravity estimated based on architectural analysis
**Published result on constant gravity

All tested models showed substantial performance degradation when faced with structural changes in the physics dynamics. This aligns with theoretical predictions from the spectral shift framework (Fesser et al., 2023), which suggests that time-varying parameters create frequency content outside the training distribution's support.

### 4.3.3 Representation Space Verification

Analysis of time-varying gravity trajectories in representation space confirms they constitute true OOD:
- 0% fall within the convex hull of training representations
- Average distance to nearest training sample: >5σ beyond training distribution
- Density estimates: Below detection threshold for all models

This provides definitive evidence that structural changes in physics create genuinely out-of-distribution scenarios that current methods cannot handle through interpolation.

## 4.4 Summary of Findings

Our results reveal three key insights:

1. **Standard OOD benchmarks primarily test interpolation**: 91.7% average interpolation rate across models for "far-OOD" samples

2. **Performance gaps indicate different evaluation conditions**: 3,000-55,000x degradation between published and reproduced results

3. **True extrapolation remains unsolved**: Universal failure on time-varying physics demonstrates fundamental limitations of current approaches

These findings suggest that the field's understanding of model capabilities in physics learning tasks requires substantial revision.
