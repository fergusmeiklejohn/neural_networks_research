# 6. A Taxonomy of Distribution Shifts

## 6.1 Existing Categorizations

The machine learning community has developed several frameworks for understanding distribution shift. We build upon these while proposing a distinction based on whether the shift requires learning new computational mechanisms.

Traditional categorizations focus on which probability distributions change:
- **Covariate shift**: P(X) changes, P(Y|X) unchanged
- **Concept drift**: P(Y|X) changes over time
- **Prior shift**: P(Y) changes, P(X|Y) unchanged

While mathematically precise, these categories do not capture whether a model trained on the source distribution can, in principle, compute correct predictions on the target distribution using its learned features.

## 6.2 Proposed Taxonomy

We propose categorizing distribution shifts based on the computational requirements:

### Level 1: Surface Variations
**Definition**: Changes that affect input appearance but not the underlying computation.

**Examples**:
- Image corruptions (noise, blur, compression artifacts)
- Style transfer (photo to painting)
- Sensor variations (different cameras)

**Model Requirements**: Robustness to input perturbations

**Why Current Methods Work**: The same features remain relevant; only their values change.

### Level 2: Statistical Shifts
**Definition**: Changes in data statistics while maintaining the same generative process.

**Examples**:
- Different object frequencies in image classification
- Demographic shifts in medical data
- Seasonal variations in time series

**Model Requirements**: Calibrated uncertainty, robust statistics

**Why Current Methods Partially Work**: The computation remains the same, but operating points shift.

### Level 3: Mechanism Changes
**Definition**: Changes in the data-generating process requiring different computations.

**Examples**:
- Constant to time-varying parameters (our physics task)
- Rule changes in games
- Economic regime changes

**Model Requirements**: Ability to learn or construct new computational pathways

**Why Current Methods Show Limited Success**: Learned features may not support the required computations.

## 6.3 Empirical Validation

We validated this taxonomy by testing how different methods perform across shift levels:

**Table: Method Performance by Shift Level**
| Shift Level | Example | ERM | TTA | MAML | Best Method |
|-------------|---------|-----|-----|------|-------------|
| Level 1 | Add noise | 1.5x | 1.3x | 1.4x | TTA (-13%) |
| Level 2 | New ball sizes | 1.8x | 1.7x | 1.9x | TTA (-6%) |
| Level 3 | Time-varying g | 27x | 69x | 6238x | ERM (baseline) |

The results suggest that adaptation methods provide benefits for Level 1-2 shifts but show substantial performance degradation on Level 3 shifts.

## 6.4 Identifying Shift Levels

How can we determine which level a distribution shift belongs to? We propose several diagnostic approaches:

### Representation Analysis
Project data into the learned representation space (e.g., penultimate layer activations):
- **Level 1-2**: Test data falls within or near the convex hull of training representations
- **Level 3**: Test data lies far outside the training representation hull

### Gradient Alignment
Compare gradients of task loss vs. self-supervised loss:
- **Level 1-2**: Positive or near-zero alignment
- **Level 3**: Negative alignment (as seen in our experiments)

### Oracle Performance
Train a model with access to a small amount of labeled target data:
- **Level 1-2**: Rapid improvement with few examples
- **Level 3**: Requires substantial retraining

## 6.5 Implications for Benchmark Design

Current benchmarks predominantly test Level 1-2 shifts:
- **PACS**: Level 1 (style changes)
- **ImageNet-C**: Level 1 (corruptions)
- **Wilds**: Mostly Level 2 (statistical shifts)

This may explain why methods successful on these benchmarks show reduced performance on physics prediction. Progress in the field may benefit from benchmarks that explicitly test Level 3 shifts.

## 6.6 Characteristics of Level 3 Benchmarks

Effective Level 3 benchmarks should:

1. **Have ground truth**: Enable verification that the shift genuinely changes the required computation
2. **Control the shift**: Allow systematic variation of the mechanism change
3. **Prohibit shortcuts**: Ensure that Level 1-2 robustness cannot solve the task
4. **Support analysis**: Enable inspection of why methods fail

Physics tasks naturally satisfy these criteria, but other domains could include:
- Logic puzzles with rule changes
- Economic models with regime shifts
- Biological systems with mutations
- Games with modified rules

## 6.7 Limitations of the Taxonomy

We acknowledge several limitations:

1. **Boundary cases**: Some shifts may fall between levels or combine multiple levels
2. **Domain dependence**: What constitutes a "mechanism" varies by domain
3. **Model dependence**: A shift's level may depend on the model architecture

Despite these limitations, the taxonomy provides a framework for understanding when adaptation methods are likely to succeed. It suggests that out-of-distribution generalization involving mechanism changes may require different approaches than those currently employed.
