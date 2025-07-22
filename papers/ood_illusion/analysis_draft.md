# 5. Analysis: Why Adaptation Methods Fail

## 5.1 The Objective Mismatch

A key issue in the failure of adaptation methods appears to be a mismatch between their optimization objectives and the task objective. Test-time adaptation methods optimize self-supervised losses such as:

- **Prediction consistency**: Encouraging similar predictions for similar inputs
- **Temporal smoothness**: Minimizing variation in sequential predictions  
- **Entropy minimization**: Reducing uncertainty in predictions

These objectives are motivated by the assumption that confident, consistent predictions are more likely to be correct. This assumption holds when the test distribution involves corruptions or style changes that increase prediction variance. However, when the underlying dynamics change, these objectives may lead to reduced performance.

## 5.2 Gradient Analysis

To understand the optimization dynamics, we analyzed the gradients of different loss functions on time-varying gravity data. Let L_task denote the true task loss (MSE) and L_self denote the self-supervised loss used for adaptation.

We computed:
- ∇L_task: Gradient that would improve prediction accuracy
- ∇L_self: Gradient actually used during adaptation
- cos(∇L_task, ∇L_self): Cosine similarity between gradients

Across test batches, we found:
- Mean cosine similarity: -0.73 ± 0.15
- Percentage with negative similarity: 89%

The negative cosine similarity indicates that optimizing the self-supervised objective moves parameters in a direction that differs substantially from what would improve accuracy. This observation is consistent with the performance degradation observed with increased adaptation steps.

## 5.3 The Degenerate Solution

We observed that adapted models converge to nearly constant predictions. To quantify this, we measured:

**Prediction Variance Analysis**
| Metric | Before Adaptation | After 50 Steps |
|--------|------------------|----------------|
| Std. dev. across inputs | 287.3 | 17.2 |
| Std. dev. across time | 156.8 | 8.4 |
| Mean prediction norm | 523.1 | 498.7 |

The model maintains similar mean predictions while substantially reducing variance. This represents a solution that minimizes the self-supervised loss (achieving high consistency) while showing poor performance on the actual task.

## 5.4 Why Constant Predictions Minimize Self-Supervised Losses

Constant predictions emerge as a local optimum for several self-supervised objectives:

1. **Consistency loss**: If f(x) = c for all x, then consistency is perfect
2. **Smoothness loss**: Constant functions have zero gradient, maximizing smoothness
3. **Entropy**: Deterministic predictions have minimum entropy

Without ground truth labels, the optimization process cannot distinguish between:
- Reducing variance due to noise (beneficial)
- Reducing variance by ignoring inputs (harmful)

On time-varying gravity data, the second pathway dominates.

## 5.5 Information-Theoretic Perspective

From an information theory standpoint, improving predictions on genuinely new dynamics requires information not present in the unlabeled test data. The self-supervised losses provide no information about the correct mapping under the new physics.

Consider the information required:
- Training data provides: P(y|x, g=9.8)
- Test data provides: P(x|g(t))
- Needed for accuracy: P(y|x, g(t))

Without labels or knowledge of the new dynamics, the unlabeled test data may not contain sufficient information to bridge this gap.

## 5.6 Why MAML Fails More Severely

MAML's larger performance degradation (623x worse than baseline) may stem from its design for rapid adaptation. The method learns to take large gradient steps that quickly minimize the loss on a few examples. When applied to self-supervised losses on OOD data, this rapid adaptation may amplify the issues observed with TTA.

We found that MAML's inner loop learning rate was 50x larger than TTA's adaptation rate. This allows MAML to reach suboptimal solutions in fewer steps, which may explain the greater performance degradation.

## 5.7 When Do These Methods Succeed?

To understand the scope of the problem, we tested adaptation methods on other distribution shifts:

**Table: Adaptation Performance on Different Shifts**
| Distribution Shift | TTA Performance | Type of Shift |
|-------------------|-----------------|---------------|
| Gaussian noise | -12% error | Corruption |
| Different ball sizes | -8% error | Covariate |
| Different g constant | +15% error | Parameter |
| Time-varying g | +235% error | Mechanism |

Adaptation appears beneficial when the shift maintains the same underlying computation but adds noise or changes input statistics. Performance degrades when the required computation changes substantially.

## 5.8 Implications for Method Design

Our analysis suggests that successful extrapolation to new mechanisms requires:

1. **Explicit modeling** of how mechanisms might change
2. **Inductive biases** aligned with the domain (e.g., physics constraints)
3. **Uncertainty awareness** to avoid confident but wrong predictions
4. **Richer feedback** than self-supervised losses provide

Current adaptation methods may lack these elements, which could explain their reduced performance on extrapolation tasks involving mechanism changes.