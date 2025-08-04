# 4. Empirical Results

We present comprehensive results across two physics systems with mechanism shifts, including tests of physics-aware adaptation variants and collapse detection.

## 4.1 Two-Ball Dynamics with Time-Varying Gravity

### 4.1.1 Baseline Performance

We first establish baseline performance using standard empirical risk minimization (ERM). Table 1 shows the model achieves low error on constant gravity while experiencing significant degradation on time-varying gravity.

**Table 1: Baseline Model Performance (Two-Ball System)**
| Test Set | MSE | 95% CI | Relative to Training |
|----------|-----|--------|---------------------|
| Training (constant g) | 89.3 | ±5.2 | 1.0x |
| In-distribution test | 99.7 | ±7.1 | 1.1x |
| Time-varying gravity | 2,721.1 | ±145.3 | 30.5x |

The 30-fold increase in error (p < 0.001, n=5 seeds) indicates that time-varying gravity represents a true mechanism shift requiring different computational operations.

### 4.1.2 Test-Time Adaptation Results

Table 2 presents results for test-time adaptation. Contrary to results on standard benchmarks, TTA consistently degrades performance.

**Table 2: Test-Time Adaptation Performance (Two-Ball System)**
| Adaptation Steps | In-Dist MSE | OOD MSE | OOD Degradation | p-value |
|-----------------|-------------|---------|-----------------|---------|
| 0 (baseline) | 99.7 ± 7.1 | 2,721.1 ± 145.3 | 1.0x | - |
| 1 | 156.2 ± 12.3 | 6,935.0 ± 412.7 | 2.55x | <0.001 |
| 10 | 234.8 ± 18.9 | 8,420.3 ± 523.8 | 3.09x | <0.001 |
| 50 | 445.3 ± 31.2 | 9,156.7 ± 601.4 | 3.36x | <0.001 |

All degradations are statistically significant. The model converges to nearly constant predictions, achieving high consistency at the cost of accuracy.

### 4.1.3 Meta-Learning Results

MAML shows even more severe degradation when adaptation is applied:

**Table 3: MAML Performance (Two-Ball System)**
| Configuration | In-Dist MSE | OOD MSE | OOD Degradation |
|--------------|-------------|---------|-----------------|
| ERM baseline | 99.7 ± 7.1 | 2,721.1 ± 145.3 | 1.0x |
| MAML (no adaptation) | 112.3 ± 8.4 | 3,018.8 ± 189.2 | 1.11x |
| MAML (10-shot adaptation) | 89,234.2 ± 5,123.7 | 1,697,689.5 ± 98,234.1 | 623.8x |

The 62,290% increase in error with adaptation represents catastrophic failure.

## 4.2 Pendulum with Time-Varying Length

To test generality and address reviewer concerns about empirical breadth, we implemented a pendulum experiment where mechanism shift involves time-varying length L(t) = L₀(1 + 0.2sin(0.1t)).

### 4.2.1 Baseline Performance

**Table 4: Baseline Performance (Pendulum System)**
| Test Set | MSE | 95% CI | Relative to Training |
|----------|-----|--------|---------------------|
| Training (fixed L) | 0.0003 | ±0.0001 | 1.0x |
| Test (fixed L) | 0.0003 | ±0.0001 | 1.0x |
| Test (varying L) | 0.0005 | ±0.0001 | 1.4x |

The milder degradation (1.4x vs 30x for two-ball) reflects the simpler system dynamics, but mechanism shift still causes measurable performance loss (p < 0.05).

### 4.2.2 Physics-Aware Test-Time Adaptation

We implemented physics-informed TTA variants using energy conservation and Hamiltonian consistency losses:

**Table 5: Physics-Aware TTA Results (Pendulum System)**
| Method | Fixed L MSE | Time-Varying L MSE | Degradation | p-value |
|--------|-------------|-------------------|-------------|---------|
| Baseline (no TTA) | 0.0003 ± 0.0001 | 0.0005 ± 0.0001 | 1.4x | - |
| Prediction TTA | 0.0003 ± 0.0001 | 0.0065 ± 0.0008 | 14.4x | <0.001 |
| Energy TTA | 0.0003 ± 0.0001 | 0.0057 ± 0.0006 | 12.6x | <0.001 |
| Hamiltonian TTA | 0.0003 ± 0.0001 | 0.0081 ± 0.0009 | 17.9x | <0.001 |

**Critical finding**: Even physics-aware losses degrade performance because they encode conservation assumptions that mechanism shifts violate. Energy is not conserved when pendulum length varies (work is done), so enforcing energy conservation misleads adaptation.

### 4.2.3 Collapse Detection Analysis

Inspired by PeTTA (Bohdal et al., 2024), we implemented collapse detection monitoring prediction entropy, variance, and parameter drift:

**Table 6: PeTTA-Inspired Collapse Detection Results**
| Method | MSE | Degradation | Collapse Events | Entropy Reduction |
|--------|-----|-------------|-----------------|-------------------|
| Standard TTA | 0.006256 ± 0.0007 | 13.90x | N/A | N/A |
| PeTTA-inspired TTA | 0.006252 ± 0.0007 | 13.89x | 0/20 | 2.0% |

**Key insight**: Collapse detection worked as designed—no degenerate solutions occurred. However, performance improvement was negligible (0.06%, p > 0.5). The model maintained diverse but systematically wrong predictions, lacking computational structure for the L̇/L term.

## 4.3 Gradient Alignment Analysis

To understand why adaptation fails, we computed the alignment between gradients of self-supervised losses and true prediction error:

**Table 7: Gradient Alignment (cosine similarity)**
| Task | Loss Type | In-Distribution | Mechanism Shift |
|------|-----------|-----------------|-----------------|
| Two-ball | Prediction consistency | 0.73 ± 0.08 | -0.41 ± 0.12 |
| Pendulum | Prediction consistency | 0.68 ± 0.09 | -0.38 ± 0.11 |
| Pendulum | Energy conservation | 0.81 ± 0.06 | -0.52 ± 0.13 |

Negative alignment on mechanism shifts means adaptation moves away from accurate solutions. This explains the systematic performance degradation.

## 4.4 Summary Across All Experiments

Figure 1 summarizes performance across both systems and all methods:

**Two-Ball System (Time-Varying Gravity):**
- Best: Deep Ensemble (0.99x baseline)
- Worst: MAML with adaptation (623.8x baseline)
- TTA methods: 2.35-3.36x degradation

**Pendulum System (Time-Varying Length):**
- Best: No adaptation (1.4x over fixed length)
- Standard TTA: 14.4x degradation
- Physics-aware TTA: 12.6-17.9x degradation
- PeTTA-inspired: 13.89x degradation

## 4.5 Statistical Summary

All reported degradations are statistically significant (p < 0.001 unless noted). Key findings:

1. **No method improves over baseline** on mechanism shifts (best case: 0.98x)
2. **All adaptation methods degrade performance** (2.35x to 623.8x worse)
3. **Physics-aware losses don't help** when physics changes (12.6-17.9x degradation)
4. **Collapse detection maintains stability** but not accuracy (13.89x degradation)

These results demonstrate that current self-supervised adaptation methods, including recent advances, face fundamental limitations when test distributions involve different generative processes than training data.
