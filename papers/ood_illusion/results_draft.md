# 4. Empirical Results

## 4.1 Baseline Performance

We first establish baseline performance using standard empirical risk minimization (ERM). Table 1 shows the model achieves low error on the in-distribution test set while experiencing significant degradation on time-varying gravity.

**Table 1: Baseline Model Performance**
| Test Set | MSE | Relative to Training |
|----------|-----|---------------------|
| Training (constant g) | 89.3 ± 5.2 | 1.0x |
| In-distribution test | 99.7 ± 7.1 | 1.1x |
| Time-varying gravity | 2,721.1 ± 145.3 | 30.5x |

The 30-fold increase in error indicates that time-varying gravity requires extrapolation. The model maintains accurate predictions for constant gravity but shows reduced performance on the modified dynamics.

## 4.2 Test-Time Adaptation Results

Table 2 presents results for test-time adaptation with different numbers of adaptation steps. Contrary to results on image corruption benchmarks, TTA consistently increases prediction error.

**Table 2: Test-Time Adaptation Performance**
| Adaptation Steps | In-Dist MSE | OOD MSE | OOD vs. Baseline |
|-----------------|-------------|---------|------------------|
| 0 (baseline) | 99.7 | 2,721.1 | 1.0x |
| 1 | 156.2 | 6,935.0 | 2.55x |
| 10 | 234.8 | 8,420.3 | 3.09x |
| 50 | 445.3 | 9,156.7 | 3.36x |

Several observations emerge:
1. Performance decreases even on in-distribution data
2. Degradation increases with more adaptation steps
3. The relative degradation is more pronounced on OOD data

## 4.3 Analysis of Adaptation Dynamics

To understand why TTA fails, we tracked the adaptation loss (prediction consistency) and task loss (MSE) during adaptation. Figure 1 (not shown) would reveal that while the consistency loss decreases as intended, the actual prediction error increases.

We examined the predictions after adaptation and found that models converge toward nearly constant outputs. After 50 adaptation steps, the standard deviation of predictions across different inputs decreased by 94%, while the mean prediction remained relatively stable. This suggests that the model achieves high consistency by making similar predictions regardless of input, at the cost of accuracy.

## 4.4 Meta-Learning Results

MAML aims to learn parameters that can quickly adapt to new tasks. Table 3 shows results with and without test-time adaptation.

**Table 3: MAML Performance**
| Configuration | In-Dist MSE | OOD MSE | OOD vs. Baseline |
|--------------|-------------|---------|------------------|
| ERM baseline | 99.7 | 2,721.1 | 1.0x |
| MAML (no adaptation) | 112.3 | 3,018.8 | 1.11x |
| MAML (10-shot adaptation) | 89,234.2 | 1,697,689.5 | 623.8x |

The results show:
1. MAML without adaptation performs similarly to ERM, with slightly worse OOD performance
2. With adaptation, MAML shows substantial performance degradation, increasing error by over 600-fold
3. The degradation exceeds that of TTA, possibly due to MAML's design for rapid adaptation

The magnitude of the error (1.7 million MSE) indicates the model's predictions diverge significantly from the true dynamics.

## 4.5 Ensemble Methods Results

Table 4 presents results for ensemble approaches.

**Table 4: Ensemble Methods Performance**
| Method | In-Dist MSE | OOD MSE | OOD vs. Baseline |
|--------|-------------|---------|------------------|
| ERM baseline | 99.7 | 2,721.1 | 1.0x |
| Deep Ensemble (5 models) | 87.4 | 2,698.3 | 0.99x |
| GFlowNet-inspired | 95.2 | 2,670.9 | 0.98x |

Ensemble methods show:
1. Modest improvements on in-distribution data through variance reduction
2. Minimal improvement on OOD data (1-2% reduction in MSE)
3. Stable performance without significant degradation

## 4.6 Summary of Results

Figure 2 summarizes all methods' performance on time-varying gravity:

```
Method                        Relative MSE vs. Baseline
ERM (baseline)               1.00x
Deep Ensemble                0.99x
GFlowNet-inspired            0.98x
MAML (no adapt)              1.11x
TTA (1 step)                 2.55x
TTA (10 steps)               3.09x
TTA (50 steps)               3.36x
MAML (10-shot adapt)         623.8x
```

Key findings:
1. No method significantly improves upon the baseline
2. Adaptation-based methods (TTA, MAML with adaptation) reduce performance
3. Performance degradation increases with adaptation intensity
4. Simple ensembling provides the most stable results, though improvements are minimal

## 4.7 Additional Analyses

We conducted several additional analyses to ensure robustness:

**Hyperparameter Sensitivity**: We varied learning rates, adaptation steps, and loss weights for TTA. Performance degradation occurred across all settings, with faster adaptation causing more severe degradation.

**Architecture Variations**: We tested different model capacities (64-1024 hidden units) and depths (2-5 layers). The pattern of results remained consistent.

**Different Physics Modifications**: We tested constant gravity with different values (g = 15 m/s²) and found better generalization (2-3x degradation), suggesting that time-varying dynamics present a qualitatively different challenge than parameter shifts.

These results indicate that current OOD methods, despite their success on standard benchmarks, may not address the type of extrapolation required for modified physical dynamics.
