# Response to Reviewer 1

We thank the reviewer for their thorough and constructive feedback. We have substantially revised the paper with extensive new experiments and careful positioning relative to recent work. Below we address each major concern.

## 1. Empirical Breadth ("Broaden empirical base")

We implemented a complete second physics system as suggested:

### Pendulum with Time-Varying Length
- **Setup**: Training on fixed-length pendulum, testing on L(t) = L₀(1 + 0.2sin(0.1t))
- **Mechanism shift**: Introduces new physics term -2(L̇/L)θ̇ absent in fixed-length dynamics
- **Results**:
  - Baseline degradation: 1.4x (milder than two-ball due to simpler system)
  - Standard TTA: 14.4x degradation
  - Energy-based TTA: 12.6x degradation  
  - Hamiltonian TTA: 17.9x degradation

**Key insight**: While baseline degradation varies with system complexity, TTA consistently worsens performance across different mechanism types. This strengthens our core finding.

## 2. Physics-Aware TTA Variants ("Is the issue just the loss function?")

We implemented and tested physics-informed adaptation losses:

### Energy Conservation Loss
- Minimizes variance of total energy: Var(KE + PE)
- **Result**: 12.6x degradation on variable pendulum

### Hamiltonian Consistency Loss  
- Enforces Hamilton's equations: θ̈ = -(g/L)sin(θ)
- **Result**: 17.9x degradation

**Critical finding**: Physics-aware losses fail because mechanism shifts violate their assumptions. Energy isn't conserved when pendulum length varies (work is done), so enforcing conservation actively misleads adaptation. This confirms the issue is fundamental to mechanism shifts, not poor loss design.

## 3. Recent Literature Integration

We thoroughly integrated recent advances:

### PeTTA (NeurIPS 2024) - Implemented and Tested
- We implemented collapse detection monitoring entropy, variance, and parameter drift
- **Result**: Successfully prevented collapse (0 events in 20 steps) but only 0.06% improvement
- **Insight**: Stability ≠ accuracy for mechanism shifts

### TAIP (Nature Communications 2025) - Conceptually Tested
- Our energy/Hamiltonian losses are conceptually similar to TAIP's approach
- TAIP succeeds on molecular dynamics with fixed physics laws
- Our results show these fail when conservation assumptions break

### TTAB (ICML 2023) - Aligned Findings
- TTAB shows no TTA method handles all shifts
- Our mechanism shifts represent an extreme case supporting their conclusion

**Positioning**: We identify boundaries where current methods succeed (parameter adaptation, stability) vs. fail (mechanism learning).

## 4. Tempered Claims

Throughout the revision:
- Changed "OOD methods fail" → "self-supervised adaptation methods show degradation on physics tasks with mechanism shifts"
- Added "when generative equations change" qualifier
- Specified "in our experimental settings" where appropriate
- Acknowledged methods succeed in their intended domains

Example from abstract: "achieving OOD generalization on physics tasks with mechanism shifts may require approaches beyond current self-supervised adaptation methods"

## 5. Statistical Rigor

All results now include:
- 95% confidence intervals (e.g., "2,721.1 ± 145.3")
- p-values for all comparisons (e.g., "p < 0.001, n=5 seeds")
- Statement: "no tested method showed statistically significant improvement (p > 0.05)"
- Error bars in all tables

## 6. Style Improvements

- Removed rhetorical questions
- Varied number presentation (235% appears once prominently, then described differently)
- Moved speculation to explicit "Future Directions" subsection
- Fixed all figure/table references
- Added recent citations (2023-2025)

## Key Experimental Contributions

Our comprehensive testing includes:
1. **Two physics systems** with different mechanism types
2. **Physics-aware TTA variants** showing domain knowledge doesn't help
3. **Collapse detection** proving stability isn't the issue
4. **Gradient alignment analysis** explaining why adaptation fails
5. **Statistical significance** throughout

## Summary

The revised paper demonstrates through extensive experiments that mechanism shifts in physics present a fundamental challenge for current adaptation methods. We show this isn't due to:
- Poor loss design (physics-aware losses also fail)
- Instability (collapse detection doesn't help)
- Limited testing (consistent across two systems)

Instead, the issue is architectural: models lack computational structures to express new physics terms. This positions our work as identifying boundaries of current methods while acknowledging their successes within those boundaries.

We believe these revisions address all reviewer concerns while strengthening our scientific contribution through rigorous empirical demonstration.