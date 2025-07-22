# Introduction (Revised with Empirical Evidence)

Machine learning models for physics prediction face a particular challenge: when the underlying physical mechanisms change over time, performance can degrade catastrophically. While recent advances in out-of-distribution (OOD) generalization—including test-time adaptation methods like PeTTA (Bohdal et al., 2024) and physics-informed approaches like TAIP (Fu et al., 2025)—show promise for many distribution shifts, we investigate a specific class that poses unique challenges: mechanism shifts in physics, where the generative equations themselves change.

We evaluated state-of-the-art adaptation methods on two physics prediction tasks with mechanism shifts. In a two-ball system with time-varying gravity, test-time adaptation (TTA) increased prediction error by 235%, while MAML with adaptation showed a 62,290% increase. To test generality, we implemented pendulum experiments with time-varying length and found that even physics-aware TTA variants—using energy conservation and Hamiltonian consistency losses—degraded performance by 12-18x. We also implemented collapse detection inspired by PeTTA, which successfully prevented degenerate solutions but provided negligible improvement (0.06%) in accuracy.

These results reveal a fundamental challenge: when test distributions require new computational operations absent from the training regime (such as the L̇/L term that emerges with time-varying pendulum length), current self-supervised adaptation methods cannot bridge this gap. This finding complements recent advances rather than contradicting them—methods like PeTTA excel at maintaining stability within the model's computational framework, while TAIP succeeds when physical laws remain fixed. Our work identifies the boundaries where new approaches are needed.

## The Challenge of Mechanism Shifts

Consider two types of distribution shift in physics:
1. **Parameter shifts**: Same equations, different constants (e.g., planets with different gravity)
2. **Mechanism shifts**: New terms in equations (e.g., time-varying gravity g(t))

Current benchmarks primarily evaluate the first type, where successful generalization requires robust parameter estimation. The second type—our focus—requires learning new functional relationships. For instance, a pendulum with time-varying length L(t) introduces a velocity-dependent term -2(L̇/L)θ̇ that doesn't exist in fixed-length dynamics. No parameter adjustment within a fixed-length model can capture this new physics.

## Our Empirical Investigation

We conducted comprehensive experiments to understand how current methods handle mechanism shifts:

**Two-ball dynamics**: Training on constant gravity (g = 9.8 m/s²), testing on g(t) = 9.8 + 2sin(0.1t)
- Standard ERM: baseline performance
- TTA methods: 235% worse than baseline
- MAML with adaptation: 62,290% worse (catastrophic failure)

**Pendulum dynamics**: Training on fixed length, testing on L(t) = L₀(1 + 0.2sin(0.1t))
- Standard ERM: 1.4x degradation (mild due to simpler system)
- Prediction consistency TTA: 14.4x degradation
- Energy conservation TTA: 12.6x degradation
- Hamiltonian consistency TTA: 17.9x degradation
- PeTTA-inspired with collapse detection: 13.9x degradation

The pendulum results are particularly revealing: physics-aware losses that should help (energy conservation) actually harm performance because they encode assumptions (energy is conserved) that mechanism shifts violate (energy changes due to work done by length variation).

## Key Insights

Our analysis reveals why adaptation fails on mechanism shifts:

1. **Gradient misalignment**: The gradient of self-supervised objectives (prediction consistency, energy conservation) becomes negatively aligned with the gradient of true prediction error, causing adaptation to move away from accurate solutions.

2. **Missing computational structure**: Models lack the architectural components to represent new physics terms. Collapse detection (à la PeTTA) maintains stable but systematically wrong predictions.

3. **Conservation assumption violation**: Physics-informed losses assume fixed conservation laws, but mechanism shifts deliberately break these assumptions.

## Contributions

This paper makes the following contributions:

1. **Empirical demonstration across multiple systems**: We show TTA degradation on two different mechanism shifts (gravity variation, pendulum length variation), with comprehensive testing including physics-aware variants and collapse detection.

2. **Mechanistic understanding**: We identify gradient misalignment as the cause of adaptation failure and show why physics-aware losses don't help when physics changes.

3. **Taxonomy of distribution shifts**: We distinguish surface variations, statistical shifts, and mechanism changes, showing current methods succeed on the former but fail on the latter.

4. **Positioning relative to recent advances**: We clarify that methods like PeTTA and TAIP succeed in their intended domains (stability preservation, parameter adaptation) while mechanism shifts require fundamentally different solutions.

## Paper Organization

Section 2 reviews OOD methods and positions our work relative to recent advances. Section 3 describes our experimental setup for both physics systems. Section 4 presents comprehensive results including physics-aware TTA and collapse detection. Section 5 analyzes gradient alignment and why adaptation fails. Section 6 proposes a taxonomy of distribution shifts. Section 7 discusses implications for future method development. Section 8 concludes.

Our findings suggest that achieving OOD generalization on physics tasks with mechanism shifts may require approaches beyond current self-supervised adaptation methods—potentially involving modular architectures that can express new computational operations or program synthesis methods that can introduce new terms at test time. This work aims to delineate the boundaries of current approaches and inspire development of methods for this challenging but important class of distribution shift.