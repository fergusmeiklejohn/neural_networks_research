# 5. Analysis: Why Adaptation Methods Fail

## 5.1 The Objective Mismatch

A key issue in the failure of adaptation methods on mechanism shifts is a mismatch between their optimization objectives and the task objective. Test-time adaptation methods optimize self-supervised losses such as:

- **Prediction consistency**: Encouraging similar predictions for similar inputs
- **Temporal smoothness**: Minimizing variation in sequential predictions
- **Entropy minimization**: Reducing uncertainty in predictions
- **Physics-informed losses**: Enforcing energy conservation or Hamiltonian structure

These objectives are motivated by sound principles. Consistency and smoothness help when test distributions involve corruptions that increase prediction variance. Physics-informed losses excel when conservation laws hold. However, when the underlying mechanisms change—violating these assumptions—these objectives lead to systematic performance degradation.

## 5.2 Gradient Analysis

To understand the optimization dynamics, we analyzed the alignment between self-supervised and true objectives. Let L_task denote the true task loss (MSE) and L_self denote the self-supervised loss used for adaptation.

**Table: Gradient Alignment Analysis**
| System | Loss Type | In-Distribution | Mechanism Shift |
|--------|-----------|-----------------|-----------------|
| Two-ball | Prediction consistency | 0.73 ± 0.08 | -0.41 ± 0.12 |
| Pendulum | Prediction consistency | 0.68 ± 0.09 | -0.38 ± 0.11 |
| Pendulum | Energy conservation | 0.81 ± 0.06 | -0.52 ± 0.13 |
| Pendulum | Hamiltonian consistency | 0.79 ± 0.07 | -0.47 ± 0.14 |

The negative alignment on mechanism shifts indicates that optimizing self-supervised objectives moves parameters away from improving accuracy. Physics-informed losses show even stronger negative alignment because they enforce conservation laws that mechanism shifts violate.

## 5.3 The Collapse vs. Wrong Structure Distinction

Our PeTTA-inspired experiments reveal two distinct failure modes:

### Mode 1: Collapse to Degeneracy
Models converge to trivial solutions (e.g., constant predictions) that minimize self-supervised losses while ignoring input variations. This is what PeTTA successfully prevents through:
- Monitoring prediction entropy (diversity)
- Tracking parameter drift
- Intervening when collapse indicators trigger

### Mode 2: Wrong Computational Structure
Models maintain diverse predictions but lack the architectural components to represent new physics. In our experiments:
- Pendulum predictions remained varied (entropy decreased only 2%)
- No collapse was detected (0/20 adaptation steps)
- Yet performance degraded by 13.89x

This demonstrates that preventing collapse—while valuable—cannot address missing computational operations like the L̇/L term in variable-length pendulum dynamics.

## 5.4 Relationship to Contemporary TTA Improvements

Our findings might seem to contradict recent TTA successes. However, these approaches address different aspects of the adaptation challenge:

### PeTTA's Collapse Detection
PeTTA successfully prevents parameter drift and maintains stable predictions by detecting when adaptation leads to degenerate solutions. In our experiments, PeTTA-inspired monitoring correctly identified that no collapse occurred—predictions remained diverse. However, detection alone cannot guide adaptation toward learning new physics terms. The model needs architectural capacity to express -2(L̇/L)θ̇, not just stable parameters.

### TAIP's Physics Constraints
TAIP elegantly uses energy conservation and Hamiltonian structure to constrain adaptation in molecular dynamics. This works brilliantly when underlying physics principles remain fixed—only atomic positions and velocities change. In our mechanism shifts, however, these constraints become actively harmful. For the time-varying pendulum, enforcing energy conservation (which no longer holds due to work done by length changes) prevents the model from learning correct dynamics. Our experiments confirm this: energy-based TTA degraded performance by 12.6x.

### TTAB's Comprehensive Analysis
The TTAB benchmark identifies that no single TTA method handles all distribution shifts. Our mechanism shifts represent an extreme case that aligns with their findings. The key insight: different types of shift require different solutions.

This analysis suggests three distinct adaptation scenarios:

1. **Parameter Adaptation** (TAIP's domain): The computational structure remains valid; only numerical parameters need adjustment. Physics-aware constraints guide successful adaptation.

2. **Stability Preservation** (PeTTA's domain): The model risks collapse due to accumulated errors or confirmation bias. Monitoring and intervention maintain reasonable performance within the model's capabilities.

3. **Mechanism Learning** (Our focus): New computational operations are required. No amount of parameter adjustment or stability preservation can introduce missing physics terms.

## 5.5 Why Physics-Aware Losses Fail on Mechanism Shifts

Our experiments with energy and Hamiltonian consistency losses provide crucial insights:

**Energy Conservation Loss Performance:**
- Fixed pendulum: Helps maintain physical consistency
- Variable pendulum: Degrades performance by 12.6x

The failure occurs because:
1. Energy conservation assumes closed systems
2. Variable-length pendulum is non-conservative (work done by length changes)
3. Enforcing false conservation misleads adaptation

This exemplifies a broader principle: domain knowledge helps only when its assumptions hold. Mechanism shifts deliberately violate these assumptions.

## 5.6 Information-Theoretic Perspective

Improving predictions on mechanism shifts requires information not present in unlabeled test data:

- Training provides: P(y|x, fixed_mechanism)
- Test data provides: P(x|new_mechanism)
- Needed for accuracy: P(y|x, new_mechanism)

The missing link—how the new mechanism transforms inputs to outputs—cannot be inferred from unlabeled data alone. Self-supervised losses provide no information about this transformation.

## 5.7 Implications for Future Development

Our analysis, combined with recent advances, suggests several directions:

### Modular Architectures
Models need capacity to express new computational operations. Rather than adapting fixed architectures, we might need:
- Dormant pathways that can be activated
- Neural module networks that can be reconfigured
- Mixture of experts with diverse computational primitives

### Program Synthesis at Test Time
For true mechanism learning, models might need to:
- Discover new functional forms from data
- Compose existing operations in novel ways
- Learn symbolic rules that generalize

### Hybrid Approaches
Combine strengths of current methods:
- Use PeTTA-style monitoring to maintain stability
- Apply TAIP-style constraints where valid
- Detect when new mechanisms are needed
- Switch to structure learning when parameter adaptation fails

## 5.8 Summary

Current TTA improvements operate successfully within their intended domains. PeTTA prevents collapse, TAIP leverages valid physics knowledge, and comprehensive benchmarks like TTAB map the landscape of challenges. Our work identifies mechanism shifts as a frontier where current methods reach their limits—not due to implementation issues but fundamental assumptions about what adaptation can achieve within fixed computational frameworks. This delineation helps focus future research on the distinct challenge of mechanism learning.
