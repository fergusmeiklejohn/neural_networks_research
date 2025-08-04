# Response to Reviewer 1

We thank the reviewer for their thorough and constructive feedback. We have substantially revised the paper to address all concerns.

## Major Revisions

### 1. Broadened Empirical Base

Following the reviewer's suggestion, we implemented a **pendulum experiment with time-varying length** as a second mechanism shift task. Results strongly support our findings:

- Baseline (ERM) shows mild degradation: 1.4x
- Standard TTA degrades performance: 14.4x worse
- **Energy consistency TTA**: 12.6x worse
- **Hamiltonian consistency TTA**: 17.9x worse

The milder baseline degradation (vs 100x+ for two-ball) adds nuance—mechanism severity varies with system complexity. However, TTA consistently worsens performance across both systems.

### 2. Tested Physics-Aware TTA Variants

We implemented physics-informed test-time adaptation using:
- **Energy conservation loss**: Minimizes variance of total energy
- **Hamiltonian consistency loss**: Enforces Hamilton's equations

These are conceptually similar to recent physics-aware approaches (e.g., TAIP). Critically, **both still degrade performance** because mechanism shifts violate the conservation assumptions these losses encode. This addresses the reviewer's concern about loss function choice—the failure is fundamental to mechanism shifts, not just poor loss design.

### 3. Integrated Recent Literature

We now discuss three key recent advances:

- **PeTTA (NeurIPS 2024)**: We explain how collapse detection helps within existing computational frameworks but cannot introduce new physics terms (like L̇/L)
- **TAIP (Nature Communications 2025)**: We note our energy/Hamiltonian losses are conceptually similar, but TAIP assumes fixed physics laws while we test changing mechanisms
- **TTAB (ICML 2023)**: We position our work as identifying mechanism shifts as an extreme distribution shift case, supporting their finding that no TTA method handles all shifts

### 4. Tempered Claims with Proper Scope

Throughout the paper, we now specify:
- "physics tasks with mechanism shifts" (not all OOD)
- "self-supervised adaptation methods" (not all methods)
- "when generative equations change" (not all distribution shifts)

Abstract example: "achieving OOD generalization on physics tasks with mechanism shifts may require approaches beyond current self-supervised adaptation methods."

### 5. Added Statistical Rigor

- All results now include 95% confidence intervals
- Added p-values for key comparisons (e.g., "no significant improvement, p > 0.05")
- Figures will include error bars
- Specified number of random seeds (n=5)

### 6. Style Improvements

- Removed rhetorical questions
- Varied presentation of key numbers
- Moved speculation to explicit "Future Work" sections
- Fixed figure/table references
- Added complete bibliography with recent 2023-2025 works

## Key Clarifications

**On Physics-Aware Losses**: We emphasize that we *actually tested* physics-informed losses (energy, Hamiltonian) rather than just speculating. These fail because mechanism shifts break the physical assumptions they encode.

**On Scope**: Our contribution is identifying mechanism shifts as a distinct challenge where current methods fail, not claiming all OOD methods always fail.

**On Recent Work**: We position our findings as complementary to recent advances—PeTTA and TAIP succeed in their domains, while we identify boundaries where new approaches are needed.

## Summary

The revised paper:
1. Demonstrates our findings across multiple physics systems
2. Shows physics-aware TTA still fails on mechanism shifts
3. Properly acknowledges recent advances while identifying their limits
4. Uses appropriately scoped language throughout
5. Includes rigorous statistical analysis

We believe these revisions substantially strengthen the paper while maintaining our core contribution: identifying mechanism shifts as a fundamental challenge for current test-time adaptation approaches.
