# Honest Literature Positioning Strategy

## What We Have Actually Done

### Confirmed Experiments:
1. **Standard TTA (TENT-style)**: Prediction consistency loss - degrades by 235% on 2-ball, 14.4x on pendulum
2. **Energy-based TTA**: Physics-inspired energy conservation loss - degrades by 12.6x on pendulum
3. **Hamiltonian TTA**: Physics-inspired Hamiltonian consistency - degrades by 17.9x on pendulum
4. **MAML baseline**: With and without adaptation - catastrophic failure with adaptation

### Key Finding:
Even physics-aware self-supervised losses fail when the physics mechanism changes (conservation laws break).

## What We Know About Recent Work

### PeTTA (NeurIPS 2024)
- **Core idea**: Detects when model is collapsing and adjusts adaptation
- **What we can say**: "PeTTA introduces collapse detection to prevent degenerate solutions during adaptation. While this addresses stability concerns, our mechanism shifts present a different challenge: the model needs new computational operations (e.g., the L̇/L term in variable pendulum), not just stable adaptation of existing parameters."
- **Honest acknowledgment**: "Future work should investigate whether PeTTA's collapse detection could identify when adaptation is moving in fundamentally wrong directions, as occurs with mechanism shifts."

### TAIP (Nature Communications 2025)
- **Core idea**: Uses self-supervised learning for molecular dynamics
- **What we can say**: "TAIP demonstrates success using physics-informed losses for molecular systems. Our energy and Hamiltonian consistency losses are conceptually similar. However, TAIP assumes fixed physical laws—our mechanism shifts violate this assumption as conservation laws themselves change."
- **Key distinction**: We actually tested physics-aware losses and showed they fail when physics changes

### TTAB (ICML 2023)
- **Core idea**: Comprehensive benchmark identifying TTA failure modes
- **What we can say**: "TTAB provides a valuable taxonomy of distribution shifts. Our mechanism shifts—where the functional form of the data-generating process changes—represent an extreme case that may extend beyond their current categories."
- **Positioning**: Our work identifies a specific, understudied failure mode

## Recommended Paper Revisions

### In Introduction:
"Recent advances in test-time adaptation have shown promise. PeTTA [1] prevents adaptation collapse through monitoring, while TAIP [2] succeeds in molecular dynamics using physics-informed losses. However, these methods assume the underlying computational structure remains valid. We investigate mechanism shifts—changes in the generative equations themselves—where we find even physics-aware adaptation fails."

### In Related Work:
"Our energy and Hamiltonian consistency losses share conceptual similarities with TAIP's physics-informed approach [2]. However, while TAIP succeeds when physical laws remain fixed (only parameters change), we show these same types of losses fail when the physics mechanism itself changes, as conservation assumptions no longer hold."

### In Discussion:
"Our findings complement recent TTA advances rather than contradicting them. PeTTA's collapse detection [1] successfully maintains stability within the model's computational framework. Our mechanism shifts require expanding this framework—adding new terms like L̇/L that don't exist in the original model. This suggests different solution strategies: modular architectures that can activate dormant pathways, or program synthesis to introduce new operations at test time."

### In Limitations:
"We have not implemented PeTTA's specific collapse detection algorithm, though our gradient alignment analysis provides complementary diagnostics. Future work should test whether collapse detection could identify when adaptation objectives fundamentally misalign with true objectives, as we observe in mechanism shifts."

## Key Messages

1. **We tested physics-aware losses**: Unlike speculation, we actually implemented and tested them
2. **Clear distinction**: Parameter adaptation (TAIP succeeds) vs mechanism adaptation (our focus)
3. **Complementary findings**: Not contradicting recent work, but identifying boundaries
4. **Honest about gaps**: Acknowledge what we didn't test while emphasizing what we did

## Action Items

1. Revise paper with this honest positioning
2. Emphasize our concrete experiments (energy/Hamiltonian TTA)
3. Draw clear conceptual distinctions
4. Acknowledge specific methods we didn't implement
5. Focus on our unique contribution: identifying and analyzing mechanism shifts

This approach maintains scientific integrity while properly positioning our work.
