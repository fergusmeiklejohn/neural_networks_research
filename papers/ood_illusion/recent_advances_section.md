# Recent Advances in Stabilized TTA (For Background Section)

## 2.3 Recent Advances in Stabilized Test-Time Adaptation

While early TTA methods showed vulnerability to distribution shift, recent work has made significant progress in addressing stability and performance issues:

### Persistent Test-Time Adaptation (PeTTA)
Bohdal et al. (2024) introduce collapse detection mechanisms to maintain model stability over extended adaptation periods. By monitoring prediction diversity and parameter drift, PeTTA prevents the degenerate solutions we observe in standard TTA. Their method shows improvements on both Level 2 (correlation shift) and Level 3 (diversity shift) distribution changes in vision benchmarks. However, PeTTA assumes the underlying computational structure remains valid—an assumption violated in mechanism shifts where new physics terms emerge.

### Physics-Aware Test-Time Adaptation (TAIP)
Fu et al. (2025) leverage domain knowledge through physics-informed consistency losses for molecular dynamics. By enforcing energy conservation and Hamiltonian structure during adaptation, TAIP successfully generalizes to new chemical environments with different atomic configurations. Crucially, TAIP's success relies on fixed physical laws with varying parameters—precisely the opposite of our mechanism shift scenario where conservation laws themselves change.

### Comprehensive Evaluation (TTAB)
Zhao et al. (2023) provide a systematic benchmark revealing that TTA success depends critically on the type of distribution shift encountered. Their taxonomy identifies scenarios where TTA helps (corruptions, style changes) versus where it struggles (label shift, temporal drift). Our mechanism shifts represent an extreme case: the computational operations required for accurate prediction fundamentally change.

### Positioning Our Contribution
These advances strengthen TTA for many real-world scenarios. However, our experiments reveal that when the data-generating mechanism itself changes—requiring different computational operations—current stabilization techniques may not suffice. This distinction helps delineate the boundaries of current approaches:

- **Parameter Adaptation** (TAIP succeeds): Same physics equations, different constants
- **Stable Adaptation** (PeTTA succeeds): Prevent collapse while maintaining performance
- **Mechanism Adaptation** (Open problem): New computational requirements emerge

Our work thus complements these advances by identifying mechanism shifts as a persistent challenge requiring fundamentally different solutions.
