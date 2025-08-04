# Summary of Reviewer-Suggested Papers

## 1. "Understanding and Mitigating Extrapolation Failures in Physics-Informed Neural Networks" (Fesser et al., 2023)
**URL**: https://arxiv.org/abs/2306.09478

### Key Insights:
- **Main Finding**: Extrapolation failure is NOT caused by high frequencies, but by "spectral shifts" - shifts in the support of the Fourier spectrum over time
- **Novel Metric**: Weighted Wasserstein-Fourier (WWF) distance to quantify spectral shifts
- **Mitigation**: Transfer learning strategy reduces extrapolation errors by up to 82%
- **Important Quote**: "In the absence of significant spectral shifts, PINN predictions stay close to the true solution even in extrapolation"

### Relevance to Our Work:
- **Strongly supports** our time-varying gravity benchmark - it creates spectral shifts that violate PINN assumptions
- Provides theoretical grounding for why structural changes cause universal failure
- Should cite when discussing why time-varying physics is fundamentally different

## 2. "Robust extrapolation using physics-related activation functions in neural networks" (Kim et al., 2025)
**URL**: https://arxiv.org/html/2505.15363v1

### Key Insights:
- **Main Finding**: Replacing standard activation functions with physics-related functions significantly improves extrapolation
- **Approach**: Used logarithmic, sinusoidal, and physics-inspired combinations
- **Success**: Achieved accurate predictions for nuclear masses in extrapolation regions
- **Key Point**: Physics knowledge helps when incorporated flexibly, not as rigid constraints

### Relevance to Our Work:
- **Nuances our finding** that physics constraints hurt - it's about HOW physics is incorporated
- Flexible physics-inspired design vs rigid constraints (like F=ma in PINNs)
- Should mention when discussing the "paradox of domain knowledge"

## 3. "An extrapolation-driven network architecture for physics-informed deep learning" (Wang et al., 2024)
**URL**: https://arxiv.org/html/2406.12460v3

### Key Insights:
- **Main Finding**: PINN extrapolation capability depends heavily on the nature of the governing equation
- **Success Cases**: Smooth, slowly changing solutions (e.g., Allen-Cahn equation)
- **Failure Cases**: Rapidly changing solutions with large |ut|
- **Solution**: E-DNN architecture with control functions to manage parameter changes

### Relevance to Our Work:
- Supports our observation that different physics scenarios have different extrapolation characteristics
- Aligns with our finding that constant vs time-varying physics represents a fundamental divide
- Should cite when discussing why some physics is "extrapolatable" while others aren't

## Integration Strategy for Our Paper:

### 1. In Related Work Section:
Add paragraph on recent advances in understanding PINN extrapolation, citing all three papers.

### 2. In Discussion Section 5.2.1 (Paradox of Domain Knowledge):
- Reference Kim et al. (2025) to nuance our statement about physics hurting performance
- Clarify: rigid constraints (F=ma) vs flexible physics-inspired design

### 3. In Discussion Section 5.5 (Connection to Recent Advances):
- Add Fesser et al. (2023) spectral shift framework as theoretical grounding
- Mention Wang et al. (2024) PDE-dependent extrapolation

### 4. When Discussing Time-Varying Gravity:
- Use spectral shift framework to explain why this creates true OOD
- Reference that this aligns with recent theoretical understanding

### 5. Soften Claims:
- Instead of "physics constraints hurt", say "rigid physics constraints can hinder adaptation, though recent work shows flexible physics-inspired designs may help (Kim et al., 2025)"
- Instead of "universal failure", say "systematic failure consistent with spectral shift theory (Fesser et al., 2023)"
