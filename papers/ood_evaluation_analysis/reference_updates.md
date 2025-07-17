# Reference Updates and Integration

## New References to Add

### 1. Primary Additions (Reviewer Suggested)

**Fesser, L., D'Amico-Wong, L., & Qiu, R. (2023).** "Understanding and mitigating extrapolation failures in physics-informed neural networks." *arXiv preprint arXiv:2306.09478*.

**Kim, C. H., Chae, K. Y., & Smith, M. S. (2025).** "Robust extrapolation using physics-related activation functions in neural networks for nuclear masses." *arXiv preprint arXiv:2505.15363*.

**Wang, Y., Yao, Y., & Gao, Z. (2024).** "An extrapolation-driven network architecture for physics-informed deep learning." *arXiv preprint arXiv:2406.12460*.

### 2. Update Existing Reference

**GraphExtrap Reference**:
- Check if "Anonymous (2023)" has been published
- If still under review, change to: "GraphExtrap: Leveraging graph neural networks for extrapolation in physics. *Manuscript under review* (2023)."

## Integration Points in the Paper

### Section 2.1 Physics-Informed Neural Networks
Add after discussing Krishnapriyan et al. (2021):

"More recently, Fesser et al. (2023) demonstrated that the failure to extrapolate is not primarily caused by high frequencies in the solution function, but rather by shifts in the support of the Fourier spectrum over time. They introduced the Weighted Wasserstein-Fourier distance (WWF) as a metric for predicting extrapolation performance. Wang et al. (2024) further showed that PINN extrapolation capability depends heavily on the nature of the governing equation itself, with smooth, slowly changing solutions being more amenable to extrapolation."

### Section 2.5 Causal Learning and Distribution Shift
Add new paragraph:

"Recent work has also explored alternative approaches to improving extrapolation. Kim et al. (2025) demonstrated that replacing standard activation functions with physics-related functions can significantly improve extrapolation performance in scientific domains. This suggests that the manner in which physics knowledge is incorporated—whether as rigid constraints or flexible design elements—may be as important as the knowledge itself."

### Section 5.2.1 The Paradox of Domain Knowledge
Revise paragraph to include:

"This paradox can be understood through the lens of recent PINN research. Fesser et al. (2023) identified spectral shifts as a primary cause of extrapolation failure, providing theoretical grounding for why time-varying physics causes systematic performance degradation. However, the relationship between physics knowledge and model performance is nuanced. Kim et al. (2025) showed that physics-related activation functions can improve extrapolation, suggesting that flexible incorporation of domain knowledge may succeed where rigid constraints fail."

### Section 5.5 Connection to Recent Advances
Expand to include:

"**Spectral Analysis**: The spectral shift framework for understanding PINN failures (Fesser et al., 2023) provides theoretical grounding for why time-varying physics causes systematic failure in our experiments. Their WWF metric could be applied to predict which physics scenarios will be challenging for current methods.

**Flexible Physics Integration**: The success of physics-related activation functions (Kim et al., 2025) suggests a path forward that balances domain knowledge with adaptability. This approach contrasts with the rigid F=ma constraints in traditional PINNs.

**PDE-Dependent Extrapolation**: Wang et al. (2024) showed that extrapolation capability varies with the governing equation's properties. Our time-varying gravity represents a challenging case with rapid temporal changes that exceed current architectural capabilities."

### Section 4.3.2 Universal Model Failure
Add context:

"All tested models showed substantial performance degradation when faced with structural changes in the physics dynamics. This aligns with theoretical predictions from the spectral shift framework (Fesser et al., 2023), which suggests that time-varying parameters create frequency content outside the training distribution's support."

### Update Citation Style

Ensure all new references follow the journal's citation style. If using author-year format:
- In text: (Fesser et al., 2023)
- In reference list: alphabetical by author surname

## Reference List Updates

Add in alphabetical order:

```
Fesser, L., D'Amico-Wong, L., & Qiu, R. (2023). Understanding and mitigating extrapolation failures in physics-informed neural networks. arXiv preprint arXiv:2306.09478.

Kim, C. H., Chae, K. Y., & Smith, M. S. (2025). Robust extrapolation using physics-related activation functions in neural networks for nuclear masses. arXiv preprint arXiv:2505.15363.

Wang, Y., Yao, Y., & Gao, Z. (2024). An extrapolation-driven network architecture for physics-informed deep learning. arXiv preprint arXiv:2406.12460.
```

Remove or update:
- "Anonymous (2023)" if GraphExtrap has been published