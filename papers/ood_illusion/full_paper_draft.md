# The OOD Illusion in Physics Learning: When Generalization Methods Make Things Worse

## Abstract

Recent advances in out-of-distribution (OOD) generalization have shown promising results on benchmarks involving style changes, corruptions, and domain shifts. However, when we evaluated these methods on physics prediction tasks with mechanism shifts—where the underlying dynamical equations change—we observed significant performance degradation: test-time adaptation (TTA) increased prediction error by 235% on time-varying gravity tasks, while Model-Agnostic Meta-Learning (MAML) with adaptation showed a 62,290% increase. To test generality, we implemented pendulum experiments with time-varying length and found that even physics-aware TTA variants (using energy and Hamiltonian consistency) degraded performance by 12-18x. We propose a taxonomy distinguishing surface variations, statistical shifts, and mechanism changes, showing that current self-supervised adaptation methods succeed on the former but show systematic limitations on the latter. Our analysis reveals that when test distributions involve different generative processes, gradient alignment between self-supervised and true objectives becomes negative, causing adaptation to move away from correct solutions. These findings suggest that achieving OOD generalization on physics tasks with mechanism shifts may require approaches beyond current self-supervised adaptation methods.

[Insert remaining sections here in order:
1. Introduction (from introduction_draft.md)
2. Background and Related Work (from background_draft.md)
3. The Physics Extrapolation Challenge (from methods_draft.md)
4. Empirical Results (from results_draft.md)
5. Analysis: Why Adaptation Methods Fail (from analysis_draft.md)
6. A Taxonomy of Distribution Shifts (from taxonomy_draft.md)
7. Implications and Future Directions (from future_directions_draft.md)
8. Conclusion (from conclusion_draft.md)]

## References

### Recent Advances in Test-Time Adaptation
1. **PeTTA**: Persistent Test-Time Adaptation in Dynamic Environments. *NeurIPS 2024*. [Shows Level 2 and 3 improvements with collapse detection]
2. **TAIP**: Test-time Augmentation for Inter-atomic Potentials. *Nature Communications 2025*. [Physics-aware consistency for molecular dynamics]
3. **TTAB**: A Comprehensive Test-Time Adaptation Benchmark. *ICML 2023*. [Identifies multiple TTA failure modes]

### Classic Test-Time Adaptation Methods
4. **TENT**: Test-Time Entropy Minimization. *ICLR 2021*
5. **MEMO**: Test-Time Model Adaptation with Marginal Entropy Minimization. *NeurIPS 2022*
6. **CoTTA**: Continual Test-Time Adaptation. *CVPR 2022*

### Meta-Learning Approaches
7. **MAML**: Model-Agnostic Meta-Learning. *ICML 2017*
8. **Reptile**: A Scalable Meta-Learning Algorithm. *2018*
9. **Meta-SGD**: Learning to Learn Quickly. *NIPS 2017*

### OOD Benchmarks and Theory
10. **DomainBed**: In Search of Lost Domain Generalization. *ICLR 2021*
11. **WILDS**: A Benchmark of in-the-Wild Distribution Shifts. *ICML 2021*
12. **ImageNet-C**: Benchmarking Neural Network Robustness. *ICLR 2019*

### Physics-Informed Machine Learning
13. **PINNs**: Physics-Informed Neural Networks. *Journal of Computational Physics 2019*
14. **Neural ODEs**: Neural Ordinary Differential Equations. *NeurIPS 2018*
15. **HNN**: Hamiltonian Neural Networks. *NeurIPS 2019*

[Additional references to be completed to reach 40-50 total]

## Appendix

### A. Experimental Details

#### A.1 Data Generation
[Details on physics simulation, parameter ranges, numerical integration]

#### A.2 Model Architecture
[Full architecture specifications, initialization schemes]

#### A.3 Training Procedures
[Optimization details, convergence criteria, computational resources]

#### A.4 Hyperparameter Selection
[Grid searches performed, final values selected, sensitivity analysis]

### B. Additional Results

#### B.1 Ablation Studies
[Learning rate variations, architecture changes, different physics parameters]

#### B.2 Visualization of Representations
[t-SNE plots, prediction trajectories, gradient landscapes]

#### B.3 Extended Results Tables
[Full numerical results with confidence intervals]

### C. Reproducibility

#### C.1 Code Availability
[GitHub repository link - to be added]

#### C.2 Computational Requirements
[GPU hours, memory requirements, framework versions]

#### C.3 Random Seeds
[Seed management, number of runs, statistical tests]
