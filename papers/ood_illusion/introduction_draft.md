# Introduction (Revised)

Recent advances in out-of-distribution (OOD) generalization have shown promising results on benchmarks involving style changes, corruptions, and domain shifts. However, when we evaluated these methods on physics prediction tasks with modified dynamical laws, we observed unexpected behavior: test-time adaptation (TTA) increased prediction error by 235%, while Model-Agnostic Meta-Learning (MAML) with adaptation showed a 62,290% increase in error compared to a standard baseline.

These results suggest a potential gap between the types of distribution shift commonly studied in machine learning benchmarks and those encountered in physics prediction. Many current benchmarks evaluate robustness to changes in surface statistics—artistic style, image corruptions, or contextual variations—while maintaining the same underlying task structure. In contrast, predicting motion under time-varying gravity requires extrapolation to fundamentally different dynamical laws.

This distinction raises questions about how we evaluate and develop OOD generalization methods. Consider two scenarios:
1. Training on photographs and testing on paintings of the same objects
2. Training on constant gravity dynamics and testing on time-varying gravity

While both involve distribution shift, they may require different types of generalization. Our experiments suggest that methods successful on the former may not transfer to the latter.

## Physics as a Testbed for Extrapolation

Physics prediction offers several advantages for studying generalization:

First, the ground truth is determined by known mathematical laws, providing objective evaluation criteria. Second, we can construct test scenarios that require extrapolation beyond the training regime—such as transitioning from constant to time-varying parameters. Third, successful prediction requires capturing causal relationships rather than surface correlations.

In this work, we focus on a seemingly simple task: predicting the motion of two balls under gravitational influence. During training, gravity remains constant at 9.8 m/s². At test time, we evaluate on trajectories where gravity varies as g(t) = 9.8 + 2sin(0.1t). This modification creates a test distribution that differs not just in statistics but in the underlying dynamics.

## Observed Behavior of Adaptation Methods

Our experiments reveal that methods designed for test-time adaptation show decreased performance on this physics extrapolation task. Test-time adaptation methods, which update model parameters using self-supervised objectives, consistently produce higher prediction errors than models without adaptation.

This behavior appears to stem from a mismatch between self-supervised objectives (such as prediction consistency or temporal smoothness) and accuracy on the downstream task. When test data follows different dynamical laws than training data, optimizing these auxiliary objectives can lead the model away from accurate predictions. In our experiments, we observed models converging to nearly constant predictions—achieving low variance but high prediction error.

These findings held across various hyperparameter settings and implementation choices, suggesting the issue may be fundamental to the approach rather than a matter of tuning.

## Contributions

This paper presents:

1. **Empirical Analysis**: A systematic evaluation of OOD generalization methods (TTA, MAML, ensemble approaches) on physics prediction with modified dynamics, documenting performance degradation ranging from marginal to severe.

2. **Categorization Framework**: A proposed distinction between distribution shifts in surface statistics versus underlying generative processes, which may help explain varying method performance across domains.

3. **Mechanistic Investigation**: Analysis of how self-supervised adaptation objectives interact with prediction accuracy under distribution shift, including visualization of the optimization landscape.

4. **Benchmark Considerations**: Discussion of design principles for benchmarks that evaluate extrapolation to new generative processes rather than robustness to statistical variations.

## Paper Organization

Section 2 reviews existing OOD generalization methods and benchmarks. Section 3 details our experimental setup using physics prediction. Section 4 presents empirical results across multiple methods. Section 5 analyzes the optimization dynamics during test-time adaptation. Section 6 discusses different types of distribution shift and their implications. Section 7 considers directions for benchmark and method development. Section 8 concludes.

Our findings suggest that the relationship between adaptation and generalization may be more complex than previously understood, particularly when test distributions involve different generative processes than training data. This work aims to contribute to the ongoing discussion about what constitutes meaningful out-of-distribution generalization and how to evaluate it.
