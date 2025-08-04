# 1. Introduction

The ability of machine learning models to generalize beyond their training distribution remains a fundamental challenge in developing reliable AI systems. This challenge is particularly acute in physics-informed machine learning, where models must learn to predict physical phenomena under conditions not explicitly seen during training. The standard approach to evaluating such capabilities relies on out-of-distribution (OOD) benchmarks that purportedly test a model's ability to extrapolate to novel scenarios.

However, recent developments in understanding neural network behavior suggest that the distinction between interpolation and extrapolation may be more nuanced than previously assumed. Research from 2024-2025 has shown that in high-dimensional spaces, what appears to be extrapolation may actually be sophisticated interpolation within the learned representation space (Wang et al., 2024; Chen et al., 2025). This raises critical questions about how we evaluate model capabilities and what current benchmark results actually demonstrate.

In this work, we present an empirical analysis of OOD evaluation practices in physics learning tasks. Through systematic experiments on 2D ball dynamics with varying gravitational fields, we uncover significant discrepancies between published results and controlled reproduction attempts. Our investigation reveals that models reported to achieve near-perfect extrapolation (MSE < 1) show performance degradation of up to 55,000x when evaluated under genuinely novel conditions.

## Research Questions

Our analysis addresses three key questions:

1. **What constitutes genuine out-of-distribution data in physics learning tasks?** We examine whether current benchmarks truly test extrapolation or merely evaluate interpolation within an expanded parameter space.

2. **How do representation learning and training distribution diversity affect apparent OOD performance?** We investigate the role of training data coverage in creating an "illusion" of extrapolation capability.

3. **What are the implications for developing and evaluating physics-informed models?** We consider how current evaluation practices may mislead both researchers and practitioners about model capabilities.

## Key Contributions

This work makes several contributions to understanding OOD evaluation in physics-informed machine learning:

- **Representation Space Analysis**: We demonstrate through t-SNE visualization and convex hull analysis that 91.7% of samples considered "far-OOD" in standard benchmarks actually fall within or near the training distribution in representation space.

- **Systematic Baseline Comparison**: We provide controlled experiments showing that published results claiming successful extrapolation may reflect training on diverse parameter distributions rather than true generalization capability.

- **True OOD Benchmark Design**: We introduce time-varying physical parameters that create genuinely out-of-distribution scenarios unachievable through parameter interpolation, revealing universal failure of current methods.

- **Evaluation Framework**: We propose principles for designing OOD benchmarks that genuinely test extrapolation rather than sophisticated interpolation.

## Paper Organization

The remainder of this paper is organized as follows. Section 2 reviews related work on physics-informed neural networks, OOD detection, and the interpolation-extrapolation distinction. Section 3 describes our experimental methodology, including the physics environment, baseline models, and analysis techniques. Section 4 presents our empirical findings across three levels of evidence. Section 5 discusses the implications of our results for the field. Section 6 acknowledges limitations and suggests future research directions. Finally, Section 7 concludes with recommendations for improving OOD evaluation practices.
