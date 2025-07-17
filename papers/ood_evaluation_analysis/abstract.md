# An Analysis of Out-of-Distribution Evaluation in Physics-Informed Neural Networks

## Abstract

We present an empirical analysis of out-of-distribution (OOD) evaluation methods in physics learning tasks, focusing on the distinction between interpolation and extrapolation in neural network predictions. Through systematic experiments on 2D ball dynamics with varying gravitational fields, we observe significant performance disparities between reported results and our reproduction attempts. 

Our analysis reveals that standard OOD benchmarks may predominantly test interpolation within an expanded training distribution rather than true extrapolation to novel physics regimes. Using representation space analysis, we find that 91.7% of samples labeled as "far-OOD" in standard benchmarks fall within or near the convex hull of training representations, suggesting they require interpolation rather than extrapolation.

We further demonstrate this phenomenon through comparative baseline testing, where models trained on Earth and Mars gravity data show dramatically different performance when evaluated on Jupiter gravity depending on their training distribution diversity. Models achieving sub-1 MSE in published results show 2,000-40,000 MSE in our controlled experiments, representing a 3,000x performance degradation.

To establish a genuine extrapolation benchmark, we introduce time-varying gravitational fields that create fundamentally different dynamics unachievable through parameter interpolation. Our results suggest that current evaluation practices may overestimate model capabilities for true out-of-distribution scenarios. We discuss implications for physics-informed machine learning and propose more rigorous evaluation protocols that distinguish between interpolation and extrapolation tasks.

Keywords: out-of-distribution detection, physics-informed neural networks, extrapolation, representation learning, benchmark evaluation