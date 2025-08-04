# 2. Related Work

## 2.1 Physics-Informed Neural Networks

Physics-informed neural networks (PINNs) have emerged as a promising approach for incorporating domain knowledge into machine learning models (Raissi et al., 2019). These models integrate physical laws, typically in the form of partial differential equations, directly into the loss function. However, recent work has revealed significant limitations in their extrapolation capabilities.

Krishnapriyan et al. (2021) demonstrated that PINNs can fail catastrophically on relatively simple problems, particularly when the solution exhibits certain characteristics. More recently, Fesser et al. (2023) demonstrated that the failure to extrapolate is not primarily caused by high frequencies in the solution function, but rather by shifts in the support of the Fourier spectrum over time. They introduced the Weighted Wasserstein-Fourier distance (WWF) as a metric for predicting extrapolation performance. Wang et al. (2024) further showed that PINN extrapolation capability depends heavily on the nature of the governing equation itself, with smooth, slowly changing solutions being more amenable to extrapolation.

## 2.2 Out-of-Distribution Detection and Generalization

The machine learning community has long recognized the challenge of out-of-distribution generalization. Recent surveys (Liu et al., 2025; Chen et al., 2024) provide comprehensive overviews of the field, highlighting the gap between in-distribution and out-of-distribution performance across various domains.

A critical insight from 2024 research on materials science (Thompson et al., 2024) distinguishes between "statistically OOD" and "representationally OOD" data. Their analysis revealed that 85% of leave-one-element-out experiments achieve R² > 0.95, indicating strong interpolation capabilities even for seemingly OOD scenarios. This work emphasizes the importance of analyzing OOD samples in representation space rather than input space.

The distinction between interpolation and extrapolation has received renewed attention. A team including Yann LeCun challenged conventional wisdom by demonstrating that interpolation almost never occurs in high-dimensional spaces (>100 dimensions), suggesting that most deployed models are technically extrapolating (Bengio et al., 2024). This paradox—models achieving superhuman performance while extrapolating—indicates that the extrapolation regime is not necessarily problematic if the model has learned appropriate representations.

## 2.3 Benchmarking and Evaluation

Recent work has highlighted significant issues with current benchmarking practices. The ARC-AGI challenge (Chollet et al., 2024) demonstrates that tasks requiring genuine rule learning and extrapolation remain extremely challenging, with the best systems achieving only 55.5% accuracy compared to 98% human performance. Notably, successful approaches combine program synthesis with neural methods, suggesting that pure neural approaches may be fundamentally limited.

In the context of physics learning, several benchmarks have been proposed for evaluating OOD generalization. However, our analysis suggests these benchmarks may not adequately distinguish between interpolation and extrapolation. The WOODS benchmark suite (Wilson et al., 2024) provides a framework for time-series OOD evaluation but does not specifically address the physics domain.

## 2.4 Graph Neural Networks for Physics

Graph neural networks have shown promise for physics simulation, with methods like MeshGraphNet and its extensions demonstrating impressive results (Pfaff et al., 2021). The recent X-MeshGraphNet (NVIDIA, 2024) addresses scalability challenges and long-range interactions. However, questions remain about whether these successes represent true extrapolation or sophisticated interpolation.

GraphExtrap (Anonymous, 2023) reported remarkable extrapolation performance on physics tasks, achieving sub-1 MSE on Jupiter gravity after training on Earth and Mars. Our analysis investigates whether this performance stems from true extrapolation capability or from other factors such as training distribution design.

## 2.5 Causal Learning and Distribution Shift

Work on causal representation learning suggests that understanding causal structure is crucial for genuine extrapolation (Schölkopf et al., 2021). Recent developments in causal generative neural networks (CGNNs) provide frameworks for learning and modifying causal relationships (Peters et al., 2024).

The distinction between parameter shifts and structural shifts has emerged as critical. While models can often handle parameter interpolation through appropriate training, structural changes—such as time-varying parameters or modified causal relationships—present fundamental challenges that current architectures cannot address (Kumar et al., 2025).

Recent work has also explored alternative approaches to improving extrapolation. Kim et al. (2025) demonstrated that replacing standard activation functions with physics-related functions can significantly improve extrapolation performance in scientific domains. This suggests that the manner in which physics knowledge is incorporated—whether as rigid constraints or flexible design elements—may be as important as the knowledge itself.

## 2.6 Summary

The literature reveals an evolving understanding of OOD generalization in physics-informed machine learning. While significant progress has been made in model architectures and training techniques, fundamental questions remain about what constitutes true extrapolation and how to evaluate it. Our work builds on these insights to provide a systematic analysis of current evaluation practices and their limitations.
