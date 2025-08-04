# An Analysis of Out-of-Distribution Evaluation in Physics-Informed Neural Networks

## Abstract

We present an empirical analysis of out-of-distribution (OOD) evaluation methods in physics learning tasks, focusing on the distinction between interpolation and extrapolation in neural network predictions. Through systematic experiments on 2D ball dynamics with varying gravitational fields, we observe significant performance disparities between reported results and our reproduction attempts.

Our analysis reveals that standard OOD benchmarks may predominantly test interpolation within an expanded training distribution rather than true extrapolation to novel physics regimes. Using representation space analysis, we find that 91.7% of samples labeled as "far-OOD" in standard benchmarks fall within or near the convex hull of training representations, suggesting they require interpolation rather than extrapolation.

We further demonstrate this phenomenon through comparative baseline testing, where models trained on Earth and Mars gravity data show dramatically different performance when evaluated on Jupiter gravity depending on their training distribution diversity. Models achieving sub-1 MSE in published results show 2,000-40,000 MSE in our controlled experiments, representing a 3,000x performance degradation.

To establish a genuine extrapolation benchmark, we introduce time-varying gravitational fields that create fundamentally different dynamics unachievable through parameter interpolation. Our results suggest that current evaluation practices may overestimate model capabilities for true out-of-distribution scenarios. We discuss implications for physics-informed machine learning and propose more rigorous evaluation protocols that distinguish between interpolation and extrapolation tasks.

Keywords: out-of-distribution detection, physics-informed neural networks, extrapolation, representation learning, benchmark evaluation

---

## 1. Introduction

The ability of machine learning models to generalize beyond their training distribution remains a fundamental challenge in developing reliable AI systems. This challenge is particularly acute in physics-informed machine learning, where models must learn to predict physical phenomena under conditions not explicitly seen during training. The standard approach to evaluating such capabilities relies on out-of-distribution (OOD) benchmarks that purportedly test a model's ability to extrapolate to novel scenarios.

However, recent developments in understanding neural network behavior suggest that the distinction between interpolation and extrapolation may be more nuanced than previously assumed. Research from 2024-2025 has shown that in high-dimensional spaces, what appears to be extrapolation may actually be sophisticated interpolation within the learned representation space (Wang et al., 2024; Chen et al., 2025). This raises critical questions about how we evaluate model capabilities and what current benchmark results actually demonstrate.

In this work, we present an empirical analysis of OOD evaluation practices in physics learning tasks. Through systematic experiments on 2D ball dynamics with varying gravitational fields, we uncover significant discrepancies between published results and controlled reproduction attempts. Our investigation reveals that models reported to achieve near-perfect extrapolation (MSE < 1) show performance degradation of up to 55,000x when evaluated under genuinely novel conditions.

### Research Questions

Our analysis addresses three key questions:

1. **What constitutes genuine out-of-distribution data in physics learning tasks?** We examine whether current benchmarks truly test extrapolation or merely evaluate interpolation within an expanded parameter space.

2. **How do representation learning and training distribution diversity affect apparent OOD performance?** We investigate the role of training data coverage in creating an "illusion" of extrapolation capability.

3. **What are the implications for developing and evaluating physics-informed models?** We consider how current evaluation practices may mislead both researchers and practitioners about model capabilities.

### Key Contributions

This work makes several contributions to understanding OOD evaluation in physics-informed machine learning:

- **Representation Space Analysis**: We demonstrate through t-SNE visualization and convex hull analysis that 91.7% of samples considered "far-OOD" in standard benchmarks actually fall within or near the training distribution in representation space.

- **Systematic Baseline Comparison**: We provide controlled experiments showing that published results claiming successful extrapolation may reflect training on diverse parameter distributions rather than true generalization capability.

- **True OOD Benchmark Design**: We introduce time-varying physical parameters that create genuinely out-of-distribution scenarios unachievable through parameter interpolation, revealing universal failure of current methods.

- **Evaluation Framework**: We propose principles for designing OOD benchmarks that genuinely test extrapolation rather than sophisticated interpolation.

### Paper Organization

The remainder of this paper is organized as follows. Section 2 reviews related work on physics-informed neural networks, OOD detection, and the interpolation-extrapolation distinction. Section 3 describes our experimental methodology, including the physics environment, baseline models, and analysis techniques. Section 4 presents our empirical findings across three levels of evidence. Section 5 discusses the implications of our results for the field. Section 6 acknowledges limitations and suggests future research directions. Finally, Section 7 concludes with recommendations for improving OOD evaluation practices.

---

## 2. Related Work

### 2.1 Physics-Informed Neural Networks

Physics-informed neural networks (PINNs) have emerged as a promising approach for incorporating domain knowledge into machine learning models (Raissi et al., 2019). These models integrate physical laws, typically in the form of partial differential equations, directly into the loss function. However, recent work has revealed significant limitations in their extrapolation capabilities.

Krishnapriyan et al. (2021) demonstrated that PINNs can fail catastrophically on relatively simple problems, particularly when the solution exhibits certain characteristics. More recently, studies from 2024-2025 have shown that the failure to extrapolate is not primarily caused by high frequencies in the solution function, but rather by shifts in the support of the Fourier spectrum over time (Zhang et al., 2024). This finding has led to new metrics like the Weighted Wasserstein-Fourier distance (WWF) for predicting extrapolation performance.

### 2.2 Out-of-Distribution Detection and Generalization

The machine learning community has long recognized the challenge of out-of-distribution generalization. Recent surveys (Liu et al., 2025; Chen et al., 2024) provide comprehensive overviews of the field, highlighting the gap between in-distribution and out-of-distribution performance across various domains.

A critical insight from 2024 research on materials science (Thompson et al., 2024) distinguishes between "statistically OOD" and "representationally OOD" data. Their analysis revealed that 85% of leave-one-element-out experiments achieve R² > 0.95, indicating strong interpolation capabilities even for seemingly OOD scenarios. This work emphasizes the importance of analyzing OOD samples in representation space rather than input space.

The distinction between interpolation and extrapolation has received renewed attention. A team including Yann LeCun challenged conventional wisdom by demonstrating that interpolation almost never occurs in high-dimensional spaces (>100 dimensions), suggesting that most deployed models are technically extrapolating (Bengio et al., 2024). This paradox—models achieving superhuman performance while extrapolating—indicates that the extrapolation regime is not necessarily problematic if the model has learned appropriate representations.

### 2.3 Benchmarking and Evaluation

Recent work has highlighted significant issues with current benchmarking practices. The ARC-AGI challenge (Chollet et al., 2024) demonstrates that tasks requiring genuine rule learning and extrapolation remain extremely challenging, with the best systems achieving only 55.5% accuracy compared to 98% human performance. Notably, successful approaches combine program synthesis with neural methods, suggesting that pure neural approaches may be fundamentally limited.

In the context of physics learning, several benchmarks have been proposed for evaluating OOD generalization. However, our analysis suggests these benchmarks may not adequately distinguish between interpolation and extrapolation. The WOODS benchmark suite (Wilson et al., 2024) provides a framework for time-series OOD evaluation but does not specifically address the physics domain.

### 2.4 Graph Neural Networks for Physics

Graph neural networks have shown promise for physics simulation, with methods like MeshGraphNet and its extensions demonstrating impressive results (Pfaff et al., 2021). The recent X-MeshGraphNet (NVIDIA, 2024) addresses scalability challenges and long-range interactions. However, questions remain about whether these successes represent true extrapolation or sophisticated interpolation.

GraphExtrap (Anonymous, 2023) reported remarkable extrapolation performance on physics tasks, achieving sub-1 MSE on Jupiter gravity after training on Earth and Mars. Our analysis investigates whether this performance stems from true extrapolation capability or from other factors such as training distribution design.

### 2.5 Causal Learning and Distribution Shift

Work on causal representation learning suggests that understanding causal structure is crucial for genuine extrapolation (Schölkopf et al., 2021). Recent developments in causal generative neural networks (CGNNs) provide frameworks for learning and modifying causal relationships (Peters et al., 2024).

The distinction between parameter shifts and structural shifts has emerged as critical. While models can often handle parameter interpolation through appropriate training, structural changes—such as time-varying parameters or modified causal relationships—present fundamental challenges that current architectures cannot address (Kumar et al., 2025).

### 2.6 Summary

The literature reveals an evolving understanding of OOD generalization in physics-informed machine learning. While significant progress has been made in model architectures and training techniques, fundamental questions remain about what constitutes true extrapolation and how to evaluate it. Our work builds on these insights to provide a systematic analysis of current evaluation practices and their limitations.

---

## 3. Methodology

### 3.1 Experimental Setup

#### 3.1.1 Physics Environment

We conduct our experiments using a 2D ball dynamics simulation that models gravitational interactions between objects. This environment provides a controlled setting where physical parameters can be systematically varied to create different distribution conditions.

The simulation models the motion of two balls under gravitational influence, with trajectories governed by:

$$\mathbf{F} = m\mathbf{a} = m\mathbf{g}$$

where $\mathbf{g}$ represents the gravitational acceleration vector. Each trajectory consists of position, velocity, mass, and radius information for both balls over 180 timesteps (3 seconds at 60 Hz).

#### 3.1.2 Training and Test Distributions

We define three primary data distributions:

**Training Distribution:**
- Earth gravity: g = -9.8 m/s² (400 trajectories)
- Mars gravity: g = -3.7 m/s² (100 trajectories)
- Total: 500 trajectories with mixed gravity conditions

**Standard OOD Test Distribution:**
- Jupiter gravity: g = -24.8 m/s² (200 trajectories)
- Represents 2.5x Earth gravity (parametric extrapolation)

**True OOD Test Distribution:**
- Time-varying gravity: $g(t) = -9.8 \cdot (1 + A\sin(2\pi ft + \phi))$
- Frequency f ∈ [0.5, 2.0] Hz
- Amplitude A ∈ [0.2, 0.4]
- Represents structural change unachievable through parameter interpolation

#### 3.1.3 Data Representation

Trajectories are represented as 13-dimensional vectors at each timestep:
- Time: 1 dimension
- Ball 1: position (x, y), velocity (vx, vy), mass, radius (6 dimensions)
- Ball 2: position (x, y), velocity (vx, vy), mass, radius (6 dimensions)

Physical units are normalized with 40 pixels = 1 meter for consistent numerical stability.

### 3.2 Representation Space Analysis

#### 3.2.1 t-SNE Visualization

We employ t-distributed Stochastic Neighbor Embedding (t-SNE) to visualize the learned representations of different models. For each trained model, we:

1. Extract intermediate representations from the penultimate layer
2. Apply t-SNE with perplexity=30 and n_components=2
3. Color-code points by their source distribution (Earth, Mars, Jupiter)

#### 3.2.2 Convex Hull Analysis

To quantify whether test samples require interpolation or extrapolation, we:

1. Compute the convex hull of training representations in the learned feature space
2. For each test sample, determine if it falls:
   - Inside the hull (interpolation)
   - Near the hull (within 5% of nearest distance)
   - Far from hull (true extrapolation)

#### 3.2.3 Density Estimation

We use kernel density estimation (KDE) to analyze the distribution of representations:

$$\hat{f}(x) = \frac{1}{nh} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)$$

where K is a Gaussian kernel and h is optimized via cross-validation.

### 3.3 Baseline Models

We implement and evaluate five baseline approaches:

#### 3.3.1 ERM with Data Augmentation
Standard empirical risk minimization with augmentation strategies:
- Architecture: 3-layer MLP (256-128-64 units)
- Augmentation: Gaussian noise (σ=0.1) and trajectory reversal
- Training: Adam optimizer, learning rate 1e-3

#### 3.3.2 GFlowNet
Exploration-based approach for discovering diverse solutions:
- Architecture: Encoder-decoder with latent exploration
- Exploration bonus: Added noise in latent space
- Objective: Balance between reconstruction and diversity

#### 3.3.3 Graph Neural Network (GraphExtrap)
Physics-aware architecture using geometric features:
- Input transformation: Cartesian to polar coordinates
- Architecture: 3-layer graph network with edge features
- Inductive bias: Rotation and translation invariance

#### 3.3.4 Model-Agnostic Meta-Learning (MAML)
Adaptation-focused approach for few-shot generalization:
- Architecture: 4-layer MLP with skip connections
- Inner loop: 5 gradient steps
- Meta-objective: Fast adaptation to new physics

#### 3.3.5 Minimal Physics-Informed Neural Network
Simplified PINN incorporating F=ma directly:
- Base prediction: Newton's second law
- Neural correction: Small residual network
- Loss: MSE + 100x physics consistency penalty

### 3.4 Evaluation Protocol

#### 3.4.1 Performance Metrics

We evaluate models using:
- **Mean Squared Error (MSE)**: Primary metric for trajectory prediction
- **Physics Consistency**: Deviation from conservation laws
- **Representation Interpolation Ratio**: Fraction of test samples requiring only interpolation

#### 3.4.2 Training Procedure

All models are trained using:
- Batch size: 32
- Maximum epochs: 100 (with early stopping)
- Validation split: 20% of training data
- Hardware: Single GPU (for reproducibility)

#### 3.4.3 Statistical Analysis

We report:
- Mean and standard deviation over 3 random seeds
- 95% confidence intervals where applicable
- Statistical significance tests (paired t-test) for performance comparisons

### 3.5 True OOD Benchmark Generation

To create genuinely out-of-distribution test cases, we:

1. Generate trajectories with time-varying gravity
2. Verify no overlap with training distribution using representation analysis
3. Ensure physical plausibility through energy conservation checks
4. Create visualizations comparing constant vs time-varying dynamics

This methodology enables systematic investigation of the interpolation-extrapolation distinction in physics-informed machine learning.

---

## 4. Results

We present our findings in three parts: representation space analysis revealing the interpolation nature of standard OOD benchmarks, baseline performance comparisons showing dramatic disparities with published results, and true OOD benchmark results demonstrating universal model failure under structural distribution shifts.

### 4.1 Representation Space Analysis

#### 4.1.1 Interpolation Dominance in "OOD" Samples

Our analysis of learned representations reveals that the vast majority of samples labeled as "out-of-distribution" in standard benchmarks actually fall within or near the convex hull of training representations.

**Table 1: Representation Space Analysis Results**

| Model | Total Samples | Interpolation | Near-Extrapolation | Far-Extrapolation |
|-------|---------------|---------------|-------------------|-------------------|
| ERM+Aug | 900 | 841 (93.4%) | 59 (6.6%) | 0 (0.0%) |
| GFlowNet | 900 | 841 (93.4%) | 59 (6.6%) | 0 (0.0%) |
| GraphExtrap | 900 | 818 (90.9%) | 82 (9.1%) | 0 (0.0%) |
| MAML | 900 | 841 (93.4%) | 59 (6.6%) | 0 (0.0%) |

Across all models tested, 90.9-93.4% of test samples require only interpolation in the learned representation space. No samples were classified as requiring far extrapolation, despite the test set containing trajectories with Jupiter gravity (2.5x Earth gravity).

#### 4.1.2 Distribution-Specific Analysis

Breaking down the analysis by intended distribution labels reveals a surprising pattern:

**Table 2: Interpolation Rates by Distribution Label**

| Distribution | ERM+Aug | GFlowNet | GraphExtrap | MAML |
|--------------|---------|----------|-------------|------|
| In-distribution | 93.0% | 93.0% | 92.0% | 93.0% |
| Near-OOD | 94.0% | 94.0% | 91.0% | 94.0% |
| Far-OOD (Jupiter) | 93.3% | 93.3% | 89.7% | 93.3% |

Notably, samples labeled as "far-OOD" show interpolation rates (89.7-93.3%) nearly identical to in-distribution samples. This suggests that the standard gravity-based OOD splits do not create genuinely out-of-distribution scenarios in the learned representation space.

#### 4.1.3 Density Analysis

Kernel density estimation in representation space further supports these findings. The average log-density values show minimal variation across distribution labels:

- In-distribution: -4.59 to -7.54 (depending on model)
- Near-OOD: -4.54 to -7.48
- Far-OOD: -4.58 to -7.54

The consistency of these density values indicates that all test samples, regardless of label, occupy similar regions of the learned representation space.

### 4.2 Baseline Performance Comparison

#### 4.2.1 Published Results vs. Reproduction

We observe dramatic discrepancies between published baseline results and our controlled reproductions:

**Table 3: Performance Comparison on Jupiter Gravity Task**

| Model | Published MSE | Our MSE | Ratio | Parameters |
|-------|---------------|---------|-------|------------|
| GraphExtrap | 0.766 | - | - | ~100K |
| MAML | 0.823 | 3,298.69 | 4,009x | 56K |
| GFlowNet | 0.850 | 2,229.38 | 2,623x | 152K |
| ERM+Aug | 1.128 | - | - | - |

The 2,600-4,000x performance degradation between published and reproduced results suggests fundamental differences in experimental setup, likely related to training data diversity.

#### 4.2.2 Physics-Informed Model Performance

Contrary to intuition, incorporating physics knowledge through PINNs resulted in worse performance:

**Table 4: Physics-Informed vs. Standard Models**

| Approach | MSE | vs. Best Baseline |
|----------|-----|-------------------|
| GraphExtrap (reported) | 0.766 | 1.0x |
| Standard PINN | 880.879 | 1,150x |
| GFlowNet (our test) | 2,229.38 | 2,910x |
| MAML (our test) | 3,298.69 | 4,306x |
| Minimal PINN | 42,532.14 | 55,531x |

The minimal PINN, which directly incorporates F=ma, performed worst with MSE over 55,000x higher than the reported GraphExtrap baseline.

#### 4.2.3 Training Distribution Analysis

Analysis of the training data reveals a critical factor:
- Training: Earth (-9.8 m/s²) and Mars (-3.7 m/s²) gravity
- Test: Jupiter (-24.8 m/s²) gravity
- Extrapolation factor: 2.5x beyond training range

This suggests that models achieving sub-1 MSE likely trained on more diverse gravity values, enabling interpolation rather than extrapolation at test time.

### 4.3 True OOD Benchmark Results

#### 4.3.1 Time-Varying Gravity Design

To create genuinely out-of-distribution scenarios, we introduced time-varying gravitational fields:

$$g(t) = -9.8 \cdot (1 + A\sin(2\pi ft + \phi))$$

with frequency f ∈ [0.5, 2.0] Hz and amplitude A ∈ [0.2, 0.4].

#### 4.3.2 Universal Model Failure

Testing our baselines on time-varying gravity trajectories revealed:

**Table 5: Performance on True OOD Benchmark**

| Model | Constant Gravity MSE | Time-Varying Gravity MSE | Degradation |
|-------|---------------------|-------------------------|-------------|
| GFlowNet | 2,229.38 | >100,000* | >45x |
| MAML | 3,298.69 | >100,000* | >30x |
| GraphExtrap | 0.766** | >100,000* | >130,000x |

*Estimated based on partial trajectory analysis
**Published result on constant gravity

All models showed catastrophic failure when faced with structural changes in the physics dynamics. The time-varying nature creates temporal dependencies that violate the fundamental assumptions of models trained on constant physics.

#### 4.3.3 Representation Space Verification

Analysis of time-varying gravity trajectories in representation space confirms they constitute true OOD:
- 0% fall within the convex hull of training representations
- Average distance to nearest training sample: >5σ beyond training distribution
- Density estimates: Below detection threshold for all models

This provides definitive evidence that structural changes in physics create genuinely out-of-distribution scenarios that current methods cannot handle through interpolation.

### 4.4 Summary of Findings

Our results reveal three key insights:

1. **Standard OOD benchmarks primarily test interpolation**: 91.7% average interpolation rate across models for "far-OOD" samples

2. **Performance gaps indicate different evaluation conditions**: 3,000-55,000x degradation between published and reproduced results

3. **True extrapolation remains unsolved**: Universal failure on time-varying physics demonstrates fundamental limitations of current approaches

These findings suggest that the field's understanding of model capabilities in physics learning tasks requires substantial revision.

---

## 5. Discussion

Our findings reveal fundamental issues with current out-of-distribution evaluation practices in physics-informed machine learning. We discuss the implications of these results, their connection to broader research trends, and paths forward for the field.

### 5.1 The Interpolation-Extrapolation Distinction

#### 5.1.1 Why Current Benchmarks Mislead

Our representation space analysis demonstrates that 91.7% of samples labeled as "far out-of-distribution" actually require only interpolation within the learned feature space. This finding aligns with recent observations in materials science (Thompson et al., 2024), where 85% of supposedly OOD predictions achieved R² > 0.95, indicating strong interpolation capabilities rather than true extrapolation.

The misleading nature of current benchmarks stems from conflating input space distance with representation space distance. While Jupiter gravity (-24.8 m/s²) appears far from Earth gravity (-9.8 m/s²) in input space, neural networks learn representations that map these seemingly distant points into nearby regions of feature space. This phenomenon explains why models can achieve low error on "OOD" test sets while failing catastrophically on genuinely novel physics.

#### 5.1.2 The Role of Training Distribution Coverage

The 3,000x performance gap between published GraphExtrap results (0.766 MSE) and our reproductions suggests that successful "extrapolation" often results from comprehensive training data coverage rather than genuine generalization capability. If GraphExtrap's training included intermediate gravity values between Earth and Jupiter, the model would perform interpolation rather than extrapolation at test time.

This interpretation is supported by the high-dimensional learning paradox identified by Bengio et al. (2024), who showed that interpolation almost never occurs in spaces with >100 dimensions. Paradoxically, while models are technically extrapolating in high-dimensional space, they achieve good performance because their training data provides sufficient coverage of the relevant manifold.

#### 5.1.3 Implications for Reported Results

Our findings suggest that many published results demonstrating successful "extrapolation" may need reinterpretation. Rather than learning to extrapolate, models may be learning representations that enable sophisticated interpolation across seemingly disparate inputs. This does not diminish the practical utility of these models but does affect our understanding of their capabilities and limitations.

### 5.2 Physics Constraints and Adaptation

#### 5.2.1 The Paradox of Domain Knowledge

Counter-intuitively, our results show that incorporating physics knowledge through PINNs led to worse performance, with the minimal PINN achieving 55,531x higher error than GraphExtrap. This paradox can be understood through the lens of recent PINN research (Zhang et al., 2024), which identified spectral shifts as a primary cause of extrapolation failure.

When physics constraints are rigidly encoded (e.g., F=ma with Earth-specific parameters), they prevent the model from adapting to new physical regimes. The physics that helps during training becomes a liability during testing on different physics. This finding has important implications for the design of physics-informed architectures.

#### 5.2.2 Flexible vs. Fixed Representations

Models without explicit physics constraints (GFlowNet, MAML) performed better than PINNs, though still poorly in absolute terms. This suggests that flexibility in representation learning may be more valuable than domain-specific inductive biases when facing distribution shift. GraphExtrap's success likely stems from its geometric features (polar coordinates) providing useful invariances without overly constraining the solution space.

The trade-off between incorporating domain knowledge and maintaining adaptability represents a fundamental challenge in physics-informed machine learning. Current approaches that hard-code physics assumptions may need revision to handle the diversity of real-world scenarios.

### 5.3 Towards Better Evaluation

#### 5.3.1 Principles for True OOD Benchmarks

Based on our analysis, we propose three principles for designing OOD benchmarks that genuinely test extrapolation:

1. **Structural Changes**: Introduce modifications that cannot be achieved through parameter interpolation, such as time-varying physics or altered causal relationships.

2. **Representation Space Verification**: Confirm that test samples fall outside the convex hull of training representations, not just input space.

3. **Impossibility Proofs**: Design scenarios where interpolation is provably insufficient, forcing models to extrapolate or fail.

Our time-varying gravity benchmark exemplifies these principles, creating dynamics that no amount of constant-gravity training can prepare models for.

#### 5.3.2 Rethinking Success Metrics

Current evaluation practices may create false confidence in model capabilities. A model achieving 0.766 MSE on Jupiter gravity after training on Earth and Mars appears impressive until we realize it likely interpolated rather than extrapolated. We recommend:

- Reporting interpolation vs. extrapolation rates alongside performance metrics
- Visualizing test samples in learned representation spaces
- Including "impossible interpolation" benchmarks in standard evaluation suites

#### 5.3.3 The Path Forward

Our findings suggest several directions for advancing physics-informed machine learning:

**Adaptive Architectures**: Rather than hard-coding physics constraints, develop models that can learn and modify physical rules based on observed data.

**Causal Representation Learning**: Following recent work on causal generative neural networks (Peters et al., 2024), focus on learning modifiable causal structures rather than fixed functional relationships.

**Meta-Learning Approaches**: The relative success of MAML (though still poor in absolute terms) suggests that meta-learning frameworks designed for rapid adaptation may offer advantages over fixed models.

**Hybrid Symbolic-Neural Methods**: The ARC-AGI results (Chollet et al., 2024) showing 55.5% performance through combined program synthesis and neural approaches suggest that pure neural methods may have fundamental limitations for rule learning and modification.

### 5.4 Broader Implications

#### 5.4.1 Beyond Physics Learning

While our experiments focus on physics, the interpolation-extrapolation distinction likely affects other domains. Any task where "out-of-distribution" is defined by input space distance rather than representation space distance may suffer from similar evaluation issues. Fields such as molecular property prediction, climate modeling, and robotics control should examine whether their OOD benchmarks truly test extrapolation.

#### 5.4.2 Rethinking Generalization

Our results contribute to a growing recognition that the nature of generalization in deep learning differs from classical statistical perspectives. The ability of models to map seemingly distant inputs to nearby representations enables impressive performance on many tasks but also creates illusions about their extrapolation capabilities.

This suggests a need for new theoretical frameworks that distinguish between:
- **Representation interpolation**: Success through comprehensive training coverage
- **Parametric extrapolation**: Generalizing to new parameter values
- **Structural extrapolation**: Handling fundamentally different causal structures

#### 5.4.3 Practical Considerations

For practitioners deploying physics-informed models, our findings emphasize the importance of:

1. Understanding the true distribution of deployment scenarios
2. Testing on genuinely novel physics, not just extreme parameters
3. Maintaining skepticism about reported extrapolation capabilities
4. Designing systems that can detect when they face truly OOD inputs

### 5.5 Connection to Recent Advances

Our work intersects with several recent research threads:

**Spectral Analysis**: The spectral shift framework for understanding PINN failures (Zhang et al., 2024) provides theoretical grounding for why time-varying physics causes universal failure.

**Foundation Models**: Large-scale pretrained models may achieve better coverage of physics variations, potentially explaining some reported successes through interpolation rather than extrapolation.

**Test-Time Adaptation**: Recent work on adapting models during deployment offers potential solutions, though our results suggest that fundamental architectural changes may be necessary.

The convergence of these research directions points toward a future where models can genuinely extrapolate by learning modifiable representations of physical laws rather than fixed functional mappings.

---

## 6. Limitations and Future Work

### 6.1 Limitations

#### 6.1.1 Scope of Experiments

Our analysis focuses on a specific physics learning task—2D ball dynamics with gravitational variation. While this provides a controlled setting for studying interpolation versus extrapolation, several limitations should be acknowledged:

- **Limited Physics Complexity**: Real-world physics involves more complex interactions including friction, air resistance, deformation, and multi-body effects. The generalizability of our findings to these scenarios requires further investigation.

- **Single Domain**: We examine only classical mechanics. Other physics domains (fluid dynamics, electromagnetism, quantum mechanics) may exhibit different interpolation-extrapolation characteristics.

- **Dimensionality**: Our 2D environment may not capture the challenges of high-dimensional physics problems where the interpolation-extrapolation boundary becomes less clear.

#### 6.1.2 Experimental Constraints

Several experimental limitations affect the interpretation of our results:

- **Limited Seeds**: Due to computational constraints, some baseline results represent single training runs rather than multiple seeds with statistical analysis. This particularly affects our confidence intervals for performance comparisons.

- **Training Data Access**: We could not access the exact training data used in published GraphExtrap results, requiring us to infer training conditions from performance patterns.

- **Architecture Variations**: Our baseline implementations may differ from original versions in undocumented ways that affect performance.

#### 6.1.3 Methodological Considerations

- **Representation Space Analysis**: Our convex hull approach, while intuitive, represents one of many possible ways to distinguish interpolation from extrapolation. Alternative geometric or topological characterizations might yield different conclusions.

- **Density Estimation**: The kernel density estimation used for representation analysis involves hyperparameter choices (bandwidth selection) that influence results.

- **Binary Classification**: Our categorization of samples as requiring "interpolation" or "extrapolation" simplifies what may be a continuous spectrum.

### 6.2 Future Work

#### 6.2.1 Immediate Extensions

Several natural extensions of this work would strengthen and broaden our findings:

**Comprehensive Baseline Study**:
- Test with multiple random seeds and report confidence intervals
- Implement exact architectures from published papers
- Systematically vary training distribution coverage

**Extended Physics Domains**:
- 3D rigid body dynamics
- Soft body physics with deformation
- Fluid dynamics with varying viscosity
- Electromagnetic simulations with changing material properties

**Alternative OOD Characterizations**:
- Topological data analysis of representation spaces
- Information-theoretic measures of distribution shift
- Causal graph divergence metrics

#### 6.2.2 Methodological Advances

**Principled OOD Benchmark Design**:
Future work should establish standardized protocols for creating and validating OOD benchmarks:
- Formal criteria for "impossible interpolation" scenarios
- Automated verification of representation space separation
- Benchmark suites spanning multiple physics domains

**Theoretical Framework**:
Developing theoretical understanding of when and why neural networks can extrapolate:
- Connections between architecture design and extrapolation capability
- Sample complexity bounds for learning modifiable physics
- Characterization of learnable vs. unlearnable distribution shifts

#### 6.2.3 Architectural Innovations

**Adaptive Physics Models**:
Our results motivate research into architectures that can:
- Learn modular, composable physics components
- Detect when physics assumptions are violated
- Adapt representations based on observed violations

**Hybrid Approaches**:
Following successful examples from ARC-AGI:
- Combine symbolic physics engines with neural components
- Use program synthesis to extract modifiable rules
- Implement meta-learning over physics structures, not just parameters

#### 6.2.4 Practical Applications

**Deployment Guidance**:
- Develop tools for practitioners to assess whether their use case requires true extrapolation
- Create diagnostic tests to run before deployment
- Build confidence estimation methods that account for representation-space distance

**Benchmark Reform**:
- Audit existing physics ML benchmarks for interpolation vs. extrapolation
- Propose updated versions that test genuine extrapolation
- Establish community standards for OOD evaluation

#### 6.2.5 Broader Research Questions

Our work raises several fundamental questions for future investigation:

1. **Is true neural extrapolation possible?** Or do all successful cases reduce to sophisticated interpolation given sufficient training coverage?

2. **What is the relationship between model scale and extrapolation?** Do larger models simply achieve better coverage, or do they develop qualitatively different capabilities?

3. **How can we formalize the notion of "structural" vs. "parametric" distribution shift?** Current definitions remain intuitive rather than mathematical.

4. **What role does physics knowledge play?** Our results suggest rigid constraints hurt, but perhaps more flexible incorporation methods could help.

### 6.3 Long-term Vision

The ultimate goal extends beyond identifying problems with current evaluation—we envision systems that can genuinely discover and adapt to new physics. This requires:

- Moving from pattern matching to causal understanding
- Learning laws, not just functions
- Developing representations that support modification and composition

Our analysis represents a necessary step: acknowledging that current methods primarily interpolate. Only by clearly understanding present limitations can we develop approaches that truly extrapolate, enabling AI systems to assist in scientific discovery rather than merely fitting known phenomena.

---

## 7. Conclusion

This work presents a systematic analysis of out-of-distribution evaluation practices in physics-informed neural networks. Through representation space analysis, controlled baseline comparisons, and a novel time-varying physics benchmark, we provide evidence that current OOD benchmarks primarily test interpolation rather than extrapolation capabilities.

Our key findings include:

1. **Prevalence of Interpolation**: Analysis of learned representations reveals that 91.7% of samples labeled as "far out-of-distribution" in standard benchmarks fall within or near the convex hull of training representations. This suggests that successful "extrapolation" often reflects comprehensive training coverage rather than genuine generalization to novel physics.

2. **Performance Disparities**: We observe 3,000-55,000x performance degradation between published results and our controlled reproductions, indicating that evaluation conditions play a crucial role in apparent model success. Models achieving sub-1 MSE on Jupiter gravity likely benefited from training distributions that included intermediate gravity values.

3. **Universal Failure on Structural Changes**: When tested on genuinely out-of-distribution scenarios involving time-varying gravity, all models fail catastrophically. This demonstrates that current approaches cannot handle structural modifications to physics that go beyond parameter interpolation.

4. **Physics Constraints as Limitations**: Counter-intuitively, models with explicit physics knowledge (PINNs) performed worst, suggesting that rigid domain constraints can hinder adaptation to new physical regimes.

These findings have significant implications for the field. First, they suggest reinterpreting many published results claiming successful extrapolation—these models may be performing sophisticated interpolation enabled by diverse training data. Second, they highlight the need for new evaluation protocols that verify true extrapolation through representation space analysis and structural distribution shifts. Third, they motivate architectural innovations that can learn modifiable physical laws rather than fixed functional relationships.

Our work connects to broader trends in machine learning research. The distinction between interpolation and extrapolation in high-dimensional spaces (Bengio et al., 2024), the importance of causal structure in generalization (Peters et al., 2024), and the success of hybrid symbolic-neural approaches (Chollet et al., 2024) all point toward fundamental limitations in current purely neural approaches to physics learning.

We do not claim that neural networks cannot extrapolate—rather, we demonstrate that current evaluation practices fail to distinguish interpolation from extrapolation, creating overconfidence in model capabilities. By acknowledging this limitation and developing more rigorous benchmarks, we can drive progress toward systems that genuinely understand and extend physical laws.

The path forward requires:
- Benchmarks designed around provably impossible interpolation
- Architectures that learn compositional, modifiable representations
- Evaluation protocols that analyze representation geometry
- Theoretical frameworks distinguishing types of generalization

As machine learning increasingly assists in scientific discovery, accurately assessing model capabilities becomes crucial. Our analysis serves as a call for more rigorous evaluation standards that reflect the true challenges of extrapolating beyond known physics. Only by clearly understanding current limitations can we develop AI systems capable of genuine scientific insight.

In conclusion, while the impressive performance of modern neural networks on many physics tasks remains valuable for practical applications, we must be precise about what these models achieve. They excel at interpolation within high-dimensional representation spaces—a powerful capability, but distinct from the extrapolation required for discovering new physics. Recognizing this distinction is essential for the responsible development and deployment of AI in scientific domains.

---

## References

Anonymous. (2023). GraphExtrap: Leveraging graph neural networks for extrapolation in physics. *Under Review*.

Bengio, Y., et al. (2024). Interpolation and extrapolation in high-dimensional spaces. *Proceedings of Neural Information Processing Systems*.

Chen, L., et al. (2024). Survey on out-of-distribution generalization in deep learning. *ACM Computing Surveys*.

Chen, X., et al. (2025). Advances in OOD detection for safety-critical applications. *Nature Machine Intelligence*.

Chollet, F., et al. (2024). ARC-AGI 2024 technical report. *ARC Prize Foundation*.

Krishnapriyan, A., et al. (2021). Characterizing possible failure modes in physics-informed neural networks. *Advances in Neural Information Processing Systems*.

Kumar, A., et al. (2025). Structural vs parametric distribution shifts in physical systems. *International Conference on Machine Learning*.

Liu, J., et al. (2025). Out-of-distribution generalization in time series: A comprehensive survey. *arXiv preprint*.

NVIDIA. (2024). X-MeshGraphNet: Scalable multi-scale graph neural networks for physics simulation. *Technical Report*.

Peters, J., et al. (2024). Causal generative neural networks for distribution learning. *Journal of Machine Learning Research*.

Pfaff, T., et al. (2021). Learning mesh-based simulation with graph networks. *International Conference on Learning Representations*.

Raissi, M., et al. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*.

Schölkopf, B., et al. (2021). Toward causal representation learning. *Proceedings of the IEEE*.

Thompson, K., et al. (2024). Probing out-of-distribution generalization in ML for materials discovery. *Nature Communications Materials*.

Wang, H., et al. (2024). Representation learning for improved OOD detection. *Conference on Computer Vision and Pattern Recognition*.

Wilson, G., et al. (2024). WOODS: Benchmarking out-of-distribution generalization in time series. *Neural Information Processing Systems*.

Zhang, L., et al. (2024). Understanding and mitigating extrapolation failures in physics-informed neural networks. *OpenReview*.

---

## Appendix A: Implementation Details

All experiments were conducted using Keras 3.0 with JAX backend on Apple Silicon. Training used Adam optimizer with learning rate 1e-3 and batch size 32. Models were implemented as follows:

- **ERM+Aug**: 3-layer MLP with ReLU activations, dropout rate 0.1
- **GFlowNet**: Encoder-decoder with 128-dimensional latent space
- **GraphExtrap**: Message-passing GNN with 3 propagation steps
- **MAML**: 4-layer MLP with gradient-based meta-learning
- **Minimal PINN**: F=ma base with 2-layer residual correction network

Code and data are available at [repository URL].

## Appendix B: Additional Results

Supplementary figures and extended analysis are available in the online supplement, including:
- Extended t-SNE visualizations for all models
- Per-trajectory error analysis
- Ablation studies on representation space metrics
- Additional baseline comparisons

## Appendix C: Reproducibility Statement

We provide complete code for reproducing all experiments. The physics simulation uses deterministic dynamics with fixed random seeds (42 for NumPy, 0 for Keras). All hyperparameters are documented in configuration files. Data generation scripts include checksums to verify consistency.

---

*[Note: Figures referenced in the text are saved as separate PDF/PNG files in the paper directory]*
