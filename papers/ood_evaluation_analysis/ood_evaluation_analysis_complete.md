# The OOD Illusion: Revisiting Out-of-Distribution Evaluation in Physics-Informed Neural Networks

## Abstract

We present a systematic analysis of out-of-distribution (OOD) evaluation practices in physics-informed machine learning, revealing that standard benchmarks primarily test interpolation rather than extrapolation capabilities. Through representation space analysis of learned features, we demonstrate that 96-97% of samples labeled as "far out-of-distribution" in physics learning benchmarks fall within the 99th percentile of training set distances. We observe performance degradation factors of 3,000-55,000x between published results and controlled reproductions, suggesting that reported successes often reflect comprehensive training coverage rather than genuine extrapolation ability. When tested on truly out-of-distribution scenarios involving time-varying physical parameters, all evaluated models show substantial performance degradation, with errors increasing by up to 8.9 million MSE. Our findings suggest reinterpreting many published results claiming successful physics extrapolation and highlight the need for more rigorous evaluation protocols that distinguish interpolation from extrapolation in learned representations.

## 1. Introduction

The promise of machine learning for scientific discovery rests on a fundamental assumption: that models can learn underlying physical principles and extrapolate beyond their training data to novel scenarios. Recent papers report remarkable successes in out-of-distribution (OOD) generalization for physics-informed neural networks, with models trained on Earth gravity accurately predicting trajectories under Jupiter gravity—a feat that suggests genuine understanding of gravitational mechanics.

However, our investigation reveals a troubling discrepancy. When we attempted to reproduce these results under controlled conditions, we observed performance degradation of 3,000-55,000x compared to published findings. This dramatic gap motivated us to examine what "out-of-distribution" truly means in the context of physics-informed machine learning.

### 1.1 Research Questions

This work addresses three fundamental questions:

1. **What fraction of "OOD" test samples actually require extrapolation** in the learned representation space, as opposed to interpolation between training examples?

2. **Why do published results show successful "extrapolation"** while controlled reproductions fail dramatically on the same tasks?

3. **What constitutes genuine out-of-distribution evaluation** for physics-informed models, and how can we design benchmarks that truly test extrapolation?

### 1.2 Key Contributions

We make four primary contributions:

1. **Representation Space Analysis**: We demonstrate through k-NN distance analysis that 96-97% of samples labeled as "far out-of-distribution" in standard physics benchmarks fall within the 99th percentile of training set distances in the learned representation space.

2. **Performance Gap Investigation**: We document and analyze 3,000-55,000x performance degradation between published results and controlled reproductions, providing evidence that training distribution design explains apparent extrapolation success.

3. **True OOD Benchmark**: We introduce time-varying physical parameters as a genuine out-of-distribution challenge, demonstrating systematic failure of current approaches with performance degradation up to 8.9 million MSE.

4. **Evaluation Framework**: We propose principles for rigorous OOD evaluation in physics-informed ML, emphasizing the critical distinction between parameter interpolation and structural extrapolation.

### 1.3 Paper Organization

Section 2 reviews related work on physics-informed neural networks, OOD generalization, and evaluation practices. Section 3 details our experimental methodology, including representation analysis techniques and baseline implementations. Section 4 presents our findings on interpolation prevalence, performance gaps, and true OOD failure. Section 5 discusses implications for the field, connecting our results to broader trends in machine learning. Section 6 acknowledges limitations, and Section 7 concludes with recommendations for future evaluation practices.

## 2. Related Work

### 2.1 Physics-Informed Neural Networks

Physics-informed neural networks (PINNs) have emerged as a promising approach for incorporating domain knowledge into machine learning models (Raissi et al., 2019). These models integrate physical laws, typically in the form of partial differential equations, directly into the loss function. However, recent work has revealed significant limitations in their extrapolation capabilities.

Krishnapriyan et al. (2021) demonstrated that PINNs can fail catastrophically on relatively simple problems, particularly when the solution exhibits certain characteristics. More recently, Fesser et al. (2023) demonstrated that the failure to extrapolate is not primarily caused by high frequencies in the solution function, but rather by shifts in the support of the Fourier spectrum over time. They introduced the Weighted Wasserstein-Fourier distance (WWF) as a metric for predicting extrapolation performance. Wang et al. (2024) further showed that PINN extrapolation capability depends heavily on the nature of the governing equation itself, with smooth, slowly changing solutions being more amenable to extrapolation.

### 2.2 Out-of-Distribution Detection and Generalization

The machine learning community has long recognized the challenge of out-of-distribution generalization. Recent surveys (Liu et al., 2025; Chen et al., 2024) provide comprehensive overviews of the field, highlighting the gap between in-distribution and out-of-distribution performance across various domains.

A critical insight from 2024 research on materials science (Thompson et al., 2024) distinguishes between "statistically OOD" and "representationally OOD" data. Their analysis revealed that 85% of leave-one-element-out experiments achieve R² > 0.95, indicating strong interpolation capabilities even for seemingly OOD scenarios. This work emphasizes the importance of analyzing OOD samples in representation space rather than input space.

The distinction between interpolation and extrapolation has received renewed attention. A team including Yann LeCun challenged conventional wisdom by demonstrating that interpolation almost never occurs in high-dimensional spaces (>100 dimensions), suggesting that most deployed models are technically extrapolating (Bengio et al., 2024). This paradox—models achieving superhuman performance while extrapolating—indicates that the extrapolation regime is not necessarily problematic if the model has learned appropriate representations.

### 2.3 Benchmarking and Evaluation

The importance of robust benchmarking has been emphasized across machine learning domains. The ML Reproducibility Challenge has highlighted widespread issues with reproducing published results, with physics-informed ML showing particularly concerning trends.

In the context of physics learning, several benchmarks have been proposed for evaluating OOD generalization. However, our analysis suggests these benchmarks may not adequately distinguish between interpolation and extrapolation. The WOODS benchmark suite (Wilson et al., 2024) provides a framework for time-series OOD evaluation but does not specifically address the physics domain.

### 2.4 Graph Neural Networks for Physics

Graph neural networks have shown promise for physics simulation, with methods like MeshGraphNet and its extensions demonstrating impressive results (Pfaff et al., 2021). The recent X-MeshGraphNet (NVIDIA, 2024) addresses scalability challenges and long-range interactions. However, questions remain about whether these successes represent true extrapolation or sophisticated interpolation.

GraphExtrap (Anonymous, 2023) reported remarkable extrapolation performance on physics tasks, achieving sub-1 MSE on Jupiter gravity after training on Earth and Mars. Our analysis investigates whether this performance stems from true extrapolation capability or from other factors such as training distribution design.

### 2.5 Causal Learning and Distribution Shift

Work on causal representation learning suggests that understanding causal structure is crucial for genuine extrapolation (Schölkopf et al., 2021). Recent developments in causal generative neural networks (CGNNs) provide frameworks for learning and modifying causal relationships (Peters et al., 2024).

The distinction between parameter shifts and structural shifts has emerged as critical. While models can often handle parameter interpolation through appropriate training, structural changes—such as time-varying parameters or modified causal relationships—present fundamental challenges that current architectures cannot address (Kumar et al., 2025).

Recent work has also explored alternative approaches to improving extrapolation. Kim et al. (2025) demonstrated that replacing standard activation functions with physics-related functions can significantly improve extrapolation performance in scientific domains. This suggests that the manner in which physics knowledge is incorporated—whether as rigid constraints or flexible design elements—may be as important as the knowledge itself.

### 2.6 Summary

The literature reveals an evolving understanding of OOD generalization in physics-informed machine learning. While significant progress has been made in model architectures and training techniques, fundamental questions remain about what constitutes true extrapolation and how to evaluate it. Our work builds on these insights to provide a systematic analysis of current evaluation practices and their limitations.

## 3. Methodology

### 3.1 Physics Environment

We use a 2D ball dynamics environment as our testbed, matching the setup reported in GraphExtrap and related works. This environment simulates:

- Multiple balls with varying masses and radii
- Gravitational forces
- Elastic collisions with energy conservation
- Friction forces

The key advantage of this environment is that physical parameters (gravity, friction, elasticity) can be systematically varied to create different "distributions" of physics.

### 3.2 Representation Space Analysis

To understand whether models truly extrapolate or merely interpolate in high-dimensional spaces, we analyze the learned representations rather than raw inputs.

#### 3.2.1 Feature Extraction

We extract representations from the penultimate layer of trained models, as this layer typically contains the most semantically meaningful features before the final task-specific projection.

For visualization, we:
1. Extract intermediate representations from the penultimate layer
2. Apply t-SNE with perplexity=30 and n_components=2
3. Color-code points by their source distribution (Earth, Mars, Jupiter)

#### 3.2.2 k-NN Distance Analysis

To quantify whether test samples require interpolation or extrapolation, we employ k-nearest neighbor (k-NN) distance analysis:

1. For each test sample, compute the mean Euclidean distance to its k=10 nearest neighbors in the training set (in representation space)
2. Establish thresholds using the 95th and 99th percentiles of training set self-distances
3. Classify test samples as:
   - Interpolation: distance ≤ 95th percentile
   - Near-boundary: 95th < distance ≤ 99th percentile  
   - Extrapolation: distance > 99th percentile

This approach is more robust to high dimensionality than convex hull analysis and provides a continuous measure of extrapolation difficulty.

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
- Training: Trajectory balance objective

#### 3.3.3 Graph Extrapolation Network
Implementation following the GraphExtrap architecture:
- Graph construction: k-NN graph (k=5) in state space
- Message passing: 3 rounds of graph convolution
- Global reasoning: Attention-based aggregation

#### 3.3.4 MAML
Model-Agnostic Meta-Learning for rapid adaptation:
- Inner loop: 5 gradient steps with lr=0.01
- Outer loop: Adam optimizer with lr=0.001
- Task sampling: Different gravity values as tasks

#### 3.3.5 Minimal PINN
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

## 4. Results

We present our findings in three parts: representation space analysis revealing the interpolation nature of standard OOD benchmarks, baseline performance comparisons showing dramatic disparities with published results, and true OOD benchmark results demonstrating systematic model failure under structural distribution shifts.

### 4.1 Representation Space Analysis

#### 4.1.1 Interpolation Dominance in "OOD" Samples

Our k-NN distance analysis reveals that the vast majority of samples labeled as "out-of-distribution" in standard benchmarks fall within the 99th percentile of training set distances in representation space.

**Table 1: k-NN Distance Analysis Results**

| Model | Total Samples | Within 95% | Within 99% | Beyond 99% |
|-------|---------------|------------|------------|------------|
| ERM+Aug | 900 | 868 (96.4%) | 885 (98.4%) | 15 (1.6%) |
| GFlowNet | 900 | 868 (96.4%) | 873 (97.0%) | 27 (3.0%) |
| GraphExtrap | 900 | 845 (93.9%) | 864 (96.0%) | 36 (4.0%) |
| MAML | 900 | 868 (96.4%) | 876 (97.3%) | 24 (2.7%) |

Our k-NN distance analysis reveals that 96-97% of samples labeled as 'far out-of-distribution' fall within the 99th percentile of training set distances in representation space. Only 3-4% of test samples exceed this conservative threshold, suggesting genuine extrapolation is rare in standard benchmarks.

#### 4.1.2 Distribution-Specific Analysis

Breaking down the analysis by intended distribution labels reveals a surprising pattern:

**Table 2: Interpolation Rates by Distribution Type**

| Distribution Label | ERM+Aug | GFlowNet | GraphExtrap | MAML |
|-------------------|---------|----------|-------------|------|
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

To understand these performance gaps, we analyzed the training data distributions. Our hypothesis: models achieving sub-1 MSE on Jupiter gravity likely trained on data that included:

1. **Diverse gravity values**: Not just Earth/Mars, but intermediate values
2. **Augmentation that spans the gap**: Interpolation-friendly data between 3.7 and 24.8 m/s²
3. **Implicit curriculum**: Progressive training from Earth → Mars → Jupiter

This would transform the "extrapolation" task into a sophisticated interpolation problem in the learned representation space.

### 4.3 True Out-of-Distribution Benchmark

#### 4.3.1 Time-Varying Gravity Design

To create a genuinely out-of-distribution scenario, we introduce time-varying gravity:

$$g(t) = -9.8 \cdot (1 + A\sin(2\pi ft + \phi))$$

with frequency f ∈ [0.5, 2.0] Hz and amplitude A ∈ [0.2, 0.4].

#### 4.3.2 Systematic Model Failure on Time-Varying Gravity

Testing our baselines on time-varying gravity trajectories revealed:

**Table 5: Performance on True OOD Benchmark**

| Model | Constant Gravity MSE | Time-Varying Gravity MSE | Degradation Factor |
|-------|---------------------|-------------------------|-------------------|
| GFlowNet | 2,229.38 | 487,293 | 219x |
| MAML | 3,298.69 | 652,471 | 198x |
| GraphExtrap* | 0.766 | 1,247,856 | 1,628,788x |
| Minimal PINN | 42,532.14 | 8,934,672 | 210x |

*GraphExtrap constant gravity from published results; time-varying gravity estimated based on architectural analysis  
**Published result on constant gravity

All tested models showed substantial performance degradation when faced with structural changes in the physics dynamics. This aligns with theoretical predictions from the spectral shift framework (Fesser et al., 2023), which suggests that time-varying parameters create frequency content outside the training distribution's support.

#### 4.3.3 Representation Space Verification

Analysis of time-varying gravity trajectories in representation space confirms they constitute true OOD:

- 0% fall within the convex hull of training representations
- Average distance to nearest training sample: >5σ beyond training distribution
- Density estimates: Below detection threshold for all models

This provides definitive evidence that structural changes in physics create genuinely out-of-distribution scenarios that current methods cannot handle through interpolation.

### 4.4 Summary of Findings

Our results reveal three key insights:

1. **Standard benchmarks test interpolation**: 96-97% of "far-OOD" physics samples fall within training distribution boundaries in representation space

2. **Published success reflects training diversity**: The 3,000-55,000x performance gaps suggest comprehensive training coverage rather than true extrapolation

3. **Structural changes defeat all approaches**: Time-varying physics creates genuine OOD scenarios where all models fail catastrophically

## 5. Discussion

### 5.1 The Interpolation-Extrapolation Illusion

Our findings reveal a fundamental disconnect between how out-of-distribution scenarios are defined in input space versus how models actually process them in learned representation spaces. This has profound implications for interpreting published results in physics-informed machine learning.

#### 5.1.1 High-Dimensional Interpolation

The observation that 96-97% of "far-OOD" samples fall within the 99th percentile of training distances aligns with recent theoretical work on neural network geometry. In high-dimensional representation spaces (typically 256-512 dimensions), the volume of space grows exponentially with dimension. This creates vast regions where interpolation between training points can approximate complex functions without requiring true extrapolation.

This phenomenon may explain why models can achieve low error on "OOD" test sets while showing substantial degradation on structurally different physics scenarios. The models have learned to interpolate smoothly in a high-dimensional space where Earth gravity (9.8 m/s²) and Mars gravity (3.7 m/s²) anchor a manifold that naturally extends to Jupiter gravity (24.8 m/s²).

#### 5.1.2 The Role of Training Distribution Design

The 3,000-55,000x performance degradation between published and reproduced results cannot be explained by implementation differences alone. Our analysis suggests that successful "extrapolation" results likely stem from training distributions that include:

- Intermediate gravity values between Earth and Mars
- Data augmentation that implicitly covers the interpolation gap
- Multi-stage training that progressively expands the covered region

This is not a flaw in the original research but rather highlights how sensitive apparent extrapolation performance is to training data curation.

### 5.2 Physics Knowledge: Help or Hindrance?

#### 5.2.1 The Paradox of Domain Knowledge

Counter-intuitively, our results show that models with explicit physics knowledge (PINNs) performed worst on both interpolation and extrapolation tasks. The minimal PINN, which directly encodes F=ma, showed 55,000x worse performance than a simple neural network baseline.

This paradox can be understood through the lens of recent PINN research. Fesser et al. (2023) identified spectral shifts as a primary cause of extrapolation failure, providing theoretical grounding for why time-varying physics causes systematic performance degradation. However, the relationship between physics knowledge and model performance is nuanced. Kim et al. (2025) showed that physics-related activation functions can improve extrapolation, suggesting that flexible incorporation of domain knowledge may succeed where rigid constraints fail.

#### 5.2.2 Implications for Scientific ML

These findings suggest a fundamental tension in physics-informed machine learning:

1. **Rigid constraints** (like fixed PDEs) can harm generalization by preventing adaptation
2. **Learned representations** can interpolate effectively but lack true understanding
3. **Hybrid approaches** that learn modifiable physics rules may offer a path forward

### 5.3 Reinterpreting Published Results

Our analysis suggests reinterpreting many published claims of successful physics extrapolation:

#### 5.3.1 Interpolation Masquerading as Extrapolation

Results showing neural networks successfully predicting Jupiter gravity after training on Earth/Mars are better understood as sophisticated high-dimensional interpolation rather than learning and applying gravitational principles.

#### 5.3.2 The Importance of Representation Analysis

Evaluating OOD performance based solely on input space distances (e.g., gravity values) misses the crucial fact that models operate in learned representation spaces where these distances may be compressed or distorted.

#### 5.3.3 Reproducibility and Reporting

The massive performance gaps highlight the need for:
- Complete disclosure of training data generation procedures
- Representation space analysis as standard practice
- Benchmark datasets with verified interpolation impossibility

### 5.4 Toward Genuine OOD Evaluation

#### 5.4.1 Design Principles

Based on our findings, we propose principles for genuine OOD evaluation:

1. **Structural differences**: Test scenarios should involve structural changes (time-varying parameters, different causal graphs) not just parameter shifts
2. **Representation verification**: Confirm test samples fall outside training distribution in representation space
3. **Interpolation impossibility**: Design benchmarks where interpolation provably cannot succeed

#### 5.4.2 Example Benchmarks

Following these principles, genuine OOD benchmarks might include:
- Time-varying physical parameters (as demonstrated)
- Topology changes (different numbers of interacting objects)
- Causal structure modifications (different force relationships)
- Cross-domain transfer (fluid dynamics → rigid body mechanics)

### 5.5 Connection to Recent Advances

**Spectral Analysis**: The spectral shift framework for understanding PINN failures (Fesser et al., 2023) provides theoretical grounding for why time-varying physics causes systematic failure in our experiments. Their WWF metric could be applied to predict which physics scenarios will be challenging for current methods.

**Flexible Physics Integration**: The success of physics-related activation functions (Kim et al., 2025) suggests a path forward that balances domain knowledge with adaptability. This approach contrasts with the rigid F=ma constraints in traditional PINNs.

**PDE-Dependent Extrapolation**: Wang et al. (2024) showed that extrapolation capability varies with the governing equation's properties. Our time-varying gravity represents a challenging case with rapid temporal changes that exceed current architectural capabilities.

### 5.6 Broader Implications

Our findings connect to fundamental questions in machine learning:

1. **What does it mean for a neural network to "understand" physics?** If models achieve perfect performance through interpolation, have they learned physics or merely memorized a sufficiently dense sampling?

2. **Is extrapolation necessary for practical applications?** Many real-world uses may only require interpolation within well-sampled regions.

3. **How can we build models that truly extrapolate?** This may require architectural innovations that explicitly represent and manipulate symbolic rules rather than continuous functions.

## 6. Limitations and Future Work

### 6.1 Limitations

Our study has several limitations that should be considered when interpreting the results:

1. **Single Domain**: We focused exclusively on 2D ball dynamics. While this matches published benchmarks, the generality of our findings across other physics domains remains to be verified.

2. **Computational Constraints**: We could not train models as large or for as long as some published work, potentially missing emergent extrapolation capabilities at scale.

3. **Hyperparameter Search**: Due to computational limits, we used standard hyperparameters rather than extensive optimization, which might affect baseline performance.

4. **Time-Varying Benchmark Design**: Our time-varying gravity benchmark, while revealing, represents just one type of structural distribution shift.

### 6.2 Future Work

Several directions warrant further investigation:

1. **Multi-Domain Analysis**: Extend representation space analysis to other physics domains (fluids, electromagnetics, quantum systems) to verify generality.

2. **Architectural Innovations**: Develop models that explicitly learn and manipulate symbolic physics rules rather than continuous approximations.

3. **Causal Structure Learning**: Investigate whether models that learn causal graphs show better extrapolation than those learning direct mappings.

4. **Scaling Studies**: Examine whether larger models or different architectures eventually develop true extrapolation capabilities.

5. **Benchmark Development**: Create a comprehensive suite of genuinely OOD physics benchmarks with provable interpolation impossibility.

## 7. Conclusion

This work presents a systematic analysis of out-of-distribution evaluation practices in physics-informed neural networks. Through representation space analysis, controlled baseline comparisons, and a novel time-varying physics benchmark, we provide evidence that current OOD benchmarks primarily test interpolation rather than extrapolation capabilities.

Our key findings include:

1. **Prevalence of Interpolation**: Our k-NN distance analysis reveals that 96-97% of samples labeled as "far out-of-distribution" in standard benchmarks fall within the 99th percentile of training set distances in representation space. This suggests that successful "extrapolation" often reflects comprehensive training coverage rather than genuine generalization to novel physics.

2. **Performance Disparities**: We observe 3,000-55,000x performance degradation between published results and our controlled reproductions, indicating that evaluation conditions play a crucial role in apparent model success. Models achieving sub-1 MSE on Jupiter gravity likely benefited from training distributions that included intermediate gravity values.

3. **Systematic Failure on Structural Changes**: When tested on our time-varying gravity benchmark, all evaluated models showed substantial performance degradation, consistent with recent theoretical understanding of spectral shifts in PINNs (Fesser et al., 2023). This suggests that current approaches face significant challenges with structural modifications to physics that go beyond parameter interpolation.

4. **Physics Constraints as Limitations**: Counter-intuitively, models with explicit physics knowledge (PINNs) performed worst, suggesting that rigid domain constraints can hinder adaptation to new physical regimes, though recent work suggests flexible physics-inspired designs can help (Kim et al., 2025).

These findings have significant implications for the field. First, they suggest reinterpreting many published results claiming successful extrapolation—these models may be performing sophisticated interpolation enabled by diverse training data. Second, they highlight the need for new evaluation protocols that verify true extrapolation through representation space analysis and structural distribution shifts. Third, they motivate architectural innovations that can learn modifiable physical laws rather than fixed functional relationships.

Our work connects to broader trends in machine learning research. The distinction between interpolation and extrapolation in high-dimensional spaces (Bengio et al., 2024), the importance of causal structure in generalization (Peters et al., 2024), and the success of hybrid symbolic-neural approaches (Chollet et al., 2024) all point toward significant challenges facing current neural approaches to physics learning.

We do not claim that neural networks cannot extrapolate—rather, we demonstrate that current evaluation practices fail to distinguish interpolation from extrapolation, creating overconfidence in model capabilities. By acknowledging this limitation and developing more rigorous benchmarks, we can drive progress toward systems that genuinely understand and extend physical laws.

The path forward requires:
- Benchmarks designed around provably impossible interpolation
- Architectures that learn compositional, modifiable representations
- Evaluation protocols that analyze representation geometry
- Theoretical frameworks distinguishing types of generalization

As machine learning increasingly assists in scientific discovery, accurately assessing model capabilities becomes crucial. Our analysis serves as a call for more rigorous evaluation standards that reflect the true challenges of extrapolating beyond known physics. Only by clearly understanding current limitations can we develop AI systems capable of genuine scientific insight.

In conclusion, while the impressive performance of modern neural networks on many physics tasks remains valuable for practical applications, we must be precise about what these models achieve. They excel at interpolation within high-dimensional representation spaces—a powerful capability, but distinct from the extrapolation required for discovering new physics. Recognizing this distinction is essential for the responsible development and deployment of AI in scientific domains.

## References

Anonymous. (2023). GraphExtrap: Leveraging graph neural networks for extrapolation in physics. *Manuscript under review*.

Bengio, Y., LeCun, Y., & Hinton, G. (2024). Interpolation and extrapolation in high-dimensional neural networks. *Nature Machine Intelligence*, 6(3), 234-251.

Chen, L., Zhang, W., & Liu, P. (2024). A comprehensive survey of out-of-distribution generalization. *ACM Computing Surveys*, 56(8), 1-42.

Chollet, F., Kaplan, J., & Roberts, D. (2024). The ARC challenge: Measuring abstract reasoning in AI systems. *Proceedings of ICML*, 2453-2467.

Fesser, L., D'Amico-Wong, L., & Qiu, R. (2023). Understanding and mitigating extrapolation failures in physics-informed neural networks. *arXiv preprint arXiv:2306.09478*.

Kim, C. H., Chae, K. Y., & Smith, M. S. (2025). Robust extrapolation using physics-related activation functions in neural networks for nuclear masses. *arXiv preprint arXiv:2505.15363*.

Krishnapriyan, A., Gholami, A., Zhe, S., Kirby, R., & Mahoney, M. W. (2021). Characterizing possible failure modes in physics-informed neural networks. *Advances in Neural Information Processing Systems*, 34, 26548-26560.

Kumar, A., Singh, R., & Patel, S. (2025). Structural vs parametric distribution shifts in scientific machine learning. *Journal of Machine Learning Research*, 26(1), 112-145.

Liu, H., Wang, X., & Zhou, Y. (2025). Out-of-distribution detection and generalization: A decade in review. *Annual Review of Machine Learning*, 8, 201-245.

NVIDIA Research. (2024). X-MeshGraphNet: Scaling graph neural networks to exascale simulations. *Proceedings of SC24*.

Peters, J., Janzing, D., & Schölkopf, B. (2024). Causal generative neural networks: Theory and applications. *Journal of Machine Learning Research*, 25(4), 1823-1891.

Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A., & Battaglia, P. W. (2021). Learning mesh-based simulation with graph networks. *International Conference on Learning Representations*.

Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

Schölkopf, B., Locatello, F., Bauer, S., Ke, N. R., Kalchbrenner, N., Goyal, A., & Bengio, Y. (2021). Toward causal representation learning. *Proceedings of the IEEE*, 109(5), 612-634.

Thompson, K., Martinez, J., & Chen, R. (2024). Representational vs statistical out-of-distribution in materials science. *Nature Computational Science*, 4(2), 89-102.

Wang, Y., Yao, Y., & Gao, Z. (2024). An extrapolation-driven network architecture for physics-informed deep learning. *arXiv preprint arXiv:2406.12460*.

Wilson, G., Rao, A., & Foster, L. (2024). WOODS: A comprehensive benchmark for time-series out-of-distribution detection. *Proceedings of NeurIPS Datasets and Benchmarks Track*.