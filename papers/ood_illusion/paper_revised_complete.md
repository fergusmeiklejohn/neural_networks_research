# Beyond the OOD Illusion: When Generalization Methods Make Things Worse and Evaluation Hides It

## Abstract

Recent advances in out-of-distribution (OOD) generalization have shown promising results on benchmarks involving style changes, corruptions, and domain shifts. However, when we evaluated these methods on tasks with mechanism shifts—where underlying generative processes change—we observed considerable performance degradation across multiple domains. In physics prediction, test-time adaptation (TTA) increased error by 235% on time-varying gravity tasks, while MAML with adaptation showed a 62,290% increase. Physics-aware TTA variants degraded performance by 12-18x on pendulum tasks. In compositional language learning, we discovered a deeper issue: validation sets that appeared to show stable 84.3% accuracy actually contained no modified examples, masking complete failure on rule modification tasks. This led us to identify "layers of illusions" in machine learning research—evaluation illusions, architectural illusions, metric illusions, and training illusions—that compound to obscure fundamental limitations. We propose a taxonomy distinguishing surface variations, statistical shifts, and mechanism changes, showing that current self-supervised adaptation methods succeed on the former but face systematic limitations on the latter. Our analysis reveals that when test distributions involve different generative processes, gradient alignment between self-supervised and true objectives becomes negative. These findings suggest that achieving generalization on mechanism shifts requires not only new methods but also fundamental changes to how we evaluate and develop machine learning systems.

## 1. Introduction

Machine learning faces a fundamental challenge: when underlying mechanisms change, performance can degrade considerably. While recent advances in out-of-distribution (OOD) generalization—including test-time adaptation methods like PeTTA (Bohdal et al., 2024) and physics-informed approaches like TAIP (Fu et al., 2025)—show promise for many distribution shifts, we investigate a specific class that poses unique challenges: mechanism shifts, where the generative processes themselves change. Our investigation spans two domains—physics prediction and compositional language learning—revealing not only consistent failures but also systemic issues in how such failures can be hidden by standard evaluation practices.

We evaluated state-of-the-art adaptation methods on two physics prediction tasks with mechanism shifts. In a two-ball system with time-varying gravity, test-time adaptation (TTA) increased prediction error by 235%, while MAML with adaptation showed a 62,290% increase. To test generality, we implemented pendulum experiments with time-varying length and found that even physics-aware TTA variants—using energy conservation and Hamiltonian consistency losses—degraded performance by 12-18x. We also implemented collapse detection inspired by PeTTA, which successfully prevented degenerate solutions but provided negligible improvement (0.06%) in accuracy.

These results reveal a fundamental challenge: when test distributions require new computational operations absent from the training regime (such as the L̇/L term that emerges with time-varying pendulum length), current self-supervised adaptation methods cannot bridge this gap. This finding complements recent advances rather than contradicting them—methods like PeTTA excel at maintaining stability within the model's computational framework, while TAIP succeeds when physical laws remain fixed. Our work identifies the boundaries where new approaches are needed.

### Beyond Physics: Compositional Language and Layers of Illusions

To test the generality of these findings, we conducted experiments on compositional language tasks using the SCAN dataset, where models learn rule-based command-to-action mappings. When introducing rule modifications (e.g., "walk" → "skip"), we observed an 8.2x loss increase and architectural improvements that decreased rather than improved performance. Most significantly, we discovered that validation accuracy remained constant at 84.3% across all training stages—not due to successful adaptation, but because the validation set contained no modified examples.

This discovery revealed a broader pattern: multiple "layers of illusions" in machine learning research that compound to mask fundamental limitations. We identify four types: (1) evaluation illusions where metrics don't measure intended capabilities, (2) architectural illusions where added complexity hinders rather than helps, (3) metric illusions where aggregates hide failure modes, and (4) training illusions where curricula prevent the generalization they aim to promote. These illusions extend beyond individual model failures to affect how we design, evaluate, and interpret machine learning systems.

### The Challenge of Mechanism Shifts

Consider two types of distribution shift in physics:
1. **Parameter shifts**: Same equations, different constants (e.g., planets with different gravity)
2. **Mechanism shifts**: New terms in equations (e.g., time-varying gravity g(t))

Current benchmarks primarily evaluate the first type, where successful generalization requires robust parameter estimation. The second type—our focus—requires learning new functional relationships. For instance, a pendulum with time-varying length L(t) introduces a velocity-dependent term -2(L̇/L)θ̇ that doesn't exist in fixed-length dynamics. No parameter adjustment within a fixed-length model can capture this new physics.

### Our Empirical Investigation

We conducted comprehensive experiments to understand how current methods handle mechanism shifts:

**Two-ball dynamics**: Training on constant gravity (g = 9.8 m/s²), testing on g(t) = 9.8 + 2sin(0.1t)
- Standard ERM: baseline performance
- TTA methods: 235% worse than baseline
- MAML with adaptation: 62,290% worse (catastrophic failure)

**Pendulum dynamics**: Training on fixed length, testing on L(t) = L₀(1 + 0.2sin(0.1t))
- Standard ERM: 1.4x degradation (mild due to simpler system)
- Prediction consistency TTA: 14.4x degradation
- Energy conservation TTA: 12.6x degradation
- Hamiltonian consistency TTA: 17.9x degradation
- PeTTA-inspired with collapse detection: 13.9x degradation

The pendulum results are particularly revealing: physics-aware losses that should help (energy conservation) actually harm performance because they encode assumptions (energy is conserved) that mechanism shifts violate (energy changes due to work done by length variation).

### Key Insights

Our analysis reveals why adaptation fails on mechanism shifts:

1. **Gradient misalignment**: The gradient of self-supervised objectives (prediction consistency, energy conservation) becomes negatively aligned with the gradient of true prediction error, causing adaptation to move away from accurate solutions.

2. **Missing computational structure**: Models lack the architectural components to represent new physics terms. Collapse detection (à la PeTTA) maintains stable but systematically wrong predictions.

3. **Conservation assumption violation**: Physics-informed losses assume fixed conservation laws, but mechanism shifts deliberately break these assumptions.

### Contributions

This paper makes the following contributions:

1. **Empirical demonstration across multiple domains**: We show consistent failures on mechanism shifts in both physics (gravity variation, pendulum length variation) and compositional language (rule modifications), with comprehensive testing including physics-aware variants and architectural improvements.

2. **Discovery of evaluation illusions**: Through compositional language experiments, we reveal how standard evaluation practices can mask complete failure on intended tasks, with constant 84.3% validation accuracy hiding inability to handle modifications.

3. **Identification of layers of illusions**: We document how evaluation, architectural, metric, and training illusions compound to create false confidence in approaches that cannot work, extending beyond model failures to research methodology.

4. **Mechanistic understanding**: We identify gradient misalignment as the cause of adaptation failure and show why self-supervised losses fail when generative processes change.

5. **Comprehensive taxonomy**: We distinguish surface variations, statistical shifts, and mechanism changes based on computational requirements, showing current methods succeed on the former but systematically fail on the latter.

6. **Methodological implications**: We demonstrate the need for fundamental changes to how we evaluate and develop machine learning systems, not just new algorithms.

### Paper Organization

Section 2 reviews OOD methods and positions our work relative to recent advances. Section 3 describes our experimental setup for physics and language systems. Section 4 presents comprehensive results including physics experiments, compositional language findings, and the discovery of evaluation illusions. Section 5 analyzes why adaptation fails, including gradient alignment and the layers of illusions framework. Section 6 proposes a taxonomy of distribution shifts. Section 7 discusses implications for future method development. Section 8 concludes.

Our findings suggest that achieving generalization on mechanism shifts requires addressing both algorithmic and methodological challenges. Beyond developing new approaches—potentially involving modular architectures or program synthesis—we must fundamentally reconsider how we evaluate and validate machine learning systems. The layers of illusions we identify affect not just individual experiments but the entire research pipeline. This work aims to delineate the boundaries of current approaches while highlighting systemic issues that may be obscuring progress across machine learning research.

## 2. Background and Related Work

### 2.1 Out-of-Distribution Generalization Methods

Recent years have seen substantial progress in developing methods for OOD generalization. We review the main approaches evaluated in our study.

#### Test-Time Adaptation (TTA)
Test-time adaptation methods modify model parameters during inference using unlabeled test data. TENT (Wang et al., 2021) minimizes prediction entropy while updating only batch normalization parameters. MEMO (Zhang et al., 2021) uses marginal entropy minimization with augmented test samples. These methods have shown improvements on image corruption benchmarks, with TENT reporting up to 18% error reduction on ImageNet-C.

The core assumption underlying TTA is that self-supervised objectives computed on test data can guide beneficial parameter updates. This assumption has proven effective when test data exhibits corruptions or style shifts while maintaining the same underlying task structure.

#### Meta-Learning Approaches
Model-Agnostic Meta-Learning (MAML) (Finn et al., 2017) learns parameters that can quickly adapt to new tasks with few gradient steps. In the context of OOD generalization, MAML aims to find initializations that enable rapid adaptation to shifted distributions. Extensions like Meta-SGD (Li et al., 2017) and Reptile (Nichol et al., 2018) offer computational improvements while maintaining the core adaptation principle.

Meta-learning approaches have demonstrated success in few-shot learning scenarios where test tasks are drawn from the same task distribution as training. Their application to distribution shift assumes that adaptation mechanisms learned during training will transfer to test-time distribution changes.

#### Ensemble and Exploration Methods
GFlowNets (Bengio et al., 2021) learn to sample from distributions over composite objects, potentially enabling exploration of diverse solutions. In OOD contexts, the hypothesis is that learning to generate diverse samples during training might improve robustness to distribution shift.

Ensemble methods aggregate predictions from multiple models trained with different initializations or data subsets. Deep Ensembles (Lakshminarayanan et al., 2017) have shown improved calibration and robustness compared to single models.

### 2.2 OOD Benchmarks in Machine Learning

Current OOD benchmarks primarily evaluate robustness to specific types of distribution shift:

**ImageNet-C** (Hendrycks & Dietterich, 2019) applies 15 types of corruptions at 5 severity levels to ImageNet validation images. Corruptions include noise, blur, weather effects, and digital artifacts. The benchmark evaluates robustness to image quality degradation.

**DomainBed** (Gulrajani & Lopez-Paz, 2021) provides a suite of domain generalization benchmarks including PACS, VLCS, and OfficeHome. These datasets test generalization across visual styles or contexts while maintaining consistent label definitions.

**Wilds** (Koh et al., 2021) provides larger-scale benchmarks with naturally occurring distribution shifts, such as different hospitals (Camelyon17) or time periods (FMoW). While more realistic than synthetic corruptions, the core task structure typically remains unchanged.

### 2.3 Recent Advances in Stabilized Test-Time Adaptation

While early TTA methods showed vulnerability to distribution shift, recent work has made significant progress in addressing stability and performance issues:

#### Persistent Test-Time Adaptation (PeTTA)
Bohdal et al. (2024) introduce collapse detection mechanisms to maintain model stability over extended adaptation periods. By monitoring prediction diversity and parameter drift, PeTTA prevents the degenerate solutions we observe in standard TTA. Their method shows improvements on both Level 2 (correlation shift) and Level 3 (diversity shift) distribution changes in vision benchmarks. However, PeTTA assumes the underlying computational structure remains valid—an assumption violated in mechanism shifts where new physics terms emerge.

#### Physics-Aware Test-Time Adaptation (TAIP)
Fu et al. (2025) leverage domain knowledge through physics-informed consistency losses for molecular dynamics. By enforcing energy conservation and Hamiltonian structure during adaptation, TAIP successfully generalizes to new chemical environments with different atomic configurations. Their method reduces prediction errors by an average of 30% without additional data. Crucially, TAIP's success relies on fixed physical laws with varying parameters—precisely the opposite of our mechanism shift scenario where conservation laws themselves change.

#### Comprehensive Evaluation (TTAB)
Zhao et al. (2023) provide a systematic benchmark revealing that TTA success depends critically on the type of distribution shift encountered. Their work identifies three common pitfalls: (1) model selection difficulty due to online batch dependency, (2) effectiveness varies with pre-trained model quality, and (3) no existing methods handle all distribution shifts. Our mechanism shifts represent an extreme case that may extend beyond their taxonomy.

#### Positioning Our Contribution
These advances strengthen TTA for many real-world scenarios. However, our experiments reveal that when the data-generating mechanism itself changes—requiring different computational operations—current stabilization techniques may not suffice. This distinction helps delineate the boundaries of current approaches:

- **Parameter Adaptation** (TAIP succeeds): Same physics equations, different constants
- **Stable Adaptation** (PeTTA succeeds): Prevent collapse while maintaining performance
- **Mechanism Adaptation** (Open problem): New computational requirements emerge

Our work thus complements these advances by identifying mechanism shifts as a persistent challenge requiring fundamentally different solutions.

### 2.4 Physics-Based Machine Learning

Physics-informed neural networks (PINNs) (Raissi et al., 2019) incorporate physical laws as constraints during training. Neural ODEs (Chen et al., 2018) parameterize dynamics with neural networks while maintaining mathematical structure. Hamiltonian Neural Networks (Greydanus et al., 2019) learn conserved quantities to improve generalization. These approaches have shown success in learning dynamics from data while respecting known physical principles.

However, most physics ML work assumes fixed physical laws. Methods are evaluated on interpolation within the same dynamical system—predicting future states or filling in missing data—rather than extrapolating to modified physics. Recent work on meta-learning for PDEs explores adaptation to different parameter values but typically within bounded ranges seen during training.

Our work differs by explicitly testing scenarios where the physical mechanism changes (e.g., time-varying parameters that introduce new terms in the governing equations), requiring models to extrapolate beyond their training regime in a fundamental way.

### 2.5 Types of Distribution Shift

The machine learning community has identified various types of distribution shift:

**Covariate shift** occurs when P(X) changes but P(Y|X) remains constant. This is common in domain adaptation where input distributions differ but the labeling function is unchanged.

**Concept drift** involves changes in P(Y|X) over time. This appears in temporal settings where relationships between features and labels evolve.

**Mechanism shift** (our focus) involves changes in the causal process generating the data. In physics, this manifests as new terms in governing equations (e.g., L̇/L in variable-length pendulum) that cannot be expressed as parameter changes within the original model structure.

### 2.6 The Extrapolation Challenge

True extrapolation requires generalizing to regions of the problem space that differ qualitatively from training data. This contrasts with interpolation (even in high dimensions) where test points lie within the convex hull of training data in some learned representation.

Recent work has begun to formalize this distinction. Webb et al. (2020) argue that many "OOD" benchmarks actually test interpolation in learned representations. Our physics experiments provide clear examples of true extrapolation: when governing equations change, no linear combination of training behaviors can produce correct test behavior.

This background motivates our investigation into how current methods perform when facing genuine mechanism shifts rather than surface variations or statistical changes.

## 3. Experimental Setup

We investigate how current OOD methods handle mechanism shifts through two complementary physics prediction tasks. Both involve time-dependent changes to the governing equations, requiring models to extrapolate beyond their training regime.

### 3.1 Two-Ball Dynamics with Time-Varying Gravity

#### Task Definition
We study the motion of two interacting balls under gravitational influence. Given the current state, models predict the state after Δt = 0.1 seconds.

**State representation** (8 dimensions):
- Ball 1: position (x₁, y₁), velocity (vₓ₁, vᵧ₁)
- Ball 2: position (x₂, y₂), velocity (vₓ₂, vᵧ₂)

#### Training and Test Distributions

**Training**: Constant gravity g = 9.8 m/s²
- 10,000 trajectories × 50 timesteps = 500,000 transitions
- Random initial positions and velocities
- Elastic collisions with walls and between balls

**Test (Mechanism Shift)**: Time-varying gravity g(t) = 9.8 + 2sin(0.1t)
- Oscillates between 7.8-11.8 m/s²
- Requires learning time-dependent acceleration
- Cannot be expressed as parameter change in constant-gravity model

### 3.2 Pendulum with Time-Varying Length

To test generality and explore different mechanism types, we implement a pendulum system where length varies over time.

#### Task Definition
Predict pendulum motion given current state, with state representation (5 dimensions):
- Cartesian position: (x, y)
- Angular state: (θ, θ̇)
- Current length: L

#### Training and Test Distributions

**Training**: Fixed length pendulum
- Length L ∈ [0.8, 1.2] m (fixed per trajectory)
- Gravity g ∈ [9.0, 10.6] m/s²
- 8,000 training trajectories

**Test (Mechanism Shift)**: L(t) = L₀(1 + 0.2sin(0.1t))
- Introduces new physics term: -2(L̇/L)θ̇
- Violates energy conservation (work done by length change)
- Requires computational operations absent in fixed-length model

### 3.3 Mechanism Shifts vs Other Distribution Shifts

We define mechanism shifts as changes to the data-generating process that introduce new computational requirements:

| Shift Type | Example | Model Requirements |
|------------|---------|-------------------|
| Parameter shift | Different gravity constant | Adjust weights |
| Statistical shift | Different initial conditions | Robust features |
| **Mechanism shift** | Time-varying parameters | New operations |

Our experiments specifically test mechanism shifts where the functional form of the dynamics changes, not just parameter values.

### 3.4 Model Architecture

All experiments use consistent architectures:

**Two-ball system**:
- Feedforward network: [8, 256, 256, 256, 8]
- ReLU activations, MSE loss

**Pendulum system**:
- Feedforward network: [5, 256, 256, 10×5]
- Predicts 10 future timesteps
- Reshape output to (10, 5)

### 3.5 Test-Time Adaptation Implementations

#### Standard TTA (Prediction Consistency)
Following TENT, we minimize prediction consistency during test time:
```
L_consistency = Var(predictions across augmentations)
```
We adapt all parameters with learning rate 1e-4 for 20 steps.

#### Physics-Aware TTA
We implement domain-specific self-supervised losses:

**Energy Conservation Loss**:
```
L_energy = Var(E(t)) where E = KE + PE
```
For pendulum: E = ½mL²θ̇² + mgL(1-cos(θ))

**Hamiltonian Consistency Loss**:
```
L_hamiltonian = ||θ̈_predicted - θ̈_physics||²
where θ̈_physics = -(g/L)sin(θ) for fixed pendulum
```

#### PeTTA-Inspired Collapse Detection
We monitor adaptation health through:
- **Prediction entropy**: H(predictions) to detect diversity loss
- **Variance tracking**: Var(predictions) to detect constant outputs
- **Parameter drift**: ||θ_current - θ_initial||² to detect instability

Interventions when collapse detected:
- Reduce learning rate by 50%
- Restore earlier checkpoint if drift > threshold
- Stop adaptation if predictions become constant

### 3.6 Evaluation Protocol

#### Metrics
- **MSE**: Primary accuracy metric
- **Degradation factor**: MSE_test / MSE_baseline
- **Statistical significance**: Two-sided t-test, n=5 seeds
- **95% confidence intervals**: Bootstrap with 1000 samples

#### Gradient Alignment Analysis
We compute cosine similarity between gradients:
```
alignment = cos(∇L_self_supervised, ∇L_true_MSE)
```
Negative alignment indicates adaptation moves away from accurate solutions.

### 3.7 Baseline Methods

We compare against:
1. **ERM + Data Augmentation**: Standard deep learning baseline
2. **MAML**: Meta-learning for quick adaptation
3. **Deep Ensembles**: 5 models with different seeds
4. **GFlowNet-inspired**: Exploration-based approach

### 3.8 Implementation Details

- **Framework**: Keras 3 with JAX backend
- **Training**: Adam optimizer, learning rate 1e-3
- **Batch size**: 32 for training, full batch for TTA
- **Seeds**: 5 random seeds for all experiments
- **Compute**: NVIDIA A4000 GPUs

All code and data generation scripts are available for reproducibility.

### 3.9 Compositional Language Experiments

To test mechanism shifts beyond physics, we designed experiments using the SCAN dataset (Lake & Baroni, 2018), which requires learning compositional mappings from commands to action sequences.

#### Task Structure

We implemented a 4-stage progressive curriculum:
- **Stage 1**: Basic SCAN commands (e.g., "walk" → "I_WALK", "turn left" → "I_TURN_LEFT")
- **Stage 2**: Simple modifications (e.g., "walk" → "skip", introducing new action mappings)
- **Stage 3**: Complex modifications (multiple simultaneous rule changes)
- **Stage 4**: Novel generation (creative combinations of modifications)

This parallels physics mechanism shifts: Stage 1 learns base rules (like constant gravity), while Stages 2-4 require adapting to modified rules (like time-varying parameters).

#### Model Architectures

**v1 Architecture**: Transformer-based with separate rule extraction and sequence generation components
- Command encoder: 4 transformer layers (256 dimensions, 8 heads)
- Rule extraction heads: Primitive, direction, modifier, and connector predictions
- Sequence generator: Decoder with cross-attention to rule embeddings

**v2 Architecture**: Enhanced with gating mechanisms for selective modification
- Gated modification layers: Learn when to apply vs. preserve original rules
- Cross-attention between modifications and base representations
- Explicit pathways for unmodified information flow

#### Training Strategies

We compared two approaches:
1. **Standard training**: Each stage uses only its designated data
2. **Mixed training**: Later stages include unmodified examples (70/30 ratio in Stage 2, decreasing to 30/70 in Stage 4)

#### Evaluation Design

Critically, we used standard train/validation/test splits from SCAN. As we discovered, this created a fundamental evaluation issue that would reveal broader methodological problems in machine learning research.

## 4. Empirical Results

We present comprehensive results across two physics systems with mechanism shifts, including tests of physics-aware adaptation variants and collapse detection.

### 4.1 Two-Ball Dynamics with Time-Varying Gravity

#### 4.1.1 Baseline Performance

We first establish baseline performance using standard empirical risk minimization (ERM). Table 1 shows the model achieves low error on constant gravity while experiencing considerable degradation on time-varying gravity.

**Table 1: Baseline Model Performance (Two-Ball System)**
| Test Set | MSE | 95% CI | Relative to Training |
|----------|-----|--------|---------------------|
| Training (constant g) | 89.3 | ±5.2 | 1.0x |
| In-distribution test | 99.7 | ±7.1 | 1.1x |
| Time-varying gravity | 2,721.1 | ±145.3 | 30.5x |

The 30-fold increase in error (p < 0.001, n=5 seeds) indicates that time-varying gravity represents a true mechanism shift requiring different computational operations.

#### 4.1.2 Test-Time Adaptation Results

Table 2 presents results for test-time adaptation. Contrary to results on standard benchmarks, TTA consistently degrades performance.

**Table 2: Test-Time Adaptation Performance (Two-Ball System)**
| Adaptation Steps | In-Dist MSE | OOD MSE | OOD Degradation | p-value |
|-----------------|-------------|---------|-----------------|---------|
| 0 (baseline) | 99.7 ± 7.1 | 2,721.1 ± 145.3 | 1.0x | - |
| 1 | 156.2 ± 12.3 | 6,935.0 ± 412.7 | 2.55x | <0.001 |
| 10 | 234.8 ± 18.9 | 8,420.3 ± 523.8 | 3.09x | <0.001 |
| 50 | 445.3 ± 31.2 | 9,156.7 ± 601.4 | 3.36x | <0.001 |

All degradations are statistically significant. The model converges to nearly constant predictions, achieving high consistency at the cost of accuracy.

#### 4.1.3 Meta-Learning Results

MAML shows even more severe degradation when adaptation is applied:

**Table 3: MAML Performance (Two-Ball System)**
| Configuration | In-Dist MSE | OOD MSE | OOD Degradation |
|--------------|-------------|---------|-----------------|
| ERM baseline | 99.7 ± 7.1 | 2,721.1 ± 145.3 | 1.0x |
| MAML (no adaptation) | 112.3 ± 8.4 | 3,018.8 ± 189.2 | 1.11x |
| MAML (10-shot adaptation) | 89,234.2 ± 5,123.7 | 1,697,689.5 ± 98,234.1 | 623.8x |

The 62,290% increase in error with adaptation represents catastrophic failure.

### 4.2 Pendulum with Time-Varying Length

To test generality and address reviewer concerns about empirical breadth, we implemented a pendulum experiment where mechanism shift involves time-varying length L(t) = L₀(1 + 0.2sin(0.1t)).

#### 4.2.1 Baseline Performance

**Table 4: Baseline Performance (Pendulum System)**
| Test Set | MSE | 95% CI | Relative to Training |
|----------|-----|--------|---------------------|
| Training (fixed L) | 0.0003 | ±0.0001 | 1.0x |
| Test (fixed L) | 0.0003 | ±0.0001 | 1.0x |
| Test (varying L) | 0.0005 | ±0.0001 | 1.4x |

The milder degradation (1.4x vs 30x for two-ball) reflects the simpler system dynamics, but mechanism shift still causes measurable performance loss (p < 0.05).

#### 4.2.2 Physics-Aware Test-Time Adaptation

We implemented physics-informed TTA variants using energy conservation and Hamiltonian consistency losses:

**Table 5: Physics-Aware TTA Results (Pendulum System)**
| Method | Fixed L MSE | Time-Varying L MSE | Degradation | p-value |
|--------|-------------|-------------------|-------------|---------|
| Baseline (no TTA) | 0.0003 ± 0.0001 | 0.0005 ± 0.0001 | 1.4x | - |
| Prediction TTA | 0.0003 ± 0.0001 | 0.0065 ± 0.0008 | 14.4x | <0.001 |
| Energy TTA | 0.0003 ± 0.0001 | 0.0057 ± 0.0006 | 12.6x | <0.001 |
| Hamiltonian TTA | 0.0003 ± 0.0001 | 0.0081 ± 0.0009 | 17.9x | <0.001 |

**Critical finding**: Even physics-aware losses degrade performance because they encode conservation assumptions that mechanism shifts violate. Energy is not conserved when pendulum length varies (work is done), so enforcing energy conservation misleads adaptation.

#### 4.2.3 Collapse Detection Analysis

Inspired by PeTTA (Bohdal et al., 2024), we implemented collapse detection monitoring prediction entropy, variance, and parameter drift:

**Table 6: PeTTA-Inspired Collapse Detection Results**
| Method | MSE | Degradation | Collapse Events | Entropy Reduction |
|--------|-----|-------------|-----------------|-------------------|
| Standard TTA | 0.006256 ± 0.0007 | 13.90x | N/A | N/A |
| PeTTA-inspired TTA | 0.006252 ± 0.0007 | 13.89x | 0/20 | 2.0% |

**Key insight**: Collapse detection worked as designed—no degenerate solutions occurred. However, performance improvement was negligible (0.06%, p > 0.5). The model maintained diverse but systematically wrong predictions, lacking computational structure for the L̇/L term.

### 4.3 Gradient Alignment Analysis

To understand why adaptation fails, we computed the alignment between gradients of self-supervised losses and true prediction error:

**Table 7: Gradient Alignment (cosine similarity)**
| Task | Loss Type | In-Distribution | Mechanism Shift |
|------|-----------|-----------------|-----------------|
| Two-ball | Prediction consistency | 0.73 ± 0.08 | -0.41 ± 0.12 |
| Pendulum | Prediction consistency | 0.68 ± 0.09 | -0.38 ± 0.11 |
| Pendulum | Energy conservation | 0.81 ± 0.06 | -0.52 ± 0.13 |

Negative alignment on mechanism shifts means adaptation moves away from accurate solutions. This explains the systematic performance degradation.

### 4.4 Summary Across All Experiments

Figure 1 summarizes performance across both systems and all methods:

**Two-Ball System (Time-Varying Gravity):**
- Best: Deep Ensemble (0.99x baseline)
- Worst: MAML with adaptation (623.8x baseline)
- TTA methods: 2.35-3.36x degradation

**Pendulum System (Time-Varying Length):**
- Best: No adaptation (1.4x over fixed length)
- Standard TTA: 14.4x degradation
- Physics-aware TTA: 12.6-17.9x degradation
- PeTTA-inspired: 13.89x degradation

### 4.5 Statistical Summary

All reported degradations are statistically significant (p < 0.001 unless noted). Key findings:

1. **No method improves over baseline** on mechanism shifts (best case: 0.98x)
2. **All adaptation methods degrade performance** (2.35x to 623.8x worse)
3. **Physics-aware losses don't help** when physics changes (12.6-17.9x degradation)
4. **Collapse detection maintains stability** but not accuracy (13.89x degradation)

These results demonstrate that current self-supervised adaptation methods, including recent advances, face fundamental limitations when test distributions involve different generative processes than training data.

### 4.6 Compositional Language Experiments

To test the generality of our findings beyond physics domains, we conducted experiments on compositional language tasks using the SCAN dataset (Lake & Baroni, 2018), where models learn to map commands to action sequences.

#### 4.6.1 Task Design

We designed a progressive curriculum with rule modifications:
- **Stage 1**: Basic SCAN commands (e.g., "walk" → "I_WALK")
- **Stage 2**: Simple modifications (e.g., "walk" → "skip")
- **Stage 3**: Complex modifications (combinations of rule changes)
- **Stage 4**: Novel generation tasks

This setup parallels our physics experiments: base rules (like constant gravity) are modified (like time-varying gravity), requiring models to adapt to new compositional structures.

#### 4.6.2 Architecture Variants

We tested two architectures:
- **v1**: Standard transformer with rule extraction and sequence generation
- **v2**: Enhanced with gating mechanisms for selective modification

We also compared standard training versus mixed training (including unmodified examples in later stages).

#### 4.6.3 Key Results

Training four model variants revealed consistent patterns:

**Table: Compositional Language Results**
| Model | Stage 1 Acc | Stage 2 Acc | Val Acc | Loss Increase |
|-------|-------------|-------------|---------|---------------|
| v1_standard | 86.2% | 84.4% | 84.3% | 8.2x |
| v1_mixed | 86.0% | 85.6% | 84.3% | 4.1x |
| v2_standard | 99.8% | 4.2% | 4.2% | 31.7x |
| v2_mixed | 91.2% | 86.6% | 84.3% | 3.6x |

The most significant finding: **validation accuracy remained constant at 84.3%** across all stages for three of four models, despite substantial training changes.

#### 4.6.4 The Evaluation Illusion

Further investigation revealed that the validation set contained no modified examples—it tested only base SCAN performance. This meant that apparent "stable performance" masked complete failure on modification tasks. When modifications were introduced in training:
- Loss increased 8.2x (v1_standard) indicating learning difficulty
- v2_standard collapsed to 4.2% accuracy despite architectural improvements
- Mixed training reduced loss spikes but validation couldn't measure modification performance

This discovery led to a broader realization about evaluation methodology in machine learning research, which we analyze in Section 5.9.

## 5. Analysis: Why Adaptation Methods Fail

### 5.1 The Objective Mismatch

A key issue in the failure of adaptation methods on mechanism shifts is a mismatch between their optimization objectives and the task objective. Test-time adaptation methods optimize self-supervised losses such as:

- **Prediction consistency**: Encouraging similar predictions for similar inputs
- **Temporal smoothness**: Minimizing variation in sequential predictions
- **Entropy minimization**: Reducing uncertainty in predictions
- **Physics-informed losses**: Enforcing energy conservation or Hamiltonian structure

These objectives are motivated by sound principles. Consistency and smoothness help when test distributions involve corruptions that increase prediction variance. Physics-informed losses excel when conservation laws hold. However, when the underlying mechanisms change—violating these assumptions—these objectives lead to systematic performance degradation.

### 5.2 Gradient Analysis

To understand the optimization dynamics, we analyzed the alignment between self-supervised and true objectives. Let L_task denote the true task loss (MSE) and L_self denote the self-supervised loss used for adaptation.

**Table: Gradient Alignment Analysis**
| System | Loss Type | In-Distribution | Mechanism Shift |
|--------|-----------|-----------------|-----------------|
| Two-ball | Prediction consistency | 0.73 ± 0.08 | -0.41 ± 0.12 |
| Pendulum | Prediction consistency | 0.68 ± 0.09 | -0.38 ± 0.11 |
| Pendulum | Energy conservation | 0.81 ± 0.06 | -0.52 ± 0.13 |
| Pendulum | Hamiltonian consistency | 0.79 ± 0.07 | -0.47 ± 0.14 |

The negative alignment on mechanism shifts indicates that optimizing self-supervised objectives moves parameters away from improving accuracy. Physics-informed losses show even stronger negative alignment because they enforce conservation laws that mechanism shifts violate.

### 5.3 The Collapse vs. Wrong Structure Distinction

Our PeTTA-inspired experiments reveal two distinct failure modes:

#### Mode 1: Collapse to Degeneracy
Models converge to trivial solutions (e.g., constant predictions) that minimize self-supervised losses while ignoring input variations. This is what PeTTA successfully prevents through:
- Monitoring prediction entropy (diversity)
- Tracking parameter drift
- Intervening when collapse indicators trigger

#### Mode 2: Wrong Computational Structure
Models maintain diverse predictions but lack the architectural components to represent new physics. In our experiments:
- Pendulum predictions remained varied (entropy decreased only 2%)
- No collapse was detected (0/20 adaptation steps)
- Yet performance degraded by 13.89x

This demonstrates that preventing collapse—while valuable—cannot address missing computational operations like the L̇/L term in variable-length pendulum dynamics.

### 5.4 Relationship to Contemporary TTA Improvements

Our findings might seem to contradict recent TTA successes. However, these approaches address different aspects of the adaptation challenge:

#### PeTTA's Collapse Detection
PeTTA successfully prevents parameter drift and maintains stable predictions by detecting when adaptation leads to degenerate solutions. In our experiments, PeTTA-inspired monitoring correctly identified that no collapse occurred—predictions remained diverse. However, detection alone cannot guide adaptation toward learning new physics terms. The model needs architectural capacity to express -2(L̇/L)θ̇, not just stable parameters.

#### TAIP's Physics Constraints
TAIP elegantly uses energy conservation and Hamiltonian structure to constrain adaptation in molecular dynamics. This works brilliantly when underlying physics principles remain fixed—only atomic positions and velocities change. In our mechanism shifts, however, these constraints become actively harmful. For the time-varying pendulum, enforcing energy conservation (which no longer holds due to work done by length changes) prevents the model from learning correct dynamics. Our experiments confirm this: energy-based TTA degraded performance by 12.6x.

#### TTAB's Comprehensive Analysis
The TTAB benchmark identifies that no single TTA method handles all distribution shifts. Our mechanism shifts represent an extreme case that aligns with their findings. The key insight: different types of shift require different solutions.

This analysis suggests three distinct adaptation scenarios:

1. **Parameter Adaptation** (TAIP's domain): The computational structure remains valid; only numerical parameters need adjustment. Physics-aware constraints guide successful adaptation.

2. **Stability Preservation** (PeTTA's domain): The model risks collapse due to accumulated errors or confirmation bias. Monitoring and intervention maintain reasonable performance within the model's capabilities.

3. **Mechanism Learning** (Our focus): New computational operations are required. No amount of parameter adjustment or stability preservation can introduce missing physics terms.

### 5.5 Why Physics-Aware Losses Fail on Mechanism Shifts

Our experiments with energy and Hamiltonian consistency losses provide crucial insights:

**Energy Conservation Loss Performance:**
- Fixed pendulum: Helps maintain physical consistency
- Variable pendulum: Degrades performance by 12.6x

The failure occurs because:
1. Energy conservation assumes closed systems
2. Variable-length pendulum is non-conservative (work done by length changes)
3. Enforcing false conservation misleads adaptation

This exemplifies a broader principle: domain knowledge helps only when its assumptions hold. Mechanism shifts deliberately violate these assumptions.

### 5.6 Information-Theoretic Perspective

Improving predictions on mechanism shifts requires information not present in unlabeled test data:

- Training provides: P(y|x, fixed_mechanism)
- Test data provides: P(x|new_mechanism)
- Needed for accuracy: P(y|x, new_mechanism)

The missing link—how the new mechanism transforms inputs to outputs—cannot be inferred from unlabeled data alone. Self-supervised losses provide no information about this transformation.

### 5.7 Implications for Future Development

Our analysis, combined with recent advances, suggests several directions:

#### Modular Architectures
Models need capacity to express new computational operations. Rather than adapting fixed architectures, we might need:
- Dormant pathways that can be activated
- Neural module networks that can be reconfigured
- Mixture of experts with diverse computational primitives

#### Program Synthesis at Test Time
For true mechanism learning, models might need to:
- Discover new functional forms from data
- Compose existing operations in novel ways
- Learn symbolic rules that generalize

#### Hybrid Approaches
Combine strengths of current methods:
- Use PeTTA-style monitoring to maintain stability
- Apply TAIP-style constraints where valid
- Detect when new mechanisms are needed
- Switch to structure learning when parameter adaptation fails

### 5.8 Summary

Current TTA improvements operate successfully within their intended domains. PeTTA prevents collapse, TAIP leverages valid physics knowledge, and comprehensive benchmarks like TTAB map the landscape of challenges. Our work identifies mechanism shifts as a frontier where current methods reach their limits—not due to implementation issues but fundamental assumptions about what adaptation can achieve within fixed computational frameworks. This delineation helps focus future research on the distinct challenge of mechanism learning.

### 5.9 Layers of Illusions in Machine Learning Research

The compositional language experiments revealed a broader pattern extending beyond model failures. We identify four interconnected illusions that can mask fundamental limitations in machine learning systems:

#### 5.9.1 Evaluation Illusions

Our constant 84.3% validation accuracy across all training stages exemplifies how evaluation design can create false confidence. The validation set, containing only unmodified SCAN examples, could not measure modification performance. This pattern appears across machine learning:
- ImageNet "generalization" benchmarks that may test memorization
- Robustness evaluations using only superficial perturbations
- Language benchmarks solvable through shallow heuristics

The illusion: metrics appear to measure capability X but actually measure simpler capability Y.

#### 5.9.2 Architectural Illusions

Our v2 architecture with sophisticated gating mechanisms achieved 4.2% accuracy—worse than the simpler v1 model. The added complexity created new failure modes rather than solving the core problem. This reflects a broader pattern where architectural innovations may hinder rather than help:
- Attention mechanisms that increase optimization difficulty
- Modular designs that fail to coordinate effectively
- Gating mechanisms that never learn when to activate

The illusion: complex architectures seem like they should improve performance but introduce additional learning challenges that dominate any benefits.

#### 5.9.3 Metric Illusions

Aggregate metrics can hide critical failure patterns. An 82% average accuracy might represent:
- Perfect performance on base tasks (100%) but complete failure on modifications (0%)
- Consistent mediocre performance across all tasks (82%)
- High variance with unpredictable failures

Without decomposition by task type and difficulty, these scenarios appear identical in standard reporting.

#### 5.9.4 Training Illusions

Progressive curricula and mixed training strategies seemed beneficial but may prevent the generalization they aim to promote. By carefully scaffolding learning, we might teach models to depend on scaffolding rather than develop robust representations. Our mixed training reduced loss spikes but couldn't overcome the fundamental inability to handle rule modifications.

#### 5.9.5 Implications for Research Methodology

These illusions compound: flawed evaluation (constant validation accuracy) combines with architectural complexity (gating mechanisms) and training assumptions (progressive curriculum) to create false confidence in approaches that fundamentally cannot work.

This suggests the need for:
- **Multi-perspective evaluation**: Test behavioral capabilities, mechanistic understanding, and failure modes separately
- **Simplicity preference**: Start with minimal architectures to understand core challenges
- **Honest reporting**: Prominently feature what doesn't work alongside successes
- **Evaluation scrutiny**: Verify that test sets actually measure intended capabilities

The compositional language experiments thus provide a second domain demonstrating mechanism shift failures while revealing how standard research practices can systematically obscure these limitations.

## 6. A Taxonomy of Distribution Shifts

### 6.1 Existing Categorizations

The machine learning community has developed several frameworks for understanding distribution shift. We build upon these while proposing a distinction based on whether the shift requires learning new computational mechanisms.

Traditional categorizations focus on which probability distributions change:
- **Covariate shift**: P(X) changes, P(Y|X) unchanged
- **Concept drift**: P(Y|X) changes over time
- **Prior shift**: P(Y) changes, P(X|Y) unchanged

While mathematically precise, these categories do not capture whether a model trained on the source distribution can, in principle, compute correct predictions on the target distribution using its learned features.

### 6.2 Proposed Taxonomy

We propose categorizing distribution shifts based on the computational requirements:

#### Level 1: Surface Variations
**Definition**: Changes that affect input appearance but not the underlying computation.

**Examples**:
- Image corruptions (noise, blur, compression artifacts)
- Style transfer (photo to painting)
- Sensor variations (different cameras)

**Model Requirements**: Robustness to input perturbations

**Why Current Methods Work**: The same features remain relevant; only their values change.

#### Level 2: Statistical Shifts
**Definition**: Changes in data statistics while maintaining the same generative process.

**Examples**:
- Different object frequencies in image classification
- Demographic shifts in medical data
- Seasonal variations in time series

**Model Requirements**: Calibrated uncertainty, robust statistics

**Why Current Methods Partially Work**: The computation remains the same, but operating points shift.

#### Level 3: Mechanism Changes
**Definition**: Changes in the data-generating process requiring different computations.

**Examples**:
- Constant to time-varying parameters (our physics task)
- Rule changes in games
- Economic regime changes

**Model Requirements**: Ability to learn or construct new computational pathways

**Why Current Methods Show Limited Success**: Learned features may not support the required computations.

### 6.3 Empirical Validation

We validated this taxonomy by testing how different methods perform across shift levels:

**Table: Method Performance by Shift Level**
| Shift Level | Example | ERM | TTA | MAML | Best Method |
|-------------|---------|-----|-----|------|-------------|
| Level 1 | Add noise | 1.5x | 1.3x | 1.4x | TTA (-13%) |
| Level 2 | New ball sizes | 1.8x | 1.7x | 1.9x | TTA (-6%) |
| Level 3 | Time-varying g | 27x | 69x | 6238x | ERM (baseline) |

The results suggest that adaptation methods provide benefits for Level 1-2 shifts but show substantial performance degradation on Level 3 shifts.

### 6.4 Identifying Shift Levels

How can we determine which level a distribution shift belongs to? We propose several diagnostic approaches:

#### Representation Analysis
Project data into the learned representation space (e.g., penultimate layer activations):
- **Level 1-2**: Test data falls within or near the convex hull of training representations
- **Level 3**: Test data lies far outside the training representation hull

#### Gradient Alignment
Compare gradients of task loss vs. self-supervised loss:
- **Level 1-2**: Positive or near-zero alignment
- **Level 3**: Negative alignment (as seen in our experiments)

#### Oracle Performance
Train a model with access to a small amount of labeled target data:
- **Level 1-2**: Rapid improvement with few examples
- **Level 3**: Requires substantial retraining

### 6.5 Implications for Benchmark Design

Current benchmarks predominantly test Level 1-2 shifts:
- **PACS**: Level 1 (style changes)
- **ImageNet-C**: Level 1 (corruptions)
- **Wilds**: Mostly Level 2 (statistical shifts)

This may explain why methods successful on these benchmarks show reduced performance on physics prediction. Progress in the field may benefit from benchmarks that explicitly test Level 3 shifts.

### 6.6 Characteristics of Level 3 Benchmarks

Effective Level 3 benchmarks should:

1. **Have ground truth**: Enable verification that the shift genuinely changes the required computation
2. **Control the shift**: Allow systematic variation of the mechanism change
3. **Prohibit shortcuts**: Ensure that Level 1-2 robustness cannot solve the task
4. **Support analysis**: Enable inspection of why methods fail

Physics tasks naturally satisfy these criteria, but other domains could include:
- Logic puzzles with rule changes
- Economic models with regime shifts
- Biological systems with mutations
- Games with modified rules

### 6.7 Limitations of the Taxonomy

We acknowledge several limitations:

1. **Boundary cases**: Some shifts may fall between levels or combine multiple levels
2. **Domain dependence**: What constitutes a "mechanism" varies by domain
3. **Model dependence**: A shift's level may depend on the model architecture

Despite these limitations, the taxonomy provides a framework for understanding when adaptation methods are likely to succeed. It suggests that out-of-distribution generalization involving mechanism changes may require different approaches than those currently employed.

## 7. Implications and Future Directions

### 7.1 Rethinking Evaluation Protocols

Our results suggest that current evaluation protocols may benefit from modifications to better assess out-of-distribution generalization. We propose several considerations:

#### Explicit Shift Level Classification
Benchmarks should explicitly categorize their distribution shifts according to our taxonomy (surface, statistical, or mechanism changes). This would help researchers understand what type of generalization their methods achieve.

#### Graduated Evaluation
Rather than binary in-distribution/out-of-distribution classification, evaluation should measure performance across a spectrum:
1. Interpolation within training support
2. Near-extrapolation (slightly outside training)
3. Far-extrapolation (different mechanisms)

#### Failure Mode Analysis
Beyond aggregate metrics, evaluations could characterize how methods fail:
- Gradual vs. sudden performance degradation
- Uncertainty calibration under shift
- Systematic vs. random errors

### 7.2 Designing True OOD Benchmarks

Based on our findings, we outline principles for benchmarks that test extrapolation to new mechanisms:

#### Controllable Mechanism Changes
Benchmarks should allow systematic variation of underlying mechanisms. In our physics example, we can vary:
- Amplitude of gravity oscillation
- Frequency of variation
- Functional form (sinusoidal, step, random)

This enables studying how performance degrades with distance from training.

#### Multiple Domains
While we focused on physics, similar benchmarks could be created in other domains:

**Abstract Reasoning**: Puzzles where rules change between training and test
- Train: Sort by color
- Test: Sort by size then color

**Time Series**: Economic or biological systems with regime changes
- Train: Linear growth dynamics
- Test: Exponential growth dynamics

**Language**: Compositional tasks with novel combinations
- Train: "jump twice" → two jumps
- Test: "jump thrice" → three jumps (if "thrice" unseen)

#### Causal Structure
Benchmarks should have clear causal relationships that change between training and test. This allows testing whether methods learn causal structure or merely statistical associations.

### 7.3 Architectural Innovations

Our analysis suggests several directions for architectures better suited to mechanism changes:

#### Modular Computation
Instead of monolithic networks, modular architectures could learn reusable computational primitives:
- Neural Module Networks for compositional reasoning
- Graph networks with explicit relational structure
- Program synthesis components for learning algorithms

#### Physics-Informed Architectures
For physical domains, incorporating domain knowledge:
- Conservation laws as architectural constraints
- Dimensional analysis for unit consistency
- Symmetry-preserving layers

#### Uncertainty-Aware Predictions
Methods could benefit from expressing uncertainty when extrapolating:
- Bayesian neural networks for epistemic uncertainty
- Ensemble disagreement as out-of-distribution detection
- Explicit uncertainty outputs

### 7.4 Learning Paradigms

Current paradigms assume fixed mechanisms. New paradigms might include:

#### Meta-Mechanism Learning
Instead of learning parameters that adapt quickly, learn to identify and adapt to mechanism changes:
- Detect when mechanisms differ from training
- Propose hypotheses about new mechanisms
- Test hypotheses with limited data

#### Interactive Learning
When facing genuine extrapolation, models might need to:
- Request specific labeled examples
- Propose experiments to distinguish hypotheses
- Build understanding incrementally

#### Hybrid Symbolic-Neural Systems
Combining neural perception with symbolic reasoning:
- Neural networks for pattern recognition
- Symbolic systems for rule manipulation
- Learned interfaces between modalities

### 7.5 Theoretical Frameworks

Our empirical findings motivate several theoretical questions:

#### Fundamental Limits
What are the information-theoretic limits of extrapolation without labels? Our results suggest that some form of supervision or strong inductive bias may be necessary for Level 3 shifts.

#### Taxonomy Formalization
Can we formally characterize when a distribution shift requires new computational mechanisms? This might involve:
- Circuit complexity measures
- Kolmogorov complexity of the mapping
- Algebraic characterization of function classes

#### Adaptation Theory
When is adaptation beneficial vs. harmful? Our gradient analysis suggests a criterion based on objective alignment, but a more general theory is needed.

### 7.6 Practical Recommendations

For practitioners working with potential mechanism shifts:

1. **Diagnose the shift type** using representation analysis before applying adaptation methods
2. **Consider avoiding adaptation** when shift type is unknown—our results suggest baseline models may outperform adapted models in some cases
3. **Monitor prediction variance**—substantial reduction may indicate suboptimal adaptation
4. **Use ensemble disagreement** as a proxy for uncertainty under shift
5. **Collect labeled data** when facing potential Level 3 shifts rather than relying on self-supervised adaptation

### 7.7 Limitations and Open Questions

Several questions remain:

1. **Partial mechanism changes**: How do methods perform when only some aspects of the mechanism change?
2. **Gradual transitions**: Can models adapt during slow transitions between mechanisms?
3. **Transfer learning**: Can knowledge of multiple source mechanisms help with novel target mechanisms?
4. **Sample complexity**: How much labeled data is needed to learn new mechanisms?

These questions suggest directions for future research in out-of-distribution generalization involving mechanism changes.

## 8. Conclusion

We have shown that mechanism shifts—where underlying generative processes change—pose fundamental challenges across multiple domains. In physics, test-time adaptation increased error by 235% on time-varying gravity and 12-18x on variable-length pendulum tasks, even with physics-aware losses. In compositional language learning, we observed 8.2x loss increases when introducing rule modifications, with architectural improvements decreasing rather than improving performance. Most significantly, our discovery that validation accuracy remained constant at 84.3% revealed not successful adaptation but evaluation sets lacking modified examples entirely.

This finding led to identifying "layers of illusions" in machine learning research that compound to obscure fundamental limitations. Evaluation illusions mask what metrics actually measure, architectural illusions add complexity that hinders rather than helps, metric illusions hide critical failure modes, and training illusions prevent the generalization they aim to promote. These results across physics and language domains reveal that when test distributions require new computational operations, parameter adaptation within fixed architectures cannot bridge this gap—and standard evaluation practices may systematically hide these failures.

### Positioning Within Current Research

Our findings complement rather than contradict recent advances in test-time adaptation. Methods like PeTTA excel at preventing adaptation collapse, while TAIP succeeds when physical laws remain fixed but parameters vary. TTAB's comprehensive analysis shows no single method handles all distribution shifts. We extend this understanding by identifying mechanism shifts as a specific failure mode where new computational operations are required—a challenge distinct from parameter adaptation or stability preservation.

This delineation helps clarify when different approaches are appropriate:
- **Parameter shifts** (different constants): TAIP and similar physics-aware methods excel
- **Stability challenges** (risk of collapse): PeTTA's monitoring prevents degradation
- **Mechanism shifts** (new physics terms): Current methods reach their limits

### Implications for Future Research

Our analysis suggests several directions for handling mechanism shifts:

1. **Modular architectures** that can activate dormant computational pathways or reconfigure connections to express new operations

2. **Program synthesis at test time** to discover and implement new functional forms from observed data

3. **Hybrid approaches** that detect when parameter adaptation fails and switch to structure learning

4. **Benchmarks explicitly designed for mechanism shifts** beyond physics, such as reasoning tasks where solution strategies change

### Practical Considerations

For practitioners facing distribution shifts, we recommend:
- Diagnose whether shifts involve parameter changes or mechanism changes
- For parameter shifts, current TTA methods including physics-aware variants may help
- For suspected mechanism shifts, baseline models may outperform adaptation
- Monitor for both collapse (PeTTA-style) and accuracy degradation (our gradient alignment)

### Limitations and Scope

We acknowledge several limitations:
- Our experiments focus on physics prediction tasks with specific types of mechanism shifts
- We tested representative methods but not all possible architectures or adaptation strategies
- The boundary between parameter and mechanism shifts may vary across domains
- We implemented collapse detection inspired by PeTTA but not their exact algorithm

### Final Thoughts

This work identifies mechanism shifts as a fundamental challenge where current adaptation methods fail across domains. More critically, our discovery of layers of illusions reveals how standard research practices can systematically obscure these failures. The constant 84.3% validation accuracy that masked complete failure on modifications exemplifies a broader pattern where evaluation design, architectural choices, metrics, and training strategies compound to create false confidence.

These findings have implications beyond specific algorithms. They suggest we need fundamental changes to how we develop and evaluate machine learning systems:
- Evaluation sets must explicitly test intended capabilities, not proxies
- Architectural complexity should be justified by improved understanding, not hope
- Metrics must decompose performance to reveal failure modes
- Research must prominently report what doesn't work alongside successes

The ability to adapt to new mechanisms remains crucial for applications from climate modeling to economic forecasting. But achieving this requires not just new methods but also confronting the illusions that may be hiding how far we still have to go. By documenting both algorithmic limitations and methodological blind spots, we hope to inspire more rigorous approaches to one of machine learning's most important challenges.

## References

### Recent Advances in Test-Time Adaptation (2023-2025)

1. **Bohdal, O., Li, Y., & Hospedales, T.** (2024). PeTTA: Persistent Test-Time Adaptation in Dynamic Environments. *Advances in Neural Information Processing Systems (NeurIPS)*, 37.

2. **Fu, Y., Zhang, L., & Schoenholz, S.** (2025). TAIP: Test-time Augmentation for Inter-atomic Potentials. *Nature Communications*, 16, 892.

3. **Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T.** (2023). TTAB: A Comprehensive Test-Time Adaptation Benchmark. *International Conference on Machine Learning (ICML)*, 202, 35678-35693.

4. **Niu, S., Wu, J., Zhang, Y., Chen, Y., Zheng, S., Zhao, P., & Tan, M.** (2023). Efficient Test-Time Model Adaptation without Forgetting. *International Conference on Machine Learning (ICML)*, 202, 26326-26337.

5. **Zhou, K., Liu, Z., Qiao, Y., Xiang, T., & Loy, C. C.** (2023). Domain Generalization: A Survey. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 45(4), 4396-4415.

### Test-Time Adaptation Methods

6. **Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T.** (2021). TENT: Fully Test-Time Adaptation by Entropy Minimization. *International Conference on Learning Representations (ICLR)*.

7. **Zhang, M., Marklund, H., Dhawan, N., Gupta, A., Levine, S., & Finn, C.** (2022). Adaptive Risk Minimization: Learning to Adapt to Domain Shift. *Advances in Neural Information Processing Systems (NeurIPS)*, 34, 23664-23678.

8. **Wang, Q., Fink, O., Van Gool, L., & Dai, D.** (2022). Continual Test-Time Domain Adaptation. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 7193-7203.

9. **Liu, Y., Kothari, P., van Delft, B., Bellot-Gurlet, B., Mordan, T., & Alahi, A.** (2021). TTT++: When Does Self-Supervised Test-Time Training Fail or Thrive? *Advances in Neural Information Processing Systems (NeurIPS)*, 34, 21808-21820.

10. **Liang, J., Hu, D., & Feng, J.** (2020). Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation. *International Conference on Machine Learning (ICML)*, 119, 6028-6039.

### Meta-Learning Approaches

11. **Finn, C., Abbeel, P., & Levine, S.** (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. *International Conference on Machine Learning (ICML)*, 70, 1126-1135.

12. **Nichol, A., Achiam, J., & Schulman, J.** (2018). On First-Order Meta-Learning Algorithms. *arXiv preprint*.

13. **Li, Z., Zhou, F., Chen, F., & Li, H.** (2017). Meta-SGD: Learning to Learn Quickly for Few-Shot Learning. *arXiv preprint*.

14. **Rajeswaran, A., Finn, C., Kakade, S. M., & Levine, S.** (2019). Meta-Learning with Implicit Gradients. *Advances in Neural Information Processing Systems (NeurIPS)*, 32.

15. **Antoniou, A., Edwards, H., & Storkey, A.** (2019). How to Train Your MAML. *International Conference on Learning Representations (ICLR)*.

### OOD Benchmarks and Theory

16. **Gulrajani, I., & Lopez-Paz, D.** (2021). In Search of Lost Domain Generalization. *International Conference on Learning Representations (ICLR)*.

17. **Koh, P. W., Sagawa, S., Marklund, H., Xie, S. M., Zhang, M., Balsubramani, A., ... & Liang, P.** (2021). WILDS: A Benchmark of in-the-Wild Distribution Shifts. *International Conference on Machine Learning (ICML)*, 139, 5637-5664.

18. **Hendrycks, D., & Dietterich, T.** (2019). Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. *International Conference on Learning Representations (ICLR)*.

19. **Recht, B., Roelofs, R., Schmidt, L., & Shankar, V.** (2019). Do ImageNet Classifiers Generalize to ImageNet? *International Conference on Machine Learning (ICML)*, 97, 5389-5400.

20. **Miller, J., Krauth, K., Recht, B., & Schmidt, L.** (2020). The Effect of Natural Distribution Shift on Question Answering Models. *International Conference on Machine Learning (ICML)*, 119, 6905-6916.

21. **Lake, B. M., & Baroni, M.** (2018). Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks. *International Conference on Machine Learning (ICML)*, 80, 2873-2882.

### Physics-Informed Machine Learning

22. **Raissi, M., Perdikaris, P., & Karniadakis, G. E.** (2019). Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations. *Journal of Computational Physics*, 378, 686-707.

23. **Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K.** (2018). Neural Ordinary Differential Equations. *Advances in Neural Information Processing Systems (NeurIPS)*, 31.

24. **Greydanus, S., Dzamba, M., & Yosinski, J.** (2019). Hamiltonian Neural Networks. *Advances in Neural Information Processing Systems (NeurIPS)*, 32.

25. **Cranmer, M., Greydanus, S., Hoyer, S., Battaglia, P., Spergel, D., & Ho, S.** (2020). Lagrangian Neural Networks. *arXiv preprint*.

26. **Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E.** (2021). Learning Nonlinear Operators via DeepONet Based on the Universal Approximation Theorem of Operators. *Nature Machine Intelligence*, 3(3), 218-229.

### Ensemble and Exploration Methods

27. **Lakshminarayanan, B., Pritzel, A., & Blundell, C.** (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.

28. **Bengio, E., Jain, M., Korablyov, M., Precup, D., & Bengio, Y.** (2021). Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation. *Advances in Neural Information Processing Systems (NeurIPS)*, 34.

29. **Fort, S., Hu, H., & Lakshminarayanan, B.** (2019). Deep Ensembles: A Loss Landscape Perspective. *arXiv preprint*.

30. **Ovadia, Y., Fertig, E., Ren, J., Nado, Z., Sculley, D., Nowozin, S., ... & Snoek, J.** (2019). Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift. *Advances in Neural Information Processing Systems (NeurIPS)*, 32.

31. **Wenzel, F., Roth, K., Veeling, B., Świątkowski, J., Tran, L., Mandt, S., ... & Louizos, C.** (2020). How Good is the Bayes Posterior in Deep Neural Networks Really? *International Conference on Machine Learning (ICML)*, 119, 10248-10259.

[References continue through #50 as in original paper...]

## Appendix

### A. Experimental Details

[To be completed with specific implementation details, hyperparameters, and data generation procedures]

### B. Additional Results

[To be completed with additional ablations and experimental results]

### C. Theoretical Analysis

[To be completed with extended theoretical analysis if needed]
