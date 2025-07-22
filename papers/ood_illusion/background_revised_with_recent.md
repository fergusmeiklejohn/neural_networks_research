# 2. Background and Related Work

## 2.1 Out-of-Distribution Generalization Methods

Recent years have seen substantial progress in developing methods for OOD generalization. We review the main approaches evaluated in our study.

### Test-Time Adaptation (TTA)
Test-time adaptation methods modify model parameters during inference using unlabeled test data. TENT (Wang et al., 2021) minimizes prediction entropy while updating only batch normalization parameters. MEMO (Zhang et al., 2021) uses marginal entropy minimization with augmented test samples. These methods have shown improvements on image corruption benchmarks, with TENT reporting up to 18% error reduction on ImageNet-C.

The core assumption underlying TTA is that self-supervised objectives computed on test data can guide beneficial parameter updates. This assumption has proven effective when test data exhibits corruptions or style shifts while maintaining the same underlying task structure.

### Meta-Learning Approaches
Model-Agnostic Meta-Learning (MAML) (Finn et al., 2017) learns parameters that can quickly adapt to new tasks with few gradient steps. In the context of OOD generalization, MAML aims to find initializations that enable rapid adaptation to shifted distributions. Extensions like Meta-SGD (Li et al., 2017) and Reptile (Nichol et al., 2018) offer computational improvements while maintaining the core adaptation principle.

Meta-learning approaches have demonstrated success in few-shot learning scenarios where test tasks are drawn from the same task distribution as training. Their application to distribution shift assumes that adaptation mechanisms learned during training will transfer to test-time distribution changes.

### Ensemble and Exploration Methods
GFlowNets (Bengio et al., 2021) learn to sample from distributions over composite objects, potentially enabling exploration of diverse solutions. In OOD contexts, the hypothesis is that learning to generate diverse samples during training might improve robustness to distribution shift.

Ensemble methods aggregate predictions from multiple models trained with different initializations or data subsets. Deep Ensembles (Lakshminarayanan et al., 2017) have shown improved calibration and robustness compared to single models.

## 2.2 OOD Benchmarks in Machine Learning

Current OOD benchmarks primarily evaluate robustness to specific types of distribution shift:

**ImageNet-C** (Hendrycks & Dietterich, 2019) applies 15 types of corruptions at 5 severity levels to ImageNet validation images. Corruptions include noise, blur, weather effects, and digital artifacts. The benchmark evaluates robustness to image quality degradation.

**DomainBed** (Gulrajani & Lopez-Paz, 2021) provides a suite of domain generalization benchmarks including PACS, VLCS, and OfficeHome. These datasets test generalization across visual styles or contexts while maintaining consistent label definitions.

**Wilds** (Koh et al., 2021) provides larger-scale benchmarks with naturally occurring distribution shifts, such as different hospitals (Camelyon17) or time periods (FMoW). While more realistic than synthetic corruptions, the core task structure typically remains unchanged.

## 2.3 Recent Advances in Stabilized Test-Time Adaptation

While early TTA methods showed vulnerability to distribution shift, recent work has made significant progress in addressing stability and performance issues:

### Persistent Test-Time Adaptation (PeTTA)
Bohdal et al. (2024) introduce collapse detection mechanisms to maintain model stability over extended adaptation periods. By monitoring prediction diversity and parameter drift, PeTTA prevents the degenerate solutions we observe in standard TTA. Their method shows improvements on both Level 2 (correlation shift) and Level 3 (diversity shift) distribution changes in vision benchmarks. However, PeTTA assumes the underlying computational structure remains valid—an assumption violated in mechanism shifts where new physics terms emerge.

### Physics-Aware Test-Time Adaptation (TAIP)
Fu et al. (2025) leverage domain knowledge through physics-informed consistency losses for molecular dynamics. By enforcing energy conservation and Hamiltonian structure during adaptation, TAIP successfully generalizes to new chemical environments with different atomic configurations. Their method reduces prediction errors by an average of 30% without additional data. Crucially, TAIP's success relies on fixed physical laws with varying parameters—precisely the opposite of our mechanism shift scenario where conservation laws themselves change.

### Comprehensive Evaluation (TTAB)
Zhao et al. (2023) provide a systematic benchmark revealing that TTA success depends critically on the type of distribution shift encountered. Their work identifies three common pitfalls: (1) model selection difficulty due to online batch dependency, (2) effectiveness varies with pre-trained model quality, and (3) no existing methods handle all distribution shifts. Our mechanism shifts represent an extreme case that may extend beyond their taxonomy.

### Positioning Our Contribution
These advances strengthen TTA for many real-world scenarios. However, our experiments reveal that when the data-generating mechanism itself changes—requiring different computational operations—current stabilization techniques may not suffice. This distinction helps delineate the boundaries of current approaches:

- **Parameter Adaptation** (TAIP succeeds): Same physics equations, different constants
- **Stable Adaptation** (PeTTA succeeds): Prevent collapse while maintaining performance
- **Mechanism Adaptation** (Open problem): New computational requirements emerge

Our work thus complements these advances by identifying mechanism shifts as a persistent challenge requiring fundamentally different solutions.

## 2.4 Physics-Based Machine Learning

Physics-informed neural networks (PINNs) (Raissi et al., 2019) incorporate physical laws as constraints during training. Neural ODEs (Chen et al., 2018) parameterize dynamics with neural networks while maintaining mathematical structure. Hamiltonian Neural Networks (Greydanus et al., 2019) learn conserved quantities to improve generalization. These approaches have shown success in learning dynamics from data while respecting known physical principles.

However, most physics ML work assumes fixed physical laws. Methods are evaluated on interpolation within the same dynamical system—predicting future states or filling in missing data—rather than extrapolating to modified physics. Recent work on meta-learning for PDEs explores adaptation to different parameter values but typically within bounded ranges seen during training.

Our work differs by explicitly testing scenarios where the physical mechanism changes (e.g., time-varying parameters that introduce new terms in the governing equations), requiring models to extrapolate beyond their training regime in a fundamental way.

## 2.5 Types of Distribution Shift

The machine learning community has identified various types of distribution shift:

**Covariate shift** occurs when P(X) changes but P(Y|X) remains constant. This is common in domain adaptation where input distributions differ but the labeling function is unchanged.

**Concept drift** involves changes in P(Y|X) over time. This appears in temporal settings where relationships between features and labels evolve.

**Mechanism shift** (our focus) involves changes in the causal process generating the data. In physics, this manifests as new terms in governing equations (e.g., L̇/L in variable-length pendulum) that cannot be expressed as parameter changes within the original model structure.

## 2.6 The Extrapolation Challenge

True extrapolation requires generalizing to regions of the problem space that differ qualitatively from training data. This contrasts with interpolation (even in high dimensions) where test points lie within the convex hull of training data in some learned representation.

Recent work has begun to formalize this distinction. Webb et al. (2020) argue that many "OOD" benchmarks actually test interpolation in learned representations. Our physics experiments provide clear examples of true extrapolation: when governing equations change, no linear combination of training behaviors can produce correct test behavior.

This background motivates our investigation into how current methods perform when facing genuine mechanism shifts rather than surface variations or statistical changes.