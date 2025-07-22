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

### Invariant Learning
Methods like Invariant Risk Minimization (IRM) (Arjovsky et al., 2019) and Risk Extrapolation (REx) (Krueger et al., 2021) aim to learn features that maintain predictive power across different environments. These approaches assume access to multiple training environments and optimize for invariance across them.

## 2.2 OOD Benchmarks in Machine Learning

Current OOD benchmarks primarily evaluate robustness to specific types of distribution shift:

**PACS** (Li et al., 2017) contains images from four domains: Photo, Art painting, Cartoon, and Sketch. Models are trained on three domains and tested on the fourth. The underlying objects and labels remain consistent across domains; only the visual style varies.

**DomainBed** (Gulrajani & Lopez-Paz, 2021) provides a suite of domain generalization benchmarks including VLCS, OfficeHome, and TerraIncognita. These datasets test generalization across contexts (e.g., different offices) or geographic locations while maintaining consistent label definitions.

**ImageNet-C** (Hendrycks & Dietterich, 2019) applies 15 types of corruptions at 5 severity levels to ImageNet validation images. Corruptions include noise, blur, weather effects, and digital artifacts. The benchmark evaluates robustness to image quality degradation.

**Wilds** (Koh et al., 2021) provides larger-scale benchmarks with naturally occurring distribution shifts, such as different hospitals (Camelyon17) or time periods (FMoW). While more realistic than synthetic corruptions, the core task structure typically remains unchanged.

## 2.3 Physics-Based Machine Learning

Physics-informed neural networks (PINNs) (Raissi et al., 2019) incorporate physical laws as constraints during training. Neural ODEs (Chen et al., 2018) parameterize dynamics with neural networks while maintaining mathematical structure. These approaches have shown success in learning dynamics from data while respecting known physical principles.

However, most physics ML work assumes fixed physical laws. Methods are evaluated on interpolation within the same dynamical system—predicting future states or filling in missing data—rather than extrapolating to modified physics. Recent work on meta-learning for PDEs (Finn et al., 2022) explores adaptation to different parameter values but typically within bounded ranges seen during training.

## 2.4 Types of Distribution Shift

The machine learning community has identified various types of distribution shift:

**Covariate shift** occurs when P(X) changes but P(Y|X) remains constant. This is common in domain adaptation where input distributions differ but the labeling function is unchanged.

**Concept drift** involves changes in P(Y|X) over time. This appears in temporal settings where relationships between features and labels evolve.

**Prior shift** or **label shift** occurs when P(Y) changes but P(X|Y) remains constant. This is relevant in settings where class proportions vary between training and test.

Our physics experiments involve what might be termed **mechanism shift**—the data-generating process itself changes. When gravity transitions from constant to time-varying, the dynamics governing the system are altered. This differs qualitatively from changes in input statistics or noise levels.

## 2.5 Evaluation Protocols

Standard evaluation of OOD methods typically reports:
- Average accuracy across multiple test domains
- Worst-case performance on any single domain
- Performance gap between in-distribution and OOD data

These metrics work well when all test domains share the same task structure. However, when the data-generating mechanism changes, we need to consider:
- Whether methods maintain stable performance or degrade substantially
- How performance scales with the degree of mechanism change
- Whether uncertainty estimates reflect the increased difficulty

Our evaluation focuses on these aspects, measuring not just average performance but the nature and magnitude of degradation under mechanism shift.