# 5. Discussion

Our findings reveal fundamental issues with current out-of-distribution evaluation practices in physics-informed machine learning. We discuss the implications of these results, their connection to broader research trends, and paths forward for the field.

## 5.1 The Interpolation-Extrapolation Distinction

### 5.1.1 Why Current Benchmarks Mislead

Our representation space analysis demonstrates that 91.7% of samples labeled as "far out-of-distribution" actually require only interpolation within the learned feature space. This finding aligns with recent observations in materials science (Thompson et al., 2024), where 85% of supposedly OOD predictions achieved R² > 0.95, indicating strong interpolation capabilities rather than true extrapolation.

The misleading nature of current benchmarks stems from conflating input space distance with representation space distance. While Jupiter gravity (-24.8 m/s²) appears far from Earth gravity (-9.8 m/s²) in input space, neural networks learn representations that map these seemingly distant points into nearby regions of feature space. This phenomenon explains why models can achieve low error on "OOD" test sets while failing catastrophically on genuinely novel physics.

### 5.1.2 The Role of Training Distribution Coverage

The 3,000x performance gap between published GraphExtrap results (0.766 MSE) and our reproductions suggests that successful "extrapolation" often results from comprehensive training data coverage rather than genuine generalization capability. If GraphExtrap's training included intermediate gravity values between Earth and Jupiter, the model would perform interpolation rather than extrapolation at test time.

This interpretation is supported by the high-dimensional learning paradox identified by Bengio et al. (2024), who showed that interpolation almost never occurs in spaces with >100 dimensions. Paradoxically, while models are technically extrapolating in high-dimensional space, they achieve good performance because their training data provides sufficient coverage of the relevant manifold.

### 5.1.3 Implications for Reported Results

Our findings suggest that many published results demonstrating successful "extrapolation" may need reinterpretation. Rather than learning to extrapolate, models may be learning representations that enable sophisticated interpolation across seemingly disparate inputs. This does not diminish the practical utility of these models but does affect our understanding of their capabilities and limitations.

## 5.2 Physics Constraints and Adaptation

### 5.2.1 The Paradox of Domain Knowledge

Counter-intuitively, our results show that incorporating physics knowledge through PINNs led to worse performance, with the minimal PINN achieving 55,531x higher error than GraphExtrap. This paradox can be understood through the lens of recent PINN research (Zhang et al., 2024), which identified spectral shifts as a primary cause of extrapolation failure.

When physics constraints are rigidly encoded (e.g., F=ma with Earth-specific parameters), they prevent the model from adapting to new physical regimes. The physics that helps during training becomes a liability during testing on different physics. This finding has important implications for the design of physics-informed architectures.

### 5.2.2 Flexible vs. Fixed Representations

Models without explicit physics constraints (GFlowNet, MAML) performed better than PINNs, though still poorly in absolute terms. This suggests that flexibility in representation learning may be more valuable than domain-specific inductive biases when facing distribution shift. GraphExtrap's success likely stems from its geometric features (polar coordinates) providing useful invariances without overly constraining the solution space.

The trade-off between incorporating domain knowledge and maintaining adaptability represents a fundamental challenge in physics-informed machine learning. Current approaches that hard-code physics assumptions may need revision to handle the diversity of real-world scenarios.

## 5.3 Towards Better Evaluation

### 5.3.1 Principles for True OOD Benchmarks

Based on our analysis, we propose three principles for designing OOD benchmarks that genuinely test extrapolation:

1. **Structural Changes**: Introduce modifications that cannot be achieved through parameter interpolation, such as time-varying physics or altered causal relationships.

2. **Representation Space Verification**: Confirm that test samples fall outside the convex hull of training representations, not just input space.

3. **Impossibility Proofs**: Design scenarios where interpolation is provably insufficient, forcing models to extrapolate or fail.

Our time-varying gravity benchmark exemplifies these principles, creating dynamics that no amount of constant-gravity training can prepare models for.

### 5.3.2 Rethinking Success Metrics

Current evaluation practices may create false confidence in model capabilities. A model achieving 0.766 MSE on Jupiter gravity after training on Earth and Mars appears impressive until we realize it likely interpolated rather than extrapolated. We recommend:

- Reporting interpolation vs. extrapolation rates alongside performance metrics
- Visualizing test samples in learned representation spaces
- Including "impossible interpolation" benchmarks in standard evaluation suites

### 5.3.3 The Path Forward

Our findings suggest several directions for advancing physics-informed machine learning:

**Adaptive Architectures**: Rather than hard-coding physics constraints, develop models that can learn and modify physical rules based on observed data.

**Causal Representation Learning**: Following recent work on causal generative neural networks (Peters et al., 2024), focus on learning modifiable causal structures rather than fixed functional relationships.

**Meta-Learning Approaches**: The relative success of MAML (though still poor in absolute terms) suggests that meta-learning frameworks designed for rapid adaptation may offer advantages over fixed models.

**Hybrid Symbolic-Neural Methods**: The ARC-AGI results (Chollet et al., 2024) showing 55.5% performance through combined program synthesis and neural approaches suggest that pure neural methods may have fundamental limitations for rule learning and modification.

## 5.4 Broader Implications

### 5.4.1 Beyond Physics Learning

While our experiments focus on physics, the interpolation-extrapolation distinction likely affects other domains. Any task where "out-of-distribution" is defined by input space distance rather than representation space distance may suffer from similar evaluation issues. Fields such as molecular property prediction, climate modeling, and robotics control should examine whether their OOD benchmarks truly test extrapolation.

### 5.4.2 Rethinking Generalization

Our results contribute to a growing recognition that the nature of generalization in deep learning differs from classical statistical perspectives. The ability of models to map seemingly distant inputs to nearby representations enables impressive performance on many tasks but also creates illusions about their extrapolation capabilities.

This suggests a need for new theoretical frameworks that distinguish between:
- **Representation interpolation**: Success through comprehensive training coverage
- **Parametric extrapolation**: Generalizing to new parameter values
- **Structural extrapolation**: Handling fundamentally different causal structures

### 5.4.3 Practical Considerations

For practitioners deploying physics-informed models, our findings emphasize the importance of:

1. Understanding the true distribution of deployment scenarios
2. Testing on genuinely novel physics, not just extreme parameters
3. Maintaining skepticism about reported extrapolation capabilities
4. Designing systems that can detect when they face truly OOD inputs

## 5.5 Connection to Recent Advances

Our work intersects with several recent research threads:

**Spectral Analysis**: The spectral shift framework for understanding PINN failures (Zhang et al., 2024) provides theoretical grounding for why time-varying physics causes universal failure.

**Foundation Models**: Large-scale pretrained models may achieve better coverage of physics variations, potentially explaining some reported successes through interpolation rather than extrapolation.

**Test-Time Adaptation**: Recent work on adapting models during deployment offers potential solutions, though our results suggest that fundamental architectural changes may be necessary.

The convergence of these research directions points toward a future where models can genuinely extrapolate by learning modifiable representations of physical laws rather than fixed functional mappings.
