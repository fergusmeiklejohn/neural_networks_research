# 7. Conclusion

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
