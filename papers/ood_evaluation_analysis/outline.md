# Paper Outline: Analysis of OOD Evaluation in Physics Learning

## 1. Introduction
- Current state of OOD evaluation in physics-informed ML
- Motivation: Understanding true model capabilities
- Research questions:
  - What constitutes genuine OOD in physics learning?
  - How do current benchmarks evaluate extrapolation?
  - What are the implications for model development?

## 2. Related Work
- Physics-informed neural networks (PINNs) and their limitations
- OOD detection and evaluation methods
- Interpolation vs extrapolation in high-dimensional spaces
- Recent work on spectral shifts and extrapolation failure (2024-2025)

## 3. Methodology
### 3.1 Experimental Setup
- 2D ball dynamics environment
- Training distributions (Earth, Mars gravity)
- Test distributions (Jupiter, time-varying gravity)

### 3.2 Representation Space Analysis
- t-SNE visualization of learned representations
- Convex hull analysis for OOD detection
- Density estimation methods

### 3.3 Baseline Models
- ERM with data augmentation
- GFlowNet for exploration
- Graph neural networks (GraphExtrap)
- MAML for adaptation
- Minimal physics-informed model

## 4. Results
### 4.1 Representation Analysis Findings
- 91.7% of "far-OOD" samples are interpolation
- Visualization of representation spaces
- Density analysis results

### 4.2 Baseline Performance Comparison
- Published results vs our reproduction
- 3,000x performance gap analysis
- Impact of training distribution diversity

### 4.3 True OOD Benchmark Results
- Time-varying gravity as structural change
- Universal failure of current methods
- Quantitative performance degradation

## 5. Discussion
### 5.1 The Interpolation-Extrapolation Distinction
- Why current benchmarks mislead
- Role of training distribution coverage
- Implications for reported results

### 5.2 Physics Constraints and Adaptation
- Why physics-informed models performed worse
- The paradox of domain knowledge
- Limits of current architectures

### 5.3 Towards Better Evaluation
- Principles for true OOD benchmarks
- Structural vs parametric changes
- Verification methods

## 6. Limitations and Future Work
- Scope limited to 2D physics
- Need for broader domain testing
- Potential solutions for true extrapolation

## 7. Conclusion
- Summary of key findings
- Implications for the field
- Call for improved evaluation standards

## Appendices
- A. Implementation Details
- B. Additional Experimental Results
- C. Code and Data Availability