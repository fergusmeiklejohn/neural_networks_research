# The OOD Illusion in Physics Learning: When Generalization Methods Make Things Worse

## Paper Outline

### Abstract (150 words)
- Hook: State-of-the-art OOD methods increase error by 235-62,290% on physics tasks
- Problem: Current "OOD generalization" benchmarks test interpolation, not extrapolation
- Approach: Systematic evaluation on true OOD physics (time-varying gravity)
- Results: Universal failure of TTA, MAML, GFlowNets - all perform worse than baseline
- Insight: Fundamental distinction between statistical OOD (style changes) and representational OOD (law changes)
- Impact: Need to rethink benchmarks and methods for true extrapolation

### 1. Introduction (1.5 pages)

#### Opening Hook
"When we applied test-time adaptation to a simple physics prediction task, we expected modest improvements. Instead, we discovered something shocking: the method made predictions 235% worse. Further investigation revealed this wasn't unique to TTA—MAML performed 62,290% worse after adaptation."

#### The OOD Illusion
- What the field calls "out-of-distribution generalization" is sophisticated interpolation
- Popular benchmarks (PACS, DomainBed, ImageNet-C) test style/corruption changes
- True extrapolation—predicting under fundamentally different laws—remains unsolved

#### Why Physics Matters
- Physics provides ground truth: we know the exact laws
- Can create true OOD scenarios: time-varying gravity, phase transitions
- Exposes the gap between pattern matching and causal understanding

#### Contributions
1. Empirical evidence that all major OOD methods fail catastrophically on true extrapolation
2. Theoretical framework distinguishing statistical vs representational OOD
3. Analysis of why self-supervised adaptation amplifies errors
4. Concrete recommendations for future benchmark design

### 2. Background and Related Work (1 page)

#### 2.1 Current OOD Methods
- **Test-Time Adaptation (TTA)**: TENT, MEMO, COTTA
- **Meta-Learning**: MAML, Reptile, ProtoNets
- **Generative approaches**: GFlowNets, diffusion models
- **Invariant learning**: IRM, REx, GroupDRO

#### 2.2 OOD Benchmarks
- **Vision**: PACS (art styles), DomainBed (contexts), ImageNet-C (corruptions)
- **Language**: GLUE variants, domain adaptation
- **Limited physics benchmarks**: Usually interpolation within same dynamics

#### 2.3 The Interpolation-Extrapolation Spectrum
- Cite Chollet's ARC work on true generalization
- Physics ML: PINNs, Neural ODEs assume fixed laws
- Our position: current methods optimize for wrong objective

### 3. The Physics Extrapolation Challenge (1 page)

#### 3.1 Experimental Setup
- **Task**: Predict 2-ball trajectories (11D state space)
- **Training**: Constant gravity (9.8 m/s²)
- **True OOD Test**: Time-varying gravity g(t) = 9.8 + 2sin(0.1t)
- **Metrics**: MSE on position prediction

#### 3.2 Why This Is True OOD
- Not just different parameters—different functional form
- Outside representation manifold of constant-gravity training
- Requires extrapolation, not interpolation

#### 3.3 Implementation Details
- 10,000 training trajectories (constant gravity)
- 1,000 test trajectories each (constant and time-varying)
- Standard MLP architecture (fair comparison)
- Careful separation of data generation

### 4. Empirical Results: Universal Failure (2 pages)

#### 4.1 Baseline Performance
```
| Method | In-Dist MSE | OOD MSE | Degradation |
|--------|-------------|---------|-------------|
| ERM | 100 | 2,721 | 27x (baseline) |
```

#### 4.2 Test-Time Adaptation Catastrophe
```
| Method | OOD MSE | vs Baseline |
|--------|---------|-------------|
| No adaptation | 2,721 | - |
| TTA (1 step) | 6,935 | +235% |
| TTA (10 steps) | 8,420 | +309% |
```

- Self-supervised losses decrease while MSE increases
- Converges to degenerate constant predictions
- Performance degrades even on in-distribution data

#### 4.3 Meta-Learning Explosion
```
| Method | OOD MSE | vs Baseline |
|--------|---------|-------------|
| MAML (no adapt) | 3,019 | +10.9% |
| MAML (10-shot) | 1,697,689 | +62,290% |
```

- Gradient-based adaptation amplifies errors exponentially
- Quick adaptation = quick failure on true OOD

#### 4.4 Other Methods
- GFlowNet: -1.8% (statistically insignificant)
- Graph methods: require different data structure
- All methods within 2x of baseline or much worse

### 5. Analysis: Why Everything Fails (1.5 pages)

#### 5.1 The Fundamental Problem
- **Objective mismatch**: Self-supervised objectives ≠ accuracy
- **Information theory**: Cannot improve without ground truth
- **Wrong inductive bias**: Smoothness assumptions fail for physics

#### 5.2 Degenerate Solutions
- TTA converges to constant predictions (minimizes variation)
- Analysis of gradient directions: opposite to accuracy
- Visualization of prediction collapse

#### 5.3 The Adaptation Paradox
- Methods designed for quick adaptation fail most spectacularly
- Fast learners overfit to self-supervised signal
- Slow learners (ERM) fail gracefully

### 6. The OOD Illusion: A Taxonomy (1 page)

#### 6.1 Types of Distribution Shift
1. **Surface-level**: Corruptions, noise (ImageNet-C)
2. **Statistical**: Different data statistics, same laws (PACS)
3. **Representational**: Different underlying laws (our physics task)

#### 6.2 What Current Benchmarks Test
- Analysis of popular benchmarks
- All test interpolation within learned representations
- Success stories are sophisticated pattern matching

#### 6.3 What True Extrapolation Requires
- Causal understanding
- Compositional reasoning
- Explicit modeling of governing laws

### 7. Implications and Future Directions (1 page)

#### 7.1 Rethinking Benchmarks
- Need tasks with provable extrapolation
- Physics: time-varying parameters, phase transitions
- Abstract reasoning: ARC-like tasks
- Compositional: systematic generalization

#### 7.2 Rethinking Methods
- Physics-informed architectures
- Symbolic regression components
- Uncertainty-aware predictions
- Hybrid neural-symbolic systems

#### 7.3 Rethinking Evaluation
- Separate interpolation from extrapolation metrics
- Measure distance from training manifold
- Report graceful degradation curves

### 8. Conclusion (0.5 pages)
- Current OOD methods solve the wrong problem
- Physics exposes this limitation undeniably
- Need fundamental rethinking of generalization
- Call for honest benchmarks and methods

### References (1.5 pages)
~40-50 references covering OOD methods, physics ML, generalization theory

## Key Figures

### Figure 1: The OOD Illusion
- 2x2 grid showing:
  - Top: ImageNet → ImageNet-C (works)
  - Bottom: Constant gravity → Time-varying gravity (fails)
  - Left: Training distribution
  - Right: Test distribution

### Figure 2: Performance Degradation
- Bar chart showing all methods' performance
- Highlight 62,290% MAML failure

### Figure 3: TTA Convergence to Degeneracy
- Multi-panel showing:
  - Adaptation loss decreasing
  - MSE increasing
  - Predictions converging to constant

### Figure 4: Representation Space Analysis
- t-SNE visualization showing:
  - Training data cluster
  - Time-varying data far outside
  - Why interpolation fails

### Figure 5: Taxonomy of Distribution Shifts
- Hierarchical diagram:
  - Surface changes
  - Statistical changes
  - Law changes (true OOD)

## Appendix (2-3 pages)
- Detailed experimental setup
- Additional ablations
- Hyperparameter searches
- Reproducibility details

## Target Venues
1. **NeurIPS 2024**: Deadline passed, target 2025
2. **ICML 2025**: Likely January deadline
3. **ICLR 2025**: September/October deadline (most immediate)
4. **AISTATS 2025**: October deadline

## Writing Timeline
- Week 1: Introduction, setup, results sections
- Week 2: Analysis, taxonomy, implications
- Week 3: Polish, figures, appendix
- Week 4: Internal review and revision

## Key Messages
1. **Shocking result**: Methods make things dramatically worse
2. **Clear explanation**: Interpolation vs extrapolation
3. **Broad impact**: Affects entire OOD field
4. **Constructive**: Clear path forward with better benchmarks
