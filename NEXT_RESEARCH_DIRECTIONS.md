# Next Research Directions: From OOD Illusion to Distribution Invention

Generated: 2025-07-27

## Executive Summary

Our discovery of the "OOD Illusion" - where models appearing to generalize are actually failing completely on true OOD tasks - aligns perfectly with recent advances in compositional generalization research (2023-2025). The literature reveals both why current approaches fail and promising paths forward. This document outlines our research strategy based on these insights.

## Core Insights from Literature Analysis

### 1. Variable Binding is the Missing Primitive
- CLIP and other SOTA models fail at basic variable binding (Lewis et al., 2024)
- Without binding, models cannot modify rules systematically
- Wu et al. (2025) show Transformers CAN learn binding with proper training

### 2. Explicit Structure Beats Implicit Learning
- NSR (Li et al., 2024) achieved >90% on SCAN using explicit symbolic parsing
- Modular architectures (Schug et al., 2024) enable exponential generalization
- Causal separation (Yin et al., 2024) prevents spurious correlation learning

### 3. Evaluation Must Test True Extrapolation
- Most "OOD" tests are interpolation in disguise (Li et al., 2025)
- Need mechanism-based splits, not just parameter variation
- ARC-AGI-2 exemplifies proper evaluation design

### 4. Meta-Learning Shapes the Learning Algorithm
- MLC (Lake & Baroni, 2023) achieved human-level generalization through curriculum
- Meta-learning teaches HOW to learn, not just what to learn
- Interpretability and adaptation can coexist (CAMEL, 2024)

## Research Roadmap

### Phase 1: Fix Evaluation (1-2 weeks)
**Goal:** Establish proper evaluation to avoid future illusions

1. **Implement Convex Hull Analysis**
   - Add representation space analysis to all experiments
   - Classify test samples: interpolation vs near vs far extrapolation
   - Create visualization tools for OOD verification

2. **Design Mechanism-Based Test Suites**
   - Physics: Time-varying forces, phase transitions, emergent properties
   - Language: Novel syntactic constructions, recursive modifications
   - Create behavioral probing suite for each domain

3. **Establish True Baselines**
   - Simple models with honest failure modes
   - Human performance benchmarks
   - Document all assumptions explicitly

### Phase 2: Variable Binding Architecture (2-3 weeks)
**Goal:** Build foundation for true compositional generalization

1. **Implement Explicit Binding Mechanism**
   ```python
   class BindingModule(nn.Module):
       def __init__(self):
           self.variable_slots = nn.Parameter(torch.randn(n_slots, d_model))
           self.binding_attention = MultiHeadAttention()
           self.pointer_network = PointerNet()
   ```

2. **Force Binding Through Training**
   - Create dereferencing tasks (Wu et al. approach)
   - Design curricula requiring variable tracking
   - Add binding consistency loss

3. **Test on SCAN Modifications**
   - Can model bind "jump" independently of context?
   - Can it apply modifications to bound variables?
   - Measure binding stability across contexts

### Phase 3: Hybrid Neuro-Symbolic System (3-4 weeks)
**Goal:** Achieve SCAN success through explicit structure

1. **Build NSR-Inspired Architecture**
   - Neural perception → Symbolic parsing → Rule execution
   - Implement deduction-abduction learning
   - Create intermediate symbolic language for SCAN

2. **Add Differentiable Rule Learning**
   - DFORL-style rule discovery
   - Represent SCAN as relational facts
   - Learn modification rules explicitly

3. **Stress Test Beyond Standard SCAN**
   - Novel command combinations
   - Recursive modifications
   - Reference resolution tasks

### Phase 4: Modular Physics Architecture (3-4 weeks)
**Goal:** Fix physics extrapolation through composable modules

1. **Implement Modular PINN**
   ```python
   class ModularPhysicsLearner:
       def __init__(self):
           self.physics_modules = {
               'gravity': GravityModule(),
               'friction': FrictionModule(),
               'collision': ElasticCollisionModule()
           }
           self.module_composer = HyperNetwork()
           self.causal_encoder = CausalFactorization()
   ```

2. **Causal Intervention Training**
   - Separate invariant laws from parameters
   - Train with active interventions
   - Verify causal structure learning

3. **Test True Physics Extrapolation**
   - Time-varying parameters
   - Novel force combinations
   - Phase transitions

### Phase 5: Meta-Learning for Distribution Invention (4-6 weeks)
**Goal:** Enable systematic modification and invention of distributions

1. **Design MLC-Style Curriculum**
   - Episodes with base distribution → modification → test
   - Progressive complexity: single rule → multiple → novel combinations
   - Cross-domain transfer tasks

2. **Implement Interpretable Meta-Learning**
   - CAMEL-inspired parameter separation
   - Explicit modification tracking
   - Verification of intended changes

3. **Create Distribution Invention Benchmark**
   - Physics: Invent consistent new force laws
   - Language: Create novel but coherent syntax rules
   - Abstract: Design new puzzle mechanics (ARC-style)

## Immediate Next Steps (This Week)

### 1. Create Minimal Proof-of-Concept
**File:** `experiments/03_binding_architecture/minimal_binding_demo.py`
- Implement basic variable binding for SCAN
- Show it can handle "jump" → "hop" modification
- Compare to our failed attempts

### 2. Fix Evaluation Infrastructure
**File:** `evaluation/convex_hull_analyzer.py`
- Add representation analysis to existing models
- Generate OOD verification plots
- Document true extrapolation requirements

### 3. Design Hybrid Architecture
**File:** `models/hybrid_neuro_symbolic.py`
- Sketch NSR-style architecture for SCAN
- Plan integration points for rule learning
- Create training pipeline

### 4. Write Research Paper Outline
**File:** `papers/ood_illusion/outline_v2.md`
- Incorporate literature findings
- Position our work in context
- Plan experiments to support claims

## Success Metrics

1. **SCAN with Modifications**: >80% on true held-out modifications
2. **Physics Extrapolation**: <10x degradation on time-varying parameters
3. **Distribution Invention**: Generate coherent novel rules verified by humans
4. **Evaluation Robustness**: No evaluation illusions in any experiment

## Risk Mitigation

1. **Complexity Creep**: Start with minimal implementations, add only what helps
2. **Evaluation Gaming**: Always include adversarial test cases
3. **Overfitting to Benchmark**: Test on multiple domains, create new tasks
4. **Computational Cost**: Use efficient implementations, leverage meta-learning

## Key Papers to Implement/Extend

1. **NSR (Li et al., 2024)**: For SCAN success
2. **Modular Meta-Learning (Schug et al., 2024)**: For compositional modules
3. **MLC (Lake & Baroni, 2023)**: For curriculum design
4. **CAMEL (Blanke & Lelarge, 2024)**: For interpretable physics
5. **Energy-Based OOD (Chen et al., 2023)**: For detection and adaptation

## Timeline

- **Month 1**: Evaluation fixes + Variable binding architecture
- **Month 2**: Hybrid neuro-symbolic SCAN + Modular physics
- **Month 3**: Meta-learning integration + Distribution invention
- **Month 4**: Paper writing + Additional experiments

## Conclusion

The literature validates our OOD Illusion discovery while providing clear paths forward. By combining explicit binding, modular architectures, neuro-symbolic reasoning, and meta-learning, we can move from exposing the illusion to achieving true distribution invention. The key is building systems that understand and manipulate compositional structure, not just memorize patterns.

Our next session should start with the minimal binding proof-of-concept to demonstrate immediate progress beyond our current failures.
