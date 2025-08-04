# Experiment Plan: Distribution Invention Mechanisms

## Objective

Develop and validate the minimal neural mechanisms required for distribution invention - the ability to create new probability distributions with modified rules rather than interpolating within training data.

## Background

This experiment emerged from a breakthrough in Experiment 03 (Variable Binding):
- We discovered that variable binding ("X means jump") IS distribution invention in miniature
- Current models fail because they try to interpolate rather than invent
- The key is explicit, discrete, stateful operations for rule modification

## Phases

### Phase 1: Proof of Concept with Variable Binding (Current)
**Goal**: Demonstrate that explicit rule modification succeeds where implicit encoding fails

1. Implement Two-Stage Compiler:
   - Stage 1: Rule-based binding extraction (discrete, 100% accurate)
   - Stage 2: Neural execution with binding context (continuous, learnable)

2. Validate on progressive complexity dataset:
   - Level 1: Simple bindings ("X means jump do X")
   - Level 2: Compositions ("do X and Y", "do X then Y")
   - Level 3: Rebinding and temporal ("X means jump do X then X means walk do X")
   - Level 4: Complex combinations

3. Success Metric: >90% accuracy across all levels

### Phase 2: Extract General Principles
**Goal**: Identify the core mechanisms that enable distribution invention

1. Analyze what makes the Two-Stage Compiler work:
   - Role of explicit vs implicit operations
   - Importance of state tracking
   - Balance between discrete and continuous processing

2. Design experiments to test each mechanism:
   - Remove explicit binding → measure performance drop
   - Replace discrete with continuous → analyze failure modes
   - Eliminate state tracking → observe consistency errors

3. Document minimal required components

### Phase 3: Scale to Physics Domain
**Goal**: Apply distribution invention to physics simulations

1. Design physics modification tasks:
   - "Set gravity to 5 m/s²"
   - "Make friction coefficient 0.1"
   - "Reverse time dynamics"

2. Implement physics-aware Two-Stage architecture:
   - Stage 1: Extract physics rules to modify
   - Stage 2: Neural simulation with modified rules

3. Compare with standard PINNs (which failed in Experiment 01)

### Phase 4: Generalize Across Domains
**Goal**: Prove distribution invention works broadly

1. Visual domain:
   - "Make all cats purple"
   - "Blend textures from two objects"
   - Measure: Coherent novel images

2. Mathematical domain:
   - "Make multiplication non-commutative"
   - "Define operations in 5D space"
   - Measure: Consistent rule application

3. Abstract reasoning:
   - ARC-like puzzles with rule modifications
   - Measure: Solve novel puzzle types

## Technical Approach

### Architecture Components

1. **Rule Extractor**:
   - Identifies modifiable aspects of distribution
   - Can be rule-based initially, neural later

2. **Modification Engine**:
   - Applies discrete changes to rules
   - Maintains consistency constraints

3. **Neural Executor**:
   - Operates within modified distribution
   - Fully differentiable for end-to-end learning

4. **State Tracker**:
   - Maintains "which distribution am I in?"
   - Handles temporal changes and scoping

### Key Innovations

1. **Explicit Binding Tables**: Replace implicit hidden states with explicit mappings
2. **Hybrid Processing**: Discrete for rule changes, continuous for execution
3. **Temporal Versioning**: Track how rules change over time
4. **Compositional Operators**: Learn to combine multiple rule modifications

## Evaluation Framework

### Metrics

1. **Accuracy**: Performance on held-out examples
2. **Consistency**: Do modified rules apply uniformly?
3. **Generalization**: Performance on increasingly complex modifications
4. **Interpretability**: Can we inspect and understand the modifications?

### Baselines

1. Standard Transformer (failed at 50% on binding)
2. Memory Networks (failed due to gradient issues)
3. Graph Neural Networks (for compositional structure)
4. Meta-learning approaches (MAML, etc.)

## Expected Outcomes

1. **Immediate**: >90% accuracy on variable binding with explicit mechanisms
2. **Medium-term**: General framework for distribution invention across domains
3. **Long-term**: Foundation for models that truly think outside their training

## Timeline

- Week 1-2: Implement and validate Two-Stage Compiler
- Week 3-4: Extract general principles, ablation studies
- Month 2: Scale to physics domain
- Month 3: Generalize to vision and mathematics
- Month 4: Write paper on distribution invention mechanisms

## Key Risks and Mitigations

1. **Risk**: Two-Stage Compiler too specialized for binding
   - **Mitigation**: Design with generalization in mind from start

2. **Risk**: Discrete operations limit end-to-end learning
   - **Mitigation**: Explore Gumbel-softmax, RL for discrete choices

3. **Risk**: State tracking becomes intractable at scale
   - **Mitigation**: Hierarchical state representations

## Success Criteria

This experiment succeeds if we:
1. Achieve >90% on variable binding with explicit mechanisms
2. Identify 3-5 core mechanisms required for distribution invention
3. Successfully apply to at least one other domain (physics)
4. Publish findings on the nature of creative extrapolation
