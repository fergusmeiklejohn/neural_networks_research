# Research Requirements Document: Rethinking Distribution Invention

## Executive Summary

Our experiments have revealed fundamental limitations in current approaches to compositional generalization and distribution invention. The "Evaluation Illusion" in SCAN and complete failure of models to handle true compositional tasks suggest we need a fundamentally different approach. This document outlines what we need to learn from contemporary literature to inform our next steps.

## Context: What We've Learned

### Key Failures
1. **Compositional Generalization**: Even simple SCAN modifications completely defeated all architectures
2. **Evaluation Practices**: Standard metrics hide complete failure on intended tasks
3. **Architectural Complexity**: More sophisticated models didn't help - they just hid problems better
4. **Memorization vs Understanding**: Models memorize patterns rather than learning compositional rules

### Core Insight
The gap between "distribution interpolation" and "distribution invention" is larger than anticipated. Current neural architectures may lack fundamental capabilities needed for true extrapolation.

## Research Questions for Literature Review

### 1. Fundamental Capabilities

**Question**: What computational primitives are required for true compositional generalization?

**Look for**:
- Papers distinguishing memorization from systematic generalization
- Work on "compositional cognitive functions"
- Neural-symbolic integration approaches
- Program synthesis and differentiable programming

**Key terms**:
- "systematic generalization"
- "compositional reasoning"
- "abstract reasoning"
- "program induction"

### 2. Architecture Innovations (2023-2025)

**Question**: What novel architectures have been proposed specifically for OOD generalization?

**Focus areas**:
- Modular networks with dynamic routing
- Explicit composition operators
- Discrete/structured latent variables
- Causal representation learning
- Energy-based models for OOD

**Avoid**:
- Standard transformer variants unless specifically designed for composition
- Simple scaling approaches

### 3. Hybrid Neuro-Symbolic Systems

**Question**: How are researchers combining neural and symbolic approaches for systematic generalization?

**Investigate**:
- Differentiable rule learning
- Neural module networks
- Program synthesis backends
- Logic-guided architectures
- Explicit binding mechanisms

**Key researchers/groups** to check:
- Josh Tenenbaum's group (MIT)
- Brenden Lake (NYU)
- Felix Hill (DeepMind)
- Anyone working on ARC-AGI

### 4. Evaluation Methodology Advances

**Question**: Has anyone else identified and addressed the evaluation problems we found?

**Look for**:
- Papers on "evaluation validity"
- Work criticizing standard benchmarks
- New evaluation protocols for compositional tasks
- "Behavioral testing" approaches

### 5. Physics-Informed Neural Networks (PINNs) Updates

**Question**: What's the latest on PINNs handling distribution shifts?

**Specific interests**:
- Time-varying parameters
- Mechanism changes vs parameter shifts
- Conservation law violations
- Hybrid physics-ML approaches

### 6. Meta-Learning and Adaptation

**Question**: Are there meta-learning approaches that handle structural changes (not just parameter adaptation)?

**Look for**:
- "Meta-learning for extrapolation"
- "Structure learning"
- "Compositional meta-learning"
- Work beyond MAML/Reptile that addresses architectural adaptation

## Specific Technical Approaches to Investigate

### 1. Binding and Variable Management
- How do recent models handle variable binding?
- Are there explicit mechanisms for reference and substitution?
- Look for: "neural binding", "variable binding problem"

### 2. Explicit Composition Operators
- Models with built-in composition functions
- Tensor product representations
- Category theory inspired architectures

### 3. Discrete Optimization in Neural Networks
- Combinatorial layers
- Differentiable sorting/matching
- Discrete latent variables with proper gradients

### 4. Causal and Mechanistic Interpretability
- Work on discovering causal mechanisms
- "Mechanistic interpretability" applied to composition
- Causal representation learning

## Literature Search Strategy

### Tier 1: Must Review (Recent & Directly Relevant)
1. **2024-2025 papers** on:
   - Compositional generalization
   - Systematic generalization
   - OOD generalization in structured domains

2. **Major venue papers**:
   - NeurIPS 2024 (especially OOD/generalization tracks)
   - ICLR 2025
   - ICML 2024
   - Recent arxiv preprints

3. **Specific benchmarks**:
   - Updates to SCAN
   - ARC-AGI approaches
   - New compositional benchmarks

### Tier 2: Important Context
1. **Theoretical foundations**:
   - Limits of neural network expressivity
   - Compositional function approximation theory
   - PAC learning for extrapolation

2. **Negative results**:
   - Papers showing what doesn't work
   - Fundamental limitations discovered

### Tier 3: Inspiration Sources
1. **Cognitive science**:
   - How humans handle novel compositions
   - Developmental psychology on rule learning

2. **Formal methods**:
   - Program synthesis
   - Automated theorem proving
   - Type systems

## Key Papers to Start With

Based on our failures, prioritize finding:

1. **"The Limits of..." papers**:
   - Limits of transformers for reasoning
   - Limits of gradient descent for discrete problems
   - Fundamental barriers to extrapolation

2. **"Rethinking..." papers**:
   - Rethinking evaluation of generalization
   - Rethinking compositional learning

3. **Novel paradigms**:
   - Papers proposing completely different approaches
   - Work that explicitly addresses our failure modes

## Questions to Answer Through Literature

### Immediate Tactical Questions
1. Has anyone solved SCAN with >90% on true held-out modifications?
2. Are there architectures that explicitly prevent memorization?
3. What's the state-of-the-art on learning new operators at test time?

### Strategic Questions
1. Is neural-only approach fundamentally limited for distribution invention?
2. What would a "correct" architecture for composition look like?
3. How do we evaluate true generalization without falling into illusions?

### Philosophical Questions
1. What is the relationship between interpolation and extrapolation?
2. Can gradient descent discover new computational primitives?
3. Is distribution invention learnable or does it require built-in priors?

## Output Format for Literature Review

For each relevant paper found, document:

```markdown
### Paper: [Title]
**Authors**:
**Venue/Date**:
**Relevance**: [Why this matters for our work]

**Key Contributions**:
- Bullet points of main ideas

**Approach**:
- Technical approach summary
- Architecture details if novel

**Results**:
- What worked
- What failed
- Evaluation methodology

**Implications for Our Work**:
- How this informs our approach
- What to adopt/avoid
- Open questions it raises

**Code/Data Available**: [Yes/No - link if yes]
```

## Success Criteria for Literature Review

The review should enable us to answer:

1. **Architecture Decision**: Should we pursue neural-only, hybrid, or fully symbolic approaches?
2. **Evaluation Design**: How do we avoid creating new evaluation illusions?
3. **Technical Approach**: What specific mechanisms/modules show promise?
4. **Theoretical Grounding**: What fundamental principles should guide our design?

## Timeline and Prioritization

**Immediate** (While I work on physics experiments):
- Search for 2024-2025 compositional generalization papers
- Check if anyone has "solved" SCAN properly
- Look for physics-based OOD work

**Next Phase**:
- Deep dive into most promising approaches
- Theoretical foundations
- Implementation details of successful methods

## Final Note

Given our discoveries about the Evaluation Illusion and complete failure of standard approaches, be especially alert for:

1. **Contrarian views**: Papers arguing against current paradigms
2. **Negative results**: What others have tried and failed
3. **Fundamental limitations**: Theoretical barriers we must respect
4. **Paradigm shifts**: Completely different ways of thinking about the problem

The goal is not incremental improvement but finding approaches that could fundamentally handle distribution invention - creating new computational patterns, not just interpolating existing ones.

---

*Remember: We're looking for breakthroughs, not variations on themes that we've already shown don't work.*
