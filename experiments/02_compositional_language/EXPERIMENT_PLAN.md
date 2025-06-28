# Compositional Language Experiment 2 - Distribution Invention

## 🎯 Experiment Overview

**Objective**: Test linguistic rule modification and novel command generation to validate distribution invention in compositional language tasks.

**Core Question**: Can our neural network learn compositional rules from the SCAN dataset and then generate consistent outputs under modified linguistic rules?

**Success Criteria**:
- Standard SCAN accuracy >95% 
- Rule modification consistency >70%
- Novel combination validity >60%
- Clear evidence of compositional understanding

**Compute**: Local development + Paperspace A4000 for full training

---

## 📋 Implementation Phases

### Phase 1: Data Preparation & Analysis

**Goal**: Load SCAN dataset and create modification pairs with proper isolation

- [ ] **Download and Explore SCAN Dataset**
  - Target: All SCAN splits (simple, length, primitive)
  - Understand compositional patterns
  - Document rule types and complexity
  - **Results**: [Dataset size: ___, Rule patterns: ___]

- [ ] **Create Train/Val/Test Splits with Isolation**
  - Ensure no compositional pattern leakage
  - Separate by primitive actions and modifiers
  - Create interpolation vs extrapolation test sets
  - **Results**: [Split sizes: ___, Isolation verified: ___]

- [ ] **Generate Rule Modification Pairs**
  - Single word swaps: "jump" ↔ "walk", "left" ↔ "right"
  - Action modifications: "jump" → "turn around"
  - Modifier changes: "twice" → "thrice"
  - Complex rules: reverse all directions, negate actions
  - **Results**: [Modification types: ___, Pairs generated: ___]

- [ ] **Create Novel Test Cases**
  - Unseen command combinations
  - New compositional patterns
  - Edge cases and stress tests
  - **Results**: [Novel cases: ___, Coverage: ___]

**Phase 1 Status**: ⏳ Not Started

---

### Phase 2: Model Implementation

**Goal**: Build transformer-based models for compositional rule learning

- [ ] **Implement Compositional Rule Extractor**
  - Transformer encoder for command understanding
  - Attention-based rule identification
  - Explicit compositional structure modeling
  - Target: ~50M parameters
  - **Results**: [Architecture: ___, Parameters: ___]

- [ ] **Implement Rule Modification Component**
  - Modification request encoder
  - Rule editing mechanism
  - Consistency preservation module
  - **Results**: [Component design: ___, Integration: ___]

- [ ] **Implement Sequence Generator**
  - Action sequence decoder
  - Beam search for quality
  - Consistency enforcement
  - **Results**: [Generation quality: ___, Speed: ___]

- [ ] **Create End-to-End Pipeline**
  - Connect all components
  - Add residual connections
  - Implement combined loss function
  - **Results**: [Pipeline complete: ___, Loss design: ___]

**Phase 2 Status**: ⏳ Not Started

---

### Phase 3: Progressive Training Curriculum

**Goal**: Train models using lessons from physics experiment success

- [ ] **Stage 1: Basic Compositional Learning (50 epochs)**
  - Train on standard SCAN mappings
  - No modifications, pure supervised learning
  - Target: >95% exact match accuracy
  - **Results**: [Accuracy: ___, Convergence: ___]

- [ ] **Stage 2: Simple Modifications (50 epochs)**
  - Introduce single word swaps
  - Gradually increase modification frequency
  - Balance standard and modified examples
  - Target: >80% modification consistency
  - **Results**: [Consistency: ___, Types mastered: ___]

- [ ] **Stage 3: Complex Modifications (50 epochs)**
  - Multi-word and structural changes
  - Compositional rule modifications
  - Domain randomization for robustness
  - Target: >70% on complex modifications
  - **Results**: [Complex accuracy: ___, Patterns: ___]

- [ ] **Stage 4: Novel Generation (50 epochs)**
  - Focus on unseen combinations
  - Reward valid novel outputs
  - Test true linguistic creativity
  - Target: >60% novel validity
  - **Results**: [Novel success: ___, Creativity: ___]

**Phase 3 Status**: ⏳ Not Started

---

### Phase 4: Comprehensive Evaluation

**Goal**: Validate compositional understanding and modification capabilities

- [ ] **Standard SCAN Performance**
  - Test on all SCAN splits
  - Compare to baseline models
  - Analyze compositional generalization
  - **Results**: [Scores by split: ___, vs baseline: ___]

- [ ] **Modification Consistency Analysis**
  - Test each modification type
  - Measure cross-context consistency
  - Identify failure patterns
  - **Results**: [Consistency by type: ___, Patterns: ___]

- [ ] **Novel Generation Quality**
  - Human evaluation of outputs
  - Grammatical validity checks
  - Semantic coherence assessment
  - **Results**: [Human scores: ___, Validity: ___]

- [ ] **Compositional Understanding Tests**
  - Probe attention patterns
  - Test systematic generalization
  - Analyze learned representations
  - **Results**: [Understanding score: ___, Evidence: ___]

**Phase 4 Status**: ⏳ Not Started

---

## 🧪 Specific Test Cases

### Test Case 1: Word Meaning Swaps

- [ ] **"jump" means "walk"**
  - Input: "jump twice" → Expected: "WALK WALK"
  - **Results**: [Success rate: ___, Consistency: ___]

- [ ] **"left" means "right"**
  - Input: "turn left" → Expected: "TURN RIGHT"
  - **Results**: [Success rate: ___, Consistency: ___]

- [ ] **"around" means "opposite"**
  - Input: "look around right" → Expected: "LOOK OPPOSITE RIGHT"
  - **Results**: [Success rate: ___, Consistency: ___]

### Test Case 2: Action Modifications

- [ ] **"jump" becomes "turn around"**
  - Input: "jump thrice" → Expected: "TURN AROUND TURN AROUND TURN AROUND"
  - **Results**: [Success rate: ___, Consistency: ___]

- [ ] **All movements reversed**
  - Input: "walk left" → Expected: "WALK RIGHT"
  - **Results**: [Success rate: ___, Pattern understanding: ___]

### Test Case 3: New Primitives

- [ ] **Add "spin" action**
  - Define: "spin" → "TURN LEFT TURN LEFT TURN LEFT TURN LEFT"
  - Test: "spin and walk" → Expected: consistent spin implementation
  - **Results**: [Learning success: ___, Usage consistency: ___]

- [ ] **Add "backwards" modifier**
  - Define: "walk backwards" → "WALK OPPOSITE"
  - Test compositional usage
  - **Results**: [Integration: ___, Generalization: ___]

### Test Case 4: Complex Combinations

- [ ] **Multiple simultaneous modifications**
  - "jump"→"walk" + "left"→"right" + "twice"→"thrice"
  - **Results**: [Success: ___, Interference: ___]

- [ ] **Recursive modifications**
  - "look around X" → "look opposite X and look X"
  - **Results**: [Recursion handling: ___, Depth limit: ___]

**Test Cases Summary**:
```
Overall Test Success Rate: ___%
Most Successful Modifications: ___
Most Challenging Modifications: ___
Compositional Patterns Discovered: ___
```

---

## 🔧 Infrastructure & Tools

### Development Setup

- [ ] **SCAN Dataset Integration**
  - Download all splits
  - Create unified data loader
  - Implement preprocessing pipeline
  - **Results**: [Loader complete: ___, Performance: ___]

- [ ] **Weights & Biases Configuration**
  - Project: "compositional-language"
  - Track: accuracy, consistency, modification success
  - **Results**: [Setup: ___, Dashboard: ___]

- [ ] **Evaluation Framework**
  - Exact match scoring
  - Consistency metrics
  - Novel generation assessment
  - **Results**: [Framework ready: ___, Metrics: ___]

### Training Infrastructure

- [ ] **Local Development** (Mac M3)
  - Small-scale testing
  - Architecture development
  - Quick iterations

- [ ] **Paperspace Setup**
  - A4000 for full training
  - Estimated: 8 hours total
  - Cost: ~$6-8
  - **Results**: [Setup complete: ___, Scripts ready: ___]

---

## 📊 Expected Results

### Quantitative Metrics
```
Target Performance:
- Standard SCAN: >95% accuracy
- Simple Modifications: >80% consistency  
- Complex Modifications: >70% success
- Novel Combinations: >60% validity
- Training Time: ~8 hours
- Model Size: ~50M parameters
```

### Qualitative Insights
```
Expected Findings:
- Compositional rules can be explicitly modified
- Consistency across contexts is challenging
- Some modifications easier than others
- Novel generation shows true understanding
```

---

## ⚠️ Risks & Mitigations

### Technical Risks

- [ ] **Discrete vs Continuous Challenge**
  - Risk: Methods from physics may not transfer
  - Mitigation: Adapt architecture for discrete tokens

- [ ] **Exact Match Strictness**
  - Risk: Small errors count as complete failure
  - Mitigation: Add partial credit metrics

- [ ] **Limited Vocabulary**
  - Risk: Overfitting to small action space
  - Mitigation: Test generalization thoroughly

### Research Risks

- [ ] **SCAN Limitations**
  - Risk: Dataset too simple/artificial
  - Mitigation: Create more complex test cases

- [ ] **Modification Ambiguity**
  - Risk: Multiple valid interpretations
  - Mitigation: Clear modification specifications

---

## 🎯 Key Differentiators from Physics

1. **Discrete Space**: Tokens vs continuous values
2. **Symbolic Rules**: Explicit vs implicit physics
3. **Exact Evaluation**: No approximation allowed
4. **Finite Vocabulary**: Limited modification space
5. **Compositional Structure**: Clear hierarchical rules

---

## 📝 Next Actions

**Immediate (Day 1-2)**:
1. [ ] Download SCAN dataset
2. [ ] Implement data loader
3. [ ] Create modification generator
4. [ ] Design model architecture

**Short-term (Day 3-6)**:
1. [ ] Implement all model components
2. [ ] Create training pipeline
3. [ ] Test on small subset
4. [ ] Prepare Paperspace scripts

**Training (Day 7)**:
1. [ ] Run full progressive curriculum
2. [ ] Monitor via W&B
3. [ ] Checkpoint best models

**Analysis (Day 8)**:
1. [ ] Evaluate all test cases
2. [ ] Document findings
3. [ ] Compare to physics results
4. [ ] Plan next experiment

---

**Experiment Status**: 🚧 Planning Complete, Ready for Implementation  
**Last Updated**: 2025-06-28  
**Principal Investigator**: Distribution Invention Research  
**Previous Success**: 83.51% extrapolation in physics  
**Target**: Similar success in compositional language