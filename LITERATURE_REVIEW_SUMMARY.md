# Literature Review Summary - Distribution Invention Research

## Overview
This document summarizes key insights from reviewing 15+ papers recommended by reviewers, focusing on recent advances in OOD generalization, compositional learning, and counterfactual generation.

## Major Findings

### 1. **Controllable Extrapolation is Achievable**
- Meta-Learning for Compositionality (MLC) achieves 99.78%+ accuracy on SCAN by training on dynamically generated tasks
- Graph structure extrapolation shows promise for non-Euclidean OOD generalization
- Diffusion models with semantic abduction can preserve identity while modifying specific attributes

### 2. **Progressive Training Works**
- Time-varying treatment models validate our curriculum learning approach
- GFlowNets show evolutionary-guided exploration can discover novel solutions
- Meta-learning over diverse tasks teaches general systematic skills

### 3. **Hybrid Architectures Are Key**
- Sparse tree operations maintain both neural flexibility and symbolic interpretability
- Standard transformers + meta-learning outperform specialized symbolic machinery
- Causal structure combined with neural generation provides best of both worlds

### 4. **Safety Through Structure**
- Grammar-based approaches ensure outputs remain valid
- Semantic preservation techniques prevent unintended modifications
- Counterfactual methods with causal constraints maintain plausibility

## Architectural Insights

### From Meta-Learning for Compositionality:
```python
# Key insight: Dynamic task generation during training
def generate_training_task():
    grammar = sample_random_grammar()
    examples = generate_from_grammar(grammar, n=few_shot_size)
    return examples, grammar

# Train on many different "physics" or "rule sets"
for task in dynamically_generated_tasks:
    model.adapt_to_task(task.examples)
    loss = model.evaluate_on_task(task.test_cases)
```

### From Diffusion Counterfactuals:
```python
# Semantic abduction for selective modification
def modify_distribution(base_dist, modification_request):
    # Preserve semantic identity
    identity_features = extract_identity(base_dist)

    # Apply targeted modifications
    modified_features = apply_semantic_abduction(
        base_dist, modification_request
    )

    # Generate while maintaining consistency
    return generate_with_constraints(
        identity_features, modified_features
    )
```

### From Sparse Tree Operations:
```python
# Unified neural-symbolic computation
class UnifiedComputation:
    def forward(self, input):
        # Neural path for flexibility
        neural_output = self.neural_encoder(input)

        # Symbolic path for interpretability
        symbolic_output = self.tree_operations(input)

        # Combine for best of both
        return self.merge(neural_output, symbolic_output)
```

## Evaluation Framework Recommendations

### 1. **Multi-Level Evaluation** (from surveys):
- In-distribution consistency (>90%)
- Near-distribution interpolation (>85%)
- Far-distribution extrapolation (>75%)
- Novel regime generation (>70%)

### 2. **Benchmark Integration**:
- SCAN/COGS for compositional generalization
- WOODS framework for time series evaluation
- ARC-AGI for abstract reasoning (when accessible)
- Custom physics-specific benchmarks

### 3. **Human Comparison** (from MLC paper):
- Test both systematic correctness AND human-like biases
- Log-likelihood of human responses as metric
- Few-shot adaptation capabilities

## Implementation Priorities

Based on literature review:

1. **Immediate Integration**:
   - Meta-learning framework for dynamic task generation
   - Semantic abduction for selective rule modification
   - Progressive curriculum with time-varying complexity

2. **Architecture Updates**:
   - Standard transformer base (no need for complex symbolic machinery)
   - Attention mechanisms for mechanism identification
   - Diffusion-based generation for smooth distributions

3. **Training Strategy**:
   - Generate diverse "physics worlds" during training
   - Include auxiliary copy tasks for grounding
   - Balance systematic behavior with flexibility

## Key Papers for Implementation Reference

1. **Meta-Learning for Compositionality** (Nature 2023)
   - Code: https://doi.org/10.5281/zenodo.8274609
   - Directly applicable training framework

2. **Diffusion Counterfactuals** (2025)
   - Semantic preservation techniques
   - Identity-preserving modifications

3. **Graph Structure Extrapolation** (2023)
   - Non-Euclidean generalization methods
   - Applicable to physics parameter spaces

4. **WOODS Benchmark** (2024)
   - Evaluation framework design
   - Time series OOD testing

## Next Steps

1. Adapt MLC's dynamic task generation to physics domains
2. Implement semantic abduction for rule modification
3. Design evaluation suite combining multiple benchmarks
4. Create safety validators based on structural constraints

## Validation of Our Approach

The literature strongly supports our core hypotheses:
- Neural networks CAN extrapolate with proper training (MLC: 99.78% on SCAN)
- Progressive curricula enable systematic generalization
- Combining neural flexibility with structural constraints is optimal
- Safety through grammatical/physical constraints is feasible

Our contribution of "controllable extrapolation" fills a genuine gap between:
- Pure interpolation (standard deep learning)
- Unconstrained generation (potentially unsafe)
- Human-like flexible but systematic generalization

Last Updated: 2025-07-12
