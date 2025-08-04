# Literature Review Notes - 2024-25 Papers

## Summary of Key Papers from Reviewer Feedback

### 1. Generative Flow Networks (GFlowNets)

#### Evolution-Guided GFlowNets (EGFN) - 2024
- **Key Innovation**: Augments GFlowNet training with Evolutionary Algorithms
- **Relevance**: Provides robust exploration of complex, high-dimensional spaces
- **Application**: Better handling of long trajectories and sparse rewards
- **Our Use**: Could enhance our distribution modification by improving exploration strategies

#### Generalization for GFlowNets - 2024
- **Focus**: Length generalization - generalizing to longer trajectories than training
- **Theory**: Links generalization with stability
- **Relevance**: Shows how to extend beyond initial training distribution
- **Our Use**: Theoretical foundation for controlled extrapolation

### 2. Compositional Generalization

#### Scale Leads to Compositional Generalization - 2025
- **Key Finding**: Simply scaling data and model size enables compositional generalization
- **Evidence**: MLPs can approximate compositional tasks with linear neurons
- **Discovery**: Successful generalization allows linear decoding of task constituents
- **Our Use**: Suggests our distribution invention might benefit from scale

#### Human-like Systematic Generalization (Lake & Baroni) - 2023
- **Approach**: Meta-learning for compositionality (MLC)
- **Achievement**: Neural networks achieve human-like systematicity
- **Key**: Optimized specifically for compositional skills
- **Our Use**: Meta-learning approach could help with rule modification

### 3. Out-of-Distribution Generalization

#### Graph Structure Extrapolation for OOD - 2023
- **Innovation**: Non-Euclidean-space linear extrapolation
- **Method**: Extrapolates both structure and feature spaces
- **Results**: Substantial improvements across graph OOD tasks
- **Our Use**: Directly relevant - shows controlled extrapolation is possible

### 4. Key Insights for Our Research

1. **Controllable Extrapolation is Achievable**: Recent work shows NNs can extrapolate with proper methods
2. **Scale Matters**: Larger models show better compositional abilities
3. **Meta-Learning Helps**: Training for compositionality improves systematic generalization
4. **Evolutionary Strategies**: Can improve exploration in sparse reward settings
5. **Graph Methods**: Provide structured approach to extrapolation

### 5. How This Changes Our Positioning

**Original Claim**: "No neural network extrapolates"
**Revised Claim**: "Few neural networks achieve *controllable* extrapolation for creative tasks"

**Original Focus**: First to enable extrapolation
**Revised Focus**: First to enable *distribution invention* through selective rule modification

**Key Differentiators**:
- We focus on creative generation, not just OOD performance
- We emphasize interpretable rule modification
- We target multiple modalities with unified approach
- We aim for human-evaluable "pocket realities"

### 6. Methods to Incorporate

1. **GFlowNets**: As a baseline and potential component
2. **Meta-learning**: For quick adaptation to new rule sets
3. **Scale considerations**: Plan for larger models in later stages
4. **Graph-based extrapolation**: For structured domains
5. **Evolutionary strategies**: For exploration enhancement

### 7. Benchmarks to Add

- ARC-AGI (recent models achieve >55%)
- gSCAN (compositional generalization)
- COGS (systematic generalization)
- WOODS (time-series OOD)

This positions our work as advancing the field rather than starting from scratch.
