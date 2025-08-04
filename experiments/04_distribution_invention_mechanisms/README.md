# Experiment 04: Distribution Invention Mechanisms

## Overview

This experiment explores the fundamental mechanisms required for neural networks to **invent new distributions** rather than merely interpolate within their training distribution. 

## Key Insight

Through our work on variable binding (Experiment 03), we discovered that **variable binding IS distribution invention in miniature**:
- Base distribution: X has no meaning
- Invented distribution: X → jump
- This is exactly the cognitive operation needed for creative extrapolation

## Research Questions

1. What are the minimal cognitive operations required for distribution invention?
2. How can we design architectures that explicitly modify rules rather than implicitly encode them?
3. Can we scale these mechanisms from simple variable binding to complex domains (physics, vision)?

## Core Principles Discovered

1. **Explicit over Implicit**: Distribution invention requires explicit rule modification, not emergent properties
2. **Discrete Operations**: Some cognitive operations cannot be made continuous without losing their essence
3. **State Tracking**: Models must know "which distribution am I in?"
4. **Hybrid Architectures**: Combine discrete rule manipulation with continuous execution

## Current Approach: Two-Stage Compiler

Based on our findings, we're implementing a Two-Stage Compiler that separates:
- **Stage 1**: Discrete rule extraction and modification (guaranteed correct)
- **Stage 2**: Continuous neural execution within the modified distribution

## Files in This Experiment

- `progressive_complexity_dataset.py` - Test suite with 4 levels of complexity
- `compositional_final_fix.py` - Working parser for Stage 1 
- `two_stage_compiler.py` - Main implementation (to be created)
- `THEORETICAL_FRAMEWORK.md` - Core theoretical insights
- `IMPLEMENTATION_PLAN.md` - Detailed technical approach

## Connection to Broader Goals

This work directly addresses our core research question: How can neural networks think outside their training distribution? By understanding the minimal mechanisms (explicit rule modification, state tracking, hybrid processing), we can build models capable of true creative extrapolation.

## Next Steps

1. Implement Two-Stage Compiler for variable binding (proof of concept)
2. Achieve >90% accuracy to validate approach
3. Extract general principles for distribution invention
4. Apply to physics domain (modify gravity, friction)
5. Scale to visual concept blending and mathematical discovery