# Two-Stage Compiler Implementation Plan

## Overview

Based on our findings that gradient descent cannot learn discrete variable bindings, we will implement a Two-Stage Compiler that separates discrete operations from continuous learning.

## Architecture Design

### Stage 1: Rule-Based Binding Extraction (Discrete)
- Use the working parser from `compositional_final_fix.py`
- Extract all variable bindings explicitly
- Create a binding table: `{"X": action_id, "Y": action_id}`
- Handle rebinding by tracking temporal positions
- 100% accurate by construction

### Stage 2: Neural Execution Engine (Continuous)
- Input: Token sequence + binding table
- Architecture: Transformer with binding-aware attention
- Can attend to both tokens and binding values
- Learns compositional operators (and, then, or)
- Fully differentiable

## Implementation Steps

### Phase 1: Basic Two-Stage Compiler
```python
class TwoStageCompiler:
    def __init__(self, vocab_size, num_actions):
        self.binding_extractor = RuleBasedBindingExtractor()  # From compositional_final_fix
        self.neural_executor = BindingAwareTransformer(vocab_size, num_actions)
        
    def forward(self, tokens):
        # Stage 1: Extract bindings (discrete, perfect)
        bindings, execution_plan = self.binding_extractor(tokens)
        
        # Stage 2: Neural execution (continuous, learnable)
        outputs = self.neural_executor(tokens, bindings, execution_plan)
        return outputs
```

### Phase 2: Key Components

1. **RuleBasedBindingExtractor**
   - Adapt from `FinalCompositionalParser` in `compositional_final_fix.py`
   - Returns: bindings dict + execution structure
   - Handles all edge cases perfectly

2. **BindingAwareTransformer**
   - Modified transformer that takes binding table as input
   - Cross-attention between tokens and binding values
   - Learns to execute compositional patterns

3. **Temporal Binding Tracker**
   - Track when bindings change (rebinding)
   - Provide temporal context to neural executor

## Expected Benefits

1. **Guaranteed Binding Accuracy**: Stage 1 is rule-based, 100% correct
2. **Focused Learning**: Neural network only learns execution, not binding
3. **Interpretability**: Can inspect binding table at any point
4. **Scalability**: Can later make Stage 1 neural once Stage 2 works

## Success Metrics

- Level 1: 100% (trivial with correct bindings)
- Level 2: >95% (learn and/then/or operators)
- Level 3: >90% (handle rebinding with temporal context)
- Level 4: >85% (complex compositions)

## Files to Create

1. `two_stage_compiler.py` - Main implementation
2. `binding_aware_transformer.py` - Neural executor
3. `train_two_stage.py` - Training script
4. `test_two_stage.py` - Evaluation on progressive dataset

## Next Action

Start by copying and adapting the parser from `compositional_final_fix.py` to create `RuleBasedBindingExtractor`.