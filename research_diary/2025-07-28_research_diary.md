# Research Diary - July 28, 2025

## Session Summary
Implemented Gumbel-Softmax for differentiable variable binding and discovered critical gradient flow issues. Made significant progress by:
1. Diagnosing zero gradients in binding projections
2. Implementing proper soft attention mechanism
3. Discovering model collapse to single action
4. Developing phased training approach

## Key Technical Discoveries

### 1. Gradient Flow Issue
- **Problem**: Hard `argmax` operation in `BindingAttention` was blocking gradients
- **Solution**: Implemented Gumbel-Softmax for differentiable discrete sampling
- **Files modified**: `train_binding_mlx_proper.py:72-126` (BindingAttention class)

### 2. Soft Attention for Slot Retrieval
- **Problem**: Hard indexing `slot_values[slot_indices]` also blocked gradients
- **Solution**: Used soft attention weights for weighted sum over slot values
- **Code location**: `train_binding_mlx_proper.py:224-268` (forward pass)

### 3. Temperature Sensitivity
- **Discovery**: Gradient magnitude heavily dependent on Gumbel-Softmax temperature
  - Temperature 5.0: gradients ~0.002-0.004 (too small)
  - Temperature 1.0: gradients ~0.02-0.03 (10x stronger)
  - Temperature 0.5: gradients ~0.03 (optimal)
- **Fix**: Changed initial temperature from 5.0 to 1.5, faster decay (0.95)

### 4. Model Collapse Pattern
- **Observation**: Model collapsed to always predicting "WALK" (100% of predictions)
- **Analysis**: Created `analyze_binding_behavior.py` revealing:
  - Perfect attention entropy (100%) - completely uniform attention
  - Consistent variable binding (X always → slot 3)
  - But no task learning whatsoever

### 5. Phased Training Success
- **Approach**: Created `train_binding_phased.py` with 3 phases:
  1. Direct word-to-action mapping (no variables)
  2. Simple single variable binding
  3. Complex multi-variable patterns
- **Results**: 50% modification success (up from 0%!)

## Code Created/Modified

### New Files
1. `test_gradient_flow.py` - Comprehensive gradient analysis tool
2. `debug_binding_gradients.py` - Isolated gradient flow debugging
3. `test_minimal_binding.py` - Minimal test of soft attention gradients
4. `analyze_binding_behavior.py` - Model behavior analysis
5. `train_binding_phased.py` - Phased training implementation

### Key Modifications
1. `train_binding_mlx_proper.py`:
   - Added Gumbel-Softmax to BindingAttention
   - Replaced hard indexing with soft attention
   - Added gradient norm computation
   - Updated temperature schedule

## Experimental Results

### Gradient Flow Tests
```
Temperature  Q_proj grad  K_proj grad
5.0         0.002334     0.004743
2.0         0.012262     0.009363
1.0         0.026891     0.019137
0.5         0.029008     0.022087
```

### Phased Training Results
- Phase 1 (direct mapping): 75% accuracy
- Phase 2 (simple binding): 25% accuracy
- Phase 3 (complex): 41% accuracy
- **Modification tests: 50% success rate**

## Current Understanding

The variable binding problem is harder than initially thought because:
1. **Gradient vanishing**: Soft attention with many slots dilutes gradients
2. **Credit assignment**: Hard to learn which slot should store which variable
3. **Task complexity**: Full SCAN-like tasks too complex to learn directly

The phased approach shows promise but needs refinement:
- Model learns patterns but not true binding
- Still predicts same action for all variables
- Needs stronger inductive bias for variable-value association

## Next Steps

1. **Improve binding mechanism**:
   - Add explicit slot-variable association loss
   - Use contrastive learning between variables
   - Implement attention regularization to encourage focused binding

2. **Enhanced phased training**:
   - Start with 2-3 slots instead of 10
   - Add explicit "same variable" recognition task
   - Gradually increase task complexity

3. **Alternative architectures**:
   - Try pointer networks for explicit variable tracking
   - Implement external memory with read/write heads
   - Explore neural module networks for compositional structure

## Key Commands for Tomorrow

```bash
# Run phased training with more epochs
python train_binding_phased.py --phase1_epochs 20 --phase2_epochs 30 --phase3_epochs 20

# Analyze learned representations
python analyze_binding_behavior.py

# Test gradient flow with different architectures
python test_minimal_binding.py
```

## Critical Context

**The 50% modification success is misleading** - the model learned to predict "JUMP" for all X/Y variables, not true binding. Real progress requires:
1. Explicit variable identity learning
2. Stronger binding-to-slot assignment
3. Contrastive objectives to distinguish variables

The Gumbel-Softmax implementation is correct and gradients are flowing, but the learning signal is too weak with 10 slots. Consider reducing to 3-4 slots initially.

## Second Session Update

Implemented several approaches to improve variable binding:

### 1. Contrastive Learning Approach (`train_binding_contrastive.py`)
- Added contrastive loss to ensure same variables get similar representations
- Reduced slots to 4 for stronger gradients
- Achieved high binding consistency (70-80%) - variables consistently use same slots
- BUT: Model still collapses to always predicting one action
- Key insight: X always binds to slot 3, regardless of associated action

### 2. Simplified Testing (`test_simple_binding.py`)
- Created minimal binding tests to isolate the problem
- Discovered models learn consistent slot assignment but not variable-value association
- All variables bind to same slot regardless of their associated values
- 50% accuracy = random chance on binary task

### 3. Explicit Memory Model (`train_explicit_memory.py`)
- Implemented explicit read/write operations
- Clear separation of storage and retrieval phases
- Pattern: "X is jump Y is walk recall X recall Y"
- Still achieved only 50% accuracy with uniform attention (0.25 for 4 slots)
- Always predicts "jump" for all recalls

## Key Insight

The fundamental challenge is that variable binding requires solving three coupled problems simultaneously:
1. **Variable Recognition**: Identifying that "X" at position 0 and position 4 are the same entity
2. **Value Storage**: Associating the recognized variable with its declared value
3. **Value Retrieval**: Using variable identity to retrieve the correct value

Current approaches solve #1 (consistent binding) but fail at #2 and #3. The model finds it easier to memorize common patterns than learn the binding mechanism.

## Technical Analysis

### Why Models Collapse
1. **Weak Learning Signal**: With cross-entropy loss, predicting the most common action gives ~50% accuracy
2. **Attention Dilution**: Even with 4 slots, soft attention spreads gradients too thin
3. **No Explicit Binding Objective**: Loss doesn't directly reward correct variable-value association

### What's Working
- Gumbel-Softmax enables gradient flow ✓
- Variables consistently map to same slots ✓
- Contrastive loss helps with variable identity ✓

### What's Not Working
- Models don't learn value association
- Soft attention becomes uniform (high entropy)
- No mechanism to enforce slot specialization

## Next Steps for Tomorrow

1. **Curriculum Learning with Explicit Stages**:
   ```python
   # Stage 1: Variable recognition only
   "Is X at position 0 same as X at position 4?" -> Yes/No
   
   # Stage 2: Direct retrieval (no intervening words)
   "X jump X" -> jump
   "Y walk Y" -> walk
   
   # Stage 3: Full binding with distractors
   "X means jump do X" -> jump
   ```

2. **Architectural Changes**:
   - Separate networks for variable identification vs value processing
   - Hard-coded positional patterns initially (variables at positions 0,4,7...)
   - External memory with discrete read/write ops
   - Consider pointer networks or neural module networks

3. **Alternative Training Objectives**:
   - Treat as program synthesis: learn to execute binding programs
   - Use reinforcement learning with explicit rewards for correct binding
   - Meta-learning: learn to bind from few examples
   - Auxiliary tasks: predict which slot contains which variable

## Code Snippets for Tomorrow

```python
# Test if model can distinguish same vs different variables
def variable_recognition_task():
    # Same: "X ... X" -> 1
    # Different: "X ... Y" -> 0
    pass

# Minimal binding with no distractors  
def direct_binding_task():
    # "X jump X" -> jump
    # No "means" or "do" words
    pass

# Slot prediction auxiliary task
def slot_prediction_loss(bindings, var_ids):
    # Predict which variable is in which slot
    pass
```

The core insight is that variable binding might require more structured inductive biases than general attention mechanisms provide. Tomorrow we should focus on explicit curriculum learning and architectural changes that enforce the binding mechanism.