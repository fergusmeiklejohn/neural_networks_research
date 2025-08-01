# Research Diary - July 27, 2025

## Session Overview: From Literature Review to Action Plan

Today's session focused on analyzing the comprehensive literature review (2023-2025) on compositional generalization and OOD, mapping these findings to our OOD Illusion discovery, and creating an actionable research plan.

## Key Activities

### 1. Literature Analysis
Thoroughly analyzed the research document covering latest developments in:
- Fundamental computational capabilities (variable binding)
- Architectural innovations for OOD generalization
- Hybrid neuro-symbolic approaches
- Evaluation methodologies
- Meta-learning for structural adaptation

### 2. Key Insights Discovered

#### Variable Binding is THE Missing Primitive
- Lewis et al. (2024): Even CLIP fails at basic variable binding
- Wu et al. (2025): Transformers CAN learn binding with proper training
- Without binding, our SCAN models couldn't modify rules (0% performance)

#### Explicit Structure Beats Complexity
- NSR achieved >90% on SCAN using symbolic parsing
- Our complex architectures (V2 gating) made things worse (4.2% vs 84.3%)
- Modular approaches enable exponential generalization

#### Most "OOD" is Actually Interpolation
- Li et al. (2025) validates our finding: 91.7% of "OOD" was interpolation
- Convex hull analysis reveals the truth about generalization claims
- Need mechanism-based evaluation, not parameter variation

#### Meta-Learning Changes Everything
- MLC achieved human-level generalization through curriculum alone
- Meta-learning shapes HOW models learn, not just what they learn
- CAMEL shows interpretability and performance can coexist

### 3. Created Comprehensive Research Plan

#### Immediate Priorities (Week 1):
1. **Minimal Variable Binding Demo**: Implement explicit binding for SCAN
2. **Fix Evaluation**: Add convex hull analysis to verify true OOD
3. **NSR-Style Parser**: Build symbolic intermediate representation
4. **Simple Meta-Learning**: Test if curriculum helps even basic models

#### Longer-term Roadmap:
- Phase 1: Fix evaluation infrastructure
- Phase 2: Variable binding architecture  
- Phase 3: Hybrid neuro-symbolic system
- Phase 4: Modular physics architecture
- Phase 5: Meta-learning for distribution invention

## Key Files Created

1. `NEXT_RESEARCH_DIRECTIONS.md` - Comprehensive research strategy
2. `IMMEDIATE_ACTION_PLAN.md` - Week 1 detailed implementation plan
3. `LITERATURE_TO_OOD_ILLUSION_MAPPING.md` - Direct connections between findings

## Critical Code Snippets for Next Session

### Variable Binding Implementation
```python
class MinimalBindingModel:
    def __init__(self):
        self.variable_memory = VariableMemory(n_slots=10)
        self.binder = BindingAttention()
        self.executor = BoundVariableExecutor()
```

### Convex Hull OOD Verification
```python
def verify_true_ood(train_data, test_data):
    hull = ConvexHull(train_embeddings)
    for point in test_embeddings:
        distance = distance_to_hull(point, hull)
        # Classify as interpolation/near/far extrapolation
```

### NSR-Style Symbolic Parser
```python
class NSRScanParser:
    def __init__(self):
        self.perception = nn.LSTM()  # Neural → Symbolic
        self.parser = SymbolicParser(grammar=SCAN_GRAMMAR)
        self.executor = SemanticExecutor()
```

## Next Session Starting Point

1. Begin with `experiments/03_binding_architecture/minimal_binding_scan.py`
2. Implement variable slots and binding mechanism
3. Create dereferencing training tasks
4. Test if explicit binding enables "jump" → "hop" modification

## Key Insight of the Day

**The OOD Illusion isn't just our discovery - it's a fundamental limitation exposed by multiple concurrent research efforts. The solution requires explicit computational primitives (binding), proper training (meta-learning), and honest evaluation (convex hull analysis).**

## Progress Metrics

- Literature review: ✓ Complete
- Architecture recommendations: ✓ Identified  
- Implementation plan: ✓ Created
- Next steps: ✓ Clear and actionable

The path forward is clear: implement variable binding, fix evaluation, and build toward true distribution invention through explicit structure rather than hoping emergence from complexity.

---

## Second Session: Variable Binding Implementation & Framework Performance

### Session Overview
Implemented the variable binding architecture from the morning's plan, encountered significant performance issues with Keras, and discovered MLX as a game-changing alternative for Apple Silicon.

### Key Achievements

1. **Successfully Bypassed Keras Compilation Issues**
   - Encountered persistent `AttributeError: module 'keras.ops' has no attribute 'GradientTape'`
   - Multi-output model complications with loss mapping
   - Solution: Manual training loops avoiding `model.compile()`
   - Result: Training works but reveals deeper performance issues

2. **TensorFlow Implementation Results**
   - File: `experiments/03_binding_architecture/train_binding_tensorflow.py`
   - Training successful: 25.7% validation accuracy after 5 epochs
   - **Critical Discovery: Only ~33% GPU utilization**
   - Performance: ~3-4 seconds per epoch
   - Modification tests: 100% success (but likely overfitting to simple cases)

3. **MLX Framework Discovery**
   - Switched to Apple's MLX for native Metal optimization
   - File: `experiments/03_binding_architecture/train_binding_mlx_simple.py`
   - **Performance Results:**
     - Training: 0.20s per epoch (15-20x faster!)
     - Throughput: ~200 steps/second
     - Inference: **284,205 samples/second**
     - Latency: 0.90ms per batch
   - Full GPU utilization on Apple Silicon

### Technical Insights

1. **Framework Overhead Matters**
   - Keras multi-backend abstraction adds significant overhead
   - Complex gradient paths limit GPU utilization
   - Research benefit doesn't justify engineering complexity

2. **MLX Advantages for Research**
   - Simple NumPy-like API
   - Native Metal optimization
   - Lazy evaluation
   - Unified memory (no CPU/GPU transfer overhead)
   - Perfect for rapid architecture prototyping

3. **Architecture Status**
   - Basic binding mechanism implemented
   - Needs refinement - current accuracy too low
   - Modification capability present but untested on real tasks

### Code Snippets for Tomorrow

```python
# MLX model structure that works
class SimpleBindingModel(nn.Module):
    def __init__(self, vocab_size, action_vocab_size):
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder1 = nn.Linear(embed_dim, hidden_dim)
        self.bind_proj = nn.Linear(hidden_dim, hidden_dim)
        # ...

# Training loop in MLX
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
loss, grads = loss_and_grad_fn(model, batch_x, batch_y)
optimizer.update(model, grads)
mx.eval(model.parameters(), optimizer.state)
```

### Next Steps for Tomorrow

1. **Refine MLX Binding Architecture**
   - Implement proper attention-based binding
   - Add explicit slot retrieval mechanism
   - Test on real dereferencing tasks

2. **Create Proper Test Suite**
   - "X means jump. Do X." → "JUMP"
   - "X means jump. Now X means hop. Do X." → "HOP"
   - Measure true modification capability

3. **Integrate with SCAN Dataset**
   - Use existing data loaders
   - Test compositional generalization
   - Compare with baselines

### Key Learning

**Research Priority Decision**: When framework complexity impedes research progress, simplify. MLX's 15-20x performance improvement and cleaner API enable faster experimentation cycles, which is more valuable than multi-backend support for research purposes.

### Files Created/Modified
- `train_binding_tensorflow.py` - Working but slow
- `train_binding_mlx.py` - Initial attempt (dimension issues)
- `train_binding_mlx_simple.py` - Working MLX implementation
- `train_binding_simple.py` - Keras debugging version

### Tomorrow's Starting Point
1. Open `train_binding_mlx_simple.py`
2. Enhance binding mechanism with proper attention
3. Create dereferencing task dataset
4. Run: `/Users/fergusmeiklejohn/miniconda3/envs/dist-invention/bin/python experiments/03_binding_architecture/train_binding_mlx_enhanced.py`

The transition from planning to implementation revealed unexpected insights about framework choice. MLX enables the rapid experimentation cycles needed for architecture research.

---

## Third Session: Proper Variable Binding Architecture Implementation

### Session Overview
Implemented the full variable binding architecture with proper components (VariableMemory, BindingAttention, BoundVariableExecutor) and identified the key gradient flow issue preventing learning.

### Major Accomplishments

1. **Comprehensive Architecture Analysis**
   - Analyzed gaps between simplified MLX version and intended design
   - Original vision: Explicit slots, hard binding, true dereferencing
   - Simplified version: Just pattern matching, no real binding
   - Created detailed improvement plan

2. **Full Architecture Implementation**
   - File: `train_binding_mlx_proper.py`
   - **VariableMemory**: Explicit slots with content-based addressing
   - **BindingAttention**: Multi-head attention with hard binding via argmax
   - **BoundVariableExecutor**: Uses retrieved slot values (not just embeddings)
   - Real dereferencing tasks from `DereferencingTaskGenerator`

3. **Comprehensive Test Suite**
   - File: `test_binding_capabilities.py`
   - Tests: Basic binding, rebinding, compositional, chained references
   - Example: "X means jump. Now X means hop. Do X." → Should output "HOP"
   - Structured evaluation by capability type

4. **Performance Results**
   - Training runs successfully: Loss decreases 1.96 → 1.35
   - Inference speed: **97,098 samples/second** (1.32ms latency)
   - BUT: 0% modification success - model not learning binding

### Critical Discovery: Gradient Flow Problem

The model fails because hard argmax blocks gradients:
```python
bindings = mx.argmax(scores, axis=-1)  # No gradients flow through this!
retrieved = slot_values[slot_indices]  # Indexing also blocks gradients
```

Without gradients, the model can't learn which slots to bind variables to.

### Solution Identified: Gumbel-Softmax

Need differentiable binding mechanism:
```python
# Forward: hard selection, Backward: soft gradients
bindings_soft = mx.softmax(scores / temperature, axis=-1)
retrieved = mx.sum(bindings_soft[:, :, :, None] * slot_values[None, None, :, :], axis=2)
```

### Key Files Created

1. **train_binding_mlx_proper.py**
   - Full architecture with all components
   - Training on real dereferencing tasks
   - 538 lines of proper implementation

2. **test_binding_capabilities.py**
   - Comprehensive test cases
   - Failure mode analysis
   - Clear success metrics

3. **BINDING_IMPROVEMENT_PLAN.md**
   - Detailed architectural gaps analysis
   - Step-by-step improvement roadmap
   - Research impact statement

4. **NEXT_STEPS_BINDING.md**
   - Immediate fixes needed (Gumbel-Softmax)
   - Architecture improvements
   - Testing protocol with commands

### Next Steps (Immediate)

1. **Fix Gradient Flow**
   ```python
   # Implement in BindingAttention
   def gumbel_softmax(logits, temperature=1.0):
       gumbel_noise = -mx.log(-mx.log(mx.random.uniform(logits.shape)))
       return mx.softmax((logits + gumbel_noise) / temperature, axis=-1)
   ```

2. **Start Simple**
   - First: Word-to-action copying (no variables)
   - Then: Single variable binding
   - Finally: Multiple variables and rebinding

3. **Add Diagnostics**
   - Log gradient norms for all parameters
   - Visualize attention patterns
   - Track slot usage statistics

### Commands for Tomorrow

```bash
# Continue from train_binding_mlx_proper.py
cd experiments/03_binding_architecture

# Add Gumbel-Softmax to BindingAttention
# Test gradient flow
python test_gradient_flow.py

# Try simplified task first
python train_binding_mlx_proper.py --task copying --epochs 10

# Once working, full training
python train_binding_mlx_proper.py --epochs 50 --lr 0.001
```

### Research Insight

We're at a critical juncture: The architecture is correct in principle, but the optimization challenge (gradient flow through discrete operations) is blocking progress. This mirrors a fundamental challenge in AI - how to learn discrete symbolic operations with gradient descent.

The solution (Gumbel-Softmax) bridges symbolic and connectionist approaches, allowing us to have discrete bindings (symbolic) while maintaining differentiability (connectionist).

### Status Summary
- ✅ Architecture design complete
- ✅ Implementation working (no crashes)  
- ✅ Fast inference (97k samples/sec)
- ❌ Not learning binding (gradient issue)
- 🔧 Clear fix identified (Gumbel-Softmax)

Tomorrow's priority: Make gradients flow through binding mechanism.