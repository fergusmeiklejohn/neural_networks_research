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