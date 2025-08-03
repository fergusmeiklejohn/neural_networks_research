# Neural Architectures for Learning Variable Bindings: Exploration Guide

## Core Problem Statement

We need a neural architecture that can:
1. **Learn** that "X means jump" creates a binding between variable X and action jump
2. **Store** these bindings throughout the command processing
3. **Apply** the correct binding when seeing "do X" later in the sequence
4. **Handle** complex patterns:
   - Multiple bindings: "X means jump Y means walk"
   - Rebinding: "X means jump do X then X means walk do X"
   - Compositional structures: "do X and Y then X"

## Why Current Approaches Fail

### 1. Implicit Encoding Failure
Current transformer models try to encode bindings implicitly in hidden states:
- Loss of information over distance
- No explicit storage mechanism
- Confusion between similar patterns

### 2. Lack of Sequential Processing
Commands like "X means jump do X then Y means walk do Y" require:
- Processing "X means jump" before "do X"
- Updating context after "then"
- Maintaining separate binding scopes

### 3. No Explicit Memory
Without memory, models must:
- Recompute bindings at every step
- Risk forgetting distant bindings
- Cannot handle rebinding properly

## Architectures to Explore and Test

### 1. Differentiable Neural Memory Networks

**Core Idea**: Explicit key-value memory for variable bindings

```python
class NeuralMemoryBinding:
    def __init__(self):
        # Fixed slots for variables X, Y, Z, W
        self.memory_keys = nn.Embedding(4, key_dim)
        self.memory_values = nn.Parameter(torch.zeros(4, value_dim))
        self.write_gate = nn.Sequential(
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid()
        )
        self.read_attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
```

**Implementation Plan**:
1. Process command with LSTM/GRU sequentially
2. On "X means jump": 
   - Encode "jump" → value vector
   - Write to memory[X] with soft gating
3. On "do X":
   - Read from memory[X] using attention
   - Decode to action prediction

**Test Strategy**:
- Start with fixed 4 variables
- Test on progressively complex patterns
- Measure: Can it remember bindings over 20+ tokens?
- Ablation: Remove memory → measure accuracy drop

### 2. Cross-Attention Binding Architecture

**Core Idea**: Learn to attend to binding declarations when executing

```python
class CrossAttentionBinder:
    def __init__(self):
        self.binding_encoder = TransformerEncoder(num_layers=2)
        self.execution_encoder = TransformerEncoder(num_layers=2)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
    def forward(self, tokens):
        # Step 1: Identify and encode all bindings
        binding_mask = find_binding_patterns(tokens)  # "X means Y"
        binding_representations = self.binding_encoder(tokens, binding_mask)
        
        # Step 2: Encode execution commands
        exec_mask = find_execution_patterns(tokens)  # "do X"
        exec_representations = self.execution_encoder(tokens, exec_mask)
        
        # Step 3: Cross-attend from execution to bindings
        attended_bindings = self.cross_attention(
            query=exec_representations,
            key=binding_representations,
            value=binding_representations
        )
        
        # Step 4: Predict actions
        return self.action_predictor(attended_bindings)
```

**Implementation Plan**:
1. Pre-process to identify binding vs execution tokens
2. Separate encoding paths for clarity
3. Learn attention patterns: "do X" should attend to "X means jump"
4. Combine attended information for prediction

**Test Strategy**:
- Visualize attention weights: Does "do X" attend to correct binding?
- Test with distractors: "X means jump Y means walk do X"
- Measure long-range attention: 50+ token sequences

### 3. Pointer Network for Binding Resolution

**Core Idea**: Point to where variables are defined instead of predicting actions

```python
class PointerBindingNetwork:
    def __init__(self):
        self.encoder = BiLSTM(hidden_dim)
        self.pointer_attention = PointerAttention(hidden_dim)
        
    def forward(self, tokens):
        # Encode entire sequence
        encoded = self.encoder(tokens)  # [seq_len, hidden_dim]
        
        outputs = []
        for i, token in enumerate(tokens):
            if is_variable_reference(token):  # e.g., "X" in "do X"
                # Compute attention over all positions
                attention_scores = self.pointer_attention(
                    query=encoded[i],
                    keys=encoded
                )
                
                # Find where this variable was defined
                pointed_position = argmax(attention_scores)
                
                # Extract action from definition
                action = extract_action_from_definition(tokens, pointed_position)
                outputs.append(action)
                
        return outputs
```

**Implementation Plan**:
1. Train to point to binding definitions
2. Use curriculum learning: start with bindings close to usage
3. Gradually increase distance between binding and usage
4. Add regularization to encourage pointing to "means" patterns

**Test Strategy**:
- Accuracy of pointing: Does it point to correct definition?
- Multiple definitions: "X means jump ... X means walk"
- Robustness: Add noise tokens between binding and usage

### 4. Two-Stage Compiler Architecture

**Core Idea**: Separate binding compilation from execution

```python
class TwoStageBindingModel:
    def __init__(self):
        self.binding_compiler = BindingCompiler()
        self.neural_executor = NeuralExecutor()
        
    def forward(self, tokens):
        # Stage 1: Compile all bindings into structured form
        binding_table = self.binding_compiler(tokens)
        # Returns: {"X": jump_embedding, "Y": walk_embedding}
        
        # Stage 2: Execute with binding table
        outputs = self.neural_executor(tokens, binding_table)
        return outputs

class BindingCompiler(nn.Module):
    def forward(self, tokens):
        # Find all "VAR means ACTION" patterns
        patterns = find_binding_patterns(tokens)
        
        binding_table = {}
        for var, action in patterns:
            # Encode action into embedding
            action_embedding = self.action_encoder(action)
            binding_table[var] = action_embedding
            
        return binding_table
```

**Implementation Plan**:
1. Pre-train compiler on synthetic binding extraction
2. Train executor with frozen compiler initially
3. Fine-tune end-to-end
4. Add noise to test robustness

**Test Strategy**:
- Compiler accuracy: Does it extract all bindings correctly?
- Executor performance: Given perfect bindings, can it execute?
- End-to-end: How much does joint training improve?
- Interpretability: Visualize extracted binding table

### 5. Graph Neural Network Architecture

**Core Idea**: Represent bindings as edges in a graph

```python
class GraphBindingNetwork:
    def __init__(self):
        self.node_encoder = nn.Embedding(vocab_size, hidden_dim)
        self.edge_encoder = nn.Embedding(edge_types, hidden_dim)
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim) for _ in range(4)
        ])
        
    def build_graph(self, tokens):
        # Create nodes for each token
        nodes = [self.node_encoder(t) for t in tokens]
        
        # Add edges for bindings
        edges = []
        for i, token in enumerate(tokens):
            if tokens[i:i+2] == ["means"]:
                # Add binding edge from var to action
                var_idx = i - 1
                action_idx = i + 1
                edges.append((var_idx, action_idx, BINDING_EDGE))
                
        return Graph(nodes, edges)
```

**Implementation Plan**:
1. Parse command into graph structure
2. Add different edge types: binding, sequence, composition
3. Use message passing to propagate binding information
4. Aggregate at variable nodes for prediction

**Test Strategy**:
- Graph construction accuracy
- Information propagation: Do variables receive action information?
- Complex graphs: Multiple bindings, rebinding
- Comparison with sequential models

### 6. Hierarchical Binding Processor

**Core Idea**: Process bindings hierarchically with scoping

```python
class HierarchicalBindingProcessor:
    def __init__(self):
        self.segment_processor = SegmentProcessor()
        self.binding_merger = BindingMerger()
        
    def process(self, tokens, parent_bindings=None):
        # Split by "then" for sequential processing
        segments = split_by_then(tokens)
        
        current_bindings = parent_bindings or {}
        outputs = []
        
        for segment in segments:
            # Extract local bindings
            local_bindings = extract_bindings(segment)
            
            # Merge with current bindings (local overwrites)
            merged_bindings = self.binding_merger(
                current_bindings, local_bindings
            )
            
            # Process segment with merged bindings
            segment_outputs = self.segment_processor(
                segment, merged_bindings
            )
            outputs.extend(segment_outputs)
            
            # Update current bindings
            current_bindings = merged_bindings
            
        return outputs
```

**Implementation Plan**:
1. Implement recursive descent parser
2. Track binding scopes explicitly
3. Handle "then" as scope boundary
4. Test on nested structures

**Test Strategy**:
- Scope handling: Are bindings properly scoped?
- Rebinding: Does local override parent?
- Complex nesting: Multiple levels of "then"
- Performance on flat vs nested structures

## Evaluation Framework

### 1. Progressive Complexity Dataset

Create controlled datasets with increasing complexity:

```python
levels = {
    "level_1": [
        ("X means jump do X", ["JUMP"]),
        ("Y means walk do Y", ["WALK"])
    ],
    "level_2": [
        ("X means jump Y means walk do X and Y", ["JUMP", "WALK"]),
        ("X means run Y means turn do Y then X", ["TURN", "RUN"])
    ],
    "level_3": [
        ("X means jump do X then X means walk do X", ["JUMP", "WALK"]),
        ("X means jump do X twice then Y means walk do Y", ["JUMP", "JUMP", "WALK"])
    ],
    "level_4": [
        ("X means jump do X then Y means walk do Y and X", ["JUMP", "WALK", "JUMP"]),
        ("X means jump Y means walk do X and Y then X or Y", ["JUMP", "WALK", "JUMP"])
    ]
}
```

### 2. Diagnostic Metrics

Beyond accuracy, measure:
- **Binding Extraction F1**: How well does model identify bindings?
- **Binding Application Accuracy**: Given correct bindings, execution accuracy?
- **Long-range Dependency**: Accuracy vs distance between binding and usage
- **Rebinding Accuracy**: Performance on variable reassignment
- **Compositional Generalization**: Novel combinations of known patterns

### 3. Ablation Studies

For each architecture:
- Remove key component (memory, attention, etc.)
- Measure performance drop
- Identify critical components
- Test simplified versions

## Implementation Strategy

### Phase 1: Proof of Concept (Week 1)
1. Implement Memory Network with 4 fixed variables
2. Test on Level 1-2 complexity
3. Establish baseline performance
4. Identify key challenges

### Phase 2: Advanced Architectures (Week 2-3)
1. Implement Cross-Attention and Pointer Network
2. Compare performance on all levels
3. Analyze failure modes
4. Select best 2 approaches

### Phase 3: Optimization and Scaling (Week 4)
1. Combine best ideas from different architectures
2. Scale to more variables (8+)
3. Test on real-world-like commands
4. Optimize for inference speed

## Expected Outcomes

### Memory Network
- **Pros**: Explicit storage, interpretable
- **Cons**: Fixed variable slots, scaling issues
- **Expected**: 80-90% on simple, 60-70% on complex

### Cross-Attention
- **Pros**: Flexible, scalable
- **Cons**: Requires learning attention patterns
- **Expected**: 70-80% on simple, 80-90% on complex

### Pointer Network
- **Pros**: Elegant, no separate memory
- **Cons**: Harder to train
- **Expected**: 60-70% on simple, 70-80% on complex

### Two-Stage
- **Pros**: Modular, debuggable
- **Cons**: Error propagation
- **Expected**: 90%+ on simple, 85%+ on complex

## Key Insights

The fundamental breakthrough needed is making binding operations **explicit** rather than implicit. Current models fail because they treat variable binding as a side effect rather than a first-class operation.

The most promising approach combines:
1. **Explicit binding extraction** (like Two-Stage compiler)
2. **Attention-based binding application** (like Cross-Attention)
3. **Sequential processing** for proper "then" handling
4. **Memory or pointing** for long-range dependencies

By making these operations explicit and differentiable, we can achieve the >90% accuracy target on compositional commands.