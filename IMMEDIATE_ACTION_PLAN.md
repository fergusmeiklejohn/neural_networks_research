# Immediate Action Plan: Breaking Through the OOD Illusion

Generated: 2025-07-27

## Context

We've discovered that our models achieving 84.3% accuracy on SCAN were completely failing (0%) on actual modifications. Similarly, our physics models showing good "OOD" performance were actually just interpolating. The recent literature (2023-2025) reveals why and shows paths forward.

## Priority 1: Minimal Variable Binding Demo (2-3 days)

### Why This First?
The literature reveals variable binding is THE fundamental missing piece. Without it, no amount of architectural complexity will enable true modification.

### Implementation Plan

```python
# File: experiments/03_binding_architecture/minimal_binding_scan.py

class MinimalBindingModel:
    """
    Based on Wu et al. (2025) - Transformers learning variable binding
    Key insight: Force binding through dereferencing tasks
    """
    def __init__(self):
        # Explicit variable slots (not just embeddings)
        self.variable_memory = VariableMemory(n_slots=10)
        
        # Binding mechanism (associates words with slots)
        self.binder = BindingAttention()
        
        # Execution with bound variables
        self.executor = BoundVariableExecutor()
    
    def forward(self, command, modification=None):
        # Parse command into variables
        variables = self.parse_to_variables(command)
        
        # Bind variables to memory slots
        bindings = self.binder(variables, self.variable_memory)
        
        # Apply modifications to bindings
        if modification:
            bindings = self.apply_modification(bindings, modification)
        
        # Execute with modified bindings
        return self.executor(bindings)
```

### Specific Tasks:
1. Implement variable slot memory (2-4 hours)
2. Create binding attention mechanism (3-4 hours)
3. Build dereferencing training tasks (4-6 hours)
4. Test on simple modifications: "jump" → "hop" (2-3 hours)
5. Compare to our failed complex architectures (1-2 hours)

### Success Criteria:
- Model can bind "jump" to a variable slot
- Can modify that binding to "hop"
- Achieves >50% on single-modification validation set

## Priority 2: Fix Physics Evaluation with Convex Hull Analysis (1-2 days)

### Why This Second?
We need to verify which of our "successes" were actually interpolation to avoid wasting time on false positives.

### Implementation Plan

```python
# File: evaluation/true_ood_verifier.py

class TrueOODVerifier:
    """
    Based on Li et al. (2025) - Most OOD is interpolation
    """
    def analyze_dataset(self, train_data, test_data, model):
        # Get representations
        train_repr = model.get_representations(train_data)
        test_repr = model.get_representations(test_data)
        
        # Compute convex hull
        hull = ConvexHull(train_repr)
        
        # Classify each test point
        results = {
            'interpolation': [],
            'near_extrapolation': [],
            'far_extrapolation': []
        }
        
        for point in test_repr:
            distance = distance_to_hull(point, hull)
            if distance < 0.01:
                category = 'interpolation'
            elif distance < 0.1:
                category = 'near_extrapolation'
            else:
                category = 'far_extrapolation'
            results[category].append(point)
        
        return results
```

### Specific Tasks:
1. Implement convex hull analysis (2-3 hours)
2. Apply to our physics "OOD" results (2-3 hours)  
3. Visualize interpolation vs extrapolation (2-3 hours)
4. Re-evaluate all baseline results (3-4 hours)
5. Document true OOD requirements (1-2 hours)

## Priority 3: NSR-Style Parser for SCAN (3-4 days)

### Why This Third?
NSR achieved >90% on SCAN by using explicit symbolic parsing. This is our most direct path to success.

### Implementation Plan

```python
# File: models/nsr_scan_parser.py

class NSRScanParser:
    """
    Based on Li et al. (2024) - Neural-Symbolic Recursive Machine
    Key: Explicit symbolic intermediate representation
    """
    def __init__(self):
        # Neural perception: string → symbols
        self.perception = nn.LSTM(input_size=vocab_size, hidden_size=256)
        
        # Symbolic parser: symbols → parse tree
        self.parser = SymbolicParser(
            grammar=SCAN_GRAMMAR,
            max_depth=5
        )
        
        # Semantic executor: parse tree → actions
        self.executor = SemanticExecutor()
        
        # Deduction-abduction learner
        self.rule_learner = DeductionAbduction()
    
    def parse_command(self, command):
        # Neural → Symbolic
        symbols = self.perception(command)
        
        # Build parse tree
        tree = self.parser(symbols)
        
        # Return structured representation
        return tree
    
    def execute(self, tree, modifications=None):
        # Apply modifications at symbolic level
        if modifications:
            tree = self.apply_symbolic_modifications(tree, modifications)
        
        # Execute symbolically
        return self.executor(tree)
```

### Specific Tasks:
1. Define SCAN grammar in symbolic form (3-4 hours)
2. Implement neural-to-symbolic perception (4-5 hours)
3. Build parse tree constructor (4-6 hours)
4. Create symbolic executor (3-4 hours)
5. Add deduction-abduction learning (6-8 hours)
6. Test on SCAN with modifications (2-3 hours)

## Priority 4: Simple Meta-Learning Experiment (2-3 days)

### Why This Fourth?
MLC showed that HOW we train matters as much as architecture. We should test if meta-learning helps even our simple models.

### Implementation Plan

```python
# File: experiments/meta_learning/simple_mlc_scan.py

def meta_learning_episode():
    """
    Based on Lake & Baroni (2023) - MLC
    Each episode teaches compositional generalization
    """
    # Sample a new "word" (e.g., "blip" means jump)
    new_word, meaning = sample_novel_word()
    
    # Create support set (few examples)
    support = [
        (f"{new_word}", meaning),
        (f"{new_word} twice", f"{meaning} {meaning}"),
        (f"{new_word} left", f"TURN_LEFT {meaning}")
    ]
    
    # Create query set (novel combinations)
    query = [
        (f"{new_word} around right", f"TURN_RIGHT {meaning} TURN_RIGHT {meaning}"),
        (f"{new_word} and walk", f"{meaning} WALK")
    ]
    
    # Meta-update to improve compositional learning
    loss = model.adapt(support, query)
    meta_optimizer.step(loss)
```

### Specific Tasks:
1. Design compositional curriculum (2-3 hours)
2. Implement episode generator (3-4 hours)
3. Add meta-optimization loop (2-3 hours)
4. Test on simple LSTM baseline (2-3 hours)
5. Compare to standard training (1-2 hours)

## Week 1 Schedule

### Monday (Day 1)
- Morning: Start minimal binding implementation
- Afternoon: Complete variable memory and binding attention

### Tuesday (Day 2)  
- Morning: Implement dereferencing tasks
- Afternoon: Test binding on simple modifications

### Wednesday (Day 3)
- Morning: Implement convex hull analyzer
- Afternoon: Apply to physics results, create visualizations

### Thursday (Day 4)
- Morning: Start NSR parser - define grammar
- Afternoon: Implement neural-to-symbolic perception

### Friday (Day 5)
- Morning: Complete NSR parse tree and executor
- Afternoon: Run initial tests, document results

## Success Metrics for Week 1

1. **Binding Demo**: >50% accuracy on single modifications
2. **Convex Hull**: Clear classification of our "OOD" results  
3. **NSR Parser**: Successfully parses basic SCAN commands
4. **Progress**: Move beyond 0% on SCAN modifications

## Key Principles

1. **Start Simple**: Minimal implementations that demonstrate core concepts
2. **Test Immediately**: Don't wait for perfect implementation
3. **Document Everything**: Both successes and failures are valuable
4. **Stay Focused**: Resist adding complexity before basics work

## Expected Outcomes

By end of Week 1, we should have:
- Proof that variable binding enables modifications
- Clear understanding of what was interpolation vs extrapolation
- Working symbolic parser for SCAN
- Foundation for meta-learning experiments

This positions us to tackle the full distribution invention challenge with proper foundations instead of complex architectures that hide fundamental limitations.