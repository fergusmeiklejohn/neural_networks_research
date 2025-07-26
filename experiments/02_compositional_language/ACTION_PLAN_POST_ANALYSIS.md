# Action Plan: Fixing Compositional Language Experiments

## Critical Discovery

Our comprehensive analysis reveals a fundamental issue with the experimental setup:

**The validation set contains NO modified examples**, making it impossible to measure whether models are learning to apply modifications. This explains the constant 84.3% validation accuracy across all experiments.

## Key Findings

### 1. Validation Set Problem ⚠️
- **Symptom**: All models show exactly 84.3% validation accuracy throughout training
- **Cause**: Validation set only contains standard SCAN examples
- **Impact**: Cannot measure modification performance

### 2. Model Behavior Patterns
- **v1 Models**: Maintain base SCAN performance but likely ignore modifications
- **v2_standard**: Complete failure (4.2%) - gating mechanism blocks all learning
- **v2_mixed**: Recovers to 84.3% but modification behavior unknown
- **Training Loss**: Increases when modifications introduced, then plateaus immediately

### 3. Convergence Patterns
- Models converge instantly (0.000000 speed) after Stage 1
- Suggests finding compromise solutions that ignore modifications
- No gradual adaptation to new rules

## Immediate Actions Required

### 1. Fix Evaluation Infrastructure (Priority: CRITICAL)

Create new evaluation module with proper modification testing:

```python
# experiments/02_compositional_language/evaluation_v2.py

class ModificationEvaluator:
    def __init__(self):
        self.base_test_set = []      # Standard SCAN
        self.modified_test_sets = {   # Modified versions
            'reverse': [],
            'double': [],
            'skip': []
        }
    
    def evaluate(self, model):
        return {
            'base_accuracy': self.eval_base(model),
            'modification_accuracy': self.eval_modifications(model),
            'consistency_score': self.eval_consistency(model)
        }
```

### 2. Create Diagnostic Training Script

```python
# experiments/02_compositional_language/train_with_diagnostics.py

# Key additions:
# 1. Log predictions on specific test cases each epoch
# 2. Track gate activations for v2 models
# 3. Separate metrics for base vs modified examples
# 4. Save example predictions for analysis
```

### 3. Simplified Initial Test

Before full experiments, run minimal test:
- Single modification type (e.g., "reverse" only)
- 1000 training examples
- Clear separation of base/modified in batches
- Log every prediction

## Proposed Experiment Redesign

### Phase 1: Validate Modification Learning (1-2 days)
1. Create proper test sets with modifications
2. Implement diagnostic training with detailed logging
3. Train on single modification type
4. Verify models can learn ANY modification

### Phase 2: Fix Architecture Issues (2-3 days)
1. Debug v2 gating mechanism
2. Add gate regularization to prevent over-blocking
3. Initialize gates to be more "open"
4. Test simplified gating alternatives

### Phase 3: Run Corrected Experiments (3-4 days)
1. Train with proper validation sets
2. Use diagnostic logging throughout
3. Compare all baselines fairly
4. Generate modification-specific metrics

## Technical Fixes Needed

### 1. Data Generation
```python
# In scan_data_generator.py
def create_validation_data(self, include_modifications=True):
    """Create validation set with both base and modified examples."""
    base_val = self.get_base_validation()
    
    if include_modifications:
        modified_val = self.apply_modifications(base_val)
        return mix_datasets(base_val, modified_val, ratio=0.5)
    
    return base_val
```

### 2. Training Loop
```python
# In training script
# Track separate metrics
base_metrics = []
mod_metrics = []

for batch in train_data:
    if batch.has_modifications:
        mod_metrics.append(train_step(batch))
    else:
        base_metrics.append(train_step(batch))

# Log both separately
wandb.log({
    'base_accuracy': np.mean(base_metrics),
    'modification_accuracy': np.mean(mod_metrics)
})
```

### 3. Model Architecture
```python
# For v2 models, add gate monitoring
class GatedModificationLayer(keras.layers.Layer):
    def call(self, inputs, training=None):
        # ... existing code ...
        
        # Add gate activation tracking
        self.add_metric(
            tf.reduce_mean(gate_values),
            name='gate_activation_mean'
        )
        
        return modified_output
```

## Success Criteria

Before proceeding with full experiments, we must verify:

1. ✓ Validation set includes modified examples
2. ✓ Can measure modification-specific accuracy
3. ✓ Models show different behavior with/without modifications
4. ✓ v2 models can learn basic SCAN with gating enabled
5. ✓ Training logs capture modification performance

## Timeline

- **Day 1**: Fix validation data and evaluation metrics
- **Day 2**: Create diagnostic training script
- **Day 3**: Run minimal tests to verify modification learning
- **Day 4-5**: Fix architectural issues based on findings
- **Day 6-8**: Run full corrected experiments
- **Day 9**: Analyze results and prepare report

## Next Immediate Step

Create `evaluation_v2.py` with proper modification testing:

```bash
cd experiments/02_compositional_language
python create_evaluation_v2.py  # To be created
```

This will give us the tools to properly measure what our models are actually learning.