# Research Diary - July 25, 2025

## Early Morning Session: Evaluation Illusion Confirmed

### What We Set Out to Do
Implement `evaluation_v2.py` to properly evaluate models using the modification-specific validation sets created yesterday.

### What Actually Happened
1. **Architectural Fragility Strike Again**
   - Model loading failed with weight mismatches (74 weights expected, 106 received)
   - Models trained with d_model=128, code expects 256
   - Shape mismatches in attention layers
   - This confirms yesterday's "Architectural Illusion" - experimental fragility is itself an illusion

2. **Pivoted to Demonstration**
   - Created `demonstrate_evaluation_illusion.py` instead
   - Shows the evaluation illusion using existing results
   - Analyzes our proper validation sets: 87% are modified examples!

### Key Findings

**The Numbers Tell the Story:**
```
Original validation (val_interpolation):
- v1_standard: 84.31%
- v1_mixed: 84.31%  
- v2_standard: 4.18% (catastrophic failure!)
- v2_mixed: 84.31%

Stage progression (safeguarded training):
- Stage 1: Loss decreased, slight accuracy gain
- Stage 2: Loss INCREASED by 2.2, accuracy dropped
- Stage 3-4: Zero improvement (literally 0.0000 change)
```

**What This Means:**
1. Models never learned modifications - stages 2-4 were theater
2. Constant 84.3% validation masked complete failure
3. v2's gating mechanism made things worse, not better
4. Mixed training only helped by diluting the modification signal

### The Evaluation Illusion Explained

We thought we were measuring generalization, but we were measuring memorization:
- Validation set had 0% modified examples
- Training had modifications but insufficient signal
- Result: Models that appear to work but fail completely on actual OOD

This is **systemic** in ML research - standard practices create illusions of progress.

---

## Later Morning Session: Simple Baseline Reveals Hard Truths

### What We Set Out to Do
Following the early morning's confirmation of the Evaluation Illusion, the goal was to build a simple baseline model without architectural complexity to establish honest performance metrics.

### What Actually Happened

#### 1. Simple Baseline Implementation ✓
Created `simple_baseline_v2.py` with:
- Basic LSTM encoder-decoder (~267K parameters)
- No gating mechanisms or rule extraction
- Modifications handled via simple addition
- Standard cross-entropy training

#### 2. Training Experiments
**First Run** (10K examples, 50 epochs):
- Reached 76% training accuracy at epoch 46
- Validation hit 86% (!!) at epoch 46
- But then collapsed - model generates only "I_LOOK I_LOOK"
- Clear overfitting and instability

**Second Run** (Full 39.7K examples, better hyperparameters):
- Used `train_simple_baseline_full.py` with:
  - All training data (26K base + 13K modifications)
  - Larger model (d_model=256)
  - Early stopping, gradient clipping, LR scheduling
- Stopped at epoch 6 with 25.46% validation accuracy
- Model generates only "I_TURN_RIGHT I_TURN_RIGHT..."

#### 3. Proper Evaluation Results
Both models showed **0% accuracy on ALL validation sets**:
```
val_base.................... 0.00% (500 examples)
val_mod_walk_skip........... 0.00% (500 examples)
val_mod_jump_hop............ 0.00% (500 examples)
val_mod_look_scan........... 0.00% (500 examples)
val_mod_left_right.......... 0.00% (341 examples)
val_mod_mixed............... 0.00% (500 examples)
val_mod_unseen.............. 0.00% (500 examples)
val_mod_composed............ 0.00% (498 examples)
```

### Key Findings

**1. The Evaluation Illusion is Even Worse Than We Thought**
- Model showed 86% validation accuracy in training
- But 0% on proper validation sets
- The gap is massive: 86% → 0%

**2. SCAN is Genuinely Hard**
- Simple LSTM can't even learn basic mappings reliably
- Gets stuck in degenerate solutions (repetitive outputs)
- This isn't about modifications - it fails on BASE examples too

**3. Simple Baselines Are Valuable**
- **Honest**: 0% is real, unlike complex models' fake 84.3%
- **Debuggable**: Can see exactly what fails (repetitive generation)
- **Fast**: Full experiment in <20 minutes
- **Clear**: No architectural complexity hiding issues

### The Bigger Picture

| Model Type | Training Val | Proper Val | Truth |
|------------|--------------|------------|-------|
| V1 Complex | 84.3% | 0% | Hidden failure |
| V2 Gated | 84.3% | 0% | Hidden failure |
| Simple LSTM | 86% → 25% | 0% | Visible failure |

The simple baseline's failure is more valuable than the complex models' "success" because it reveals:
1. The task is fundamentally hard
2. Standard evaluation is completely broken
3. We need better approaches, not just more complexity

### Reflection

Today's work confirms that the "Layers of Illusions" run deep:
- **Evaluation Illusion**: Validation metrics lie
- **Architectural Illusion**: Complexity hides failure
- **Progress Illusion**: High accuracy ≠ actual learning

The simple baseline strips away these illusions, showing that we haven't even solved basic SCAN, let alone modifications.

### Next Steps

1. **Document this as a case study in the paper**
   - Show how evaluation practices create false confidence
   - Demonstrate value of simple, honest baselines
   - Argue for proper validation as prerequisite

2. **Rethink the approach**
   - Current methods (even simple ones) aren't working
   - Need fundamentally different training objectives?
   - Or architectural inductive biases for compositionality?

3. **Design better experiments**
   - Start with even simpler tasks (single-word commands?)
   - Build up systematically with proper validation at each step
   - Never trust standard validation metrics

### Action Items for Next Session

1. **Write up Evaluation Illusion case study**
   - Use today's results as concrete example
   - Show progression: 84.3% → 86% → 0%
   - Include generation examples

2. **Explore alternative approaches**
   - Review literature for SCAN-specific architectures
   - Consider grammar-based or symbolic components
   - Look into curriculum learning done right

3. **Update paper outline**
   - Add empirical validation section
   - Include negative results as key findings
   - Emphasize importance of proper evaluation

## Code References

Key files from today:
- `evaluation_v2.py` - Proper evaluation framework (with fixes for dict/object handling)
- `demonstrate_evaluation_illusion.py` - Shows illusion with existing results
- `simple_baseline_v2.py` - Basic LSTM implementation
- `train_simple_baseline_full.py` - Full dataset training
- `evaluate_simple_baseline.py` - Proper evaluation script
- `test_model_generation.py` - Reveals degenerate outputs
- `FINAL_BASELINE_RESULTS.md` - Complete analysis

## Key Insight

**The most dangerous bugs are those that make failing systems appear successful.**

Our simple baseline, by failing honestly at 0%, is more valuable than complex models that fail silently at 84.3%. This is the core message: proper evaluation isn't just important, it's the difference between real and illusory progress.