# Research Diary: July 19, 2025

## Today's Focus: Fixed TTA Weight Restoration Bug & Hyperparameter Tuning

### Summary
Successfully identified and fixed the weight restoration bug in Test-Time Adaptation implementation. The issue was that only trainable variables were being saved/restored, which meant BatchNormalization moving statistics weren't properly reset after adaptation. This is now fixed in both BaseTTA and BaseTTAJax classes. Also attempted hyperparameter tuning but discovered limitations with JAX backend for gradient-based adaptation.

### Key Accomplishments

1. **Identified the Bug**:
   - Created `test_weight_restoration_bug.py` to reproduce the issue
   - Confirmed that BatchNorm statistics weren't being restored
   - TENT wrapper showed incorrect predictions after reset

2. **Fixed the Bug**:
   - Updated `models/test_time_adaptation/base_tta.py`:
     - `_copy_weights()` now stores ALL variables (not just trainable)
     - `_restore_weights()` now restores ALL variables
   - Updated `models/test_time_adaptation/base_tta_jax.py` with same fixes
   - Added shape checking to prevent assignment errors

3. **Verified the Fix**:
   - Created `test_tta_fixed.py` to verify on true OOD physics data
   - Weight restoration now working correctly (error = 0.0)
   - Model properly resets to original state after adaptation

4. **Attempted Hyperparameter Tuning**:
   - Created regression-specific TTA methods (`regression_tta.py`)
   - Discovered TENT (designed for classification) doesn't work well for regression
   - Found that JAX backend limits gradient-based adaptation
   - Current TTA only updates BatchNorm statistics, not other parameters

### Technical Details

The bug was in:
```python
# OLD (BUGGY):
def _copy_weights(self):
    return {var.name: ops.convert_to_numpy(var.value) 
            for var in self.model.trainable_variables}  # Missing non-trainable!

# NEW (FIXED):
def _copy_weights(self):
    return {var.name: ops.convert_to_numpy(var.value).copy() 
            for var in self.model.variables}  # Includes BatchNorm stats
```

### Current Status

- ✅ Weight restoration bug FIXED (verified with 0.0 error)
- ✅ TTA wrapper now properly resets model state
- ❌ TTA not showing improvement on OOD data (limited by JAX backend)
- 🔍 Key discovery: TENT is inappropriate for regression tasks
- 📊 Test results on true OOD physics:
  - In-distribution MSE: 495.62
  - OOD MSE (no TTA): 808.64 (1.63x degradation)
  - OOD MSE (with TTA): 1001.51 (-23.9% worse)
- ⚠️ JAX backend limitation: Can't compute gradients easily for adaptation

### Next Steps (Priority Order)

1. **Implement Proper JAX Gradient Computation** (NEW HIGH PRIORITY):
   - Current `adapt_step_simple` only updates BatchNorm stats
   - Need to implement JAX-compatible gradient computation
   - Consider using JAX's `value_and_grad` properly
   - Or switch to TensorFlow backend for full TTA capabilities

2. **Run Full Baseline Evaluation**:
   - Train all 4 baselines on true OOD physics data
   - Compare with/without TTA
   - Use `experiments/01_physics_worlds/evaluate_tta_on_true_ood.py`

3. **Test More Extreme OOD Scenarios**:
   - Rotating frames (Coriolis forces)
   - Spring coupling (new interactions)
   - These should show clearer TTA benefits

### Key Files Modified
- `models/test_time_adaptation/base_tta.py`: Fixed weight restoration
- `models/test_time_adaptation/base_tta_jax.py`: Fixed JAX version + simplified adaptation
- `models/test_time_adaptation/base_tta_jax_v2.py`: NEW - Full JAX gradient implementation
- `models/test_time_adaptation/regression_tta.py`: New regression-specific TTA methods
- `models/test_time_adaptation/regression_tta_v2.py`: NEW - Improved regression TTA with JAX
- `experiments/01_physics_worlds/test_tta_fixed.py`: Comprehensive test
- `experiments/01_physics_worlds/test_regression_tta.py`: Regression TTA evaluation
- `experiments/01_physics_worlds/test_jax_tta_v2.py`: NEW - Test for JAX implementation
- `experiments/01_physics_worlds/implement_jax_tta.py`: Implementation plan
- `experiments/01_physics_worlds/jax_tta_implementation_summary.md`: Documentation

### Commands for Tomorrow
```bash
# Test the new JAX TTA implementation
/Users/fergusmeiklejohn/miniconda3/envs/dist-invention/bin/python experiments/01_physics_worlds/test_jax_tta_v2.py

# Re-tune hyperparameters for gradient-based TTA
/Users/fergusmeiklejohn/miniconda3/envs/dist-invention/bin/python experiments/01_physics_worlds/tune_tta_hyperparameters_v2.py

# Update TTA wrappers to use V2 implementation
# Edit models/test_time_adaptation/tta_wrappers.py

# Test on more extreme OOD scenarios
/Users/fergusmeiklejohn/miniconda3/envs/dist-invention/bin/python experiments/01_physics_worlds/generate_extreme_ood_data.py
```

### Lessons Learned
- Always save/restore ALL model variables, not just trainable ones
- BatchNorm statistics are crucial for model behavior
- Weight restoration bugs can be subtle - always verify with tests
- TENT (entropy minimization) is designed for classification, not regression
- JAX backend has limitations for gradient-based TTA without proper implementation
- Different ML backends (JAX vs TensorFlow) have different capabilities for dynamic adaptation

### Research Direction
The weight restoration fix is essential infrastructure. However, current TTA implementation is limited by:
1. **JAX Backend Limitations**: Need proper gradient computation implementation
2. **Method Mismatch**: TENT is for classification; we need regression-specific methods
3. **Minimal Adaptation**: Currently only updating BatchNorm stats, not model parameters

To move forward, we should either:
- Implement proper JAX gradient computation for full TTA
- Switch to TensorFlow backend for easier gradient-based adaptation
- Focus on BatchNorm-only adaptation and test on more extreme OOD scenarios
- Explore alternative adaptation mechanisms that don't require gradients

### Update: JAX TTA Implementation Complete! ✅

Successfully implemented full JAX gradient computation for TTA:

1. **Created `base_tta_jax_v2.py`**: 
   - Uses `model.stateless_call()` for JAX compatibility
   - Proper gradient computation with `jax.value_and_grad()`
   - Complete state management (model + optimizer)
   - Supports selective parameter updates

2. **Created `regression_tta_v2.py`**:
   - Regression-specific losses (consistency, smoothness)
   - Physics-aware version with conservation laws
   - Can update all parameters or just BatchNorm

3. **Test Results**:
   - ✅ Gradient computation: WORKING
   - ✅ State restoration: PERFECT (0.0 error)
   - ✅ BatchNorm-only mode: WORKING
   - ⚠️ Performance: Still needs hyperparameter tuning

### Key Implementation Details

The solution involved using Keras 3's stateless API:
```python
# Define loss function for JAX
def compute_loss_and_updates(trainable_vars, non_trainable_vars, x):
    y_pred, updated_non_trainable = model.stateless_call(
        trainable_vars, non_trainable_vars, x, training=True
    )
    loss = self.compute_adaptation_loss(x, y_pred)
    return loss, (y_pred, updated_non_trainable)

# Create gradient function
grad_fn = jax.value_and_grad(compute_loss_and_updates, has_aux=True)
```

### Next Steps with Full Gradient TTA
1. **Re-tune hyperparameters** - gradient-based updates need different learning rates
2. **Test on extreme OOD** - rotating frames, spring coupling
3. **Compare backends** - benchmark JAX vs TensorFlow performance
4. **Optimize adaptation losses** - current losses may be too restrictive

The infrastructure is now complete - we have full gradient-based TTA working with JAX!