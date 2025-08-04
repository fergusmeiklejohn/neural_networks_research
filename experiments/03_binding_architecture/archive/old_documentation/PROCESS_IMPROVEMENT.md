# Process Improvement: Preventing API Errors

## Issue Identified
The error `AttributeError: module 'keras.utils' has no attribute 'function'` slipped through our initial implementation because:

1. **No execution testing**: Code was written but not run before committing
2. **Incorrect API assumptions**: Used non-existent Keras 3 APIs
3. **Environment mismatch**: Testing wasn't done in the correct conda environment

## Process Improvements Implemented

### 1. Mandatory Pre-Commit Testing
Created `test_minimal_training.py` that must pass before committing any training code:
- Tests all major components (model creation, training step, evaluation)
- Runs with minimal data (20 samples) for quick feedback
- Returns exit code 0/1 for CI integration

### 2. Environment-Aware Testing
Always use the correct Python environment:
```bash
/Users/fergusmeiklejohn/miniconda3/envs/dist-invention/bin/python test_script.py
```

### 3. API-Aware Implementation
For Keras 3 compatibility:
- Avoid backend-specific code (GradientTape, @tf.function)
- Use model.compile() and model.fit() for standard training
- Override compute_loss() for custom loss computation
- Test on actual Keras 3 installation before assuming APIs exist

### 4. Simplified Implementation First
Created `train_binding_model_simple.py` that:
- Uses Keras built-in training loop
- Avoids manual gradient computation
- Works across all backends (JAX, TensorFlow, PyTorch)

## Updated Workflow

1. **Write code** with API documentation open
2. **Run minimal test** immediately:
   ```bash
   /Users/fergusmeiklejohn/miniconda3/envs/dist-invention/bin/python test_minimal_training.py
   ```
3. **Fix any errors** before proceeding
4. **Run full training** only after tests pass
5. **Commit** with confidence

## Checklist for Future Implementations

- [ ] Check Keras 3 documentation for API changes
- [ ] Write test script that exercises all code paths
- [ ] Run test in correct environment before committing
- [ ] Use built-in Keras training when possible
- [ ] Document any backend-specific requirements

This process ensures we catch API errors early and maintain working code throughout development.
