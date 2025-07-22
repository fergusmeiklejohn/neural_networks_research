# Compositional Language Experiment - Local Testing Guide

## Overview
We've created a comprehensive local testing framework to catch bugs before expensive Paperspace runs. This guide shows how to test everything locally.

## Components Created

1. **Local Testing Strategy** (`LOCAL_TESTING_STRATEGY.md`)
   - Comprehensive testing plan with 3 stages
   - Memory monitoring and error simulation
   - Pre-Paperspace checklist

2. **Test Suite** (`run_local_tests.py`)
   - 7 comprehensive tests covering all components
   - Environment setup, data generation, model building
   - Memory usage monitoring and checkpoint recovery

3. **Data Generator** (`prepare_scan_data.py`)
   - Self-contained SCAN data generation
   - Rule modifications (5 types)
   - Works both locally and on Paperspace

4. **Training Script** (`train_compositional_model.py`)
   - Baseline seq2seq + distribution invention components
   - Automatic path detection (local vs Paperspace)
   - Comprehensive checkpointing and logging

## Quick Start - Local Testing

### 1. Generate Test Data (tiny subset)
```bash
cd experiments/02_compositional_language
python prepare_scan_data.py --subset 100 --output ./test_data
```

### 2. Run Comprehensive Tests
```bash
./run_local_tests.py
```

This will:
- Test environment setup
- Verify path flexibility
- Test data generation
- Build and test models
- Test checkpointing/recovery
- Monitor memory usage
- Run integration test

### 3. Test Training Pipeline (minimal)
```bash
# First generate slightly larger test data
python prepare_scan_data.py --subset 500 --output ./data

# Run training in test mode (2 epochs, small model)
python train_compositional_model.py --test
```

## Expected Test Results

All tests should pass with output like:
```
✓ Environment Setup: PASSED (0.52s)
✓ Path Flexibility: PASSED (0.03s)
✓ SCAN Data Generation: PASSED (0.12s)
✓ Model Building: PASSED (1.24s)
✓ Checkpoint/Recovery: PASSED (2.31s)
✓ Memory Usage: PASSED (3.45s)
✓ Integration Test: PASSED (1.02s)

Overall: 7/7 tests passed
Total time: 8.69s

✅ ALL TESTS PASSED - Ready for Paperspace deployment!
```

## Memory Usage Guidelines

Local testing should show:
- Initial memory: ~200-400 MB
- After data load: +50-100 MB
- After model build: +100-200 MB
- After training: +50-100 MB
- Net increase: <300 MB

If memory usage exceeds these ranges, investigate before Paperspace deployment.

## Common Issues and Solutions

1. **Import Errors**
   - Ensure you're in the conda environment: `conda activate dist-invention`
   - Run from project root or use full paths

2. **Memory Issues**
   - Reduce subset size for data generation
   - Decrease batch size in training

3. **Keras Backend Issues**
   - Check `~/.keras/keras.json` for correct backend
   - Set `KERAS_BACKEND=jax` explicitly

## Paperspace Deployment

Once all local tests pass:

1. **Create deployment package**:
```bash
cd experiments/02_compositional_language
zip -r compositional_deploy.zip *.py data/ LOCAL_TESTING_STRATEGY.md
```

2. **Create Paperspace run script**:
```python
# paperspace_run.py
import subprocess
import sys

# Generate full dataset
subprocess.run([sys.executable, "prepare_scan_data.py", "--subset", "10000"])

# Run full training
subprocess.run([sys.executable, "train_compositional_model.py", 
                "--epochs", "100", "--batch-size", "64"])
```

3. **Upload and run on Paperspace**:
- Upload the zip file and run script
- Monitor GPU usage and memory
- Check /storage for saved checkpoints

## Key Lessons Applied

From previous experiments, we've implemented:
- ✅ Self-contained data generation (no git issues)
- ✅ Frequent checkpointing to /storage
- ✅ Path flexibility for different environments
- ✅ Memory monitoring and cleanup
- ✅ Comprehensive error handling
- ✅ Test mode for quick validation

This approach should prevent the issues encountered in previous Paperspace runs!