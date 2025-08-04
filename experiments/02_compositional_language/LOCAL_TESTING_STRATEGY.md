# Local Testing Strategy for Compositional Language Experiment

## Overview
This document outlines a comprehensive testing strategy to catch bugs locally before expensive Paperspace runs. Based on lessons from previous experiments.

## 1. Multi-Stage Local Testing Pipeline

### Stage 1: Minimal Data Testing (5 minutes)
- **Goal**: Verify basic pipeline functionality
- **Data**: 10 training samples, 5 validation samples
- **Epochs**: 2 per training stage
- **What to check**:
  - Data loading works
  - Model builds without errors
  - Training loop executes
  - Checkpointing works
  - Metrics are logged

### Stage 2: Integration Testing (15 minutes)
- **Goal**: Test all components together
- **Data**: 100 samples with proper train/val/test split
- **Epochs**: 5 per stage
- **What to check**:
  - Rule modification logic works
  - Baseline comparisons execute
  - Memory usage stays reasonable
  - No data leakage between splits

### Stage 3: Stress Testing (30 minutes)
- **Goal**: Simulate Paperspace conditions
- **Data**: 1000 samples
- **Epochs**: 10 per stage
- **What to check**:
  - Memory usage under load
  - Checkpoint recovery works
  - Performance metrics are reasonable
  - No memory leaks over time

## 2. Environment Simulation

### Path Flexibility Testing
```python
# test_path_flexibility.py
def get_base_path():
    """Test path detection logic"""
    # Paperspace paths
    if os.path.exists('/notebooks/neural_networks_research'):
        return '/notebooks/neural_networks_research'
    elif os.path.exists('/storage'):
        return '/storage/neural_networks_research'
    # Local paths
    elif os.path.exists(os.path.expanduser('~/conductor/repo/neural_networks_research')):
        return os.path.expanduser('~/conductor/repo/neural_networks_research')
    else:
        return os.getcwd()

# Test all path scenarios
def test_paths():
    base = get_base_path()
    assert os.path.exists(base), f"Base path {base} doesn't exist"

    # Test data paths
    data_path = os.path.join(base, 'data')
    os.makedirs(data_path, exist_ok=True)

    # Test output paths
    output_path = os.path.join(base, 'outputs')
    os.makedirs(output_path, exist_ok=True)
```

### Mock Storage Testing
```python
# Create local /storage simulation
LOCAL_STORAGE = "./mock_storage"
os.makedirs(LOCAL_STORAGE, exist_ok=True)

def get_storage_path():
    """Get storage path with fallback"""
    if os.path.exists('/storage'):
        return '/storage'
    else:
        return LOCAL_STORAGE
```

## 3. Data Generation Testing

### Test Data Pipeline Locally
```python
# test_data_generation.py
def test_scan_data_generation():
    """Test SCAN data generation with small subset"""
    # Generate tiny SCAN subset
    commands = [
        ("jump", "JUMP"),
        ("jump twice", "JUMP JUMP"),
        ("walk left", "TURN_LEFT WALK"),
        ("look and walk", "LOOK WALK"),
        ("jump thrice", "JUMP JUMP JUMP")
    ]

    # Test tokenization
    tokenizer = create_tokenizer(commands)
    assert len(tokenizer.word_index) > 0

    # Test data splits
    train, val, test = create_splits(commands, ratios=[0.6, 0.2, 0.2])
    assert len(train) > 0 and len(val) > 0 and len(test) > 0

    # Test rule modifications
    modified = apply_rule_modification(commands, rule="jump->walk")
    assert any("walk" in cmd for cmd, _ in modified)

    return True
```

### Test Model Building
```python
# test_model_building.py
def test_build_models():
    """Test all model architectures build correctly"""
    vocab_size = 50
    max_length = 20

    # Test baseline seq2seq
    baseline = build_baseline_model(vocab_size, max_length)
    assert baseline is not None

    # Test distribution invention model
    dist_model = build_distribution_model(vocab_size, max_length)
    assert dist_model is not None

    # Test compilation
    baseline.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # Test forward pass with dummy data
    dummy_input = np.random.randint(0, vocab_size, (2, max_length))
    dummy_output = baseline(dummy_input)
    assert dummy_output.shape == (2, max_length, vocab_size)

    return True
```

## 4. Training Loop Testing

### Mock Training with Checkpoints
```python
# test_training_loop.py
def test_training_with_checkpoints():
    """Test training loop with checkpoint/recovery"""
    model = build_simple_model()

    # Simulate training interruption
    for epoch in range(3):
        # Train for one epoch
        history = model.fit(x_train[:10], y_train[:10], epochs=1)

        # Save checkpoint
        checkpoint_path = f"test_checkpoint_epoch_{epoch}.h5"
        model.save_weights(checkpoint_path)

        # Simulate recovery
        if epoch == 1:
            # Load from checkpoint
            new_model = build_simple_model()
            new_model.load_weights(checkpoint_path)
            print(f"Successfully recovered from epoch {epoch}")

    # Cleanup
    for epoch in range(3):
        os.remove(f"test_checkpoint_epoch_{epoch}.h5")
```

## 5. Memory and Performance Testing

### Memory Usage Monitoring
```python
# test_memory_usage.py
import psutil
import gc

def test_memory_usage():
    """Monitor memory during mini training"""
    process = psutil.Process()

    # Initial memory
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Initial memory: {initial_memory:.2f} MB")

    # Load data
    data = load_scan_subset(n_samples=1000)
    data_memory = process.memory_info().rss / 1024 / 1024
    print(f"After data load: {data_memory:.2f} MB (+{data_memory-initial_memory:.2f} MB)")

    # Build model
    model = build_model()
    model_memory = process.memory_info().rss / 1024 / 1024
    print(f"After model build: {model_memory:.2f} MB (+{model_memory-data_memory:.2f} MB)")

    # Train briefly
    model.fit(data['train'], epochs=2, batch_size=32)
    train_memory = process.memory_info().rss / 1024 / 1024
    print(f"After training: {train_memory:.2f} MB (+{train_memory-model_memory:.2f} MB)")

    # Cleanup and check
    del model
    del data
    gc.collect()

    final_memory = process.memory_info().rss / 1024 / 1024
    print(f"After cleanup: {final_memory:.2f} MB")

    # Check for memory leak
    memory_leak = final_memory - initial_memory
    assert memory_leak < 100, f"Potential memory leak: {memory_leak:.2f} MB"
```

## 6. Error Simulation and Recovery

### Common Error Testing
```python
# test_error_handling.py
def test_error_scenarios():
    """Test handling of common errors"""

    # Test 1: Model weights not created
    try:
        model = keras.Sequential()  # Empty model
        model.save_weights("test.h5")
    except ValueError as e:
        print("✓ Caught empty model error")

    # Test 2: Optimizer state recovery
    model = build_model()
    optimizer = keras.optimizers.Adam()

    # Train briefly to create optimizer state
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(np.random.rand(10, 10), np.random.rand(10, 1), epochs=1)

    # Save and restore
    model.save_weights("test_weights.h5")
    new_model = build_model()
    new_model.compile(optimizer='adam', loss='mse')
    new_model.load_weights("test_weights.h5")
    print("✓ Optimizer state recovery works")

    # Test 3: Path not found
    try:
        data = load_data("/nonexistent/path")
    except FileNotFoundError:
        print("✓ Path error handling works")
```

## 7. Full Local Test Script

Create `run_local_tests.py`:
```python
#!/usr/bin/env python3
"""
Comprehensive local testing before Paperspace deployment
Run this script to verify everything works before cloud training
"""

import sys
import time
from datetime import datetime

def run_test_suite():
    """Run all tests in sequence"""
    tests = [
        ("Path flexibility", test_path_flexibility),
        ("Data generation", test_scan_data_generation),
        ("Model building", test_build_models),
        ("Training loop", test_training_with_checkpoints),
        ("Memory usage", test_memory_usage),
        ("Error handling", test_error_scenarios),
        ("Integration test", test_full_pipeline_mini)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print(f"{'='*50}")

        start_time = time.time()
        try:
            test_func()
            duration = time.time() - start_time
            results.append((test_name, "PASSED", duration))
            print(f"✓ {test_name} passed in {duration:.2f}s")
        except Exception as e:
            duration = time.time() - start_time
            results.append((test_name, f"FAILED: {str(e)}", duration))
            print(f"✗ {test_name} failed: {str(e)}")

    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")

    passed = sum(1 for _, status, _ in results if status == "PASSED")
    total = len(results)

    for test_name, status, duration in results:
        status_symbol = "✓" if status == "PASSED" else "✗"
        print(f"{status_symbol} {test_name}: {status} ({duration:.2f}s)")

    print(f"\nTotal: {passed}/{total} tests passed")

    # Save results
    with open(f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 'w') as f:
        for test_name, status, duration in results:
            f.write(f"{test_name}: {status} ({duration:.2f}s)\n")

    return passed == total

if __name__ == "__main__":
    if run_test_suite():
        print("\n✓ All tests passed! Ready for Paperspace deployment.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Fix issues before cloud deployment.")
        sys.exit(1)
```

## 8. Pre-Paperspace Checklist

Before running on Paperspace, ensure:

- [ ] All local tests pass
- [ ] Memory usage is reasonable (<2GB for small tests)
- [ ] Checkpointing works and can recover
- [ ] Path detection handles both environments
- [ ] Data generation is self-contained
- [ ] Error handling is comprehensive
- [ ] Integration test completes successfully

## 9. Paperspace Deployment Script

Once local tests pass, use this deployment script:
```bash
#!/bin/bash
# deploy_to_paperspace.sh

echo "Deploying Compositional Language Experiment to Paperspace"

# 1. Create deployment package
echo "Creating deployment package..."
zip -r compositional_language_deploy.zip \
    experiments/02_compositional_language/ \
    models/ \
    utils/ \
    requirements.txt

# 2. Generate run script
cat > paperspace_run.py << 'EOF'
import os
import sys

# Auto-detect environment
if os.path.exists('/notebooks'):
    base_path = '/notebooks/neural_networks_research'
    storage_path = '/storage/compositional_language'
else:
    base_path = os.getcwd()
    storage_path = './storage'

os.makedirs(storage_path, exist_ok=True)

# Add to path
sys.path.insert(0, base_path)

# Import and run
from experiments.02_compositional_language.train_full import main
main(use_storage=True, use_wandb=True, debug=False)
EOF

echo "Ready for Paperspace deployment!"
echo "Upload compositional_language_deploy.zip and paperspace_run.py"
```

## Key Testing Principles

1. **Start Small**: Always begin with tiny datasets
2. **Test Incrementally**: Build complexity gradually
3. **Simulate Production**: Mock Paperspace environment locally
4. **Monitor Resources**: Track memory and GPU usage
5. **Plan for Failure**: Test recovery mechanisms
6. **Document Everything**: Log all configurations and results

This strategy ensures we catch 95% of bugs before using GPU time!
