#!/bin/bash
# Pre-merge testing automation script
# Runs essential tests before daily merge to production

set -e  # Exit on error

echo "🧪 Starting pre-merge tests..."
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run a test and track results
run_test() {
    local test_name=$1
    local test_command=$2
    
    echo -e "\n${YELLOW}Running: $test_name${NC}"
    echo "Command: $test_command"
    
    if eval "$test_command"; then
        echo -e "${GREEN}✅ PASSED: $test_name${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}❌ FAILED: $test_name${NC}"
        ((TESTS_FAILED++))
        # Don't exit on failure, continue with other tests
    fi
}

# 1. Check Python environment
run_test "Python environment check" "python -c 'import keras; print(f\"Keras backend: {keras.backend.backend()}\")'"

# 2. Import tests
echo -e "\n${YELLOW}Testing critical imports...${NC}"
run_test "Import models" "python -c 'from models.baseline_models import BaselineModel'"
run_test "Import TTA wrappers" "python -c 'from models.test_time_adaptation.tta_wrappers import TTAWrapper'"
run_test "Import utilities" "python -c 'from utils.imports import setup_project_paths'"

# 3. Code quality checks (non-blocking)
echo -e "\n${YELLOW}Running code quality checks...${NC}"
if command -v black &> /dev/null; then
    run_test "Black formatting check" "black --check --diff . 2>/dev/null || true"
fi

if command -v flake8 &> /dev/null; then
    run_test "Flake8 linting" "flake8 --max-line-length=88 --extend-ignore=E203,W503 models/ experiments/ 2>/dev/null | head -20 || true"
fi

# 4. Quick functionality tests
echo -e "\n${YELLOW}Running quick functionality tests...${NC}"

# Test data loading
cat > /tmp/test_data_loading.py << 'EOF'
import pickle
from pathlib import Path

try:
    data_path = Path("data/physics_2ball_gravity_variations.pkl")
    if data_path.exists():
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Loaded data with {len(data['train_trajectories'])} training trajectories")
    else:
        print("⚠ Data file not found (expected for fresh clone)")
except Exception as e:
    print(f"✗ Data loading error: {e}")
    exit(1)
EOF

run_test "Data loading test" "python /tmp/test_data_loading.py"

# Test TTA weight restoration
cat > /tmp/test_tta_restoration.py << 'EOF'
import os
os.environ['KERAS_BACKEND'] = 'jax'

import keras
import numpy as np

try:
    # Create simple model
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(2)
    ])
    
    # Initialize model
    dummy_input = np.random.randn(10, 4)
    _ = model(dummy_input)
    
    # Test weight saving/restoration
    from models.test_time_adaptation.base_tta import BaseTTA
    
    class TestTTA(BaseTTA):
        def _adaptation_loss(self, outputs, inputs):
            return keras.ops.mean(outputs)
    
    tta = TestTTA(model)
    original_weights = tta._copy_weights()
    
    # Modify weights
    for var in model.trainable_variables:
        var.assign(var + 0.1)
    
    # Restore
    tta._restore_weights()
    
    # Check restoration
    max_diff = 0
    for var, orig in zip(model.variables, original_weights):
        if var.shape == orig.shape:
            diff = np.max(np.abs(var.numpy() - orig))
            max_diff = max(max_diff, diff)
    
    if max_diff < 1e-6:
        print(f"✓ Weight restoration working (max diff: {max_diff:.2e})")
    else:
        print(f"✗ Weight restoration issue (max diff: {max_diff:.2e})")
        exit(1)
        
except Exception as e:
    print(f"✗ TTA restoration test error: {e}")
    exit(1)
EOF

run_test "TTA weight restoration" "python /tmp/test_tta_restoration.py"

# 5. Check for common issues
echo -e "\n${YELLOW}Checking for common issues...${NC}"

# Check for uncommitted changes to critical files
CRITICAL_FILES="models/test_time_adaptation/base_tta.py models/test_time_adaptation/base_tta_jax.py"
for file in $CRITICAL_FILES; do
    if git diff --name-only | grep -q "$file"; then
        echo -e "${YELLOW}⚠ Warning: Uncommitted changes in $file${NC}"
    fi
done

# Check for large files
LARGE_FILES=$(find . -type f -size +10M -not -path "./.git/*" -not -path "./data/*" -not -name "*.pkl" 2>/dev/null)
if [ ! -z "$LARGE_FILES" ]; then
    echo -e "${YELLOW}⚠ Warning: Large files detected:${NC}"
    echo "$LARGE_FILES" | head -5
fi

# 6. Summary
echo -e "\n================================"
echo -e "Test Summary:"
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "\n${GREEN}✅ All tests passed! Ready for merge.${NC}"
    exit 0
else
    echo -e "\n${YELLOW}⚠ Some tests failed. Review before merging.${NC}"
    echo "You can still proceed with merge if failures are expected."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    else
        exit 1
    fi
fi