# Paperspace Script Validation Process

## The Problem
We've been pushing scripts to Paperspace with runtime errors that waste GPU hours and break our workflow. Two recent examples:
1. `ImportError: cannot import name 'create_model' from 'train_progressive_minimal'`
2. `AttributeError: 'SCANTokenizer' object has no attribute 'save'`

Both could have been caught with proper local testing.

## Mandatory Validation Process

**NEVER push a script to Paperspace without completing ALL these steps:**

### 1. Pre-Flight Checklist ✈️

```bash
# Activate the correct environment
source ~/miniconda3/etc/profile.d/conda.sh && conda activate dist-invention

# Navigate to experiment directory
cd experiments/02_compositional_language/
```

### 2. Static Validation (Catches syntax and import errors)

```bash
# Run the comprehensive validator
python validate_before_paperspace.py your_script.py

# Must see: "✅ VALIDATION PASSED"
```

This catches:
- Syntax errors
- Import errors
- Missing dependencies
- Code style issues
- Common antipatterns

### 3. Local Runtime Test (Catches attribute errors and runtime issues)

```bash
# Test with minimal data
python test_script_locally.py your_script.py

# Must see: "✅ Basic validation complete!"
```

This catches:
- AttributeError (like tokenizer.save() vs save_vocabulary())
- Runtime errors with actual data
- Missing methods or functions
- Model building issues

### 4. Dry Run Test (5 minutes max)

```bash
# Set test environment variables
export DRY_RUN=1
export MAX_SAMPLES=100
export MAX_EPOCHS=1

# Run the actual script
python your_script.py

# Should complete within 5 minutes with no errors
```

### 5. Final Checks

Before pushing to git:

```bash
# Check for hardcoded paths
grep -n "/notebooks\|/storage\|/home\|/Users" your_script.py

# Check for proper error handling
grep -n "try:\|except" your_script.py

# Verify save methods
grep -n "\.save(" your_script.py
```

## Common Fixes for Frequent Issues

### Import Errors
```python
# ❌ Wrong
from train_progressive_minimal import create_model

# ✅ Correct
from models import create_model
```

### Tokenizer Save
```python
# ❌ Wrong
tokenizer.save(path)

# ✅ Correct
tokenizer.save_vocabulary(path)
```

### Path Handling
```python
# ❌ Wrong
base_path = '/notebooks/neural_networks_research'

# ✅ Correct
if os.path.exists('/notebooks/neural_networks_research'):
    base_path = '/notebooks/neural_networks_research'
elif os.path.exists('/workspace/neural_networks_research'):
    base_path = '/workspace/neural_networks_research'
else:
    base_path = Path(__file__).parent.parent.parent
```

### Error Handling
```python
# ❌ Wrong
def main():
    train_model()

# ✅ Correct
def main():
    try:
        train_model()
    except Exception as e:
        print(f"❌ ERROR: {e}")
        # Save what we can
        save_emergency_checkpoint()
        raise
```

## Setting Up Automated Validation

### Install Development Dependencies
```bash
pip install mypy flake8 black isort pre-commit
```

### Enable Pre-commit Hooks
```bash
pre-commit install
```

### Configure VS Code / Cursor
Add to `.vscode/settings.json`:
```json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black"
}
```

## Emergency Debugging on Paperspace

If a script fails on Paperspace:

1. **Check the exact error**:
   ```bash
   tail -n 50 error.log
   ```

2. **Test the specific failing component**:
   ```python
   # Create test_debug.py
   from failing_module import failing_function
   print(dir(failing_function))  # See available methods
   ```

3. **Quick fix and test**:
   ```bash
   # Edit the file
   nano your_script.py

   # Test just the fixed part
   python -c "from your_script import fixed_function; fixed_function()"
   ```

## The Golden Rule

**Time spent on validation saves GPU hours.**

A 5-minute local test can prevent:
- 4-6 hours of wasted GPU time
- Lost results from crashes
- Debugging on expensive hardware
- Frustration and delays

## Checklist Summary

- [ ] Run `validate_before_paperspace.py`
- [ ] Run `test_script_locally.py`
- [ ] Do a 5-minute dry run
- [ ] Check for hardcoded paths
- [ ] Verify error handling
- [ ] Test in correct conda environment

Only when ALL checks pass, push to Paperspace.
