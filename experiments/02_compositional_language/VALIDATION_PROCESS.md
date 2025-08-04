# Comprehensive Validation Process for Compositional Language Experiment

## Overview

This document establishes a comprehensive validation process to catch errors before deployment to cloud environments (Paperspace, Colab, etc.). It was created after discovering that `ModificationGenerator.load_modifications()` was called but didn't exist.

## Pre-Deployment Validation Checklist

### 1. Static Analysis
- [ ] Run `flake8` for syntax and style issues
- [ ] Run `mypy` for type checking (if using type hints)
- [ ] Check all imports are valid

### 2. Method Existence Validation
- [ ] Verify all method calls on custom classes exist
- [ ] Check that method signatures match usage
- [ ] Validate attribute access on objects

### 3. Data Flow Validation
- [ ] Ensure data generation creates all required files
- [ ] Verify file paths are consistent across scripts
- [ ] Check pickle file compatibility

### 4. Environment Compatibility
- [ ] Test with target Keras backend (TensorFlow for Paperspace)
- [ ] Verify GPU memory requirements
- [ ] Check for platform-specific code

## Automated Validation Script

Run before any cloud deployment:

```bash
# Full validation
python validate_training_script.py paperspace_train_with_safeguards.py

# Quick test with minimal data
python test_script_locally.py paperspace_train_with_safeguards.py
```

## Common Issues and Solutions

### 1. Missing Methods
**Issue**: `AttributeError: 'ModificationGenerator' object has no attribute 'load_modifications'`

**Detection**:
```python
# In validation script
if 'generator.load_modifications()' in content:
    from modification_generator import ModificationGenerator
    if not hasattr(ModificationGenerator, 'load_modifications'):
        raise ValidationError("Method doesn't exist!")
```

**Solution**: Either add the method or use direct pickle loading.

### 2. Tokenizer Save Methods
**Issue**: `AttributeError: 'SCANTokenizer' object has no attribute 'save'`

**Detection**:
```python
if 'tokenizer.save(' in content:
    # Should be tokenizer.save_vocabulary()
    raise ValidationError("Wrong method name!")
```

### 3. File Path Issues
**Issue**: Files not found due to path differences between environments

**Detection**:
```python
# Check for hardcoded paths
hardcoded_paths = re.findall(r'["\']\/(?:home|Users|notebooks|workspace)', content)
if hardcoded_paths:
    raise ValidationError(f"Hardcoded paths found: {hardcoded_paths}")
```

## Validation Process Flow

1. **Local Development**
   ```bash
   # Write/modify code
   python your_script.py --test  # Test mode with 100 samples
   ```

2. **Pre-Commit Validation**
   ```bash
   # Run all checks
   ./scripts/pre_merge_tests.sh
   ```

3. **Script-Specific Validation**
   ```bash
   # Test the exact script you'll run on Paperspace
   python test_script_locally.py paperspace_train_with_safeguards.py
   ```

4. **Cloud Deployment**
   - Only deploy after all validations pass
   - Keep validation logs for debugging

## Adding New Validations

When you encounter a new type of error:

1. Add detection logic to `test_script_locally.py`
2. Document the issue and solution here
3. Create a test case to prevent regression

## Method Discovery Helper

To discover available methods on a class:

```python
def discover_methods(class_name, module_name):
    """Discover all public methods of a class"""
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)

    methods = []
    for name in dir(cls):
        if not name.startswith('_'):
            attr = getattr(cls, name)
            if callable(attr):
                methods.append(name)

    return methods

# Usage
methods = discover_methods('ModificationGenerator', 'modification_generator')
print(f"Available methods: {methods}")
```

## Emergency Recovery

If a script fails on Paperspace:

1. Check `/storage` for any saved outputs
2. Review logs in Paperspace console
3. Run validation locally to reproduce
4. Fix and re-validate before redeploying

## Continuous Improvement

- After each cloud run, update this document with new learnings
- Add new validation checks for discovered issues
- Share validation scripts across experiments

---

Remember: **Every minute of cloud GPU time is valuable**. Thorough validation saves both time and money.
