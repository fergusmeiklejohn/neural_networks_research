# Compositional Language Experiment Action Plan
Date: 2025-07-22

## Current Status
✅ **Implementation Complete**: All components fully implemented and tested
✅ **Data Generated**: SCAN dataset processed with 39,708 training samples
✅ **Quick Test Passed**: Model creation, data loading, and training pipeline verified
✅ **Safeguards Added**: New training script with comprehensive result preservation

## Immediate Next Steps

### 1. Run Full Training on Paperspace (Priority: HIGH)
**Command to run on Paperspace:**
```bash
cd /notebooks/neural_networks_research/experiments/02_compositional_language
python paperspace_train_with_safeguards.py
```

**Key Features of New Script:**
- Automatic persistent storage detection and use
- Checkpoint saving after every epoch to /storage
- Multiple backup mechanisms
- Comprehensive evaluation on all test sets
- Emergency save on failure

**Expected Duration:** 4-6 hours on A4000 GPU

### 2. Monitor Training Progress
- Watch GPU usage: `watch -n 1 nvidia-smi`
- Check /storage for checkpoint saves
- Monitor loss/accuracy progression

### 3. Post-Training Verification
After training completes:
1. Verify results in `/storage/compositional_language_[timestamp]/`
2. Check evaluation_results.json for performance metrics
3. Download backup: `zip -r results.zip /storage/compositional_language_*/`

## Key Improvements from Previous Attempt

1. **Persistent Storage**: All checkpoints saved to /storage immediately
2. **Reduced Epochs**: 5 per stage for faster iteration (vs 50 before)
3. **Multiple Backups**: Local + persistent storage + final zip
4. **Comprehensive Evaluation**: Automatic testing on all splits
5. **Emergency Recovery**: Saves progress even on crash

## Success Metrics

### Expected Performance:
- **Interpolation** (standard SCAN): >95% accuracy
- **Extrapolation** (new primitives): >70% accuracy
- **Rule Modifications**: >60% consistency
- **Novel Generation**: >50% validity

### Key Comparisons:
- Previous LSTM run: 96.2% final accuracy (lost to shutdown)
- Target: Match or exceed with result preservation

## Future Enhancements (After Successful Run)

### 1. Test-Time Adaptation (TTA)
- Apply insights from physics experiment
- Add adaptation during inference for better extrapolation
- Implementation sketch already in physics experiment

### 2. True OOD Scenarios
- Design "time-varying" linguistic rules
- Example: Grammar rules that change mid-sequence
- Create genuinely out-of-distribution test cases

### 3. Baseline Comparisons
- Train all 4 baseline models on same data
- Use unified evaluation framework
- Generate comparison table

## Risk Mitigation

1. **Auto-shutdown**: Script saves every epoch + final emergency save
2. **Memory issues**: Using proven minimal LSTM architecture
3. **Data loss**: Multiple backup locations + immediate saves
4. **Training failure**: Fallback to simpler architectures

## Commands Reference

### Local Testing:
```bash
# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh && conda activate dist-invention

# Quick test
python quick_test.py

# Generate data if needed
python scan_data_loader.py
```

### Paperspace Execution:
```bash
# Main training with safeguards
python paperspace_train_with_safeguards.py

# Alternative: Original script
python paperspace_generate_and_train.py

# Emergency result save
python save_all_results.py
```

### Result Retrieval:
```bash
# Check storage
ls -la /storage/

# Create downloadable archive
zip -r compositional_results_$(date +%Y%m%d_%H%M%S).zip /storage/compositional_language_*/

# Verify contents
unzip -l compositional_results_*.zip
```

## Next Session Checklist

When returning to this work:
1. Check if Paperspace training completed
2. Download and analyze results
3. Compare with baseline models
4. Plan TTA implementation if results are good
5. Document findings in research diary

The experiment is fully ready for execution. The new safeguarded training script addresses all issues from the previous attempt and should successfully preserve results even with Paperspace's auto-shutdown behavior.
