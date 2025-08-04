# Paperspace Training Best Practices Guide

Based on hard-won experience from the Compositional Language experiment, this guide ensures you never lose training results again.

## Pre-Training Checklist

### 1. Environment Setup
```bash
# Always start with GPU memory growth
export TF_CPP_MIN_LOG_LEVEL=2
```

### 2. Directory Structure
```
/notebooks/neural_networks_research/experiment_XX/
├── data/                    # Will be generated
├── outputs/                 # Training outputs
├── /storage/experiment_XX/  # Persistent storage backup
└── wandb/                   # Experiment tracking
```

### 3. Verify Storage Access
```python
import os
assert os.path.exists('/storage'), "No persistent storage available!"
print(f"Storage available: {os.listdir('/storage')}")
```

## During Training Best Practices

### 1. Save Checkpoints Frequently
```python
# After each stage
checkpoint_path = f'/storage/experiment_name/stage_{stage}_model.h5'
model.save_weights(checkpoint_path)
print(f"Checkpoint saved to persistent storage: {checkpoint_path}")
```

### 2. Log Everything
- Use wandb for metrics
- Also save local JSON logs
- Print key metrics to console (in case of wandb failure)

### 3. Monitor GPU Usage
```bash
# In separate terminal
watch -n 1 nvidia-smi
```
If GPU usage < 20%, increase batch size!

### 4. Create Stage Summaries
```python
# After each training stage
stage_summary = {
    'stage': stage,
    'final_loss': float(train_loss),
    'val_accuracy': float(val_acc),
    'timestamp': datetime.now().isoformat()
}
with open(f'/storage/stage_{stage}_summary.json', 'w') as f:
    json.dump(stage_summary, f)
```

## Post-Training Procedures

### 1. Immediate Actions (Before Shutdown!)

```bash
# 1. Run comprehensive save script
python save_all_results.py

# 2. Create backup zip
zip -r results_backup_$(date +%Y%m%d_%H%M%S).zip outputs/ data/processed/

# 3. Copy to storage
cp results_backup*.zip /storage/

# 4. Verify storage
ls -la /storage/
```

### 2. Results Validation
```python
# Verify what was saved
import glob
print("Model files:", glob.glob('/storage/**/*.h5', recursive=True))
print("Results:", glob.glob('/storage/**/*results*.json', recursive=True))
```

## Common Pitfalls and Solutions

### Problem: "Model weights not created"
**Cause**: Complex nested models, Sequential layers in dicts
**Solution**: Use simple architectures, explicit layer definitions

### Problem: Mixed precision errors
**Cause**: Not all ops support float16
**Solution**: Disable mixed precision for proof-of-concept

### Problem: Optimizer doesn't recognize variables
**Cause**: Keras 3 stricter variable tracking
**Solution**: Use legacy optimizers or explicit optimizer.build()

### Problem: Results not saved
**Cause**: Looking in wrong directory, shutdown too quick
**Solution**: Save to multiple locations, use storage during training

## Template Training Script Structure

```python
def train_with_safety():
    # 1. Setup storage paths
    storage_dir = Path(f'/storage/{experiment_name}_{timestamp}')
    storage_dir.mkdir(exist_ok=True)

    # 2. Training loop with saves
    for stage in range(4):
        # ... training code ...

        # Save after EVERY stage
        save_checkpoint(model, storage_dir / f'stage_{stage}.h5')
        save_metrics(metrics, storage_dir / f'stage_{stage}_metrics.json')

    # 3. Final comprehensive save
    save_everything_to_storage(storage_dir)
    create_downloadable_zip()
```

## Paperspace-Specific Tips

1. **Notebooks vs Jobs**: Notebooks give you 6 hours, jobs can run longer
2. **Instance Types**: A4000 (16GB) good for most experiments
3. **Storage Persistence**: Only `/storage` persists between sessions
4. **Auto-shutdown**: Save early, save often!

## Emergency Recovery

If training completes but save fails:
```bash
# Try to recover from wandb
wandb sync wandb/latest-run/

# Check for auto-saved checkpoints
find . -name "*.h5" -o -name "*.keras" -mmin -360

# Look for TensorFlow autosaves
ls -la /tmp/tensorflow_saves/
```

## Recommended Workflow

1. **Start Small**: Test with 100 samples, 1 epoch
2. **Verify Saves**: Ensure checkpointing works
3. **Scale Up**: Increase data/epochs only after verification
4. **Monitor Progress**: Check intermediate results
5. **Save Redundantly**: Local + storage + download

Remember: It's better to have too many backups than to lose 4 hours of GPU training!
