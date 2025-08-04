# Instructions for Simple Baseline Training

## Quick Start

From the `experiments/02_compositional_language/` directory, run:

```bash
/Users/fergusmeiklejohn/miniconda3/envs/dist-invention/bin/python simple_baseline_v2.py \
    --epochs 50 \
    --batch_size 32 \
    --d_model 128 \
    --learning_rate 0.001
```

## Full Training Script (Recommended)

For a more comprehensive training run with all data:

```bash
# First, create a training script that uses all data
cat > train_simple_baseline_full.py << 'EOF'
#!/usr/bin/env python3
"""
Full training run for simple baseline model.
This uses all available training data (not just 10K subset).
"""

import sys
import os

# Modify simple_baseline_v2.py to use full dataset
with open('simple_baseline_v2.py', 'r') as f:
    content = f.read()

# Change the subset limit
content = content.replace(
    'for sample in train_data[:10000]:  # Use subset for testing',
    'for sample in train_data:  # Use full dataset'
)
content = content.replace(
    'for item in mod_pairs[:2000]:  # Add some modified examples',
    'for item in mod_pairs[:len(train_data)//2]:  # Add 50% modified examples'
)

# Write modified version
with open('simple_baseline_v2_full.py', 'w') as f:
    f.write(content)

# Run the full training
os.system(f'{sys.executable} simple_baseline_v2_full.py --epochs 50 --batch_size 64')
EOF

# Make it executable and run
chmod +x train_simple_baseline_full.py
/Users/fergusmeiklejohn/miniconda3/envs/dist-invention/bin/python train_simple_baseline_full.py
```

## Parameters to Try

### Option 1: Conservative (Recommended for first run)
```bash
--epochs 30
--batch_size 32
--d_model 128
--learning_rate 0.001
```
Expected time on Mac: ~1-2 hours

### Option 2: Full Training
```bash
--epochs 50
--batch_size 64
--d_model 256
--learning_rate 0.001
```
Expected time on Mac: ~3-5 hours

### Option 3: Quick Test (to estimate timing)
```bash
--epochs 5
--batch_size 32
--d_model 128
--learning_rate 0.001
```
Expected time on Mac: ~10-15 minutes

## Monitoring Progress

The script will:
1. Print epoch-by-epoch training metrics
2. Save model to `outputs/simple_baseline_v2_YYYYMMDD_HHMMSS/`
3. Save training history to `training_history.json`

## After Training Completes

1. **Run Evaluation**:
```bash
/Users/fergusmeiklejohn/miniconda3/envs/dist-invention/bin/python evaluate_simple_baseline.py \
    --model_dir outputs/simple_baseline_v2_YYYYMMDD_HHMMSS \
    --batch_size 32
```

2. **Check Results**:
- Look for `evaluation_results.json` in the model directory
- Key metrics to report:
  - `val_base` accuracy (should be >50% if training worked)
  - Average modification accuracy
  - Performance drop on modifications

## Expected Results

After proper training, we expect:
- **Base accuracy**: 70-85% (on standard SCAN examples)
- **Modification accuracy**: 5-20% (much lower, revealing the true challenge)
- **Performance drop**: 50-70% (showing modifications are truly hard)

This will establish a honest baseline showing:
1. The model CAN learn basic SCAN
2. But CANNOT handle modifications well
3. No evaluation illusion - just honest performance

## Moving to Paperspace

If training takes >2 hours on your Mac, consider moving to Paperspace:

1. Create `paperspace_train_simple_baseline.py` that:
   - Generates all data on the cloud machine
   - Runs the full training pipeline
   - Saves results to `/storage/`

2. Use the template from `paperspace_generate_and_train.py` as reference

## What to Report Back

Please share:
1. Training time for 5 epochs (to estimate full run)
2. Final training/validation accuracy
3. Evaluation results on all validation sets
4. Any errors or issues encountered

This will help us understand if the simple baseline can achieve reasonable performance with proper training!
