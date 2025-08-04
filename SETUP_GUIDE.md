# Quick Setup Guide for Physics Worlds Experiment

## Environment Activation

You need to activate the `dist-invention` conda environment before running any experiments.

### Step 1: Activate Environment
```bash
conda activate dist-invention
```

### Step 2: Verify Environment
```bash
python check_environment.py
```
You should see "✅ Environment check PASSED" and "dist-invention" as the current environment.

### Step 3: Run Tests
```bash
cd experiments/01_physics_worlds
python simple_test.py
```

## If Activation Doesn't Work

### Option 1: Initialize Conda
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dist-invention
```

### Option 2: Use Direct Path
```bash
/Users/fergusmeiklejohn/miniconda3/envs/dist-invention/bin/python check_environment.py
```

### Option 3: Use Activation Script
```bash
./activate_and_test.sh
```

## Expected Output

When everything is working correctly, you should see:
```
✅ Environment check PASSED
All required packages are available!
You're in the correct conda environment!
```

## Next Steps After Environment is Working

1. **Test Components**: `cd experiments/01_physics_worlds && python simple_test.py`
2. **Generate Data**: `python data_generator.py`
3. **Train Model**: `python train_physics.py --epochs 10 --batch_size 4`
4. **Evaluate**: `python evaluate_physics.py --model_path outputs/checkpoints/model.keras`

## Troubleshooting

- **"conda: command not found"**: Make sure conda is installed and in your PATH
- **"Environment not found"**: Run `conda info --envs` to see available environments
- **"Package missing"**: The environment may not be fully set up, re-run the installation steps

## Package List

The environment should have these packages:
- keras, torch, jax (Deep learning frameworks)
- numpy, scipy, matplotlib (Scientific computing)
- pymunk, pygame (Physics simulation)
- transformers, wandb (ML utilities)
- tqdm (Progress bars)
