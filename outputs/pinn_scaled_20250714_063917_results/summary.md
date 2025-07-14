
# Scaled PINN Training Results

## Model Configuration
- Parameters: 1,925,708
- Hidden dimension: 512
- Layers: 6
- Training time: 9.0 minutes

## Progressive Training Results

### Stage 1: Earth Only
- Earth MSE: 1122.2981
- Jupiter MSE: 1220.6532

### Stage 2: Earth + Mars + Moon  
- Earth MSE: 905.3689
- Moon MSE: 892.1095
- Jupiter MSE: 874.5259

### Stage 3: Full Curriculum
- Earth MSE: 908.5435
- Moon MSE: 889.1826
- Jupiter MSE: 880.8787

## Physics Understanding
- Gravity prediction error (Jupiter): 24.80 m/s²
- Trajectory smoothness maintained: 0.1776

## Comparison with Baselines
Best baseline (GraphExtrap) Jupiter MSE: 0.7660
PINN Jupiter MSE: 880.8787
Further training needed
