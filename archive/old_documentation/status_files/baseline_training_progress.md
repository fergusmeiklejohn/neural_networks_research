# Baseline Training Progress Report

## Date: 2025-01-15

### GFlowNet Training Status
- **Model Parameters**: 152,464
- **Training Data**: 5,678 2-ball trajectories
- **Validation Data**: 823 trajectories
- **Jupiter Test Data**: 57 trajectories (gravity range: -40 to -50 m/s²)

### Training Progress (First 40 epochs)
- Initial loss: ~33,000
- Best validation loss so far: ~389.75 (Epoch 39)
- Model is showing steady improvement
- Using exploration noise during training as expected for GFlowNet

### Key Observations
1. **Data Loading Success**: Successfully loaded real physics data with proper gravity conversions
2. **Jupiter Samples Found**: 57 samples with Jupiter-like gravity in test set
3. **Model Architecture**: Simplified GFlowNet with encoder + flow network
4. **Training Stability**: No NaN losses or training failures

### Comparison Points
When training completes, we'll compare:
- **GFlowNet**: (training in progress)
- **GraphExtrap**: 0.766 MSE (best baseline)
- **Failed PINN**: 880.879 MSE
- **Minimal PINN**: (training separately)

### Next Steps While Waiting
1. Check on Minimal PINN training progress
2. Prepare MAML baseline training
3. Begin implementing True OOD benchmark

### Environment Note
Both training scripts may face conda environment issues. If interrupted, user should run:
```bash
# For GFlowNet
conda activate dist-invention
python train_baseline_with_real_data.py --model gflownet

# For Minimal PINN
python train_minimal_pinn.py
```
