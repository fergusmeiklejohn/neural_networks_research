# TTA Test Results:
## test_regression_tta.py - Tests the new regression-specific TTA
Testing Regression-Specific TTA
============================================================
Model created successfully

Training model...

Baseline predictions:
  Shape: (5, 10, 8)
  Mean: 0.0083
  Std: 0.1967

Testing TTA configurations:
------------------------------------------------------------

Config: lr=0.0001, steps=5
  TENT change: 0.217619
  PhysicsTENT change: 0.217619

Config: lr=0.001, steps=10
  TENT change: 0.217619
  PhysicsTENT change: 0.217619

Config: lr=0.01, steps=10
  TENT change: 0.217619
  PhysicsTENT change: 0.217619

Config: lr=0.01, steps=20
  TENT change: 0.217619
  PhysicsTENT change: 0.217619


Batch adaptation test:
------------------------------------------------------------
Average change per sample: 0.374636
Max change: 0.522977

============================================================
CONCLUSION
============================================================
✓ TTA is working! Predictions are being adapted.
  Higher learning rates and more steps = more adaptation


## debug_tta_convergence.py - Tracks adaptation dynamics
TTA Convergence Debugging
============================================================
Loaded time-varying gravity data: (100, 50, 8)

Creating test model...

Training on constant gravity...

Initial Model Weight Analysis:
  Layer 1 (dense) - kernel:
    Shape: (8, 128), Mean: -0.002184, Std: 0.120763
  Layer 1 (dense) - bias:
    Shape: (128,), Mean: 0.001193, Std: 0.007255
  Layer 2 (batch_normalization) - gamma:
    Shape: (128,), Mean: 0.999226, Std: 0.010921
    Min: 0.981296, Max: 1.018766
  Layer 2 (batch_normalization) - beta:
    Shape: (128,), Mean: -0.001167, Std: 0.009367
    Min: -0.017630, Max: 0.017469
  Layer 2 (batch_normalization) - moving_mean:
    Shape: (128,), Mean: 1.258707, Std: 1.595452
    Min: 0.000000, Max: 6.284874
  Layer 2 (batch_normalization) - moving_variance:
    Shape: (128,), Mean: 6.485753, Std: 6.788924
    Min: 0.817907, Max: 32.311920
  Layer 3 (dense_1) - kernel:
    Shape: (128, 64), Mean: 0.000012, Std: 0.101647
  Layer 3 (dense_1) - bias:
    Shape: (64,), Mean: 0.001856, Std: 0.009048
  Layer 4 (batch_normalization_1) - gamma:
    Shape: (64,), Mean: 1.009377, Std: 0.011605
    Min: 0.982955, Max: 1.020629
  Layer 4 (batch_normalization_1) - beta:
    Shape: (64,), Mean: -0.003750, Std: 0.017410
    Min: -0.020254, Max: 0.020168
  Layer 4 (batch_normalization_1) - moving_mean:
    Shape: (64,), Mean: 0.077153, Std: 0.018769
    Min: 0.046358, Max: 0.160760
  Layer 4 (batch_normalization_1) - moving_variance:
    Shape: (64,), Mean: 0.890620, Std: 0.033735
    Min: 0.840258, Max: 1.017379
  Layer 5 (dense_2) - kernel:
    Shape: (64, 80), Mean: -0.000715, Std: 0.118021
  Layer 5 (dense_2) - bias:
    Shape: (80,), Mean: 0.010306, Std: 0.014824

============================================================
Testing TENT
============================================================


Tracking TENT adaptation over 10 steps...

Adaptation progress:
  Step 1: Loss = 0.000833
    gamma shape changed: (64,) -> (128,)
    beta shape changed: (64,) -> (128,)
    moving_mean shape changed: (64,) -> (128,)
    moving_variance shape changed: (64,) -> (128,)
    moving_mean changed by 0.000620
    moving_variance changed by 0.008906
  Step 2: Loss = 0.000833
    gamma shape changed: (64,) -> (128,)
    beta shape changed: (64,) -> (128,)
    moving_mean shape changed: (64,) -> (128,)
    moving_variance shape changed: (64,) -> (128,)
    moving_mean changed by 0.001234
    moving_variance changed by 0.017723
  Step 3: Loss = 0.000833
    gamma shape changed: (64,) -> (128,)
    beta shape changed: (64,) -> (128,)
    moving_mean shape changed: (64,) -> (128,)
    moving_variance shape changed: (64,) -> (128,)
    moving_mean changed by 0.001842
    moving_variance changed by 0.026452
  Step 4: Loss = 0.000833
    gamma shape changed: (64,) -> (128,)
    beta shape changed: (64,) -> (128,)
    moving_mean shape changed: (64,) -> (128,)
    moving_variance shape changed: (64,) -> (128,)
    moving_mean changed by 0.002444
    moving_variance changed by 0.035094
  Step 5: Loss = 0.000833
    gamma shape changed: (64,) -> (128,)
    beta shape changed: (64,) -> (128,)
    moving_mean shape changed: (64,) -> (128,)
    moving_variance shape changed: (64,) -> (128,)
    moving_mean changed by 0.003039
    moving_variance changed by 0.043649
  Step 6: Loss = 0.000833
    gamma shape changed: (64,) -> (128,)
    beta shape changed: (64,) -> (128,)
    moving_mean shape changed: (64,) -> (128,)
    moving_variance shape changed: (64,) -> (128,)
    moving_mean changed by 0.003629
    moving_variance changed by 0.052119
  Step 7: Loss = 0.000833
    gamma shape changed: (64,) -> (128,)
    beta shape changed: (64,) -> (128,)
    moving_mean shape changed: (64,) -> (128,)
    moving_variance shape changed: (64,) -> (128,)
    moving_mean changed by 0.004213
    moving_variance changed by 0.060504
  Step 8: Loss = 0.000833
    gamma shape changed: (64,) -> (128,)
    beta shape changed: (64,) -> (128,)
    moving_mean shape changed: (64,) -> (128,)
    moving_variance shape changed: (64,) -> (128,)
    moving_mean changed by 0.004791
    moving_variance changed by 0.068805
  Step 9: Loss = 0.000833
    gamma shape changed: (64,) -> (128,)
    beta shape changed: (64,) -> (128,)
    moving_mean shape changed: (64,) -> (128,)
    moving_variance shape changed: (64,) -> (128,)
    moving_mean changed by 0.005363
    moving_variance changed by 0.077023
  Step 10: Loss = 0.000833
    gamma shape changed: (64,) -> (128,)
    beta shape changed: (64,) -> (128,)
    moving_mean shape changed: (64,) -> (128,)
    moving_variance shape changed: (64,) -> (128,)
    moving_mean changed by 0.005930
    moving_variance changed by 0.085159

Prediction changes:
  Initial prediction mean: 0.277871
  Final prediction mean: 0.277756
  Total change: 0.001915

After TENT Weight Analysis:
  Layer 1 (dense) - kernel:
    Shape: (8, 128), Mean: -0.002184, Std: 0.120763
  Layer 1 (dense) - bias:
    Shape: (128,), Mean: 0.001193, Std: 0.007255
  Layer 2 (batch_normalization) - gamma:
    Shape: (128,), Mean: 0.999226, Std: 0.010921
    Min: 0.981296, Max: 1.018766
  Layer 2 (batch_normalization) - beta:
    Shape: (128,), Mean: -0.001167, Std: 0.009367
    Min: -0.017630, Max: 0.017469
  Layer 2 (batch_normalization) - moving_mean:
    Shape: (128,), Mean: 2.027480, Std: 2.600987
    Min: 0.000000, Max: 10.438675
  Layer 2 (batch_normalization) - moving_variance:
    Shape: (128,), Mean: 5.865600, Std: 6.139782
    Min: 0.739700, Max: 29.222322
  Layer 3 (dense_1) - kernel:
    Shape: (128, 64), Mean: 0.000012, Std: 0.101647
  Layer 3 (dense_1) - bias:
    Shape: (64,), Mean: 0.001856, Std: 0.009048
  Layer 4 (batch_normalization_1) - gamma:
    Shape: (64,), Mean: 1.009377, Std: 0.011605
    Min: 0.982955, Max: 1.020629
  Layer 4 (batch_normalization_1) - beta:
    Shape: (64,), Mean: -0.003750, Std: 0.017410
    Min: -0.020254, Max: 0.020168
  Layer 4 (batch_normalization_1) - moving_mean:
    Shape: (64,), Mean: 0.071223, Std: 0.017133
    Min: 0.041925, Max: 0.148609
  Layer 4 (batch_normalization_1) - moving_variance:
    Shape: (64,), Mean: 0.805460, Std: 0.030510
    Min: 0.759914, Max: 0.920099
  Layer 5 (dense_2) - kernel:
    Shape: (64, 80), Mean: -0.000715, Std: 0.118021
  Layer 5 (dense_2) - bias:
    Shape: (80,), Mean: 0.010306, Std: 0.014824

============================================================
Testing PHYSICS_TENT
============================================================


Tracking PHYSICS_TENT adaptation over 10 steps...

Adaptation progress:
  Step 1: Loss = 0.584229
    gamma shape changed: (64,) -> (128,)
    beta shape changed: (64,) -> (128,)
    moving_mean shape changed: (64,) -> (128,)
    moving_variance shape changed: (64,) -> (128,)
    moving_mean changed by 0.000620
    moving_variance changed by 0.008906
  Step 2: Loss = 0.584229
    gamma shape changed: (64,) -> (128,)
    beta shape changed: (64,) -> (128,)
    moving_mean shape changed: (64,) -> (128,)
    moving_variance shape changed: (64,) -> (128,)
    moving_mean changed by 0.001234
    moving_variance changed by 0.017723
  Step 3: Loss = 0.584229
    gamma shape changed: (64,) -> (128,)
    beta shape changed: (64,) -> (128,)
    moving_mean shape changed: (64,) -> (128,)
    moving_variance shape changed: (64,) -> (128,)
    moving_mean changed by 0.001842
    moving_variance changed by 0.026452
  Step 4: Loss = 0.584229
    gamma shape changed: (64,) -> (128,)
    beta shape changed: (64,) -> (128,)
    moving_mean shape changed: (64,) -> (128,)
    moving_variance shape changed: (64,) -> (128,)
    moving_mean changed by 0.002444
    moving_variance changed by 0.035094
  Step 5: Loss = 0.584229
    gamma shape changed: (64,) -> (128,)
    beta shape changed: (64,) -> (128,)
    moving_mean shape changed: (64,) -> (128,)
    moving_variance shape changed: (64,) -> (128,)
    moving_mean changed by 0.003039
    moving_variance changed by 0.043649
  Step 6: Loss = 0.584229
    gamma shape changed: (64,) -> (128,)
    beta shape changed: (64,) -> (128,)
    moving_mean shape changed: (64,) -> (128,)
    moving_variance shape changed: (64,) -> (128,)
    moving_mean changed by 0.003629
    moving_variance changed by 0.052119
  Step 7: Loss = 0.584229
    gamma shape changed: (64,) -> (128,)
    beta shape changed: (64,) -> (128,)
    moving_mean shape changed: (64,) -> (128,)
    moving_variance shape changed: (64,) -> (128,)
    moving_mean changed by 0.004213
    moving_variance changed by 0.060504
  Step 8: Loss = 0.584229
    gamma shape changed: (64,) -> (128,)
    beta shape changed: (64,) -> (128,)
    moving_mean shape changed: (64,) -> (128,)
    moving_variance shape changed: (64,) -> (128,)
    moving_mean changed by 0.004791
    moving_variance changed by 0.068805
  Step 9: Loss = 0.584229
    gamma shape changed: (64,) -> (128,)
    beta shape changed: (64,) -> (128,)
    moving_mean shape changed: (64,) -> (128,)
    moving_variance shape changed: (64,) -> (128,)
    moving_mean changed by 0.005363
    moving_variance changed by 0.077023
  Step 10: Loss = 0.584229
    gamma shape changed: (64,) -> (128,)
    beta shape changed: (64,) -> (128,)
    moving_mean shape changed: (64,) -> (128,)
    moving_variance shape changed: (64,) -> (128,)
    moving_mean changed by 0.005930
    moving_variance changed by 0.085159

Prediction changes:
  Initial prediction mean: 0.277871
  Final prediction mean: 0.277756
  Total change: 0.001915

After PHYSICS_TENT Weight Analysis:
  Layer 1 (dense) - kernel:
    Shape: (8, 128), Mean: -0.002184, Std: 0.120763
  Layer 1 (dense) - bias:
    Shape: (128,), Mean: 0.001193, Std: 0.007255
  Layer 2 (batch_normalization) - gamma:
    Shape: (128,), Mean: 0.999226, Std: 0.010921
    Min: 0.981296, Max: 1.018766
  Layer 2 (batch_normalization) - beta:
    Shape: (128,), Mean: -0.001167, Std: 0.009367
    Min: -0.017630, Max: 0.017469
  Layer 2 (batch_normalization) - moving_mean:
    Shape: (128,), Mean: 2.027480, Std: 2.600987
    Min: 0.000000, Max: 10.438675
  Layer 2 (batch_normalization) - moving_variance:
    Shape: (128,), Mean: 5.865600, Std: 6.139782
    Min: 0.739700, Max: 29.222322
  Layer 3 (dense_1) - kernel:
    Shape: (128, 64), Mean: 0.000012, Std: 0.101647
  Layer 3 (dense_1) - bias:
    Shape: (64,), Mean: 0.001856, Std: 0.009048
  Layer 4 (batch_normalization_1) - gamma:
    Shape: (64,), Mean: 1.009377, Std: 0.011605
    Min: 0.982955, Max: 1.020629
  Layer 4 (batch_normalization_1) - beta:
    Shape: (64,), Mean: -0.003750, Std: 0.017410
    Min: -0.020254, Max: 0.020168
  Layer 4 (batch_normalization_1) - moving_mean:
    Shape: (64,), Mean: 0.071223, Std: 0.017133
    Min: 0.041925, Max: 0.148609
  Layer 4 (batch_normalization_1) - moving_variance:
    Shape: (64,), Mean: 0.805460, Std: 0.030510
    Min: 0.759914, Max: 0.920099
  Layer 5 (dense_2) - kernel:
    Shape: (64, 80), Mean: -0.000715, Std: 0.118021
  Layer 5 (dense_2) - bias:
    Shape: (80,), Mean: 0.010306, Std: 0.014824

============================================================
Testing TTT
============================================================


Tracking TTT adaptation over 10 steps...

Adaptation progress:
  Step 1: Loss = nan
    gamma shape changed: (64,) -> (128,)
    beta shape changed: (64,) -> (128,)
    moving_mean shape changed: (64,) -> (128,)
    moving_variance shape changed: (64,) -> (128,)
    moving_mean changed by 0.001234
    moving_variance changed by 0.017723
  Step 2: Loss = nan
    gamma shape changed: (64,) -> (128,)
    beta shape changed: (64,) -> (128,)
    moving_mean shape changed: (64,) -> (128,)
    moving_variance shape changed: (64,) -> (128,)
    moving_mean changed by 0.002444
    moving_variance changed by 0.035094
  Step 3: Loss = nan
    gamma shape changed: (64,) -> (128,)
    beta shape changed: (64,) -> (128,)
    moving_mean shape changed: (64,) -> (128,)
    moving_variance shape changed: (64,) -> (128,)
    moving_mean changed by 0.003629
    moving_variance changed by 0.052119
  Step 4: Loss = nan
    gamma shape changed: (64,) -> (128,)
    beta shape changed: (64,) -> (128,)
    moving_mean shape changed: (64,) -> (128,)
    moving_variance shape changed: (64,) -> (128,)
    moving_mean changed by 0.004791
    moving_variance changed by 0.068805
  Step 5: Loss = nan
    gamma shape changed: (64,) -> (128,)
    beta shape changed: (64,) -> (128,)
    moving_mean shape changed: (64,) -> (128,)
    moving_variance shape changed: (64,) -> (128,)
    moving_mean changed by 0.005930
    moving_variance changed by 0.085159
  Step 6: Loss = nan
    gamma shape changed: (64,) -> (128,)
    beta shape changed: (64,) -> (128,)
    moving_mean shape changed: (64,) -> (128,)
    moving_variance shape changed: (64,) -> (128,)
    moving_mean changed by 0.007046
    moving_variance changed by 0.101188
  Step 7: Loss = nan
    gamma shape changed: (64,) -> (128,)
    beta shape changed: (64,) -> (128,)
    moving_mean shape changed: (64,) -> (128,)
    moving_variance shape changed: (64,) -> (128,)
    moving_mean changed by 0.008140
    moving_variance changed by 0.116897
  Step 8: Loss = nan
    gamma shape changed: (64,) -> (128,)
    beta shape changed: (64,) -> (128,)
    moving_mean shape changed: (64,) -> (128,)
    moving_variance shape changed: (64,) -> (128,)
    moving_mean changed by 0.009212
    moving_variance changed by 0.132294
  Step 9: Loss = nan
    gamma shape changed: (64,) -> (128,)
    beta shape changed: (64,) -> (128,)
    moving_mean shape changed: (64,) -> (128,)
    moving_variance shape changed: (64,) -> (128,)
    moving_mean changed by 0.010263
    moving_variance changed by 0.147385
  Step 10: Loss = nan
    gamma shape changed: (64,) -> (128,)
    beta shape changed: (64,) -> (128,)
    moving_mean shape changed: (64,) -> (128,)
    moving_variance shape changed: (64,) -> (128,)
    moving_mean changed by 0.011293
    moving_variance changed by 0.162175

Prediction changes:
  Initial prediction mean: 0.277871
  Final prediction mean: 0.277644
  Total change: 0.003696

After TTT Weight Analysis:
  Layer 1 (dense) - kernel:
    Shape: (8, 128), Mean: -0.002184, Std: 0.120763
  Layer 1 (dense) - bias:
    Shape: (128,), Mean: 0.001193, Std: 0.007255
  Layer 2 (batch_normalization) - gamma:
    Shape: (128,), Mean: 0.999226, Std: 0.010921
    Min: 0.981296, Max: 1.018766
  Layer 2 (batch_normalization) - beta:
    Shape: (128,), Mean: -0.001167, Std: 0.009367
    Min: -0.017630, Max: 0.017469
  Layer 2 (batch_normalization) - moving_mean:
    Shape: (128,), Mean: 2.722745, Std: 3.559715
    Min: 0.000000, Max: 14.630692
  Layer 2 (batch_normalization) - moving_variance:
    Shape: (128,), Mean: 5.304744, Std: 5.552709
    Min: 0.668972, Max: 26.428146
  Layer 3 (dense_1) - kernel:
    Shape: (128, 64), Mean: 0.000012, Std: 0.101647
  Layer 3 (dense_1) - bias:
    Shape: (64,), Mean: 0.001856, Std: 0.009048
  Layer 4 (batch_normalization_1) - gamma:
    Shape: (64,), Mean: 1.009377, Std: 0.011605
    Min: 0.982955, Max: 1.020629
  Layer 4 (batch_normalization_1) - beta:
    Shape: (64,), Mean: -0.003750, Std: 0.017410
    Min: -0.020254, Max: 0.020168
  Layer 4 (batch_normalization_1) - moving_mean:
    Shape: (64,), Mean: 0.065860, Std: 0.015804
    Min: 0.037916, Max: 0.137621
  Layer 4 (batch_normalization_1) - moving_variance:
    Shape: (64,), Mean: 0.728444, Std: 0.027592
    Min: 0.687253, Max: 0.832121
  Layer 5 (dense_2) - kernel:
    Shape: (64, 80), Mean: -0.000715, Std: 0.118021
  Layer 5 (dense_2) - bias:
    Shape: (80,), Mean: 0.010306, Std: 0.014824

============================================================
COMPARISON OF TTA METHODS
============================================================

TENT:
  Predictions changed: True
  Loss trajectory: [0.0008325268281623721, 0.0008325268281623721, 0.0008325268281623721, 0.0008325268281623721, 0.0008325268281623721]...
  Initial vs Final prediction diff: 0.001915

PHYSICS_TENT:
  Predictions changed: True
  Loss trajectory: [0.5842287540435791, 0.5842287540435791, 0.5842287540435791, 0.5842287540435791, 0.5842287540435791]...
  Initial vs Final prediction diff: 0.001915

TTT:
  Predictions changed: True
  Loss trajectory: [nan, nan, nan, nan, nan]...
  Initial vs Final prediction diff: 0.003696

============================================================
HYPOTHESIS TESTING
============================================================

- Are BatchNorm statistics being updated?

- Is the learning rate too small?
  Current LR: 1e-4
  Try: 1e-3, 1e-2 for stronger adaptation

- Is single timestep input limiting?
  Current: 1 timestep → predict 10
  Try: 5 timesteps → predict 10

- Are we in a local minimum?
  All methods converging to same value suggests yes

Recommendations:
- Increase learning rate (1e-3 or 1e-2)
- Use multi-timestep inputs
- Add gradient clipping to prevent instability
- Try different optimizers (SGD with momentum)


## evaluate_tta_multistep.py - Tests with temporal context
Multi-Step TTA Evaluation
======================================================================
Timestamp: 2025-07-19 07:03:25

Loading data...

======================================================================
Testing with 1 input timesteps
======================================================================

Training data shape: X=(3200, 1, 8), y=(3200, 10, 8)
Training model...
Final training loss: 1602.2776

Baseline evaluation (no TTA):
  MSE: 2186.93 (±686.40)

TTA Results:

TENT:
  MSE: 4458.53 (±1644.03)
  Average improvement: -123.3%
  Samples improved: 3/20

PhysicsTENT:
  MSE: 4442.43 (±1640.19)
  Average improvement: -123.0%
  Samples improved: 3/20

TTT:
  MSE: 4448.30 (±1599.40)
  Average improvement: -124.0%
  Samples improved: 3/20

======================================================================
Testing with 3 input timesteps
======================================================================

Training data shape: X=(3040, 3, 8), y=(3040, 10, 8)
Training model...
Final training loss: 1406.5481

Baseline evaluation (no TTA):
  MSE: 1839.25 (±576.15)

TTA Results:

TENT:
  MSE: 4788.30 (±1518.32)
  Average improvement: -188.3%
  Samples improved: 1/20

PhysicsTENT:
  MSE: 4788.63 (±1518.12)
  Average improvement: -189.0%
  Samples improved: 1/20

TTT:
  MSE: 4784.83 (±1517.04)
  Average improvement: -168.8%
  Samples improved: 4/20

======================================================================
Testing with 5 input timesteps
======================================================================

Training data shape: X=(2880, 5, 8), y=(2880, 10, 8)
Training model...
Final training loss: 1406.5396

Baseline evaluation (no TTA):
  MSE: 2334.37 (±870.75)

TTA Results:

TENT:
  MSE: 4731.10 (±1412.23)
  Average improvement: -138.8%
  Samples improved: 3/20

PhysicsTENT:
  MSE: 4764.89 (±1407.64)
  Average improvement: -140.9%
  Samples improved: 3/20

TTT:
  MSE: 4760.75 (±1416.54)
  Average improvement: -125.6%
  Samples improved: 4/20

======================================================================
Testing with 10 input timesteps
======================================================================

Training data shape: X=(2480, 10, 8), y=(2480, 10, 8)
Training model...
Final training loss: 1539.4158

Baseline evaluation (no TTA):
  MSE: 2832.06 (±917.08)

TTA Results:

TENT:
  MSE: 4657.62 (±771.42)
  Average improvement: -80.9%
  Samples improved: 1/20

PhysicsTENT:
  MSE: 4636.33 (±773.74)
  Average improvement: -80.3%
  Samples improved: 1/20

TTT:
  MSE: 4647.58 (±771.25)
  Average improvement: -70.0%
  Samples improved: 3/20

======================================================================
Hyperparameter Search (5-step input)
======================================================================

======================================================================
KEY FINDINGS
======================================================================

1. Multi-step inputs should provide richer adaptation signal
2. LSTM can capture temporal patterns better
3. Higher learning rates (1e-3) may work better with more information
4. Look for configurations where >50% of samples improve


## tta_hyperparameter_search.py - Systematic parameter search
TTA Hyperparameter Search
======================================================================
Timestamp: 2025-07-19 07:19:56

Searching TENT configurations...
----------------------------------------------------------------------

Testing TENT_lr1e-05_s1_dense...
  Baseline MSE: 8649.26
  TTA MSE: 8730.45
  Improvement: -1.0% (±0.8%)
  Improved ratio: 0.10

Testing TENT_lr1e-05_s1_deep...
  Baseline MSE: 8636.59
  TTA MSE: 8730.86
  Improvement: -1.1% (±0.9%)
  Improved ratio: 0.10

Testing TENT_lr1e-05_s5_dense...
  Baseline MSE: 8638.67
  TTA MSE: 8731.34
  Improvement: -1.1% (±0.8%)
  Improved ratio: 0.10

Testing TENT_lr1e-05_s5_deep...
  Baseline MSE: 8698.56
  TTA MSE: 8732.22
  Improvement: -0.4% (±1.0%)
  Improved ratio: 0.40

Testing TENT_lr1e-05_s10_dense...
  Baseline MSE: 8671.80
  TTA MSE: 8730.88
  Improvement: -0.7% (±0.8%)
  Improved ratio: 0.30

Testing TENT_lr1e-05_s10_deep...
  Baseline MSE: 8690.47
  TTA MSE: 8731.83
  Improvement: -0.5% (±0.7%)
  Improved ratio: 0.10

Testing TENT_lr1e-05_s20_dense...
  Baseline MSE: 8661.07
  TTA MSE: 8731.83
  Improvement: -0.8% (±0.7%)
  Improved ratio: 0.10

Testing TENT_lr1e-05_s20_deep...
  Baseline MSE: 8707.67
  TTA MSE: 8731.86
  Improvement: -0.3% (±0.5%)
  Improved ratio: 0.20

Testing TENT_lr0.0001_s1_dense...
  Baseline MSE: 8658.70
  TTA MSE: 8732.28
  Improvement: -0.9% (±0.7%)
  Improved ratio: 0.00

Testing TENT_lr0.0001_s1_deep...
  Baseline MSE: 8700.01
  TTA MSE: 8731.42
  Improvement: -0.4% (±0.7%)
  Improved ratio: 0.30

Testing TENT_lr0.0001_s5_dense...
  Baseline MSE: 8674.53
  TTA MSE: 8730.79
  Improvement: -0.7% (±0.6%)
  Improved ratio: 0.10

Testing TENT_lr0.0001_s5_deep...
  Baseline MSE: 8628.89
  TTA MSE: 8730.35
  Improvement: -1.2% (±1.0%)
  Improved ratio: 0.10

Testing TENT_lr0.0001_s10_dense...
  Baseline MSE: 8605.07
  TTA MSE: 8729.77
  Improvement: -1.4% (±1.2%)
  Improved ratio: 0.10

Testing TENT_lr0.0001_s10_deep...
  Baseline MSE: 8712.33
  TTA MSE: 8732.22
  Improvement: -0.2% (±0.5%)
  Improved ratio: 0.30

Testing TENT_lr0.0001_s20_dense...
  Baseline MSE: 8655.26
  TTA MSE: 8730.84
  Improvement: -0.9% (±0.9%)
  Improved ratio: 0.20

Testing TENT_lr0.0001_s20_deep...
  Baseline MSE: 8658.74
  TTA MSE: 8731.88
  Improvement: -0.9% (±1.1%)
  Improved ratio: 0.10

Testing TENT_lr0.001_s1_dense...
  Baseline MSE: 8720.55
  TTA MSE: 8731.87
  Improvement: -0.1% (±0.8%)
  Improved ratio: 0.40

Testing TENT_lr0.001_s1_deep...
  Baseline MSE: 8675.42
  TTA MSE: 8730.88
  Improvement: -0.7% (±0.7%)
  Improved ratio: 0.10

Testing TENT_lr0.001_s5_dense...
  Baseline MSE: 8624.71
  TTA MSE: 8731.18
  Improvement: -1.3% (±1.4%)
  Improved ratio: 0.10

Testing TENT_lr0.001_s5_deep...
  Baseline MSE: 8706.28
  TTA MSE: 8731.98
  Improvement: -0.3% (±0.4%)
  Improved ratio: 0.30

Testing TENT_lr0.001_s10_dense...
  Baseline MSE: 8679.26
  TTA MSE: 8731.54
  Improvement: -0.7% (±1.2%)
  Improved ratio: 0.20

Testing TENT_lr0.001_s10_deep...
  Baseline MSE: 8676.83
  TTA MSE: 8732.36
  Improvement: -0.7% (±0.6%)
  Improved ratio: 0.00

Testing TENT_lr0.001_s20_dense...
  Baseline MSE: 8655.37
  TTA MSE: 8732.33
  Improvement: -0.9% (±1.1%)
  Improved ratio: 0.10

Testing TENT_lr0.001_s20_deep...
  Baseline MSE: 8697.90
  TTA MSE: 8732.29
  Improvement: -0.4% (±0.6%)
  Improved ratio: 0.10

Testing TENT_lr0.01_s1_dense...
  Baseline MSE: 8607.71
  TTA MSE: 8730.83
  Improvement: -1.5% (±1.0%)
  Improved ratio: 0.00

Testing TENT_lr0.01_s1_deep...
  Baseline MSE: 8641.64
  TTA MSE: 8730.80
  Improvement: -1.1% (±0.9%)
  Improved ratio: 0.00

Testing TENT_lr0.01_s5_dense...
  Baseline MSE: 8694.94
  TTA MSE: 8730.90
  Improvement: -0.5% (±1.0%)
  Improved ratio: 0.20

Testing TENT_lr0.01_s5_deep...
  Baseline MSE: 8738.35
  TTA MSE: 8730.91
  Improvement: 0.1% (±0.5%)
  Improved ratio: 0.50

Testing TENT_lr0.01_s10_dense...
  Baseline MSE: 8663.76
  TTA MSE: 8732.27
  Improvement: -0.8% (±0.7%)
  Improved ratio: 0.10

Testing TENT_lr0.01_s10_deep...
  Baseline MSE: 8645.23
  TTA MSE: 8731.48
  Improvement: -1.1% (±1.0%)
  Improved ratio: 0.00

Testing TENT_lr0.01_s20_dense...
  Baseline MSE: 8658.09
  TTA MSE: 8730.91
  Improvement: -0.9% (±0.9%)
  Improved ratio: 0.10

Testing TENT_lr0.01_s20_deep...
  Baseline MSE: 8688.08
  TTA MSE: 8730.45
  Improvement: -0.5% (±0.9%)
  Improved ratio: 0.40

======================================================================
Testing PhysicsTENT variations...
----------------------------------------------------------------------

Testing PhysicsTENT_pw0.01...
  Improvement: -0.7%

Testing PhysicsTENT_pw0.1...
  Improvement: -0.3%

Testing PhysicsTENT_pw0.5...
  Improvement: -1.2%

Testing PhysicsTENT_pw1.0...
  Improvement: -0.8%

======================================================================
SUMMARY
======================================================================

Top 5 Configurations:

1. TENT_lr0.01_s5_deep
   Improvement: 0.1% (±0.5%)
   Improved ratio: 0.50
   Params: {'learning_rate': 0.01, 'adaptation_steps': 5, 'architecture': 'deep', 'hidden_units': 128}

2. TENT_lr0.001_s1_dense
   Improvement: -0.1% (±0.8%)
   Improved ratio: 0.40
   Params: {'learning_rate': 0.001, 'adaptation_steps': 1, 'architecture': 'dense', 'hidden_units': 128}

3. TENT_lr0.0001_s10_deep
   Improvement: -0.2% (±0.5%)
   Improved ratio: 0.30
   Params: {'learning_rate': 0.0001, 'adaptation_steps': 10, 'architecture': 'deep', 'hidden_units': 128}

4. TENT_lr1e-05_s20_deep
   Improvement: -0.3% (±0.5%)
   Improved ratio: 0.20
   Params: {'learning_rate': 1e-05, 'adaptation_steps': 20, 'architecture': 'deep', 'hidden_units': 128}

5. PhysicsTENT_pw0.1
   Improvement: -0.3% (±0.6%)
   Improved ratio: 0.40

Results saved to: /Users/fergusmeiklejohn/dev/neural_networks_research/experiments/01_physics_worlds/outputs/tta_evaluation/hyperparameter_search_20250719_072029.json

======================================================================
KEY INSIGHTS
======================================================================

✓ TTA CAN improve performance!
Best improvement: 0.1%
Best config: TENT_lr0.01_s5_deep