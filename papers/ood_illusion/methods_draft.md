# 3. The Physics Extrapolation Challenge

## 3.1 Task Definition

We study a fundamental physics prediction task: forecasting the motion of two interacting balls under gravitational influence. Given the current state of the system (positions and velocities of both balls), the model must predict the state after a fixed time interval.

The state vector x ∈ ℝ¹¹ consists of:
- Ball 1: position (x₁, y₁), velocity (vₓ₁, vᵧ₁), radius r₁
- Ball 2: position (x₂, y₂), velocity (vₓ₂, vᵧ₂), radius r₂  
- Gravity parameter: g

The prediction target is the state after Δt = 0.1 seconds, accounting for gravitational acceleration and elastic collisions between balls and with boundaries.

## 3.2 Training and Test Distributions

### Training Distribution
During training, we generate trajectories with:
- Constant gravity: g = 9.8 m/s²
- Ball positions: uniformly sampled in [100, 900] × [100, 900] pixels
- Ball velocities: uniformly sampled in [-5, 5] m/s per component
- Ball radii: uniformly sampled in [10, 30] pixels
- Trajectory length: 50 time steps (5 seconds)

We generate 10,000 training trajectories, resulting in 500,000 state transition pairs. The dynamics follow standard Newtonian mechanics with elastic collisions.

### Test Distributions
We evaluate on two test sets:

**In-Distribution Test Set**: 1,000 trajectories generated with the same process as training data (constant g = 9.8 m/s²).

**Time-Varying Gravity Test Set**: 1,000 trajectories where gravity varies as:
g(t) = 9.8 + 2sin(0.1t) m/s²

This creates gravity that oscillates between 7.8 and 11.8 m/s² with a period of approximately 63 seconds. Importantly, the model never observes time-varying gravity during training.

## 3.3 Characterizing the Distribution Shift

To understand why time-varying gravity represents out-of-distribution data, we analyze the learned representations. A model trained on constant gravity learns features that capture:
- Relative positions and velocities
- Collision dynamics
- Constant downward acceleration

When gravity becomes time-varying, the required computation changes. The model must now predict acceleration that varies with time—a computation for which it has learned no features. This represents not merely different input statistics but a different computational requirement.

We verify this distinction empirically. Using the penultimate layer representations of a trained model, we compute:
1. The convex hull of training data representations
2. The distance from test points to this hull

For standard corruptions (e.g., adding Gaussian noise to inputs), test points remain within or near the training hull. For time-varying gravity, test points lie significantly outside, confirming they require extrapolation rather than interpolation.

## 3.4 Model Architecture

We use a standard feedforward neural network to ensure fair comparison across methods:
- Input layer: 11 dimensions
- Hidden layers: 3 layers of 256 units each
- Activation: ReLU
- Output layer: 11 dimensions (predicting full state)
- Loss function: Mean squared error

This architecture is sufficient to achieve low error on the training distribution while being simple enough to isolate the effects of different OOD methods. All methods use identical architectures, differing only in training procedures or adaptation mechanisms.

## 3.5 Baseline Training

Our baseline model uses standard empirical risk minimization:
- Optimizer: Adam with learning rate 0.001
- Batch size: 256
- Training epochs: 100
- Data augmentation: None

We train until convergence on the training set, achieving mean squared error below 100 on held-out constant-gravity validation data. This represents accurate prediction within the training distribution.

## 3.6 OOD Methods Implementation

We implement each OOD method following published protocols:

**Test-Time Adaptation (TTA)**: We adapt the TENT approach to regression by minimizing prediction consistency loss on test batches. We update only batch normalization parameters to limit model changes. We test adaptation for 1, 10, and 50 gradient steps.

**MAML**: We implement few-shot adaptation where the model takes 10 gradient steps on a support set before making predictions. During training, we simulate this by splitting trajectories into support and query sets.

**GFlowNet-inspired**: Since standard GFlowNets are designed for discrete spaces, we implement an ensemble approach inspired by GFlowNet's exploration principle, training multiple models with different random seeds and aggregating predictions.

**Ensemble Baseline**: We train 5 models with different initializations and average their predictions.

## 3.7 Evaluation Metrics

We report:
- **Mean Squared Error (MSE)**: Average squared difference between predicted and true states
- **Relative Performance**: Ratio of method MSE to baseline MSE
- **Degradation Analysis**: How performance changes with adaptation steps
- **Component Analysis**: Error breakdown by position vs. velocity predictions

All experiments use fixed random seeds for reproducibility. We report means and standard deviations across multiple random trajectory samples from the test distribution.