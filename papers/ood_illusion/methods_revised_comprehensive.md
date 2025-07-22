# 3. Experimental Setup

We investigate how current OOD methods handle mechanism shifts through two complementary physics prediction tasks. Both involve time-dependent changes to the governing equations, requiring models to extrapolate beyond their training regime.

## 3.1 Two-Ball Dynamics with Time-Varying Gravity

### Task Definition
We study the motion of two interacting balls under gravitational influence. Given the current state, models predict the state after Δt = 0.1 seconds.

**State representation** (8 dimensions):
- Ball 1: position (x₁, y₁), velocity (vₓ₁, vᵧ₁)
- Ball 2: position (x₂, y₂), velocity (vₓ₂, vᵧ₂)

### Training and Test Distributions

**Training**: Constant gravity g = 9.8 m/s²
- 10,000 trajectories × 50 timesteps = 500,000 transitions
- Random initial positions and velocities
- Elastic collisions with walls and between balls

**Test (Mechanism Shift)**: Time-varying gravity g(t) = 9.8 + 2sin(0.1t)
- Oscillates between 7.8-11.8 m/s²
- Requires learning time-dependent acceleration
- Cannot be expressed as parameter change in constant-gravity model

## 3.2 Pendulum with Time-Varying Length

To test generality and explore different mechanism types, we implement a pendulum system where length varies over time.

### Task Definition
Predict pendulum motion given current state, with state representation (5 dimensions):
- Cartesian position: (x, y)
- Angular state: (θ, θ̇)
- Current length: L

### Training and Test Distributions

**Training**: Fixed length pendulum
- Length L ∈ [0.8, 1.2] m (fixed per trajectory)
- Gravity g ∈ [9.0, 10.6] m/s²
- 8,000 training trajectories

**Test (Mechanism Shift)**: L(t) = L₀(1 + 0.2sin(0.1t))
- Introduces new physics term: -2(L̇/L)θ̇
- Violates energy conservation (work done by length change)
- Requires computational operations absent in fixed-length model

## 3.3 Mechanism Shifts vs Other Distribution Shifts

We define mechanism shifts as changes to the data-generating process that introduce new computational requirements:

| Shift Type | Example | Model Requirements |
|------------|---------|-------------------|
| Parameter shift | Different gravity constant | Adjust weights |
| Statistical shift | Different initial conditions | Robust features |
| **Mechanism shift** | Time-varying parameters | New operations |

Our experiments specifically test mechanism shifts where the functional form of the dynamics changes, not just parameter values.

## 3.4 Model Architecture

All experiments use consistent architectures:

**Two-ball system**: 
- Feedforward network: [8, 256, 256, 256, 8]
- ReLU activations, MSE loss

**Pendulum system**:
- Feedforward network: [5, 256, 256, 10×5]
- Predicts 10 future timesteps
- Reshape output to (10, 5)

## 3.5 Test-Time Adaptation Implementations

### Standard TTA (Prediction Consistency)
Following TENT, we minimize prediction consistency during test time:
```
L_consistency = Var(predictions across augmentations)
```
We adapt all parameters with learning rate 1e-4 for 20 steps.

### Physics-Aware TTA
We implement domain-specific self-supervised losses:

**Energy Conservation Loss**:
```
L_energy = Var(E(t)) where E = KE + PE
```
For pendulum: E = ½mL²θ̇² + mgL(1-cos(θ))

**Hamiltonian Consistency Loss**:
```
L_hamiltonian = ||θ̈_predicted - θ̈_physics||²
where θ̈_physics = -(g/L)sin(θ) for fixed pendulum
```

### PeTTA-Inspired Collapse Detection
We monitor adaptation health through:
- **Prediction entropy**: H(predictions) to detect diversity loss
- **Variance tracking**: Var(predictions) to detect constant outputs
- **Parameter drift**: ||θ_current - θ_initial||² to detect instability

Interventions when collapse detected:
- Reduce learning rate by 50%
- Restore earlier checkpoint if drift > threshold
- Stop adaptation if predictions become constant

## 3.6 Evaluation Protocol

### Metrics
- **MSE**: Primary accuracy metric
- **Degradation factor**: MSE_test / MSE_baseline
- **Statistical significance**: Two-sided t-test, n=5 seeds
- **95% confidence intervals**: Bootstrap with 1000 samples

### Gradient Alignment Analysis
We compute cosine similarity between gradients:
```
alignment = cos(∇L_self_supervised, ∇L_true_MSE)
```
Negative alignment indicates adaptation moves away from accurate solutions.

## 3.7 Baseline Methods

We compare against:
1. **ERM + Data Augmentation**: Standard deep learning baseline
2. **MAML**: Meta-learning for quick adaptation
3. **Deep Ensembles**: 5 models with different seeds
4. **GFlowNet-inspired**: Exploration-based approach

## 3.8 Implementation Details

- **Framework**: Keras 3 with JAX backend
- **Training**: Adam optimizer, learning rate 1e-3
- **Batch size**: 32 for training, full batch for TTA
- **Seeds**: 5 random seeds for all experiments
- **Compute**: NVIDIA A4000 GPUs

All code and data generation scripts are available for reproducibility.