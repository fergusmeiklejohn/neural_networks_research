# 3. Methodology

## 3.1 Experimental Setup

### 3.1.1 Physics Environment

We conduct our experiments using a 2D ball dynamics simulation that models gravitational interactions between objects. This environment provides a controlled setting where physical parameters can be systematically varied to create different distribution conditions.

The simulation models the motion of two balls under gravitational influence, with trajectories governed by:

$$\mathbf{F} = m\mathbf{a} = m\mathbf{g}$$

where $\mathbf{g}$ represents the gravitational acceleration vector. Each trajectory consists of position, velocity, mass, and radius information for both balls over 180 timesteps (3 seconds at 60 Hz).

### 3.1.2 Training and Test Distributions

We define three primary data distributions:

**Training Distribution:**
- Earth gravity: g = -9.8 m/s² (400 trajectories)
- Mars gravity: g = -3.7 m/s² (100 trajectories)
- Total: 500 trajectories with mixed gravity conditions

**Standard OOD Test Distribution:**
- Jupiter gravity: g = -24.8 m/s² (200 trajectories)
- Represents 2.5x Earth gravity (parametric extrapolation)

**True OOD Test Distribution:**
- Time-varying gravity: $g(t) = -9.8 \cdot (1 + A\sin(2\pi ft + \phi))$
- Frequency f ∈ [0.5, 2.0] Hz
- Amplitude A ∈ [0.2, 0.4]
- Represents structural change unachievable through parameter interpolation

### 3.1.3 Data Representation

Trajectories are represented as 13-dimensional vectors at each timestep:
- Time: 1 dimension
- Ball 1: position (x, y), velocity (vx, vy), mass, radius (6 dimensions)
- Ball 2: position (x, y), velocity (vx, vy), mass, radius (6 dimensions)

Physical units are normalized with 40 pixels = 1 meter for consistent numerical stability.

## 3.2 Representation Space Analysis

### 3.2.1 t-SNE Visualization

We employ t-distributed Stochastic Neighbor Embedding (t-SNE) to visualize the learned representations of different models. For each trained model, we:

1. Extract intermediate representations from the penultimate layer
2. Apply t-SNE with perplexity=30 and n_components=2
3. Color-code points by their source distribution (Earth, Mars, Jupiter)

### 3.2.2 k-NN Distance Analysis

To quantify whether test samples require interpolation or extrapolation, we employ k-nearest neighbor (k-NN) distance analysis:

1. For each test sample, compute the mean Euclidean distance to its k=10 nearest neighbors in the training set (in representation space)
2. Establish thresholds using the 95th and 99th percentiles of training set self-distances
3. Classify test samples as:
   - Interpolation: distance ≤ 95th percentile
   - Near-boundary: 95th < distance ≤ 99th percentile
   - Extrapolation: distance > 99th percentile

This approach is more robust to high dimensionality than convex hull analysis and provides a continuous measure of extrapolation difficulty.

### 3.2.3 Density Estimation

We use kernel density estimation (KDE) to analyze the distribution of representations:

$$\hat{f}(x) = \frac{1}{nh} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)$$

where K is a Gaussian kernel and h is optimized via cross-validation.

## 3.3 Baseline Models

We implement and evaluate five baseline approaches:

### 3.3.1 ERM with Data Augmentation
Standard empirical risk minimization with augmentation strategies:
- Architecture: 3-layer MLP (256-128-64 units)
- Augmentation: Gaussian noise (σ=0.1) and trajectory reversal
- Training: Adam optimizer, learning rate 1e-3

### 3.3.2 GFlowNet
Exploration-based approach for discovering diverse solutions:
- Architecture: Encoder-decoder with latent exploration
- Exploration bonus: Added noise in latent space
- Objective: Balance between reconstruction and diversity

### 3.3.3 Graph Neural Network (GraphExtrap)
Physics-aware architecture using geometric features:
- Input transformation: Cartesian to polar coordinates
- Architecture: 3-layer graph network with edge features
- Inductive bias: Rotation and translation invariance

### 3.3.4 Model-Agnostic Meta-Learning (MAML)
Adaptation-focused approach for few-shot generalization:
- Architecture: 4-layer MLP with skip connections
- Inner loop: 5 gradient steps
- Meta-objective: Fast adaptation to new physics

### 3.3.5 Minimal Physics-Informed Neural Network
Simplified PINN incorporating F=ma directly:
- Base prediction: Newton's second law
- Neural correction: Small residual network
- Loss: MSE + 100x physics consistency penalty

## 3.4 Evaluation Protocol

### 3.4.1 Performance Metrics

We evaluate models using:
- **Mean Squared Error (MSE)**: Primary metric for trajectory prediction
- **Physics Consistency**: Deviation from conservation laws
- **Representation Interpolation Ratio**: Fraction of test samples requiring only interpolation

### 3.4.2 Training Procedure

All models are trained using:
- Batch size: 32
- Maximum epochs: 100 (with early stopping)
- Validation split: 20% of training data
- Hardware: Single GPU (for reproducibility)

### 3.4.3 Statistical Analysis

We report:
- Mean and standard deviation over 3 random seeds
- 95% confidence intervals where applicable
- Statistical significance tests (paired t-test) for performance comparisons

## 3.5 True OOD Benchmark Generation

To create genuinely out-of-distribution test cases, we:

1. Generate trajectories with time-varying gravity
2. Verify no overlap with training distribution using representation analysis
3. Ensure physical plausibility through energy conservation checks
4. Create visualizations comparing constant vs time-varying dynamics

This methodology enables systematic investigation of the interpolation-extrapolation distinction in physics-informed machine learning.
