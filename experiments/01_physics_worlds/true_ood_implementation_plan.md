# True OOD Benchmark Implementation Plan

## Phase 1: Time-Varying Gravity (Level 2)

### Overview
Implement a physics benchmark where gravity varies sinusoidally over time, creating genuinely out-of-distribution test cases that current models cannot solve through interpolation.

### Key Innovation
- **Training**: Constant gravity values (g ∈ [-10, -5] m/s²)
- **Testing**: Time-varying gravity g(t) = -9.8 * (1 + 0.1*sin(0.5*t))
- **Expected OOD Rate**: ~60% of test samples outside training manifold

## Implementation Steps

### Step 1: Data Generation Infrastructure
```python
# File: experiments/01_physics_worlds/generate_true_ood_data.py

def generate_time_varying_physics(
    n_trajectories: int = 1000,
    duration: float = 10.0,
    dt: float = 0.1,
    gravity_fn: Callable[[float], float] = lambda t: -9.8 * (1 + 0.1*np.sin(0.5*t))
) -> Dict[str, np.ndarray]:
    """Generate physics data with time-varying gravity."""
    # Key: gravity changes DURING each trajectory
    # This is fundamentally different from constant gravity
```

### Step 2: Representation Extraction
```python
# File: experiments/01_physics_worlds/verify_true_ood.py

class TrueOODVerifier:
    def __init__(self, model, train_data):
        self.model = model
        self.train_representations = self._extract_representations(train_data)
        self.density_model = self._fit_density_model()
    
    def verify_ood_percentage(self, test_data):
        """Verify what percentage of test data is truly OOD."""
        # Use k-NN distances + density estimation
        # Return percentage outside training manifold
```

### Step 3: Baseline Evaluation
1. **GraphExtrap**: Test if geometric features handle time-varying forces
2. **MAML**: Check if meta-learning adapts to changing dynamics
3. **GFlowNet**: Evaluate exploration in non-stationary environments
4. **ERM + Augmentation**: Baseline for standard deep learning

### Step 4: Visualization Tools
```python
# File: experiments/01_physics_worlds/visualize_true_ood.py

def plot_gravity_evolution(trajectory_data):
    """Show how gravity changes during a trajectory."""
    
def plot_representation_space(train_repr, test_repr):
    """Visualize OOD samples in representation space."""
    
def animate_trajectory_comparison(standard_physics, time_varying_physics):
    """Side-by-side animation showing difference."""
```

## Technical Details

### 1. Physics Implementation
```python
def simulate_time_varying_trajectory(
    initial_state: np.ndarray,
    gravity_fn: Callable,
    duration: float,
    dt: float
) -> np.ndarray:
    """
    Simulate 2-ball system with time-varying gravity.
    
    Key differences from constant gravity:
    - Acceleration changes each timestep
    - Energy is not conserved
    - Periodic motion emerges
    """
    trajectory = []
    state = initial_state.copy()
    
    for t in np.arange(0, duration, dt):
        # Gravity changes with time!
        g = gravity_fn(t)
        
        # Update physics with current gravity
        state = physics_step(state, g, dt)
        trajectory.append(state)
    
    return np.array(trajectory)
```

### 2. OOD Verification Protocol
```python
def compute_ood_metrics(train_data, test_data, model):
    """
    Comprehensive OOD verification using multiple metrics.
    """
    metrics = {
        'knn_distance': compute_knn_distance(train_repr, test_repr, k=10),
        'density_score': density_model.score_samples(test_repr),
        'mahalanobis': compute_mahalanobis_distance(train_repr, test_repr),
        'energy_score': compute_energy_score(model, test_data)
    }
    
    # Combine metrics for robust OOD detection
    is_ood = (
        (metrics['knn_distance'] > percentile_95) & 
        (metrics['density_score'] < percentile_5)
    )
    
    return {
        'ood_percentage': np.mean(is_ood) * 100,
        'metrics': metrics
    }
```

### 3. Expected Challenges

**For Models**:
- Must learn that gravity can change over time
- Cannot rely on energy conservation
- Need to infer time-dependency from observations

**For Implementation**:
- Ensure sufficient trajectory length to observe variation
- Balance variation amplitude (too small = undetectable, too large = chaotic)
- Verify that standard models truly fail

## Experiment Design

### Training Data
- 10,000 trajectories with constant gravity
- Gravity values: uniform sampling from [-10, -5] m/s²
- Duration: 10 seconds per trajectory
- Initial conditions: varied positions and velocities

### Test Sets
1. **Near-OOD**: g(t) = -9.8 * (1 + 0.05*sin(t)) - subtle variation
2. **True-OOD**: g(t) = -9.8 * (1 + 0.1*sin(0.5*t)) - clear variation
3. **Far-OOD**: g(t) = -9.8 * (1 + 0.3*sin(2*t)) - strong variation

### Evaluation Metrics
1. **Trajectory MSE**: Point-by-point prediction error
2. **Physical Consistency**: Conservation law violations
3. **OOD Detection**: Can model recognize distribution shift?
4. **Adaptation Speed**: How quickly does error grow?

## Implementation Timeline

### Day 1: Core Infrastructure
- [ ] Create `generate_true_ood_data.py`
- [ ] Implement time-varying physics simulation
- [ ] Generate training data (constant gravity)
- [ ] Generate test sets (time-varying gravity)

### Day 2: OOD Verification
- [ ] Implement `verify_true_ood.py`
- [ ] Extract representations from trained models
- [ ] Compute OOD percentages for each test set
- [ ] Create visualization tools

### Day 3: Baseline Testing
- [ ] Test GraphExtrap on true OOD
- [ ] Test other baselines
- [ ] Generate comprehensive results
- [ ] Document failure modes

### Day 4: Analysis & Documentation
- [ ] Analyze why models fail
- [ ] Create publication-quality figures
- [ ] Write up findings
- [ ] Plan next experiments

## Success Criteria

1. **True OOD Verified**: >60% of time-varying samples outside training manifold
2. **Model Failure**: All baselines show >10x performance degradation
3. **Clear Visualization**: Representation space clearly separates OOD
4. **Reproducible**: Complete pipeline from data to results

## Next Steps After Implementation

### Level 3: New Physics (Magnetic Forces)
- Add charged particles with B-field interactions
- Completely new force type not in training

### Level 4: Causal Reversal
- Time-reversed physics
- Tests fundamental understanding of causality

### Paper: "True OOD Benchmarks for Physics Learning"
- Document the benchmark design
- Show empirical results on all levels
- Propose evaluation standards for field

## Key Files to Reference
- `models/baseline_models.py` - Existing model implementations
- `data/physics_2ball_4gravity.pkl` - Current data format
- `utils/physics_utils.py` - Physics simulation code
- `RepresentationSpaceAnalyzer` - For OOD verification

## Commands to Get Started

```bash
# Create new implementation file
cd experiments/01_physics_worlds
touch generate_true_ood_data.py

# Test with minimal data first
python generate_true_ood_data.py --n_trajectories 100 --test_mode

# Verify OOD percentage
python verify_true_ood.py --model_path outputs/models/graph_extrap_model.keras

# Run full experiment
python run_true_ood_experiment.py
```