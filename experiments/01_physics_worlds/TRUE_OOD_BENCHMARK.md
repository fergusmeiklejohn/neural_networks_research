# True OOD Physics Benchmark: Ensuring Genuine Extrapolation

## Problem Statement

Our analysis revealed that 91.7% of "far-OOD" Jupiter gravity samples were actually interpolation in representation space. This means current benchmarks don't test true extrapolation. We need benchmarks where test samples are genuinely outside the training manifold.

## Design Principles

### 1. Representation-Based Data Splitting
Instead of splitting by parameter values, split by learned representations:
- Train model on initial data
- Extract representations
- Use density estimation to identify training manifold
- Create test sets guaranteed to be outside this manifold

### 2. Causal Structure Changes
Test modifications that change causal relationships, not just parameters:
- **Reverse Gravity**: Objects fall up (changes causal direction)
- **Magnetic Forces**: Add new force type (new causal factor)
- **Non-Conservative Forces**: Break energy conservation (violates assumption)
- **Variable Gravity**: g changes with height (new functional form)

### 3. Compositional Extrapolation
Test novel combinations that can't be decomposed:
- **Rotating Reference Frame**: Coriolis and centrifugal forces
- **Charged Particles**: Electromagnetic interactions
- **Relativistic Speeds**: Lorentz transformations needed
- **Quantum Effects**: Uncertainty principle emerges

## Benchmark Structure

### Level 1: Parameter Extrapolation (Baseline)
**Training**: g ∈ [-10, -5] m/s²
**Test**: g ∈ [-25, -20] m/s²
**Expected**: Most samples still interpolation

### Level 2: Functional Form Changes
**Training**: g = constant
**Test**: g = -9.8 * (1 + 0.1*sin(t))
**True OOD**: ~60% of samples

### Level 3: New Physics
**Training**: Gravity only
**Test**: Gravity + magnetic field
**True OOD**: ~90% of samples

### Level 4: Causal Reversal
**Training**: Normal physics
**Test**: Time-reversed physics
**True OOD**: ~95% of samples

## Implementation

### Step 1: Create Base Dataset
```python
def create_true_ood_benchmark():
    # Standard physics for training
    train_data = generate_standard_physics(
        gravity_range=(-10, -5),
        n_samples=10000
    )

    # Extract representations after initial training
    model = train_baseline(train_data)
    representations = model.get_representations(train_data)

    # Fit density model
    density_model = fit_kde(representations)

    return train_data, density_model
```

### Step 2: Generate True OOD Test Sets
```python
def generate_true_ood_tests(density_model):
    test_sets = {}

    # Level 2: Time-varying gravity
    test_sets['varying_gravity'] = generate_physics_with(
        gravity_fn=lambda t: -9.8 * (1 + 0.1*np.sin(0.5*t)),
        filter_fn=lambda x: density_model.score(x) < threshold
    )

    # Level 3: Add magnetic forces
    test_sets['magnetic'] = generate_electromagnetic_physics(
        B_field=0.1,  # Tesla
        charge=1e-6,  # Coulombs
        filter_fn=lambda x: density_model.score(x) < threshold
    )

    # Level 4: Reversed causality
    test_sets['reversed'] = generate_reversed_physics(
        time_direction=-1,
        filter_fn=lambda x: density_model.score(x) < threshold
    )

    return test_sets
```

### Step 3: Verify True OOD
```python
def verify_ood_percentage(train_repr, test_data, model):
    test_repr = model.get_representations(test_data)

    # Use multiple metrics
    knn_distances = compute_knn_distances(train_repr, test_repr, k=10)
    densities = density_model.score_samples(test_repr)

    # Classify as OOD if far from all training points
    is_ood = (knn_distances > np.percentile(train_distances, 95)) &
             (densities < np.percentile(train_densities, 5))

    return np.mean(is_ood) * 100
```

## Specific Test Cases

### Test 1: Harmonic Gravity
- **Physics**: g(t) = -9.8 * (1 + 0.3*sin(πt))
- **Why OOD**: No periodic gravity in training
- **Challenge**: Requires learning time-dependent forces

### Test 2: Coupled Oscillators
- **Physics**: Springs connect balls with k=100 N/m
- **Why OOD**: New interaction type
- **Challenge**: Must learn spring forces

### Test 3: Dissipative Forces
- **Physics**: Quadratic drag ∝ v²
- **Why OOD**: Training has linear friction
- **Challenge**: Different velocity dependence

### Test 4: Frame Transformations
- **Physics**: Rotating reference frame (ω = 0.1 rad/s)
- **Why OOD**: Fictitious forces appear
- **Challenge**: Non-inertial dynamics

## Evaluation Protocol

### 1. Train Model
- Use standard physics only
- No hints about test modifications
- Standard architectures allowed

### 2. Extract Representations
- Get embeddings from trained model
- Compute training manifold boundary
- Verify test is outside manifold

### 3. Test Extrapolation
- Measure performance on each level
- Report % true OOD for each test
- Compare multiple architectures

### 4. Validate Physics
- Check if predictions are physically plausible
- Measure conservation law violations
- Assess trajectory smoothness

## Expected Results

### Current Models (Predicted)
- Level 1: 80% accuracy (mostly interpolation)
- Level 2: 40% accuracy (some extrapolation)
- Level 3: 15% accuracy (true extrapolation)
- Level 4: 5% accuracy (causal confusion)

### Ideal Model Should
- Recognize when physics changes
- Adapt internal model accordingly
- Maintain physical consistency
- Flag impossible scenarios

## Success Metrics

A model passes if it:
1. **Achieves >50% on Level 2** (functional changes)
2. **Achieves >30% on Level 3** (new physics)
3. **Degrades gracefully** (not catastrophically)
4. **Maintains physics constraints** where applicable

## Implementation Checklist

- [ ] Create standard physics generator
- [ ] Implement representation extractor
- [ ] Build density estimation pipeline
- [ ] Generate Level 2 test sets (functional)
- [ ] Generate Level 3 test sets (new physics)
- [ ] Generate Level 4 test sets (causal)
- [ ] Implement evaluation metrics
- [ ] Create visualization tools
- [ ] Document failure modes

## Key Innovation

This benchmark ensures true OOD by:
1. **Representation-based filtering** - Not just parameter-based
2. **Causal structure changes** - Not just parameter shifts
3. **Verification protocol** - Proves samples are OOD
4. **Hierarchical difficulty** - From simple to impossible

## Conclusion

Current "OOD" benchmarks test interpolation in disguise. This true OOD benchmark forces models to extrapolate by:
- Changing functional forms
- Adding new physics
- Reversing causality
- Verifying representation novelty

Models that succeed here truly understand physics, not just pattern matching.
