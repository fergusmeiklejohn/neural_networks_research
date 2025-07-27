# Data Isolation Fix Plan for Physics Worlds Experiment

## Executive Summary

Our current Physics Worlds experiment has a critical flaw: **there is no true test set isolation**. All datasets (train, validation, test) are randomly sampled from the same distribution with identical parameter ranges. This means we're not actually testing the model's ability to *invent* new distributions, but merely its ability to interpolate within the training distribution.

This document outlines the problems, explains why they matter, and provides a comprehensive plan to fix them.

## 1. Current Problems Explained

### 1.1 The Fundamental Issue: Identical Distribution Sampling

Currently, all our datasets are generated using the same process:

```python
# Current approach - PROBLEMATIC
def _generate_random_physics_config(self) -> PhysicsConfig:
    return PhysicsConfig(
        gravity=random.uniform(*self.config.gravity_range),    # Same range for all splits
        friction=random.uniform(*self.config.friction_range),  # Same range for all splits
        elasticity=random.uniform(*self.config.elasticity_range), # Same range for all splits
        damping=random.uniform(*self.config.damping_range)     # Same range for all splits
    )
```

**Why this is a problem**: 
- Test samples are statistically indistinguishable from training samples
- We can't measure true generalization or distribution invention
- High "test accuracy" may be meaningless - just interpolation

### 1.2 No Systematic Coverage of Parameter Space

Current generation for all splits:
```python
# Random sampling means gaps and overlaps
train_physics = [random_physics() for _ in range(10000)]
val_physics = [random_physics() for _ in range(2000)]  
test_physics = [random_physics() for _ in range(1000)]
```

**Problems**:
- Random sampling doesn't guarantee coverage
- Test set may randomly exclude important parameter combinations
- No way to test specific physics regimes (e.g., "low gravity + high friction")

### 1.3 Modification Pairs Share Base Distribution

```python
# Modification pairs generated from same distribution
def generate_modification_pairs(self, base_samples: int = 1000):
    for sample_id in range(base_samples):
        base_config = self._generate_random_physics_config()  # Same random process!
```

**Why this matters**:
- Model learns modifications on same physics it will be tested on
- Can't test novel modification types or physics regimes
- Defeats the purpose of "distribution invention"

### 1.4 Quality Validation Creates Selection Bias

```python
# Same validation thresholds for all splits
if abs(energy_conservation) > self.config.energy_conservation_threshold:
    return False, f"Poor energy conservation"  # Sample discarded
```

**Hidden assumption**: By discarding "invalid" samples uniformly, we may be:
- Creating different effective distributions for each split
- Biasing towards "easy" physics parameters
- Hiding regions where our physics simulation breaks down

## 2. Proper Test Set Design Principles

### 2.1 Interpolation vs Extrapolation

**Interpolation Test Set**: Tests within the convex hull of training data
- Purpose: Verify basic learning
- Example: Gravity=-800 when trained on range [-1500, -200]

**Extrapolation Test Set**: Tests outside training distribution
- Purpose: Verify distribution invention capability  
- Example: Gravity=-2000 or Gravity=-100 (outside training range)

### 2.2 Systematic Parameter Space Coverage

Instead of random sampling, use systematic approaches:

1. **Grid-based sampling**: Cover parameter space systematically
2. **Latin Hypercube Sampling**: Ensure even coverage with fewer samples
3. **Boundary testing**: Explicitly test parameter space edges
4. **Combination testing**: Test specific parameter combinations

### 2.3 Held-Out Domains

Reserve entire physics regimes for testing:
- **Moon physics**: Very low gravity (-50), no air resistance
- **Underwater physics**: High damping (0.5), reduced gravity (-600)
- **Superbounce world**: Elasticity > 1.0 (energy-gaining collisions)

## 3. Proposed Solution: Three-Tier Data Strategy

### 3.1 Training Set (70% of data)

**Parameter ranges**:
```python
TRAIN_RANGES = {
    'gravity': (-1200, -300),      # Narrower than current
    'friction': (0.1, 0.8),        # Exclude extremes
    'elasticity': (0.2, 0.9),      # Exclude extremes
    'damping': (0.85, 0.98)        # Typical air resistance
}
```

**Sampling strategy**: 
- Random uniform sampling within ranges
- Ensure minimum coverage via grid initialization

### 3.2 Validation Set (15% of data)

Two types of validation data:

**3.2.1 In-Distribution Validation** (10%)
- Same ranges as training
- Different random seed
- For hyperparameter tuning

**3.2.2 Near-Distribution Validation** (5%)
- Slightly outside training ranges
- Tests early generalization
```python
NEAR_VAL_RANGES = {
    'gravity': (-1400, -200),      # 15% extension
    'friction': (0.05, 0.9),       # Include more extremes
    'elasticity': (0.1, 0.95),     
    'damping': (0.82, 0.99)        
}
```

### 3.3 Test Set (15% of data)

Three distinct test sets for comprehensive evaluation:

**3.3.1 Interpolation Test Set** (5%)
- Within training ranges but systematic sampling
- Grid-based to ensure coverage
- Tests basic learning

**3.3.2 Extrapolation Test Set** (5%)
- Outside training ranges
```python
EXTRAP_TEST_RANGES = {
    'gravity': [(-2000, -1500), (-200, -50)],  # Very high/low gravity
    'friction': [(0.0, 0.05), (0.9, 1.0)],     # Frictionless/sticky
    'elasticity': [(0.0, 0.1), (0.95, 1.1)],   # No bounce/super bounce
    'damping': [(0.7, 0.82), (0.99, 1.0)]      # Thick air/vacuum
}
```

**3.3.3 Novel Physics Regimes** (5%)
Predefined physics scenarios never seen in training:
```python
NOVEL_REGIMES = {
    'moon': {'gravity': -100, 'friction': 0.7, 'elasticity': 0.8, 'damping': 1.0},
    'jupiter': {'gravity': -3000, 'friction': 0.5, 'elasticity': 0.6, 'damping': 0.9},
    'underwater': {'gravity': -600, 'friction': 0.3, 'elasticity': 0.4, 'damping': 0.7},
    'ice_rink': {'gravity': -981, 'friction': 0.02, 'elasticity': 0.9, 'damping': 0.98},
    'rubber_room': {'gravity': -981, 'friction': 0.9, 'elasticity': 0.99, 'damping': 0.95}
}
```

## 4. Implementation Plan

### 4.1 New Data Generator Structure

```python
class ImprovedPhysicsDataGenerator:
    def __init__(self, config: DataConfig):
        self.config = config
        self.define_data_splits()
        
    def define_data_splits(self):
        """Define parameter ranges for each data split"""
        self.splits = {
            'train': {
                'ranges': TRAIN_RANGES,
                'sampling': 'random_uniform',
                'size': 0.7
            },
            'val_in_dist': {
                'ranges': TRAIN_RANGES,
                'sampling': 'random_uniform', 
                'size': 0.1
            },
            'val_near_dist': {
                'ranges': NEAR_VAL_RANGES,
                'sampling': 'random_uniform',
                'size': 0.05
            },
            'test_interpolation': {
                'ranges': TRAIN_RANGES,
                'sampling': 'grid',
                'size': 0.05
            },
            'test_extrapolation': {
                'ranges': EXTRAP_TEST_RANGES,
                'sampling': 'systematic',
                'size': 0.05
            },
            'test_novel': {
                'regimes': NOVEL_REGIMES,
                'sampling': 'predefined',
                'size': 0.05
            }
        }
```

### 4.2 Modification Pairs Strategy

```python
def generate_modification_pairs_v2(self):
    """Generate modification pairs with proper train/test split"""
    
    # Training modifications: Standard types on training physics
    train_modifications = {
        'base_physics': 'train_ranges',
        'modification_types': [
            'gravity_increase_20%',
            'gravity_decrease_20%', 
            'friction_increase',
            'friction_decrease',
            # ... standard modifications
        ]
    }
    
    # Test modifications: Novel types and out-of-distribution base physics
    test_modifications = {
        'base_physics': 'test_ranges',
        'modification_types': [
            'gravity_reverse',  # Negative to positive gravity
            'remove_all_damping',  # Perfect vacuum
            'make_sticky',  # Friction -> 1.0
            'quantum_bounce',  # Elasticity -> 1.5
            'combine_moon_underwater',  # Complex modification
        ]
    }
```

### 4.3 Evaluation Metrics Update

New metrics to properly measure distribution invention:

```python
class DistributionInventionMetrics:
    def __init__(self):
        self.metrics = {
            'interpolation_accuracy': None,      # Performance on in-distribution test
            'extrapolation_accuracy': None,      # Performance on out-of-distribution  
            'novel_regime_success': None,        # Performance on predefined regimes
            'modification_consistency': None,     # Rule modification accuracy
            'distribution_distance': None,       # KL divergence from training
            'physics_plausibility': None,        # Energy conservation, smoothness
            'invention_score': None              # Combined metric for paper
        }
        
    def compute_invention_score(self):
        """Weighted combination emphasizing true generalization"""
        weights = {
            'interpolation': 0.2,    # Less important
            'extrapolation': 0.4,    # Most important  
            'novel_regime': 0.3,     # Important
            'consistency': 0.1       # Supporting metric
        }
```

## 5. Backwards Compatibility

To maintain compatibility while fixing issues:

1. **Keep old data files**: Don't delete existing datasets
2. **Version new data**: Use `_v2` suffix for new data files
3. **Add compatibility flag**: 
```python
use_improved_data_split = True  # Flag in config
```
4. **Provide conversion script**: Transform old results to new metrics

## 6. Implementation Checklist

- [ ] Create new `ImprovedPhysicsDataGenerator` class
- [ ] Implement systematic sampling methods (grid, LHS)
- [ ] Define and implement three-tier data splits
- [ ] Generate new datasets with proper isolation
- [ ] Update training scripts to use new data structure
- [ ] Implement new evaluation metrics
- [ ] Create visualization tools for parameter space coverage
- [ ] Document parameter range choices and rationale
- [ ] Add tests to verify no data leakage between splits
- [ ] Update EXPERIMENT_PLAN.md with new methodology
- [ ] Re-run all experiments with proper data isolation
- [ ] Compare old vs new results to understand impact

## 7. Expected Impact

With proper data isolation:

1. **Lower test scores**: We expect significantly lower "accuracy" on true test sets
2. **Clearer research direction**: Will reveal where the model actually fails
3. **Valid conclusions**: Can make legitimate claims about distribution invention
4. **Better model development**: Focus on architectures that truly generalize

## 8. Timeline

1. **Week 1**: Implement new data generator and create datasets
2. **Week 2**: Update training/evaluation code, re-run experiments
3. **Week 3**: Analyze results, iterate on model architecture
4. **Week 4**: Document findings and prepare for next experiments

## Conclusion

Fixing data isolation is critical for the validity of our research. While it may initially show worse results, it will provide genuine insights into distribution invention and guide us toward architectures that truly generalize beyond their training distribution. This is essential for achieving our goal of neural networks that can think outside their training box.