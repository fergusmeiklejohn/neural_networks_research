# Variable Pendulum Length Experiment Plan

## Motivation
Address reviewer's concern about single-task evidence by adding a conceptually different mechanism shift: time-varying pendulum length.

## Physics Background

### Fixed-Length Pendulum (Training Distribution)
- Equation of motion: θ̈ = -(g/L)sin(θ)
- For small angles: θ̈ ≈ -(g/L)θ
- Period: T = 2π√(L/g)
- Energy conserved: E = ½mL²θ̇² + mgL(1-cos(θ))

### Variable-Length Pendulum (Test Distribution - Mechanism Shift)
- Length varies: L(t) = L₀(1 + αsin(ωt))
- Equation becomes: θ̈ = -(g/L(t))sin(θ) - 2(L̇/L)θ̇
- Additional term from changing length (angular momentum not conserved)
- Energy no longer conserved due to work done changing length

## Why This is a True Mechanism Shift
1. **New Physical Term**: The 2(L̇/L)θ̇ term doesn't exist in training
2. **Changed Conservation Laws**: Energy conservation breaks
3. **Different Computational Graph**: Model must learn time-dependent operations
4. **Non-interpolatable**: No weighted combination of fixed-length behaviors produces variable-length dynamics

## Implementation Details

### Data Generation
```python
def generate_pendulum_data(n_trajectories, timesteps, mechanism='fixed'):
    """Generate pendulum trajectories with fixed or variable length"""
    
    L0 = 1.0  # Base length (m)
    g = 9.8   # Gravity (m/s²)
    dt = 0.01 # Time step
    
    if mechanism == 'fixed':
        L_func = lambda t: L0
        L_dot_func = lambda t: 0
    else:  # time-varying
        alpha = 0.2  # 20% variation
        omega = 0.1  # Frequency of length variation
        L_func = lambda t: L0 * (1 + alpha * np.sin(omega * t))
        L_dot_func = lambda t: L0 * alpha * omega * np.cos(omega * t)
    
    trajectories = []
    for _ in range(n_trajectories):
        # Random initial conditions
        theta0 = np.random.uniform(-np.pi/6, np.pi/6)  # ±30 degrees
        theta_dot0 = np.random.uniform(-1, 1)
        
        states = integrate_pendulum(theta0, theta_dot0, L_func, L_dot_func, 
                                   timesteps, dt, g)
        trajectories.append(states)
    
    return np.array(trajectories)

def integrate_pendulum(theta0, theta_dot0, L_func, L_dot_func, steps, dt, g):
    """RK4 integration of variable-length pendulum"""
    # Implementation details...
```

### Model Architecture
Same CNN architecture as 2-ball experiment, but:
- Input: Sequence of (x, y) pendulum positions
- Output: Next position prediction
- Can reuse existing architecture code

### Training Protocol
1. **Phase 1**: Train on fixed-length pendulum (10,000 trajectories)
2. **Phase 2**: Evaluate on:
   - In-distribution: New fixed-length trajectories
   - Near-OOD: Slightly different fixed lengths
   - True OOD: Time-varying length

### Expected Results
If our hypothesis is correct:
- ERM: High error on variable length (30-50x baseline)
- TTA: Performance degradation (2-3x worse than no adaptation)
- MAML: Catastrophic failure with adaptation
- Physics-aware TTA: Might show less degradation but still fail

## Advantages Over 2-Ball System
1. **Simpler Physics**: One object, cleaner interpretation
2. **Well-Studied**: Extensive physics literature on variable-length pendulum
3. **Different Mechanism Type**: Length change vs force change
4. **Energy Analysis**: Can directly compute energy non-conservation

## Implementation Priority
1. Start with data generation script
2. Verify physics correctness with known solutions
3. Train baseline models
4. Test adaptation methods
5. Add to paper results

## Connection to Reviewer Comments
This directly addresses:
- "Hard to generalize from single mechanism-shift task"
- "Demonstrate phenomenon is not simulator-specific"
- Different type of mechanism change (geometric vs force)

## Minimal Version for Quick Testing
```python
# Quick test: 100 trajectories, 100 timesteps
# Just ERM and TTA to verify the effect exists
# Full version: 10,000 trajectories, proper statistics
```