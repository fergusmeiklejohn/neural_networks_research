# PINN Lessons Learned: From Catastrophic Failure to Success

## Overview

Our first PINN attempt failed catastrophically (880.879 MSE vs 0.766 baseline). This document captures the lessons learned and how we fixed them.

## Major Lessons

### 1. Feature Representation Matters More Than Physics Losses

**Failed Approach:**
- Raw Cartesian coordinates (x, y, vx, vy)
- Expected model to learn coordinate transformations
- Physics losses on top of poor features

**Successful Approach:**
- Physics-aware features (r, θ, vr, vθ, L)
- Natural representation for central forces
- Conservation quantities as features

**Key Insight:** Good features encode more physics than complex losses.

### 2. Loss Balance is Critical

**Failed Approach:**
```python
total_loss = mse_loss + 0.01 * physics_loss
# Result: MSE ~1000, Physics ~10
# Physics effectively ignored!
```

**Successful Approach:**
```python
total_loss = 1.0 * mse_loss + 100.0 * physics_loss
# Physics losses dominate optimization
```

**Key Insight:** Physics losses must be 100-1000x larger than reconstruction losses.

### 3. Architecture Should Encode Physics

**Failed Approach:**
- Generic deep network (6 layers, 1.9M params)
- No physics structure
- Too many degrees of freedom

**Successful Approach:**
- Minimal network (2 layers, <100K params)
- F=ma as base + learned corrections
- Symplectic integrator for trajectories

**Key Insight:** Start with physics equations, add neural corrections.

### 4. Predict Physics Quantities, Not Raw States

**Failed Approach:**
- Predict next positions directly
- No physical constraints on predictions
- Accumulating errors

**Successful Approach:**
- Predict accelerations
- Integrate using physics
- Conserve energy/momentum by construction

**Key Insight:** Neural networks should predict forces/accelerations, not positions.

### 5. Progressive Training Helps (But Not Enough)

**Failed Approach:**
- Progressive curriculum reduced MSE by 28%
- But 1220 → 880 MSE still catastrophic

**Successful Approach:**
- Start with simple physics (1D motion)
- Gradually add complexity
- Validate physics at each stage

**Key Insight:** Progressive training helps but can't fix fundamental issues.

## Technical Details

### What Doesn't Work

1. **Deep Networks for Physics**
   - Too many parameters
   - Can satisfy MSE while violating physics
   - Hard to interpret

2. **Multiple Competing Losses**
   - Hard to balance
   - Optimization gets stuck
   - Physics constraints ignored

3. **Raw State Prediction**
   - No structure
   - Errors compound
   - Unphysical trajectories

4. **Complex Physics Losses**
   - Energy + momentum + smoothness + ...
   - Each needs different scaling
   - Gradients conflict

### What Works

1. **Physics-Inspired Features**
   ```python
   features = [
       r,           # radius (distance from origin)
       theta,       # angle
       v_r,         # radial velocity
       v_theta,     # angular velocity
       L,           # angular momentum (r × v)
   ]
   ```

2. **Minimal Corrections to Known Physics**
   ```python
   acceleration = F_physics + small_neural_correction
   ```

3. **Single Dominant Physics Loss**
   ```python
   loss = mse + 100 * energy_conservation
   ```

4. **Symplectic Integration**
   - Preserves energy
   - Stable over long trajectories
   - Physics-consistent

## Architectural Guidelines

### DO:
- Use coordinate systems natural to the problem
- Encode symmetries (rotation, translation)
- Predict intensive quantities (force/mass, not force)
- Start with known physics equations
- Keep networks small (<100K params)

### DON'T:
- Use generic deep architectures
- Predict extensive quantities directly
- Balance many physics losses
- Ignore domain knowledge
- Over-parameterize

## Loss Function Design

### Principles:
1. **One primary physics constraint** (e.g., energy)
2. **Weight it 100-1000x more** than reconstruction
3. **Use gradient normalization** if needed
4. **Monitor individual loss components**
5. **Stop if physics loss dominates** without improving

### Example:
```python
def loss_fn(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    energy_violation = compute_energy_loss(y_pred)

    # Heavy physics weighting
    total_loss = 1.0 * mse + 100.0 * energy_violation

    # Log components separately
    log('mse', mse)
    log('energy', energy_violation)

    return total_loss
```

## Debugging PINNs

### Red Flags:
1. MSE orders of magnitude larger than baselines
2. Physics losses near zero while MSE huge
3. Predicted parameters unrealistic (e.g., gravity = -9.8 for Jupiter)
4. No improvement with more physics losses
5. Deeper networks perform worse

### Diagnostic Steps:
1. **Check feature representation** - Are they physics-aware?
2. **Examine loss balance** - Is physics loss significant?
3. **Validate predictions** - Do they conserve quantities?
4. **Simplify architecture** - Can you remove layers?
5. **Start smaller** - 1D before 2D, one ball before two

## Success Metrics

A successful PINN should:
1. **Beat naive baselines** on extrapolation
2. **Learn correct physics parameters** (±20%)
3. **Conserve quantities** (energy, momentum)
4. **Generate smooth trajectories**
5. **Extrapolate to new conditions**

## Conclusion

The key lesson: **Physics-informed doesn't mean physics-aware**.

Successful PINNs require:
- Physics-aware features
- Proper loss weighting
- Minimal architectures
- Domain-appropriate design

Our catastrophic failure taught us that throwing physics losses at a generic deep network doesn't work. The path forward is thoughtful integration of physics into every aspect of the model.
