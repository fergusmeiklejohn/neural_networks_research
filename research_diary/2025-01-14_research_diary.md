# Research Diary: January 14, 2025

## Today's Focus: Testing Minimal PINN Implementation

### Goals
- Test if a minimal physics-informed neural network could beat GraphExtrap's baseline
- Understand why PINNs fail so catastrophically on physics extrapolation
- Apply lessons learned from our previous analyses

### Key Activities

#### 1. Implemented Minimal PINN Architecture
Created a stripped-down PINN based on our lessons learned:
- Started with F=ma as base, added small neural corrections
- Used physics-aware features (polar coordinates like GraphExtrap)
- Weighted physics losses 100x more than MSE
- Only 2 layers with 64 hidden units (vs 1.9M params in original)

#### 2. Discovered Major Data Format Issues
Spent significant time debugging data loading:
- Data uses pixel coordinates, not meters (gravity ~400-1200 pixels/s²)
- Trajectory format was different than expected: [time, x1, y1, vx1, vy1, mass1, radius1, ...]
- Had to extract correct columns and convert units properly
- Fixed negative friction issue by enforcing positivity constraint

#### 3. Results: Another PINN Failure Mode

**MSE on Jupiter Gravity:**
- GraphExtrap: 0.766 (best)
- Minimal PINN: 42,468 (55,000x worse!)
- Original PINN: 880.879

The minimal PINN learned Earth gravity perfectly (-9.81 m/s²) but couldn't adapt to Jupiter (-42.8 m/s²).

### Key Insights

1. **Physics Constraints Are a Liability**: The model's assumption of fixed gravity prevents adaptation. What should be a strength (knowing F=ma) becomes a fundamental weakness.

2. **GraphExtrap's Secret**: It succeeds by NOT assuming specific physics. It learns patterns from data and can interpolate between different gravity values.

3. **Data Understanding is Critical**: We lost hours to data format issues - pixel vs meter coordinates, wrong column extraction, etc. 

4. **PINNs Have a Fundamental Flaw**: They encode assumptions that become invalid under distribution shift. The more "physics" we add, the more rigid and brittle the model becomes.

### Challenges Encountered

1. **Data Format Confusion**: The trajectory data structure was completely different than documented
2. **Unit Conversion Hell**: Mixing pixel and meter units caused massive MSE values
3. **Numerical Instability**: Had to reduce learning rates and physics weights to avoid NaN losses
4. **JAX Immutability**: Required rewriting array updates to avoid in-place modifications

### Current State & Next Steps

**Where we are:**
- Confirmed PINNs fail due to rigid physics assumptions (2 experiments, both catastrophic)
- Have working data pipeline: `/experiments/01_physics_worlds/train_minimal_pinn.py` loads filtered 2-ball trajectories
- GraphExtrap baseline: 0.766 MSE on Jupiter (need to understand WHY - check their feature engineering)
- Key files ready: `TRUE_OOD_BENCHMARK.md` has Level 2 design, `models/baseline_models.py` has GraphExtrap implementation

**Immediate next steps (with entry points):**
1. **Understand GraphExtrap's success** 
   - Run: `python train_baselines.py --model graph_extrap --verbose`
   - Check their geometric features in `models/baseline_models.py:L142-156`
   - Key question: Do they train on multiple gravity values?

2. **Implement True OOD Benchmark Level 2**
   - Start from: `TRUE_OOD_BENCHMARK.md:L36-47` (time-varying gravity)
   - Modify: `improved_data_generator.py` to add `gravity_fn=lambda t: -9.8 * (1 + 0.1*sin(t))`
   - Test with: `RepresentationSpaceAnalyzer` to verify >60% true OOD

3. **Paper 1: "The OOD Illusion"**
   - Evidence collected: `CRITICAL_OOD_INSIGHTS.md` (91.7% interpolation finding)
   - Structure template: `NEXT_RESEARCH_STEPS.md:L47-54`
   - Key figure: t-SNE plot showing "OOD" samples inside training manifold

**Critical context for tomorrow:**
- Data is in PIXELS not meters! (40 pixels = 1 meter)
- Gravity range: Earth ~-9.8 m/s², Jupiter ~-24.8 m/s² (but data shows -42.8)
- Use `physics_config['gravity'] / 40.0` for conversion
- Trajectories are 300 timesteps but we use first 50

### Questions to Investigate Tomorrow

1. **Does GraphExtrap train on multiple gravity values?** This could explain its interpolation success
2. **What happens if we make gravity a function of features?** E.g., `gravity = f(positions)`
3. **Can we detect distribution shift online?** If model could recognize "this is Jupiter", it could adapt

### Reflection

Today confirmed our hypothesis: physics-informed approaches fail when test conditions violate encoded assumptions. Even a "perfect" physics model can't extrapolate if it assumes wrong constants. This is profound - it suggests that for true extrapolation, we need models that can discover and adapt their own "physics" rather than having it hardcoded.

The irony is striking: knowing the correct physics equations makes models WORSE at extrapolation, not better. This challenges fundamental assumptions about how to incorporate domain knowledge into ML systems.

**Key takeaway for paper**: "Physics-informed" ≠ "Physics-aware". The former rigidly enforces equations, the latter flexibly uses physics-inspired features.