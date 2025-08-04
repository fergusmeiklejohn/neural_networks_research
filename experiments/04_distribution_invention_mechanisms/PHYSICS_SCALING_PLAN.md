# Physics Domain Scaling Plan: Two-Stage Compiler for Physical Laws

## Executive Summary

We've successfully demonstrated that the Two-Stage Compiler achieves 79% accuracy on variable binding tasks by using explicit, discrete mechanisms instead of implicit neural representations. Now we scale this approach to physics, where "X means jump" becomes "gravity = 5 m/s²".

**Key Insight**: Physics law modification is variable binding at a higher level of abstraction.

## Core Parallels

### Variable Binding → Physics Laws

| Variable Binding | Physics Domain |
|-----------------|----------------|
| "X means jump" | "gravity = 9.8 m/s²" |
| Variable scope | Physical context |
| Temporal rebinding | Time-varying physics |
| Compositional operators | Combined forces |
| THEN sequencing | Causal chains |

### Why Two-Stage Works

1. **Stage 1 (Discrete)**: Extract modifiable physics parameters
   - Just as we extract "X → jump" bindings
   - Identify gravity, friction, elasticity values
   - Track which parameters apply when

2. **Stage 2 (Neural)**: Execute physics with modified parameters
   - Similar to executing "do X" with binding context
   - Run physics simulation with extracted parameters
   - Fully differentiable for learning dynamics

## Implementation Architecture

### Stage 1: Physics Rule Extractor

```python
@dataclass
class PhysicsParameter:
    name: str  # "gravity", "friction", etc.
    value: float
    unit: str
    context_start: int  # When this parameter applies
    context_end: Optional[int]

@dataclass
class PhysicsModification:
    parameter: str
    operation: str  # "set", "increase", "decrease"
    value: float
    scope: str  # "global", "object", "temporal"

class PhysicsRuleExtractor:
    """Extract physics parameters and modifications from commands."""

    def extract(self, command: str, trajectory: np.ndarray) -> Tuple[List[PhysicsParameter], List[PhysicsModification]]:
        # Parse natural language commands
        # "Set gravity to 5 m/s²" → PhysicsModification("gravity", "set", 5.0, "global")
        # "Increase friction by 20%" → PhysicsModification("friction", "increase", 0.2, "global")

        # Extract current physics from trajectory (if needed)
        # Use physics-informed analysis to identify parameters

        return parameters, modifications
```

### Stage 2: Neural Physics Executor

```python
class NeuralPhysicsExecutor(nn.Module):
    """Execute physics simulation with explicit parameter context."""

    def __init__(self, state_dim: int, param_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Physics state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Parameter context encoder
        self.param_encoder = nn.Sequential(
            nn.Linear(param_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Cross-attention: state attends to active parameters
        self.cross_attention = nn.MultiHeadAttention(
            hidden_dim, num_heads=8
        )

        # Physics predictor
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state: Tensor, parameters: Dict[str, float], timestep: int) -> Tensor:
        # Encode current state
        state_encoded = self.state_encoder(state)

        # Encode active parameters for this timestep
        param_vector = self.encode_parameters(parameters, timestep)
        param_encoded = self.param_encoder(param_vector)

        # Cross-attention: how do parameters affect state?
        attended, _ = self.cross_attention(
            state_encoded.unsqueeze(0),
            param_encoded.unsqueeze(0),
            param_encoded.unsqueeze(0)
        )

        # Predict next state
        next_state = self.predictor(attended.squeeze(0))
        return next_state
```

### Full Two-Stage Physics Model

```python
class TwoStagePhysicsModel:
    """Complete model for physics distribution invention."""

    def __init__(self):
        self.extractor = PhysicsRuleExtractor()
        self.executor = NeuralPhysicsExecutor(
            state_dim=4,  # x, y, vx, vy
            param_dim=10  # encoded physics parameters
        )

    def forward(self, command: str, initial_state: Tensor, timesteps: int) -> Tensor:
        # Stage 1: Extract physics modifications
        parameters, modifications = self.extractor.extract(command, initial_state)

        # Apply modifications to create new physics context
        physics_context = self.apply_modifications(parameters, modifications)

        # Stage 2: Execute physics with modified parameters
        trajectory = [initial_state]
        state = initial_state

        for t in range(timesteps):
            # Get active parameters for this timestep
            active_params = self.get_active_parameters(physics_context, t)

            # Neural physics step with explicit parameter context
            state = self.executor(state, active_params, t)
            trajectory.append(state)

        return torch.stack(trajectory)
```

## Key Design Decisions

### 1. Explicit Parameter Representation
- **Why**: Just as "X → jump" must be explicit, "gravity → 9.8" must be explicit
- **How**: Structured parameter dictionary, not hidden embeddings
- **Benefit**: Perfect parameter extraction, interpretable modifications

### 2. Temporal Parameter Tracking
- **Why**: Physics can change over time (like variable rebinding)
- **How**: Each parameter has context_start/context_end
- **Benefit**: Handles "set gravity to 5 for 2 seconds then 10"

### 3. Cross-Attention Mechanism
- **Why**: State evolution depends on active physics parameters
- **How**: Neural executor attends to parameter context
- **Benefit**: Learns parameter-state interactions

### 4. Discrete Modifications
- **Why**: "Set gravity to 5" is discrete, not continuous
- **How**: Explicit modification operations
- **Benefit**: Precise control, no ambiguity

## Training Strategy

### Phase 1: Parameter Extraction (1 week)
1. Create synthetic dataset of physics commands
2. Train extractor to identify parameters from descriptions
3. Validate on held-out modification types
4. Target: 95%+ extraction accuracy

### Phase 2: Neural Dynamics (1 week)
1. Use ground-truth physics parameters
2. Train executor to predict trajectories
3. Include energy conservation loss
4. Target: <0.1 MSE on standard physics

### Phase 3: End-to-End Training (1 week)
1. Joint training with modification commands
2. Balance extraction and execution losses
3. Test on novel parameter combinations
4. Target: 80%+ on modified physics

### Phase 4: True OOD Testing (1 week)
1. Implement TRUE_OOD_BENCHMARK.md tests
2. Time-varying gravity
3. New force types
4. Causal reversals
5. Target: 50%+ on Level 2, 30%+ on Level 3

## Datasets

### Training Data
```python
training_modifications = [
    # Simple parameter changes
    ("Set gravity to 5 m/s²", {"gravity": 5.0}),
    ("Double the friction", {"friction": "*2"}),
    ("Remove air resistance", {"damping": 0.0}),

    # Combinations
    ("Underwater physics", {"gravity": 7.0, "damping": 0.5}),
    ("Space physics", {"gravity": 0.1, "friction": 0.0}),

    # Temporal changes
    ("Increase gravity over time", {"gravity": "9.8 + 0.1*t"}),
    ("Pulse gravity every second", {"gravity": "9.8 * (1 + 0.5*sin(2π*t))"}),
]
```

### Evaluation Data
```python
eval_modifications = [
    # Test interpolation
    ("Set gravity to 7.5 m/s²", {"gravity": 7.5}),

    # Test extrapolation
    ("Reverse gravity", {"gravity": -9.8}),
    ("Extreme friction", {"friction": 10.0}),

    # Test novel combinations
    ("Bouncy vacuum", {"elasticity": 1.0, "damping": 0.0}),

    # Test true OOD
    ("Oscillating gravity", {"gravity": "9.8*sin(t)"}),
    ("Add magnetic field", {"B_field": 0.1}),
]
```

## Evaluation Metrics

### 1. Parameter Extraction Accuracy
- Correct parameter identification: 95%+
- Correct modification parsing: 90%+
- Temporal scope accuracy: 85%+

### 2. Physics Execution Quality
- Trajectory MSE: <0.1 for standard physics
- Energy conservation: <10% violation
- Smoothness: No discontinuities

### 3. Modification Success
- Directional correctness: 90%+ (gravity increases when commanded)
- Magnitude accuracy: ±20% of target
- Combination coherence: All parameters applied correctly

### 4. True OOD Performance
- Level 1 (parameter shift): 80%+
- Level 2 (functional change): 50%+
- Level 3 (new physics): 30%+
- Level 4 (causal reversal): 10%+

## Implementation Timeline

### Week 1: Basic Infrastructure
- [ ] Port PhysicsRuleExtractor from variable binding extractor
- [ ] Implement parameter parsing for physics commands
- [ ] Create synthetic physics modification dataset
- [ ] Set up evaluation pipeline

### Week 2: Neural Executor
- [ ] Implement NeuralPhysicsExecutor with cross-attention
- [ ] Train on ground-truth parameters
- [ ] Validate energy conservation
- [ ] Test on standard physics

### Week 3: Integration
- [ ] Connect extractor and executor
- [ ] Implement modification application logic
- [ ] End-to-end training pipeline
- [ ] Test on modification commands

### Week 4: Evaluation
- [ ] Run TRUE_OOD_BENCHMARK tests
- [ ] Ablation studies (remove explicit parameters, etc.)
- [ ] Compare with baseline PINNs
- [ ] Document findings

## Expected Outcomes

### Success Indicators
1. **Explicit beats implicit**: 80%+ with explicit parameters vs 40% without
2. **Temporal tracking works**: Handles time-varying physics correctly
3. **True extrapolation**: Achieves 50%+ on functional form changes
4. **Interpretability**: Can inspect extracted parameters

### Key Insights Expected
1. Physics laws are high-level variable bindings
2. Discrete extraction enables better generalization
3. Explicit state tracking prevents parameter confusion
4. Cross-attention learns parameter-dynamics relationships

## Risk Mitigation

### Risk 1: Physics extraction harder than variable binding
- **Mitigation**: Start with synthetic commands, gradually increase complexity
- **Fallback**: Use ground-truth parameters initially

### Risk 2: Neural dynamics unstable
- **Mitigation**: Include physics-informed losses (energy, momentum)
- **Fallback**: Use simpler integrator architectures

### Risk 3: True OOD too difficult
- **Mitigation**: Create intermediate difficulty levels
- **Fallback**: Focus on near-distribution success first

## Conclusion

The Two-Stage Compiler's success on variable binding provides a clear path for physics domain scaling. By maintaining explicit parameter representation and discrete modifications, we expect to achieve true distribution invention for physical laws. This approach should succeed where implicit PINNs failed, demonstrating that explicit mechanisms are necessary for genuine extrapolation.

**Next Step**: Begin implementing PhysicsRuleExtractor using lessons from variable binding.
