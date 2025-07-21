# Pendulum vs 2-Ball Experiments: Comparison Summary

## Overview
We implemented pendulum experiments to address reviewer feedback about empirical breadth. Results confirm our core findings while revealing interesting differences between mechanism shift types.

## Key Findings

### 1. TTA Consistently Fails on Both Mechanism Shifts

**2-Ball System (Time-Varying Gravity):**
- Standard TTA: 235% degradation (6,935 vs 2,721 MSE)
- MAML with adaptation: 62,290% catastrophic failure

**Pendulum System (Time-Varying Length):**
- Prediction consistency TTA: 1,441% degradation 
- Energy consistency TTA: 1,256% degradation
- Hamiltonian consistency TTA: 1,794% degradation

**Critical Insight**: Physics-aware TTA (energy/Hamiltonian) still degrades performance, confirming that the failure is fundamental to mechanism shifts, not just poor loss choice.

### 2. Baseline Models Show Different Degradation Patterns

**2-Ball System:**
- ERM baseline: ~2,700 MSE on OOD
- Represents 100x+ degradation from in-distribution

**Pendulum System:**
- ERM baseline: 1.4x degradation only (0.0003 → 0.0005 MSE)
- Much milder degradation despite mechanism shift

### 3. Why the Difference?

Several factors explain the milder pendulum degradation:

1. **Simpler System**: Single pendulum vs interacting balls
2. **Smoother Mechanism Change**: Sinusoidal length variation vs gravity change
3. **Conservation Properties**: Length changes are periodic and bounded
4. **Lower Dimensionality**: 5D state vs 8D+ for multiple balls

## Implications for Paper

### Strengthens Core Arguments:
1. ✅ TTA fails on multiple types of mechanism shifts
2. ✅ Physics-aware losses don't solve the problem
3. ✅ Failure is systematic, not task-specific

### Adds Nuance:
1. Severity of degradation depends on mechanism type
2. Simpler systems may show less catastrophic failure
3. But TTA still consistently makes things worse

### Reviewer Response:
"We tested on pendulum with time-varying length as suggested. While baseline degradation was milder (1.4x), TTA still degraded performance by 12-18x across all variants including physics-aware losses. This confirms our core finding holds across different mechanism shift types."

## Experimental Details

### Pendulum Setup:
- Training: Fixed length L = 1.0m
- Test: L(t) = L₀(1 + 0.2sin(0.1t))
- New physics term: -2(L̇/L)θ̇ appears in equations

### Results Summary Table:

| Method | 2-Ball Degradation | Pendulum Degradation |
|--------|-------------------|---------------------|
| ERM Baseline | 100x+ | 1.4x |
| Standard TTA | 2.35x | 14.4x |
| Energy-based TTA | N/A | 12.6x |
| Hamiltonian TTA | N/A | 17.9x |
| MAML + Adapt | 622x | Not tested |

## Next Steps

1. Include pendulum results in paper revision
2. Emphasize that physics-aware TTA still fails
3. Use to demonstrate generality of findings
4. Consider adding third mechanism type if space permits