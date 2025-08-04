# Baseline Models Comparison on Physics Extrapolation

Generated: 2025-07-13 07:16:22

## Task Description

- **In-distribution**: Earth (9.8 m/s²) and Mars (3.7 m/s²) gravity
- **Near-OOD**: Moon gravity (1.6 m/s²)
- **Far-OOD**: Jupiter gravity (24.8 m/s²)

## Results Summary

| Model | Overall MSE | In-Dist MSE | Near-OOD MSE | Far-OOD MSE | Training Time |
|-------|-------------|-------------|--------------|-------------|---------------|
| ERM+Aug | 0.4313 | 0.0910 | 0.0745 | 1.1284 | 5.5s |
| GFlowNet | 0.3120 | 0.0253 | 0.0608 | 0.8500 | 2.9s |
| GraphExtrap | 0.3166 | 0.0600 | 0.1236 | 0.7663 | 3.2s |
| MAML | 0.3054 | 0.0251 | 0.0684 | 0.8228 | 4.1s |

## Key Insights

- **Best overall performance**: MAML
- **Best far-OOD extrapolation**: GraphExtrap (MSE: 0.7663)

### Extrapolation Degradation

- ERM+Aug: 1140.7% increase from in-dist to far-OOD
- GFlowNet: 3265.6% increase from in-dist to far-OOD
- GraphExtrap: 1177.7% increase from in-dist to far-OOD
- MAML: 3179.8% increase from in-dist to far-OOD
