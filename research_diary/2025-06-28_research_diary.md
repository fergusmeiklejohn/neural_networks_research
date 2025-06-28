# Research Diary - June 28, 2025

## Today's Focus: Progressive Training Curriculum for Physics Extrapolation

### What We're Trying to Prove/Disprove

**Hypothesis**: Neural networks can learn to extrapolate beyond their training distribution in physics simulations by progressively introducing domain knowledge through a structured curriculum, rather than learning everything at once.

**Specific Claims to Test**:
1. **Physics constraints improve extrapolation**: Models trained with physics-informed losses (conservation of energy/momentum) will generalize better to unseen parameter ranges than purely data-driven models
2. **Progressive curriculum beats joint training**: Gradually introducing complexity (basic patterns → physics constraints → domain randomization → extrapolation focus) will outperform training with all objectives simultaneously
3. **Domain randomization enables transfer**: Training on mixed physics regimes (varying gravity/friction) will help the model learn parameter-invariant representations

**What We're Disproving**:
- The null hypothesis that neural networks fundamentally cannot extrapolate in physics domains
- That more data alone (without structured inductive biases) is sufficient for extrapolation
- That physics-informed approaches only work for interpolation, not true extrapolation

### Why This Experiment Now?

1. **Foundation is Ready**: We've implemented both the Distribution Modification Component and PINN architecture with proper data isolation
2. **Clear Baseline**: Current 0% extrapolation accuracy provides an unambiguous starting point
3. **Methodological Rigor**: We fixed the data leakage issue, ensuring our results will be scientifically valid
4. **Progressive Complexity**: Physics worlds are the simplest domain - if we can't solve extrapolation here, more complex domains (language, visual concepts) will be even harder
5. **Theoretical Grounding**: PINN literature suggests 70-85% extrapolation is achievable with proper training strategies

### Computational Platform Decision

**Recommendation: Start on Mac, then move to Paperspace**

**Phase 1 - Development & Testing (Mac)**:
- Implement the progressive curriculum script
- Test with small subsets (100-1000 samples)
- Debug stage transitions and loss balancing
- Verify metrics and checkpointing
- Expected time: 1-2 hours

**Phase 2 - Full Training (Paperspace)**:
- Run complete 4-stage curriculum on full dataset (9,712 samples)
- 50+ epochs per stage = 200+ total epochs
- GPU acceleration critical for physics loss computations
- P4000 instance ($0.51/hr) should suffice for our model size
- Expected time: 4-8 hours

**Rationale**:
- Mac's M3 Max is excellent for development but limited for long training runs
- TensorFlow backend (already configured) works well on both platforms
- Paperspace provides consistent GPU performance and won't tie up local machine
- Cost-effective: ~$4 for a full training run vs. days on CPU

### Success Criteria

1. **Quantitative**: Achieve 70%+ extrapolation accuracy (up from 0%)
2. **Qualitative**: Model predictions should respect physics constraints even in unseen regimes
3. **Scientific**: Results should be reproducible with fixed seeds and proper train/test isolation
4. **Practical**: Training should complete within 8 hours on P4000 GPU

### Risk Mitigation

- **Risk**: Curriculum might be too aggressive → **Mitigation**: Implement adaptive stage transitions based on validation metrics
- **Risk**: Physics losses might dominate → **Mitigation**: Careful loss weighting with gradual ramping
- **Risk**: Overfitting to training distribution → **Mitigation**: Strong regularization and early stopping per stage

### Next Steps

1. Implement `train_progressive_curriculum.py` with 4-stage pipeline
2. Run quick validation on Mac with subset
3. Transfer to Paperspace for full training
4. Document results and iterate on curriculum design

## End of Entry