# Experiment Execution Checklist

This checklist ensures all experiments follow our standardized evaluation protocol with baseline comparisons and proper OOD verification.

## Pre-Experiment Setup

- [ ] **Review Key Documents**
  - [ ] Read `DOCUMENTATION_INDEX.md` for navigation
  - [ ] Review `CRITICAL_OOD_INSIGHTS.md` for representation space analysis
  - [ ] Check `COMPLETE_LITERATURE_INSIGHTS.md` for implementation strategies
  - [ ] Consult `PAPERSPACE_TRAINING_GUIDE.md` if using cloud resources

- [ ] **Prepare Baseline Models**
  - [ ] Import `models/baseline_models.py`
  - [ ] Configure all 4 baselines for your experiment:
    - [ ] ERM + Data Augmentation
    - [ ] GFlowNet-guided Search
    - [ ] Graph Extrapolation
    - [ ] MAML
  - [ ] Verify baseline configurations match experiment requirements

## Data Generation Phase

- [ ] **Generate Training Data**
  - [ ] Create base distribution samples
  - [ ] Document parameter ranges and distributions
  - [ ] Generate modification pairs for training

- [ ] **Create Proper Test Splits**
  - [ ] Interpolation test set (within training manifold)
  - [ ] Near-extrapolation test set (boundary cases)
  - [ ] Far-extrapolation test set (novel regimes)
  - [ ] Use `RepresentationSpaceAnalyzer` to verify categories

- [ ] **Prepare Modification Test Suite**
  - [ ] Simple parameter changes
  - [ ] Rule swaps
  - [ ] Combined modifications
  - [ ] Novel rule combinations

## Training Phase

- [ ] **Train Our Model**
  - [ ] Phase 1: Base distribution learning
  - [ ] Phase 2: Controlled modifications
  - [ ] Phase 3: Creative generation
  - [ ] Save checkpoints regularly

- [ ] **Train All Baselines**
  ```python
  from models.baseline_models import ERMWithAugmentation, GFlowNetBaseline, etc.
  
  baselines = {
      'ERM+Aug': ERMWithAugmentation(config),
      'GFlowNet': GFlowNetBaseline(config),
      'GraphExtrap': GraphExtrapolationBaseline(config),
      'MAML': MAMLBaseline(config)
  }
  
  for name, model in baselines.items():
      model.build_model()
      model.train(train_data, val_data)
  ```

## Evaluation Phase

- [ ] **Run Unified Evaluation**
  ```python
  from models.unified_evaluation import UnifiedEvaluator
  
  evaluator = UnifiedEvaluator(experiment_type)
  results = evaluator.evaluate_all_models(
      models={**baselines, 'DistInvention': our_model},
      train_data=train_data,
      test_data=test_data
  )
  ```

- [ ] **Verify Representation Space Analysis**
  - [ ] Check UMAP visualizations
  - [ ] Confirm true OOD categorization
  - [ ] Validate density estimates

- [ ] **Test Modification Capabilities**
  - [ ] Run modification test suite
  - [ ] Compare adaptation times
  - [ ] Measure consistency scores

## Results Documentation

- [ ] **Generate Standard Reports**
  - [ ] Baseline comparison table
  - [ ] Interpolation vs extrapolation breakdown
  - [ ] Modification success rates
  - [ ] Representation space visualizations

- [ ] **Required Metrics to Report**
  - [ ] Interpolation accuracy (all models)
  - [ ] Near-extrapolation accuracy (all models)
  - [ ] Far-extrapolation accuracy (all models)
  - [ ] Modification success rate (all models)
  - [ ] Our advantage over best baseline

- [ ] **Create Visualizations**
  - [ ] Training curves comparison
  - [ ] Representation space plots
  - [ ] Modification examples
  - [ ] Failure case analysis

## Post-Experiment

- [ ] **Update Documentation**
  - [ ] Update experiment's `EXPERIMENT_PLAN.md` with results
  - [ ] Add entry to research diary
  - [ ] Update `FEEDBACK_INTEGRATION.md` if addressing feedback

- [ ] **Save All Artifacts**
  - [ ] Model checkpoints (all models)
  - [ ] Evaluation reports
  - [ ] Visualization figures
  - [ ] Raw result data

- [ ] **Prepare for Next Steps**
  - [ ] Identify failure modes
  - [ ] Plan improvements
  - [ ] Document lessons learned

## Critical Reminders

⚠️ **Always verify true OOD**: Use representation space analysis, not just parameter differences
⚠️ **Include all baselines**: Never report results without baseline comparisons
⚠️ **Save during training**: Don't wait until end to save results
⚠️ **Document modifications**: Keep detailed logs of all modification tests

## Quick Commands Reference

```bash
# Train baselines
python train_baselines.py --experiment physics_worlds

# Run evaluation
python run_unified_evaluation.py --models all --output results/

# Generate report
python generate_report.py --experiment physics_worlds --format latex

# Visualize representations
python visualize_representations.py --model our_model --test_data test.pkl
```

---

*This checklist ensures consistent, rigorous evaluation across all experiments with proper baseline comparisons and OOD verification.*