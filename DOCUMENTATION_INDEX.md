# Distribution Invention Project - Master Documentation Index

## 📚 Quick Navigation Guide

### Getting Started
- **Project Overview**: `CLAUDE.md` - Start here for project goals and navigation
- **Code Reliability**: `CODE_RELIABILITY_GUIDE.md` - ⚠️ MUST READ before coding
- **Research Plan**: `distribution_invention_research_plan.md` - Complete technical approach
- **Setup Guide**: `setup_distribution_invention.md` - Development environment setup
- **What to Do Next**: Check experiment `CURRENT_STATUS.md` files

### Current Status & Progress
- **Feedback Integration**: `FEEDBACK_INTEGRATION.md` - 100% complete, all reviewer feedback integrated
- **Research Diary**: `research_diary/` - Daily progress and decisions
  - Latest: `2025-07-16_research_diary.md` - Started scholarly paper on OOD evaluation analysis
- **Papers in Progress**: `papers/ood_evaluation_analysis/` - Technical report on interpolation vs extrapolation

### 🔬 Experiments

#### Experiment 01: Physics Worlds
- **Current Status**: `experiments/01_physics_worlds/CURRENT_STATUS.md` - ⭐ Check this first!
- **Plan**: `experiments/01_physics_worlds/EXPERIMENT_PLAN.md`
- **Major Finding**: Discovered "OOD Illusion" - most OOD is actually interpolation
- **PINN Analysis**: Physics-informed models failed catastrophically (1,150x worse)
- **Key Documents**: `TRUE_OOD_BENCHMARK.md`, `PINN_LESSONS_LEARNED.md`

#### Experiment 02: Compositional Language  
- **Plan**: `experiments/02_compositional_language/EXPERIMENT_PLAN.md`
- **Status**: Ready for retry (infrastructure fixed after lost results)
- **Training Template**: `train_template.py` - Use this for safe cloud training

### 🧠 Key Insights & Learning

#### Literature Reviews
1. **Complete Summary**: `COMPLETE_LITERATURE_INSIGHTS.md` - All 15 papers synthesized
2. **Critical OOD Insight**: `CRITICAL_OOD_INSIGHTS.md` - Interpolation vs extrapolation
3. **ARC-AGI Insights**: `ARC_AGI_INSIGHTS.md` - 55.5% SOTA validates our approach
4. **Quick Summary**: `LITERATURE_REVIEW_SUMMARY.md` - Key takeaways

#### Technical Designs
- **CausalRuleExtractor**: `models/causal_rule_extractor_design.md` - CGNN-based with MMD loss
- **Baseline Models**: `models/baseline_models.py` - 4 baseline implementations
- **Unified Evaluation**: `models/unified_evaluation.py` - Standardized evaluation framework
- **Safety Protocols**: `SAFETY_PROTOCOLS.md` - Comprehensive safety framework

### 📋 Process Documents
- **Experiment Checklist**: `EXPERIMENT_CHECKLIST.md` - Step-by-step guide for running experiments
- **Milestone Timeline**: `MILESTONE_TIMELINE.md` - 24-month plan with GPU estimates

### 🚀 Cloud Training
- **CRITICAL**: `PAPERSPACE_TRAINING_GUIDE.md` - READ BEFORE ANY CLOUD TRAINING
  - Contains hard-won lessons about GPU memory, saving results, recovery procedures
  - Prevents loss of valuable training time and results

### 📊 What We've Learned

#### Major Revelations
1. **Most "OOD" is interpolation** - Must verify in representation space
2. **Hybrid approaches win** - Pure neural/symbolic cap at 40%, hybrid reaches 55.5%
3. **Meta-learning is key** - MLC achieved 99.78% on SCAN with dynamic task generation
4. **MMD loss for distributions** - CGNN provides concrete implementation

#### Updated Approach
- Dynamic physics generation during training (not fixed parameters)
- Three-tier evaluation: interpolation, near-extrapolation, far-extrapolation
- Standard transformer + meta-learning (not complex architectures)
- Test-time adaptation for rule recombination

### 📝 Research History

#### Proposals & Feedback
- **Original Proposal**: `Research_proposals_feedback/RESEARCH_PROPOSAL.md`
- **Reviewer Feedback**: `Research_proposals_feedback/FEEDBACK.md` - Overwhelmingly positive
- **Integration Status**: 90% complete (40/45 items done)

#### Key Milestones
1. **June 27**: Initial physics experiment setup
2. **June 28**: Achieved 83.51% extrapolation (but discovered data leakage)
3. **July 12**: Comprehensive literature review transformed approach
4. **July 14-15**: Discovered OOD illusion through baseline testing
5. **July 16**: Began scholarly documentation of findings

### 🔍 Finding Information

#### By Topic
- **Physics Experiments**: `experiments/01_physics_worlds/`
- **Language Experiments**: `experiments/02_compositional_language/`
- **Safety & Ethics**: `SAFETY_PROTOCOLS.md`
- **Literature Insights**: `*_INSIGHTS.md` files
- **Daily Progress**: `research_diary/YYYY-MM-DD_research_diary.md`

#### By Purpose
- **Implementation**: Check experiment folders and `models/`
- **Theory & Justification**: Literature reviews and research plan
- **Progress Tracking**: `FEEDBACK_INTEGRATION.md` and diary
- **Cloud Setup**: `PAPERSPACE_TRAINING_GUIDE.md`

### ⚠️ Critical Documents

**Must Read Before**:
- Starting development: `CLAUDE.md`
- Cloud training: `PAPERSPACE_TRAINING_GUIDE.md`
- Implementing models: `models/causal_rule_extractor_design.md`
- Running experiments: Respective `EXPERIMENT_PLAN.md`

### 📈 Next Steps
Based on our current 90% feedback integration:
1. Create realistic milestone timeline
2. Implement MMD loss and meta-learning
3. Update baseline models
4. Complete safety documentation

### 🚀 Fast Access - Common Commands & Paths

#### Quick Test Commands
```bash
# Test minimal physics model
cd experiments/01_physics_worlds
python train_minimal_pinn.py --test_mode

# Train baseline model
python train_baselines.py --model graph_extrap --verbose

# Check environment
python -c "import keras; print(keras.backend.backend())"
```

#### Key Paths
- Latest findings: `research_diary/2025-07-16_research_diary.md`
- Physics status: `experiments/01_physics_worlds/CURRENT_STATUS.md`
- Code pitfalls: `CODE_RELIABILITY_GUIDE.md`
- Cloud setup: `PAPERSPACE_TRAINING_GUIDE.md`
- Paper draft: `papers/ood_evaluation_analysis/`

#### Data Format Reminders
- Physics data: 40 pixels = 1 meter
- Gravity: ~400-1200 pixels/s² (divide by 40 for m/s²)
- Trajectory: [time, x1, y1, vx1, vy1, mass1, radius1, x2, y2, vx2, vy2, mass2, radius2]

---

*This index is the single source of truth for finding project documentation. Update it when adding major new documents.*

Last Updated: 2025-07-16