# Distribution Invention Project - Master Documentation Index

## 📚 Quick Navigation Guide

### Getting Started
- **Project Overview**: `CLAUDE.md` - Start here for project goals and current state
- **Research Plan**: `distribution_invention_research_plan.md` - Complete technical approach
- **Setup Guide**: `setup_distribution_invention.md` - Development environment setup
- **What to Do Next**: `NEXT_STEPS.md` - Clear action items for implementation

### Current Status & Progress
- **Feedback Integration**: `FEEDBACK_INTEGRATION.md` - 90% complete, tracks all reviewer feedback
- **Research Diary**: `research_diary/` - Daily progress and decisions
  - Latest: `2025-07-12_research_diary.md` - Literature review transformation

### 🔬 Experiments

#### Experiment 01: Physics Worlds
- **Plan**: `experiments/01_physics_worlds/EXPERIMENT_PLAN.md`
- **Status**: Phase 1 ✅ Complete, Phase 2 🚧 In Progress (data isolation fixed)
- **Key Results**: 83.51% extrapolation accuracy achieved
- **Critical Fix**: `DATA_ISOLATION_FIX_PLAN.md` - Addresses data leakage

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

---

*This index is the single source of truth for finding project documentation. Update it when adding major new documents.*

Last Updated: 2025-07-12