# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

This is a neural networks research project focused on developing models that can
**invent new distributions** rather than merely interpolate within their
training distribution. The goal is to create neural networks that can build
"pocket realities" with modified rules to explore novel ideas - mimicking human
creativity and extrapolation capabilities.

### Research Objectives

- **Distribution Construction**: Build models that create coherent new
  probability distributions with selectively modified constraints
- **Controlled Extrapolation**: Enable principled exploration of the "adjacent
  possible" - ideas just outside training boundaries
- **Rule Modification**: Develop architectures that can identify, modify, and
  consistently apply rule changes
- **Insight Transfer**: Create mechanisms for mapping insights from invented
  distributions back to base distributions

# Most important thing to remember is that we are doing research. We are not trying to get the code to run or finish a task to complete a Jira ticket. We are trying to discover new things. This means that often we must persevere and figure out how to do difficult things. Cutting corners is not a good idea because it will not lead us to truth.

## Second most important thing to remember is that we need to consult the documentation when coding new and difficult things and we can fetch it from Context7 MCP.

## Development Environment

### Python Environment Setup

The project uses a Conda environment named `dist-invention` with Python 3.11:

```bash
# Activate environment
conda activate dist-invention

# Key dependencies
pip install keras>=3.0 torch jax[metal] transformers wandb
```
Sometimes Claude cannot run scripts and sees errors like that a package isn't available in the python environment. Often this is an environment error and Claude should ask for help running files in these cases rather than change our strategy or scripts to get round the problem.

### Backend Configuration

The project uses **Keras 3** with multi-backend support:

- **Primary Backend**: JAX with Metal acceleration (Mac)
- **Alternative Backends**: PyTorch, TensorFlow
- Configure via `~/.keras/keras.json` or `KERAS_BACKEND` environment variable

### Development Tools

- **IDE**: VS Code with Python Interactive files (`.py` with `# %%` cells)
- **Notebooks**: Jupyter for final presentations/sharing
- **Experiment Tracking**: Weights & Biases (`wandb`)
- **Code Quality**: black, flake8, isort, pytest

## Project Structure

### Current State

The project is in **active implementation** with two experiments completed:

**Experiment 01: Physics Worlds** ✅
- Progressive curriculum achieved **83.51% extrapolation accuracy**
- Proved neural networks can extrapolate with proper inductive biases
- 4-stage curriculum successfully implemented

**Experiment 02: Compositional Language** 🚧
- Successfully trained minimal LSTM model (267K params)
- Training loss: 0.4026 → 0.038 (excellent convergence)
- Progressive curriculum completed all 4 stages
- Final metrics lost due to save script issues (lesson learned!)
- Ready to retry with improved infrastructure

### Architecture (partially implemented)

```
experiments/                    # 6 major experiments
├── 01_physics_worlds/         # Physics simulation with rule modifications
├── 02_compositional_language/ # Language tasks with compositional rules
├── 03_visual_concepts/        # Visual concept blending
├── 04_abstract_reasoning/     # ARC-like puzzle solving
├── 05_mathematical_extension/ # Mathematical concept extension
└── 06_multimodal_transfer/    # Cross-modal rule transfer

models/
├── core/
│   ├── physics_rule_extractor.py  # ✅ Transformer-based rule extraction
│   ├── distribution_generator.py   # Main distribution invention model
│   ├── consistency_checker.py     # Ensures rule consistency
│   └── insight_extractor.py       # Maps insights back to base
├── physics_informed_components.py  # ✅ HamiltonianNN, attention layers
├── collision_models.py            # ✅ Soft collision handling
├── physics_losses.py              # ✅ Conservation losses, ReLoBRaLo
├── physics_informed_transformer.py # ✅ Hybrid PINN-Transformer
└── utils/                         # Shared utilities

configs/
└── experiment_configs.yaml        # Experiment hyperparameters

data/
├── raw/                           # Original datasets
├── processed/                     # Preprocessed data
└── generated/                     # Generated distributions

scripts/
├── train_paperspace.py           # Training scripts for cloud
└── evaluate_distribution.py      # Evaluation utilities
```

## Core Architecture Components

### 1. DistributionInventor Model

```python
class DistributionInventor(keras.Model):
    def __init__(self):
        self.rule_extractor = CausalRuleExtractor()
        self.modifier = SelectiveRuleModifier()
        self.generator = DistributionGenerator()
        self.consistency_checker = ConsistencyNetwork()
        self.insight_mapper = InsightExtractor()
```

### 2. Key Components

- **Causal Disentanglement Module**: Separates causal mechanisms from parameters
- **Distribution Generator Network**: Creates new coherent distributions from
  base + modification requests
- **Consistency Enforcer**: Ensures non-modified rules remain intact using
  energy-based formulation
- **Insight Extractor**: Maps patterns from generated distributions back to base
  distribution

## Development Commands

### Setup Commands

```bash
# Initial setup (run once)
source scripts/setup_env.sh              # Load environment variables
pip install -e .                         # Install project in development mode

# Development environment verification
python test_setup.py                     # Test installation
python -c "import keras; print(keras.backend.backend())"  # Check backend
```

### Running Experiments

```bash
# Physics Worlds Training (implemented)
cd experiments/01_physics_worlds
python train_pinn_extractor.py      # Full PINN training (run separately)
python train_modifier_numpy.py      # Distribution modifier training

# IMPORTANT: Always train and evaluate baselines for comparison
python train_baselines.py           # Train all 4 baseline models
python run_unified_evaluation.py    # Run comprehensive evaluation

# Evaluation commands
python evaluate_pinn_performance.py --model_path outputs/checkpoints/pinn_final.keras
python evaluate_modifier.py
```

### Code Quality

```bash
# Formatting and linting
black .                                   # Format code
isort .                                   # Sort imports
flake8 --max-line-length=88 .            # Lint code

# Testing
pytest tests/ -v                         # Run tests
pytest tests/test_distribution_generator.py  # Run specific test
```

### Jupyter/Interactive Development

```bash
# Launch Jupyter
jupyter lab                              # Full Jupyter environment
jupyter notebook                         # Classic notebook interface

# VS Code Interactive Python
# Open .py files with # %% cells and run with Shift+Enter
```

## Experiment Workflow

### 3-Phase Training Strategy

1. **Phase 1: Learn Base Distributions**
   - Train on standard datasets to learn rule extraction
   - Use reconstruction tasks to ensure rule completeness
   - **NEW**: Train all 4 baseline models on same data

2. **Phase 2: Controlled Modification**
   - Synthetic tasks with known rule modifications
   - Train to maintain consistency while changing specific rules
   - **NEW**: Test baselines on modification tasks

3. **Phase 3: Creative Generation**
   - Open-ended modification requests
   - Reward novel-but-coherent outputs
   - Use human feedback for quality assessment
   - **NEW**: Run unified evaluation comparing all models

### Evaluation Protocol (MANDATORY)

For EVERY experiment:
1. **Train Baselines**: Use `models/baseline_models.py` to train all 4 baselines
2. **Representation Analysis**: Use `RepresentationSpaceAnalyzer` to verify true OOD
3. **Modification Testing**: Create modification suite for experiment domain
4. **Unified Evaluation**: Run `UnifiedEvaluator` for comprehensive comparison
5. **Report Generation**: Include baseline comparison table in all results

This ensures:
- Fair comparison with strong baselines
- Verification of true extrapolation (not just interpolation)
- Systematic evaluation of modification capabilities
- Clear demonstration of our approach's advantages

### Experiment Progression

1. **Physics Worlds**: 2D ball dynamics with gravity/friction modifications
2. **Compositional Language**: SCAN dataset with rule modifications
3. **Visual Concepts**: ImageNet with object attribute blending
4. **Abstract Reasoning**: ARC-like puzzles requiring novel rule combinations
5. **Mathematical Extension**: Non-commutative multiplication, 4D geometry
6. **Multi-Modal Transfer**: Cross-modal rule application

## Success Metrics

**IMPORTANT**: All experiments must compare against 4 baselines using the unified evaluation framework:

### Primary Metrics (via unified_evaluation.py):
- **Interpolation Accuracy**: Performance within learned representation space (>90%)
- **Near-Extrapolation Accuracy**: Just outside training manifold (>75%)
- **Far-Extrapolation Accuracy**: True novel regimes (>70%)
- **Modification Success Rate**: Ability to adapt to rule changes (>baseline+20%)
- **Consistency Score**: Internal coherence of modified distributions

### Baseline Comparisons (always report):
1. **ERM + Augmentation**: Standard deep learning baseline
2. **GFlowNet**: Exploration-based baseline
3. **Graph Extrapolation**: OOD-focused baseline
4. **MAML**: Adaptation-focused baseline

### Additional Metrics:
- **Novel Coherence**: Models generate novel but coherent outputs
- **Graceful Degradation**: Performance degrades gracefully outside training
- **Human Evaluation**: Outputs rated as "surprisingly sensible" and useful

## Compute Requirements

### Development Phases

- **Local (Mac)**: Initial prototyping, small-scale tests
- **Colab**: Quick experiments, proof of concepts
- **Paperspace**: Main development and training
  - Phase 1: P4000 ($0.51/hr) for models <100M parameters
  - Phase 2: A4000 ($0.76/hr) for models up to 1B parameters

## Key Files to Reference

**IMPORTANT**: For a complete guide to all documentation, see `DOCUMENTATION_INDEX.md` - this is the master index of all project documentation and should be your first stop when looking for information.

### Critical Documents:
- `DOCUMENTATION_INDEX.md`: Master documentation index - START HERE to find anything
- `distribution_invention_research_plan.md`: Complete research plan and
  technical approach
- `setup_distribution_invention.md`: Detailed development environment setup
- `PAPERSPACE_TRAINING_GUIDE.md`: Essential guide for cloud training - READ BEFORE
  any Paperspace runs!
- `FEEDBACK_INTEGRATION.md`: Tracks integration of reviewer feedback (90% complete)
- `experiments/02_compositional_language/train_template.py`: Reusable training
  template with all safety features

### Key Insights Documents:
- `COMPLETE_LITERATURE_INSIGHTS.md`: Synthesis of 15+ papers from literature review
- `CRITICAL_OOD_INSIGHTS.md`: Critical distinction between interpolation vs extrapolation
- `ARC_AGI_INSIGHTS.md`: Insights from ARC-AGI showing 55.5% SOTA performance
- `models/causal_rule_extractor_design.md`: CGNN-based architecture with MMD loss

### Progress Tracking:
- `research_diary/`: Daily progress entries
- Latest experiment status in respective `EXPERIMENT_PLAN.md` files

## Environment Variables

Set these in `.env` file:

```bash
KERAS_BACKEND=jax                        # Primary backend
WANDB_PROJECT=distribution-invention     # Experiment tracking
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8      # JAX memory settings
DATA_DIR=./data                          # Data directory
OUTPUT_DIR=./outputs                     # Output directory
```

## Next Implementation Steps

1. **Setup Development Environment**: Execute setup guide to create directory
   structure
2. **Implement Core Architecture**: Create `DistributionInventor` base classes
3. **Physics Experiment**: Start with simplest experiment (2D physics worlds)
4. **Evaluation Framework**: Build consistent evaluation metrics across
   experiments
5. **Scale Gradually**: Move from local prototyping to cloud training

## Notes

- **Current State**: Planning phase complete, ready for implementation
- **Multi-Backend**: Keras 3 provides flexibility across JAX/PyTorch/TensorFlow
- **Interactive Development**: Use VS Code with Python Interactive files for
  rapid prototyping
- **Experiment Tracking**: All experiments should log to Weights & Biases
- **Timeline**: 4-month research plan with systematic experiment progression

## Notes for execution

We always keep an EXPERIMENT_PLAN.md file in each Experiment folder. We keep it
up to date with the current state of the experiment. And use it to guide the
execution of the experiment.

### Planning Documentation

When planning next steps for experiments, we create timestamped plan documents in
a `claude-plans` folder within each experiment directory. These plans should:
- Include the date and time in the filename (e.g., `2025-06-27_physics_worlds_next_steps.md`)
- Provide context about the current state
- List specific, actionable steps in priority order
- Include success metrics and risk mitigation strategies
- Emphasize scientific validity and proper data isolation

### Training Runs

For efficient development, we separate testing from full training:

**Testing (Claude runs)**:
- Small data subsets (100-1000 samples)
- 1-2 epochs per stage
- Verify implementation correctness
- Quick iteration cycles

**Full Training (User runs separately)**:
- Complete datasets
- 50+ epochs per stage
- Enable wandb logging
- Use GPU/TPU acceleration
- May take hours/days to complete

### Cloud Training Best Practices

**CRITICAL**: Before running any training on Paperspace or other cloud platforms, **ALWAYS consult `PAPERSPACE_TRAINING_GUIDE.md`** in the project root. This guide contains hard-won lessons from actual training runs and will save you from losing valuable GPU hours and results.

**Why this guide is essential**:
- Prevents loss of training results due to instance auto-shutdown
- Ensures proper GPU memory configuration
- Provides tested solutions to common errors
- Includes emergency recovery procedures

**When to reference it**:
1. Before starting any cloud training run
2. When encountering GPU memory or optimizer errors
3. When planning post-training data retrieval
4. For debugging common Paperspace-specific issues

**Key Principles**:

1. **Self-Contained Scripts**: Create scripts like `paperspace_generate_and_train.py` that:
   - Generate all required data on the cloud machine
   - Run the complete training pipeline
   - Handle path differences between environments (/notebooks vs /workspace)
   - Include fallback options for testing

2. **Avoid Large Data in Git**: Since processed data files are gitignored:
   - Always include data generation as part of the cloud pipeline
   - This ensures reproducibility and avoids transfer bottlenecks
   - Document the data generation process clearly

3. **Path Flexibility**: Cloud environments vary, so:
   - Auto-detect paths (e.g., `/notebooks` on Paperspace Gradient)
   - Use relative paths where possible
   - Test for multiple common locations

4. **Example Pattern**:
   ```python
   # Auto-detect base path
   if os.path.exists('/notebooks/neural_networks_research'):
       base_path = '/notebooks/neural_networks_research'
   elif os.path.exists('/workspace/neural_networks_research'):
       base_path = '/workspace/neural_networks_research'
   else:
       base_path = os.path.abspath('../..')
   ```

5. **ALWAYS Save to Persistent Storage**:
   ```python
   # Save during training, not just after!
   if os.path.exists('/storage'):
       storage_path = f'/storage/experiment_{timestamp}/checkpoint_{epoch}.h5'
       model.save_weights(storage_path)
   ```

This approach ensures experiments are fully reproducible and can be run with a single command on any cloud platform.

### Research Diary

We maintain a research diary in `research_diary/` with daily entries documenting:
- Goals and objectives for the day
- Key decisions made and rationale
- What worked and what didn't
- Challenges encountered and solutions
- Results and metrics
- Next steps

**Process**:
1. Make brief notes throughout the day as work progresses
2. Reference these notes when writing the daily summary
3. Use format: `YYYY-MM-DD_research_diary.md`
4. Keep entries concise but comprehensive
5. Include specific code changes, test results, and insights
