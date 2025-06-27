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

## Development Environment

### Python Environment Setup

The project uses a Conda environment named `dist-invention` with Python 3.11:

```bash
# Activate environment
conda activate dist-invention

# Key dependencies
pip install keras>=3.0 torch jax[metal] transformers wandb
```

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

The project is in **active implementation** with Physics Worlds experiment underway:
- Phase 1 (Data Generation) ✅ Complete - 9,712 training samples
- Phase 2 (Model Training) 🚧 In Progress
  - Distribution Modification Component ✅ Complete
  - Physics-Informed Neural Network (PINN) ✅ Complete
  - Progressive Training Curriculum ⏳ Ready to start

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

2. **Phase 2: Controlled Modification**
   - Synthetic tasks with known rule modifications
   - Train to maintain consistency while changing specific rules

3. **Phase 3: Creative Generation**
   - Open-ended modification requests
   - Reward novel-but-coherent outputs
   - Use human feedback for quality assessment

### Experiment Progression

1. **Physics Worlds**: 2D ball dynamics with gravity/friction modifications
2. **Compositional Language**: SCAN dataset with rule modifications
3. **Visual Concepts**: ImageNet with object attribute blending
4. **Abstract Reasoning**: ARC-like puzzles requiring novel rule combinations
5. **Mathematical Extension**: Non-commutative multiplication, 4D geometry
6. **Multi-Modal Transfer**: Cross-modal rule application

## Success Metrics

- **Novel Coherence**: Models generate novel but coherent outputs when modifying
  specific constraints
- **Graceful Degradation**: Performance degrades gracefully (not
  catastrophically) outside training distribution
- **Internal Consistency**: Generated distributions maintain consistency while
  violating specified rules
- **Human Evaluation**: Outputs are rated as "surprisingly sensible" and useful

## Compute Requirements

### Development Phases

- **Local (Mac)**: Initial prototyping, small-scale tests
- **Colab**: Quick experiments, proof of concepts
- **Paperspace**: Main development and training
  - Phase 1: P4000 ($0.51/hr) for models <100M parameters
  - Phase 2: A4000 ($0.76/hr) for models up to 1B parameters

## Key Files to Reference

- `distribution_invention_research_plan.md`: Complete research plan and
  technical approach
- `setup_distribution_invention.md`: Detailed development environment setup
- `configs/experiment_configs.yaml`: Hyperparameter configurations (to be
  created)
- `models/core/distribution_generator.py`: Core model implementation (to be
  created)

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
