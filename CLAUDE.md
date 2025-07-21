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

## Core Research Philosophy

**We are doing research, not engineering.** We seek truth through rigorous experimentation. This means:
- Persevere through difficult problems - don't cut corners
- Document everything - failures teach as much as successes
- Verify assumptions - what seems like OOD might be interpolation
- Consult documentation (eg Context7 MCP or if that doesn't work the internet) when implementing new approaches

### Scholarly Approach
- **Avoid assumptions of novelty**: Others may have made similar observations - search thoroughly
- **Present findings objectively**: Use measured language, not sensational claims
- **Acknowledge limitations**: Be explicit about what we don't know
- **Build on existing work**: Position findings within broader research context
- **Maintain scientific skepticism**: Even about our own results
- **Focus on reproducibility**: Evidence matters more than claims

## 🧭 Knowledge Navigation

### Where to Find Current Information
- **Latest Research Status**: `research_diary/` - Check most recent entry
- **Experiment Progress**: `experiments/*/CURRENT_STATUS.md` - Real-time updates
- **All Documentation**: `DOCUMENTATION_INDEX.md` - Master index of everything
- **Code Reliability**: `CODE_RELIABILITY_GUIDE.md` - MUST READ before coding
- **Cloud Training**: `PAPERSPACE_TRAINING_GUIDE.md` - Essential for GPU runs

### Critical Process Reminders
1. **Before Starting Work**: Read latest research diary entry for context
2. **Before Writing Code**: 
   - Check CODE_RELIABILITY_GUIDE.md for common pitfalls
   - Use centralized imports from `utils/` (see CODE_QUALITY_SETUP.md)
3. **Before Training**: Review PAPERSPACE_TRAINING_GUIDE.md for cloud setup
4. **When Lost**: DOCUMENTATION_INDEX.md has links to everything
5. **For Code Quality**: See CODE_QUALITY_SETUP.md for new infrastructure

## Development Environment

### Python Environment Setup

The project uses a Conda environment named `dist-invention` with Python 3.11:

```bash
# Activate environment
conda activate dist-invention

# Key dependencies
pip install keras>=3.0 torch jax[metal] transformers wandb

# IMPORTANT: When Claude needs to run Python scripts, use the full path:
/Users/fergusmeiklejohn/miniconda3/envs/dist-invention/bin/python script.py
```
Sometimes Claude cannot run scripts and sees errors like that a package isn't available in the python environment. Often this is an environment error and Claude should ask for help running files in these cases rather than change our strategy or scripts to get round the problem.

### Backend Configuration

The project uses **Keras 3** with multi-backend support:

- **Primary Backend**: JAX with Metal acceleration (Mac)
- **Alternative Backends**: PyTorch, TensorFlow
- Configure via `~/.keras/keras.json` or `KERAS_BACKEND` environment variable

### Development Tools

- **IDE**: Cursor/VS Code with Python Interactive files (`.py` with `# %%` cells)
  - Configured with strict Pylance type checking (see `.vscode/settings.json`)
- **Notebooks**: Jupyter for final presentations/sharing
- **Experiment Tracking**: Weights & Biases (`wandb`)
- **Code Quality**: 
  - **Automated**: Pre-commit hooks (black, flake8, isort, mypy)
  - **Real-time**: Pylance strict mode in Cursor/VS Code
  - **Centralized**: See `utils/` for imports, config, and paths
  - **Setup Guide**: CODE_QUALITY_SETUP.md

## Project Structure

### Current Research Status
The project is in **active experimentation**. For latest status:
- Check `experiments/*/CURRENT_STATUS.md` for each experiment
- Read most recent `research_diary/` entry for overall progress
- See `DOCUMENTATION_INDEX.md` for comprehensive status tracking

**Key Learning**: Most "OOD" benchmarks actually test interpolation. We discovered the "OOD Illusion" - true extrapolation requires careful experimental design.

### Architecture (evolving - check experiment folders for latest)

```
experiments/                    # Each has CURRENT_STATUS.md and EXPERIMENT_PLAN.md
├── 01_physics_worlds/         # Active: Discovered PINN failures, designing true OOD
├── 02_compositional_language/ # Ready to retry with improved infrastructure
├── 03_visual_concepts/        # Planned: Visual concept blending
├── 04_abstract_reasoning/     # Planned: ARC-like puzzle solving
├── 05_mathematical_extension/ # Planned: Mathematical concept extension
└── 06_multimodal_transfer/    # Planned: Cross-modal rule transfer

models/                        # Check git status for latest implementations
├── core/                      # Core architecture components
├── baseline_models.py         # All 4 baseline implementations
├── unified_evaluation.py      # Standardized evaluation framework
└── minimal_physics_model.py   # Recent PINN experiment

data/                          # Note: processed data is gitignored
scripts/                       # Training and evaluation scripts
research_diary/                # Daily progress and insights
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

## Research Process

### Daily Workflow
1. **Start**: Read latest research diary entry for context
2. **Plan**: Check experiment CURRENT_STATUS.md for next steps
3. **Code**: Reference CODE_RELIABILITY_GUIDE.md to avoid known issues
4. **Test**: Run minimal tests (100 samples, 2 epochs) before full training
5. **Document**: Update research diary with findings and next steps

### When Starting New Work
1. Check DOCUMENTATION_INDEX.md for relevant existing work
2. Read the experiment's EXPERIMENT_PLAN.md
3. Review any existing code in the experiment folder
4. Create timestamped plan in `claude-plans/` if needed
5. Always test with minimal data first

## Key Principles

- **Research Focus**: We prioritize understanding over implementation speed
- **Reproducibility**: Every experiment must be reproducible from scratch
- **Failure Documentation**: Negative results are as valuable as positive ones
- **Progressive Complexity**: Start simple, add complexity gradually
- **Continuous Learning**: Update guides as we discover new patterns

## End of Session Protocol

**IMPORTANT**: Before ending any work session, complete these documentation updates to ensure the next session starts with accurate information:

### 1. Update Research Diary
- Create or update today's diary entry in `research_diary/YYYY-MM-DD_research_diary.md`
- Ensure it includes all actionable elements from the template
- Focus on making it actionable for tomorrow (specific commands, file paths, line numbers)

### 2. Update Experiment Status
If you worked on an experiment:
- Update the relevant `experiments/*/CURRENT_STATUS.md` file
- Include latest results, what's working, known issues, and immediate next steps
- Update status tags (Active, Complete, Blocked, etc.)

### 3. Update Documentation Index
If any of these changed:
- Latest diary date in `DOCUMENTATION_INDEX.md`
- Experiment status or major findings
- New key documents created
- Important file paths or commands discovered

### 4. Update CLAUDE.md (if needed)
Update this file if:
- Experiment phases changed significantly
- New critical process discovered
- Important warnings or gotchas found
- Key commands or paths changed

### 5. Verify Consistency
Quick check that:
- All cross-references between documents are accurate
- Status descriptions match across files
- Latest findings are reflected in relevant places
- No conflicting information exists

### 6. Commit Changes
```bash
git add -A
git commit -m "Update documentation: [brief description of session work]

- Updated research diary with [key findings]
- Updated [experiment] status
- [Other key updates]

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

This protocol ensures documentation stays synchronized and the next session can start immediately with full context.

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

### Code Patterns - IMPORTANT UPDATE

**We now use centralized utilities for consistent code quality across the project.**

Instead of scattered `sys.path.append` calls, use:
```python
# At the top of every script
from utils.imports import setup_project_paths
setup_project_paths()

# Then import project modules normally
from utils.config import setup_environment
from utils.paths import get_data_path, get_output_path
from models.baseline_models import BaselineModel

# Set up environment (includes logging, random seeds, Keras backend)
config = setup_environment()
```

Key benefits:
- **No more import errors** from path issues
- **Consistent configuration** across environments
- **Type checking** catches errors before runtime
- **Pre-commit hooks** ensure code quality

See `CODE_QUALITY_SETUP.md` for complete guide.

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

**CRITICAL: Make diary entries actionable for tomorrow**
The diary must serve as a practical working document that enables immediate productivity the next day. Include:

1. **Specific file paths and line numbers** - Enable jumping directly to relevant code
   - Example: "Check geometric features in `models/baseline_models.py:L142-156`"

2. **Exact commands to run** - No searching needed
   - Example: "Run: `python train_baselines.py --model graph_extrap --verbose`"

3. **Current state summary** - What's working, what's ready
   - Example: "Have working data pipeline: `train_minimal_pinn.py` loads 2-ball trajectories"

4. **Critical context** - Quirks and conversions to remember
   - Example: "Data is in PIXELS not meters! Use `physics_config['gravity'] / 40.0`"

5. **Open questions with hypotheses** - Guide tomorrow's investigation
   - Example: "Does GraphExtrap train on multiple gravity values? This could explain interpolation success"

6. **Next steps with entry points** - Specific starting locations
   - Example: "Start from: `TRUE_OOD_BENCHMARK.md:L36-47` for time-varying gravity"

The goal is zero friction when resuming work - the diary should contain everything needed to pick up exactly where you left off with full context and clear direction.

### Daily Workflow (Git Worktree Setup)

We use git worktrees for parallel development, with automated tools for daily merging to production:

**End-of-Day Workflow:**
```bash
# Single command handles everything:
./scripts/daily_merge.sh
```

This automated workflow:
1. **Updates Research Diary** (optional prompt)
   - Auto-generates template with git activity
   - Tracks modified files and recent commits
   - Opens in editor for completion
   
2. **Runs Pre-Merge Tests**
   - Python environment verification
   - Critical imports testing
   - Code quality checks (black, flake8)
   - TTA weight restoration tests
   - Warns about uncommitted changes
   
3. **Creates and Merges PR**
   - Pushes current branch to origin
   - Auto-generates PR description from commits
   - Merges to production branch
   - Updates local branches

**Manual Commands (if needed):**
```bash
# Update research diary separately
./scripts/update_research_diary.sh

# Run tests without merging
./scripts/pre_merge_tests.sh

# Switch between worktrees
cd /Users/fergusmeiklejohn/conductor/repo/neural_networks_research/.main   # production
cd /Users/fergusmeiklejohn/conductor/repo/neural_networks_research/vienna  # feature branch
```

**Important Notes:**
- Always commit all changes before running daily merge
- Tests must pass (or be manually overridden) to proceed
- Branch remains active for continued work next day
- Research diary helps maintain continuity between sessions

### Research Paper Writing Style

**CRITICAL**: When writing research papers, always follow the comprehensive style guide in `papers/ood_illusion/SCIENTIFIC_WRITING_NOTES.md`. This guide incorporates both our internal standards and valuable feedback from external reviewers.

**Key Principles**:
1. **Measured, Objective Tone**: Let data speak for itself without emotional language
2. **Statistical Rigor**: Include confidence intervals, p-values, and error bars
3. **Precise Scope**: Claims must match evidence (e.g., "in physics tasks with mechanism shifts")
4. **Recent Citations**: Include 2023-2025 work to show current relevance
5. **No Rhetorical Questions**: Use declarative statements instead
6. **Avoid Repetition**: State key numbers prominently once, then vary presentation

**Before Writing Papers**:
- Read `SCIENTIFIC_WRITING_NOTES.md` thoroughly
- Review examples of good vs bad phrasing
- Check scope of claims against actual evidence
- Ensure all figures/tables are referenced in text
- Move speculation to clearly labeled Future Work sections

**Common Pitfalls to Avoid**:
- Universal claims from limited evidence
- Emotional/hyperbolic language ("catastrophic", "shocking")
- Unsupported absolute statements ("proves", "cannot work")
- Missing statistical support for comparative claims
- Speculation mixed with results

The style guide includes specific examples and transformations to ensure professional, defensible scientific writing that will withstand peer review
