# Scientific Research with Claude: A Recipe for Success

This guide distills hard-won lessons from developing neural networks for distribution invention into a reusable framework for conducting rigorous scientific research with Claude.

## Table of Contents

1. [Core Research Philosophy](#core-research-philosophy)
2. [Documentation Infrastructure](#documentation-infrastructure)
3. [Scientific Writing Standards](#scientific-writing-standards)
4. [Code Reliability Practices](#code-reliability-practices)
5. [Experiment Workflow](#experiment-workflow)
6. [Daily Research Process](#daily-research-process)
7. [End-of-Session Protocol](#end-of-session-protocol)
8. [Cloud Training Best Practices](#cloud-training-best-practices)
9. [Key Success Patterns](#key-success-patterns)
10. [Tools and Infrastructure](#tools-and-infrastructure)

## Core Research Philosophy

### 1. Research vs Engineering Mindset

**We are doing research, not engineering.** This fundamental principle shapes everything:

- **Seek truth through rigorous experimentation** - Don't optimize for quick wins
- **Persevere through difficult problems** - Don't cut corners when things get hard
- **Document everything** - Failures teach as much as successes
- **Verify assumptions** - What seems like breakthrough might be data leakage
- **Consult documentation** - Use tools like Context7 MCP or web search for established methods

### 2. The Power of Theory-Driven Implementation

Combine deep theoretical analysis with targeted implementation:

1. **First understand WHY something fails** - Prove mathematically why approaches don't work
2. **Let theory guide minimal solutions** - Understanding limitations leads to principled fixes
3. **Don't just try random architectures** - Design based on what's fundamentally missing
4. **Document theoretical insights** - Create formal analyses before implementing
5. **Theory reveals hidden assumptions** - Uncover implicit constraints in existing approaches

Example: In variable binding, we proved mathematically why static memory cannot work due to contradictory optimization objectives. This led directly to dynamic memory as the minimal necessary extension.

### 3. Scholarly Approach

Maintain scientific rigor and humility:

- **Avoid assumptions of novelty** - Others may have made similar observations
- **Present findings objectively** - Use measured language, not sensational claims
- **Acknowledge limitations** - Be explicit about what you don't know
- **Build on existing work** - Position findings within broader research context
- **Maintain scientific skepticism** - Even about your own results
- **Focus on reproducibility** - Evidence matters more than claims

## Documentation Infrastructure

### 1. Master Documentation Index

Create a `DOCUMENTATION_INDEX.md` that serves as the single source of truth:

```markdown
# Project Documentation Index

## Quick Navigation Guide
- **Project Overview**: `CLAUDE.md` - Start here
- **Current Status**: Check experiment `CURRENT_STATUS.md` files
- **Code Reliability**: `CODE_RELIABILITY_GUIDE.md` - MUST READ before coding

## Experiments
### Experiment 01: [Name]
- **Status**: `experiments/01_*/CURRENT_STATUS.md`
- **Plan**: `experiments/01_*/EXPERIMENT_PLAN.md`
- **Key Finding**: [One-line summary]
```

### 2. Research Diary Best Practices

Maintain daily entries in `research_diary/YYYY-MM-DD_research_diary.md`:

**Make entries actionable for tomorrow:**

1. **Specific file paths and line numbers**
   - Example: "Check geometric features in `models/baseline_models.py:L142-156`"

2. **Exact commands to run**
   - Example: "Run: `python train_baselines.py --model graph_extrap --verbose`"

3. **Current state summary**
   - Example: "Have working data pipeline: `train_minimal.py` loads trajectories"

4. **Critical context**
   - Example: "Data is in PIXELS not meters! Use `gravity / 40.0` for conversion"

5. **Open questions with hypotheses**
   - Example: "Does model train on multiple values? This could explain success"

6. **Next steps with entry points**
   - Example: "Start from: `BENCHMARK.md:L36-47` for time-varying physics"

### 3. Experiment Planning Documents

For each experiment, maintain:

- `EXPERIMENT_PLAN.md` - Overall strategy and phases
- `CURRENT_STATUS.md` - Real-time progress updates
- `claude-plans/` folder - Timestamped detailed plans

Include in plans:
- Context about current state
- Specific, actionable steps in priority order
- Success metrics and risk mitigation
- Scientific validity considerations

## Scientific Writing Standards

### 1. Measured, Objective Tone

**Let data speak for itself:**

- ✅ "Methods amplify errors by 235%"
- ❌ "Methods catastrophically explode"

- ✅ "This discovery led to a deeper realization"
- ❌ "This shocking discovery exposes the illusion"

- ✅ "Results indicate"
- ❌ "Results prove"

### 2. Statistical Rigor

Support all claims with statistics:

- Include confidence intervals: "235% worse (95% CI: 220-250%)"
- Report p-values: "No significant improvement (p > 0.05)"
- Use multiple seeds: "Results averaged over 5 seeds"
- Add error bars to all figures
- Report mean ± standard deviation

### 3. Scope Management

Calibrate claims to evidence:

- "In physics tasks with mechanism shifts..." not "In all OOD settings..."
- "Our experiments show..." not "This proves..."
- "The studied setting..." not "All settings..."
- Acknowledge boundaries: "While our results apply to X, Y remains open"

### 4. Professional Standards

- **No rhetorical questions** - Use declarative statements
- **Avoid repetition** - State key numbers once prominently
- **Reference all figures** - "As shown in Table 2..."
- **Recent citations** - Include 2023-2025 work
- **Move speculation** - Keep results factual, speculation in Future Work

## Code Reliability Practices

### 1. Always Verify Data First

Start every implementation with data inspection:

```python
# ALWAYS start with data inspection
print(f"Data shape: {data.shape}")
print(f"First sample: {data[0]}")
print(f"Column meanings: {column_names}")
print(f"Value ranges: min={data.min(axis=0)}, max={data.max(axis=0)}")
```

### 2. Use Centralized Utilities

Eliminate common issues with standardized imports:

```python
# At the top of every script
from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
from utils.paths import get_data_path, get_output_path
config = setup_environment()
```

### 3. Document Units and Conversions

Be explicit about units in variable names:

```python
# Clear unit indication
gravity_pixels_per_s2 = 400.0
gravity_m_per_s2 = gravity_pixels_per_s2 / 40.0  # Conversion factor

# Document column meanings
X1_COL, Y1_COL = 1, 2  # Ball 1 position columns
```

### 4. Test Minimally First

Before full training runs:

- Use 100-1000 samples (not full dataset)
- Run 1-2 epochs (not 50+)
- Verify shapes and losses decrease
- Check outputs are reasonable

## Experiment Workflow

### 1. Three-Phase Training Strategy

Structure experiments progressively:

**Phase 1: Learn Base Distributions**
- Train on standard datasets
- Verify reconstruction ability
- Train all baseline models

**Phase 2: Controlled Modification**
- Synthetic tasks with known changes
- Test modification capabilities
- Compare against baselines

**Phase 3: Creative Generation**
- Open-ended modifications
- Evaluate coherence
- Human assessment if needed

### 2. Mandatory Baseline Comparisons

For EVERY experiment:

1. **Train Baselines** - At least 4 standard approaches
2. **Representation Analysis** - Verify true OOD vs interpolation
3. **Modification Testing** - Create modification test suite
4. **Unified Evaluation** - Standardized comparison framework
5. **Report Generation** - Include comparison tables

### 3. Progressive Complexity

- Start with simplest possible version
- Verify it works completely
- Add one complexity at a time
- Document what each addition brings

## Daily Research Process

### 1. Morning Startup Routine

1. **Read latest research diary** - Get context from yesterday
2. **Check experiment status** - Read `CURRENT_STATUS.md`
3. **Review documentation index** - Ensure you know where things are
4. **Plan today's work** - Create specific goals

### 2. Before Writing Code

1. **Check reliability guide** - Avoid known pitfalls
2. **Use centralized imports** - No `sys.path.append`
3. **Verify data format** - Always inspect first
4. **Plan minimal test** - How to verify quickly?

### 3. During Development

1. **Test frequently** - Every significant change
2. **Document inline** - Explain non-obvious decisions
3. **Update status** - Keep `CURRENT_STATUS.md` current
4. **Commit often** - Preserve working states

## End-of-Session Protocol

Complete these before ending any session:

### 1. Update Research Diary

Create/update `research_diary/YYYY-MM-DD_research_diary.md`:

- Summary of what was accomplished
- Key findings and insights
- Problems encountered and solutions
- Specific next steps with file locations
- Critical context for tomorrow

### 2. Update Experiment Status

If you worked on an experiment:

- Update `experiments/*/CURRENT_STATUS.md`
- Include latest results
- Document known issues
- List immediate next steps

### 3. Synchronize Documentation

Ensure consistency:

- Update `DOCUMENTATION_INDEX.md` if needed
- Check cross-references are accurate
- Verify no conflicting information

### 4. Commit with Descriptive Message

```bash
git add -A
git commit -m "Update documentation: [brief description]

- Updated research diary with [findings]
- Updated [experiment] status
- [Other key updates]

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

## Cloud Training Best Practices

### 1. Self-Contained Scripts

Create scripts that include everything:

```python
# paperspace_train.py
def generate_data():
    """Generate all data on cloud machine"""
    pass

def train_model():
    """Complete training pipeline"""
    pass

if __name__ == "__main__":
    generate_data()  # Don't assume data exists
    train_model()
```

### 2. Path Flexibility

Handle different environments:

```python
# Auto-detect base path
if os.path.exists('/notebooks/project'):
    base_path = '/notebooks/project'
elif os.path.exists('/workspace/project'):
    base_path = '/workspace/project'
else:
    base_path = os.path.abspath('.')
```

### 3. Save During Training

Don't wait until the end:

```python
# Save checkpoints frequently
if os.path.exists('/storage'):
    checkpoint_path = f'/storage/exp_{timestamp}/epoch_{epoch}.h5'
    model.save_weights(checkpoint_path)
```

### 4. Document Requirements

Always include:

- Exact commands to run
- Expected runtime
- Memory requirements
- Recovery procedures

## Key Success Patterns

### 1. Theory Guides Implementation

- Don't implement blind
- Understand the problem deeply first
- Let understanding guide minimal solution
- Document theoretical insights

### 2. Document Everything

- Failed experiments are valuable
- Record unexpected behaviors
- Keep timestamped logs
- Make notes actionable

### 3. Verify Assumptions

- What looks like success might be leakage
- Check in representation space
- Use multiple validation approaches
- Be your own skeptic

### 4. Start Simple

- Minimal viable experiment first
- One complexity addition at a time
- Verify each step works
- Build understanding progressively

### 5. Make Documentation Actionable

- Include exact commands
- Provide specific file locations
- Document quirks and gotchas
- Enable zero-friction resumption

## Tools and Infrastructure

### 1. Pre-commit Hooks

Automated quality checks:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    hooks:
      - id: isort
```

### 2. Centralized Configuration

Standardize across environments:

```python
# utils/config.py
def setup_environment():
    """Configure logging, seeds, backends"""
    setup_logging()
    set_random_seeds(42)
    configure_keras_backend()
    return config
```

### 3. Unified Evaluation

Compare approaches fairly:

```python
# models/unified_evaluation.py
class UnifiedEvaluator:
    def evaluate_all_models(self, models, test_sets):
        """Run standardized evaluation"""
        results = {}
        for model in models:
            for test_set in test_sets:
                results[model][test_set] = evaluate(model, test_set)
        return results
```

### 4. Experiment Tracking

Use tools like Weights & Biases:

```python
import wandb

wandb.init(project="research", name=experiment_name)
wandb.config.update(config)
wandb.log({"loss": loss, "accuracy": accuracy})
```

## Conclusion

Successful scientific research with Claude requires:

1. **Rigorous methodology** - Theory-driven, well-documented, skeptical
2. **Robust infrastructure** - Reliable code, comprehensive testing, version control
3. **Clear communication** - Actionable documentation, objective writing, reproducible results
4. **Progressive development** - Start simple, verify thoroughly, add complexity carefully
5. **Continuous learning** - Document failures, update processes, share insights

By following these practices, you can conduct research that not only achieves results but also advances scientific understanding in a reproducible, reliable way.

Remember: **We are doing research, not engineering.** The goal is truth through rigorous experimentation, not just working code.
