# Quick Start Guide - Imagination Engine V4

## 30-Second Overview
We built a system that **learns HOW to solve problems** rather than memorizing solutions. Current performance: 10% on ARC tasks (up from 0%). The learning infrastructure works - it successfully stores and retrieves solutions.

## 5-Minute Orientation

### What's Working
- ✅ **No crashes** - All bounds checking fixed
- ✅ **Learning works** - Successfully accumulates knowledge  
- ✅ **Memory works** - Stores and retrieves solutions
- ✅ **10% accuracy** - First successful task solutions!

### What Needs Work
- ⚠️ Hypothesis generator score attribute issue
- ⚠️ Limited pattern coverage (need more strategies)
- ⚠️ Partial solution collection for composition
- ⚠️ Better marker detection for regions

### Key Files You'll Touch
1. **imagination_engine_v4.py** - Main orchestrator (start here)
2. **invention_strategies.py** - Add new strategies here
3. **meta_learner.py** - Learning logic
4. **hypothesis_generator.py** - Needs score fix

## Instant Start Commands

```bash
# 1. Activate environment
conda activate dist-invention
cd experiments/05_imagination/imagination_engine

# 2. Test system health (should work without errors)
/Users/fergusmeiklejohn/miniconda3/envs/dist-invention/bin/python test_single_task.py

# 3. Check learning progress
/Users/fergusmeiklejohn/miniconda3/envs/dist-invention/bin/python -c "
from meta_learner import MetaLearner
m = MetaLearner()
m.load()
print(m.get_learning_summary())
"

# 4. Run quick evaluation (5 tasks)
/Users/fergusmeiklejohn/miniconda3/envs/dist-invention/bin/python evaluate_v4_comprehensive.py --max-tasks 5 --rounds 2 --timeout 5

# 5. Run full evaluation (20 tasks)
/Users/fergusmeiklejohn/miniconda3/envs/dist-invention/bin/python evaluate_v4_comprehensive.py --max-tasks 20 --rounds 3 --timeout 10
```

## Most Important Code Sections

### 1. Where strategies are defined
**File**: `imagination_engine_v4.py`
**Lines**: 211-222 (meta-learning strategies), 317-327 (all strategies)

### 2. How learning works
**File**: `meta_learner.py`
**Lines**: 140-160 (strategy prediction), 250-300 (failure learning)

### 3. Where to add new strategies
**File**: `invention_strategies.py`
**Lines**: 627+ (new strategies start here)

### 4. The bug to fix first
**File**: `hypothesis_generator.py`
**Issue**: Hypothesis class needs score attribute
**Location**: Where Hypothesis objects are created

## Current Strategy Arsenal

### Strategies That Work
1. **geometric_reasoning** - ✅ Successfully solved task_009
2. **memory_retrieval** - ✅ Successfully reused solution

### Strategies Available
- pattern_decomposition
- abstraction_discovery
- multi_object_coordination (NEW)
- conditional_transformation (NEW)
- recursive_patterns (NEW)
- boundary_operations (NEW)
- region_extraction
- trace
- search

## Quick Improvements (Under 30 min each)

### 1. Fix Hypothesis Score (10 min)
```python
# In hypothesis_generator.py, add to Hypothesis class:
class Hypothesis:
    def __init__(self, ...):
        self.score = 0.0  # Add this
```

### 2. Add Symmetry Strategy (20 min)
```python
# In invention_strategies.py, add:
def symmetry_operations(self, examples):
    # Check for vertical, horizontal, diagonal symmetry
    # Return transformation function
```

### 3. Better Debug Output (15 min)
```python
# In imagination_engine_v4.py solve(), add:
self._log(f"  Strategy {name} score: {score}")
self._log(f"  Partial solutions collected: {len(partials)}")
```

## Understanding the Learning Flow

```
1. Task arrives → Extract features
2. Meta-learner predicts best strategies
3. Try strategies in order of confidence
4. If success → Store in memory
5. If failure → Learn what didn't work
6. Next task → Use accumulated knowledge
```

## Key Metrics to Watch

### During Development
- **Strategy success rate**: Which strategies actually work?
- **Memory hit rate**: Are we reusing solutions?
- **Learning velocity**: Does performance improve over rounds?

### For Evaluation
- **Overall accuracy**: Currently 10%
- **Tasks solved**: Currently 1/10
- **Strategies used**: Distribution of successful strategies

## Common Patterns in ARC Tasks

### Currently Handled (10% coverage)
- Geometric transformations (rotation, reflection)
- Simple patterns with clear rules

### Need to Add (for 20%+ coverage)
- Symmetry operations
- Counting and arithmetic
- Pattern completion
- Color mapping rules
- Grid subdivisions

## Debugging Checklist

If something breaks:

1. **Check bounds**: Are all grid accesses protected?
2. **Check None**: Are all returns checked for None?
3. **Check shapes**: Do input/output shapes match?
4. **Check memory**: Is the memory file corrupted?
5. **Check strategies**: Did a new strategy crash?

## Next Best Actions

### If you have 15 minutes
- Fix the hypothesis score issue
- Add debug output to see what's failing

### If you have 1 hour
- Add 1-2 new strategies (symmetry, counting)
- Improve partial solution collection

### If you have 2 hours
- Full strategy expansion (add 5+ strategies)
- Implement better composition logic

### If you have a day
- Major enhancement to abstraction engine
- Implement learnable hypothesis space
- Add neural components for pattern recognition

## Philosophy Reminder

**We optimize for learning, not performance.**

Bad approach:
- Add 100 hand-coded strategies
- Optimize for specific ARC tasks
- Focus only on accuracy

Good approach:
- Make strategies learn from experience
- Focus on generalization
- Build mechanisms that discover patterns

## Success Indicators

You're on the right track if:
- Memory hit rate increases over time
- Different strategies start succeeding
- Performance improves across rounds
- Novel solutions emerge unexpectedly

## Contact Points

### Key Documentation
- `COMPLETE_IMPLEMENTATION_GUIDE.md` - Full technical details
- `STRATEGIC_ROADMAP.md` - Long-term vision
- `research_diary/2025-08-22_research_diary.md` - Latest session details

### Critical Code Files
- `imagination_engine_v4.py` - Main engine
- `meta_learner.py` - Learning system
- `invention_strategies.py` - Strategy library

## Final Tip

**Start with the evaluation.** Run it first to see current performance, then make ONE change, then run again. This tight feedback loop is key to progress.

Remember: We went from 0% to 10% in one session. The next 10% will be easier because the system is now learning.

---

*"The system that learns fastest wins."*

Good luck!