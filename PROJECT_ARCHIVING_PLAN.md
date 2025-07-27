# Project Archiving and Organization Plan

Generated: 2025-07-27

## Overview

Before starting our new research directions, we need to archive outdated experiments and organize the project for clarity. This will reduce clutter by ~70% and make navigation much easier.

## Archiving Strategy

### 1. Archive Structure
```
archive/
├── README.md (explains archive contents and when they were created)
├── pre_ood_illusion_discovery/
│   ├── pinn_experiments/         # Failed physics-informed neural networks
│   ├── progressive_curriculum/   # Early training approaches
│   ├── complex_architectures/    # V1/V2 models that failed
│   └── early_experiments/        # Initial attempts
├── failed_approaches/
│   ├── tta_experiments/          # Test-time adaptation (failed)
│   ├── distribution_modifiers/   # Complex modification attempts
│   └── training_variants/        # Multiple training script iterations
├── old_documentation/
│   ├── status_files/            # Superseded status documents
│   ├── implementation_plans/     # Completed/abandoned plans
│   └── experiment_summaries/     # Old summaries
└── analysis_outputs/
    ├── failed_model_outputs/     # Results from failed approaches
    └── intermediate_results/     # Superseded analysis
```

### 2. What to Archive

#### High Priority (Immediate Archive)
- **All PINN-related files** (~15 files) - Conclusively failed approach
- **TTA experiments** (~25 files) - Failed adaptation attempts  
- **Multiple training script variants** (~20 files) - Superseded by template
- **Old status documents** (~10 files) - Replaced by current docs

#### Medium Priority (Archive After Review)
- **Early modifier attempts** (~10 files) - Keep only final versions
- **Test scripts for failed approaches** (~20 files)
- **Intermediate outputs** - Keep only final results

#### Low Priority (Keep for Now)
- **Data generation scripts** - May need for reproducibility
- **Analysis scripts** - Useful for future analysis
- **Research diary** - Essential for continuity

### 3. What to Keep Active

#### Essential Documentation
- CLAUDE.md (project guidance)
- DOCUMENTATION_INDEX.md (master index)
- CODE_RELIABILITY_GUIDE.md (critical learnings)
- PAPERSPACE_TRAINING_GUIDE.md (cloud training)
- SCIENTIFIC_WRITING_GUIDE.md (paper writing style - relocated from papers/ood_illusion/)
- Current research plans and directions

#### Active Code
- `models/baseline_models.py` (working baselines)
- `models/unified_evaluation.py` (evaluation framework)
- `utils/` directory (all utilities)
- Template training scripts
- True OOD data generation

#### Current Experiments
- `experiments/*/EXPERIMENT_PLAN.md`
- `experiments/*/CURRENT_STATUS.md`
- Working evaluation scripts
- Successful baseline implementations

### 4. Reorganization Plan

Create cleaner structure in active directories:
```
experiments/01_physics_worlds/
├── README.md                    # Explains current state
├── core/                        # Essential active code
│   ├── data_generation/
│   ├── evaluation/
│   └── training/
├── baselines/                   # Working baseline implementations
├── analysis/                    # Analysis scripts
└── results/                     # Current results only

experiments/02_compositional_language/
├── README.md
├── core/                        # Essential SCAN code
├── evaluation_illusion_study/   # Our key findings
└── results/
```

## Implementation Steps

### Step 1: Create Archive Structure (10 mins)
```bash
mkdir -p archive/{pre_ood_illusion_discovery,failed_approaches,old_documentation,relocated_documents,analysis_outputs}
mkdir -p archive/pre_ood_illusion_discovery/{pinn_experiments,progressive_curriculum,complex_architectures,early_experiments}
mkdir -p archive/failed_approaches/{tta_experiments,distribution_modifiers,training_variants}
mkdir -p archive/old_documentation/{status_files,implementation_plans,experiment_summaries}
mkdir -p archive/analysis_outputs/{failed_model_outputs,intermediate_results}
```

### Step 2: Create Archive Script (see below)

### Step 3: Archive in Batches
1. First batch: PINN and TTA files (highest clutter)
2. Second batch: Old training scripts
3. Third batch: Obsolete documentation
4. Fourth batch: Old outputs and results

### Step 4: Reorganize Active Files
1. Create new directory structure
2. Move files to appropriate locations
3. Update imports and references
4. Test that everything still works

### Step 5: Update Documentation
1. Update DOCUMENTATION_INDEX.md
2. Create README files in each directory
3. Update CLAUDE.md with new structure
4. Commit with clear message

## Archive Safety Script

```python
#!/usr/bin/env python3
"""
Safe archiving script that:
1. Creates archive directories
2. Copies files (doesn't move until confirmed)
3. Verifies copies
4. Updates git
5. Only deletes originals after confirmation
"""

import os
import shutil
import hashlib
from pathlib import Path
import json
from datetime import datetime

class SafeArchiver:
    def __init__(self, dry_run=True):
        self.dry_run = dry_run
        self.archive_log = []
        
    def calculate_checksum(self, filepath):
        """Calculate MD5 checksum of file"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def archive_file(self, source, destination):
        """Safely archive a file with verification"""
        if self.dry_run:
            print(f"[DRY RUN] Would archive: {source} -> {destination}")
            return True
            
        # Create destination directory
        dest_dir = os.path.dirname(destination)
        os.makedirs(dest_dir, exist_ok=True)
        
        # Calculate source checksum
        source_checksum = self.calculate_checksum(source)
        
        # Copy file
        shutil.copy2(source, destination)
        
        # Verify copy
        dest_checksum = self.calculate_checksum(destination)
        
        if source_checksum == dest_checksum:
            self.archive_log.append({
                'source': source,
                'destination': destination,
                'checksum': source_checksum,
                'timestamp': datetime.now().isoformat()
            })
            return True
        else:
            print(f"ERROR: Checksum mismatch for {source}")
            return False
    
    def save_log(self):
        """Save archive log for recovery"""
        log_path = 'archive/ARCHIVE_LOG.json'
        with open(log_path, 'w') as f:
            json.dump(self.archive_log, f, indent=2)
        print(f"Archive log saved to {log_path}")

# Usage
archiver = SafeArchiver(dry_run=True)
# First run with dry_run=True to see what will happen
# Then run with dry_run=False to actually archive
```

## Timeline

1. **Immediate (Today)**: 
   - Review and approve archiving plan
   - Run archiving script in dry-run mode
   - Archive PINN and TTA experiments

2. **Before Next Work Session**:
   - Complete archiving of failed approaches
   - Reorganize active experiments
   - Update documentation

3. **Ongoing**:
   - Archive failed experiments as we go
   - Keep only working/active code in main directories
   - Regular cleanup every 2 weeks

## Benefits

1. **70% reduction in file clutter**
2. **Clear separation of working vs historical code**
3. **Faster navigation to relevant files**
4. **Preserves all work for future reference**
5. **Easier onboarding for collaborators**

## Next Steps

1. Review this plan and adjust as needed
2. Run archiving script in dry-run mode
3. Execute archiving in batches
4. Update all documentation
5. Commit with clear history

This organized structure will make our new research directions much easier to implement and track.