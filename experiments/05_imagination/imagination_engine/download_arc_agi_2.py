"""Download and setup ARC-AGI-2 dataset with STRICT data isolation.

CRITICAL: 
- Training data: Can be used for training and validation
- Evaluation data: MUST remain unseen until final black-box evaluation
- We will create a separate evaluation script that prevents data leakage
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

def setup_arc_agi_2():
    """Download and organize ARC-AGI-2 with proper data isolation."""
    
    print("=" * 80)
    print("SETTING UP ARC-AGI-2 WITH STRICT DATA ISOLATION")
    print("=" * 80)
    
    # Create data directory structure
    base_dir = Path(__file__).parent / "arc_agi_2_data"
    train_dir = base_dir / "training"
    eval_dir = base_dir / "evaluation_BLACKBOX"
    
    print(f"\nCreating directory structure...")
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Clone repository to temp location
    temp_dir = Path("/tmp/arc_agi_2_temp")
    
    if temp_dir.exists():
        print(f"Removing existing temp directory...")
        shutil.rmtree(temp_dir)
    
    print(f"\nCloning ARC-AGI-2 repository...")
    try:
        subprocess.run([
            "git", "clone", 
            "https://github.com/arcprize/ARC-AGI-2.git",
            str(temp_dir)
        ], check=True)
        print("✓ Repository cloned successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to clone repository: {e}")
        return False
    
    # Copy training data
    print(f"\nCopying training data...")
    train_source = temp_dir / "data" / "training"
    if train_source.exists():
        # Copy all training files
        for file in train_source.glob("*.json"):
            shutil.copy2(file, train_dir / file.name)
            print(f"  ✓ Copied {file.name}")
    else:
        print("  ⚠ Training data not found in expected location")
    
    # Copy evaluation data WITH WARNING
    print(f"\n" + "!" * 60)
    print("COPYING EVALUATION DATA TO BLACKBOX DIRECTORY")
    print("WARNING: This data must NEVER be accessed during development!")
    print("!" * 60)
    
    eval_source = temp_dir / "data" / "evaluation"
    if eval_source.exists():
        for file in eval_source.glob("*.json"):
            shutil.copy2(file, eval_dir / file.name)
            print(f"  ✓ Copied {file.name} to BLACKBOX")
    else:
        print("  ⚠ Evaluation data not found")
    
    # Create access control file
    blackbox_warning = eval_dir / "DO_NOT_ACCESS_UNTIL_FINAL_EVAL.txt"
    with open(blackbox_warning, "w") as f:
        f.write("""
================================================================================
                            BLACK BOX EVALUATION DATA
================================================================================

This directory contains the ARC-AGI-2 evaluation dataset.

CRITICAL RULES:
1. DO NOT access this data during development
2. DO NOT look at these files during training  
3. DO NOT use this data for validation or debugging
4. ONLY use for final black-box evaluation

The integrity of our research depends on keeping this data unseen.

To run final evaluation:
  python run_blackbox_evaluation.py

================================================================================
""")
    
    # Clean up temp directory
    print(f"\nCleaning up temporary files...")
    shutil.rmtree(temp_dir)
    
    # Create data statistics (training only!)
    print(f"\nAnalyzing training data (evaluation remains unseen)...")
    train_files = list(train_dir.glob("*.json"))
    
    total_tasks = 0
    total_examples = 0
    
    for file in train_files[:5]:  # Sample a few files
        with open(file, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                total_tasks += len(data)
                for task in data.values():
                    if 'train' in task:
                        total_examples += len(task['train'])
    
    print(f"\nTraining Data Statistics:")
    print(f"  Files: {len(train_files)}")
    print(f"  Tasks (sample): ~{total_tasks * len(train_files) // 5}")
    print(f"  Examples (sample): ~{total_examples * len(train_files) // 5}")
    
    # Create safe training data loader
    create_safe_loader(base_dir)
    
    print(f"\n" + "=" * 80)
    print("SETUP COMPLETE")
    print("=" * 80)
    print(f"\n✅ Training data available at: {train_dir}")
    print(f"🔒 Evaluation data locked at: {eval_dir}")
    print(f"\nNext steps:")
    print(f"  1. Use load_arc_training_data() for training")
    print(f"  2. Train and validate ONLY on training data")
    print(f"  3. Run final evaluation with run_blackbox_evaluation.py")
    
    return True


def create_safe_loader(base_dir: Path):
    """Create a safe data loader that prevents access to evaluation data."""
    
    loader_code = '''"""Safe ARC-AGI-2 data loader that prevents evaluation data access."""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


def load_arc_training_data(max_tasks: int = None) -> List[Dict]:
    """Load ONLY training data. Evaluation data remains inaccessible."""
    
    train_dir = Path(__file__).parent / "arc_agi_2_data" / "training"
    
    if not train_dir.exists():
        raise FileNotFoundError(
            "Training data not found. Run download_arc_agi_2.py first."
        )
    
    all_tasks = []
    
    for file in train_dir.glob("*.json"):
        with open(file, 'r') as f:
            data = json.load(f)
            
            if isinstance(data, dict):
                for task_id, task_data in data.items():
                    task = {
                        'id': task_id,
                        'train': task_data.get('train', []),
                        'test': task_data.get('test', [])
                    }
                    all_tasks.append(task)
    
    if max_tasks:
        all_tasks = all_tasks[:max_tasks]
    
    print(f"Loaded {len(all_tasks)} training tasks")
    return all_tasks


def prepare_task_for_hti(task: Dict) -> Tuple[List, List]:
    """Convert ARC task format to HTI format."""
    
    train_examples = []
    for example in task['train']:
        inp = np.array(example['input'], dtype=np.float32)
        out = np.array(example['output'], dtype=np.float32)
        train_examples.append((inp, out))
    
    test_examples = []
    for example in task['test']:
        inp = np.array(example['input'], dtype=np.float32)
        out = np.array(example['output'], dtype=np.float32)
        test_examples.append((inp, out))
    
    return train_examples, test_examples


def split_training_data(tasks: List[Dict], val_split: float = 0.1):
    """Split training data into train/validation sets."""
    
    n_val = int(len(tasks) * val_split)
    
    # Shuffle for random split
    import random
    random.shuffle(tasks)
    
    val_tasks = tasks[:n_val]
    train_tasks = tasks[n_val:]
    
    print(f"Split: {len(train_tasks)} training, {len(val_tasks)} validation")
    
    return train_tasks, val_tasks


# IMPORTANT: No function to load evaluation data!
# Evaluation data access is restricted to run_blackbox_evaluation.py only
'''
    
    loader_file = base_dir.parent / "arc_data_loader.py"
    with open(loader_file, 'w') as f:
        f.write(loader_code)
    
    print(f"✓ Created safe data loader: {loader_file}")


if __name__ == "__main__":
    success = setup_arc_agi_2()
    
    if success:
        print("\n✅ ARC-AGI-2 setup successful with proper data isolation!")
    else:
        print("\n❌ Setup failed. Please check errors above.")