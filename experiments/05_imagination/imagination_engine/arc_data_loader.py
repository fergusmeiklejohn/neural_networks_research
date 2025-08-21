"""Safe ARC-AGI-2 data loader that prevents evaluation data access."""

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
            
            # Each file contains a single task with train/test keys
            if isinstance(data, dict) and 'train' in data:
                task = {
                    'id': file.stem,  # Use filename as task ID
                    'train': data.get('train', []),
                    'test': data.get('test', [])
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
