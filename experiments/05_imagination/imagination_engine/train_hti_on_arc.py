"""Train HTI on ARC-AGI-2 TRAINING data only.

This script implements proper training for the HTI system using
only the training portion of ARC-AGI-2. Evaluation data remains
completely unseen.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent))

from integrated_hti_system import IntegratedHTI
from arc_data_loader import load_arc_training_data, prepare_task_for_hti, split_training_data


class HTITrainer:
    """Trainer for the Hierarchical Transform Inventor."""
    
    def __init__(self, hti_system: IntegratedHTI):
        self.hti = hti_system
        self.training_history = []
        self.validation_history = []
        
    def train_epoch(self, train_tasks: List[Dict], verbose: bool = True) -> float:
        """Train HTI for one epoch on training tasks."""
        
        total_score = 0.0
        successful_tasks = 0
        
        for i, task in enumerate(train_tasks):
            if verbose and i % 10 == 0:
                print(f"  Training on task {i+1}/{len(train_tasks)}...")
            
            # Prepare task
            train_examples, test_examples = prepare_task_for_hti(task)
            
            # Solve with HTI (this updates memory and learns)
            transform, info = self.hti.solve_with_memory(train_examples, task['id'])
            
            # Evaluate on task's test examples
            task_score = 0.0
            for test_input, expected in test_examples:
                predicted = transform(test_input)
                if predicted.shape == expected.shape:
                    task_score += np.mean(predicted == expected)
            
            if test_examples:
                task_score /= len(test_examples)
            
            total_score += task_score
            if task_score > 0.5:
                successful_tasks += 1
            
            # Learn from success/failure
            if task_score > 0.7:
                # Successful solution - reinforce in memory
                self.hti.memory.retrieve(self.hti.hti.encode_task(train_examples), k=1)
        
        avg_score = total_score / len(train_tasks) if train_tasks else 0.0
        success_rate = successful_tasks / len(train_tasks) if train_tasks else 0.0
        
        return avg_score, success_rate
    
    def validate(self, val_tasks: List[Dict], verbose: bool = True) -> float:
        """Validate HTI on validation tasks."""
        
        total_score = 0.0
        perfect_solutions = 0
        
        for i, task in enumerate(val_tasks):
            if verbose and i % 5 == 0:
                print(f"  Validating task {i+1}/{len(val_tasks)}...")
            
            # Prepare task
            train_examples, test_examples = prepare_task_for_hti(task)
            
            # Solve (without updating memory as aggressively)
            transform, info = self.hti.solve_with_memory(train_examples, f"val_{task['id']}")
            
            # Evaluate
            task_score = 0.0
            for test_input, expected in test_examples:
                predicted = transform(test_input)
                if predicted.shape == expected.shape:
                    score = np.mean(predicted == expected)
                    task_score += score
                    if score > 0.99:
                        perfect_solutions += 1
            
            if test_examples:
                task_score /= len(test_examples)
            
            total_score += task_score
        
        avg_score = total_score / len(val_tasks) if val_tasks else 0.0
        perfect_rate = perfect_solutions / (len(val_tasks) * len(test_examples)) if val_tasks else 0.0
        
        return avg_score, perfect_rate
    
    def train(
        self,
        train_tasks: List[Dict],
        val_tasks: List[Dict],
        epochs: int = 5,
        save_best: bool = True
    ):
        """Full training loop."""
        
        print("\n" + "=" * 80)
        print("TRAINING HTI ON ARC-AGI-2")
        print("=" * 80)
        
        print(f"\nTraining configuration:")
        print(f"  Training tasks: {len(train_tasks)}")
        print(f"  Validation tasks: {len(val_tasks)}")
        print(f"  Epochs: {epochs}")
        print(f"  Memory capacity: {self.hti.memory.capacity}")
        
        best_val_score = 0.0
        
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'='*60}")
            
            # Training
            start_time = time.time()
            train_score, train_success = self.train_epoch(train_tasks, verbose=(epoch == 0))
            train_time = time.time() - start_time
            
            self.training_history.append(train_score)
            
            print(f"\nTraining results:")
            print(f"  Average score: {train_score:.1%}")
            print(f"  Success rate: {train_success:.1%}")
            print(f"  Time: {train_time:.1f}s")
            
            # Validation
            start_time = time.time()
            val_score, val_perfect = self.validate(val_tasks, verbose=(epoch == 0))
            val_time = time.time() - start_time
            
            self.validation_history.append(val_score)
            
            print(f"\nValidation results:")
            print(f"  Average score: {val_score:.1%}")
            print(f"  Perfect solutions: {val_perfect:.1%}")
            print(f"  Time: {val_time:.1f}s")
            
            # Memory statistics
            mem_stats = self.hti.memory.get_statistics()
            print(f"\nMemory statistics:")
            print(f"  Stored transforms: {mem_stats['total_transforms']}")
            print(f"  Retrieved: {mem_stats['total_retrieved']}")
            print(f"  Composed: {mem_stats['total_composed']}")
            
            # Save best model
            if save_best and val_score > best_val_score:
                best_val_score = val_score
                self.save_checkpoint(f"hti_best_val_{val_score:.3f}.json")
                print(f"\n✓ Saved best model (val: {val_score:.1%})")
            
            # Early stopping check
            if len(self.validation_history) > 3:
                recent_scores = self.validation_history[-3:]
                if all(s <= recent_scores[0] for s in recent_scores[1:]):
                    print("\n⚠ No improvement in 3 epochs, consider stopping")
    
    def save_checkpoint(self, filename: str):
        """Save HTI state."""
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'training_history': self.training_history,
            'validation_history': self.validation_history,
            'memory_stats': self.hti.memory.get_statistics(),
            'system_stats': self.hti.get_statistics()
        }
        
        checkpoint_file = Path(__file__).parent / "checkpoints" / filename
        checkpoint_file.parent.mkdir(exist_ok=True)
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        # Also save memory
        self.hti.memory.save(checkpoint_file.with_suffix('.memory.json'))


def main():
    """Main training script."""
    
    print("\n" + "=" * 80)
    print("HTI TRAINING ON ARC-AGI-2 (TRAINING DATA ONLY)")
    print("=" * 80)
    
    # Load training data
    print("\nLoading ARC training data...")
    try:
        all_tasks = load_arc_training_data(max_tasks=100)  # Start with subset
    except FileNotFoundError:
        print("\n❌ Training data not found!")
        print("Please run: python download_arc_agi_2.py")
        return
    
    # Split into train/validation
    print("\nSplitting data...")
    train_tasks, val_tasks = split_training_data(all_tasks, val_split=0.2)
    
    # Initialize HTI
    print("\nInitializing HTI system...")
    hti_system = IntegratedHTI()
    
    # Create trainer
    trainer = HTITrainer(hti_system)
    
    # Train
    trainer.train(
        train_tasks,
        val_tasks,
        epochs=3,  # Start with few epochs
        save_best=True
    )
    
    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    
    if trainer.training_history:
        print(f"\nTraining progression:")
        for i, score in enumerate(trainer.training_history):
            print(f"  Epoch {i+1}: {score:.1%}")
    
    if trainer.validation_history:
        print(f"\nValidation progression:")
        for i, score in enumerate(trainer.validation_history):
            print(f"  Epoch {i+1}: {score:.1%}")
        
        best_val = max(trainer.validation_history)
        print(f"\nBest validation score: {best_val:.1%}")
    
    # Memory analysis
    mem_stats = hti_system.memory.get_statistics()
    print(f"\nFinal memory state:")
    print(f"  Total transforms: {mem_stats['total_transforms']}")
    print(f"  Unique discoveries: {hti_system.transforms_discovered}")
    print(f"  Average performance: {mem_stats['average_performance']:.1%}")
    
    print(f"\nNext steps:")
    print(f"  1. Train on more data (we used {len(all_tasks)} tasks)")
    print(f"  2. Implement meta-learning for faster adaptation")
    print(f"  3. Run black-box evaluation when ready")
    print(f"\n⚠️ Remember: Do NOT access evaluation data until final testing!")


if __name__ == "__main__":
    main()