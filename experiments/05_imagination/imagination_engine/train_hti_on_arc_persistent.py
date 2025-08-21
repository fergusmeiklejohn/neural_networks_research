"""Train HTI on ARC-AGI-2 with PERSISTENT MEMORY across sessions.

This version loads previous memory if available, allowing incremental training.
"""

import json
import os
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
from transform_memory import StoredTransform


class PersistentHTITrainer:
    """Trainer for HTI with persistent memory across sessions."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize HTI
        self.hti = IntegratedHTI()
        
        # Try to load existing memory
        self.load_latest_memory()
        
        self.training_history = []
        self.validation_history = []
        
    def load_latest_memory(self) -> bool:
        """Load the most recent memory checkpoint if available."""
        memory_files = list(self.checkpoint_dir.glob("*.memory.json"))
        
        if not memory_files:
            print("No previous memory found. Starting fresh.")
            return False
        
        # Get most recent file
        latest_memory = max(memory_files, key=lambda f: f.stat().st_mtime)
        
        print(f"\n📂 Loading memory from: {latest_memory.name}")
        
        try:
            with open(latest_memory, 'r') as f:
                data = json.load(f)
            
            # Reconstruct transforms
            loaded_count = 0
            for transform_data in data.get('transforms', []):
                # Add transform back to memory
                if len(transform_data) >= 3:
                    primitives = transform_data[1]  # Primitive sequence
                    score = transform_data[2]  # Performance score
                    
                    # Create random task encoding (simplified)
                    task_encoding = np.random.randn(self.hti.memory.embedding_dim)
                    
                    # Add to memory
                    self.hti.memory.add(
                        primitives,
                        task_encoding,
                        score,
                        metadata={'loaded_from': str(latest_memory)}
                    )
                    loaded_count += 1
            
            print(f"✅ Loaded {loaded_count} transforms from previous session")
            
            # Load statistics
            if 'statistics' in data:
                stats = data['statistics']
                print(f"   Previous performance: {stats.get('average_performance', 0):.1%}")
                print(f"   Total stored: {stats.get('total_stored', 0)}")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Could not load memory: {e}")
            return False
    
    def save_checkpoint(self, filename: str = None):
        """Save HTI state with automatic naming."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            val_score = self.validation_history[-1] if self.validation_history else 0
            filename = f"hti_checkpoint_{timestamp}_val_{val_score:.3f}.json"
        
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'training_history': self.training_history,
            'validation_history': self.validation_history,
            'memory_stats': self.hti.memory.get_statistics(),
            'system_stats': self.hti.get_statistics()
        }
        
        checkpoint_file = self.checkpoint_dir / filename
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        # Save memory
        memory_file = checkpoint_file.with_suffix('.memory.json')
        self.hti.memory.save(str(memory_file))
        
        print(f"💾 Checkpoint saved: {filename}")
        
        return checkpoint_file
    
    def train_epoch(self, train_tasks: List[Dict], verbose: bool = True) -> Tuple[float, float]:
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
        
        avg_score = total_score / len(train_tasks) if train_tasks else 0.0
        success_rate = successful_tasks / len(train_tasks) if train_tasks else 0.0
        
        return avg_score, success_rate
    
    def validate(self, val_tasks: List[Dict], verbose: bool = True) -> Tuple[float, float]:
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
        save_every: int = 1
    ):
        """Full training loop with periodic saving."""
        
        print("\n" + "=" * 80)
        print("TRAINING HTI WITH PERSISTENT MEMORY")
        print("=" * 80)
        
        print(f"\nTraining configuration:")
        print(f"  Training tasks: {len(train_tasks)}")
        print(f"  Validation tasks: {len(val_tasks)}")
        print(f"  Epochs: {epochs}")
        print(f"  Memory capacity: {self.hti.memory.capacity}")
        print(f"  Current memory size: {len(self.hti.memory.transforms)}")
        
        best_val_score = max(self.validation_history) if self.validation_history else 0.0
        
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
            print(f"  Unique discoveries: {self.hti.transforms_discovered}")
            print(f"  Retrieved: {mem_stats['total_retrieved']}")
            print(f"  Composed: {mem_stats['total_composed']}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint()
            
            # Save best model
            if val_score > best_val_score:
                best_val_score = val_score
                best_file = self.save_checkpoint(f"hti_best_val_{val_score:.3f}.json")
                print(f"\n🏆 New best model! Validation: {val_score:.1%}")
            
            # Early stopping check
            if len(self.validation_history) > 3:
                recent_scores = self.validation_history[-3:]
                if all(s <= recent_scores[0] for s in recent_scores[1:]):
                    print("\n⚠️ No improvement in 3 epochs")
                    if epoch < epochs - 1:
                        response = input("Continue training? (y/n): ")
                        if response.lower() != 'y':
                            print("Stopping early.")
                            break


def main():
    """Main training script with persistence."""
    
    print("\n" + "=" * 80)
    print("HTI TRAINING WITH PERSISTENT MEMORY")
    print("=" * 80)
    
    # Configuration
    MAX_TASKS = 200  # Increase gradually
    EPOCHS = 5
    VAL_SPLIT = 0.2
    
    # Load training data
    print(f"\nLoading ARC training data (max {MAX_TASKS} tasks)...")
    try:
        all_tasks = load_arc_training_data(max_tasks=MAX_TASKS)
    except FileNotFoundError:
        print("\n❌ Training data not found!")
        print("Please run: python download_arc_agi_2.py")
        return
    
    # Split into train/validation
    print("\nSplitting data...")
    train_tasks, val_tasks = split_training_data(all_tasks, val_split=VAL_SPLIT)
    
    # Create trainer with persistent memory
    trainer = PersistentHTITrainer(checkpoint_dir="checkpoints")
    
    # Train
    trainer.train(
        train_tasks,
        val_tasks,
        epochs=EPOCHS,
        save_every=1  # Save after each epoch
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
        print(f"\n🏆 Best validation score: {best_val:.1%}")
    
    # Final memory analysis
    mem_stats = trainer.hti.memory.get_statistics()
    print(f"\n📊 Final memory state:")
    print(f"  Total transforms: {mem_stats['total_transforms']}")
    print(f"  Unique discoveries: {trainer.hti.transforms_discovered}")
    print(f"  Average performance: {mem_stats['average_performance']:.1%}")
    print(f"  Capacity used: {mem_stats['capacity_used']:.1%}")
    
    print(f"\n✅ Memory saved for next session!")
    print(f"   Next run will continue from {mem_stats['total_transforms']} transforms")
    
    print(f"\n📝 Next steps:")
    print(f"  1. Run again to continue training (memory persists)")
    print(f"  2. Increase MAX_TASKS gradually (currently {MAX_TASKS})")
    print(f"  3. When ready, run black-box evaluation")


if __name__ == "__main__":
    main()