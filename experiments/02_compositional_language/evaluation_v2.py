#!/usr/bin/env python3
"""
Evaluation V2: Proper evaluation using modification-specific validation sets.

This script addresses the "Evaluation Illusion" by measuring model performance
on validation sets that contain actual modified examples, not just interpolation.
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

# Import models and data utilities
from scan_data_loader import SCANDataLoader, SCANSample
from train_progressive_curriculum import SCANTokenizer
from models import CompositionalLanguageModel, create_model
from models_v2 import CompositionalLanguageModelV2, create_model_v2


class ModificationAwareEvaluator:
    """Evaluator that properly measures performance on modified examples."""
    
    def __init__(self, tokenizer: SCANTokenizer):
        self.tokenizer = tokenizer
        self.results = {}
        
    def load_validation_set(self, set_name: str, base_path: Path) -> List[SCANSample]:
        """Load a specific validation set."""
        pkl_path = base_path / f"{set_name}.pkl"
        
        if not pkl_path.exists():
            raise FileNotFoundError(f"Validation set not found: {pkl_path}")
        
        with open(pkl_path, 'rb') as f:
            samples = pickle.load(f)
            
        print(f"Loaded {set_name}: {len(samples)} samples")
        return samples
    
    def evaluate_exact_match(self, 
                           model: keras.Model, 
                           samples: List[SCANSample],
                           batch_size: int = 32,
                           verbose: bool = True) -> Dict[str, float]:
        """Evaluate exact match accuracy on samples."""
        correct = 0
        total = 0
        
        # Process in batches
        num_batches = (len(samples) + batch_size - 1) // batch_size
        
        if verbose:
            progress_bar = tqdm(range(num_batches), desc="Evaluating")
        else:
            progress_bar = range(num_batches)
        
        for batch_idx in progress_bar:
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(samples))
            batch_samples = samples[start_idx:end_idx]
            
            # Tokenize batch
            commands = []
            expected_actions = []
            modifications = []
            
            for sample in batch_samples:
                # Handle both dict and object formats
                if isinstance(sample, dict):
                    command = sample['command']
                    action = sample['action']
                    modification = sample.get('modification', None)
                else:
                    command = sample.command
                    action = sample.action
                    modification = getattr(sample, 'modification', None)
                
                # Encode command
                cmd_tokens = self.tokenizer.encode_command(command)
                commands.append(cmd_tokens)
                
                # Encode expected action
                act_tokens = self.tokenizer.encode_action(action)
                expected_actions.append(act_tokens)
                
                # Handle modification if present
                if modification:
                    # Create one-hot modification vector for simple baseline
                    mod_types = ['walk_skip', 'jump_hop', 'look_scan', 'left_right']
                    mod_vector = np.zeros(8, dtype=np.float32)
                    if modification in mod_types:
                        mod_vector[mod_types.index(modification)] = 1.0
                    modifications.append(mod_vector)
                else:
                    modifications.append(np.zeros(8, dtype=np.float32))
            
            # Pad sequences
            max_cmd_len = max(len(cmd) for cmd in commands)
            max_act_len = max(len(act) for act in expected_actions)
            
            # Create tensors
            cmd_tensor = np.zeros((len(batch_samples), max_cmd_len), dtype=np.int32)
            for i, cmd in enumerate(commands):
                cmd_tensor[i, :len(cmd)] = cmd
            
            # Create modification tensor (one-hot vectors)
            mod_tensor = np.array(modifications, dtype=np.float32)
            
            # Generate predictions
            try:
                # Check if model has generate_action method (v1) or needs direct call
                if hasattr(model, 'generate_action'):
                    predictions = model.generate_action(
                        tf.constant(cmd_tensor),
                        modification=tf.constant(mod_tensor) if mod_tensor is not None else None,
                        start_token=self.tokenizer.action_to_id['<START>'],
                        end_token=self.tokenizer.action_to_id['<END>'],
                        max_length=max_act_len + 10
                    )
                else:
                    # Handle v2 model or other architectures
                    # This is a placeholder - actual generation logic may vary
                    predictions = self._generate_v2(model, cmd_tensor, mod_tensor, max_act_len)
                
                # Compare predictions with expected
                # Handle both tensor and numpy array cases
                if hasattr(predictions, 'numpy'):
                    predictions_np = predictions.numpy()
                else:
                    predictions_np = predictions
                    
                for i, (pred, expected) in enumerate(zip(predictions_np, expected_actions)):
                    # Remove padding and special tokens for comparison
                    pred_clean = self._clean_sequence(pred)
                    expected_clean = self._clean_sequence(expected[1:-1])  # Remove START/END
                    
                    if np.array_equal(pred_clean, expected_clean):
                        correct += 1
                    total += 1
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                total += len(batch_samples)
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def _clean_sequence(self, seq: np.ndarray) -> np.ndarray:
        """Remove padding and special tokens."""
        # Remove padding (0s)
        seq = seq[seq > 0]
        
        # Remove END token if present
        if len(seq) > 0 and seq[-1] == self.tokenizer.action_to_id.get('<END>', -1):
            seq = seq[:-1]
            
        return seq
    
    def _generate_v2(self, model, commands, modifications, max_length):
        """Placeholder for v2 model generation - needs implementation based on model."""
        # This would need to be implemented based on the specific v2 model architecture
        batch_size = commands.shape[0]
        return tf.zeros((batch_size, max_length), dtype=tf.int32)
    
    def evaluate_all_sets(self, 
                         model: keras.Model,
                         validation_dir: Path,
                         batch_size: int = 32) -> Dict[str, Dict[str, float]]:
        """Evaluate model on all validation sets."""
        
        # Define validation sets to evaluate
        val_sets = [
            'val_base',
            'val_mod_walk_skip', 
            'val_mod_jump_hop',
            'val_mod_look_scan',
            'val_mod_left_right',
            'val_mod_mixed',
            'val_mod_unseen',
            'val_mod_composed'
        ]
        
        all_results = {}
        
        print("\n" + "="*60)
        print("MODIFICATION-AWARE EVALUATION")
        print("="*60)
        
        for set_name in val_sets:
            print(f"\nEvaluating {set_name}...")
            
            try:
                # Load validation set
                samples = self.load_validation_set(set_name, validation_dir)
                
                # Evaluate
                results = self.evaluate_exact_match(
                    model, samples, batch_size=batch_size
                )
                
                all_results[set_name] = results
                
                # Print results
                print(f"  Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")
                
            except Exception as e:
                print(f"  ERROR: {e}")
                all_results[set_name] = {'error': str(e)}
        
        # Calculate aggregate metrics
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        # Base accuracy (no modifications)
        if 'val_base' in all_results and 'accuracy' in all_results['val_base']:
            print(f"Base accuracy: {all_results['val_base']['accuracy']:.2%}")
        
        # Average modification accuracy
        mod_accuracies = []
        for set_name in val_sets:
            if set_name != 'val_base' and set_name in all_results:
                if 'accuracy' in all_results[set_name]:
                    mod_accuracies.append(all_results[set_name]['accuracy'])
        
        if mod_accuracies:
            avg_mod_accuracy = np.mean(mod_accuracies)
            print(f"Average modification accuracy: {avg_mod_accuracy:.2%}")
            
            # Performance drop
            if 'val_base' in all_results and 'accuracy' in all_results['val_base']:
                drop = all_results['val_base']['accuracy'] - avg_mod_accuracy
                print(f"Performance drop on modifications: {drop:.2%}")
        
        return all_results


def load_model_with_config(model_path: Path, vocab_path: Path) -> Tuple[keras.Model, Dict]:
    """Load a model and its configuration."""
    
    # Load vocabulary
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    
    command_vocab_size = len(vocab_data['command_to_id'])
    action_vocab_size = len(vocab_data['action_to_id'])
    
    print(f"Vocabulary sizes - Command: {command_vocab_size}, Action: {action_vocab_size}")
    
    # Try to determine model version from path or config
    if 'v2' in str(model_path) or 'gated' in str(model_path):
        print("Loading as V2 model...")
        model = create_model_v2(
            command_vocab_size=command_vocab_size,
            action_vocab_size=action_vocab_size,
            d_model=128  # From training config
        )
    else:
        print("Loading as V1 model...")
        model = create_model(
            command_vocab_size=command_vocab_size,
            action_vocab_size=action_vocab_size,
            d_model=128  # From training config
        )
    
    # Load weights
    try:
        model.load_weights(str(model_path))
        print(f"Successfully loaded weights from {model_path}")
    except Exception as e:
        print(f"Warning: Could not load weights directly: {e}")
        print("Attempting to build model first...")
        
        # Build model with dummy inputs
        dummy_inputs = {
            'command': tf.zeros((1, 10), dtype=tf.int32),
            'target': tf.zeros((1, 10), dtype=tf.int32),
            'modification': tf.zeros((1, 5), dtype=tf.int32)
        }
        _ = model(dummy_inputs)
        
        # Try loading again
        model.load_weights(str(model_path))
    
    return model, vocab_data


def main():
    """Run comprehensive evaluation."""
    
    # Setup paths
    base_dir = Path(__file__).parent
    validation_dir = base_dir / 'data' / 'processed' / 'proper_validation_sets'
    # Use vocabulary from the same training run as the model
    vocab_path = base_dir / 'compositional_language_complete_20250722_185804' / 'outputs' / 'safeguarded_training' / 'vocabulary.json'
    
    # Model path - use the final model from safeguarded training
    model_path = base_dir / 'compositional_language_complete_20250722_185804' / 'outputs' / 'safeguarded_training' / 'final_model.h5'
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Please specify a valid model path.")
        return
    
    # Load tokenizer
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    
    tokenizer = SCANTokenizer(vocab_path)
    tokenizer.command_to_id = vocab_data['command_to_id']
    tokenizer.action_to_id = vocab_data['action_to_id']
    tokenizer.id_to_command = {v: k for k, v in vocab_data['command_to_id'].items()}
    tokenizer.id_to_action = {v: k for k, v in vocab_data['action_to_id'].items()}
    
    # Load model
    print(f"Loading model from {model_path}...")
    model, _ = load_model_with_config(model_path, vocab_path)
    
    # Create evaluator
    evaluator = ModificationAwareEvaluator(tokenizer)
    
    # Run evaluation
    results = evaluator.evaluate_all_sets(
        model=model,
        validation_dir=validation_dir,
        batch_size=32
    )
    
    # Save results
    output_path = base_dir / 'evaluation_v2_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Check if we've resolved the evaluation illusion
    print("\n" + "="*60)
    print("EVALUATION ILLUSION CHECK")
    print("="*60)
    
    if 'val_base' in results and 'accuracy' in results['val_base']:
        base_acc = results['val_base']['accuracy']
        print(f"Previous 'constant' validation accuracy: 84.3%")
        print(f"New base validation accuracy: {base_acc:.1%}")
        
        if abs(base_acc - 0.843) < 0.05:
            print("✓ Base accuracy similar - good sanity check")
        else:
            print("⚠ Base accuracy differs significantly - investigate")
    
    # Check modification performance
    mod_accs = [v['accuracy'] for k, v in results.items() 
                if k != 'val_base' and 'accuracy' in v]
    
    if mod_accs:
        print(f"\nModification accuracies range: {min(mod_accs):.1%} - {max(mod_accs):.1%}")
        if max(mod_accs) < 0.843:
            print("✓ Evaluation illusion confirmed: modifications perform worse!")
        else:
            print("⚠ Some modifications perform well - investigate further")


if __name__ == '__main__':
    main()