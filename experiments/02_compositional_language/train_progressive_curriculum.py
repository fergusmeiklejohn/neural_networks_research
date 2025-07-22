#!/usr/bin/env python3
"""
Progressive Training Curriculum for Compositional Language

Based on the successful physics experiment approach, this implements a 4-stage
progressive curriculum for learning compositional rules in SCAN.
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import sys
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import wandb
from tqdm import tqdm

# Import from current directory
from models import create_model
from scan_data_loader import SCANDataLoader


class SCANTokenizer:
    """Tokenizer for SCAN commands and actions"""
    
    def __init__(self, vocab_path: Optional[Path] = None):
        self.vocab_path = vocab_path
        self.command_to_id = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
        self.action_to_id = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
        self.id_to_command = {}
        self.id_to_action = {}
        
        if vocab_path and vocab_path.exists():
            self.load_vocabulary()
    
    def build_vocabulary(self, samples: List[Dict]):
        """Build vocabulary from samples"""
        command_words = set()
        action_words = set()
        
        for sample in samples:
            command_words.update(sample['command'].lower().split())
            action_words.update(sample['action'].split())
        
        # Add words to vocabulary
        for word in sorted(command_words):
            if word not in self.command_to_id:
                self.command_to_id[word] = len(self.command_to_id)
        
        for word in sorted(action_words):
            if word not in self.action_to_id:
                self.action_to_id[word] = len(self.action_to_id)
        
        # Create reverse mappings
        self.id_to_command = {v: k for k, v in self.command_to_id.items()}
        self.id_to_action = {v: k for k, v in self.action_to_id.items()}
    
    def encode_command(self, command: str, max_length: int = 50) -> np.ndarray:
        """Encode command to token ids"""
        tokens = command.lower().split()
        ids = [self.command_to_id.get(token, self.command_to_id['<UNK>']) for token in tokens]
        
        # Pad or truncate
        if len(ids) < max_length:
            ids.extend([self.command_to_id['<PAD>']] * (max_length - len(ids)))
        else:
            ids = ids[:max_length]
        
        return np.array(ids, dtype=np.int32)
    
    def encode_action(self, action: str, max_length: int = 100) -> np.ndarray:
        """Encode action to token ids with START and END tokens"""
        tokens = ['<START>'] + action.split() + ['<END>']
        ids = [self.action_to_id.get(token, self.action_to_id['<UNK>']) for token in tokens]
        
        # Pad or truncate
        if len(ids) < max_length:
            ids.extend([self.action_to_id['<PAD>']] * (max_length - len(ids)))
        else:
            ids = ids[:max_length-1] + [self.action_to_id['<END>']]
        
        return np.array(ids, dtype=np.int32)
    
    def decode_action(self, ids: np.ndarray) -> str:
        """Decode action ids to string"""
        tokens = []
        for id in ids:
            if id == self.action_to_id['<PAD>']:
                break
            if id == self.action_to_id['<START>']:
                continue
            if id == self.action_to_id['<END>']:
                break
            tokens.append(self.id_to_action.get(id, '<UNK>'))
        return ' '.join(tokens)
    
    def save_vocabulary(self, path: Path):
        """Save vocabulary to file"""
        vocab_data = {
            'command_to_id': self.command_to_id,
            'action_to_id': self.action_to_id
        }
        with open(path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
    
    def load_vocabulary(self):
        """Load vocabulary from file"""
        with open(self.vocab_path, 'r') as f:
            vocab_data = json.load(f)
        self.command_to_id = vocab_data['command_to_id']
        self.action_to_id = vocab_data['action_to_id']
        self.id_to_command = {v: k for k, v in self.command_to_id.items()}
        self.id_to_action = {v: k for k, v in self.action_to_id.items()}


class ProgressiveCurriculum:
    """Progressive training curriculum for compositional language"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.current_stage = 0
        self.stage_configs = {
            1: {
                "name": "Stage 1: Basic Compositional Learning",
                "description": "Learn standard SCAN mappings",
                "use_modifications": False,
                "epochs": config.get('stage1_epochs', 50),
                "lr": config.get('stage1_lr', 1e-3)
            },
            2: {
                "name": "Stage 2: Simple Modifications",
                "description": "Introduce single word swaps",
                "use_modifications": True,
                "modification_types": ['simple_swaps'],
                "modification_ratio": 0.3,
                "epochs": config.get('stage2_epochs', 50),
                "lr": config.get('stage2_lr', 5e-4)
            },
            3: {
                "name": "Stage 3: Complex Modifications",
                "description": "Multi-word and structural changes",
                "use_modifications": True,
                "modification_types": ['simple_swaps', 'action_modifications', 'structural'],
                "modification_ratio": 0.5,
                "epochs": config.get('stage3_epochs', 50),
                "lr": config.get('stage3_lr', 2e-4)
            },
            4: {
                "name": "Stage 4: Novel Generation",
                "description": "Focus on unseen combinations",
                "use_modifications": True,
                "modification_types": 'all',
                "modification_ratio": 0.7,
                "focus_extrapolation": True,
                "epochs": config.get('stage4_epochs', 50),
                "lr": config.get('stage4_lr', 1e-4)
            }
        }
    
    def get_stage_config(self, stage: int) -> Dict:
        """Get configuration for a specific stage"""
        return self.stage_configs.get(stage, {})


def create_dataset(samples: List[Dict], 
                   tokenizer: SCANTokenizer,
                   modifications: Optional[List[Dict]] = None,
                   modification_ratio: float = 0.0,
                   batch_size: int = 32) -> tf.data.Dataset:
    """Create TensorFlow dataset from samples"""
    
    # Prepare data
    commands = []
    actions = []
    has_modification = []
    modification_commands = []
    
    for sample in samples:
        command_enc = tokenizer.encode_command(sample['command'])
        action_enc = tokenizer.encode_action(sample['action'])
        
        commands.append(command_enc)
        actions.append(action_enc)
        has_modification.append(0)  # No modification
        modification_commands.append(np.zeros(20, dtype=np.int32))  # Dummy
    
    # Add modifications if requested
    if modifications and modification_ratio > 0:
        num_modifications = int(len(samples) * modification_ratio)
        selected_mods = np.random.choice(modifications, size=min(num_modifications, len(modifications)), replace=False)
        
        for mod in selected_mods:
            # Original sample
            orig_command_enc = tokenizer.encode_command(mod['original']['command'])
            orig_action_enc = tokenizer.encode_action(mod['original']['action'])
            
            # Modified sample
            mod_command_enc = tokenizer.encode_command(mod['modified']['command'])
            mod_action_enc = tokenizer.encode_action(mod['modified']['action'])
            
            # Modification description (simplified for now)
            mod_desc = tokenizer.encode_command(mod['modification_description'], max_length=20)
            
            commands.append(orig_command_enc)
            actions.append(mod_action_enc)
            has_modification.append(1)
            modification_commands.append(mod_desc)
    
    # Convert to numpy arrays
    commands = np.array(commands)
    actions = np.array(actions)
    has_modification = np.array(has_modification)
    modification_commands = np.array(modification_commands)
    
    # Create dataset with proper format for model.fit()
    # The model expects inputs as a dict and targets as the shifted action sequence
    def prepare_for_training(data):
        # Extract the action sequence for teacher forcing
        # Input: action[:-1], Target: action[1:]
        action = data['action']
        
        # Prepare inputs dict for the model
        inputs = {
            'command': data['command'],
            'target': action[:, :-1],  # All tokens except the last
            'modification': data['modification']
        }
        
        # Targets are the shifted action sequence
        targets = action[:, 1:]  # All tokens except the first
        
        return inputs, targets
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices({
        'command': commands,
        'action': actions,
        'has_modification': has_modification,
        'modification': modification_commands
    })
    
    # Shuffle and batch
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    
    # Map to prepare for training
    dataset = dataset.map(prepare_for_training, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def compute_accuracy(model, dataset, tokenizer, max_samples=1000):
    """Compute exact match accuracy on dataset"""
    correct = 0
    total = 0
    
    for batch_inputs, batch_targets in dataset:
        if total >= max_samples:
            break
            
        # Generate predictions
        commands = batch_inputs['command']
        batch_size = tf.shape(commands)[0]
        
        # Get model predictions
        generated = model.generate_action(
            commands, 
            modification=batch_inputs.get('modification', None),
            start_token=tokenizer.action_to_id['<START>'],
            end_token=tokenizer.action_to_id['<END>']
        )
        
        # Reconstruct full action sequences from targets
        # targets are action[:, 1:], so we need to add START token back
        start_tokens = tf.fill([batch_size, 1], tokenizer.action_to_id['<START>'])
        full_targets = tf.concat([start_tokens, batch_targets], axis=1)
        
        for i in range(batch_size):
            pred_str = tokenizer.decode_action(generated[i].numpy())
            target_str = tokenizer.decode_action(full_targets[i].numpy())
            
            if pred_str == target_str:
                correct += 1
            total += 1
    
    return correct / total if total > 0 else 0.0


def train_progressive_curriculum(config: Dict):
    """Main training function"""
    
    # Setup
    print("Setting up progressive curriculum training...")
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb if enabled
    if config.get('use_wandb', False):
        wandb.init(
            project=config.get('wandb_project', 'compositional-language'),
            name=f"progressive_curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config
        )
    
    # Load data
    print("\nLoading data...")
    loader = SCANDataLoader(data_dir='data')
    splits = loader.load_processed_splits()
    
    # Load modifications
    mod_path = Path('data/processed/modification_pairs.pkl')
    if mod_path.exists():
        with open(mod_path, 'rb') as f:
            modifications = pickle.load(f)
    else:
        modifications = []
    
    # Create tokenizer
    print("Building vocabulary...")
    tokenizer = SCANTokenizer()
    tokenizer.build_vocabulary(splits['train'])
    tokenizer.save_vocabulary(output_dir / 'vocabulary.json')
    
    print(f"Command vocabulary size: {len(tokenizer.command_to_id)}")
    print(f"Action vocabulary size: {len(tokenizer.action_to_id)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        command_vocab_size=len(tokenizer.command_to_id),
        action_vocab_size=len(tokenizer.action_to_id),
        d_model=config.get('d_model', 256)
    )
    
    # Build the model with dummy inputs to initialize weights
    print("Building model...")
    dummy_command = tf.zeros((1, 50), dtype=tf.int32)
    dummy_target = tf.zeros((1, 99), dtype=tf.int32)
    dummy_modification = tf.zeros((1, 20), dtype=tf.int32)
    
    # Call model to build it
    _ = model({
        'command': dummy_command,
        'target': dummy_target,
        'modification': dummy_modification
    }, training=False)
    
    # Try to count parameters, but don't fail if we can't
    try:
        total_params = sum(tf.size(v).numpy() for v in model.trainable_variables)
        print(f"Model built with {total_params:,} parameters")
    except:
        print("Model built successfully")
    
    # Initialize curriculum
    curriculum = ProgressiveCurriculum(config)
    
    # Training loop
    for stage in range(1, 5):
        stage_config = curriculum.get_stage_config(stage)
        print(f"\n{'='*60}")
        print(f"{stage_config['name']}")
        print(f"{stage_config['description']}")
        print(f"{'='*60}")
        
        # Create stage-specific dataset
        if stage_config.get('use_modifications', False):
            dataset = create_dataset(
                splits['train'],
                tokenizer,
                modifications=modifications,
                modification_ratio=stage_config.get('modification_ratio', 0.0),
                batch_size=config['batch_size']
            )
        else:
            dataset = create_dataset(
                splits['train'],
                tokenizer,
                batch_size=config['batch_size']
            )
        
        # Create validation datasets
        val_dataset = create_dataset(
            splits['val_interpolation'],
            tokenizer,
            batch_size=config['batch_size']
        )
        
        # Setup optimizer and loss
        optimizer = keras.optimizers.Adam(learning_rate=stage_config['lr'])
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        # Training epochs
        for epoch in range(stage_config['epochs']):
            print(f"\nEpoch {epoch + 1}/{stage_config['epochs']}")
            
            # Training step
            train_loss = 0
            num_batches = 0
            
            for batch_inputs, batch_targets in tqdm(dataset, desc="Training"):
                with tf.GradientTape() as tape:
                    # Forward pass
                    outputs = model(batch_inputs, training=True)
                    
                    # Compute loss
                    logits = outputs['logits']
                    
                    # Mask padding
                    mask = tf.cast(batch_targets != tokenizer.action_to_id['<PAD>'], tf.float32)
                    loss = loss_fn(batch_targets, logits) * mask
                    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
                    
                    train_loss += loss.numpy()
                    num_batches += 1
                
                # Backward pass
                gradients = tape.gradient(loss, model.trainable_variables)
                
                # Filter out None gradients (from unused components)
                valid_gradients_and_vars = [
                    (grad, var) for grad, var in zip(gradients, model.trainable_variables)
                    if grad is not None
                ]
                
                optimizer.apply_gradients(valid_gradients_and_vars)
            
            # Validation
            if epoch % 5 == 0:
                val_acc = compute_accuracy(model, val_dataset, tokenizer, max_samples=100)
                print(f"  Train Loss: {train_loss/num_batches:.4f}")
                print(f"  Val Accuracy: {val_acc:.2%}")
                
                if config.get('use_wandb', False):
                    wandb.log({
                        f'stage_{stage}/train_loss': train_loss/num_batches,
                        f'stage_{stage}/val_accuracy': val_acc,
                        'epoch': sum([curriculum.get_stage_config(s)['epochs'] for s in range(1, stage)]) + epoch
                    })
        
        # Save stage checkpoint
        checkpoint_path = output_dir / f"stage_{stage}_model.keras"
        model.save_weights(str(checkpoint_path))
        print(f"Saved checkpoint: {checkpoint_path}")
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Test on different splits
    test_results = {}
    
    for split_name in ['test_interpolation', 'test_primitive_extrap']:
        if split_name in splits:
            test_dataset = create_dataset(
                splits[split_name],
                tokenizer,
                batch_size=config['batch_size']
            )
            accuracy = compute_accuracy(model, test_dataset, tokenizer)
            test_results[split_name] = accuracy
            print(f"{split_name}: {accuracy:.2%}")
    
    # Save results
    results_path = output_dir / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'config': config,
            'test_results': test_results,
            'vocabulary_sizes': {
                'command': len(tokenizer.command_to_id),
                'action': len(tokenizer.action_to_id)
            }
        }, f, indent=2)
    
    if config.get('use_wandb', False):
        wandb.finish()
    
    print(f"\nTraining complete! Results saved to {output_dir}")


def main():
    """Run progressive curriculum training"""
    
    config = {
        # Model parameters
        'd_model': 256,
        'batch_size': 32,
        
        # Stage-specific parameters
        'stage1_epochs': 2,  # Reduced for testing
        'stage2_epochs': 2,
        'stage3_epochs': 2,
        'stage4_epochs': 2,
        'stage1_lr': 1e-3,
        'stage2_lr': 5e-4,
        'stage3_lr': 2e-4,
        'stage4_lr': 1e-4,
        
        # Output
        'output_dir': 'outputs/progressive_curriculum',
        'use_wandb': False  # Set to True for full runs
    }
    
    train_progressive_curriculum(config)


if __name__ == "__main__":
    main()