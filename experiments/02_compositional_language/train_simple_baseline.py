#!/usr/bin/env python3
"""
Simple Baseline Model for Compositional Language

This implements a straightforward sequence-to-sequence model without architectural
complexity. The goal is to establish a true baseline for the compositional
language task that:

1. Avoids complex gating mechanisms
2. Trains on mixed data from the start
3. Uses proper validation sets for evaluation
4. Serves as a comparison point for more complex architectures

Key simplifications:
- Basic LSTM encoder-decoder architecture
- Modifications handled through simple concatenation
- No separate rule extraction/modification components
- Single training phase (no curriculum)
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime
import sys

# Import from current directory
from evaluation_v2 import ModificationAwareEvaluator
from train_progressive_curriculum import SCANTokenizer


class SimpleSeq2SeqModel(keras.Model):
    """
    Simple LSTM-based sequence-to-sequence model.
    
    Architecture:
    1. Embed command + modification (concatenated)
    2. LSTM encoder
    3. LSTM decoder with attention
    4. Output projection
    """
    
    def __init__(self, 
                 command_vocab_size: int,
                 action_vocab_size: int,
                 d_model: int = 128,
                 num_layers: int = 2,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.command_vocab_size = command_vocab_size
        self.action_vocab_size = action_vocab_size
        self.d_model = d_model
        
        # Embeddings
        self.command_embedding = keras.layers.Embedding(command_vocab_size, d_model)
        self.modification_embedding = keras.layers.Dense(d_model, activation='tanh')
        
        # Encoder
        self.encoder_lstm = keras.layers.LSTM(
            d_model, 
            return_sequences=True,
            return_state=True,
            dropout=dropout_rate
        )
        
        # Decoder
        self.decoder_lstm = keras.layers.LSTM(
            d_model,
            return_sequences=True,
            return_state=True,
            dropout=dropout_rate
        )
        
        # Simple attention
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=4,
            key_dim=d_model // 4,
            dropout=dropout_rate
        )
        
        # Output projection
        self.output_projection = keras.layers.Dense(action_vocab_size)
        
        # Dropout
        self.dropout = keras.layers.Dropout(dropout_rate)
        
    def encode(self, command_ids, modification=None, training=None):
        """Encode command and optional modification."""
        # Embed command
        command_embedded = self.command_embedding(command_ids)
        command_embedded = self.dropout(command_embedded, training=training)
        
        # If modification provided, add it to embeddings
        if modification is not None:
            # Project modification to embedding space
            mod_signal = self.modification_embedding(modification)
            mod_signal = tf.expand_dims(mod_signal, axis=1)  # Add sequence dimension
            
            # Simple addition (no gating)
            command_embedded = command_embedded + mod_signal
        
        # Encode with LSTM
        encoder_output, state_h, state_c = self.encoder_lstm(
            command_embedded, training=training
        )
        
        return encoder_output, [state_h, state_c]
    
    def decode_step(self, decoder_input, encoder_output, decoder_state, training=None):
        """Single decoder step with attention."""
        # LSTM step
        lstm_output, new_h, new_c = self.decoder_lstm(
            decoder_input, 
            initial_state=decoder_state,
            training=training
        )
        
        # Attention over encoder output
        attended_output = self.attention(
            query=lstm_output,
            value=encoder_output,
            key=encoder_output,
            training=training
        )
        
        # Combine LSTM output and attention
        combined = lstm_output + attended_output
        
        # Project to vocabulary
        logits = self.output_projection(combined)
        
        return logits, [new_h, new_c]
    
    def call(self, inputs, training=None):
        """Forward pass for training."""
        command_ids = inputs['commands']
        action_ids = inputs['actions']
        modification = inputs.get('modification', None)
        
        # Encode
        encoder_output, encoder_state = self.encode(
            command_ids, modification, training=training
        )
        
        # Prepare decoder inputs (shift right with BOS token)
        batch_size = tf.shape(action_ids)[0]
        bos_tokens = tf.zeros([batch_size, 1], dtype=action_ids.dtype)
        decoder_inputs = tf.concat([bos_tokens, action_ids[:, :-1]], axis=1)
        
        # Embed decoder inputs
        decoder_embedded = self.command_embedding(decoder_inputs)  # Reuse embeddings
        
        # Decode all at once for training
        # Use decoder LSTM directly for simplicity during training
        decoder_output, _, _ = self.decoder_lstm(
            decoder_embedded, 
            initial_state=encoder_state,
            training=training
        )
        
        # Apply attention
        attended_output = self.attention(
            query=decoder_output,
            value=encoder_output,
            key=encoder_output,
            training=training
        )
        
        # Combine and project
        combined = decoder_output + attended_output
        logits = self.output_projection(combined)
        
        return logits
    
    def generate_action(self, command_ids, modification=None, max_length=50):
        """Generate action sequence given command."""
        # Encode
        encoder_output, decoder_state = self.encode(command_ids, modification, training=False)
        
        # Start with BOS token
        batch_size = tf.shape(command_ids)[0]
        current_token = tf.zeros([batch_size, 1], dtype=tf.int32)
        
        generated_tokens = []
        
        for _ in range(max_length):
            # Embed current token
            token_embedded = self.command_embedding(current_token)
            
            # Decode step
            logits, decoder_state = self.decode_step(
                token_embedded, encoder_output, decoder_state, training=False
            )
            
            # Get next token (greedy)
            next_token = tf.argmax(logits[:, -1, :], axis=-1)
            next_token = tf.expand_dims(next_token, axis=1)
            
            generated_tokens.append(next_token)
            
            # Stop if all sequences have generated EOS (assuming EOS=0)
            if tf.reduce_all(next_token == 0):
                break
                
            current_token = next_token
        
        # Concatenate all tokens
        if generated_tokens:
            return tf.concat(generated_tokens, axis=1)
        else:
            return tf.zeros([batch_size, 1], dtype=tf.int32)


def create_mixed_dataset(data_path: Path, tokenizer: SCANTokenizer, 
                        batch_size: int = 32, include_modifications: bool = True):
    """Create dataset mixing base and modified examples."""
    # Load base training data
    with open(data_path / 'train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    # Load modification pairs if requested
    all_examples = []
    
    # Add base examples
    for sample in train_data:
        # Handle both dict and object formats
        if isinstance(sample, dict):
            command = sample['command']
            action = sample['action']
        else:
            command = sample.command
            action = sample.action
            
        all_examples.append({
            'command': command,
            'action': action,
            'modification': None
        })
    
    if include_modifications:
        # Load modification pairs
        mod_path = data_path / 'modification_pairs.pkl'
        if mod_path.exists():
            with open(mod_path, 'rb') as f:
                mod_pairs = pickle.load(f)
            
            # Add modified examples
            for item in mod_pairs[:len(train_data) // 2]:
                if len(item) == 3:
                    base, modified, mod_type = item
                else:
                    # Handle different formats
                    continue
                    
                # Handle both dict and object formats
                if isinstance(base, dict):
                    base_command = base['command']
                else:
                    base_command = base.command
                    
                if isinstance(modified, dict):
                    modified_action = modified['action']
                else:
                    modified_action = modified.action
                
                all_examples.append({
                    'command': base_command,
                    'action': modified_action,
                    'modification': mod_type
                })
    
    # Shuffle examples
    np.random.shuffle(all_examples)
    
    # Create TF dataset
    def generator():
        for example in all_examples:
            # Tokenize
            command_ids = tokenizer.encode_command(example['command'])
            action_ids = tokenizer.encode_action(example['action'])
            
            # Create modification vector
            if example['modification'] is not None:
                # Simple one-hot encoding for modifications
                mod_types = ['walk_skip', 'jump_hop', 'look_scan', 'left_right']
                mod_vector = np.zeros(8, dtype=np.float32)
                if example['modification'] in mod_types:
                    mod_vector[mod_types.index(example['modification'])] = 1.0
            else:
                mod_vector = np.zeros(8, dtype=np.float32)
            
            yield {
                'commands': command_ids,
                'actions': action_ids,
                'modification': mod_vector
            }
    
    # Create dataset with proper types
    output_signature = {
        'commands': tf.TensorSpec(shape=(None,), dtype=tf.int32),
        'actions': tf.TensorSpec(shape=(None,), dtype=tf.int32),
        'modification': tf.TensorSpec(shape=(8,), dtype=tf.float32)
    }
    
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    
    # Pad and batch
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes={
            'commands': [None],
            'actions': [None],
            'modification': [8]
        },
        padding_values={
            'commands': 0,
            'actions': 0,
            'modification': 0.0
        }
    )
    
    return dataset, len(all_examples)


def train_simple_baseline(args):
    """Train the simple baseline model."""
    print("Training Simple Baseline Model")
    print("=" * 50)
    
    # Setup paths
    base_dir = Path(__file__).parent
    data_path = base_dir / 'data' / 'processed'
    output_dir = base_dir / 'outputs' / f'simple_baseline_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to use vocabulary from a trained model first, fallback to general vocabulary
    trained_model_vocab = base_dir / 'compositional_language_complete_20250722_185804' / 'outputs' / 'safeguarded_training' / 'vocabulary.json'
    if trained_model_vocab.exists():
        vocab_path = trained_model_vocab
        print(f"Using vocabulary from trained model: {vocab_path}")
    else:
        vocab_path = data_path / 'vocabulary.json'
        print(f"Using default vocabulary: {vocab_path}")
    
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    
    # Initialize tokenizer with None (we'll set vocabulary manually)
    tokenizer = SCANTokenizer(None)
    
    # Set vocabulary based on format
    if 'command_to_id' in vocab_data:
        tokenizer.command_to_id = vocab_data['command_to_id']
        tokenizer.action_to_id = vocab_data['action_to_id']
    else:
        # Build from lists
        tokenizer.command_to_id = {token: i for i, token in enumerate(vocab_data['command_vocab'])}
        tokenizer.action_to_id = {token: i for i, token in enumerate(vocab_data['action_vocab'])}
    
    # Build reverse mappings
    tokenizer.id_to_command = {v: k for k, v in tokenizer.command_to_id.items()}
    tokenizer.id_to_action = {v: k for k, v in tokenizer.action_to_id.items()}
    
    command_vocab_size = len(tokenizer.command_to_id)
    action_vocab_size = len(tokenizer.action_to_id)
    
    print(f"Command vocabulary size: {command_vocab_size}")
    print(f"Action vocabulary size: {action_vocab_size}")
    
    # Create model
    model = SimpleSeq2SeqModel(
        command_vocab_size=command_vocab_size,
        action_vocab_size=action_vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate
    )
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Create datasets
    print("\nCreating mixed dataset...")
    train_dataset, num_train = create_mixed_dataset(
        data_path, tokenizer, args.batch_size, 
        include_modifications=args.mixed_from_start
    )
    
    print(f"Training examples: {num_train}")
    print(f"Include modifications: {args.mixed_from_start}")
    
    # Training loop with periodic evaluation
    history = {'loss': [], 'accuracy': [], 'val_results': []}
    
    # Initialize evaluator for proper validation
    if args.use_proper_validation:
        print("\nInitializing proper validation...")
        validation_dir = data_path / 'proper_validation_sets'
        evaluator = ModificationAwareEvaluator(tokenizer)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train for one epoch
        epoch_history = model.fit(
            train_dataset,
            epochs=1,
            verbose=1
        )
        
        history['loss'].append(epoch_history.history['loss'][0])
        history['accuracy'].append(epoch_history.history['accuracy'][0])
        
        # Evaluate on proper validation sets every few epochs
        if args.use_proper_validation and (epoch + 1) % 5 == 0:
            print("\nEvaluating on proper validation sets...")
            val_results = evaluator.evaluate_all_sets(model, validation_dir)
            history['val_results'].append({
                'epoch': epoch + 1,
                'results': val_results
            })
            
            # Print summary
            print(f"\nValidation Results (Epoch {epoch + 1}):")
            print(f"Base accuracy: {val_results.get('val_base', {}).get('accuracy', 0):.4f}")
            if 'aggregate_metrics' in val_results:
                metrics = val_results['aggregate_metrics']
                print(f"Average modification accuracy: {metrics.get('average_modification_accuracy', 0):.4f}")
                print(f"Performance drop: {metrics.get('performance_drop', 0):.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch + 1}.h5'
            model.save_weights(str(checkpoint_path))
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_model_path = output_dir / 'final_model.h5'
    model.save_weights(str(final_model_path))
    print(f"\nSaved final model: {final_model_path}")
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save model configuration
    config = {
        'model_type': 'simple_baseline',
        'command_vocab_size': command_vocab_size,
        'action_vocab_size': action_vocab_size,
        'd_model': args.d_model,
        'num_layers': args.num_layers,
        'dropout_rate': args.dropout_rate,
        'mixed_from_start': args.mixed_from_start,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate
    }
    
    config_path = output_dir / 'model_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Final evaluation on all validation sets
    if args.use_proper_validation:
        print("\nFinal evaluation on all validation sets...")
        final_results = evaluator.evaluate_all_sets(model, validation_dir)
        
        results_path = output_dir / 'final_evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print("\nFinal Results Summary:")
        print("=" * 50)
        for set_name, results in final_results.items():
            if isinstance(results, dict) and 'accuracy' in results:
                print(f"{set_name}: {results['accuracy']:.4f}")
        
        if 'aggregate_metrics' in final_results:
            print("\nAggregate Metrics:")
            metrics = final_results['aggregate_metrics']
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")
    
    print(f"\nTraining complete! Results saved to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Train simple baseline model')
    parser.add_argument('--d_model', type=int, default=128,
                       help='Model dimension (default: 128)')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of LSTM layers (default: 2)')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                       help='Dropout rate (default: 0.1)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--mixed_from_start', action='store_true',
                       help='Include modifications in training from start')
    parser.add_argument('--use_proper_validation', action='store_true', default=True,
                       help='Use proper validation sets (default: True)')
    
    args = parser.parse_args()
    
    # Run training
    train_simple_baseline(args)


if __name__ == '__main__':
    main()