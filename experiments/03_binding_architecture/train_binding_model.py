"""
Training script for Minimal Binding Model

This script trains the binding model on dereferencing tasks to force
explicit variable binding, then tests modification capabilities.
"""

from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
from utils.paths import get_output_path
import numpy as np
import keras
from keras import ops
import logging
import json
import os
from datetime import datetime

from minimal_binding_scan import MinimalBindingModel
from dereferencing_tasks import DereferencingTaskGenerator

# Set up environment
config = setup_environment()
logger = logging.getLogger(__name__)


class BindingTrainer:
    """Handles training of the binding model."""
    
    def __init__(
        self,
        model: MinimalBindingModel,
        generator: DereferencingTaskGenerator,
        output_dir: str
    ):
        self.model = model
        self.generator = generator
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up optimizer and loss
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        # Metrics
        self.train_loss_metric = keras.metrics.Mean()
        self.train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        
    @keras.utils.function  
    def train_step(self, commands, actions):
        """Single training step."""
        with keras.GradientTape() as tape:
            # Forward pass
            outputs = self.model({'command': commands}, training=True)
            action_logits = outputs['action_logits']
            
            # Compute loss
            loss = self.loss_fn(actions, action_logits)
            
        # Compute gradients and update
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply(gradients, self.model.trainable_variables)
        
        # Update metrics
        self.train_loss_metric.update_state(loss)
        self.train_acc_metric.update_state(actions, action_logits)
        
        return loss, outputs['bindings']
    
    def evaluate(self, dataset, split_name='val'):
        """Evaluate model on dataset."""
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        commands = dataset['commands']
        actions = dataset['actions']
        
        # Process in batches
        batch_size = 32
        n_batches = (len(commands) + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(commands))
            
            batch_commands = commands[start_idx:end_idx]
            batch_actions = actions[start_idx:end_idx]
            
            # Forward pass
            outputs = self.model({'command': batch_commands}, training=False)
            action_logits = outputs['action_logits']
            
            # Compute loss and accuracy
            loss = self.loss_fn(batch_actions, action_logits)
            predictions = ops.argmax(action_logits, axis=-1)
            
            # Count correct predictions (all actions must match)
            for j in range(len(batch_commands)):
                pred = predictions[j]
                target = batch_actions[j]
                
                # Find non-padding positions
                non_pad = target != 0
                
                if ops.all(pred[non_pad] == target[non_pad]):
                    total_correct += 1
                    
            total_loss += float(loss) * len(batch_commands)
            total_samples += len(batch_commands)
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def test_modification_capability(self):
        """Test if model can handle modifications."""
        test_cases = self.generator.generate_modification_test_set()
        
        results = []
        
        for test_case in test_cases:
            # Test original
            orig_cmd = self.generator.encode_words(test_case['original']['command'])
            orig_act = self.generator.encode_actions(test_case['original']['actions'])
            
            orig_cmd = np.expand_dims(orig_cmd, 0)  # Add batch dimension
            outputs = self.model({'command': orig_cmd}, training=False)
            orig_pred = ops.argmax(outputs['action_logits'], axis=-1)
            
            # Test modified  
            mod_cmd = self.generator.encode_words(test_case['modified']['command'])
            mod_act = self.generator.encode_actions(test_case['modified']['actions'])
            
            mod_cmd = np.expand_dims(mod_cmd, 0)
            outputs = self.model({'command': mod_cmd}, training=False)
            mod_pred = ops.argmax(outputs['action_logits'], axis=-1)
            
            # Check if predictions match expected
            orig_correct = ops.all(orig_pred[0, :len(orig_act)] == orig_act)
            mod_correct = ops.all(mod_pred[0, :len(mod_act)] == mod_act)
            
            results.append({
                'modification_type': test_case['modification_type'],
                'original_correct': bool(orig_correct),
                'modified_correct': bool(mod_correct),
                'original_command': ' '.join(test_case['original']['command']),
                'modified_command': ' '.join(test_case['modified']['command'])
            })
            
        return results
    
    def train(self, n_epochs: int = 50, batch_size: int = 32):
        """Full training loop."""
        logger.info("Generating training data...")
        dataset = self.generator.generate_dataset(n_samples=1000)
        
        train_data = dataset['train']
        val_data = dataset['val']
        
        logger.info(f"Train samples: {len(train_data['commands'])}")
        logger.info(f"Val samples: {len(val_data['commands'])}")
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'modification_results': []
        }
        
        # Training loop
        for epoch in range(n_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{n_epochs}")
            
            # Reset metrics
            self.train_loss_metric.reset_state()
            self.train_acc_metric.reset_state()
            
            # Training
            n_batches = (len(train_data['commands']) + batch_size - 1) // batch_size
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(train_data['commands']))
                
                batch_commands = train_data['commands'][start_idx:end_idx]
                batch_actions = train_data['actions'][start_idx:end_idx]
                
                loss, bindings = self.train_step(batch_commands, batch_actions)
                
                if batch_idx % 10 == 0:
                    logger.info(
                        f"Batch {batch_idx}/{n_batches} - "
                        f"Loss: {float(self.train_loss_metric.result()):.4f}, "
                        f"Acc: {float(self.train_acc_metric.result()):.4f}"
                    )
            
            # Validation
            val_loss, val_acc = self.evaluate(val_data, 'val')
            
            # Record history
            history['train_loss'].append(float(self.train_loss_metric.result()))
            history['train_acc'].append(float(self.train_acc_metric.result()))
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            logger.info(
                f"Epoch {epoch + 1} - "
                f"Train Loss: {history['train_loss'][-1]:.4f}, "
                f"Train Acc: {history['train_acc'][-1]:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.4f}"
            )
            
            # Test modification capability every 10 epochs
            if (epoch + 1) % 10 == 0:
                logger.info("\nTesting modification capability...")
                mod_results = self.test_modification_capability()
                
                # Calculate modification success rate
                n_correct = sum(r['modified_correct'] for r in mod_results)
                mod_success_rate = n_correct / len(mod_results) if mod_results else 0
                
                logger.info(f"Modification success rate: {mod_success_rate:.2%}")
                
                for result in mod_results[:2]:  # Show first 2 examples
                    logger.info(f"  {result['original_command']} -> {result['modified_command']}")
                    logger.info(f"  Original: {'✓' if result['original_correct'] else '✗'}, "
                              f"Modified: {'✓' if result['modified_correct'] else '✗'}")
                
                history['modification_results'].append({
                    'epoch': epoch + 1,
                    'success_rate': mod_success_rate,
                    'details': mod_results
                })
            
            # Save checkpoint
            if (epoch + 1) % 25 == 0:
                checkpoint_path = os.path.join(
                    self.output_dir, 
                    f'binding_model_epoch_{epoch + 1}.keras'
                )
                self.model.save(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save final model and history
        final_model_path = os.path.join(self.output_dir, 'binding_model_final.keras')
        self.model.save(final_model_path)
        
        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"\nTraining complete!")
        logger.info(f"Final model saved to: {final_model_path}")
        logger.info(f"History saved to: {history_path}")
        
        return history


def main():
    """Main training script."""
    logger.info("Starting Minimal Binding Model training...")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = get_output_path() / f"binding_model_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    generator = DereferencingTaskGenerator(seed=42)
    
    model = MinimalBindingModel(
        vocab_size=len(generator.word_vocab),
        action_vocab_size=len(generator.action_vocab),
        n_slots=10,
        embed_dim=128,
        hidden_dim=256
    )
    
    # Build model with dummy input
    dummy_input = {'command': np.zeros((1, 10), dtype=np.int32)}
    _ = model(dummy_input)
    
    logger.info(f"Model initialized with {model.count_params()} parameters")
    
    # Create trainer
    trainer = BindingTrainer(model, generator, str(output_dir))
    
    # Train model
    history = trainer.train(n_epochs=50, batch_size=32)
    
    # Final modification test
    logger.info("\nFinal modification capability test:")
    final_results = trainer.test_modification_capability()
    
    n_correct = sum(r['modified_correct'] for r in final_results)
    final_success_rate = n_correct / len(final_results) if final_results else 0
    
    logger.info(f"Final modification success rate: {final_success_rate:.2%}")
    
    # Save final results
    results_path = os.path.join(output_dir, 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'final_modification_success_rate': final_success_rate,
            'modification_test_results': final_results,
            'final_val_accuracy': history['val_acc'][-1] if history['val_acc'] else 0
        }, f, indent=2)
    
    logger.info(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()