#!/usr/bin/env python3
"""
Paperspace training script with comprehensive safeguards from physics experiments.
Ensures results are preserved even with auto-shutdown.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

# Set backend before any keras imports
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def setup_environment():
    """Setup the environment and paths"""
    print("Setting up environment...")
    
    # Detect base path
    if os.path.exists('/notebooks/neural_networks_research'):
        base_path = '/notebooks/neural_networks_research'
    elif os.path.exists('/workspace/neural_networks_research'):
        base_path = '/workspace/neural_networks_research'
    else:
        base_path = os.path.abspath('../..')
    
    # Change to experiment directory
    exp_dir = os.path.join(base_path, 'experiments/02_compositional_language')
    os.chdir(exp_dir)
    
    print(f"Working directory: {os.getcwd()}")
    
    # Verify persistent storage
    if os.path.exists('/storage'):
        print("✓ Persistent storage available at /storage")
        # Create experiment folder in storage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        storage_dir = f'/storage/compositional_language_{timestamp}'
        os.makedirs(storage_dir, exist_ok=True)
        print(f"✓ Created storage directory: {storage_dir}")
    else:
        print("⚠️  WARNING: No persistent storage found! Results may be lost!")
        storage_dir = None
    
    return exp_dir, storage_dir

def generate_data():
    """Generate SCAN data and modifications"""
    print("\n" + "="*60)
    print("STEP 1: Generating SCAN Dataset")
    print("="*60)
    
    # Check if data already exists
    data_path = Path("data/processed/train.pkl")
    if data_path.exists():
        print("Processed data already exists. Skipping generation.")
        return True
    
    print("Generating SCAN data...")
    
    # Import and run data loader
    from scan_data_loader import SCANDataLoader
    
    loader = SCANDataLoader(data_dir='data')
    
    # Load all data
    print("Downloading SCAN dataset...")
    all_data = loader.load_all_data()
    
    # Create isolated splits
    print("Creating train/test splits with proper isolation...")
    splits = loader.create_isolated_splits()
    
    # Save processed data
    print("Saving processed data...")
    loader.save_processed_data(splits)
    
    # Generate modifications
    print("\nGenerating modification pairs...")
    from modification_generator import ModificationGenerator
    
    generator = ModificationGenerator()
    
    # Load the processed splits
    processed_splits = loader.load_processed_splits()
    
    # Convert back to SCANSample objects for modification generation
    from scan_data_loader import SCANSample
    train_samples = []
    for data in processed_splits['train'][:5000]:  # Use subset for faster generation
        sample = SCANSample(
            command=data['command'],
            action=data['action'],
            primitives=set(data['primitives']),
            modifiers=set(data['modifiers']),
            connectors=set(data['connectors']),
            length=data['length']
        )
        train_samples.append(sample)
    
    # Generate modifications
    modifications = generator.generate_all_modifications(train_samples)
    
    # Save modifications
    generator.save_modifications(modifications)
    
    print("Data generation complete!")
    return True

def save_checkpoint(model, stage, epoch, metrics, storage_dir, output_dir):
    """Save checkpoint with multiple backups"""
    checkpoint_name = f"stage_{stage}_epoch_{epoch}.h5"
    
    # Save locally first
    local_path = os.path.join(output_dir, 'checkpoints', checkpoint_name)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    model.save_weights(local_path)
    print(f"✓ Saved local checkpoint: {local_path}")
    
    # Save to persistent storage if available
    if storage_dir:
        storage_path = os.path.join(storage_dir, 'checkpoints', checkpoint_name)
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        model.save_weights(storage_path)
        print(f"✓ Saved to persistent storage: {storage_path}")
        
        # Also save metrics
        metrics_path = os.path.join(storage_dir, 'metrics', f'stage_{stage}_epoch_{epoch}.json')
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    return local_path

def train_with_safeguards(storage_dir):
    """Run training with comprehensive safeguards"""
    print("\n" + "="*60)
    print("STEP 2: Training with Safeguards")
    print("="*60)
    
    # Import the minimal training (most reliable)
    from train_progressive_minimal import create_model, create_dataset, SCANTokenizer
    from scan_data_loader import SCANDataLoader
    from modification_generator import ModificationGenerator
    
    # Configuration
    config = {
        'd_model': 128,
        'batch_size': 32,
        'stage1_epochs': 5,  # Reduced for testing
        'stage2_epochs': 5,
        'stage3_epochs': 5,
        'stage4_epochs': 5,
        'stage1_lr': 1e-3,
        'stage2_lr': 5e-4,
        'stage3_lr': 2e-4,
        'stage4_lr': 1e-4,
        'output_dir': 'outputs/safeguarded_training',
        'storage_dir': storage_dir
    }
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Load data
    print("Loading data...")
    loader = SCANDataLoader()
    splits = loader.load_processed_splits()
    
    # Create tokenizer
    print("Building tokenizer...")
    tokenizer = SCANTokenizer()
    tokenizer.build_vocabulary(splits['train'])
    
    # Save tokenizer
    tokenizer_path = os.path.join(config['output_dir'], 'tokenizer.json')
    tokenizer.save(tokenizer_path)
    if storage_dir:
        shutil.copy(tokenizer_path, os.path.join(storage_dir, 'tokenizer.json'))
    
    # Create model
    print("Creating model...")
    model = create_model(
        command_vocab_size=len(tokenizer.command_to_id),
        action_vocab_size=len(tokenizer.action_to_id),
        d_model=config['d_model']
    )
    
    # Load modifications
    mod_generator = ModificationGenerator()
    modifications = mod_generator.load_modifications()
    
    # Training stages
    stages = [
        ("Stage 1: Basic SCAN", splits['train'], [], config['stage1_epochs'], config['stage1_lr']),
        ("Stage 2: Simple Modifications", splits['train'], modifications[:100], config['stage2_epochs'], config['stage2_lr']),
        ("Stage 3: Complex Modifications", splits['train'], modifications[:500], config['stage3_epochs'], config['stage3_lr']),
        ("Stage 4: Novel Generation", splits['train'][:1000], modifications, config['stage4_epochs'], config['stage4_lr'])
    ]
    
    # Training history
    training_history = {
        'config': config,
        'stages': []
    }
    
    # Train through stages
    for stage_idx, (stage_name, base_data, mods, epochs, lr) in enumerate(stages, 1):
        print(f"\n{stage_name}")
        print("-" * 40)
        
        stage_history = {
            'name': stage_name,
            'epochs': [],
            'start_time': datetime.now().isoformat()
        }
        
        # Create dataset for this stage
        if mods:
            # Mix base data with modifications
            stage_data = base_data[:int(len(base_data) * 0.7)]  # 70% base
            stage_data.extend([m for m in mods if m['modification']])  # 30% mods
        else:
            stage_data = base_data
        
        dataset = create_dataset(stage_data, tokenizer, batch_size=config['batch_size'])
        
        # Configure optimizer
        import tensorflow as tf
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train epochs
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train one epoch
            history = model.fit(
                dataset,
                epochs=1,
                verbose=1
            )
            
            # Save metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                'loss': float(history.history['loss'][0]),
                'accuracy': float(history.history['accuracy'][0]),
                'timestamp': datetime.now().isoformat()
            }
            stage_history['epochs'].append(epoch_metrics)
            
            # Save checkpoint every epoch
            save_checkpoint(
                model, stage_idx, epoch + 1, epoch_metrics, 
                storage_dir, config['output_dir']
            )
            
            # Quick evaluation on validation
            if epoch == epochs - 1:  # Last epoch
                print("\nEvaluating on validation...")
                val_dataset = create_dataset(
                    splits['val_interpolation'][:100], 
                    tokenizer, 
                    batch_size=config['batch_size']
                )
                val_metrics = model.evaluate(val_dataset, verbose=0)
                stage_history['val_loss'] = float(val_metrics[0])
                stage_history['val_accuracy'] = float(val_metrics[1])
                print(f"Validation - Loss: {val_metrics[0]:.4f}, Accuracy: {val_metrics[1]:.4f}")
        
        stage_history['end_time'] = datetime.now().isoformat()
        training_history['stages'].append(stage_history)
        
        # Save training history after each stage
        history_path = os.path.join(config['output_dir'], 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        if storage_dir:
            shutil.copy(history_path, os.path.join(storage_dir, 'training_history.json'))
    
    # Final model save
    final_path = os.path.join(config['output_dir'], 'final_model.h5')
    model.save_weights(final_path)
    if storage_dir:
        shutil.copy(final_path, os.path.join(storage_dir, 'final_model.h5'))
    
    print("\n✓ Training completed successfully!")
    return True, training_history

def comprehensive_evaluation(storage_dir):
    """Run comprehensive evaluation on all test sets"""
    print("\n" + "="*60)
    print("STEP 3: Comprehensive Evaluation")
    print("="*60)
    
    from train_progressive_minimal import create_model, create_dataset, SCANTokenizer
    from scan_data_loader import SCANDataLoader
    
    # Load everything
    loader = SCANDataLoader()
    splits = loader.load_processed_splits()
    
    # Load tokenizer
    tokenizer = SCANTokenizer()
    tokenizer_path = 'outputs/safeguarded_training/tokenizer.json'
    if os.path.exists(tokenizer_path):
        tokenizer.load(tokenizer_path)
    else:
        tokenizer.build_vocabulary(splits['train'])
    
    # Create model and load weights
    model = create_model(
        command_vocab_size=len(tokenizer.command_to_id),
        action_vocab_size=len(tokenizer.action_to_id),
        d_model=128
    )
    
    model_path = 'outputs/safeguarded_training/final_model.h5'
    if os.path.exists(model_path):
        model.load_weights(model_path)
        print("✓ Loaded trained model")
    else:
        print("⚠️  No trained model found, using random weights")
    
    # Evaluate on all test sets
    test_sets = [
        ('val_interpolation', 'Validation (Interpolation)'),
        ('val_extrapolation', 'Validation (Extrapolation)'),
        ('test_interpolation', 'Test (Interpolation)'),
        ('test_primitive_extrap', 'Test (Primitive Extrapolation)'),
    ]
    
    results = {}
    
    for split_name, display_name in test_sets:
        if split_name in splits and splits[split_name]:
            print(f"\nEvaluating on {display_name}...")
            
            # Create dataset
            test_data = splits[split_name][:500]  # Limit for speed
            test_dataset = create_dataset(test_data, tokenizer, batch_size=32)
            
            # Evaluate
            import tensorflow as tf
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            metrics = model.evaluate(test_dataset, verbose=1)
            
            results[split_name] = {
                'name': display_name,
                'samples': len(test_data),
                'loss': float(metrics[0]),
                'accuracy': float(metrics[1])
            }
    
    # Save results
    results_path = os.path.join('outputs/safeguarded_training', 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    if storage_dir:
        shutil.copy(results_path, os.path.join(storage_dir, 'evaluation_results.json'))
    
    # Print summary
    print("\n" + "="*40)
    print("EVALUATION SUMMARY")
    print("="*40)
    for split_name, res in results.items():
        print(f"{res['name']:30} - Accuracy: {res['accuracy']:.2%}")
    
    return results

def save_all_results(storage_dir):
    """Final comprehensive save of all results"""
    print("\n" + "="*60)
    print("STEP 4: Final Results Preservation")
    print("="*60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create backup of entire outputs directory
    if os.path.exists('outputs'):
        backup_name = f'compositional_language_complete_{timestamp}.zip'
        
        # Create local backup
        os.system(f'zip -r {backup_name} outputs/ data/processed/')
        print(f"✓ Created local backup: {backup_name}")
        
        # Copy to storage
        if storage_dir:
            shutil.copy(backup_name, os.path.join(storage_dir, backup_name))
            print(f"✓ Copied backup to persistent storage")
            
            # List all files in storage
            print(f"\nFiles in persistent storage ({storage_dir}):")
            for root, dirs, files in os.walk(storage_dir):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), storage_dir)
                    print(f"  - {rel_path}")
    
    return True

def main():
    """Main pipeline with comprehensive safeguards"""
    print("="*60)
    print("Compositional Language Training with Safeguards")
    print("="*60)
    
    # Setup
    exp_dir, storage_dir = setup_environment()
    
    # Check GPU
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPU Available: {gpus}")
    if gpus:
        print(f"GPU Details: {gpus[0]}")
    
    try:
        # Generate data
        if not generate_data():
            print("Data generation failed!")
            return 1
        
        # Train with safeguards
        success, history = train_with_safeguards(storage_dir)
        if not success:
            print("Training failed!")
            return 1
        
        # Comprehensive evaluation
        results = comprehensive_evaluation(storage_dir)
        
        # Final save
        save_all_results(storage_dir)
        
        print("\n" + "="*60)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        if storage_dir:
            print(f"\nResults preserved in: {storage_dir}")
            print("To download from Paperspace:")
            print(f"  1. zip -r results.zip {storage_dir}/")
            print(f"  2. Download the zip file")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Emergency save attempt
        print("\nAttempting emergency save...")
        try:
            save_all_results(storage_dir)
        except:
            pass
        
        return 1

if __name__ == "__main__":
    sys.exit(main())