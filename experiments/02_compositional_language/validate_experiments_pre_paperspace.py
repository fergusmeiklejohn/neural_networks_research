#!/usr/bin/env python3
"""
Comprehensive validation script to run before Paperspace experiments.
Tests all components with minimal data to catch errors before GPU time.
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import sys
import json
import traceback
from pathlib import Path
from datetime import datetime
import tensorflow as tf
import numpy as np

# Import all our modules to test
from models import create_model, CompositionalLanguageModel
from models_v2 import create_model_v2, CompositionalLanguageModelV2
from train_progressive_curriculum import SCANTokenizer, create_dataset
from scan_data_loader import SCANDataLoader
from modification_generator import ModificationGenerator


class ExperimentValidator:
    """Validates all components of the experiment before Paperspace deployment."""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'errors': [],
            'warnings': []
        }
        
    def log_test(self, name, success, details=""):
        """Log test result."""
        self.results['tests'][name] = {
            'success': success,
            'details': details
        }
        
        status = "✓" if success else "✗"
        print(f"{status} {name}")
        if details and not success:
            print(f"  → {details}")
    
    def test_data_loading(self):
        """Test data loading and tokenization."""
        print("\n1. Testing Data Loading...")
        print("-" * 50)
        
        try:
            # Load data
            data_loader = SCANDataLoader()
            data_loader.load_all_data()
            splits = data_loader.create_isolated_splits()
            
            # Check splits
            required_splits = ['train', 'val_interpolation', 'test_interpolation']
            for split in required_splits:
                if split not in splits:
                    raise ValueError(f"Missing required split: {split}")
                if len(splits[split]) == 0:
                    raise ValueError(f"Empty split: {split}")
            
            self.log_test("Data loading", True, 
                         f"Loaded {len(splits['train'])} train samples")
            
            # Test tokenizer
            tokenizer = SCANTokenizer()
            train_data = [{'command': s.command, 'action': s.action} 
                         for s in splits['train'][:100]]
            tokenizer.build_vocabulary(train_data)
            
            # Test encoding/decoding
            test_command = "walk twice and jump"
            encoded = tokenizer.encode_command(test_command)
            
            if len(encoded) == 0:
                raise ValueError("Tokenizer encoding failed")
            
            self.log_test("Tokenizer", True, 
                         f"Vocab sizes: cmd={len(tokenizer.command_to_id)}, "
                         f"act={len(tokenizer.action_to_id)}")
            
            return True, splits, tokenizer
            
        except Exception as e:
            self.log_test("Data loading", False, str(e))
            self.results['errors'].append(f"Data loading: {str(e)}")
            return False, None, None
    
    def test_modifications(self, train_samples):
        """Test modification generation."""
        print("\n2. Testing Modification Generation...")
        print("-" * 50)
        
        try:
            mod_gen = ModificationGenerator()
            
            # Test with small sample
            small_sample = train_samples[:50]
            mod_dict = mod_gen.generate_all_modifications(small_sample)
            
            # Check all modification types generated
            expected_types = ['simple_swaps', 'action_modifications', 
                            'structural_modifications', 'novel_combinations']
            
            total_mods = 0
            for mod_type in expected_types:
                if mod_type not in mod_dict:
                    raise ValueError(f"Missing modification type: {mod_type}")
                count = len(mod_dict[mod_type])
                total_mods += count
                print(f"  {mod_type}: {count} modifications")
            
            if total_mods == 0:
                raise ValueError("No modifications generated")
            
            # Test modification format
            sample_mod = mod_dict['simple_swaps'][0] if mod_dict['simple_swaps'] else None
            if sample_mod:
                required_attrs = ['original_sample', 'modified_sample', 
                                'modification_description', 'modification_type']
                for attr in required_attrs:
                    if not hasattr(sample_mod, attr):
                        raise ValueError(f"ModificationPair missing attribute: {attr}")
            
            self.log_test("Modification generation", True, 
                         f"Generated {total_mods} total modifications")
            
            return True, mod_dict
            
        except Exception as e:
            self.log_test("Modification generation", False, str(e))
            self.results['errors'].append(f"Modifications: {str(e)}")
            return False, None
    
    def test_models(self, tokenizer):
        """Test both v1 and v2 models."""
        print("\n3. Testing Model Architectures...")
        print("-" * 50)
        
        cmd_vocab = len(tokenizer.command_to_id)
        act_vocab = len(tokenizer.action_to_id)
        
        # Test inputs
        test_inputs = {
            'command': tf.constant([[1, 2, 3, 4, 5]]),
            'target': tf.constant([[1, 2, 3, 4, 5, 6]]),
            'modification': tf.constant([[7, 8, 9]])
        }
        
        # Test v1 model
        try:
            model_v1 = create_model(cmd_vocab, act_vocab, d_model=64)
            output_v1 = model_v1(test_inputs, training=True)
            
            if output_v1.shape != (1, 6, act_vocab):
                raise ValueError(f"Unexpected output shape: {output_v1.shape}")
            
            self.log_test("Model v1", True, 
                         f"Parameters: {model_v1.count_params():,}")
            
        except Exception as e:
            self.log_test("Model v1", False, str(e))
            self.results['errors'].append(f"Model v1: {str(e)}")
        
        # Test v2 model
        try:
            model_v2 = create_model_v2(cmd_vocab, act_vocab, d_model=64)
            output_v2 = model_v2(test_inputs, training=True)
            
            if output_v2.shape != (1, 6, act_vocab):
                raise ValueError(f"Unexpected output shape: {output_v2.shape}")
            
            # Test gating mechanism
            test_analysis = model_v2({
                'command': test_inputs['command'],
                'modification': test_inputs['modification']
            }, training=False)
            
            if 'modification_gates' not in test_analysis:
                self.results['warnings'].append("Model v2 gates not accessible")
            
            self.log_test("Model v2", True, 
                         f"Parameters: {model_v2.count_params():,}")
            
        except Exception as e:
            self.log_test("Model v2", False, str(e))
            self.results['errors'].append(f"Model v2: {str(e)}")
        
        return True
    
    def test_dataset_creation(self, train_data, modifications, tokenizer):
        """Test dataset creation and batching."""
        print("\n4. Testing Dataset Creation...")
        print("-" * 50)
        
        try:
            # Test base dataset
            base_dataset = create_dataset(train_data[:100], tokenizer, batch_size=8)
            
            # Get one batch
            for batch in base_dataset.take(1):
                inputs, targets = batch
                
                if 'command' not in inputs:
                    raise ValueError("Missing 'command' in inputs")
                if inputs['command'].shape[0] != 8:
                    raise ValueError(f"Incorrect batch size: {inputs['command'].shape[0]}")
            
            self.log_test("Base dataset creation", True, "Batch size correct")
            
            # Test mixed dataset
            from train_with_mixed_strategy import create_mixed_dataset
            
            # Convert modifications to training format
            mod_data = []
            for pair in modifications[:50]:
                mod_data.append({
                    'command': pair.modified_sample.command,
                    'action': pair.modified_sample.action,
                    'modification': pair.modification_description
                })
            
            mixed_dataset = create_mixed_dataset(
                train_data[:100], mod_data, 
                mix_ratio=0.5, tokenizer=tokenizer, batch_size=8
            )
            
            # Verify mixed dataset
            for batch in mixed_dataset.take(1):
                inputs, targets = batch
                if 'modification' not in inputs:
                    raise ValueError("Missing 'modification' in mixed dataset")
            
            self.log_test("Mixed dataset creation", True, "Modifications included")
            
            return True
            
        except Exception as e:
            self.log_test("Dataset creation", False, str(e))
            self.results['errors'].append(f"Dataset: {str(e)}")
            return False
    
    def test_training_loop(self, model, dataset):
        """Test minimal training loop."""
        print("\n5. Testing Training Loop...")
        print("-" * 50)
        
        try:
            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train for 1 step
            history = model.fit(dataset.take(2), epochs=1, verbose=0)
            
            if 'loss' not in history.history:
                raise ValueError("No loss in training history")
            
            loss = history.history['loss'][0]
            acc = history.history['accuracy'][0]
            
            self.log_test("Training loop", True, 
                         f"Loss: {loss:.4f}, Acc: {acc:.4f}")
            
            # Test saving/loading
            temp_path = Path('temp_model_weights.h5')
            model.save_weights(temp_path)
            model.load_weights(temp_path)
            temp_path.unlink()  # Clean up
            
            self.log_test("Model save/load", True, "Weights saved and loaded")
            
            return True
            
        except Exception as e:
            self.log_test("Training loop", False, str(e))
            self.results['errors'].append(f"Training: {str(e)}")
            return False
    
    def test_paperspace_paths(self):
        """Test Paperspace-specific path handling."""
        print("\n6. Testing Paperspace Path Handling...")
        print("-" * 50)
        
        try:
            # Test path detection logic
            paths_to_test = [
                '/notebooks/neural_networks_research',
                '/workspace/neural_networks_research',
                os.path.abspath('../..')
            ]
            
            found_path = None
            for path in paths_to_test:
                if os.path.exists(path):
                    found_path = path
                    break
            
            self.log_test("Path detection", True, f"Using: {found_path}")
            
            # Test storage directory creation
            from pathlib import Path
            test_storage = Path('/tmp/test_storage_dir')
            test_storage.mkdir(exist_ok=True)
            test_storage.rmdir()
            
            self.log_test("Directory creation", True, "Can create directories")
            
            return True
            
        except Exception as e:
            self.log_test("Paperspace paths", False, str(e))
            self.results['warnings'].append(f"Path handling: {str(e)}")
            return True  # Warning only
    
    def run_all_tests(self):
        """Run all validation tests."""
        print("=" * 60)
        print("PAPERSPACE EXPERIMENT VALIDATION")
        print("=" * 60)
        
        # Test 1: Data loading
        success, splits, tokenizer = self.test_data_loading()
        if not success:
            print("\n❌ Critical failure in data loading. Stopping.")
            return False
        
        # Test 2: Modifications
        train_samples = splits['train']
        success, mod_dict = self.test_modifications(train_samples)
        if not success:
            print("\n❌ Critical failure in modifications. Stopping.")
            return False
        
        # Flatten modifications
        all_mods = []
        for mods in mod_dict.values():
            all_mods.extend(mods)
        
        # Test 3: Models
        self.test_models(tokenizer)
        
        # Test 4: Dataset creation
        train_data = [{'command': s.command, 'action': s.action} 
                     for s in train_samples[:200]]
        self.test_dataset_creation(train_data, all_mods, tokenizer)
        
        # Test 5: Training loop (with v2 model)
        try:
            model = create_model_v2(
                len(tokenizer.command_to_id),
                len(tokenizer.action_to_id),
                d_model=64
            )
            dataset = create_dataset(train_data[:50], tokenizer, batch_size=4)
            self.test_training_loop(model, dataset)
        except Exception as e:
            print(f"Training test skipped: {e}")
        
        # Test 6: Paperspace paths
        self.test_paperspace_paths()
        
        # Summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.results['tests'])
        passed_tests = sum(1 for t in self.results['tests'].values() if t['success'])
        
        print(f"\nTests passed: {passed_tests}/{total_tests}")
        
        if self.results['errors']:
            print(f"\n❌ Errors ({len(self.results['errors'])}):")
            for error in self.results['errors']:
                print(f"  - {error}")
        
        if self.results['warnings']:
            print(f"\n⚠️  Warnings ({len(self.results['warnings'])}):")
            for warning in self.results['warnings']:
                print(f"  - {warning}")
        
        # Save results
        with open('validation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        ready = len(self.results['errors']) == 0
        if ready:
            print("\n✅ ALL CRITICAL TESTS PASSED - Ready for Paperspace!")
        else:
            print("\n❌ CRITICAL ERRORS FOUND - Fix before Paperspace deployment!")
        
        return ready


def main():
    """Run validation suite."""
    validator = ExperimentValidator()
    ready = validator.run_all_tests()
    
    if ready:
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("1. Review validation_results.json for any warnings")
        print("2. Run: python paperspace_experiment_runner.py")
        print("3. Monitor GPU memory usage in first few minutes")
        print("="*60)
    
    return 0 if ready else 1


if __name__ == '__main__':
    sys.exit(main())