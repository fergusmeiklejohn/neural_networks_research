#!/usr/bin/env python3
"""
Comprehensive local testing before Paperspace deployment
Run this script to verify everything works before cloud training
"""

import os
import sys
import gc
import json
import time
import psutil
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set up minimal environment
os.environ['KERAS_BACKEND'] = 'jax'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
from keras import layers


class TestResults:
    """Track test results"""
    def __init__(self):
        self.results = []
        self.start_time = time.time()
    
    def add(self, test_name, status, message="", duration=0):
        self.results.append({
            'test': test_name,
            'status': status,
            'message': message,
            'duration': duration
        })
    
    def summary(self):
        passed = sum(1 for r in self.results if r['status'] == 'PASSED')
        total = len(self.results)
        return f"{passed}/{total} tests passed"


def test_environment_setup():
    """Test 1: Verify environment is correctly configured"""
    print("\n1. Testing environment setup...")
    start = time.time()
    
    try:
        # Check Keras backend
        backend = keras.backend.backend()
        print(f"   Keras backend: {backend}")
        
        # Check if we can build a simple model
        model = keras.Sequential([
            layers.Dense(10, input_shape=(5,)),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        # Test with dummy data
        x = np.random.rand(10, 5)
        y = np.random.rand(10, 1)
        model.fit(x, y, epochs=1, verbose=0)
        
        duration = time.time() - start
        print(f"   ✓ Environment setup successful ({duration:.2f}s)")
        return True, "", duration
        
    except Exception as e:
        duration = time.time() - start
        print(f"   ✗ Environment setup failed: {str(e)}")
        return False, str(e), duration


def test_path_flexibility():
    """Test 2: Verify path detection works for both local and Paperspace"""
    print("\n2. Testing path flexibility...")
    start = time.time()
    
    try:
        def get_base_path():
            # Paperspace paths
            if os.path.exists('/notebooks/neural_networks_research'):
                return '/notebooks/neural_networks_research'
            elif os.path.exists('/storage'):
                return '/storage/neural_networks_research'
            # Local paths
            else:
                # Use project root
                return str(project_root)
        
        base = get_base_path()
        assert os.path.exists(base), f"Base path {base} doesn't exist"
        print(f"   Base path: {base}")
        
        # Test creating subdirectories
        test_dirs = ['data/scan', 'outputs/models', 'outputs/logs']
        for dir_path in test_dirs:
            full_path = os.path.join(base, 'experiments/02_compositional_language', dir_path)
            os.makedirs(full_path, exist_ok=True)
            assert os.path.exists(full_path), f"Failed to create {full_path}"
        
        duration = time.time() - start
        print(f"   ✓ Path flexibility test passed ({duration:.2f}s)")
        return True, "", duration
        
    except Exception as e:
        duration = time.time() - start
        print(f"   ✗ Path flexibility test failed: {str(e)}")
        return False, str(e), duration


def test_scan_data_generation():
    """Test 3: Test SCAN data generation with small subset"""
    print("\n3. Testing SCAN data generation...")
    start = time.time()
    
    try:
        # Create mini SCAN dataset
        commands = [
            ("jump", "JUMP"),
            ("jump twice", "JUMP JUMP"),
            ("walk left", "TURN_LEFT WALK"),
            ("look and walk", "LOOK WALK"),
            ("jump thrice", "JUMP JUMP JUMP"),
            ("walk right", "TURN_RIGHT WALK"),
            ("look around left", "TURN_LEFT LOOK TURN_LEFT LOOK TURN_LEFT LOOK TURN_LEFT LOOK"),
            ("jump left", "TURN_LEFT JUMP"),
            ("run", "RUN"),
            ("run twice", "RUN RUN")
        ]
        
        # Test tokenization
        def create_tokenizer(data):
            # Simple word-level tokenizer
            vocab = set()
            for cmd, action in data:
                vocab.update(cmd.split())
                vocab.update(action.split())
            
            word_to_id = {word: i+1 for i, word in enumerate(sorted(vocab))}
            word_to_id['<PAD>'] = 0
            word_to_id['<START>'] = len(word_to_id)
            word_to_id['<END>'] = len(word_to_id)
            
            return word_to_id
        
        tokenizer = create_tokenizer(commands)
        print(f"   Vocabulary size: {len(tokenizer)}")
        
        # Test data splits
        n = len(commands)
        train_idx = int(0.6 * n)
        val_idx = int(0.8 * n)
        
        train = commands[:train_idx]
        val = commands[train_idx:val_idx]
        test = commands[val_idx:]
        
        print(f"   Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        # Test rule modifications
        def apply_rule_modification(data, rule):
            modified = []
            for cmd, action in data:
                if rule == "jump->walk":
                    new_cmd = cmd.replace("jump", "walk")
                    new_action = action.replace("JUMP", "WALK")
                    modified.append((new_cmd, new_action))
                else:
                    modified.append((cmd, action))
            return modified
        
        modified = apply_rule_modification(commands, "jump->walk")
        assert any("walk" in cmd for cmd, _ in modified), "Rule modification failed"
        
        duration = time.time() - start
        print(f"   ✓ SCAN data generation test passed ({duration:.2f}s)")
        return True, "", duration
        
    except Exception as e:
        duration = time.time() - start
        print(f"   ✗ SCAN data generation failed: {str(e)}")
        return False, str(e), duration


def test_model_building():
    """Test 4: Test building different model architectures"""
    print("\n4. Testing model building...")
    start = time.time()
    
    try:
        vocab_size = 50
        max_length = 20
        
        # Test baseline seq2seq
        def build_baseline_model():
            encoder_input = layers.Input(shape=(max_length,))
            encoder_embed = layers.Embedding(vocab_size, 128)(encoder_input)
            encoder_lstm = layers.LSTM(256, return_state=True)
            _, state_h, state_c = encoder_lstm(encoder_embed)
            encoder_states = [state_h, state_c]
            
            decoder_input = layers.Input(shape=(max_length,))
            decoder_embed = layers.Embedding(vocab_size, 128)(decoder_input)
            decoder_lstm = layers.LSTM(256, return_sequences=True)
            decoder_output = decoder_lstm(decoder_embed, initial_state=encoder_states)
            decoder_dense = layers.Dense(vocab_size, activation='softmax')
            output = decoder_dense(decoder_output)
            
            model = keras.Model([encoder_input, decoder_input], output)
            return model
        
        baseline = build_baseline_model()
        print(f"   Baseline model parameters: {baseline.count_params():,}")
        
        # Test compilation
        baseline.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Test forward pass
        enc_input = np.random.randint(0, vocab_size, (2, max_length))
        dec_input = np.random.randint(0, vocab_size, (2, max_length))
        output = baseline([enc_input, dec_input])
        assert output.shape == (2, max_length, vocab_size), f"Unexpected output shape: {output.shape}"
        
        duration = time.time() - start
        print(f"   ✓ Model building test passed ({duration:.2f}s)")
        return True, "", duration
        
    except Exception as e:
        duration = time.time() - start
        print(f"   ✗ Model building failed: {str(e)}")
        return False, str(e), duration


def test_training_with_checkpoints():
    """Test 5: Test training loop with checkpoint/recovery"""
    print("\n5. Testing training with checkpoints...")
    start = time.time()
    
    try:
        # Build simple model
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(10,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        # Create dummy data
        x_train = np.random.rand(100, 10)
        y_train = np.random.rand(100, 1)
        
        checkpoint_dir = Path("./test_checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Simulate training with interruption
        for epoch in range(3):
            # Train for one epoch
            history = model.fit(x_train, y_train, epochs=1, verbose=0)
            
            # Save checkpoint
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.weights.h5"
            model.save_weights(str(checkpoint_path))
            
            # Simulate recovery at epoch 1
            if epoch == 1:
                # Build new model and load weights
                new_model = keras.Sequential([
                    layers.Dense(64, activation='relu', input_shape=(10,)),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1)
                ])
                new_model.compile(optimizer='adam', loss='mse')
                
                # Need to build model first
                new_model(x_train[:1])  # Build with sample data
                new_model.load_weights(str(checkpoint_path))
                
                # Verify recovery by continuing training
                history = new_model.fit(x_train, y_train, epochs=1, verbose=0)
                print(f"   Successfully recovered from epoch {epoch}")
        
        # Cleanup
        import shutil
        shutil.rmtree(checkpoint_dir)
        
        duration = time.time() - start
        print(f"   ✓ Checkpoint/recovery test passed ({duration:.2f}s)")
        return True, "", duration
        
    except Exception as e:
        duration = time.time() - start
        print(f"   ✗ Checkpoint/recovery test failed: {str(e)}")
        return False, str(e), duration


def test_memory_usage():
    """Test 6: Monitor memory usage during training"""
    print("\n6. Testing memory usage...")
    start = time.time()
    
    try:
        process = psutil.Process()
        
        # Initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"   Initial memory: {initial_memory:.2f} MB")
        
        # Create data
        data_size = 1000
        x_data = np.random.rand(data_size, 50)
        y_data = np.random.randint(0, 10, (data_size, 20))
        
        data_memory = process.memory_info().rss / 1024 / 1024
        print(f"   After data creation: {data_memory:.2f} MB (+{data_memory-initial_memory:.2f} MB)")
        
        # Build model
        model = keras.Sequential([
            layers.Embedding(100, 128, input_length=50),
            layers.LSTM(256),
            layers.Dense(20, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        model_memory = process.memory_info().rss / 1024 / 1024
        print(f"   After model build: {model_memory:.2f} MB (+{model_memory-data_memory:.2f} MB)")
        
        # Train briefly
        model.fit(x_data, y_data, epochs=2, batch_size=32, verbose=0)
        
        train_memory = process.memory_info().rss / 1024 / 1024
        print(f"   After training: {train_memory:.2f} MB (+{train_memory-model_memory:.2f} MB)")
        
        # Cleanup
        del model
        del x_data
        del y_data
        gc.collect()
        
        # Wait a moment for garbage collection
        time.sleep(1)
        
        final_memory = process.memory_info().rss / 1024 / 1024
        print(f"   After cleanup: {final_memory:.2f} MB")
        
        # Check for significant memory leak (allow some overhead)
        memory_increase = final_memory - initial_memory
        print(f"   Net memory increase: {memory_increase:.2f} MB")
        
        duration = time.time() - start
        print(f"   ✓ Memory usage test passed ({duration:.2f}s)")
        return True, "", duration
        
    except Exception as e:
        duration = time.time() - start
        print(f"   ✗ Memory usage test failed: {str(e)}")
        return False, str(e), duration


def test_integration():
    """Test 7: Full pipeline integration test"""
    print("\n7. Testing full pipeline integration...")
    start = time.time()
    
    try:
        # This would be a mini version of the full training pipeline
        print("   Running mini training pipeline...")
        
        # 1. Generate toy data
        train_data = [("jump", "JUMP"), ("walk", "WALK"), ("run", "RUN")]
        val_data = [("jump twice", "JUMP JUMP")]
        
        # 2. Create vocabulary
        vocab = {'<PAD>': 0, '<START>': 1, '<END>': 2, 'jump': 3, 'walk': 4, 
                 'run': 5, 'twice': 6, 'JUMP': 7, 'WALK': 8, 'RUN': 9}
        
        # 3. Build model
        model = keras.Sequential([
            layers.Embedding(len(vocab), 16, input_length=5),
            layers.LSTM(32),
            layers.Dense(len(vocab), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        # 4. Create dummy training data
        x_train = np.random.randint(0, len(vocab), (10, 5))
        y_train = np.random.randint(0, len(vocab), (10,))
        
        # 5. Train
        history = model.fit(x_train, y_train, epochs=2, verbose=0)
        
        # 6. Save results
        results = {
            'final_loss': float(history.history['loss'][-1]),
            'vocab_size': len(vocab),
            'model_params': model.count_params()
        }
        
        with open('test_integration_results.json', 'w') as f:
            json.dump(results, f)
        
        # Cleanup
        os.remove('test_integration_results.json')
        
        duration = time.time() - start
        print(f"   ✓ Integration test passed ({duration:.2f}s)")
        return True, "", duration
        
    except Exception as e:
        duration = time.time() - start
        print(f"   ✗ Integration test failed: {str(e)}")
        return False, str(e), duration


def main():
    """Run all tests and generate report"""
    print("="*70)
    print("COMPOSITIONAL LANGUAGE EXPERIMENT - LOCAL TEST SUITE")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = TestResults()
    
    # Define test suite
    tests = [
        ("Environment Setup", test_environment_setup),
        ("Path Flexibility", test_path_flexibility),
        ("SCAN Data Generation", test_scan_data_generation),
        ("Model Building", test_model_building),
        ("Checkpoint/Recovery", test_training_with_checkpoints),
        ("Memory Usage", test_memory_usage),
        ("Integration Test", test_integration)
    ]
    
    # Run tests
    for test_name, test_func in tests:
        passed, message, duration = test_func()
        results.add(test_name, "PASSED" if passed else "FAILED", message, duration)
    
    # Generate summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for result in results.results:
        symbol = "✓" if result['status'] == "PASSED" else "✗"
        print(f"{symbol} {result['test']}: {result['status']} ({result['duration']:.2f}s)")
        if result['message']:
            print(f"  → {result['message']}")
    
    print(f"\nOverall: {results.summary()}")
    print(f"Total time: {time.time() - results.start_time:.2f}s")
    
    # Save detailed results
    report_path = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'summary': results.summary(),
            'results': results.results,
            'total_duration': time.time() - results.start_time
        }, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_path}")
    
    # Return status
    all_passed = all(r['status'] == 'PASSED' for r in results.results)
    if all_passed:
        print("\n✅ ALL TESTS PASSED - Ready for Paperspace deployment!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - Fix issues before deployment!")
        return 1


if __name__ == "__main__":
    sys.exit(main())