"""
Quick test of pendulum baseline training with minimal data
"""

from train_pendulum_baselines import PendulumDataLoader, PendulumBaselineTrainer
from pendulum_data_generator import PendulumDataConfig, PendulumDataGenerator
from pathlib import Path

def main():
    """Quick test with minimal data"""
    print("Running quick pendulum baseline test...")
    
    # Generate small test datasets
    test_data_path = Path("data/processed/pendulum_test_quick")
    
    if not test_data_path.exists():
        print("Generating minimal test datasets...")
        config = PendulumDataConfig(
            num_samples=100,
            sequence_length=60,  # 1 second
            output_dir=str(test_data_path)
        )
        generator = PendulumDataGenerator(config)
        
        # Generate minimal datasets
        train_data = generator.generate_dataset(mechanism='fixed', num_samples=80)
        generator.save_dataset(train_data, 'pendulum_train')
        
        val_data = generator.generate_dataset(mechanism='fixed', num_samples=20)
        generator.save_dataset(val_data, 'pendulum_val')
        
        test_fixed = generator.generate_dataset(mechanism='fixed', num_samples=20)
        generator.save_dataset(test_fixed, 'pendulum_test_fixed')
        
        test_ood = generator.generate_dataset(mechanism='time-varying', num_samples=20)
        generator.save_dataset(test_ood, 'pendulum_test_ood')
    
    # Load data
    loader = PendulumDataLoader(data_dir=str(test_data_path))
    data = loader.prepare_data()
    
    # Quick training with fewer epochs
    trainer = PendulumBaselineTrainer(output_dir="outputs/pendulum_test_quick")
    
    # Override epochs for quick test
    import baseline_models_physics
    
    # Monkey patch for quick test
    def quick_train(self, train_data, val_data, epochs=5, **kwargs):
        """Quick training for test"""
        return self._original_train(train_data, val_data, epochs=5, **kwargs)
    
    # Patch all models
    for model_class in [baseline_models_physics.PhysicsGFlowNetBaseline,
                       baseline_models_physics.PhysicsGraphExtrapolationBaseline,
                       baseline_models_physics.PhysicsMAMLBaseline]:
        if hasattr(model_class, 'train'):
            model_class._original_train = model_class.train
            model_class.train = quick_train
    
    # Train just ERM for quick test
    print("\nTraining ERM baseline only for quick test...")
    erm_model = trainer.train_erm_baseline(data)
    results = {'ERM+Aug': trainer.evaluate_model(erm_model, data, "ERM+Aug")}
    
    # Save results
    trainer.save_results(results)
    
    print("\nQuick test complete!")
    print(f"Results: {results}")

if __name__ == "__main__":
    main()