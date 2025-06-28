#!/usr/bin/env python3
"""
Quick script to save all important results before Paperspace shutdown
Run this immediately after training completes!
"""

import os
import shutil
import json
from datetime import datetime
from pathlib import Path

def save_critical_results():
    """Save all critical results to a single directory"""
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path(f'compositional_language_results_{timestamp}')
    results_dir.mkdir(exist_ok=True)
    
    print(f"Saving results to: {results_dir}")
    
    # 1. Copy all model checkpoints
    checkpoints_dir = Path('outputs/full_training')
    if checkpoints_dir.exists():
        shutil.copytree(checkpoints_dir, results_dir / 'checkpoints')
        print("✓ Saved model checkpoints")
    
    # 2. Copy training results JSON
    results_file = checkpoints_dir / 'training_results.json'
    if results_file.exists():
        shutil.copy(results_file, results_dir / 'training_results.json')
        print("✓ Saved training results")
    
    # 3. Create quick summary
    summary = {
        'experiment': 'Compositional Language - Distribution Invention',
        'timestamp': timestamp,
        'training_complete': results_file.exists()
    }
    
    # Try to load and summarize results
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
            
        summary.update({
            'test_results': results.get('test_results', {}),
            'final_scores': {
                'interpolation': results.get('test_results', {}).get('test_interpolation', 0),
                'extrapolation': results.get('test_results', {}).get('test_primitive_extrap', 0)
            }
        })
    
    # 4. Save summary
    with open(results_dir / 'SUMMARY.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    if 'test_results' in summary:
        for test_name, accuracy in summary['test_results'].items():
            print(f"{test_name}: {accuracy:.2%}")
    
    print(f"\nAll results saved to: {results_dir}")
    print("\nTo download:")
    print(f"1. Zip the directory: zip -r {results_dir}.zip {results_dir}/")
    print(f"2. Download before shutdown!")
    
    return results_dir

if __name__ == "__main__":
    save_critical_results()