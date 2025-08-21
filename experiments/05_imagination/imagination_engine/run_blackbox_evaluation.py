"""Black-box evaluation script for ARC-AGI-2.

CRITICAL: This script accesses evaluation data.
Only run AFTER all development and training is complete.
Running this during development will contaminate results!
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np


def confirm_evaluation_intent():
    """Require explicit confirmation before accessing evaluation data."""
    
    print("=" * 80)
    print("                    BLACK BOX EVALUATION WARNING")
    print("=" * 80)
    print()
    print("You are about to run evaluation on the UNSEEN ARC-AGI-2 test set.")
    print()
    print("CRITICAL REQUIREMENTS:")
    print("  1. All development must be complete")
    print("  2. Model must be fully trained")
    print("  3. No further changes after seeing results")
    print()
    print("Running this evaluation will:")
    print("  - Access previously unseen evaluation data")
    print("  - Test your model's true generalization")
    print("  - Provide final benchmark scores")
    print()
    
    response = input("Type 'EVALUATE' to proceed (or anything else to cancel): ")
    
    if response != "EVALUATE":
        print("\n❌ Evaluation cancelled. Good choice to wait!")
        return False
    
    print("\n✅ Proceeding with black-box evaluation...")
    return True


def load_evaluation_data() -> List[Dict]:
    """Load the black-box evaluation data."""
    
    eval_dir = Path(__file__).parent / "arc_agi_2_data" / "evaluation_BLACKBOX"
    
    if not eval_dir.exists():
        raise FileNotFoundError(
            "Evaluation data not found. Run download_arc_agi_2.py first."
        )
    
    all_tasks = []
    
    for file in eval_dir.glob("*.json"):
        with open(file, 'r') as f:
            data = json.load(f)
            
            if isinstance(data, dict):
                for task_id, task_data in data.items():
                    task = {
                        'id': task_id,
                        'train': task_data.get('train', []),
                        'test': task_data.get('test', [])
                    }
                    all_tasks.append(task)
    
    return all_tasks


def evaluate_model_on_task(model, task: Dict) -> float:
    """Evaluate model on a single ARC task."""
    
    # Convert to model format
    train_examples = []
    for example in task['train']:
        inp = np.array(example['input'], dtype=np.float32)
        out = np.array(example['output'], dtype=np.float32)
        train_examples.append((inp, out))
    
    # Get model prediction
    try:
        transform, info = model.solve_with_memory(train_examples, task['id'])
        
        # Evaluate on test examples
        scores = []
        for test_example in task['test']:
            test_input = np.array(test_example['input'], dtype=np.float32)
            expected = np.array(test_example['output'], dtype=np.float32)
            
            predicted = transform(test_input)
            
            if predicted.shape == expected.shape:
                score = np.mean(predicted == expected)
            else:
                score = 0.0
            
            scores.append(score)
        
        return np.mean(scores) if scores else 0.0
        
    except Exception as e:
        print(f"  Error on task {task['id']}: {e}")
        return 0.0


def run_blackbox_evaluation(model_path: str = None):
    """Run complete black-box evaluation."""
    
    # Get confirmation
    if not confirm_evaluation_intent():
        return
    
    print("\n" + "=" * 80)
    print("STARTING BLACK-BOX EVALUATION")
    print("=" * 80)
    
    # Load model
    print("\nLoading model...")
    if model_path:
        # Load from checkpoint
        print(f"Loading from: {model_path}")
        # TODO: Implement model loading
        raise NotImplementedError("Model loading not yet implemented")
    else:
        # Use current HTI system
        sys.path.append(str(Path(__file__).parent))
        from integrated_hti_system import IntegratedHTI
        model = IntegratedHTI()
        print("Using current HTI system (not trained)")
    
    # Load evaluation data
    print("\nLoading evaluation data...")
    eval_tasks = load_evaluation_data()
    print(f"Loaded {len(eval_tasks)} evaluation tasks")
    
    # Run evaluation
    print("\nEvaluating...")
    print("-" * 60)
    
    results = []
    start_time = time.time()
    
    for i, task in enumerate(eval_tasks):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(eval_tasks)} tasks...")
        
        score = evaluate_model_on_task(model, task)
        results.append({
            'task_id': task['id'],
            'score': score
        })
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    scores = [r['score'] for r in results]
    perfect = sum(1 for s in scores if s > 0.99)
    partial = sum(1 for s in scores if 0.5 < s <= 0.99)
    failed = sum(1 for s in scores if s <= 0.5)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(__file__).parent / f"blackbox_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'total_tasks': len(eval_tasks),
            'total_time': total_time,
            'results': results,
            'statistics': {
                'mean_score': float(np.mean(scores)),
                'median_score': float(np.median(scores)),
                'std_score': float(np.std(scores)),
                'perfect_solutions': perfect,
                'partial_solutions': partial,
                'failed_solutions': failed
            }
        }, f, indent=2)
    
    # Print final results
    print("\n" + "=" * 80)
    print("BLACK-BOX EVALUATION COMPLETE")
    print("=" * 80)
    
    print(f"\nTasks evaluated: {len(eval_tasks)}")
    print(f"Time taken: {total_time:.1f} seconds")
    
    print(f"\n📊 RESULTS:")
    print(f"  Mean score: {np.mean(scores):.1%}")
    print(f"  Median score: {np.median(scores):.1%}")
    print(f"  Std deviation: {np.std(scores):.1%}")
    
    print(f"\n📈 BREAKDOWN:")
    print(f"  Perfect (>99%): {perfect} ({perfect/len(eval_tasks):.1%})")
    print(f"  Partial (50-99%): {partial} ({partial/len(eval_tasks):.1%})")
    print(f"  Failed (<50%): {failed} ({failed/len(eval_tasks):.1%})")
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Context
    print(f"\n📌 CONTEXT:")
    print(f"  Human performance: ~85%")
    print(f"  SOTA (2024): ~30%")
    print(f"  Random baseline: ~5%")
    print(f"  Your score: {np.mean(scores):.1%}")
    
    if np.mean(scores) > 0.30:
        print(f"\n🎉 CONGRATULATIONS! You've matched/exceeded SOTA!")
    elif np.mean(scores) > 0.20:
        print(f"\n✅ Strong performance! Close to SOTA levels.")
    elif np.mean(scores) > 0.10:
        print(f"\n🔶 Decent performance, better than random.")
    else:
        print(f"\n📝 Room for improvement, but learning is happening!")
    
    return results


if __name__ == "__main__":
    # Only run if explicitly executed
    print("\n" + "!" * 80)
    print("                           FINAL WARNING")
    print("!" * 80)
    print("\nThis will access the evaluation data.")
    print("Only proceed if you are ready for final black-box testing!")
    print()
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--confirm":
        results = run_blackbox_evaluation()
    else:
        print("To run evaluation, use: python run_blackbox_evaluation.py --confirm")
        print("\nExiting safely without accessing evaluation data.")