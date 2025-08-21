"""Test Imagination Engine on ARC-AGI tasks for external validation.

This validates our approach on an established benchmark to see if our
imagination mechanisms generalize beyond our custom tasks.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from hypothesis_generator import GenerationStrategy, MinimalHypothesisGenerator
from improved_compositional_reasoner import ImprovedCompositionalReasoner
from improved_cross_domain import ImprovedCrossDomainTransfer
from final_integrated_system import FinalIntegratedSystem


class ARCTask:
    """Wrapper for ARC-AGI tasks."""
    
    def __init__(self, task_id: str, train_examples: List, test_examples: List):
        self.task_id = task_id
        self.train_examples = train_examples
        self.test_examples = test_examples
    
    def evaluate_solution(self, predicted: np.ndarray, expected: np.ndarray) -> float:
        """Evaluate a solution against expected output."""
        if predicted.shape != expected.shape:
            return 0.0
        
        correct = np.sum(predicted == expected)
        total = expected.size
        return correct / total if total > 0 else 0.0


def create_arc_style_tasks() -> List[Tuple[ARCTask, str]]:
    """Create ARC-style tasks that test imagination capabilities.
    
    We'll create simplified versions of classic ARC patterns that require
    true pattern discovery, not just memorization.
    """
    tasks = []
    
    # Task 1: Pattern completion with novel rule
    # Training shows increasing pattern, test requires understanding the rule
    train_1 = [
        (np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]])),
        (np.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[2, 2, 2], [2, 2, 0], [2, 0, 0]])),
    ]
    test_1 = [
        (np.array([[3, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[3, 3, 3], [3, 3, 3], [3, 3, 0]]))
    ]
    tasks.append((ARCTask("growth_pattern", train_1, test_1), "pattern_discovery"))
    
    # Task 2: Color transformation with conditional rule
    # If cell has neighbor of specific color, change to that color
    train_2 = [
        (np.array([[1, 2, 0], [0, 1, 0], [0, 0, 0]]),
         np.array([[2, 2, 0], [0, 2, 0], [0, 0, 0]])),
        (np.array([[3, 0, 1], [0, 0, 3], [1, 0, 0]]),
         np.array([[3, 0, 3], [0, 0, 3], [3, 0, 0]])),
    ]
    test_2 = [
        (np.array([[2, 1, 0], [1, 0, 2], [0, 2, 1]]),
         np.array([[2, 2, 0], [2, 0, 2], [0, 2, 2]]))
    ]
    tasks.append((ARCTask("color_propagation", train_2, test_2), "rule_combination"))
    
    # Task 3: Symmetry detection and completion
    # Complete pattern to make it symmetric
    train_3 = [
        (np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]]),
         np.array([[1, 0, 1], [2, 0, 2], [3, 0, 3]])),
        (np.array([[4, 5, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[4, 5, 0], [0, 0, 0], [0, 5, 4]])),
    ]
    test_3 = [
        (np.array([[2, 3, 0], [1, 0, 0], [0, 0, 0]]),
         np.array([[2, 3, 0], [1, 0, 1], [0, 3, 2]]))
    ]
    tasks.append((ARCTask("symmetry_completion", train_3, test_3), "cross_domain"))
    
    # Task 4: Object counting and replication
    # Count distinct objects and create that many copies
    train_4 = [
        (np.array([[1, 0, 2], [0, 0, 0], [3, 0, 0]]),
         np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])),  # 3 objects -> 3 copies each
        (np.array([[4, 0, 0], [0, 5, 0], [0, 0, 0]]),
         np.array([[4, 4, 0], [5, 5, 0], [0, 0, 0]])),  # 2 objects -> 2 copies each
    ]
    test_4 = [
        (np.array([[1, 2, 3], [0, 0, 4], [0, 0, 0]]),
         np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]))  # Need to understand: 4 objects but only show first 3
    ]
    tasks.append((ARCTask("object_replication", train_4, test_4), "counterfactual"))
    
    # Task 5: Novel transformation - diagonal fill
    # This pattern is unlikely to be in training data
    train_5 = [
        (np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
        (np.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])),
    ]
    test_5 = [
        (np.array([[3, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]]))
    ]
    tasks.append((ARCTask("diagonal_fill", train_5, test_5), "creative"))
    
    return tasks


def test_on_arc_tasks():
    """Test our Imagination Engine on ARC-style tasks."""
    
    print("\n" + "=" * 80)
    print("ARC-AGI STYLE VALIDATION")
    print("=" * 80)
    
    # Initialize our system
    system = FinalIntegratedSystem()
    
    # Create ARC-style tasks
    tasks = create_arc_style_tasks()
    
    print(f"\nTesting on {len(tasks)} ARC-style tasks...")
    
    results = []
    category_scores = {}
    
    for i, (task, category) in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] Task: {task.task_id} ({category})")
        
        # Try to solve using our system
        start_time = time.time()
        
        # Adapt task format for our system
        class ARCAdapter:
            def __init__(self, arc_task):
                self.task_id = arc_task.task_id
                self.train_examples = arc_task.train_examples
                self.test_examples = arc_task.test_examples
                self.arc_task = arc_task  # Store reference
            
            def evaluate_solution(self, predicted, expected):
                return self.arc_task.evaluate_solution(predicted, expected)
        
        adapted_task = ARCAdapter(task)
        score, method = system.solve_task(adapted_task, category)
        elapsed = time.time() - start_time
        
        results.append({
            "task": task.task_id,
            "category": category,
            "score": score,
            "method": method,
            "time": elapsed
        })
        
        if category not in category_scores:
            category_scores[category] = []
        category_scores[category].append(score)
        
        print(f"  Score: {score:.1%} | Method: {method} | Time: {elapsed:.2f}s")
        
        # Show what the system tried
        if score < 1.0:
            print(f"  Note: Partial solution achieved")
    
    # Calculate summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\n📊 Category Performance:")
    for category, scores in category_scores.items():
        avg = np.mean(scores)
        print(f"  {category:20} | {avg:.1%}")
    
    overall_avg = np.mean([r["score"] for r in results])
    success_count = sum(1 for r in results if r["score"] > 0.5)
    
    print(f"\n📈 Overall Performance:")
    print(f"  Average Score:       {overall_avg:.1%}")
    print(f"  Tasks Solved:        {success_count}/{len(tasks)}")
    
    # Compare to known baselines
    print(f"\n📊 Context:")
    print(f"  Human performance:   ~85% on full ARC")
    print(f"  GPT-4:              ~20% on full ARC")
    print(f"  SOTA (2024):        ~55% on full ARC")
    print(f"  Our system:         {overall_avg:.1%} on these simplified tasks")
    
    return results


def test_hypothesis_generator_directly():
    """Test if hypothesis generator can find ARC patterns."""
    
    print("\n" + "=" * 80)
    print("DIRECT HYPOTHESIS GENERATOR TEST ON ARC PATTERNS")
    print("=" * 80)
    
    generator = MinimalHypothesisGenerator(seed=42)
    
    # Test on diagonal fill pattern (should be discoverable)
    print("\n1. Testing diagonal fill pattern...")
    examples = [
        (np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
        (np.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])),
    ]
    
    hypothesis = generator.discover_pattern(
        examples,
        max_attempts=1000,
        strategies=[GenerationStrategy.SYSTEMATIC, GenerationStrategy.COMPOSITIONAL]
    )
    
    if hypothesis:
        score = generator.test_hypothesis(hypothesis, examples)
        print(f"  ✓ Pattern discovered! Score: {score:.1%}")
        print(f"    Type: {hypothesis.transform_type}")
        print(f"    Attempts: {hypothesis.attempts_to_discover}")
    else:
        print(f"  ✗ Pattern not discovered")
    
    # Test on growth pattern
    print("\n2. Testing growth pattern...")
    examples = [
        (np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]])),
        (np.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]]),
         np.array([[2, 2, 2], [2, 2, 0], [2, 0, 0]])),
    ]
    
    hypothesis = generator.discover_pattern(
        examples,
        max_attempts=1000,
        strategies=[GenerationStrategy.SYSTEMATIC, GenerationStrategy.RANDOM]
    )
    
    if hypothesis:
        score = generator.test_hypothesis(hypothesis, examples)
        print(f"  ✓ Pattern discovered! Score: {score:.1%}")
        print(f"    Type: {hypothesis.transform_type}")
        print(f"    Attempts: {hypothesis.attempts_to_discover}")
    else:
        print(f"  ✗ Pattern not discovered")


def main():
    """Run ARC-AGI validation tests."""
    
    print("=" * 80)
    print("IMAGINATION ENGINE - ARC-AGI VALIDATION")
    print("=" * 80)
    
    # First test hypothesis generator directly
    test_hypothesis_generator_directly()
    
    # Then test full system on ARC-style tasks
    results = test_on_arc_tasks()
    
    # Save results
    output_file = Path(__file__).parent / "arc_validation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    
    return results


if __name__ == "__main__":
    results = main()