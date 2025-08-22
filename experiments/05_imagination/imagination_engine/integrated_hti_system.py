"""Integrated HTI System with all components.

Combines the hierarchical planner/executor, adaptive computation,
and transform memory into a complete learnable transform system.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from adaptive_computation import AdaptiveComputationTime, ComputationState
from hierarchical_transform_inventor import HierarchicalTransformInventor
from transform_memory import TransformMemory

# Import benchmark tasks
from core.imagination_benchmark import (
    CounterfactualTasks,
    CreativeProblemTasks,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegratedHTI:
    """Fully integrated HTI system with memory and adaptive computation."""
    
    def __init__(self):
        # Core components
        self.hti = HierarchicalTransformInventor()
        self.memory = TransformMemory(capacity=1000)
        self.act = AdaptiveComputationTime(max_segments=50)
        
        # Learning statistics
        self.tasks_solved = 0
        self.transforms_discovered = 0
        self.total_reasoning_cycles = 0
        
        logger.info("Integrated HTI system initialized")
    
    def solve_with_memory(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        task_name: str = "unknown"
    ) -> Tuple[Callable, Dict]:
        """Solve a task using memory-augmented invention."""
        logger.info(f"Solving task: {task_name}")
        
        # Encode task
        task_encoding = self.hti.encode_task(examples)
        
        # Estimate task complexity
        task_complexity = self._estimate_complexity(examples)
        logger.info(f"Estimated complexity: {task_complexity:.2f}")
        
        # Adapt ACT for this task
        self.act.adapt_for_task_complexity(task_complexity)
        
        # Try to retrieve MANY relevant transforms from memory
        retrieved_transforms = self.memory.retrieve(task_encoding, k=30)
        
        best_transform = None
        best_score = 0.0
        best_info = {}
        
        # First, try retrieved transforms
        if retrieved_transforms:
            logger.info(f"Testing {len(retrieved_transforms)} retrieved transforms")
            
            for stored_transform in retrieved_transforms:
                # Create transform function from stored primitives
                def make_transform(primitives):
                    def transform(grid):
                        result = grid.copy()
                        for prim in primitives:
                            if prim in self.hti.executor.primitives:
                                result = self.hti.executor.primitives[prim](result)
                        return result
                    return transform
                
                transform_fn = make_transform(stored_transform.primitive_sequence)
                
                # Test on first example
                test_input = examples[0][0]
                predicted = transform_fn(test_input)
                score = self.hti.evaluate_transform(predicted, examples[0][1])
                
                if score > best_score:
                    best_score = score
                    best_transform = transform_fn
                    best_info = {
                        'source': 'memory',
                        'transform_id': stored_transform.id,
                        'primitives': stored_transform.primitive_sequence
                    }
                    
                    if score > 0.95:
                        logger.info(f"Found excellent transform in memory: {stored_transform.id}")
                        break
        
        # If memory didn't provide good solution, invent new transform
        if best_score < 0.9:
            logger.info("Memory insufficient, inventing new transform...")
            
            # Use adaptive computation to determine reasoning depth
            improvement_history = []
            reasoning_cycles = 0
            
            while reasoning_cycles < self.act.max_segments:
                # Create computation state
                current_improvement = improvement_history[-1] if improvement_history else 0.0
                comp_state = ComputationState(
                    task_complexity=task_complexity,
                    current_confidence=best_score,
                    cycles_used=reasoning_cycles,
                    improvement_rate=current_improvement,
                    exploration_diversity=0.5
                )
                
                # Check if should halt
                if self.act.should_halt(comp_state):
                    logger.info(f"ACT halting at cycle {reasoning_cycles}")
                    break
                
                # Invent transform with limited cycles
                invented_transform, invention_info = self.hti.invent_transform(
                    examples,
                    max_cycles=20  # EXTREME reasoning depth for M3 Max
                )
                
                # Evaluate
                test_input = examples[0][0]
                predicted = invented_transform(test_input)
                score = self.hti.evaluate_transform(predicted, examples[0][1])
                
                # Track improvement
                improvement = score - best_score if best_score > 0 else score
                improvement_history.append(improvement)
                
                if score > best_score:
                    best_score = score
                    best_transform = invented_transform
                    best_info = invention_info
                    best_info['source'] = 'invented'
                    logger.info(f"New best score: {best_score:.2%}")
                
                reasoning_cycles += 1
                self.total_reasoning_cycles += 1
                
                # Update ACT with reward
                reward = self.act.compute_reward(score, reasoning_cycles, improvement)
                self.act.update_q_function(comp_state, 0, reward)  # 0 = continue action
                
                # Break if excellent solution found
                if score > 0.95:
                    break
            
            # Store successful invention in memory
            if best_score > 0.7 and 'primitives' in best_info:
                primitives = best_info.get('primitives', [])
                if not primitives and 'executor_trace' in best_info:
                    # Extract primitives from trace
                    primitives = []
                    for trace_item in best_info['executor_trace']:
                        primitives.extend(trace_item.get('primitives', []))
                
                if primitives:
                    transform_id = self.memory.add(
                        primitives,
                        task_encoding,
                        best_score,
                        metadata={'task': task_name}
                    )
                    logger.info(f"Stored transform {transform_id} in memory")
                    self.transforms_discovered += 1
        
        # Update statistics
        if best_score > 0.5:
            self.tasks_solved += 1
        
        # Add final info
        best_info.update({
            'final_score': best_score,
            'task_name': task_name,
            'reasoning_cycles': reasoning_cycles if 'reasoning_cycles' in locals() else 0,
            'memory_stats': self.memory.get_statistics()
        })
        
        return best_transform or (lambda x: x), best_info
    
    def _estimate_complexity(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Estimate task complexity from examples."""
        complexities = []
        
        for inp, out in examples[:3]:
            # Shape change complexity
            shape_diff = (inp.shape != out.shape)
            complexities.append(0.3 if shape_diff else 0.0)
            
            # Value change complexity
            if inp.shape == out.shape:
                change_ratio = np.sum(inp != out) / inp.size
                complexities.append(change_ratio)
            
            # Pattern complexity (entropy-like)
            if np.any(out):
                unique_values = len(np.unique(out))
                complexities.append(unique_values / 10.0)
        
        return min(1.0, np.mean(complexities) * 2)
    
    def test_on_negative_counting(self) -> Dict:
        """Test HTI on the negative counting task."""
        print("\n" + "=" * 60)
        print("TESTING HTI ON NEGATIVE COUNTING")
        print("=" * 60)
        
        # Create the negative counting task
        task = CounterfactualTasks.create_negative_counting()
        
        # Extract examples
        examples = task.train_examples
        
        print("Task: Count objects, but negative values mean impossible")
        print(f"Training shows: n → n+1 (increment by 1)")
        print(f"Test asks: 1 → -1 (requires understanding impossibility)")
        
        # Solve with HTI
        transform, info = self.solve_with_memory(examples, "negative_counting")
        
        # Test on the actual test case
        test_input = task.test_examples[0][0]
        predicted = transform(test_input)
        expected = task.test_examples[0][1]
        
        print(f"\nTest input (1 object):")
        print(test_input)
        print(f"\nExpected (negative/impossible):")
        print(expected)
        print(f"\nPredicted:")
        print(predicted)
        
        # Calculate score
        score = np.mean(predicted == expected) if predicted.shape == expected.shape else 0.0
        
        print(f"\nScore: {score:.1%}")
        print(f"Source: {info.get('source', 'unknown')}")
        print(f"Reasoning cycles: {info.get('reasoning_cycles', 0)}")
        
        return {
            'task': 'negative_counting',
            'score': score,
            'info': info
        }
    
    def test_on_creative_sorting(self) -> Dict:
        """Test HTI on the creative sorting task."""
        print("\n" + "=" * 60)
        print("TESTING HTI ON CREATIVE SORTING")
        print("=" * 60)
        
        # Create the creative sorting task
        task = CreativeProblemTasks.create_sort_without_compare()
        
        # Extract examples
        examples = task.train_examples
        
        print("Task: Sort without comparison")
        print("Must invent novel sorting algorithm")
        
        # Solve with HTI
        transform, info = self.solve_with_memory(examples, "creative_sorting")
        
        # Test on the actual test case
        test_input = task.test_examples[0][0]
        predicted = transform(test_input)
        expected = task.test_examples[0][1]
        
        print(f"\nTest input (unsorted):")
        print(test_input)
        print(f"\nExpected (sorted):")
        print(expected)
        print(f"\nPredicted:")
        print(predicted)
        
        # Calculate score (partial credit for order)
        if predicted.shape == expected.shape:
            # Check if values are in ascending order
            flat_predicted = predicted.flatten()
            flat_expected = expected.flatten()
            
            # Count correctly ordered pairs
            correct_pairs = 0
            total_pairs = 0
            
            for i in range(len(flat_predicted)):
                for j in range(i + 1, len(flat_predicted)):
                    if flat_expected[i] <= flat_expected[j]:
                        if flat_predicted[i] <= flat_predicted[j]:
                            correct_pairs += 1
                        total_pairs += 1
            
            score = correct_pairs / total_pairs if total_pairs > 0 else 0.0
        else:
            score = 0.0
        
        print(f"\nScore: {score:.1%}")
        print(f"Source: {info.get('source', 'unknown')}")
        print(f"Reasoning cycles: {info.get('reasoning_cycles', 0)}")
        
        return {
            'task': 'creative_sorting',
            'score': score,
            'info': info
        }
    
    def get_statistics(self) -> Dict:
        """Get overall system statistics."""
        return {
            'tasks_solved': self.tasks_solved,
            'transforms_discovered': self.transforms_discovered,
            'total_reasoning_cycles': self.total_reasoning_cycles,
            'average_cycles': self.total_reasoning_cycles / max(1, self.tasks_solved),
            'memory_stats': self.memory.get_statistics(),
            'act_stats': self.act.get_statistics()
        }


def main():
    """Test integrated HTI on failed benchmark tasks."""
    print("\n" + "=" * 80)
    print("INTEGRATED HTI SYSTEM - TESTING ON FAILED TASKS")
    print("=" * 80)
    
    # Create integrated system
    system = IntegratedHTI()
    
    # Test on negative counting (currently 0%)
    result1 = system.test_on_negative_counting()
    
    # Test on creative sorting (currently 0%)
    result2 = system.test_on_creative_sorting()
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\nNegative Counting:")
    print(f"  Previous: 0%")
    print(f"  HTI Score: {result1['score']:.1%}")
    print(f"  Improvement: {result1['score']:.1%}")
    
    print(f"\nCreative Sorting:")
    print(f"  Previous: 0%")
    print(f"  HTI Score: {result2['score']:.1%}")
    print(f"  Improvement: {result2['score']:.1%}")
    
    # System statistics
    stats = system.get_statistics()
    print(f"\nSystem Statistics:")
    print(f"  Tasks attempted: 2")
    print(f"  Tasks solved (>50%): {stats['tasks_solved']}")
    print(f"  Transforms discovered: {stats['transforms_discovered']}")
    print(f"  Average reasoning cycles: {stats['average_cycles']:.1f}")
    
    # Memory statistics
    mem_stats = stats['memory_stats']
    print(f"\nMemory Statistics:")
    print(f"  Total transforms: {mem_stats.get('total_transforms', 0)}")
    print(f"  Capacity used: {mem_stats.get('capacity_used', 0):.1%}")
    
    # Overall assessment
    print("\n" + "-" * 60)
    if result1['score'] > 0 or result2['score'] > 0:
        print("✅ HTI shows improvement on previously failed tasks!")
        print("   The learnable transform space is working!")
    else:
        print("📝 HTI needs more training or semantic understanding")
        print("   Consider adding LLM bridge for these semantic tasks")
    
    return result1, result2


if __name__ == "__main__":
    results = main()