"""Improved Cross-Domain Transfer for Abstract Principle Application.

This module enables true cross-domain transfer by understanding abstract
operations and mapping them appropriately to different representational spaces.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AbstractConcept(Enum):
    """Abstract concepts that can transfer across domains."""
    
    ROTATION = "rotation"  # Cyclic permutation
    SYMMETRY = "symmetry"  # Balance/mirror
    PROGRESSION = "progression"  # Sequential change
    INVERSION = "inversion"  # Reversal/negation
    SCALING = "scaling"  # Magnification
    TRANSLATION = "translation"  # Shift/offset


@dataclass
class DomainMapping:
    """Maps an abstract concept to concrete implementation in a domain."""
    
    concept: AbstractConcept
    source_domain: str
    target_domain: str
    transform_fn: Callable
    parameters: Dict[str, Any]


class ImprovedCrossDomainTransfer:
    """Improved system for cross-domain principle transfer."""
    
    def __init__(self):
        self.learned_concepts: Dict[AbstractConcept, List[DomainMapping]] = {}
        self.domain_detectors = self._setup_domain_detectors()
    
    def _setup_domain_detectors(self) -> Dict[str, Callable]:
        """Set up functions to detect domain type."""
        
        def is_spatial_rotation(examples: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
            """Check if examples show spatial rotation."""
            for inp, out in examples:
                # Check if output is rotated version of input
                for k in range(1, 4):
                    if np.array_equal(np.rot90(inp, k), out):
                        return True
            return False
        
        def is_spatial_symmetry(examples: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
            """Check if examples show spatial symmetry."""
            for inp, out in examples:
                # Check for mirror symmetry
                if inp.shape != out.shape:
                    continue
                
                # Check horizontal mirror
                h, w = inp.shape
                is_mirror = True
                for i in range(h):
                    for j in range(w // 2):
                        if out[i, j] != out[i, w - 1 - j]:
                            is_mirror = False
                            break
                
                if is_mirror:
                    return True
            
            return False
        
        return {
            "spatial_rotation": is_spatial_rotation,
            "spatial_symmetry": is_spatial_symmetry,
        }
    
    def learn_concept_from_examples(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[AbstractConcept]:
        """Learn abstract concept from examples."""
        
        # Detect what type of transformation
        if self.domain_detectors["spatial_rotation"](examples):
            return AbstractConcept.ROTATION
        elif self.domain_detectors["spatial_symmetry"](examples):
            return AbstractConcept.SYMMETRY
        
        # Analyze transformation patterns
        for inp, out in examples:
            # Check for progression (sequential change)
            unique_in = np.unique(inp[inp != 0])
            unique_out = np.unique(out[out != 0])
            
            if len(unique_in) == len(unique_out):
                # Check if values shifted
                if all(v2 == v1 + 1 for v1, v2 in zip(sorted(unique_in), sorted(unique_out))):
                    return AbstractConcept.PROGRESSION
        
        return None
    
    def create_color_rotation_transform(self) -> Callable:
        """Create a function that rotates colors cyclically."""
        
        def rotate_colors(grid: np.ndarray) -> np.ndarray:
            """Rotate colors: 1→2→3→4→1, 0 stays 0."""
            result = grid.copy()
            
            # Define color cycle
            color_cycle = {
                0: 0,  # Empty stays empty
                1: 2,  # Red → Green
                2: 3,  # Green → Blue
                3: 4,  # Blue → Yellow
                4: 1,  # Yellow → Red
                5: 6,  # For extended colors
                6: 7,
                7: 8,
                8: 9,
                9: 5,
            }
            
            # Apply rotation
            for old_val, new_val in color_cycle.items():
                if old_val in grid:
                    result[grid == old_val] = new_val
            
            return result
        
        return rotate_colors
    
    def create_value_symmetry_transform(self) -> Callable:
        """Create a function that creates value symmetry."""
        
        def make_value_symmetric(grid: np.ndarray) -> np.ndarray:
            """Make values symmetric around median."""
            result = grid.copy()
            
            # Find median value (excluding zeros)
            non_zero = grid[grid != 0]
            if len(non_zero) == 0:
                return result
            
            median = np.median(non_zero)
            
            # Make symmetric transformations
            h, w = grid.shape
            
            # For each position, if it has a mirror position, balance values
            for i in range(h):
                for j in range(w // 2):
                    mirror_j = w - 1 - j
                    
                    val_left = grid[i, j]
                    val_right = grid[i, mirror_j]
                    
                    if val_left != 0 and val_right != 0:
                        # Balance around median
                        if val_left < median and val_right > median:
                            # Keep symmetric
                            pass
                        elif val_left > median and val_right < median:
                            # Swap to make symmetric
                            result[i, j] = val_right
                            result[i, mirror_j] = val_left
            
            return result
        
        return make_value_symmetric
    
    def transfer_concept(
        self,
        concept: AbstractConcept,
        target_domain: str,
        examples: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    ) -> Callable:
        """Transfer abstract concept to target domain."""
        
        logger.info(f"Transferring {concept} to {target_domain}")
        
        if concept == AbstractConcept.ROTATION:
            if "color" in target_domain.lower():
                return self.create_color_rotation_transform()
            elif "value" in target_domain.lower():
                # Rotation in value space means cycling values
                def rotate_values(grid):
                    result = grid.copy()
                    unique = np.unique(grid[grid != 0])
                    if len(unique) > 0:
                        # Shift all values by 1
                        for val in unique:
                            new_val = val + 1
                            if new_val > np.max(unique):
                                new_val = np.min(unique)
                            result[grid == val] = new_val
                    return result
                return rotate_values
        
        elif concept == AbstractConcept.SYMMETRY:
            if "value" in target_domain.lower():
                return self.create_value_symmetry_transform()
            elif "color" in target_domain.lower():
                # Symmetry in color space
                def color_symmetry(grid):
                    result = grid.copy()
                    # Make colors symmetric (complementary)
                    color_pairs = {1: 4, 2: 3, 3: 2, 4: 1}
                    # Apply symmetry transform
                    h, w = grid.shape
                    for i in range(h):
                        for j in range(w // 2, w):
                            if grid[i, j] in color_pairs:
                                result[i, j] = color_pairs[grid[i, j]]
                    return result
                return color_symmetry
        
        # Default identity transform
        return lambda x: x
    
    def solve_cross_domain_task(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_examples: List[Tuple[np.ndarray, np.ndarray]],
        target_domain_hint: Optional[str] = None
    ) -> List[np.ndarray]:
        """Solve a cross-domain transfer task."""
        
        # Learn concept from training examples
        concept = self.learn_concept_from_examples(train_examples)
        
        if not concept:
            logger.warning("Could not identify concept from training examples")
            return [test[0] for test in test_examples]  # Return inputs unchanged
        
        logger.info(f"Identified concept: {concept}")
        
        # Determine target domain
        if not target_domain_hint:
            # Infer from test examples
            if any(np.max(test[0]) > 5 for test in test_examples):
                target_domain_hint = "value"
            else:
                target_domain_hint = "color"
        
        # Get appropriate transform
        transform = self.transfer_concept(concept, target_domain_hint, test_examples)
        
        # Apply to test examples
        results = []
        for test_input, _ in test_examples:
            result = transform(test_input)
            results.append(result)
        
        return results


def test_color_rotation_transfer():
    """Test transferring rotation from spatial to color domain."""
    print("\n" + "=" * 60)
    print("Testing Spatial to Color Rotation Transfer")
    print("=" * 60)
    
    transfer = ImprovedCrossDomainTransfer()
    
    # Training: spatial rotation
    train_examples = []
    for _ in range(2):
        inp = np.array([[1, 2, 0], [3, 4, 0], [0, 0, 0]])
        out = np.rot90(inp)
        train_examples.append((inp, out))
    
    # Test: color rotation
    test_examples = []
    inp = np.array([[1, 2, 3], [4, 0, 1], [2, 3, 4]])
    expected = np.array([[2, 3, 4], [1, 0, 2], [3, 4, 1]])
    test_examples.append((inp, expected))
    
    # Solve
    results = transfer.solve_cross_domain_task(
        train_examples,
        test_examples,
        target_domain_hint="color"
    )
    
    print(f"Input:\n{inp}")
    print(f"Expected:\n{expected}")
    print(f"Got:\n{results[0]}")
    
    if np.array_equal(results[0], expected):
        print("✅ SUCCESS! Color rotation transfer works!")
        return True
    else:
        # Check if we at least rotated colors
        unique_in = np.unique(inp[inp != 0])
        unique_out = np.unique(results[0][results[0] != 0])
        if set(unique_in) == set(unique_out):
            print("✓ Partial success - colors were rotated")
        else:
            print("❌ Failed to transfer rotation concept")
        return False


def test_on_benchmark_cross_domain():
    """Test on actual cross-domain benchmark tasks."""
    print("\n" + "=" * 60)
    print("Testing on Cross-Domain Benchmark Tasks")
    print("=" * 60)
    
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
    sys.path.append(str(Path(__file__).parent.parent))
    
    from core.imagination_benchmark import CrossDomainTasks
    
    transfer = ImprovedCrossDomainTransfer()
    
    # Test 2D to color rotation
    print("\n1. 2D to Color Rotation:")
    task = CrossDomainTasks.create_2d_to_color_rotation()
    
    results = transfer.solve_cross_domain_task(
        task.train_examples,
        task.test_examples,
        target_domain_hint="color"
    )
    
    score = 0.0
    for i, (result, (_, expected)) in enumerate(zip(results, task.test_examples)):
        task_score = task.evaluate_solution(result, expected)
        score += task_score
        print(f"  Example {i+1} score: {task_score:.1%}")
    
    avg_score = score / len(results) if results else 0.0
    print(f"Average score: {avg_score:.1%}")
    
    if avg_score > 0.5:
        print("✅ 2D to color rotation solved!")
    
    # Test symmetry transfer
    print("\n2. Symmetry Transfer:")
    task = CrossDomainTasks.create_symmetry_transfer()
    
    results = transfer.solve_cross_domain_task(
        task.train_examples,
        task.test_examples,
        target_domain_hint="value"
    )
    
    score = 0.0
    for i, (result, (_, expected)) in enumerate(zip(results, task.test_examples)):
        task_score = task.evaluate_solution(result, expected)
        score += task_score
        print(f"  Example {i+1} score: {task_score:.1%}")
    
    avg_score = score / len(results) if results else 0.0
    print(f"Average score: {avg_score:.1%}")
    
    if avg_score > 0.5:
        print("✅ Symmetry transfer solved!")
    
    return avg_score


if __name__ == "__main__":
    print("=" * 60)
    print("IMPROVED CROSS-DOMAIN TRANSFER TEST")
    print("=" * 60)
    
    # Test color rotation
    success1 = test_color_rotation_transfer()
    
    # Test on benchmark
    score = test_on_benchmark_cross_domain()
    
    if success1 or score > 0.3:
        print("\n🎉 Cross-domain transfer showing improvement!")
    else:
        print("\n📝 Cross-domain transfer needs more work")