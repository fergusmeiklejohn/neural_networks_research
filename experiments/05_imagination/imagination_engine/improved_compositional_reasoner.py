"""Improved Compositional Reasoner for multi-attribute transformations.

This version specifically handles:
1. Simultaneous color and size changes
2. Multiple conditional rules
3. Learning from separated examples
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Transformation:
    """Represents a learned transformation."""
    
    name: str
    transform_type: str  # 'color', 'size', 'conditional'
    parameters: Dict[str, Any]
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        """Apply transformation to a grid."""
        if self.transform_type == "color":
            return self._apply_color_change(grid)
        elif self.transform_type == "size":
            return self._apply_size_change(grid)
        elif self.transform_type == "conditional":
            return self._apply_conditional(grid)
        else:
            return grid
    
    def _apply_color_change(self, grid: np.ndarray) -> np.ndarray:
        """Apply color transformation."""
        color_map = self.parameters.get("color_map", {})
        result = grid.copy()
        
        for old_color, new_color in color_map.items():
            result[grid == old_color] = new_color
        
        return result
    
    def _apply_size_change(self, grid: np.ndarray) -> np.ndarray:
        """Apply size transformation."""
        scale = self.parameters.get("scale", 1)
        
        if scale == 1:
            return grid
        
        # Simple scaling by repetition
        h, w = grid.shape
        new_h, new_w = h * scale, w * scale
        result = np.zeros((new_h, new_w), dtype=grid.dtype)
        
        for i in range(h):
            for j in range(w):
                result[i*scale:(i+1)*scale, j*scale:(j+1)*scale] = grid[i, j]
        
        return result
    
    def _apply_conditional(self, grid: np.ndarray) -> np.ndarray:
        """Apply conditional transformation."""
        condition = self.parameters.get("condition")
        action = self.parameters.get("action")
        
        if not condition or not action:
            return grid
        
        result = grid.copy()
        
        # Apply condition and action
        if condition(grid):
            result = action(result)
        
        return result


class ImprovedCompositionalReasoner:
    """Improved version that can learn and combine transformations."""
    
    def __init__(self):
        self.learned_transformations: List[Transformation] = []
        self.combined_transformations: List[List[Transformation]] = []
    
    def learn_from_examples(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Transformation]:
        """Learn transformations from examples."""
        
        transformations = []
        
        for inp, out in examples:
            # Detect type of transformation
            transform = self._detect_transformation(inp, out)
            if transform:
                transformations.append(transform)
        
        # Consolidate similar transformations
        consolidated = self._consolidate_transformations(transformations)
        self.learned_transformations.extend(consolidated)
        
        return consolidated
    
    def _detect_transformation(
        self,
        inp: np.ndarray,
        out: np.ndarray
    ) -> Optional[Transformation]:
        """Detect what transformation occurred."""
        
        # Check for color change
        if inp.shape == out.shape:
            unique_in = np.unique(inp[inp != 0])
            unique_out = np.unique(out[out != 0])
            
            if len(unique_in) == 1 and len(unique_out) == 1 and unique_in[0] != unique_out[0]:
                # Simple color change
                return Transformation(
                    name=f"color_{unique_in[0]}_to_{unique_out[0]}",
                    transform_type="color",
                    parameters={"color_map": {int(unique_in[0]): int(unique_out[0])}}
                )
        
        # Check for size change
        if out.shape[0] > inp.shape[0] or out.shape[1] > inp.shape[1]:
            # Size increased
            scale_h = out.shape[0] // inp.shape[0]
            scale_w = out.shape[1] // inp.shape[1]
            
            if scale_h == scale_w:
                # Uniform scaling
                # Verify it's actually a scale by checking pattern
                is_scale = True
                for i in range(inp.shape[0]):
                    for j in range(inp.shape[1]):
                        expected = inp[i, j]
                        actual_region = out[i*scale_h:(i+1)*scale_h, j*scale_w:(j+1)*scale_w]
                        if not np.all(actual_region == expected):
                            is_scale = False
                            break
                
                if is_scale:
                    return Transformation(
                        name=f"scale_{scale_h}x",
                        transform_type="size",
                        parameters={"scale": scale_h}
                    )
        
        # Check for conditional patterns
        if self._is_conditional_pattern(inp, out):
            condition, action = self._extract_conditional(inp, out)
            if condition and action:
                return Transformation(
                    name="conditional_rule",
                    transform_type="conditional",
                    parameters={"condition": condition, "action": action}
                )
        
        return None
    
    def _is_conditional_pattern(self, inp: np.ndarray, out: np.ndarray) -> bool:
        """Check if this looks like a conditional transformation."""
        # Simple heuristic: output has more non-zero elements in specific positions
        return np.sum(out != 0) > np.sum(inp != 0)
    
    def _extract_conditional(
        self,
        inp: np.ndarray,
        out: np.ndarray
    ) -> Tuple[Optional[Callable], Optional[Callable]]:
        """Extract conditional rule from example."""
        
        # Check for center-based rule
        if inp.shape[0] >= 3 and inp.shape[1] >= 3:
            center = (inp.shape[0] // 2, inp.shape[1] // 2)
            if inp[center] == 1:
                # Check if corners changed
                corners_changed = (
                    out[0, 0] != inp[0, 0] or
                    out[0, -1] != inp[0, -1] or
                    out[-1, 0] != inp[-1, 0] or
                    out[-1, -1] != inp[-1, -1]
                )
                
                if corners_changed:
                    def condition(g):
                        c = (g.shape[0] // 2, g.shape[1] // 2)
                        return g[c] == 1
                    
                    def action(g):
                        result = g.copy()
                        # Fill corners
                        result[0, 0] = 1
                        result[0, -1] = 1
                        result[-1, 0] = 1
                        result[-1, -1] = 1
                        return result
                    
                    return condition, action
        
        return None, None
    
    def _consolidate_transformations(
        self,
        transformations: List[Transformation]
    ) -> List[Transformation]:
        """Consolidate similar transformations into general rules."""
        
        consolidated = []
        
        # Group by type
        by_type = {}
        for t in transformations:
            if t.transform_type not in by_type:
                by_type[t.transform_type] = []
            by_type[t.transform_type].append(t)
        
        # Consolidate each type
        for transform_type, transforms in by_type.items():
            if transform_type == "color":
                # Merge color mappings
                color_map = {}
                for t in transforms:
                    color_map.update(t.parameters.get("color_map", {}))
                
                if color_map:
                    consolidated.append(Transformation(
                        name="color_mapping",
                        transform_type="color",
                        parameters={"color_map": color_map}
                    ))
            
            elif transform_type == "size":
                # Take the most common scale
                scales = [t.parameters.get("scale", 1) for t in transforms]
                if scales:
                    most_common_scale = max(set(scales), key=scales.count)
                    consolidated.append(Transformation(
                        name=f"scale_{most_common_scale}x",
                        transform_type="size",
                        parameters={"scale": most_common_scale}
                    ))
            
            else:
                # Keep all other transformations
                consolidated.extend(transforms)
        
        return consolidated
    
    def combine_transformations(
        self,
        transformations: List[Transformation]
    ) -> Callable:
        """Combine multiple transformations into a single function."""
        
        def combined_transform(grid: np.ndarray) -> np.ndarray:
            result = grid.copy()
            
            # Apply transformations in order
            # Size changes should typically come last
            size_transforms = [t for t in transformations if t.transform_type == "size"]
            other_transforms = [t for t in transformations if t.transform_type != "size"]
            
            # Apply non-size transformations first
            for transform in other_transforms:
                result = transform.apply(result)
            
            # Apply size transformations last
            for transform in size_transforms:
                result = transform.apply(result)
            
            return result
        
        return combined_transform
    
    def solve_combination_task(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray
    ) -> np.ndarray:
        """Solve a rule combination task by learning from training examples."""
        
        # Learn transformations from training examples
        transformations = self.learn_from_examples(train_examples)
        
        if not transformations:
            logger.warning("No transformations learned")
            return test_input
        
        logger.info(f"Learned {len(transformations)} transformations:")
        for t in transformations:
            logger.info(f"  - {t.name}: {t.transform_type}")
        
        # Check if we need to combine transformations
        if len(transformations) > 1:
            # Create combined transformation
            combined = self.combine_transformations(transformations)
            result = combined(test_input)
            logger.info("Applied combined transformation")
        else:
            # Apply single transformation
            result = transformations[0].apply(test_input)
            logger.info(f"Applied single transformation: {transformations[0].name}")
        
        return result
    
    def solve_conditional_task(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray
    ) -> np.ndarray:
        """Solve a conditional combination task."""
        
        # Learn all conditional rules
        rules = []
        for inp, out in train_examples:
            transform = self._detect_transformation(inp, out)
            if transform and transform.transform_type == "conditional":
                rules.append(transform)
        
        # Apply all applicable rules to test input
        result = test_input.copy()
        
        for rule in rules:
            result = rule.apply(result)
        
        return result


def test_color_size_combination():
    """Test on the color-size combination task."""
    print("\n" + "=" * 60)
    print("Testing Improved Color-Size Combination")
    print("=" * 60)
    
    reasoner = ImprovedCompositionalReasoner()
    
    # Create training examples
    train_examples = []
    
    # Color change examples
    for _ in range(2):
        inp = np.array([[1, 1], [1, 1]])
        out = np.array([[2, 2], [2, 2]])
        train_examples.append((inp, out))
    
    # Size change examples
    for _ in range(2):
        inp = np.array([[3, 3], [3, 3]])
        out = np.array([[3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]])
        train_examples.append((inp, out))
    
    # Test: combine both transformations
    test_input = np.array([[1, 1], [1, 1]])
    expected = np.array([[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]])
    
    result = reasoner.solve_combination_task(train_examples, test_input)
    
    print(f"Test input:\n{test_input}")
    print(f"Expected:\n{expected}")
    print(f"Got:\n{result}")
    
    if np.array_equal(result, expected):
        print("✅ SUCCESS! Color-size combination solved!")
        return True
    else:
        print("❌ Failed to combine transformations correctly")
        return False


def test_on_benchmark():
    """Test improved reasoner on actual benchmark tasks."""
    print("\n" + "=" * 60)
    print("Testing on Benchmark Tasks")
    print("=" * 60)
    
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
    sys.path.append(str(Path(__file__).parent.parent))
    
    from core.imagination_benchmark import RuleCombinationTasks
    
    # Test color-size task
    print("\n1. Color-Size Combination Task:")
    task = RuleCombinationTasks.create_color_size_combo()
    reasoner = ImprovedCompositionalReasoner()
    
    # Use train examples to learn, test on test examples
    test_input, expected = task.test_examples[0]
    result = reasoner.solve_combination_task(task.train_examples, test_input)
    
    score = task.evaluate_solution(result, expected)
    print(f"Score: {score:.1%}")
    
    if score > 0.8:
        print("✅ Color-size task solved!")
    
    # Test conditional task
    print("\n2. Conditional Combination Task:")
    task = RuleCombinationTasks.create_conditional_combo()
    reasoner = ImprovedCompositionalReasoner()
    
    test_input, expected = task.test_examples[0]
    result = reasoner.solve_conditional_task(task.train_examples, test_input)
    
    score = task.evaluate_solution(result, expected)
    print(f"Score: {score:.1%}")
    
    if score > 0.8:
        print("✅ Conditional task solved!")
    
    return score


if __name__ == "__main__":
    print("=" * 60)
    print("IMPROVED COMPOSITIONAL REASONER TEST")
    print("=" * 60)
    
    # Test basic color-size combination
    success1 = test_color_size_combination()
    
    # Test on actual benchmark
    score = test_on_benchmark()
    
    if success1:
        print("\n🎉 Major improvement in compositional reasoning!")
    else:
        print("\n📝 Still needs refinement")