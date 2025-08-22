"""Program Synthesis DSL for ARC tasks.

This module defines a domain-specific language for composing primitives
into complex programs that can solve ARC tasks.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from arc_primitives import ARCPrimitives, Component, Region
import copy


class Transform(ABC):
    """Base class for all transformations."""
    
    @abstractmethod
    def apply(self, grid: np.ndarray) -> np.ndarray:
        """Apply this transformation to a grid."""
        pass
    
    @abstractmethod
    def to_string(self) -> str:
        """Return a human-readable string representation."""
        pass


class Primitive(Transform):
    """Atomic operation from our primitive library."""
    
    def __init__(self, name: str, func: Callable, params: Dict[str, Any]):
        self.name = name
        self.func = func
        self.params = params
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        """Apply the primitive operation."""
        # If func only takes grid, don't pass params
        import inspect
        sig = inspect.signature(self.func)
        if len(sig.parameters) == 1:
            return self.func(grid)
        else:
            return self.func(grid, **self.params)
    
    def to_string(self) -> str:
        param_str = ', '.join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({param_str})"


class Sequence(Transform):
    """Sequential composition: T1 then T2 then ..."""
    
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        """Apply transforms in sequence."""
        result = grid
        for transform in self.transforms:
            result = transform.apply(result)
        return result
    
    def to_string(self) -> str:
        steps = ' -> '.join(t.to_string() for t in self.transforms)
        return f"Sequence[{steps}]"


class Conditional(Transform):
    """If-then-else logic."""
    
    def __init__(self, condition: Callable[[np.ndarray], bool], 
                 then_branch: Transform, 
                 else_branch: Optional[Transform] = None):
        self.condition = condition
        self.then_branch = then_branch
        self.else_branch = else_branch
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        """Apply conditional transformation."""
        if self.condition(grid):
            return self.then_branch.apply(grid)
        elif self.else_branch:
            return self.else_branch.apply(grid)
        return grid
    
    def to_string(self) -> str:
        else_str = f" else {self.else_branch.to_string()}" if self.else_branch else ""
        return f"If(condition) then {self.then_branch.to_string()}{else_str}"


class Loop(Transform):
    """Repeat transformation until condition or fixed number of times."""
    
    def __init__(self, body: Transform, 
                 condition: Optional[Callable[[np.ndarray], bool]] = None,
                 max_iterations: int = 10):
        self.body = body
        self.condition = condition
        self.max_iterations = max_iterations
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        """Apply loop transformation."""
        result = grid
        for i in range(self.max_iterations):
            if self.condition and not self.condition(result):
                break
            new_result = self.body.apply(result)
            if np.array_equal(new_result, result):
                # Fixed point reached
                break
            result = new_result
        return result
    
    def to_string(self) -> str:
        return f"Loop({self.body.to_string()}, max={self.max_iterations})"


class ForEach(Transform):
    """Apply transformation to each object/component."""
    
    def __init__(self, object_finder: Callable[[np.ndarray], List[Any]],
                 transform: Transform):
        self.object_finder = object_finder
        self.transform = transform
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        """Apply transformation to each found object."""
        result = grid.copy()
        objects = self.object_finder(grid)
        
        for obj in objects:
            # Create a sub-grid for this object
            if isinstance(obj, Component):
                # Extract component as sub-grid
                min_r, min_c, max_r, max_c = obj.bounding_box
                sub_grid = grid[min_r:max_r+1, min_c:max_c+1].copy()
                
                # Apply transformation
                transformed = self.transform.apply(sub_grid)
                
                # Put back in result
                result[min_r:max_r+1, min_c:max_c+1] = transformed
        
        return result
    
    def to_string(self) -> str:
        return f"ForEach(objects, {self.transform.to_string()})"


class Let(Transform):
    """Variable binding for reuse."""
    
    def __init__(self, var_name: str, value: Transform, body: Transform):
        self.var_name = var_name
        self.value = value
        self.body = body
        self._stored_value = None
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        """Apply with variable binding."""
        # Compute and store value
        self._stored_value = self.value.apply(grid)
        # Apply body (which may reference the stored value)
        return self.body.apply(grid)
    
    def to_string(self) -> str:
        return f"Let {self.var_name} = {self.value.to_string()} in {self.body.to_string()}"


class ProgramSynthesizer:
    """Synthesizes programs to solve ARC tasks."""
    
    def __init__(self):
        self.primitive_library = self._build_primitive_library()
        self.synthesized_programs = []
    
    def _build_primitive_library(self) -> List[Primitive]:
        """Build library of available primitives."""
        library = []
        
        # Object detection primitives
        library.append(Primitive(
            "find_components",
            lambda g, **kw: ARCPrimitives.find_connected_components(g, **kw),
            {}
        ))
        
        # Region operations
        library.append(Primitive(
            "fill_enclosed",
            lambda g, boundary, fill: ARCPrimitives.fill_enclosed_regions(g, boundary, fill),
            {'boundary': 3, 'fill': 4}  # Common defaults
        ))
        
        library.append(Primitive(
            "flood_fill",
            lambda g, r, c, color: ARCPrimitives.flood_fill(g, r, c, color),
            {'r': 0, 'c': 0, 'color': 4}
        ))
        
        # Pattern operations
        library.append(Primitive(
            "tile_regular",
            lambda g: ARCPrimitives.tile_pattern(g, (g.shape[0]*3, g.shape[1]*3), 'regular'),
            {}
        ))
        
        # Add more primitives as needed...
        
        return library
    
    def synthesize(self, examples: List[Tuple[np.ndarray, np.ndarray]], 
                  max_depth: int = 3) -> Optional[Transform]:
        """Synthesize a program that solves the given examples."""
        
        # Start with primitive operations
        candidates = self._enumerate_primitives(examples)
        
        # Evaluate primitives
        best_program = None
        best_score = 0.0
        
        for candidate in candidates:
            score = self._evaluate_program(candidate, examples)
            if score > best_score:
                best_score = score
                best_program = candidate
                if score == 1.0:
                    print(f"Found perfect solution: {candidate.to_string()}")
                    return candidate
        
        # Try compositions if no primitive works perfectly
        if best_score < 1.0 and max_depth > 1:
            composed_candidates = self._enumerate_compositions(candidates, examples, max_depth)
            
            for candidate in composed_candidates:
                score = self._evaluate_program(candidate, examples)
                if score > best_score:
                    best_score = score
                    best_program = candidate
                    if score == 1.0:
                        print(f"Found perfect composition: {candidate.to_string()}")
                        return candidate
        
        if best_program and best_score > 0.5:
            print(f"Best program (score={best_score:.2f}): {best_program.to_string()}")
            return best_program
        
        return None
    
    def _enumerate_primitives(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> List[Transform]:
        """Enumerate primitive operations with parameter variations."""
        candidates = []
        
        # Analyze ALL examples to find consistent patterns
        all_input_colors = set()
        all_output_colors = set()
        
        for input_grid, output_grid in examples:
            all_input_colors.update(np.unique(input_grid))
            all_output_colors.update(np.unique(output_grid))
        
        # Colors that appear in output but not input (or more frequently)
        new_colors = all_output_colors - {0}  # Include colors that might already exist
        boundary_colors = all_input_colors - {0}
        
        # Try fill operations with different color combinations
        for boundary_color in boundary_colors:
            for fill_color in all_output_colors:
                if fill_color != boundary_color and fill_color != 0:
                    # Create a closure properly
                    def make_fill_func(b, f):
                        return lambda g: ARCPrimitives.fill_enclosed_regions(g, b, f)
                    
                    candidates.append(Primitive(
                        f"fill_enclosed_{boundary_color}_{fill_color}",
                        make_fill_func(boundary_color, fill_color),
                        {'boundary': boundary_color, 'fill': fill_color}
                    ))
        
        # Also add simple replace operations
        for color_from in all_input_colors:
            for color_to in all_output_colors:
                if color_from != color_to:
                    def make_replace_func(cf, ct):
                        return lambda g: np.where(g == cf, ct, g)
                    
                    candidates.append(Primitive(
                        f"replace_{color_from}_with_{color_to}",
                        make_replace_func(color_from, color_to),
                        {'from': color_from, 'to': color_to}
                    ))
        
        # Try flood fill from different positions (use first example for shape)
        input_grid = examples[0][0]
        h, w = input_grid.shape
        for r in range(0, h, h//3 if h > 3 else 1):
            for c in range(0, w, w//3 if w > 3 else 1):
                if input_grid[r, c] == 0:  # Empty cell
                    for color in new_colors:
                        candidates.append(Primitive(
                            "flood_fill",
                            lambda g, rr=r, cc=c, col=color:
                                ARCPrimitives.flood_fill(g, rr, cc, col),
                            {'r': r, 'c': c, 'color': color}
                        ))
        
        # Try tiling if output is larger than input
        if output_grid.shape[0] > input_grid.shape[0]:
            candidates.append(Primitive(
                "tile_3x3",
                lambda g: self._custom_tile_3x3(g),
                {}
            ))
        
        return candidates
    
    def _custom_tile_3x3(self, grid: np.ndarray) -> np.ndarray:
        """Custom 3x3 tiling for ARC tasks."""
        h, w = grid.shape
        result = np.zeros((3*h, 3*w), dtype=grid.dtype)
        
        # Common ARC pattern: place at corners and center
        positions = [
            (0, 0), (0, 2*w), (h, w), (2*h, 0), (2*h, 2*w)
        ]
        
        for r, c in positions:
            result[r:r+h, c:c+w] = grid
        
        return result
    
    def _enumerate_compositions(self, primitives: List[Transform], 
                               examples: List[Tuple[np.ndarray, np.ndarray]],
                               max_depth: int) -> List[Transform]:
        """Enumerate compositions of primitives."""
        compositions = []
        
        # Try sequences of 2 primitives
        for p1 in primitives[:10]:  # Limit to avoid explosion
            for p2 in primitives[:10]:
                compositions.append(Sequence([p1, p2]))
        
        # Try conditional compositions if we detect pattern
        if self._looks_conditional(examples):
            # Simple heuristic: if input has multiple colors
            for p1 in primitives[:5]:
                for p2 in primitives[:5]:
                    condition = lambda g: len(np.unique(g)) > 2
                    compositions.append(Conditional(condition, p1, p2))
        
        return compositions
    
    def _looks_conditional(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """Heuristic to detect if task might need conditional logic."""
        # Check if different examples have very different transformations
        if len(examples) < 2:
            return False
        
        ratios = []
        for inp, out in examples:
            ratio = out.size / inp.size
            ratios.append(ratio)
        
        # If ratios vary significantly, might be conditional
        return max(ratios) / min(ratios) > 1.5 if min(ratios) > 0 else False
    
    def _evaluate_program(self, program: Transform, 
                         examples: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Evaluate how well a program solves the examples."""
        total_score = 0.0
        
        for input_grid, expected_output in examples:
            try:
                predicted = program.apply(input_grid)
                
                # Check if shapes match
                if predicted.shape != expected_output.shape:
                    continue
                
                # Calculate accuracy
                accuracy = np.mean(predicted == expected_output)
                total_score += accuracy
            except Exception as e:
                # Program failed on this example
                continue
        
        return total_score / len(examples) if examples else 0.0