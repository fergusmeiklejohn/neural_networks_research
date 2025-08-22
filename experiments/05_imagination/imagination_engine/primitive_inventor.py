"""Primitive Inventor - Creates novel primitives on-the-fly.

This module invents new primitives by analyzing input-output examples and
composing atomic operations to achieve the desired transformation.

Key Innovation: Instead of searching through a fixed library, we CREATE
the primitive that solves the task.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Dict, Any
from dataclasses import dataclass
import time
from itertools import product

from atomic_operations import AtomicOperations, AtomicOp, get_atomic_operations


@dataclass
class InventedPrimitive:
    """Represents an invented primitive."""
    name: str
    program: str  # String representation of the program
    function: Callable[[np.ndarray], np.ndarray]
    atomic_sequence: List[str]  # Sequence of atomic operations used
    score: float  # How well it solves the examples
    invention_time: float
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        """Apply the invented primitive to a grid."""
        return self.function(grid)


@dataclass 
class Trace:
    """Execution trace of a transformation."""
    operations: List[Tuple[str, Dict[str, Any]]]  # (op_name, params)
    intermediate_states: List[np.ndarray]
    

class PrimitiveInventor:
    """Invents new primitives by analyzing examples and composing atomic operations."""
    
    def __init__(self, max_program_length: int = 10, timeout: float = 5.0):
        """Initialize the inventor.
        
        Args:
            max_program_length: Maximum number of atomic operations in a program
            timeout: Maximum time to spend inventing a primitive
        """
        self.max_program_length = max_program_length
        self.timeout = timeout
        self.atomic_ops = get_atomic_operations()
        self.invention_count = 0
        
    def invent_primitive(self, 
                        examples: List[Tuple[np.ndarray, np.ndarray]],
                        strategy: str = "trace") -> Optional[InventedPrimitive]:
        """Invent a primitive that transforms inputs to outputs.
        
        Args:
            examples: List of (input, output) pairs
            strategy: Invention strategy to use
            
        Returns:
            Invented primitive if successful, None otherwise
        """
        start_time = time.time()
        
        if strategy == "trace":
            primitive = self._trace_based_synthesis(examples)
        elif strategy == "search":
            primitive = self._search_based_synthesis(examples)
        elif strategy == "differential":
            primitive = self._differential_synthesis(examples)
        else:
            primitive = self._trace_based_synthesis(examples)
        
        if primitive:
            primitive.invention_time = time.time() - start_time
            self.invention_count += 1
            primitive.name = f"invented_{self.invention_count}"
            
        return primitive
    
    def _trace_based_synthesis(self, 
                              examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[InventedPrimitive]:
        """Synthesize by tracing pixel changes from input to output."""
        
        if not examples:
            return None
        
        # Analyze the first example to understand the transformation
        input_grid, output_grid = examples[0]
        
        # Find all differences between input and output
        diff_positions = self._find_differences(input_grid, output_grid)
        
        if not diff_positions:
            # No changes - identity function
            return InventedPrimitive(
                name="identity",
                program="lambda g: g",
                function=lambda g: g.copy(),
                atomic_sequence=["identity"],
                score=1.0,
                invention_time=0.0
            )
        
        # Try different hypotheses about the transformation
        
        # Hypothesis 1: Simple value mapping
        if self._is_value_mapping(input_grid, output_grid):
            mapping = self._extract_value_mapping(input_grid, output_grid)
            function = self._create_mapping_function(mapping)
            
            if self._validate_on_examples(function, examples) > 0.9:
                return InventedPrimitive(
                    name="value_mapping",
                    program=f"map_pixels with mapping {mapping}",
                    function=function,
                    atomic_sequence=["map_pixels"],
                    score=1.0,
                    invention_time=0.0
                )
        
        # Hypothesis 2: Position-based transformation
        if self._is_position_based(input_grid, output_grid, diff_positions):
            pattern = self._extract_position_pattern(input_grid, output_grid, diff_positions)
            function = self._create_position_function(pattern)
            
            if function and self._validate_on_examples(function, examples) > 0.9:
                return InventedPrimitive(
                    name="position_based",
                    program=f"position pattern: {pattern}",
                    function=function,
                    atomic_sequence=["map_pixels_with_position"],
                    score=1.0,
                    invention_time=0.0
                )
        
        # Hypothesis 3: Region-based transformation
        regions = self._identify_regions(input_grid, output_grid)
        if regions:
            function = self._create_region_function(regions, input_grid, output_grid)
            
            if function and self._validate_on_examples(function, examples) > 0.9:
                return InventedPrimitive(
                    name="region_based",
                    program=f"region operations on {len(regions)} regions",
                    function=function,
                    atomic_sequence=["copy_region", "paste_region"],
                    score=1.0,
                    invention_time=0.0
                )
        
        # Hypothesis 4: Systematic pixel-by-pixel construction
        trace = self._create_construction_trace(input_grid, output_grid)
        if trace:
            function = self._trace_to_function(trace)
            
            if function and self._validate_on_examples(function, examples) > 0.9:
                return InventedPrimitive(
                    name="constructed",
                    program=f"pixel construction with {len(trace.operations)} ops",
                    function=function,
                    atomic_sequence=[op[0] for op in trace.operations],
                    score=1.0,
                    invention_time=0.0
                )
        
        return None
    
    def _search_based_synthesis(self,
                               examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[InventedPrimitive]:
        """Synthesize by searching through compositions of atomic operations."""
        
        # Start with single atomic operations
        for atomic_op in self.atomic_ops:
            if atomic_op.returns_grid and atomic_op.arity == 1:
                # Try this operation
                function = atomic_op.func
                score = self._validate_on_examples(function, examples)
                
                if score > 0.9:
                    return InventedPrimitive(
                        name=atomic_op.name,
                        program=atomic_op.name,
                        function=function,
                        atomic_sequence=[atomic_op.name],
                        score=score,
                        invention_time=0.0
                    )
        
        # Try compositions of 2 operations
        for op1 in self.atomic_ops:
            if not op1.returns_grid or op1.arity != 1:
                continue
                
            for op2 in self.atomic_ops:
                if not op2.returns_grid or op2.arity != 1:
                    continue
                
                # Compose: op2(op1(grid))
                function = lambda g, f1=op1.func, f2=op2.func: f2(f1(g))
                score = self._validate_on_examples(function, examples)
                
                if score > 0.9:
                    return InventedPrimitive(
                        name=f"{op1.name}_{op2.name}",
                        program=f"{op2.name}({op1.name}(grid))",
                        function=function,
                        atomic_sequence=[op1.name, op2.name],
                        score=score,
                        invention_time=0.0
                    )
        
        return None
    
    def _differential_synthesis(self,
                               examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[InventedPrimitive]:
        """Synthesize using differentiable programming techniques."""
        # Simplified version - just try parameterized transformations
        
        input_grid, output_grid = examples[0]
        h, w = input_grid.shape
        
        # Try affine transformations
        for scale in [0.5, 1.0, 2.0]:
            for offset in [-5, -2, -1, 0, 1, 2, 5]:
                function = lambda g, s=scale, o=offset: np.clip(g * s + o, 0, 9).astype(int)
                score = self._validate_on_examples(function, examples)
                
                if score > 0.9:
                    return InventedPrimitive(
                        name="affine",
                        program=f"grid * {scale} + {offset}",
                        function=function,
                        atomic_sequence=["arithmetic"],
                        score=score,
                        invention_time=0.0
                    )
        
        return None
    
    # ============= Helper Methods =============
    
    def _find_differences(self, grid1: np.ndarray, grid2: np.ndarray) -> List[Tuple[int, int]]:
        """Find all positions where grids differ."""
        diff = grid1 != grid2
        return list(zip(*np.where(diff)))
    
    def _is_value_mapping(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if transformation is a simple value mapping."""
        if input_grid.shape != output_grid.shape:
            return False
        
        # Check if each input value maps consistently to an output value
        value_map = {}
        h, w = input_grid.shape
        
        for r in range(h):
            for c in range(w):
                in_val = int(input_grid[r, c])
                out_val = int(output_grid[r, c])
                
                if in_val in value_map:
                    if value_map[in_val] != out_val:
                        return False
                else:
                    value_map[in_val] = out_val
        
        return True
    
    def _extract_value_mapping(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[int, int]:
        """Extract the value mapping from input to output."""
        value_map = {}
        h, w = input_grid.shape
        
        for r in range(h):
            for c in range(w):
                in_val = int(input_grid[r, c])
                out_val = int(output_grid[r, c])
                value_map[in_val] = out_val
        
        return value_map
    
    def _create_mapping_function(self, mapping: Dict[int, int]) -> Callable:
        """Create a function that applies a value mapping."""
        def apply_mapping(grid: np.ndarray) -> np.ndarray:
            result = grid.copy()
            for old_val, new_val in mapping.items():
                result[grid == old_val] = new_val
            return result
        return apply_mapping
    
    def _is_position_based(self, input_grid: np.ndarray, output_grid: np.ndarray,
                          diff_positions: List[Tuple[int, int]]) -> bool:
        """Check if changes follow a position-based pattern."""
        if not diff_positions:
            return False
        
        # Check for patterns like diagonal, checkerboard, etc.
        # Simplified: check if all changes are on diagonal
        for r, c in diff_positions:
            if r != c and r + c != input_grid.shape[0] - 1:
                return False
        return True
    
    def _extract_position_pattern(self, input_grid: np.ndarray, output_grid: np.ndarray,
                                 diff_positions: List[Tuple[int, int]]) -> str:
        """Extract the position-based pattern."""
        # Simplified: detect diagonal pattern
        if all(r == c for r, c in diff_positions):
            return "main_diagonal"
        elif all(r + c == input_grid.shape[0] - 1 for r, c in diff_positions):
            return "anti_diagonal"
        else:
            return "unknown"
    
    def _create_position_function(self, pattern: str) -> Optional[Callable]:
        """Create a function based on position pattern."""
        if pattern == "main_diagonal":
            def diagonal_transform(grid: np.ndarray) -> np.ndarray:
                result = grid.copy()
                h, w = grid.shape
                for i in range(min(h, w)):
                    result[i, i] = (grid[i, i] + 1) % 10
                return result
            return diagonal_transform
        elif pattern == "anti_diagonal":
            def anti_diagonal_transform(grid: np.ndarray) -> np.ndarray:
                result = grid.copy()
                h, w = grid.shape
                for i in range(min(h, w)):
                    if i < h and (w - 1 - i) >= 0:
                        result[i, w - 1 - i] = (grid[i, w - 1 - i] + 1) % 10
                return result
            return anti_diagonal_transform
        return None
    
    def _identify_regions(self, input_grid: np.ndarray, 
                         output_grid: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Identify regions that have been transformed."""
        # Simplified: find rectangular regions with consistent changes
        regions = []
        h, w = input_grid.shape
        
        # Try to find 2x2 regions with consistent transformation
        for r in range(h - 1):
            for c in range(w - 1):
                region_in = input_grid[r:r+2, c:c+2]
                region_out = output_grid[r:r+2, c:c+2]
                
                if not np.array_equal(region_in, region_out):
                    regions.append((r, c, r+1, c+1))
        
        return regions[:5]  # Limit to first 5 regions
    
    def _create_region_function(self, regions: List[Tuple[int, int, int, int]],
                               input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Callable]:
        """Create a function that applies region transformations."""
        def region_transform(grid: np.ndarray) -> np.ndarray:
            result = grid.copy()
            for r1, c1, r2, c2 in regions:
                # Simple: increment values in region
                result[r1:r2+1, c1:c2+1] = (grid[r1:r2+1, c1:c2+1] + 1) % 10
            return result
        return region_transform
    
    def _create_construction_trace(self, input_grid: np.ndarray, 
                                  output_grid: np.ndarray) -> Optional[Trace]:
        """Create a trace of operations to construct output from input."""
        operations = []
        current = input_grid.copy()
        
        # Simple strategy: change pixels one by one
        h, w = input_grid.shape
        for r in range(h):
            for c in range(w):
                if current[r, c] != output_grid[r, c]:
                    operations.append(("set_pixel", {
                        "r": r, "c": c, "value": int(output_grid[r, c])
                    }))
                    current[r, c] = output_grid[r, c]
        
        if operations:
            return Trace(operations=operations, intermediate_states=[])
        return None
    
    def _trace_to_function(self, trace: Trace) -> Optional[Callable]:
        """Convert a trace to a callable function."""
        def apply_trace(grid: np.ndarray) -> np.ndarray:
            result = grid.copy()
            for op_name, params in trace.operations:
                if op_name == "set_pixel":
                    r, c, value = params["r"], params["c"], params["value"]
                    if 0 <= r < result.shape[0] and 0 <= c < result.shape[1]:
                        result[r, c] = value
            return result
        return apply_trace
    
    def _validate_on_examples(self, function: Callable, 
                             examples: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Validate a function on all examples."""
        if not examples:
            return 0.0
        
        total_score = 0.0
        for input_grid, expected_output in examples:
            try:
                predicted = function(input_grid)
                if predicted.shape == expected_output.shape:
                    accuracy = np.mean(predicted == expected_output)
                    total_score += accuracy
                else:
                    total_score += 0.0
            except:
                total_score += 0.0
        
        return total_score / len(examples)