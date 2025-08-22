"""Advanced invention strategies for creating generalizable primitives.

These strategies go beyond simple trace-based synthesis to discover
elegant, reusable transformations that capture the essence of tasks.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Dict, Any, Set
from dataclasses import dataclass
import time
from collections import defaultdict

from atomic_operations import AtomicOperations
from primitive_inventor import InventedPrimitive


@dataclass
class Pattern:
    """Represents a discovered pattern in the transformation."""
    type: str  # 'object', 'line', 'region', 'symmetry', etc.
    properties: Dict[str, Any]
    transformation: str  # Description of how it transforms


class InventionStrategies:
    """Advanced strategies for inventing primitives."""
    
    def __init__(self):
        self.atoms = AtomicOperations()
        
    def pattern_decomposition(self, 
                            examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[InventedPrimitive]:
        """Decompose transformation into sub-patterns and compose them.
        
        This strategy:
        1. Identifies objects/patterns in input
        2. Tracks how each transforms
        3. Composes object-level transformations
        """
        
        if not examples:
            return None
            
        input_grid, output_grid = examples[0]
        
        # Identify patterns in input
        input_patterns = self._identify_patterns(input_grid)
        
        # Track transformations for each pattern
        transformations = []
        
        for pattern in input_patterns:
            if pattern.type == 'colored_points':
                # Check if lines are drawn through points
                if self._check_line_drawing(input_grid, output_grid, pattern):
                    transformations.append(('draw_lines', pattern))
                    
            elif pattern.type == 'object':
                # Check object transformations
                transform = self._analyze_object_transform(input_grid, output_grid, pattern)
                if transform:
                    transformations.append((transform, pattern))
                    
            elif pattern.type == 'region':
                # Check region fills or modifications
                transform = self._analyze_region_transform(input_grid, output_grid, pattern)
                if transform:
                    transformations.append((transform, pattern))
        
        # Compose transformations into a primitive
        if transformations:
            function = self._compose_transformations(transformations)
            
            # Validate on all examples
            score = self._validate(function, examples)
            
            if score > 0.9:
                return InventedPrimitive(
                    name="pattern_decomposed",
                    program=f"Composed: {[t[0] for t in transformations]}",
                    function=function,
                    atomic_sequence=self._get_atomic_sequence(transformations),
                    score=score,
                    invention_time=0.0
                )
        
        return None
    
    def abstraction_discovery(self,
                            examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[InventedPrimitive]:
        """Discover abstract patterns that generalize across examples.
        
        This strategy:
        1. Finds invariants across examples
        2. Identifies parameterized patterns
        3. Creates abstract programs
        """
        
        if len(examples) < 2:
            return self.pattern_decomposition(examples)
        
        # Find common patterns across all examples
        common_patterns = self._find_common_patterns(examples)
        
        if not common_patterns:
            return None
        
        # Try to create an abstract program
        for pattern_type, pattern_data in common_patterns.items():
            if pattern_type == 'value_dependent':
                # Create value-dependent transformation
                function = self._create_value_dependent_transform(pattern_data)
                
            elif pattern_type == 'position_dependent':
                # Create position-dependent transformation
                function = self._create_position_dependent_transform(pattern_data)
                
            elif pattern_type == 'object_wise':
                # Create object-wise transformation
                function = self._create_object_wise_transform(pattern_data)
                
            else:
                continue
            
            if function:
                score = self._validate(function, examples)
                if score > 0.9:
                    return InventedPrimitive(
                        name=f"abstract_{pattern_type}",
                        program=f"Abstract pattern: {pattern_type}",
                        function=function,
                        atomic_sequence=["abstraction"],
                        score=score,
                        invention_time=0.0
                    )
        
        return None
    
    def geometric_reasoning(self,
                          examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[InventedPrimitive]:
        """Use geometric reasoning to understand spatial transformations.
        
        This strategy:
        1. Identifies geometric relationships
        2. Finds geometric transformations (rotation, reflection, scaling)
        3. Handles spatial patterns (lines, shapes, symmetries)
        """
        
        if not examples:
            return None
        
        input_grid, output_grid = examples[0]
        
        # Check for geometric transformations
        transforms = []
        
        # Check rotation
        for k in [1, 2, 3]:
            rotated = np.rot90(input_grid, k)
            if np.array_equal(rotated, output_grid):
                function = lambda g: np.rot90(g, k)
                return InventedPrimitive(
                    name=f"rotate_{k*90}",
                    program=f"rotate_{k*90}_degrees",
                    function=function,
                    atomic_sequence=["rotate_90"] * k,
                    score=1.0,
                    invention_time=0.0
                )
        
        # Check reflection
        if np.array_equal(np.flip(input_grid, axis=0), output_grid):
            function = lambda g: np.flip(g, axis=0)
            return InventedPrimitive(
                name="flip_vertical",
                program="flip_vertical",
                function=function,
                atomic_sequence=["flip_vertical"],
                score=1.0,
                invention_time=0.0
            )
        
        if np.array_equal(np.flip(input_grid, axis=1), output_grid):
            function = lambda g: np.flip(g, axis=1)
            return InventedPrimitive(
                name="flip_horizontal",
                program="flip_horizontal",
                function=function,
                atomic_sequence=["flip_horizontal"],
                score=1.0,
                invention_time=0.0
            )
        
        # Check for line drawing patterns
        colored_pixels = self._find_colored_pixels(input_grid)
        if colored_pixels and self._output_has_lines(output_grid, colored_pixels):
            function = self._create_line_drawing_function(colored_pixels)
            
            score = self._validate(function, examples)
            if score > 0.9:
                return InventedPrimitive(
                    name="draw_lines_through_points",
                    program="Draw lines through colored points",
                    function=function,
                    atomic_sequence=["find_points", "draw_lines"],
                    score=score,
                    invention_time=0.0
                )
        
        return None
    
    # ============= Helper Methods =============
    
    def _identify_patterns(self, grid: np.ndarray) -> List[Pattern]:
        """Identify patterns in a grid."""
        patterns = []
        
        # Find colored points
        colored_positions = []
        h, w = grid.shape
        for r in range(h):
            for c in range(w):
                if grid[r, c] != 0:
                    colored_positions.append((r, c, int(grid[r, c])))
        
        if colored_positions:
            patterns.append(Pattern(
                type='colored_points',
                properties={'positions': colored_positions},
                transformation='unknown'
            ))
        
        # Find connected objects
        objects = self._find_connected_objects(grid)
        for obj in objects:
            patterns.append(Pattern(
                type='object',
                properties=obj,
                transformation='unknown'
            ))
        
        # Find rectangular regions
        regions = self._find_regions(grid)
        for region in regions:
            patterns.append(Pattern(
                type='region',
                properties=region,
                transformation='unknown'
            ))
        
        return patterns
    
    def _find_connected_objects(self, grid: np.ndarray) -> List[Dict]:
        """Find connected components in grid."""
        objects = []
        h, w = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        
        def bfs(start_r, start_c, color):
            """Find all pixels in connected component."""
            from collections import deque
            queue = deque([(start_r, start_c)])
            pixels = []
            
            while queue:
                r, c = queue.popleft()
                if visited[r, c]:
                    continue
                    
                visited[r, c] = True
                pixels.append((r, c))
                
                # Check 4-connected neighbors
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < h and 0 <= nc < w and 
                        not visited[nr, nc] and grid[nr, nc] == color):
                        queue.append((nr, nc))
            
            return pixels
        
        # Find all objects
        for r in range(h):
            for c in range(w):
                if not visited[r, c] and grid[r, c] != 0:
                    pixels = bfs(r, c, grid[r, c])
                    if pixels:
                        objects.append({
                            'pixels': pixels,
                            'color': int(grid[r, c]),
                            'size': len(pixels)
                        })
        
        return objects
    
    def _find_regions(self, grid: np.ndarray) -> List[Dict]:
        """Find rectangular regions of consistent color."""
        regions = []
        h, w = grid.shape
        
        # Simple approach: find 2x2 or larger rectangles
        for r in range(h - 1):
            for c in range(w - 1):
                # Check 2x2
                color = grid[r, c]
                if (color != 0 and 
                    grid[r, c+1] == color and
                    grid[r+1, c] == color and
                    grid[r+1, c+1] == color):
                    
                    # Extend to find full rectangle
                    r2, c2 = r + 1, c + 1
                    
                    # Extend right
                    while c2 < w - 1 and all(grid[rr, c2+1] == color for rr in range(r, r2+1)):
                        c2 += 1
                    
                    # Extend down
                    while r2 < h - 1 and all(grid[r2+1, cc] == color for cc in range(c, c2+1)):
                        r2 += 1
                    
                    regions.append({
                        'bounds': (r, c, r2, c2),
                        'color': int(color),
                        'area': (r2 - r + 1) * (c2 - c + 1)
                    })
        
        # Remove overlapping regions (keep larger ones)
        filtered_regions = []
        for region in sorted(regions, key=lambda x: -x['area']):
            r1, c1, r2, c2 = region['bounds']
            overlaps = False
            
            for existing in filtered_regions:
                er1, ec1, er2, ec2 = existing['bounds']
                if not (r2 < er1 or r1 > er2 or c2 < ec1 or c1 > ec2):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_regions.append(region)
        
        return filtered_regions
    
    def _check_line_drawing(self, input_grid: np.ndarray, output_grid: np.ndarray,
                           pattern: Pattern) -> bool:
        """Check if output has lines drawn through points."""
        positions = pattern.properties['positions']
        h_out, w_out = output_grid.shape
        
        for r, c, color in positions:
            # Check bounds
            if r >= h_out or c >= w_out:
                continue
                
            # Check if there's a line through this point
            # Vertical line
            if all(output_grid[rr, c] == color or output_grid[rr, c] == 0 
                  for rr in range(h_out)):
                return True
            # Horizontal line
            if all(output_grid[r, cc] == color or output_grid[r, cc] == 0
                  for cc in range(w_out)):
                return True
        
        return False
    
    def _analyze_object_transform(self, input_grid: np.ndarray, output_grid: np.ndarray,
                                 pattern: Pattern) -> Optional[str]:
        """Analyze how an object transforms."""
        # Simplified: check if object moves, rotates, or changes color
        pixels = pattern.properties['pixels']
        input_color = pattern.properties['color']
        h_out, w_out = output_grid.shape
        
        # Filter pixels to those within output bounds
        valid_pixels = [(r, c) for r, c in pixels if r < h_out and c < w_out]
        
        if not valid_pixels:
            return 'delete'  # Object is outside output bounds
        
        # Check if object disappears
        all_zero = all(output_grid[r, c] == 0 for r, c in valid_pixels)
        if all_zero:
            return 'delete'
        
        # Check if object changes color
        new_colors = set(output_grid[r, c] for r, c in valid_pixels)
        if len(new_colors) == 1 and list(new_colors)[0] != input_color:
            return f'recolor_to_{list(new_colors)[0]}'
        
        return None
    
    def _analyze_region_transform(self, input_grid: np.ndarray, output_grid: np.ndarray,
                                 pattern: Pattern) -> Optional[str]:
        """Analyze how a region transforms."""
        r1, c1, r2, c2 = pattern.properties['bounds']
        h_out, w_out = output_grid.shape
        
        # Clip bounds to output grid size
        r1_clip = max(0, min(r1, h_out - 1))
        c1_clip = max(0, min(c1, w_out - 1))
        r2_clip = max(0, min(r2, h_out - 1))
        c2_clip = max(0, min(c2, w_out - 1))
        
        # Check if region is valid after clipping
        if r1_clip > r2_clip or c1_clip > c2_clip:
            return None  # Region is outside bounds
        
        region_out = output_grid[r1_clip:r2_clip+1, c1_clip:c2_clip+1]
        
        # Check if region is filled with new color
        unique_colors = np.unique(region_out)
        if len(unique_colors) == 1:
            return f'fill_with_{unique_colors[0]}'
        
        return None
    
    def _compose_transformations(self, transformations: List[Tuple[str, Pattern]]) -> Callable:
        """Compose multiple transformations into a single function."""
        
        def composed_function(grid: np.ndarray) -> np.ndarray:
            result = grid.copy()
            h, w = result.shape
            
            for transform_type, pattern in transformations:
                if transform_type == 'draw_lines':
                    # Draw lines through colored points
                    for r, c, color in pattern.properties['positions']:
                        if r < h and c < w:
                            # Draw vertical line
                            result[:, c] = color
                            # Draw horizontal line
                            result[r, :] = color
                        
                elif transform_type.startswith('recolor_to_'):
                    new_color = int(transform_type.split('_')[-1])
                    for r, c in pattern.properties['pixels']:
                        if r < h and c < w:
                            result[r, c] = new_color
                        
                elif transform_type.startswith('fill_with_'):
                    fill_color = int(transform_type.split('_')[-1])
                    r1, c1, r2, c2 = pattern.properties['bounds']
                    # Clip bounds to grid size
                    r1 = max(0, min(r1, h - 1))
                    c1 = max(0, min(c1, w - 1))
                    r2 = max(0, min(r2, h - 1))
                    c2 = max(0, min(c2, w - 1))
                    if r1 <= r2 and c1 <= c2:
                        result[r1:r2+1, c1:c2+1] = fill_color
                    
                elif transform_type == 'delete':
                    for r, c in pattern.properties['pixels']:
                        if r < h and c < w:
                            result[r, c] = 0
            
            return result
        
        return composed_function
    
    def _get_atomic_sequence(self, transformations: List[Tuple[str, Pattern]]) -> List[str]:
        """Get sequence of atomic operations for transformations."""
        sequence = []
        
        for transform_type, _ in transformations:
            if transform_type == 'draw_lines':
                sequence.extend(['find_value', 'fill_region'])
            elif transform_type.startswith('recolor'):
                sequence.append('set_pixel')
            elif transform_type.startswith('fill'):
                sequence.append('fill_region')
            elif transform_type == 'delete':
                sequence.append('set_pixel')
        
        return sequence
    
    def _find_common_patterns(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict:
        """Find patterns common across all examples."""
        common = {}
        
        # Check if all examples have same type of transformation
        all_value_mappings = True
        value_maps = []
        
        for input_grid, output_grid in examples:
            if input_grid.shape != output_grid.shape:
                all_value_mappings = False
                break
                
            # Check value mapping
            value_map = {}
            h, w = input_grid.shape
            
            for r in range(h):
                for c in range(w):
                    in_val = int(input_grid[r, c])
                    out_val = int(output_grid[r, c])
                    
                    if in_val in value_map:
                        if value_map[in_val] != out_val:
                            all_value_mappings = False
                            break
                    else:
                        value_map[in_val] = out_val
                        
            if all_value_mappings:
                value_maps.append(value_map)
        
        if all_value_mappings and value_maps:
            # Check if mappings are consistent
            first_map = value_maps[0]
            if all(vm == first_map for vm in value_maps):
                common['value_dependent'] = first_map
        
        return common
    
    def _create_value_dependent_transform(self, value_map: Dict) -> Callable:
        """Create transformation based on value mapping."""
        def transform(grid: np.ndarray) -> np.ndarray:
            result = grid.copy()
            for old_val, new_val in value_map.items():
                result[grid == old_val] = new_val
            return result
        return transform
    
    def _create_position_dependent_transform(self, pattern_data: Any) -> Optional[Callable]:
        """Create transformation based on position patterns."""
        # Placeholder for position-based transforms
        return None
    
    def _create_object_wise_transform(self, pattern_data: Any) -> Optional[Callable]:
        """Create transformation that operates on each object."""
        # Placeholder for object-wise transforms
        return None
    
    def _find_colored_pixels(self, grid: np.ndarray) -> List[Tuple[int, int, int]]:
        """Find all non-zero pixels with their colors."""
        pixels = []
        h, w = grid.shape
        for r in range(h):
            for c in range(w):
                if grid[r, c] != 0:
                    pixels.append((r, c, int(grid[r, c])))
        return pixels
    
    def _output_has_lines(self, output_grid: np.ndarray, 
                         colored_pixels: List[Tuple[int, int, int]]) -> bool:
        """Check if output has lines through the colored pixel positions."""
        h_out, w_out = output_grid.shape
        
        for r, c, color in colored_pixels:
            # Check bounds
            if r >= h_out or c >= w_out:
                continue
                
            # Check for vertical line
            vertical_line = sum(1 for rr in range(h_out) 
                              if output_grid[rr, c] == color) > 3
            # Check for horizontal line
            horizontal_line = sum(1 for cc in range(w_out)
                                if output_grid[r, cc] == color) > 3
            
            if vertical_line or horizontal_line:
                return True
        
        return False
    
    def _create_line_drawing_function(self, 
                                     colored_pixels: List[Tuple[int, int, int]]) -> Callable:
        """Create function that draws lines through colored pixels."""
        
        def draw_lines(grid: np.ndarray) -> np.ndarray:
            result = grid.copy()
            h, w = grid.shape
            
            # Find colored pixels in input
            input_pixels = []
            for r in range(h):
                for c in range(w):
                    if grid[r, c] != 0:
                        input_pixels.append((r, c, int(grid[r, c])))
            
            # Draw lines through each
            for r, c, color in input_pixels:
                # Draw vertical line
                for rr in range(h):
                    if result[rr, c] == 0:
                        result[rr, c] = color
                # Draw horizontal line
                for cc in range(w):
                    if result[r, cc] == 0:
                        result[r, cc] = color
            
            # Mark intersections
            for i, (r1, c1, color1) in enumerate(input_pixels):
                for r2, c2, color2 in input_pixels[i+1:]:
                    # Intersection of lines
                    if color1 != color2:
                        # Vertical from first, horizontal from second
                        result[r2, c1] = 2  # Mark with special color
                        # Horizontal from first, vertical from second
                        result[r1, c2] = 2
            
            return result
        
        return draw_lines
    
    def _validate(self, function: Callable,
                 examples: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Validate function on examples."""
        if not examples:
            return 0.0
        
        total_score = 0.0
        for input_grid, expected_output in examples:
            try:
                predicted = function(input_grid)
                if predicted.shape == expected_output.shape:
                    accuracy = np.mean(predicted == expected_output)
                    total_score += accuracy
            except:
                pass
        
        return total_score / len(examples)
    
    def multi_object_coordination(self,
                                 examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[InventedPrimitive]:
        """Handle multiple objects with coordinated transformations.
        
        This strategy:
        1. Identifies multiple objects
        2. Learns relationships between them
        3. Creates coordinated transformations
        """
        
        if not examples:
            return None
        
        # Analyze first example to understand multi-object patterns
        input_grid, output_grid = examples[0]
        
        # Find objects in input
        input_objects = self._find_connected_objects(input_grid)
        output_objects = self._find_connected_objects(output_grid)
        
        if len(input_objects) < 2:
            return None  # Need multiple objects
        
        # Try to identify relationship
        # Simple example: objects swap positions
        if len(input_objects) == 2 and len(output_objects) == 2:
            obj1_in, obj2_in = input_objects[:2]
            
            # Check if objects swapped
            def swap_objects(grid: np.ndarray) -> np.ndarray:
                result = np.zeros_like(grid)
                objects = self._find_connected_objects(grid)
                
                if len(objects) >= 2:
                    # Get centroids
                    obj1, obj2 = objects[:2]
                    pixels1 = obj1['pixels']
                    pixels2 = obj2['pixels']
                    
                    # Calculate centroids
                    c1_r = sum(r for r, c in pixels1) // len(pixels1)
                    c1_c = sum(c for r, c in pixels1) // len(pixels1)
                    c2_r = sum(r for r, c in pixels2) // len(pixels2)
                    c2_c = sum(c for r, c in pixels2) // len(pixels2)
                    
                    # Swap positions
                    offset_r = c2_r - c1_r
                    offset_c = c2_c - c1_c
                    
                    # Move obj1 to obj2's position
                    for r, c in pixels1:
                        new_r = r + offset_r
                        new_c = c + offset_c
                        if 0 <= new_r < grid.shape[0] and 0 <= new_c < grid.shape[1]:
                            result[new_r, new_c] = obj1['color']
                    
                    # Move obj2 to obj1's position
                    for r, c in pixels2:
                        new_r = r - offset_r
                        new_c = c - offset_c
                        if 0 <= new_r < grid.shape[0] and 0 <= new_c < grid.shape[1]:
                            result[new_r, new_c] = obj2['color']
                
                return result
            
            # Validate
            score = self._validate(swap_objects, examples)
            
            if score > 0.5:
                return InventedPrimitive(
                    name="multi_object_swap",
                    program="Swap positions of two objects",
                    function=swap_objects,
                    atomic_sequence=["find_objects", "calculate_offset", "move_pixels"],
                    score=score,
                    invention_time=0.0
                )
        
        return None
    
    def conditional_transformation(self,
                                 examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[InventedPrimitive]:
        """Create transformations based on conditions.
        
        This strategy:
        1. Identifies conditions (color, position, neighbors)
        2. Learns conditional rules
        3. Creates branching transformations
        """
        
        if not examples:
            return None
        
        # Analyze patterns to find conditions
        input_grid, output_grid = examples[0]
        
        # Simple example: transform based on color
        unique_colors = np.unique(input_grid)
        
        if len(unique_colors) > 2:  # Need multiple colors for conditions
            
            def conditional_color_transform(grid: np.ndarray) -> np.ndarray:
                result = grid.copy()
                
                for r in range(grid.shape[0]):
                    for c in range(grid.shape[1]):
                        value = grid[r, c]
                        
                        # Conditional rules based on color
                        if value == 1:
                            # Rule 1: Color 1 becomes 2
                            result[r, c] = 2
                        elif value == 2:
                            # Rule 2: Color 2 becomes 3
                            result[r, c] = 3
                        elif value == 3:
                            # Rule 3: Color 3 becomes 1 (cycle)
                            result[r, c] = 1
                        # else: keep as is (0 or other)
                
                return result
            
            # Validate
            score = self._validate(conditional_color_transform, examples)
            
            if score > 0.5:
                return InventedPrimitive(
                    name="conditional_color",
                    program="Transform colors conditionally",
                    function=conditional_color_transform,
                    atomic_sequence=["check_color", "apply_rule"],
                    score=score,
                    invention_time=0.0
                )
        
        # Try position-based conditions
        def conditional_position_transform(grid: np.ndarray) -> np.ndarray:
            result = grid.copy()
            h, w = grid.shape
            
            for r in range(h):
                for c in range(w):
                    if grid[r, c] != 0:
                        # Condition based on position
                        if r < h // 2:  # Top half
                            result[r, c] = min(grid[r, c] + 1, 9)
                        else:  # Bottom half
                            result[r, c] = max(grid[r, c] - 1, 1)
            
            return result
        
        # Validate position-based
        score = self._validate(conditional_position_transform, examples)
        
        if score > 0.5:
            return InventedPrimitive(
                name="conditional_position",
                program="Transform based on position",
                function=conditional_position_transform,
                atomic_sequence=["check_position", "apply_rule"],
                score=score,
                invention_time=0.0
            )
        
        return None
    
    def recursive_patterns(self,
                          examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[InventedPrimitive]:
        """Create self-similar recursive transformations.
        
        This strategy:
        1. Identifies recursive/fractal patterns
        2. Creates self-similar transformations
        3. Applies at multiple scales
        """
        
        if not examples:
            return None
        
        input_grid, output_grid = examples[0]
        
        # Simple recursive pattern: duplicate and scale
        def recursive_duplicate(grid: np.ndarray) -> np.ndarray:
            result = grid.copy()
            h, w = grid.shape
            
            # Find non-zero region
            non_zero = np.argwhere(grid != 0)
            if len(non_zero) == 0:
                return result
            
            min_r, min_c = non_zero.min(axis=0)
            max_r, max_c = non_zero.max(axis=0)
            
            # Extract pattern
            pattern = grid[min_r:max_r+1, min_c:max_c+1]
            ph, pw = pattern.shape
            
            # Try to place scaled copies
            if ph <= h // 2 and pw <= w // 2:
                # Place in quadrants
                result[0:ph, 0:pw] = pattern
                result[0:ph, w-pw:w] = pattern
                result[h-ph:h, 0:pw] = pattern
                result[h-ph:h, w-pw:w] = pattern
            
            return result
        
        # Validate
        score = self._validate(recursive_duplicate, examples)
        
        if score > 0.5:
            return InventedPrimitive(
                name="recursive_pattern",
                program="Create recursive/fractal pattern",
                function=recursive_duplicate,
                atomic_sequence=["extract_pattern", "scale", "duplicate"],
                score=score,
                invention_time=0.0
            )
        
        return None
    
    def boundary_operations(self,
                          examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[InventedPrimitive]:
        """Special operations on boundaries and edges.
        
        This strategy:
        1. Identifies boundaries
        2. Applies special edge transformations
        3. Handles frame/border operations
        """
        
        if not examples:
            return None
        
        # Simple boundary operation: highlight edges
        def highlight_boundaries(grid: np.ndarray) -> np.ndarray:
            result = grid.copy()
            h, w = grid.shape
            
            for r in range(h):
                for c in range(w):
                    if grid[r, c] != 0:
                        # Check if on boundary
                        is_boundary = False
                        
                        # Check 4-neighbors
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if nr < 0 or nr >= h or nc < 0 or nc >= w or grid[nr, nc] == 0:
                                is_boundary = True
                                break
                        
                        if is_boundary:
                            # Highlight boundary pixels
                            result[r, c] = min(grid[r, c] + 3, 9)
            
            return result
        
        # Validate
        score = self._validate(highlight_boundaries, examples)
        
        if score > 0.5:
            return InventedPrimitive(
                name="boundary_highlight",
                program="Highlight object boundaries",
                function=highlight_boundaries,
                atomic_sequence=["detect_boundary", "modify_pixel"],
                score=score,
                invention_time=0.0
            )
        
        # Try frame operation
        def add_frame(grid: np.ndarray) -> np.ndarray:
            result = grid.copy()
            h, w = grid.shape
            
            # Add frame of color 1
            result[0, :] = 1  # Top
            result[h-1, :] = 1  # Bottom
            result[:, 0] = 1  # Left
            result[:, w-1] = 1  # Right
            
            return result
        
        # Validate frame
        score = self._validate(add_frame, examples)
        
        if score > 0.5:
            return InventedPrimitive(
                name="add_frame",
                program="Add frame around grid",
                function=add_frame,
                atomic_sequence=["draw_line"],
                score=score,
                invention_time=0.0
            )
        
        return None
    
    def symmetry_operations(self, examples):
        """Strategy: Try various symmetry operations (reflection, rotation)."""
        
        if not examples:
            return None
        
        # Try horizontal reflection
        def horizontal_reflection(grid: np.ndarray) -> np.ndarray:
            return np.fliplr(grid)
        
        score = self._validate(horizontal_reflection, examples)
        if score > 0.7:
            return InventedPrimitive(
                name="horizontal_reflection",
                program="Reflect grid horizontally (left-right)",
                function=horizontal_reflection,
                atomic_sequence=["flip_horizontal"],
                score=score,
                invention_time=0.0
            )
        
        # Try vertical reflection
        def vertical_reflection(grid: np.ndarray) -> np.ndarray:
            return np.flipud(grid)
        
        score = self._validate(vertical_reflection, examples)
        if score > 0.7:
            return InventedPrimitive(
                name="vertical_reflection",
                program="Reflect grid vertically (up-down)",
                function=vertical_reflection,
                atomic_sequence=["flip_vertical"],
                score=score,
                invention_time=0.0
            )
        
        # Try 90-degree rotation
        def rotate_90(grid: np.ndarray) -> np.ndarray:
            return np.rot90(grid, k=1)
        
        score = self._validate(rotate_90, examples)
        if score > 0.7:
            return InventedPrimitive(
                name="rotate_90",
                program="Rotate grid 90 degrees clockwise",
                function=rotate_90,
                atomic_sequence=["rotate"],
                score=score,
                invention_time=0.0
            )
        
        return None
    
    def counting_arithmetic(self, examples):
        """Strategy: Try counting objects and arithmetic operations."""
        
        if not examples:
            return None
        
        # Try counting objects and filling based on count
        def count_and_fill(grid: np.ndarray) -> np.ndarray:
            unique_colors = np.unique(grid)
            non_bg_colors = [c for c in unique_colors if c != 0]
            
            if len(non_bg_colors) == 0:
                return grid.copy()
            
            # Count objects of each color
            result = grid.copy()
            h, w = grid.shape
            
            for color in non_bg_colors:
                count = np.sum(grid == color)
                
                # Simple pattern: add count indicator
                if count > 0 and count < min(h, w):
                    for i in range(min(count, h)):
                        if result[i, 0] == 0:
                            result[i, 0] = color
            
            return result
        
        score = self._validate(count_and_fill, examples)
        if score > 0.5:
            return InventedPrimitive(
                name="count_and_fill",
                program="Count objects and fill based on count",
                function=count_and_fill,
                atomic_sequence=["count", "fill"],
                score=score,
                invention_time=0.0
            )
        
        # Try arithmetic scaling
        def arithmetic_scale(grid: np.ndarray) -> np.ndarray:
            result = grid.copy()
            unique_vals = np.unique(grid)
            
            # Try doubling non-zero values
            for val in unique_vals:
                if val > 0 and val < 5:  # Only scale small values
                    mask = grid == val
                    if np.any(mask):
                        new_val = min(val * 2, 9)  # Cap at 9 for ARC
                        result[mask] = new_val
            
            return result
        
        score = self._validate(arithmetic_scale, examples)
        if score > 0.5:
            return InventedPrimitive(
                name="arithmetic_scale",
                program="Scale color values arithmetically",
                function=arithmetic_scale,
                atomic_sequence=["multiply"],
                score=score,
                invention_time=0.0
            )
        
        return None
    
    def pattern_completion(self, examples):
        """Strategy: Complete partial patterns (grids, sequences, symmetries)."""
        
        if not examples:
            return None
        
        # Try completing checkerboard patterns
        def complete_checkerboard(grid: np.ndarray) -> np.ndarray:
            result = grid.copy()
            h, w = grid.shape
            
            # Detect and complete checkerboard pattern
            for i in range(h):
                for j in range(w):
                    if result[i, j] == 0:  # Empty cell
                        # Check if it should be filled based on neighbors
                        expected_val = ((i + j) % 2) + 1
                        # Check if neighbors follow this pattern
                        neighbors_match = False
                        if i > 0 and result[i-1, j] != 0:
                            neighbors_match = True
                        if j > 0 and result[i, j-1] != 0:
                            neighbors_match = True
                        
                        if neighbors_match:
                            result[i, j] = expected_val
            
            return result
        
        score = self._validate(complete_checkerboard, examples)
        if score > 0.6:
            return InventedPrimitive(
                name="complete_checkerboard",
                program="Complete checkerboard pattern",
                function=complete_checkerboard,
                atomic_sequence=["detect_pattern", "fill"],
                score=score,
                invention_time=0.0
            )
        
        return None
    
    def grid_subdivision(self, examples):
        """Strategy: Divide grid into regions and transform each."""
        
        if not examples:
            return None
        
        # Try quadrant-based transformation
        def transform_quadrants(grid: np.ndarray) -> np.ndarray:
            result = grid.copy()
            h, w = grid.shape
            mid_h, mid_w = h // 2, w // 2
            
            # Transform each quadrant differently
            # Top-right: increment colors
            for i in range(mid_h):
                for j in range(mid_w, w):
                    if result[i, j] > 0:
                        result[i, j] = min(result[i, j] + 1, 9)
            
            # Bottom-left: double colors
            for i in range(mid_h, h):
                for j in range(mid_w):
                    if result[i, j] > 0:
                        result[i, j] = min(result[i, j] * 2, 9)
            
            return result
        
        score = self._validate(transform_quadrants, examples)
        if score > 0.5:
            return InventedPrimitive(
                name="transform_quadrants",
                program="Transform grid quadrants differently",
                function=transform_quadrants,
                atomic_sequence=["divide_grid", "transform_region"],
                score=score,
                invention_time=0.0
            )
        
        return None
    
    def color_mapping(self, examples):
        """Strategy: Map colors based on rules."""
        
        if not examples:
            return None
        
        # Try color replacement mapping
        def color_replacement(grid: np.ndarray) -> np.ndarray:
            result = grid.copy()
            
            # Learn color mapping from first example
            if len(examples) > 0:
                inp, out = examples[0]
                # Build color map
                color_map = {}
                for i in range(min(inp.shape[0], out.shape[0])):
                    for j in range(min(inp.shape[1], out.shape[1])):
                        if inp[i, j] != 0 and inp[i, j] != out[i, j]:
                            color_map[inp[i, j]] = out[i, j]
                
                # Apply mapping
                for old_color, new_color in color_map.items():
                    result[grid == old_color] = new_color
            
            return result
        
        score = self._validate(color_replacement, examples)
        if score > 0.7:
            return InventedPrimitive(
                name="color_replacement",
                program="Replace colors based on mapping",
                function=color_replacement,
                atomic_sequence=["map_colors"],
                score=score,
                invention_time=0.0
            )
        
        return None