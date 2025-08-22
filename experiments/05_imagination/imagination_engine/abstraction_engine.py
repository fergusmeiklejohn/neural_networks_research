"""Abstraction Engine - Learns to extract abstract patterns from concrete examples.

This module enables the system to:
1. Find minimal abstract patterns that explain examples
2. Identify what varies vs what's invariant
3. Build compositional rules from simpler ones
4. Learn constraints that define valid transformations

Key Innovation: Move from concrete inventions to abstract, parameterized patterns.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import itertools


@dataclass
class AbstractPattern:
    """Represents an abstract pattern that can be instantiated."""
    
    name: str
    pattern_type: str  # 'spatial', 'value', 'relational', 'compositional'
    
    # What's constant across all instances
    invariants: Dict[str, Any]
    
    # What varies (parameters)
    parameters: Dict[str, Any]
    
    # Constraints on valid instantiations
    constraints: List[Callable]
    
    # How to instantiate the pattern
    instantiation_function: Optional[Callable] = None
    
    # Examples this pattern was derived from
    source_examples: List[Any] = field(default_factory=list)
    
    # Success metrics
    coverage: float = 0.0  # What fraction of examples it explains
    accuracy: float = 0.0  # How accurately it reproduces outputs
    
    def instantiate(self, **params) -> Callable:
        """Create a concrete instance with given parameters."""
        if not self.instantiation_function:
            raise NotImplementedError("No instantiation function defined")
        
        # Check constraints
        for constraint in self.constraints:
            if not constraint(params):
                raise ValueError(f"Parameters violate constraints: {params}")
        
        return self.instantiation_function(**params)
    
    def matches(self, example: Tuple[np.ndarray, np.ndarray]) -> bool:
        """Check if this pattern could explain the given example."""
        # This would be implemented based on pattern type
        return False


class AbstractionEngine:
    """Engine for learning abstract patterns from concrete examples."""
    
    def __init__(self):
        self.learned_patterns: List[AbstractPattern] = []
        self.pattern_library: Dict[str, List[AbstractPattern]] = defaultdict(list)
    
    def learn_abstraction(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        concrete_solution: Optional[Callable] = None
    ) -> Optional[AbstractPattern]:
        """Learn an abstract pattern from examples.
        
        Args:
            examples: Input-output pairs demonstrating the pattern
            concrete_solution: Optional concrete solution to abstract from
            
        Returns:
            Abstract pattern if one is found
        """
        
        if not examples:
            return None
        
        # Try different abstraction strategies
        pattern = None
        
        # Strategy 1: Variable extraction
        pattern = self._extract_variables(examples)
        if pattern and pattern.coverage > 0.8:
            return pattern
        
        # Strategy 2: Relational abstraction
        pattern = self._abstract_relations(examples)
        if pattern and pattern.coverage > 0.8:
            return pattern
        
        # Strategy 3: Spatial abstraction
        pattern = self._abstract_spatial(examples)
        if pattern and pattern.coverage > 0.8:
            return pattern
        
        # Strategy 4: Value mapping abstraction
        pattern = self._abstract_value_mapping(examples)
        if pattern and pattern.coverage > 0.8:
            return pattern
        
        return pattern
    
    def _extract_variables(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[AbstractPattern]:
        """Extract what varies vs what's constant."""
        
        if len(examples) < 2:
            return None
        
        # Analyze what changes across examples
        input_shapes = [inp.shape for inp, _ in examples]
        output_shapes = [out.shape for _, out in examples]
        
        # Check for variable size extraction
        if len(set(input_shapes)) == 1 and len(set(output_shapes)) > 1:
            # Input same size, output varies - might be extraction
            return self._learn_extraction_pattern(examples)
        
        # Check for variable position patterns
        if self._has_positional_variation(examples):
            return self._learn_positional_pattern(examples)
        
        # Check for variable color patterns
        if self._has_color_variation(examples):
            return self._learn_color_pattern(examples)
        
        return None
    
    def _abstract_relations(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[AbstractPattern]:
        """Abstract relational patterns between objects."""
        
        # Find objects in examples
        all_relations = []
        
        for inp, out in examples:
            input_objects = self._find_objects(inp)
            output_objects = self._find_objects(out)
            
            if not input_objects or not output_objects:
                continue
            
            # Extract relations
            relations = self._extract_relations(input_objects, output_objects)
            all_relations.append(relations)
        
        if not all_relations:
            return None
        
        # Find common relational pattern
        common_pattern = self._find_common_relations(all_relations)
        
        if common_pattern:
            def instantiate(**params):
                def apply(grid):
                    objects = self._find_objects(grid)
                    return self._apply_relational_pattern(grid, objects, common_pattern, params)
                return apply
            
            pattern = AbstractPattern(
                name="relational_transform",
                pattern_type="relational",
                invariants={"relations": common_pattern},
                parameters={"object_colors": None, "positions": None},
                constraints=[],
                instantiation_function=instantiate,
                source_examples=examples,
                coverage=self._calculate_coverage(examples, instantiate())
            )
            
            return pattern
        
        return None
    
    def _abstract_spatial(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[AbstractPattern]:
        """Abstract spatial transformation patterns."""
        
        # Check for region-based patterns
        region_pattern = self._learn_region_pattern(examples)
        if region_pattern:
            return region_pattern
        
        # Check for geometric patterns
        geometric_pattern = self._learn_geometric_pattern(examples)
        if geometric_pattern:
            return geometric_pattern
        
        # Check for symmetry patterns
        symmetry_pattern = self._learn_symmetry_pattern(examples)
        if symmetry_pattern:
            return symmetry_pattern
        
        return None
    
    def _abstract_value_mapping(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[AbstractPattern]:
        """Abstract value transformation patterns."""
        
        # Collect all value mappings
        mappings = []
        
        for inp, out in examples:
            if inp.shape != out.shape:
                continue
            
            # Find value correspondence
            mapping = {}
            for val in np.unique(inp):
                mask = (inp == val)
                out_vals = out[mask]
                if len(np.unique(out_vals)) == 1:
                    mapping[val] = out_vals[0]
            
            mappings.append(mapping)
        
        if not mappings:
            return None
        
        # Check if mapping is consistent
        if all(m == mappings[0] for m in mappings):
            # Fixed mapping
            fixed_map = mappings[0]
            
            def instantiate(**params):
                def apply(grid):
                    result = grid.copy()
                    for from_val, to_val in fixed_map.items():
                        result[grid == from_val] = to_val
                    return result
                return apply
            
            pattern = AbstractPattern(
                name="fixed_value_mapping",
                pattern_type="value",
                invariants={"mapping": fixed_map},
                parameters={},
                constraints=[],
                instantiation_function=instantiate,
                source_examples=examples,
                coverage=1.0
            )
            
            return pattern
        
        # Check for parameterized mapping (e.g., increment by N)
        if self._is_arithmetic_mapping(mappings):
            operation, param = self._extract_arithmetic_pattern(mappings)
            
            def instantiate(**params):
                offset = params.get('offset', param)
                
                def apply(grid):
                    if operation == 'add':
                        return grid + offset
                    elif operation == 'multiply':
                        return grid * offset
                    else:
                        return grid
                return apply
            
            pattern = AbstractPattern(
                name=f"arithmetic_{operation}",
                pattern_type="value",
                invariants={"operation": operation},
                parameters={"offset": param},
                constraints=[lambda p: isinstance(p.get('offset'), (int, float))],
                instantiation_function=instantiate,
                source_examples=examples,
                coverage=0.9
            )
            
            return pattern
        
        return None
    
    def _learn_extraction_pattern(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[AbstractPattern]:
        """Learn pattern for extracting regions."""
        
        # Analyze what regions are extracted
        extraction_rules = []
        
        for inp, out in examples:
            # Find markers that might define extraction
            markers = self._find_marker_positions(inp)
            
            if markers:
                # Check if output corresponds to region around markers
                if self._is_region_around_markers(inp, out, markers):
                    extraction_rules.append({
                        'marker_color': inp[markers[0]],
                        'region_size': out.shape,
                        'relative_position': self._get_relative_position(inp, out, markers)
                    })
        
        if extraction_rules:
            def instantiate(**params):
                marker_color = params.get('marker_color', extraction_rules[0]['marker_color'])
                
                def apply(grid):
                    markers = np.argwhere(grid == marker_color)
                    if len(markers) > 0:
                        return self._extract_region_around(grid, markers)
                    return grid
                return apply
            
            pattern = AbstractPattern(
                name="marker_based_extraction",
                pattern_type="spatial",
                invariants={"extraction_type": "marker_based"},
                parameters={"marker_color": None, "region_size": None},
                constraints=[],
                instantiation_function=instantiate,
                source_examples=examples,
                coverage=len(extraction_rules) / len(examples)
            )
            
            return pattern
        
        return None
    
    def _learn_positional_pattern(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[AbstractPattern]:
        """Learn patterns based on position variation."""
        
        # Check if transformation depends on position
        position_dependent = True
        
        for inp, out in examples:
            if inp.shape != out.shape:
                position_dependent = False
                break
            
            # Check if same input value produces different outputs at different positions
            for val in np.unique(inp):
                if val == 0:
                    continue
                positions = np.argwhere(inp == val)
                if len(positions) > 1:
                    out_vals = [out[tuple(pos)] for pos in positions]
                    if len(set(out_vals)) > 1:
                        # Same input, different outputs - position matters
                        break
            else:
                position_dependent = False
        
        if position_dependent:
            def instantiate(**params):
                def apply(grid):
                    result = grid.copy()
                    h, w = grid.shape
                    
                    for r in range(h):
                        for c in range(w):
                            # Transform based on position
                            if r == c:  # Diagonal
                                result[r, c] = params.get('diagonal_val', grid[r, c] + 1)
                            elif r == 0 or r == h-1 or c == 0 or c == w-1:  # Border
                                result[r, c] = params.get('border_val', grid[r, c])
                    
                    return result
                return apply
            
            pattern = AbstractPattern(
                name="position_dependent_transform",
                pattern_type="spatial",
                invariants={"position_based": True},
                parameters={"diagonal_val": None, "border_val": None},
                constraints=[],
                instantiation_function=instantiate,
                source_examples=examples,
                coverage=0.7
            )
            
            return pattern
        
        return None
    
    def _learn_color_pattern(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[AbstractPattern]:
        """Learn patterns based on color relationships."""
        
        # Analyze color usage patterns
        color_rules = []
        
        for inp, out in examples:
            input_colors = set(np.unique(inp))
            output_colors = set(np.unique(out))
            
            # New colors introduced
            new_colors = output_colors - input_colors
            
            # Colors removed
            removed_colors = input_colors - output_colors
            
            color_rules.append({
                'new': new_colors,
                'removed': removed_colors,
                'preserved': input_colors & output_colors
            })
        
        # Find common pattern
        if all(len(r['new']) == 1 for r in color_rules):
            # Consistent new color introduction
            
            def instantiate(**params):
                new_color = params.get('new_color', list(color_rules[0]['new'])[0])
                
                def apply(grid):
                    # Add new color in specific pattern
                    result = grid.copy()
                    # Simple example: replace background with new color at edges
                    if 0 in grid:
                        mask = (grid == 0)
                        h, w = grid.shape
                        edge_mask = np.zeros_like(mask)
                        edge_mask[0, :] = edge_mask[-1, :] = True
                        edge_mask[:, 0] = edge_mask[:, -1] = True
                        result[mask & edge_mask] = new_color
                    return result
                return apply
            
            pattern = AbstractPattern(
                name="color_introduction",
                pattern_type="value",
                invariants={"introduces_new_color": True},
                parameters={"new_color": None},
                constraints=[],
                instantiation_function=instantiate,
                source_examples=examples,
                coverage=0.6
            )
            
            return pattern
        
        return None
    
    def _learn_region_pattern(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[AbstractPattern]:
        """Learn region-based transformation patterns."""
        
        # Check if outputs are regions of inputs
        all_extractions = True
        
        for inp, out in examples:
            if not self._is_subregion(inp, out):
                all_extractions = False
                break
        
        if all_extractions:
            # Learn extraction rules
            rules = []
            
            for inp, out in examples:
                # Find where output came from in input
                position = self._find_subregion_position(inp, out)
                if position:
                    rules.append(position)
            
            if rules:
                def instantiate(**params):
                    def apply(grid):
                        # Extract based on learned rules
                        # This is simplified - real implementation would be more sophisticated
                        h, w = out.shape
                        if grid.shape[0] >= h and grid.shape[1] >= w:
                            return grid[:h, :w]
                        return grid
                    return apply
                
                pattern = AbstractPattern(
                    name="region_extraction",
                    pattern_type="spatial",
                    invariants={"operation": "extract"},
                    parameters={"position": None, "size": None},
                    constraints=[],
                    instantiation_function=instantiate,
                    source_examples=examples,
                    coverage=len(rules) / len(examples)
                )
                
                return pattern
        
        return None
    
    def _learn_geometric_pattern(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[AbstractPattern]:
        """Learn geometric transformation patterns."""
        
        # Check for rotations
        all_rotations = True
        rotation_amounts = []
        
        for inp, out in examples:
            if inp.shape[0] != inp.shape[1] or out.shape[0] != out.shape[1]:
                all_rotations = False
                break
            
            # Check rotation amounts
            for k in [1, 2, 3]:
                if np.array_equal(np.rot90(inp, k), out):
                    rotation_amounts.append(k)
                    break
            else:
                all_rotations = False
                break
        
        if all_rotations and rotation_amounts:
            # Consistent rotation pattern
            if len(set(rotation_amounts)) == 1:
                k = rotation_amounts[0]
                
                def instantiate(**params):
                    rotation = params.get('rotation', k)
                    
                    def apply(grid):
                        return np.rot90(grid, rotation)
                    return apply
                
                pattern = AbstractPattern(
                    name=f"rotation_{k*90}",
                    pattern_type="spatial",
                    invariants={"transformation": "rotation", "amount": k*90},
                    parameters={},
                    constraints=[],
                    instantiation_function=instantiate,
                    source_examples=examples,
                    coverage=1.0,
                    accuracy=1.0
                )
                
                return pattern
        
        # Check for reflections
        all_flips = True
        flip_axes = []
        
        for inp, out in examples:
            if np.array_equal(np.flip(inp, axis=0), out):
                flip_axes.append(0)
            elif np.array_equal(np.flip(inp, axis=1), out):
                flip_axes.append(1)
            else:
                all_flips = False
                break
        
        if all_flips and flip_axes:
            if len(set(flip_axes)) == 1:
                axis = flip_axes[0]
                
                def instantiate(**params):
                    def apply(grid):
                        return np.flip(grid, axis=axis)
                    return apply
                
                pattern = AbstractPattern(
                    name=f"flip_axis_{axis}",
                    pattern_type="spatial",
                    invariants={"transformation": "flip", "axis": axis},
                    parameters={},
                    constraints=[],
                    instantiation_function=instantiate,
                    source_examples=examples,
                    coverage=1.0,
                    accuracy=1.0
                )
                
                return pattern
        
        return None
    
    def _learn_symmetry_pattern(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[AbstractPattern]:
        """Learn symmetry-based patterns."""
        
        # Check if outputs have symmetry not present in inputs
        symmetry_additions = []
        
        for inp, out in examples:
            input_sym = self._check_symmetries(inp)
            output_sym = self._check_symmetries(out)
            
            # New symmetries in output
            new_sym = output_sym - input_sym
            if new_sym:
                symmetry_additions.append(new_sym)
        
        if symmetry_additions:
            # Common symmetry addition
            common_sym = set.intersection(*symmetry_additions) if symmetry_additions else set()
            
            if common_sym:
                sym_type = list(common_sym)[0]
                
                def instantiate(**params):
                    def apply(grid):
                        return self._make_symmetric(grid, sym_type)
                    return apply
                
                pattern = AbstractPattern(
                    name=f"add_{sym_type}_symmetry",
                    pattern_type="spatial",
                    invariants={"adds_symmetry": sym_type},
                    parameters={},
                    constraints=[],
                    instantiation_function=instantiate,
                    source_examples=examples,
                    coverage=len(symmetry_additions) / len(examples)
                )
                
                return pattern
        
        return None
    
    # Helper methods
    
    def _find_objects(self, grid: np.ndarray) -> List[Dict]:
        """Find distinct objects in grid."""
        from scipy.ndimage import label
        
        objects = []
        unique_colors = [c for c in np.unique(grid) if c != 0]
        
        for color in unique_colors:
            binary = (grid == color).astype(int)
            labeled, num_features = label(binary)
            
            for i in range(1, num_features + 1):
                mask = (labeled == i)
                positions = np.argwhere(mask)
                
                objects.append({
                    'color': color,
                    'positions': positions,
                    'bbox': (positions.min(axis=0), positions.max(axis=0))
                })
        
        return objects
    
    def _extract_relations(self, input_objects: List[Dict], output_objects: List[Dict]) -> Dict:
        """Extract relations between input and output objects."""
        relations = {
            'preserved': [],
            'transformed': [],
            'new': [],
            'removed': []
        }
        
        # Simple matching based on color and position overlap
        for in_obj in input_objects:
            matched = False
            for out_obj in output_objects:
                if self._objects_match(in_obj, out_obj):
                    relations['preserved'].append((in_obj, out_obj))
                    matched = True
                    break
                elif self._objects_related(in_obj, out_obj):
                    relations['transformed'].append((in_obj, out_obj))
                    matched = True
                    break
            
            if not matched:
                relations['removed'].append(in_obj)
        
        for out_obj in output_objects:
            if not any(self._objects_related(in_obj, out_obj) 
                      for in_obj in input_objects):
                relations['new'].append(out_obj)
        
        return relations
    
    def _objects_match(self, obj1: Dict, obj2: Dict) -> bool:
        """Check if two objects match exactly."""
        return (obj1['color'] == obj2['color'] and 
                np.array_equal(obj1['positions'], obj2['positions']))
    
    def _objects_related(self, obj1: Dict, obj2: Dict) -> bool:
        """Check if two objects are related (transformed version)."""
        # Simple check: same color or overlapping positions
        if obj1['color'] == obj2['color']:
            return True
        
        # Check position overlap
        pos1_set = set(map(tuple, obj1['positions']))
        pos2_set = set(map(tuple, obj2['positions']))
        
        return len(pos1_set & pos2_set) > 0
    
    def _find_common_relations(self, all_relations: List[Dict]) -> Optional[Dict]:
        """Find common relational pattern across examples."""
        if not all_relations:
            return None
        
        # Simple check: do all examples preserve same number of objects?
        preserve_counts = [len(r['preserved']) for r in all_relations]
        if len(set(preserve_counts)) == 1:
            return {'preserve_count': preserve_counts[0]}
        
        return None
    
    def _apply_relational_pattern(
        self,
        grid: np.ndarray,
        objects: List[Dict],
        pattern: Dict,
        params: Dict
    ) -> np.ndarray:
        """Apply relational pattern to transform grid."""
        # Simplified implementation
        return grid
    
    def _calculate_coverage(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        function: Callable
    ) -> float:
        """Calculate what fraction of examples the pattern explains."""
        if not examples:
            return 0.0
        
        correct = 0
        for inp, expected in examples:
            try:
                predicted = function(inp)
                if np.array_equal(predicted, expected):
                    correct += 1
            except:
                pass
        
        return correct / len(examples)
    
    def _has_positional_variation(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """Check if examples show position-dependent variation."""
        for inp, out in examples:
            if inp.shape != out.shape:
                continue
            
            # Check if diagonal is treated differently
            h, w = inp.shape
            if h == w:
                diag_in = np.diag(inp)
                diag_out = np.diag(out)
                if not np.array_equal(diag_in, diag_out):
                    return True
        
        return False
    
    def _has_color_variation(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """Check if examples show color-based variation."""
        color_changes = []
        
        for inp, out in examples:
            in_colors = set(np.unique(inp))
            out_colors = set(np.unique(out))
            
            if in_colors != out_colors:
                color_changes.append((in_colors, out_colors))
        
        return len(color_changes) > len(examples) / 2
    
    def _find_marker_positions(self, grid: np.ndarray) -> List[Tuple[int, int]]:
        """Find positions that might be markers."""
        # Look for isolated non-zero pixels
        markers = []
        h, w = grid.shape
        
        for r in range(h):
            for c in range(w):
                if grid[r, c] != 0:
                    # Check if isolated
                    neighbors = 0
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if grid[nr, nc] == grid[r, c]:
                                neighbors += 1
                    
                    if neighbors == 0:
                        markers.append((r, c))
        
        return markers
    
    def _is_region_around_markers(
        self,
        inp: np.ndarray,
        out: np.ndarray,
        markers: List[Tuple[int, int]]
    ) -> bool:
        """Check if output is region around markers."""
        # Simplified check
        return len(markers) > 0 and out.size < inp.size
    
    def _get_relative_position(
        self,
        inp: np.ndarray,
        out: np.ndarray,
        markers: List[Tuple[int, int]]
    ) -> Dict:
        """Get relative position of extraction."""
        # Simplified
        return {'centered': True}
    
    def _extract_region_around(
        self,
        grid: np.ndarray,
        markers: np.ndarray
    ) -> np.ndarray:
        """Extract region around markers."""
        if len(markers) == 0:
            return grid
        
        # Simple: extract bounding box around markers
        min_r, min_c = markers.min(axis=0)
        max_r, max_c = markers.max(axis=0)
        
        # Add padding
        pad = 2
        min_r = max(0, min_r - pad)
        min_c = max(0, min_c - pad)
        max_r = min(grid.shape[0] - 1, max_r + pad)
        max_c = min(grid.shape[1] - 1, max_c + pad)
        
        return grid[min_r:max_r+1, min_c:max_c+1]
    
    def _is_arithmetic_mapping(self, mappings: List[Dict]) -> bool:
        """Check if mappings follow arithmetic pattern."""
        if not mappings:
            return False
        
        # Check if all mappings are consistent offset
        offsets = []
        for mapping in mappings:
            for from_val, to_val in mapping.items():
                if from_val != 0:  # Ignore background
                    offsets.append(to_val - from_val)
        
        return len(set(offsets)) == 1 if offsets else False
    
    def _extract_arithmetic_pattern(self, mappings: List[Dict]) -> Tuple[str, int]:
        """Extract arithmetic operation and parameter."""
        # Simplified: assume addition
        offset = 0
        for mapping in mappings:
            for from_val, to_val in mapping.items():
                if from_val != 0:
                    offset = to_val - from_val
                    break
            if offset != 0:
                break
        
        return ('add', offset)
    
    def _is_subregion(self, large: np.ndarray, small: np.ndarray) -> bool:
        """Check if small is a subregion of large."""
        if small.shape[0] > large.shape[0] or small.shape[1] > large.shape[1]:
            return False
        
        # Check if small appears anywhere in large
        h, w = small.shape
        for r in range(large.shape[0] - h + 1):
            for c in range(large.shape[1] - w + 1):
                if np.array_equal(large[r:r+h, c:c+w], small):
                    return True
        
        return False
    
    def _find_subregion_position(
        self,
        large: np.ndarray,
        small: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """Find where small appears in large."""
        h, w = small.shape
        
        for r in range(large.shape[0] - h + 1):
            for c in range(large.shape[1] - w + 1):
                if np.array_equal(large[r:r+h, c:c+w], small):
                    return (r, c)
        
        return None
    
    def _check_symmetries(self, grid: np.ndarray) -> Set[str]:
        """Check what symmetries a grid has."""
        symmetries = set()
        
        # Horizontal symmetry
        if np.array_equal(grid, np.flip(grid, axis=1)):
            symmetries.add('horizontal')
        
        # Vertical symmetry
        if np.array_equal(grid, np.flip(grid, axis=0)):
            symmetries.add('vertical')
        
        # Diagonal symmetry (if square)
        if grid.shape[0] == grid.shape[1]:
            if np.array_equal(grid, grid.T):
                symmetries.add('diagonal')
        
        return symmetries
    
    def _make_symmetric(self, grid: np.ndarray, sym_type: str) -> np.ndarray:
        """Make grid symmetric in specified way."""
        result = grid.copy()
        
        if sym_type == 'horizontal':
            # Mirror left half to right
            w = grid.shape[1]
            half_w = w // 2
            result[:, w-half_w:] = np.flip(result[:, :half_w], axis=1)
        
        elif sym_type == 'vertical':
            # Mirror top half to bottom
            h = grid.shape[0]
            half_h = h // 2
            result[h-half_h:, :] = np.flip(result[:half_h, :], axis=0)
        
        elif sym_type == 'diagonal' and grid.shape[0] == grid.shape[1]:
            # Make diagonally symmetric
            result = (result + result.T) // 2
        
        return result