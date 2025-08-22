"""
Region Extraction Learner

Learns to extract regions based on markers or patterns.
This addresses a key limitation in current ARC performance.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Callable, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class RegionMarker:
    """Describes how a region is marked."""
    marker_type: str  # 'corners', 'boundary', 'single_point', 'color_based'
    properties: Dict[str, Any]
    extraction_rule: str  # How to extract from marker


@dataclass 
class ExtractionRule:
    """A learned rule for extracting regions."""
    name: str
    marker_pattern: RegionMarker
    size_rule: str  # 'fixed', 'relative_to_grid', 'marker_dependent'
    size_params: Dict[str, Any]
    position_rule: str  # 'at_marker', 'offset_from_marker', 'between_markers'
    position_params: Dict[str, Any]
    confidence: float


class RegionExtractionLearner:
    """Learn to extract regions based on markers or patterns."""
    
    def __init__(self):
        self.learned_rules: List[ExtractionRule] = []
        self.extraction_strategies = {
            'corners': self._extract_by_corners,
            'boundary': self._extract_by_boundary,
            'single_point': self._extract_by_single_point,
            'color_based': self._extract_by_color,
            'pattern_based': self._extract_by_pattern
        }
    
    def learn_extraction_rules(self, 
                              examples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> List[ExtractionRule]:
        """
        Learn extraction rules from examples.
        
        Args:
            examples: List of (full_grid, markers, extracted_region) tuples
            
        Returns:
            List of learned extraction rules
        """
        rules = []
        
        for full_grid, markers, extracted_region in examples:
            # Identify marker type
            marker_type = self._identify_marker_type(markers, full_grid)
            
            # Learn size rule
            size_rule, size_params = self._learn_size_rule(
                full_grid, markers, extracted_region
            )
            
            # Learn position rule  
            position_rule, position_params = self._learn_position_rule(
                full_grid, markers, extracted_region
            )
            
            # Create extraction rule
            rule = ExtractionRule(
                name=f"{marker_type}_{size_rule}_{position_rule}",
                marker_pattern=RegionMarker(
                    marker_type=marker_type,
                    properties=self._analyze_marker_properties(markers),
                    extraction_rule=f"Extract {size_rule} region {position_rule}"
                ),
                size_rule=size_rule,
                size_params=size_params,
                position_rule=position_rule,
                position_params=position_params,
                confidence=1.0
            )
            
            # Test if rule generalizes
            if self._test_rule(rule, examples):
                rules.append(rule)
        
        # Merge similar rules
        self.learned_rules = self._merge_similar_rules(rules)
        return self.learned_rules
    
    def extract_marked_region(self, 
                            grid: np.ndarray, 
                            markers: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Extract a region from grid based on markers.
        
        Args:
            grid: Full grid to extract from
            markers: Optional marker grid (if None, detect from grid)
            
        Returns:
            Extracted region or None if no rule applies
        """
        if markers is None:
            markers = self._detect_markers(grid)
        
        # Try each learned rule
        for rule in sorted(self.learned_rules, key=lambda r: -r.confidence):
            if rule.marker_pattern.marker_type in self.extraction_strategies:
                extractor = self.extraction_strategies[rule.marker_pattern.marker_type]
                region = extractor(grid, markers, rule)
                
                if region is not None:
                    return region
        
        # Fallback: try heuristic extraction
        return self._heuristic_extraction(grid, markers)
    
    def _identify_marker_type(self, markers: np.ndarray, full_grid: np.ndarray) -> str:
        """Identify what type of markers are present."""
        unique_markers = np.unique(markers[markers != 0])
        
        if len(unique_markers) == 0:
            # No explicit markers, look for patterns in full grid
            return 'pattern_based'
        
        # Count marker pixels
        marker_positions = np.argwhere(markers != 0)
        n_markers = len(marker_positions)
        
        if n_markers == 1:
            return 'single_point'
        elif n_markers == 2:
            # Check if diagonal corners
            r1, c1 = marker_positions[0]
            r2, c2 = marker_positions[1]
            if abs(r1 - r2) > 1 and abs(c1 - c2) > 1:
                return 'corners'
            else:
                return 'boundary'
        elif n_markers == 4:
            # Likely corners
            return 'corners'
        elif n_markers > 4:
            # Check if they form a boundary
            if self._forms_boundary(marker_positions):
                return 'boundary'
            else:
                return 'color_based'
        
        return 'color_based'
    
    def _learn_size_rule(self, 
                        full_grid: np.ndarray,
                        markers: np.ndarray, 
                        extracted: np.ndarray) -> Tuple[str, Dict]:
        """Learn how region size relates to markers/grid."""
        h_full, w_full = full_grid.shape
        h_ext, w_ext = extracted.shape
        
        # Check if fixed size
        if h_ext < h_full / 3 and w_ext < w_full / 3:
            return 'fixed', {'height': h_ext, 'width': w_ext}
        
        # Check if relative to grid
        h_ratio = h_ext / h_full
        w_ratio = w_ext / w_full
        
        if 0.2 < h_ratio < 0.8 and 0.2 < w_ratio < 0.8:
            return 'relative_to_grid', {'h_ratio': h_ratio, 'w_ratio': w_ratio}
        
        # Check if marker dependent
        marker_positions = np.argwhere(markers != 0)
        if len(marker_positions) >= 2:
            # Region defined by markers
            min_r = marker_positions[:, 0].min()
            max_r = marker_positions[:, 0].max()
            min_c = marker_positions[:, 1].min()
            max_c = marker_positions[:, 1].max()
            
            if (max_r - min_r + 1) == h_ext and (max_c - min_c + 1) == w_ext:
                return 'marker_dependent', {'expand': 0}
            elif abs((max_r - min_r + 1) - h_ext) <= 2:
                return 'marker_dependent', {'expand': 1}
        
        return 'fixed', {'height': h_ext, 'width': w_ext}
    
    def _learn_position_rule(self,
                            full_grid: np.ndarray,
                            markers: np.ndarray,
                            extracted: np.ndarray) -> Tuple[str, Dict]:
        """Learn how region position relates to markers."""
        # Find where extracted region comes from in full grid
        h_ext, w_ext = extracted.shape
        h_full, w_full = full_grid.shape
        
        # Try to locate extracted region in full grid
        for r in range(h_full - h_ext + 1):
            for c in range(w_full - w_ext + 1):
                if np.array_equal(full_grid[r:r+h_ext, c:c+w_ext], extracted):
                    # Found match
                    region_r, region_c = r, c
                    break
        else:
            # Couldn't find exact match, approximate
            region_r, region_c = 0, 0
        
        marker_positions = np.argwhere(markers != 0)
        
        if len(marker_positions) == 0:
            return 'at_marker', {'offset_r': 0, 'offset_c': 0}
        
        # Check relationship to markers
        if len(marker_positions) == 1:
            mr, mc = marker_positions[0]
            offset_r = region_r - mr
            offset_c = region_c - mc
            return 'offset_from_marker', {'offset_r': offset_r, 'offset_c': offset_c}
        
        # Multiple markers - check if between them
        min_mr = marker_positions[:, 0].min()
        max_mr = marker_positions[:, 0].max()
        min_mc = marker_positions[:, 1].min()
        max_mc = marker_positions[:, 1].max()
        
        if min_mr <= region_r <= max_mr and min_mc <= region_c <= max_mc:
            return 'between_markers', {'alignment': 'contained'}
        
        return 'at_marker', {'offset_r': 0, 'offset_c': 0}
    
    def _extract_by_corners(self, 
                           grid: np.ndarray, 
                           markers: np.ndarray,
                           rule: ExtractionRule) -> Optional[np.ndarray]:
        """Extract region defined by corner markers."""
        marker_positions = np.argwhere(markers != 0)
        
        if len(marker_positions) < 2:
            return None
        
        # Find bounding box
        min_r = marker_positions[:, 0].min()
        max_r = marker_positions[:, 0].max()
        min_c = marker_positions[:, 1].min()
        max_c = marker_positions[:, 1].max()
        
        # Apply expansion if needed
        if rule.size_rule == 'marker_dependent':
            expand = rule.size_params.get('expand', 0)
            min_r = max(0, min_r - expand)
            max_r = min(grid.shape[0] - 1, max_r + expand)
            min_c = max(0, min_c - expand)
            max_c = min(grid.shape[1] - 1, max_c + expand)
        
        return grid[min_r:max_r+1, min_c:max_c+1]
    
    def _extract_by_boundary(self,
                           grid: np.ndarray,
                           markers: np.ndarray, 
                           rule: ExtractionRule) -> Optional[np.ndarray]:
        """Extract region defined by boundary markers."""
        # Find enclosed region
        h, w = grid.shape
        marker_positions = set(map(tuple, np.argwhere(markers != 0)))
        
        if not marker_positions:
            return None
        
        # Find interior region (flood fill from non-marker positions)
        visited = np.zeros_like(grid, dtype=bool)
        interior = []
        
        for r in range(h):
            for c in range(w):
                if (r, c) not in marker_positions and not visited[r, c]:
                    # Check if this is interior
                    if self._is_interior((r, c), marker_positions, grid.shape):
                        interior.append((r, c))
                        visited[r, c] = True
        
        if not interior:
            return self._extract_by_corners(grid, markers, rule)
        
        # Extract bounding box of interior
        interior = np.array(interior)
        min_r, min_c = interior.min(axis=0)
        max_r, max_c = interior.max(axis=0)
        
        return grid[min_r:max_r+1, min_c:max_c+1]
    
    def _extract_by_single_point(self,
                                grid: np.ndarray,
                                markers: np.ndarray,
                                rule: ExtractionRule) -> Optional[np.ndarray]:
        """Extract region around a single marker point."""
        marker_positions = np.argwhere(markers != 0)
        
        if len(marker_positions) != 1:
            return None
        
        mr, mc = marker_positions[0]
        
        # Use rule parameters for size
        if rule.size_rule == 'fixed':
            h = rule.size_params['height']
            w = rule.size_params['width']
        else:
            # Default 3x3 around marker
            h, w = 3, 3
        
        # Calculate region bounds
        offset_r = rule.position_params.get('offset_r', -h//2)
        offset_c = rule.position_params.get('offset_c', -w//2)
        
        r1 = max(0, mr + offset_r)
        c1 = max(0, mc + offset_c)
        r2 = min(grid.shape[0], r1 + h)
        c2 = min(grid.shape[1], c1 + w)
        
        return grid[r1:r2, c1:c2]
    
    def _extract_by_color(self,
                         grid: np.ndarray,
                         markers: np.ndarray,
                         rule: ExtractionRule) -> Optional[np.ndarray]:
        """Extract regions based on color patterns."""
        # Find all positions of marker color
        marker_color = markers[markers != 0][0] if np.any(markers != 0) else 0
        
        if marker_color == 0:
            return None
        
        # Find connected component of this color
        color_positions = np.argwhere(grid == marker_color)
        
        if len(color_positions) == 0:
            return None
        
        # Extract bounding box
        min_r = color_positions[:, 0].min()
        max_r = color_positions[:, 0].max()
        min_c = color_positions[:, 1].min()
        max_c = color_positions[:, 1].max()
        
        return grid[min_r:max_r+1, min_c:max_c+1]
    
    def _extract_by_pattern(self,
                           grid: np.ndarray,
                           markers: np.ndarray,
                           rule: ExtractionRule) -> Optional[np.ndarray]:
        """Extract based on detected patterns."""
        # Look for patterns like symmetric objects, repeated structures
        h, w = grid.shape
        
        # Try to find a notable subregion
        for size in [3, 5, 7]:
            if size > min(h, w):
                continue
                
            for r in range(h - size + 1):
                for c in range(w - size + 1):
                    region = grid[r:r+size, c:c+size]
                    
                    # Check if region has interesting pattern
                    if self._has_pattern(region):
                        return region
        
        return None
    
    def _detect_markers(self, grid: np.ndarray) -> np.ndarray:
        """Auto-detect markers in grid."""
        # Simple heuristic: isolated pixels or special colors
        markers = np.zeros_like(grid)
        h, w = grid.shape
        
        for r in range(h):
            for c in range(w):
                if grid[r, c] != 0:
                    # Check if isolated
                    neighbors = 0
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != 0:
                            neighbors += 1
                    
                    if neighbors <= 1:
                        markers[r, c] = grid[r, c]
        
        return markers
    
    def _heuristic_extraction(self, grid: np.ndarray, markers: np.ndarray) -> Optional[np.ndarray]:
        """Fallback heuristic extraction when no rules apply."""
        # Try to extract the most interesting region
        h, w = grid.shape
        
        # Find non-zero region
        non_zero = np.argwhere(grid != 0)
        if len(non_zero) == 0:
            return None
        
        # Extract bounding box of non-zero elements
        min_r = non_zero[:, 0].min()
        max_r = non_zero[:, 0].max()
        min_c = non_zero[:, 1].min()
        max_c = non_zero[:, 1].max()
        
        # Add small padding
        min_r = max(0, min_r - 1)
        max_r = min(h - 1, max_r + 1)
        min_c = max(0, min_c - 1)
        max_c = min(w - 1, max_c + 1)
        
        return grid[min_r:max_r+1, min_c:max_c+1]
    
    def _test_rule(self, rule: ExtractionRule, examples: List[Tuple]) -> bool:
        """Test if a rule generalizes across examples."""
        successes = 0
        
        for full_grid, markers, expected_region in examples:
            # Apply rule
            if rule.marker_pattern.marker_type in self.extraction_strategies:
                extractor = self.extraction_strategies[rule.marker_pattern.marker_type]
                extracted = extractor(full_grid, markers, rule)
                
                if extracted is not None and np.array_equal(extracted, expected_region):
                    successes += 1
        
        return successes >= len(examples) * 0.8
    
    def _merge_similar_rules(self, rules: List[ExtractionRule]) -> List[ExtractionRule]:
        """Merge similar rules to avoid redundancy."""
        if not rules:
            return []
        
        merged = []
        used = set()
        
        for i, rule1 in enumerate(rules):
            if i in used:
                continue
                
            similar = [rule1]
            for j, rule2 in enumerate(rules[i+1:], i+1):
                if j not in used and self._rules_similar(rule1, rule2):
                    similar.append(rule2)
                    used.add(j)
            
            # Create merged rule
            if len(similar) > 1:
                merged_rule = self._create_merged_rule(similar)
                merged.append(merged_rule)
            else:
                merged.append(rule1)
        
        return merged
    
    def _rules_similar(self, rule1: ExtractionRule, rule2: ExtractionRule) -> bool:
        """Check if two rules are similar enough to merge."""
        return (rule1.marker_pattern.marker_type == rule2.marker_pattern.marker_type and
                rule1.size_rule == rule2.size_rule and
                rule1.position_rule == rule2.position_rule)
    
    def _create_merged_rule(self, rules: List[ExtractionRule]) -> ExtractionRule:
        """Create a merged rule from similar rules."""
        # Average parameters
        size_params = {}
        position_params = {}
        
        for key in rules[0].size_params:
            if isinstance(rules[0].size_params[key], (int, float)):
                size_params[key] = np.mean([r.size_params.get(key, 0) for r in rules])
            else:
                size_params[key] = rules[0].size_params[key]
        
        for key in rules[0].position_params:
            if isinstance(rules[0].position_params[key], (int, float)):
                position_params[key] = np.mean([r.position_params.get(key, 0) for r in rules])
            else:
                position_params[key] = rules[0].position_params[key]
        
        return ExtractionRule(
            name=f"merged_{rules[0].marker_pattern.marker_type}",
            marker_pattern=rules[0].marker_pattern,
            size_rule=rules[0].size_rule,
            size_params=size_params,
            position_rule=rules[0].position_rule,
            position_params=position_params,
            confidence=np.mean([r.confidence for r in rules])
        )
    
    def _analyze_marker_properties(self, markers: np.ndarray) -> Dict[str, Any]:
        """Analyze properties of markers."""
        unique_values = np.unique(markers[markers != 0])
        positions = np.argwhere(markers != 0)
        
        return {
            'n_markers': len(positions),
            'colors': list(unique_values),
            'positions': positions.tolist(),
            'density': len(positions) / (markers.shape[0] * markers.shape[1])
        }
    
    def _forms_boundary(self, positions: np.ndarray) -> bool:
        """Check if positions form a boundary."""
        if len(positions) < 4:
            return False
        
        # Check if positions form a connected path
        # Simple heuristic: many positions are adjacent
        adjacent_count = 0
        position_set = set(map(tuple, positions))
        
        for r, c in positions:
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                if (r+dr, c+dc) in position_set:
                    adjacent_count += 1
                    break
        
        return adjacent_count >= len(positions) * 0.7
    
    def _is_interior(self, pos: Tuple[int, int], 
                    boundary: set, 
                    grid_shape: Tuple[int, int]) -> bool:
        """Check if position is interior to boundary."""
        r, c = pos
        h, w = grid_shape
        
        # Ray casting algorithm - count boundary crossings
        crossings = 0
        for c2 in range(c + 1, w):
            if (r, c2) in boundary:
                crossings += 1
        
        # Odd crossings = interior
        return crossings % 2 == 1
    
    def _has_pattern(self, region: np.ndarray) -> bool:
        """Check if region has an interesting pattern."""
        # Simple checks for now
        unique_values = np.unique(region)
        
        # Has multiple colors
        if len(unique_values) > 2:
            return True
        
        # Has symmetry
        if np.array_equal(region, np.fliplr(region)) or np.array_equal(region, np.flipud(region)):
            return True
        
        # Has structure (not all same value)
        if len(unique_values) > 1 and np.sum(region != 0) > region.size * 0.2:
            return True
        
        return False