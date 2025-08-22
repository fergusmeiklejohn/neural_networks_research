"""Enhanced Primitive Library for ARC-AGI Tasks.

This module provides high-level primitives that ARC tasks actually require,
moving beyond simple geometric transforms to object detection, pattern
recognition, and logical operations.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set, Any
from dataclasses import dataclass
from collections import deque


@dataclass
class Component:
    """Represents a connected component in the grid."""
    pixels: List[Tuple[int, int]]
    color: int
    bounding_box: Tuple[int, int, int, int]  # min_row, min_col, max_row, max_col
    
    @property
    def size(self) -> int:
        return len(self.pixels)
    
    @property
    def center(self) -> Tuple[float, float]:
        rows = [p[0] for p in self.pixels]
        cols = [p[1] for p in self.pixels]
        return (sum(rows) / len(rows), sum(cols) / len(cols))


@dataclass
class Region:
    """Represents a region (possibly enclosed) in the grid."""
    pixels: List[Tuple[int, int]]
    boundary: List[Tuple[int, int]]
    is_enclosed: bool
    

class ARCPrimitives:
    """Collection of high-level primitives for ARC tasks."""
    
    # ============= Object Detection Primitives =============
    
    @staticmethod
    def find_connected_components(grid: np.ndarray, color: Optional[int] = None) -> List[Component]:
        """Find all connected components of a specific color (or all non-zero if color=None)."""
        h, w = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        components = []
        
        def bfs(start_r: int, start_c: int, target_color: int) -> Component:
            """BFS to find all pixels in a connected component."""
            queue = deque([(start_r, start_c)])
            pixels = []
            min_r, min_c = start_r, start_c
            max_r, max_c = start_r, start_c
            
            while queue:
                r, c = queue.popleft()
                if visited[r, c]:
                    continue
                    
                visited[r, c] = True
                pixels.append((r, c))
                
                # Update bounding box
                min_r, min_c = min(min_r, r), min(min_c, c)
                max_r, max_c = max(max_r, r), max(max_c, c)
                
                # Check 4-connected neighbors
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < h and 0 <= nc < w and 
                        not visited[nr, nc] and grid[nr, nc] == target_color):
                        queue.append((nr, nc))
            
            return Component(pixels, target_color, (min_r, min_c, max_r, max_c))
        
        # Find all components
        for r in range(h):
            for c in range(w):
                if not visited[r, c]:
                    if color is None:
                        if grid[r, c] != 0:  # Any non-zero color
                            components.append(bfs(r, c, grid[r, c]))
                    elif grid[r, c] == color:
                        components.append(bfs(r, c, color))
        
        return components
    
    @staticmethod
    def find_enclosed_regions(grid: np.ndarray, boundary_color: int) -> List[Region]:
        """Find regions enclosed by a specific boundary color."""
        h, w = grid.shape
        regions = []
        visited = np.zeros_like(grid, dtype=bool)
        
        # First, find all boundary pixels
        boundary_mask = (grid == boundary_color)
        
        # Find potential enclosed regions (non-boundary pixels)
        for r in range(1, h-1):  # Skip edges
            for c in range(1, w-1):
                if not visited[r, c] and not boundary_mask[r, c]:
                    # BFS to explore this region
                    region_pixels = []
                    boundary_pixels = set()
                    is_enclosed = True
                    queue = deque([(r, c)])
                    
                    while queue:
                        cr, cc = queue.popleft()
                        if visited[cr, cc]:
                            continue
                        
                        # Check if we reached the edge (not enclosed)
                        if cr == 0 or cr == h-1 or cc == 0 or cc == w-1:
                            if not boundary_mask[cr, cc]:
                                is_enclosed = False
                        
                        visited[cr, cc] = True
                        region_pixels.append((cr, cc))
                        
                        # Check neighbors
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                if boundary_mask[nr, nc]:
                                    boundary_pixels.add((nr, nc))
                                elif not visited[nr, nc]:
                                    queue.append((nr, nc))
                    
                    if region_pixels and is_enclosed:
                        regions.append(Region(region_pixels, list(boundary_pixels), is_enclosed))
        
        return regions
    
    @staticmethod
    def get_bounding_box(component: Component) -> Tuple[int, int, int, int]:
        """Get the bounding box of a component."""
        return component.bounding_box
    
    # ============= Object Manipulation Primitives =============
    
    @staticmethod
    def move_object(grid: np.ndarray, component: Component, dx: int, dy: int) -> np.ndarray:
        """Move an object (connected component) by (dx, dy)."""
        result = grid.copy()
        
        # Clear original position
        for r, c in component.pixels:
            result[r, c] = 0
        
        # Place at new position
        h, w = grid.shape
        for r, c in component.pixels:
            new_r, new_c = r + dy, c + dx
            if 0 <= new_r < h and 0 <= new_c < w:
                result[new_r, new_c] = component.color
        
        return result
    
    @staticmethod
    def copy_object(grid: np.ndarray, component: Component, 
                   positions: List[Tuple[int, int]]) -> np.ndarray:
        """Copy an object to multiple positions."""
        result = grid.copy()
        
        # Get component's relative positions
        min_r, min_c, _, _ = component.bounding_box
        relative_pixels = [(r - min_r, c - min_c) for r, c in component.pixels]
        
        # Place copies at each position
        h, w = grid.shape
        for pos_r, pos_c in positions:
            for rel_r, rel_c in relative_pixels:
                new_r, new_c = pos_r + rel_r, pos_c + rel_c
                if 0 <= new_r < h and 0 <= new_c < w:
                    result[new_r, new_c] = component.color
        
        return result
    
    # ============= Region Operations =============
    
    @staticmethod
    def flood_fill(grid: np.ndarray, start_r: int, start_c: int, new_color: int) -> np.ndarray:
        """Flood fill from a starting point with a new color."""
        result = grid.copy()
        h, w = grid.shape
        
        if not (0 <= start_r < h and 0 <= start_c < w):
            return result
        
        old_color = grid[start_r, start_c]
        if old_color == new_color:
            return result
        
        queue = deque([(start_r, start_c)])
        visited = set()
        
        while queue:
            r, c = queue.popleft()
            if (r, c) in visited:
                continue
            
            visited.add((r, c))
            result[r, c] = new_color
            
            # Check 4-connected neighbors
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < h and 0 <= nc < w and 
                    (nr, nc) not in visited and grid[nr, nc] == old_color):
                    queue.append((nr, nc))
        
        return result
    
    @staticmethod
    def fill_enclosed_regions(grid: np.ndarray, boundary_color: int, fill_color: int) -> np.ndarray:
        """Fill all enclosed regions with a specific color."""
        result = grid.copy()
        regions = ARCPrimitives.find_enclosed_regions(grid, boundary_color)
        
        for region in regions:
            if region.is_enclosed:
                for r, c in region.pixels:
                    result[r, c] = fill_color
        
        return result
    
    # ============= Pattern Detection Primitives =============
    
    @staticmethod
    def find_repeating_pattern(grid: np.ndarray) -> Optional[Tuple[np.ndarray, int, int]]:
        """Find if the grid contains a repeating pattern. 
        Returns (pattern, period_x, period_y) or None."""
        h, w = grid.shape
        
        # Try different pattern sizes
        for ph in range(1, h // 2 + 1):  # Pattern height
            for pw in range(1, w // 2 + 1):  # Pattern width
                # Check if this pattern size tiles the grid
                if h % ph == 0 and w % pw == 0:
                    pattern = grid[:ph, :pw]
                    is_repeating = True
                    
                    # Check all tiles
                    for i in range(0, h, ph):
                        for j in range(0, w, pw):
                            tile = grid[i:i+ph, j:j+pw]
                            if not np.array_equal(tile, pattern):
                                is_repeating = False
                                break
                        if not is_repeating:
                            break
                    
                    if is_repeating:
                        return (pattern, pw, ph)
        
        return None
    
    @staticmethod
    def find_symmetry_axes(grid: np.ndarray) -> Dict[str, bool]:
        """Find symmetry axes in the grid."""
        h, w = grid.shape
        symmetries = {}
        
        # Check horizontal symmetry (flip up-down)
        symmetries['horizontal'] = np.array_equal(grid, np.flipud(grid))
        
        # Check vertical symmetry (flip left-right)
        symmetries['vertical'] = np.array_equal(grid, np.fliplr(grid))
        
        # Check diagonal symmetries (only for square grids)
        if h == w:
            symmetries['diagonal_main'] = np.array_equal(grid, grid.T)
            symmetries['diagonal_anti'] = np.array_equal(grid, np.rot90(grid, 2).T)
        
        # Check rotational symmetries
        if h == w:
            symmetries['rot90'] = np.array_equal(grid, np.rot90(grid))
            symmetries['rot180'] = np.array_equal(grid, np.rot90(grid, 2))
            symmetries['rot270'] = np.array_equal(grid, np.rot90(grid, 3))
        
        return symmetries
    
    # ============= Pattern Application Primitives =============
    
    @staticmethod
    def tile_pattern(pattern: np.ndarray, grid_size: Tuple[int, int], 
                    arrangement: str = 'regular') -> np.ndarray:
        """Tile a pattern to fill a grid of given size."""
        target_h, target_w = grid_size
        pattern_h, pattern_w = pattern.shape
        
        if arrangement == 'regular':
            # Simple tiling
            result = np.zeros((target_h, target_w), dtype=pattern.dtype)
            for i in range(0, target_h, pattern_h):
                for j in range(0, target_w, pattern_w):
                    h_end = min(i + pattern_h, target_h)
                    w_end = min(j + pattern_w, target_w)
                    result[i:h_end, j:w_end] = pattern[:h_end-i, :w_end-j]
            return result
            
        elif arrangement == '3x3':
            # Special 3x3 arrangement (common in ARC)
            if target_h == 3 * pattern_h and target_w == 3 * pattern_w:
                result = np.zeros((target_h, target_w), dtype=pattern.dtype)
                # Place pattern in specific positions (e.g., corners and center)
                positions = [(0, 0), (0, 2*pattern_w), (pattern_h, pattern_w),
                           (2*pattern_h, 0), (2*pattern_h, 2*pattern_w)]
                for r, c in positions:
                    result[r:r+pattern_h, c:c+pattern_w] = pattern
                return result
        
        return np.zeros((target_h, target_w), dtype=pattern.dtype)
    
    @staticmethod
    def extend_pattern(grid: np.ndarray, direction: str, steps: int) -> np.ndarray:
        """Extend a pattern in a given direction."""
        h, w = grid.shape
        
        if direction == 'right':
            # Detect pattern period
            period = 1
            for p in range(1, w):
                if np.array_equal(grid[:, :p], grid[:, p:2*p]):
                    period = p
                    break
            
            # Extend
            new_w = w + steps * period
            result = np.zeros((h, new_w), dtype=grid.dtype)
            result[:, :w] = grid
            for i in range(steps):
                start = w + i * period
                result[:, start:start+period] = grid[:, :period]
            return result
            
        elif direction == 'down':
            # Similar for downward extension
            period = 1
            for p in range(1, h):
                if np.array_equal(grid[:p, :], grid[p:2*p, :]):
                    period = p
                    break
            
            new_h = h + steps * period
            result = np.zeros((new_h, w), dtype=grid.dtype)
            result[:h, :] = grid
            for i in range(steps):
                start = h + i * period
                result[start:start+period, :] = grid[:period, :]
            return result
        
        return grid
    
    # ============= Counting Primitives =============
    
    @staticmethod
    def count_objects(grid: np.ndarray, min_size: int = 1) -> int:
        """Count distinct objects (connected components) in the grid."""
        components = ARCPrimitives.find_connected_components(grid)
        return len([c for c in components if c.size >= min_size])
    
    @staticmethod
    def count_colors(grid: np.ndarray) -> Dict[int, int]:
        """Count occurrences of each color."""
        unique, counts = np.unique(grid, return_counts=True)
        return dict(zip(unique, counts))
    
    @staticmethod
    def count_neighbors(grid: np.ndarray, r: int, c: int, 
                       color: Optional[int] = None, distance: int = 1) -> int:
        """Count neighbors of a specific color within a distance."""
        h, w = grid.shape
        count = 0
        
        for dr in range(-distance, distance + 1):
            for dc in range(-distance, distance + 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if color is None or grid[nr, nc] == color:
                        count += 1
        
        return count
    
    # ============= Logical Operations =============
    
    @staticmethod
    def apply_rule(grid: np.ndarray, condition: callable, action: callable) -> np.ndarray:
        """Apply an action where a condition is true."""
        result = grid.copy()
        h, w = grid.shape
        
        for r in range(h):
            for c in range(w):
                if condition(grid, r, c):
                    result[r, c] = action(grid, r, c)
        
        return result
    
    @staticmethod
    def combine_grids(grid1: np.ndarray, grid2: np.ndarray, 
                     operation: str = 'or') -> np.ndarray:
        """Combine two grids using logical operations."""
        if operation == 'or':
            return np.maximum(grid1, grid2)
        elif operation == 'and':
            return np.minimum(grid1, grid2)
        elif operation == 'xor':
            return np.where(grid1 != grid2, np.maximum(grid1, grid2), 0)
        else:
            return grid1
    
    # ============= Topological Primitives =============
    
    @staticmethod
    def is_inside(point: Tuple[int, int], region: Region) -> bool:
        """Check if a point is inside a region."""
        return point in region.pixels
    
    @staticmethod
    def is_adjacent(obj1: Component, obj2: Component) -> bool:
        """Check if two objects are adjacent (8-connected)."""
        for r1, c1 in obj1.pixels:
            for r2, c2 in obj2.pixels:
                if abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1:
                    return True
        return False
    
    @staticmethod
    def find_path(grid: np.ndarray, start: Tuple[int, int], 
                 end: Tuple[int, int], walkable_colors: Set[int]) -> Optional[List[Tuple[int, int]]]:
        """Find a path from start to end through walkable colors."""
        h, w = grid.shape
        
        # A* pathfinding
        from heapq import heappush, heappop
        
        def heuristic(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, end)}
        
        while open_set:
            _, current = heappop(open_set)
            
            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return list(reversed(path))
            
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dr, current[1] + dc)
                
                if (0 <= neighbor[0] < h and 0 <= neighbor[1] < w and
                    grid[neighbor[0], neighbor[1]] in walkable_colors):
                    
                    tentative_g = g_score[current] + 1
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + heuristic(neighbor, end)
                        heappush(open_set, (f_score[neighbor], neighbor))
        
        return None