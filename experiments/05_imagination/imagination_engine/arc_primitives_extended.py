"""Extended ARC Primitives based on failure analysis.

Adds critical missing operations:
- Object manipulation (rotate, scale, mirror)
- Grid resizing operations
- Object duplication and repetition
- Color mapping and transformation
- Pattern continuation
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
from arc_primitives import ARCPrimitives, Component

class ARCPrimitivesExtended(ARCPrimitives):
    """Extended primitives for ARC tasks based on failure analysis."""
    
    # ============= Object Manipulation Primitives =============
    
    @staticmethod
    def rotate_object(grid: np.ndarray, component: Component, 
                     angle: int, center: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Rotate a specific object by 90, 180, or 270 degrees."""
        result = grid.copy()
        
        # Extract the object
        min_r, min_c, max_r, max_c = component.bounding_box
        object_grid = grid[min_r:max_r+1, min_c:max_c+1].copy()
        
        # Clear original position
        for r, c in component.pixels:
            result[r, c] = 0
        
        # Rotate the object
        rotations = angle // 90
        rotated = np.rot90(object_grid, rotations)
        
        # Calculate new position (centered if no center specified)
        if center is None:
            center_r = (min_r + max_r) // 2
            center_c = (min_c + max_c) // 2
        else:
            center_r, center_c = center
        
        new_h, new_w = rotated.shape
        new_min_r = center_r - new_h // 2
        new_min_c = center_c - new_w // 2
        
        # Place rotated object
        h, w = result.shape
        for dr in range(new_h):
            for dc in range(new_w):
                r, c = new_min_r + dr, new_min_c + dc
                if 0 <= r < h and 0 <= c < w and rotated[dr, dc] != 0:
                    result[r, c] = rotated[dr, dc]
        
        return result
    
    @staticmethod
    def mirror_object(grid: np.ndarray, component: Component, 
                     axis: str = 'horizontal') -> np.ndarray:
        """Mirror an object horizontally or vertically."""
        result = grid.copy()
        
        # Extract the object
        min_r, min_c, max_r, max_c = component.bounding_box
        object_grid = grid[min_r:max_r+1, min_c:max_c+1].copy()
        
        # Clear original position
        for r, c in component.pixels:
            result[r, c] = 0
        
        # Mirror the object
        if axis == 'horizontal':
            mirrored = np.flip(object_grid, axis=1)
        else:  # vertical
            mirrored = np.flip(object_grid, axis=0)
        
        # Place mirrored object
        result[min_r:max_r+1, min_c:max_c+1] = mirrored
        
        return result
    
    @staticmethod
    def scale_object(grid: np.ndarray, component: Component, 
                    scale_factor: int) -> np.ndarray:
        """Scale an object by an integer factor."""
        result = np.zeros_like(grid)
        
        # Extract the object
        min_r, min_c, max_r, max_c = component.bounding_box
        object_grid = grid[min_r:max_r+1, min_c:max_c+1]
        
        # Scale using repeat
        scaled = np.repeat(np.repeat(object_grid, scale_factor, axis=0), 
                          scale_factor, axis=1)
        
        # Calculate placement
        h, w = result.shape
        scaled_h, scaled_w = scaled.shape
        
        # Try to center the scaled object
        start_r = max(0, min_r - (scaled_h - (max_r - min_r + 1)) // 2)
        start_c = max(0, min_c - (scaled_w - (max_c - min_c + 1)) // 2)
        
        # Place scaled object (clip if needed)
        end_r = min(h, start_r + scaled_h)
        end_c = min(w, start_c + scaled_w)
        
        result[start_r:end_r, start_c:end_c] = scaled[:end_r-start_r, :end_c-start_c]
        
        # Copy rest of grid
        for r in range(h):
            for c in range(w):
                if result[r, c] == 0 and grid[r, c] != 0:
                    if not any((r, c) == p for p in component.pixels):
                        result[r, c] = grid[r, c]
        
        return result
    
    # ============= Grid Resizing Operations =============
    
    @staticmethod
    def resize_grid(grid: np.ndarray, new_shape: Tuple[int, int], 
                   mode: str = 'crop') -> np.ndarray:
        """Resize grid by cropping, padding, or repeating."""
        h, w = grid.shape
        new_h, new_w = new_shape
        
        if mode == 'crop':
            # Crop from center
            start_r = max(0, (h - new_h) // 2)
            start_c = max(0, (w - new_w) // 2)
            result = np.zeros(new_shape, dtype=grid.dtype)
            
            copy_h = min(new_h, h - start_r)
            copy_w = min(new_w, w - start_c)
            
            result[:copy_h, :copy_w] = grid[start_r:start_r+copy_h, 
                                           start_c:start_c+copy_w]
            
        elif mode == 'pad':
            # Pad with zeros
            if new_h >= h and new_w >= w:
                result = np.zeros(new_shape, dtype=grid.dtype)
                start_r = (new_h - h) // 2
                start_c = (new_w - w) // 2
                result[start_r:start_r+h, start_c:start_c+w] = grid
            else:
                # If smaller, crop
                result = ARCPrimitivesExtended.resize_grid(grid, new_shape, 'crop')
                
        elif mode == 'repeat':
            # Repeat pattern to fill
            result = np.zeros(new_shape, dtype=grid.dtype)
            for r in range(new_h):
                for c in range(new_w):
                    result[r, c] = grid[r % h, c % w]
        else:
            result = grid.copy()
        
        return result
    
    @staticmethod
    def extract_subgrid(grid: np.ndarray, 
                       top_left: Tuple[int, int], 
                       size: Tuple[int, int]) -> np.ndarray:
        """Extract a subgrid from specified position."""
        r, c = top_left
        h, w = size
        return grid[r:r+h, c:c+w].copy()
    
    # ============= Object Duplication and Repetition =============
    
    @staticmethod
    def duplicate_object(grid: np.ndarray, component: Component, 
                        positions: List[Tuple[int, int]]) -> np.ndarray:
        """Duplicate an object to multiple positions."""
        result = grid.copy()
        
        # Extract the object pattern
        min_r, min_c, max_r, max_c = component.bounding_box
        object_pattern = np.zeros((max_r - min_r + 1, max_c - min_c + 1), 
                                 dtype=grid.dtype)
        
        for r, c in component.pixels:
            object_pattern[r - min_r, c - min_c] = grid[r, c]
        
        # Place at each position
        h, w = result.shape
        obj_h, obj_w = object_pattern.shape
        
        for pos_r, pos_c in positions:
            for dr in range(obj_h):
                for dc in range(obj_w):
                    r, c = pos_r + dr, pos_c + dc
                    if 0 <= r < h and 0 <= c < w and object_pattern[dr, dc] != 0:
                        result[r, c] = object_pattern[dr, dc]
        
        return result
    
    @staticmethod
    def create_grid_of_objects(grid: np.ndarray, component: Component,
                              rows: int, cols: int, 
                              spacing: int = 1) -> np.ndarray:
        """Create a grid arrangement of an object."""
        # Extract object
        min_r, min_c, max_r, max_c = component.bounding_box
        obj_h = max_r - min_r + 1
        obj_w = max_c - min_c + 1
        
        # Calculate required grid size
        total_h = rows * obj_h + (rows - 1) * spacing
        total_w = cols * obj_w + (cols - 1) * spacing
        
        # Create result grid
        result = np.zeros((total_h, total_w), dtype=grid.dtype)
        
        # Extract object pattern
        object_pattern = np.zeros((obj_h, obj_w), dtype=grid.dtype)
        for r, c in component.pixels:
            object_pattern[r - min_r, c - min_c] = grid[r, c]
        
        # Place objects in grid
        for i in range(rows):
            for j in range(cols):
                start_r = i * (obj_h + spacing)
                start_c = j * (obj_w + spacing)
                result[start_r:start_r+obj_h, start_c:start_c+obj_w] = object_pattern
        
        return result
    
    # ============= Color Mapping and Transformation =============
    
    @staticmethod
    def map_colors(grid: np.ndarray, color_map: Dict[int, int]) -> np.ndarray:
        """Map colors according to a dictionary."""
        result = grid.copy()
        for old_color, new_color in color_map.items():
            result[grid == old_color] = new_color
        return result
    
    @staticmethod
    def swap_colors(grid: np.ndarray, color1: int, color2: int) -> np.ndarray:
        """Swap two colors in the grid."""
        result = grid.copy()
        mask1 = grid == color1
        mask2 = grid == color2
        result[mask1] = color2
        result[mask2] = color1
        return result
    
    @staticmethod
    def apply_color_gradient(grid: np.ndarray, component: Component,
                            colors: List[int], direction: str = 'horizontal') -> np.ndarray:
        """Apply a color gradient to an object."""
        result = grid.copy()
        
        if not colors or not component.pixels:
            return result
        
        min_r, min_c, max_r, max_c = component.bounding_box
        
        if direction == 'horizontal':
            width = max_c - min_c + 1
            for r, c in component.pixels:
                idx = int((c - min_c) * len(colors) / width)
                idx = min(idx, len(colors) - 1)
                result[r, c] = colors[idx]
        else:  # vertical
            height = max_r - min_r + 1
            for r, c in component.pixels:
                idx = int((r - min_r) * len(colors) / height)
                idx = min(idx, len(colors) - 1)
                result[r, c] = colors[idx]
        
        return result
    
    # ============= Pattern Continuation =============
    
    @staticmethod
    def continue_pattern(grid: np.ndarray, direction: str, 
                        steps: int = 1) -> np.ndarray:
        """Continue a detected pattern in a direction."""
        h, w = grid.shape
        
        if direction == 'right':
            # Detect period by comparing columns
            period = 1
            for p in range(1, w // 2):
                if np.array_equal(grid[:, :p], grid[:, p:2*p]):
                    period = p
                    break
            
            # Extend pattern
            new_w = w + steps * period
            result = np.zeros((h, new_w), dtype=grid.dtype)
            result[:, :w] = grid
            
            for i in range(steps):
                start_c = w + i * period
                result[:, start_c:start_c+period] = grid[:, :period]
                
        elif direction == 'down':
            # Detect period by comparing rows
            period = 1
            for p in range(1, h // 2):
                if np.array_equal(grid[:p, :], grid[p:2*p, :]):
                    period = p
                    break
            
            # Extend pattern
            new_h = h + steps * period
            result = np.zeros((new_h, w), dtype=grid.dtype)
            result[:h, :] = grid
            
            for i in range(steps):
                start_r = h + i * period
                result[start_r:start_r+period, :] = grid[:period, :]
        
        elif direction == 'radial':
            # Simple radial continuation (copy center pattern outward)
            result = grid.copy()
            center_r, center_c = h // 2, w // 2
            
            # Get center pattern (3x3 or 5x5)
            size = min(3, min(h, w) // 2)
            pattern = grid[center_r-size//2:center_r+size//2+1,
                          center_c-size//2:center_c+size//2+1]
            
            # Apply pattern at corners
            if steps > 0:
                positions = [(0, 0), (0, w-size), (h-size, 0), (h-size, w-size)]
                for r, c in positions:
                    if 0 <= r < h-size+1 and 0 <= c < w-size+1:
                        result[r:r+size, c:c+size] = pattern
        else:
            result = grid.copy()
        
        return result
    
    @staticmethod
    def interpolate_pattern(grid: np.ndarray, start_pos: Tuple[int, int],
                           end_pos: Tuple[int, int], steps: int = 3) -> np.ndarray:
        """Interpolate a pattern between two positions."""
        result = grid.copy()
        
        r1, c1 = start_pos
        r2, c2 = end_pos
        
        # Simple linear interpolation of positions
        for i in range(1, steps + 1):
            t = i / (steps + 1)
            r = int(r1 + t * (r2 - r1))
            c = int(c1 + t * (c2 - c1))
            
            # Copy pattern from start position
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                # Get 3x3 pattern around start
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        sr, sc = r1 + dr, c1 + dc
                        tr, tc = r + dr, c + dc
                        if (0 <= sr < grid.shape[0] and 0 <= sc < grid.shape[1] and
                            0 <= tr < grid.shape[0] and 0 <= tc < grid.shape[1]):
                            if grid[sr, sc] != 0:
                                result[tr, tc] = grid[sr, sc]
        
        return result
    
    # ============= Advanced Grouping Operations =============
    
    @staticmethod
    def group_by_color(grid: np.ndarray) -> Dict[int, List[Component]]:
        """Group all components by their color."""
        colors = np.unique(grid)
        groups = {}
        
        for color in colors:
            if color != 0:  # Skip background
                components = ARCPrimitives.find_connected_components(grid, color)
                if components:
                    groups[color] = components
        
        return groups
    
    @staticmethod
    def align_objects(grid: np.ndarray, components: List[Component],
                     alignment: str = 'horizontal') -> np.ndarray:
        """Align multiple objects horizontally or vertically."""
        if not components:
            return grid
        
        result = np.zeros_like(grid)
        
        if alignment == 'horizontal':
            # Align to same row
            target_row = sum(c.center[0] for c in components) // len(components)
            current_col = 1
            
            for comp in components:
                min_r, min_c, max_r, max_c = comp.bounding_box
                obj_h = max_r - min_r + 1
                obj_w = max_c - min_c + 1
                
                # Extract object
                for r, c in comp.pixels:
                    if 0 <= target_row + (r - min_r) < grid.shape[0]:
                        if current_col + (c - min_c) < grid.shape[1]:
                            result[int(target_row + (r - min_r)), 
                                  current_col + (c - min_c)] = grid[r, c]
                
                current_col += obj_w + 1
                
        elif alignment == 'vertical':
            # Align to same column
            target_col = sum(c.center[1] for c in components) // len(components)
            current_row = 1
            
            for comp in components:
                min_r, min_c, max_r, max_c = comp.bounding_box
                obj_h = max_r - min_r + 1
                obj_w = max_c - min_c + 1
                
                # Extract object
                for r, c in comp.pixels:
                    if 0 <= target_col + (c - min_c) < grid.shape[1]:
                        if current_row + (r - min_r) < grid.shape[0]:
                            result[current_row + (r - min_r),
                                  int(target_col + (c - min_c))] = grid[r, c]
                
                current_row += obj_h + 1
        
        return result
    
    # ============= Sorting and Ordering =============
    
    @staticmethod
    def sort_objects_by_size(grid: np.ndarray, 
                            direction: str = 'horizontal') -> np.ndarray:
        """Sort objects by size and arrange them."""
        components = ARCPrimitives.find_connected_components(grid)
        if not components:
            return grid
        
        # Sort by size
        components.sort(key=lambda c: c.size)
        
        # Arrange sorted objects
        return ARCPrimitivesExtended.align_objects(grid, components, direction)
    
    @staticmethod
    def sort_colors_by_frequency(grid: np.ndarray) -> np.ndarray:
        """Remap colors based on their frequency (most frequent = 1, etc)."""
        result = grid.copy()
        
        # Count frequencies
        unique, counts = np.unique(grid[grid != 0], return_counts=True)
        if len(unique) == 0:
            return result
        
        # Sort by frequency
        sorted_indices = np.argsort(-counts)  # Descending order
        
        # Create mapping
        color_map = {}
        for new_val, idx in enumerate(sorted_indices, start=1):
            old_val = unique[idx]
            color_map[old_val] = new_val
        
        # Apply mapping
        for old_color, new_color in color_map.items():
            result[grid == old_color] = new_color
        
        return result