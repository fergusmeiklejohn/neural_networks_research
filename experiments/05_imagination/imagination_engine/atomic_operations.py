"""Atomic operations for primitive invention.

These are the fundamental, indivisible operations that all invented primitives
will be composed from. They operate at the pixel level and cannot be decomposed
further.

Key principle: Any ARC transformation can be expressed as a composition of these atoms.
"""

import numpy as np
from typing import Callable, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class AtomicOp:
    """Represents an atomic operation with its signature."""
    name: str
    func: Callable
    arity: int  # Number of arguments
    returns_grid: bool  # True if returns a grid, False if returns a value
    description: str


class AtomicOperations:
    """Collection of truly atomic operations for grid manipulation."""
    
    # ============= Pixel Access Operations =============
    
    @staticmethod
    def set_pixel(grid: np.ndarray, r: int, c: int, value: int) -> np.ndarray:
        """Set a single pixel value."""
        result = grid.copy()
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
            result[r, c] = value
        return result
    
    @staticmethod
    def get_pixel(grid: np.ndarray, r: int, c: int) -> int:
        """Get a single pixel value."""
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
            return int(grid[r, c])
        return 0
    
    @staticmethod
    def swap_pixels(grid: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> np.ndarray:
        """Swap two pixels."""
        result = grid.copy()
        h, w = grid.shape
        if (0 <= r1 < h and 0 <= c1 < w and 
            0 <= r2 < h and 0 <= c2 < w):
            result[r1, c1], result[r2, c2] = grid[r2, c2], grid[r1, c1]
        return result
    
    # ============= Region Operations =============
    
    @staticmethod
    def copy_region(grid: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> np.ndarray:
        """Copy a rectangular region."""
        if (0 <= r1 <= r2 < grid.shape[0] and 
            0 <= c1 <= c2 < grid.shape[1]):
            return grid[r1:r2+1, c1:c2+1].copy()
        return np.array([[0]])
    
    @staticmethod
    def paste_region(grid: np.ndarray, region: np.ndarray, r: int, c: int) -> np.ndarray:
        """Paste a region at specified position."""
        result = grid.copy()
        rh, rw = region.shape
        gh, gw = grid.shape
        
        # Calculate valid paste area
        r_end = min(r + rh, gh)
        c_end = min(c + rw, gw)
        r_start = max(0, r)
        c_start = max(0, c)
        
        # Calculate region slice
        region_r_start = max(0, -r)
        region_c_start = max(0, -c)
        region_r_end = region_r_start + (r_end - r_start)
        region_c_end = region_c_start + (c_end - c_start)
        
        if r_start < r_end and c_start < c_end:
            result[r_start:r_end, c_start:c_end] = region[
                region_r_start:region_r_end,
                region_c_start:region_c_end
            ]
        
        return result
    
    @staticmethod
    def fill_region(grid: np.ndarray, r1: int, c1: int, r2: int, c2: int, value: int) -> np.ndarray:
        """Fill a rectangular region with a value."""
        result = grid.copy()
        r1 = max(0, r1)
        c1 = max(0, c1)
        r2 = min(grid.shape[0] - 1, r2)
        c2 = min(grid.shape[1] - 1, c2)
        
        if r1 <= r2 and c1 <= c2:
            result[r1:r2+1, c1:c2+1] = value
        return result
    
    # ============= Iteration Operations =============
    
    @staticmethod
    def map_pixels(grid: np.ndarray, transform_fn: Callable[[int], int]) -> np.ndarray:
        """Apply a transformation function to each pixel."""
        result = grid.copy()
        h, w = grid.shape
        for r in range(h):
            for c in range(w):
                result[r, c] = transform_fn(grid[r, c])
        return result
    
    @staticmethod
    def map_pixels_with_position(grid: np.ndarray, 
                                 transform_fn: Callable[[int, int, int], int]) -> np.ndarray:
        """Apply a transformation function to each pixel with its position."""
        result = grid.copy()
        h, w = grid.shape
        for r in range(h):
            for c in range(w):
                result[r, c] = transform_fn(grid[r, c], r, c)
        return result
    
    @staticmethod
    def filter_pixels(grid: np.ndarray, condition_fn: Callable[[int], bool], 
                     true_val: int, false_val: int) -> np.ndarray:
        """Set pixels based on a condition."""
        result = grid.copy()
        h, w = grid.shape
        for r in range(h):
            for c in range(w):
                if condition_fn(grid[r, c]):
                    result[r, c] = true_val
                else:
                    result[r, c] = false_val
        return result
    
    # ============= Comparison Operations =============
    
    @staticmethod
    def pixels_equal(grid: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> bool:
        """Check if two pixels have the same value."""
        h, w = grid.shape
        if (0 <= r1 < h and 0 <= c1 < w and 
            0 <= r2 < h and 0 <= c2 < w):
            return grid[r1, c1] == grid[r2, c2]
        return False
    
    @staticmethod
    def count_value(grid: np.ndarray, value: int) -> int:
        """Count occurrences of a value."""
        return int(np.sum(grid == value))
    
    @staticmethod
    def find_value(grid: np.ndarray, value: int) -> List[Tuple[int, int]]:
        """Find all positions of a value."""
        positions = []
        h, w = grid.shape
        for r in range(h):
            for c in range(w):
                if grid[r, c] == value:
                    positions.append((r, c))
        return positions
    
    # ============= Boundary Operations =============
    
    @staticmethod
    def get_neighbors(grid: np.ndarray, r: int, c: int, 
                      connectivity: int = 4) -> List[Tuple[int, int, int]]:
        """Get neighboring pixels and their values."""
        neighbors = []
        h, w = grid.shape
        
        if connectivity == 4:
            deltas = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        else:  # 8-connectivity
            deltas = [(0, 1), (1, 0), (0, -1), (-1, 0),
                     (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                neighbors.append((nr, nc, int(grid[nr, nc])))
        
        return neighbors
    
    @staticmethod
    def is_boundary(grid: np.ndarray, r: int, c: int) -> bool:
        """Check if a pixel is on the grid boundary."""
        h, w = grid.shape
        return r == 0 or r == h - 1 or c == 0 or c == w - 1
    
    # ============= Grid Operations =============
    
    @staticmethod
    def create_grid(h: int, w: int, fill_value: int = 0) -> np.ndarray:
        """Create a new grid with specified dimensions."""
        return np.full((h, w), fill_value, dtype=np.int32)
    
    @staticmethod
    def get_shape(grid: np.ndarray) -> Tuple[int, int]:
        """Get grid dimensions."""
        return grid.shape
    
    @staticmethod
    def transpose(grid: np.ndarray) -> np.ndarray:
        """Transpose the grid."""
        return grid.T.copy()
    
    @staticmethod
    def flip_horizontal(grid: np.ndarray) -> np.ndarray:
        """Flip grid horizontally."""
        return np.flip(grid, axis=1).copy()
    
    @staticmethod
    def flip_vertical(grid: np.ndarray) -> np.ndarray:
        """Flip grid vertically."""
        return np.flip(grid, axis=0).copy()
    
    @staticmethod
    def rotate_90(grid: np.ndarray) -> np.ndarray:
        """Rotate grid 90 degrees clockwise."""
        return np.rot90(grid, -1).copy()
    
    # ============= Logical Operations =============
    
    @staticmethod
    def logical_and(grid1: np.ndarray, grid2: np.ndarray) -> np.ndarray:
        """Pixel-wise logical AND (non-zero values)."""
        return ((grid1 != 0) & (grid2 != 0)).astype(np.int32)
    
    @staticmethod
    def logical_or(grid1: np.ndarray, grid2: np.ndarray) -> np.ndarray:
        """Pixel-wise logical OR (non-zero values)."""
        return ((grid1 != 0) | (grid2 != 0)).astype(np.int32)
    
    @staticmethod
    def logical_xor(grid1: np.ndarray, grid2: np.ndarray) -> np.ndarray:
        """Pixel-wise logical XOR (non-zero values)."""
        return ((grid1 != 0) ^ (grid2 != 0)).astype(np.int32)
    
    @staticmethod
    def mask_where(grid: np.ndarray, mask: np.ndarray, value: int) -> np.ndarray:
        """Set pixels to value where mask is non-zero."""
        result = grid.copy()
        result[mask != 0] = value
        return result


def get_atomic_operations() -> List[AtomicOp]:
    """Get list of all atomic operations with metadata."""
    
    ops = [
        # Pixel operations
        AtomicOp("set_pixel", AtomicOperations.set_pixel, 4, True, 
                "Set pixel at (r,c) to value"),
        AtomicOp("get_pixel", AtomicOperations.get_pixel, 3, False,
                "Get pixel value at (r,c)"),
        AtomicOp("swap_pixels", AtomicOperations.swap_pixels, 5, True,
                "Swap pixels at two positions"),
        
        # Region operations
        AtomicOp("copy_region", AtomicOperations.copy_region, 5, True,
                "Copy rectangular region"),
        AtomicOp("paste_region", AtomicOperations.paste_region, 4, True,
                "Paste region at position"),
        AtomicOp("fill_region", AtomicOperations.fill_region, 6, True,
                "Fill rectangular region"),
        
        # Iteration operations
        AtomicOp("map_pixels", AtomicOperations.map_pixels, 2, True,
                "Transform each pixel"),
        AtomicOp("map_pixels_with_position", AtomicOperations.map_pixels_with_position, 2, True,
                "Transform each pixel using position"),
        AtomicOp("filter_pixels", AtomicOperations.filter_pixels, 4, True,
                "Filter pixels by condition"),
        
        # Comparison operations
        AtomicOp("pixels_equal", AtomicOperations.pixels_equal, 5, False,
                "Check if two pixels equal"),
        AtomicOp("count_value", AtomicOperations.count_value, 2, False,
                "Count occurrences of value"),
        AtomicOp("find_value", AtomicOperations.find_value, 2, False,
                "Find all positions of value"),
        
        # Boundary operations
        AtomicOp("get_neighbors", AtomicOperations.get_neighbors, 4, False,
                "Get neighboring pixels"),
        AtomicOp("is_boundary", AtomicOperations.is_boundary, 3, False,
                "Check if pixel on boundary"),
        
        # Grid operations
        AtomicOp("create_grid", AtomicOperations.create_grid, 3, True,
                "Create new grid"),
        AtomicOp("get_shape", AtomicOperations.get_shape, 1, False,
                "Get grid dimensions"),
        AtomicOp("transpose", AtomicOperations.transpose, 1, True,
                "Transpose grid"),
        AtomicOp("flip_horizontal", AtomicOperations.flip_horizontal, 1, True,
                "Flip horizontally"),
        AtomicOp("flip_vertical", AtomicOperations.flip_vertical, 1, True,
                "Flip vertically"),
        AtomicOp("rotate_90", AtomicOperations.rotate_90, 1, True,
                "Rotate 90 degrees"),
        
        # Logical operations
        AtomicOp("logical_and", AtomicOperations.logical_and, 2, True,
                "Logical AND of grids"),
        AtomicOp("logical_or", AtomicOperations.logical_or, 2, True,
                "Logical OR of grids"),
        AtomicOp("logical_xor", AtomicOperations.logical_xor, 2, True,
                "Logical XOR of grids"),
        AtomicOp("mask_where", AtomicOperations.mask_where, 3, True,
                "Apply mask to grid"),
    ]
    
    return ops