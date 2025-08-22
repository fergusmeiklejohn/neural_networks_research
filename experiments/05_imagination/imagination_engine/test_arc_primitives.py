"""Test suite for ARC primitives using actual ARC tasks."""

import numpy as np
import json
from pathlib import Path
from arc_primitives import ARCPrimitives
from arc_data_loader import load_arc_training_data, prepare_task_for_hti


def test_object_detection():
    """Test object detection primitives."""
    print("\n=== Testing Object Detection ===")
    
    # Create a simple test grid with multiple objects
    grid = np.array([
        [0, 1, 1, 0, 0, 2, 2],
        [0, 1, 1, 0, 0, 0, 2],
        [0, 0, 0, 0, 3, 3, 0],
        [4, 4, 0, 0, 3, 3, 0],
        [4, 0, 0, 0, 0, 0, 0]
    ])
    
    # Find all components
    components = ARCPrimitives.find_connected_components(grid)
    print(f"Found {len(components)} components")
    for i, comp in enumerate(components):
        print(f"  Component {i}: color={comp.color}, size={comp.size}, "
              f"center={comp.center}, bbox={comp.bounding_box}")
    
    # Find components of specific color
    blue_components = ARCPrimitives.find_connected_components(grid, color=1)
    print(f"\nFound {len(blue_components)} blue (1) components")
    
    assert len(components) == 4, "Should find 4 components"
    assert len(blue_components) == 1, "Should find 1 blue component"
    print("✓ Object detection tests passed")


def test_region_operations():
    """Test region filling operations."""
    print("\n=== Testing Region Operations ===")
    
    # Create a grid with enclosed regions
    grid = np.array([
        [3, 3, 3, 3, 3, 0, 0],
        [3, 0, 0, 0, 3, 0, 0],
        [3, 0, 0, 0, 3, 0, 0],
        [3, 0, 0, 0, 3, 0, 0],
        [3, 3, 3, 3, 3, 0, 0]
    ])
    
    # Find enclosed regions
    regions = ARCPrimitives.find_enclosed_regions(grid, boundary_color=3)
    print(f"Found {len(regions)} enclosed regions")
    
    # Fill enclosed regions
    filled = ARCPrimitives.fill_enclosed_regions(grid, boundary_color=3, fill_color=4)
    print("\nOriginal grid:")
    print(grid)
    print("\nFilled grid:")
    print(filled)
    
    # Test flood fill
    flood_filled = ARCPrimitives.flood_fill(grid, 1, 1, 5)
    print("\nFlood filled from (1,1) with color 5:")
    print(flood_filled)
    
    assert len(regions) == 1, "Should find 1 enclosed region"
    assert filled[2, 2] == 4, "Enclosed region should be filled with color 4"
    print("✓ Region operation tests passed")


def test_pattern_detection():
    """Test pattern detection primitives."""
    print("\n=== Testing Pattern Detection ===")
    
    # Create a repeating pattern
    pattern = np.array([[1, 2], [2, 1]])
    grid = ARCPrimitives.tile_pattern(pattern, (6, 6), 'regular')
    print("Tiled grid:")
    print(grid)
    
    # Detect the pattern
    detected = ARCPrimitives.find_repeating_pattern(grid)
    if detected:
        detected_pattern, pw, ph = detected
        print(f"\nDetected pattern with period ({pw}, {ph}):")
        print(detected_pattern)
        assert np.array_equal(detected_pattern, pattern), "Should detect correct pattern"
    
    # Test symmetry detection
    symmetric_grid = np.array([
        [1, 2, 3, 2, 1],
        [2, 3, 4, 3, 2],
        [3, 4, 5, 4, 3],
        [2, 3, 4, 3, 2],
        [1, 2, 3, 2, 1]
    ])
    
    symmetries = ARCPrimitives.find_symmetry_axes(symmetric_grid)
    print(f"\nSymmetries found: {symmetries}")
    assert symmetries['horizontal'], "Should detect horizontal symmetry"
    assert symmetries['vertical'], "Should detect vertical symmetry"
    
    print("✓ Pattern detection tests passed")


def test_counting_operations():
    """Test counting primitives."""
    print("\n=== Testing Counting Operations ===")
    
    grid = np.array([
        [1, 1, 0, 2, 2],
        [1, 0, 0, 2, 2],
        [0, 3, 3, 0, 0],
        [0, 3, 3, 0, 4],
        [0, 0, 0, 4, 4]
    ])
    
    # Count objects
    num_objects = ARCPrimitives.count_objects(grid)
    print(f"Number of objects: {num_objects}")
    
    # Count colors
    color_counts = ARCPrimitives.count_colors(grid)
    print(f"Color counts: {color_counts}")
    
    # Count neighbors
    neighbors = ARCPrimitives.count_neighbors(grid, 2, 2, color=3)
    print(f"Neighbors of (2,2) with color 3: {neighbors}")
    
    assert num_objects == 4, "Should count 4 objects"
    assert color_counts[0] == 11, "Should have 11 zeros"
    assert neighbors == 3, "Should have 3 neighbors with color 3"
    
    print("✓ Counting operation tests passed")


def test_on_real_arc_task():
    """Test primitives on a real ARC task."""
    print("\n=== Testing on Real ARC Task ===")
    
    # Load the task we examined earlier (filling enclosed regions)
    task_file = Path(__file__).parent / "arc_agi_2_data" / "training" / "00d62c1b.json"
    
    if task_file.exists():
        with open(task_file, 'r') as f:
            task = json.load(f)
        
        # Get first training example
        input_grid = np.array(task['train'][0]['input'])
        output_grid = np.array(task['train'][0]['output'])
        
        print("Input shape:", input_grid.shape)
        print("Output shape:", output_grid.shape)
        
        # This task is about filling enclosed regions with color 4
        # Let's test our primitives
        filled = ARCPrimitives.fill_enclosed_regions(input_grid, boundary_color=3, fill_color=4)
        
        # Check if our solution matches
        matches = np.array_equal(filled, output_grid)
        print(f"Our solution matches expected output: {matches}")
        
        if not matches:
            # Show differences
            diff = np.where(filled != output_grid, 1, 0)
            print("Differences:")
            print(diff)
            print("Our output:")
            print(filled)
            print("Expected output:")
            print(output_grid)
    else:
        print("ARC task file not found - skipping real task test")


def test_tiling_task():
    """Test on the 3x3 tiling task."""
    print("\n=== Testing 3x3 Tiling Task ===")
    
    task_file = Path(__file__).parent / "arc_agi_2_data" / "training" / "007bbfb7.json"
    
    if task_file.exists():
        with open(task_file, 'r') as f:
            task = json.load(f)
        
        # Get first training example
        input_grid = np.array(task['train'][0]['input'])
        output_grid = np.array(task['train'][0]['output'])
        
        print("Input shape:", input_grid.shape)
        print("Output shape:", output_grid.shape)
        
        # This task creates a 3x3 tiling with specific placement
        # The pattern is placed at (0,0), (0,6), (3,3), (6,0), (6,6)
        h, w = input_grid.shape
        result = np.zeros((3*h, 3*w), dtype=input_grid.dtype)
        
        # Custom tiling for this specific pattern
        positions = [
            (0, 0), (0, 6), (3, 3), (6, 0), (6, 6)
        ]
        
        # Find non-zero component
        components = ARCPrimitives.find_connected_components(input_grid)
        if components:
            # Copy the whole input pattern to specific positions
            for r, c in positions:
                result[r:r+h, c:c+w] = input_grid
        
        matches = np.array_equal(result, output_grid)
        print(f"Our solution matches expected output: {matches}")
        
        if not matches:
            print("Note: This task requires specific placement logic beyond simple tiling")
    else:
        print("ARC task file not found - skipping tiling test")


def run_all_tests():
    """Run all primitive tests."""
    print("=" * 60)
    print("Testing ARC Primitives")
    print("=" * 60)
    
    test_object_detection()
    test_region_operations()
    test_pattern_detection()
    test_counting_operations()
    test_on_real_arc_task()
    test_tiling_task()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()