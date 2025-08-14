#!/usr/bin/env python3
"""Enhanced object manipulation for ARC tasks.

This module provides intelligent object detection and manipulation by:
1. Learning transformations from training examples
2. Tracking object properties and relationships
3. Applying learned rules to test inputs
"""

from utils.imports import setup_project_paths

setup_project_paths()

import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

warnings.filterwarnings("ignore")


@dataclass
class Object:
    """Represents a discrete object in the grid."""

    id: int
    color: int
    pixels: np.ndarray  # Coordinates of all pixels
    bbox: Tuple[
        Tuple[int, int], Tuple[int, int]
    ]  # ((min_row, min_col), (max_row, max_col))
    center: Tuple[float, float]
    size: int
    shape_signature: str  # Shape identifier for matching

    def __hash__(self):
        return hash((self.id, self.color, self.shape_signature))


@dataclass
class ObjectTransformation:
    """Represents a transformation applied to an object."""

    type: str  # 'move', 'recolor', 'rotate', 'scale', 'delete', 'duplicate'
    params: Dict  # Transformation-specific parameters
    confidence: float


class SmartObjectManipulator:
    """Intelligent object manipulation based on learned patterns."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.transformation_cache = {}

    def extract_objects(self, grid: np.ndarray, background: int = 0) -> List[Object]:
        """Extract all discrete objects from grid."""
        objects = []
        obj_id = 0

        # Find connected components for each color
        for color in np.unique(grid):
            if color == background:
                continue

            mask = grid == color
            labeled, num_features = ndimage.label(mask)

            for feat_id in range(1, num_features + 1):
                obj_mask = labeled == feat_id
                pixels = np.argwhere(obj_mask)

                if len(pixels) > 0:
                    bbox_min = pixels.min(axis=0)
                    bbox_max = pixels.max(axis=0)
                    center = pixels.mean(axis=0)

                    # Create shape signature for matching
                    shape_sig = self._get_shape_signature(pixels)

                    obj = Object(
                        id=obj_id,
                        color=int(color),
                        pixels=pixels,
                        bbox=(tuple(bbox_min), tuple(bbox_max)),
                        center=tuple(center),
                        size=len(pixels),
                        shape_signature=shape_sig,
                    )
                    objects.append(obj)
                    obj_id += 1

        return objects

    def _get_shape_signature(self, pixels: np.ndarray) -> str:
        """Create a normalized shape signature for object matching."""
        if len(pixels) == 0:
            return "empty"

        # Normalize to origin
        min_r, min_c = pixels.min(axis=0)
        normalized = pixels - [min_r, min_c]

        # Sort pixels for consistent signature
        normalized = normalized[np.lexsort((normalized[:, 1], normalized[:, 0]))]

        # Create string signature
        return str(normalized.tolist())

    def learn_transformations(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[ObjectTransformation]:
        """Learn object transformations from training examples."""
        all_transformations = []

        for inp, out in examples:
            input_objects = self.extract_objects(inp)
            output_objects = self.extract_objects(out)

            # Match objects between input and output
            matches = self._match_objects(input_objects, output_objects)

            # Detect transformations for matched objects
            for inp_obj, out_obj in matches:
                if out_obj is None:
                    # Object was deleted
                    trans = ObjectTransformation(
                        type="delete",
                        params={"color": inp_obj.color, "size": inp_obj.size},
                        confidence=0.9,
                    )
                    all_transformations.append(trans)
                else:
                    # Detect type of transformation
                    trans = self._detect_transformation(
                        inp_obj, out_obj, inp.shape, out.shape
                    )
                    if trans:
                        all_transformations.append(trans)

            # Check for new objects (duplication or creation)
            matched_out_ids = {out_obj.id for _, out_obj in matches if out_obj}
            for out_obj in output_objects:
                if out_obj.id not in matched_out_ids:
                    # This is a new object
                    trans = self._detect_new_object_rule(
                        out_obj, input_objects, output_objects
                    )
                    if trans:
                        all_transformations.append(trans)

        # Consolidate and validate transformations
        return self._consolidate_transformations(all_transformations)

    def _match_objects(
        self, input_objects: List[Object], output_objects: List[Object]
    ) -> List[Tuple[Object, Optional[Object]]]:
        """Match objects between input and output grids."""
        matches = []
        used_output_ids = set()

        for inp_obj in input_objects:
            best_match = None
            best_score = 0.0

            for out_obj in output_objects:
                if out_obj.id in used_output_ids:
                    continue

                # Calculate match score
                score = self._calculate_match_score(inp_obj, out_obj)

                if score > best_score and score > 0.5:  # Threshold
                    best_score = score
                    best_match = out_obj

            if best_match:
                used_output_ids.add(best_match.id)
                matches.append((inp_obj, best_match))
            else:
                matches.append((inp_obj, None))  # Object deleted

        return matches

    def _calculate_match_score(self, obj1: Object, obj2: Object) -> float:
        """Calculate similarity score between two objects."""
        score = 0.0

        # Shape match (most important)
        if obj1.shape_signature == obj2.shape_signature:
            score += 0.5

        # Color match
        if obj1.color == obj2.color:
            score += 0.3

        # Size similarity
        size_ratio = min(obj1.size, obj2.size) / max(obj1.size, obj2.size)
        score += 0.1 * size_ratio

        # Position similarity (relative to grid)
        dist = np.linalg.norm(np.array(obj1.center) - np.array(obj2.center))
        max_dist = np.sqrt(2) * 30  # Max possible distance in 30x30 grid
        position_sim = 1.0 - (dist / max_dist)
        score += 0.1 * position_sim

        return score

    def _detect_transformation(
        self, inp_obj: Object, out_obj: Object, inp_shape: Tuple, out_shape: Tuple
    ) -> Optional[ObjectTransformation]:
        """Detect what transformation was applied to an object."""

        # Check for color change
        if inp_obj.color != out_obj.color:
            return ObjectTransformation(
                type="recolor",
                params={
                    "from_color": inp_obj.color,
                    "to_color": out_obj.color,
                    "condition": self._detect_recolor_condition(inp_obj, out_obj),
                },
                confidence=0.9,
            )

        # Check for movement
        inp_center = np.array(inp_obj.center)
        out_center = np.array(out_obj.center)

        # Adjust for grid size change if necessary
        if inp_shape != out_shape:
            scale = np.array(out_shape) / np.array(inp_shape)
            expected_center = inp_center * scale
        else:
            expected_center = inp_center

        movement = out_center - expected_center
        if np.linalg.norm(movement) > 0.5:
            return ObjectTransformation(
                type="move",
                params={
                    "delta": tuple(movement),
                    "absolute": tuple(out_center),
                    "relative_to": "center",
                },
                confidence=0.8,
            )

        # Check for rotation
        if self._is_rotated(inp_obj, out_obj):
            angle = self._detect_rotation_angle(inp_obj, out_obj)
            return ObjectTransformation(
                type="rotate",
                params={"angle": angle, "center": "object_center"},
                confidence=0.7,
            )

        # Check for scaling
        if inp_obj.size != out_obj.size:
            scale_factor = out_obj.size / inp_obj.size
            return ObjectTransformation(
                type="scale", params={"factor": scale_factor}, confidence=0.7
            )

        return None

    def _detect_recolor_condition(self, inp_obj: Object, out_obj: Object) -> str:
        """Detect condition for recoloring."""
        # Simple heuristics for now
        if inp_obj.size < 5:
            return "small_object"
        elif inp_obj.color == 1:
            return "was_blue"
        else:
            return "always"

    def _is_rotated(self, obj1: Object, obj2: Object) -> bool:
        """Check if object was rotated."""
        # Simple check: same size but different pixel arrangement
        if obj1.size != obj2.size:
            return False

        # Check if shape can be rotated to match
        shape1 = self._pixels_to_grid(obj1.pixels)
        shape2 = self._pixels_to_grid(obj2.pixels)

        for rot in range(4):
            if np.array_equal(shape1, shape2):
                return rot > 0
            shape1 = np.rot90(shape1)

        return False

    def _detect_rotation_angle(self, obj1: Object, obj2: Object) -> int:
        """Detect rotation angle (90, 180, or 270 degrees)."""
        shape1 = self._pixels_to_grid(obj1.pixels)
        shape2 = self._pixels_to_grid(obj2.pixels)

        for rot in range(1, 4):
            if np.array_equal(np.rot90(shape1, rot), shape2):
                return rot * 90

        return 0

    def _pixels_to_grid(self, pixels: np.ndarray) -> np.ndarray:
        """Convert pixel coordinates to a minimal bounding grid."""
        if len(pixels) == 0:
            return np.array([[]])

        min_r, min_c = pixels.min(axis=0)
        max_r, max_c = pixels.max(axis=0)

        grid = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=bool)
        for r, c in pixels:
            grid[r - min_r, c - min_c] = True

        return grid

    def _detect_new_object_rule(
        self, new_obj: Object, input_objects: List[Object], output_objects: List[Object]
    ) -> Optional[ObjectTransformation]:
        """Detect rule for creating new objects."""

        # Check if it's a duplicate of an existing object
        for inp_obj in input_objects:
            if inp_obj.shape_signature == new_obj.shape_signature:
                # It's a duplicate
                offset = np.array(new_obj.center) - np.array(inp_obj.center)
                return ObjectTransformation(
                    type="duplicate",
                    params={
                        "source_color": inp_obj.color,
                        "offset": tuple(offset),
                        "new_color": new_obj.color,
                    },
                    confidence=0.8,
                )

        # Check if it's a combination of objects
        # (More complex - simplified for now)

        return None

    def _consolidate_transformations(
        self, transformations: List[ObjectTransformation]
    ) -> List[ObjectTransformation]:
        """Consolidate and validate transformations across examples."""
        # Group by transformation type
        by_type = defaultdict(list)
        for trans in transformations:
            by_type[trans.type].append(trans)

        consolidated = []

        for trans_type, trans_list in by_type.items():
            if len(trans_list) == 0:
                continue

            # Find most common pattern
            if trans_type == "recolor":
                # Find consistent color mappings
                color_map = {}
                for trans in trans_list:
                    from_c = trans.params["from_color"]
                    to_c = trans.params["to_color"]
                    if from_c in color_map and color_map[from_c] != to_c:
                        continue  # Inconsistent
                    color_map[from_c] = to_c

                if color_map:
                    consolidated.append(
                        ObjectTransformation(
                            type="recolor",
                            params={"color_map": color_map},
                            confidence=0.9,
                        )
                    )

            elif trans_type == "move":
                # Find consistent movement pattern
                movements = [trans.params["delta"] for trans in trans_list]
                if len(set(movements)) == 1:
                    # All objects move the same
                    consolidated.append(
                        ObjectTransformation(
                            type="move",
                            params={"delta": movements[0], "apply_to": "all"},
                            confidence=0.9,
                        )
                    )

            elif trans_type == "delete":
                # Find deletion pattern
                conditions = []
                for trans in trans_list:
                    if trans.params["size"] < 5:
                        conditions.append("small")
                    if trans.params["color"] == 0:
                        conditions.append("black")

                if conditions:
                    consolidated.append(
                        ObjectTransformation(
                            type="delete",
                            params={
                                "condition": max(set(conditions), key=conditions.count)
                            },
                            confidence=0.7,
                        )
                    )

            else:
                # Add the most confident transformation
                best = max(trans_list, key=lambda t: t.confidence)
                consolidated.append(best)

        return consolidated

    def apply_transformations(
        self, test_input: np.ndarray, transformations: List[ObjectTransformation]
    ) -> np.ndarray:
        """Apply learned transformations to test input."""
        output = test_input.copy()
        objects = self.extract_objects(test_input)

        # Track which objects have been processed

        for trans in transformations:
            if trans.type == "recolor":
                output = self._apply_recolor(output, objects, trans.params)

            elif trans.type == "move":
                output = self._apply_move(output, objects, trans.params)

            elif trans.type == "delete":
                output = self._apply_delete(output, objects, trans.params)

            elif trans.type == "duplicate":
                output = self._apply_duplicate(output, objects, trans.params)

            elif trans.type == "rotate":
                output = self._apply_rotate(output, objects, trans.params)

        return output

    def _apply_recolor(
        self, grid: np.ndarray, objects: List[Object], params: Dict
    ) -> np.ndarray:
        """Apply recolor transformation."""
        output = grid.copy()
        color_map = params.get("color_map", {})

        for obj in objects:
            if obj.color in color_map:
                new_color = color_map[obj.color]
                for r, c in obj.pixels:
                    if 0 <= r < output.shape[0] and 0 <= c < output.shape[1]:
                        output[r, c] = new_color

        return output

    def _apply_move(
        self, grid: np.ndarray, objects: List[Object], params: Dict
    ) -> np.ndarray:
        """Apply movement transformation."""
        output = np.zeros_like(grid)
        delta = params.get("delta", (0, 0))

        for obj in objects:
            for r, c in obj.pixels:
                new_r = int(r + delta[0])
                new_c = int(c + delta[1])
                if 0 <= new_r < output.shape[0] and 0 <= new_c < output.shape[1]:
                    output[new_r, new_c] = obj.color

        return output

    def _apply_delete(
        self, grid: np.ndarray, objects: List[Object], params: Dict
    ) -> np.ndarray:
        """Apply deletion transformation."""
        output = grid.copy()
        condition = params.get("condition", "none")

        for obj in objects:
            should_delete = False

            if condition == "small" and obj.size < 5:
                should_delete = True
            elif condition == "black" and obj.color == 0:
                should_delete = True
            elif condition == "always":
                should_delete = True

            if should_delete:
                for r, c in obj.pixels:
                    if 0 <= r < output.shape[0] and 0 <= c < output.shape[1]:
                        output[r, c] = 0  # Set to background

        return output

    def _apply_duplicate(
        self, grid: np.ndarray, objects: List[Object], params: Dict
    ) -> np.ndarray:
        """Apply duplication transformation."""
        output = grid.copy()
        source_color = params.get("source_color")
        offset = params.get("offset", (0, 0))
        new_color = params.get("new_color")

        for obj in objects:
            if source_color is None or obj.color == source_color:
                for r, c in obj.pixels:
                    new_r = int(r + offset[0])
                    new_c = int(c + offset[1])
                    if 0 <= new_r < output.shape[0] and 0 <= new_c < output.shape[1]:
                        output[new_r, new_c] = new_color if new_color else obj.color

        return output

    def _apply_rotate(
        self, grid: np.ndarray, objects: List[Object], params: Dict
    ) -> np.ndarray:
        """Apply rotation transformation."""
        output = grid.copy()
        angle = params.get("angle", 90)

        for obj in objects:
            # Extract object region
            min_r, min_c = obj.pixels.min(axis=0)
            max_r, max_c = obj.pixels.max(axis=0)

            # Create object grid
            obj_h = max_r - min_r + 1
            obj_w = max_c - min_c + 1
            obj_grid = np.zeros((obj_h, obj_w), dtype=grid.dtype)

            for r, c in obj.pixels:
                obj_grid[r - min_r, c - min_c] = obj.color

            # Rotate
            rotations = angle // 90
            rotated = np.rot90(obj_grid, rotations)

            # Clear original position
            for r, c in obj.pixels:
                output[r, c] = 0

            # Place rotated object (centered at same position)
            center_r, center_c = obj.center
            new_h, new_w = rotated.shape
            start_r = int(center_r - new_h // 2)
            start_c = int(center_c - new_w // 2)

            for i in range(new_h):
                for j in range(new_w):
                    if rotated[i, j] != 0:
                        r = start_r + i
                        c = start_c + j
                        if 0 <= r < output.shape[0] and 0 <= c < output.shape[1]:
                            output[r, c] = rotated[i, j]

        return output

    def solve(
        self, examples: List[Tuple[np.ndarray, np.ndarray]], test_input: np.ndarray
    ) -> np.ndarray:
        """Main solving method - learn and apply transformations."""
        if self.verbose:
            print(f"Analyzing {len(examples)} training examples...")

        # Learn transformations from examples
        transformations = self.learn_transformations(examples)

        if self.verbose:
            print(f"Learned {len(transformations)} transformations:")
            for trans in transformations:
                print(f"  - {trans.type}: {trans.params}")

        # Apply transformations to test input
        output = self.apply_transformations(test_input, transformations)

        return output


if __name__ == "__main__":
    print("Smart Object Manipulator")
    print("=" * 60)

    # Test with a simple example
    manipulator = SmartObjectManipulator(verbose=True)

    # Create test data
    test_input = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0], [2, 2, 0, 0]])

    test_output = np.array(
        [[0, 3, 3, 0], [0, 3, 3, 0], [0, 0, 0, 0], [0, 0, 2, 2]]  # Moved right
    )

    examples = [(test_input, test_output)]

    print("\nTest input:")
    print(test_input)

    print("\nExpected output:")
    print(test_output)

    # Solve
    result = manipulator.solve(examples, test_input)

    print("\nPredicted output:")
    print(result)

    print("\nMatch:", np.array_equal(result, test_output))
