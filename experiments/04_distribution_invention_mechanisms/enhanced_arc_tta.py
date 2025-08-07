#!/usr/bin/env python3
"""Enhanced ARC Test-Time Adapter with advanced hypothesis generation.

This module uses the enhanced perception capabilities to generate and test
more sophisticated transformation hypotheses.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import itertools
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from arc_grid_extractor import ARCRule
from enhanced_neural_perception import EnhancedNeuralPerception
from object_manipulation import ObjectManipulator


@dataclass
class TransformationHypothesis:
    """Represents a transformation hypothesis."""

    name: str
    transform_fn: Callable
    confidence: float
    evidence: List[str]  # What evidence supports this hypothesis


@dataclass
class EnhancedAdaptationResult:
    """Result of enhanced test-time adaptation."""

    best_hypothesis: TransformationHypothesis
    all_hypotheses: List[TransformationHypothesis]
    adapted_rules: ARCRule
    confidence: float
    discovered_patterns: List[str]
    adaptation_steps: int


class EnhancedARCTestTimeAdapter:
    """Enhanced test-time adapter with sophisticated hypothesis generation."""

    def __init__(self):
        self.perception = EnhancedNeuralPerception()
        self.object_manipulator = ObjectManipulator()
        self.hypotheses = []

    def adapt(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        initial_rules: Optional[ARCRule] = None,
        max_steps: int = 10,
    ) -> EnhancedAdaptationResult:
        """Adapt rules using enhanced perception and hypothesis testing."""

        # Analyze all training examples
        analyses = []
        for input_grid, output_grid in train_examples:
            input_analysis = self.perception.analyze_grid(input_grid)
            output_analysis = self.perception.analyze_grid(output_grid)
            comparison = self.perception.compare_grids(input_grid, output_grid)

            analyses.append(
                {
                    "input": input_analysis,
                    "output": output_analysis,
                    "comparison": comparison,
                }
            )

        # Generate hypotheses based on analyses
        self.hypotheses = self._generate_hypotheses(analyses)

        # Test and rank hypotheses
        ranked_hypotheses = self._test_hypotheses(self.hypotheses, train_examples)

        # Select best hypothesis
        best_hypothesis = ranked_hypotheses[0] if ranked_hypotheses else None

        # Create adapted rules
        adapted_rules = self._create_adapted_rules(best_hypothesis, initial_rules)

        # Collect discovered patterns
        discovered_patterns = self._collect_discovered_patterns(
            analyses, best_hypothesis
        )

        return EnhancedAdaptationResult(
            best_hypothesis=best_hypothesis,
            all_hypotheses=ranked_hypotheses,
            adapted_rules=adapted_rules,
            confidence=best_hypothesis.confidence if best_hypothesis else 0.0,
            discovered_patterns=discovered_patterns,
            adaptation_steps=len(ranked_hypotheses),
        )

    def _generate_hypotheses(
        self, analyses: List[Dict]
    ) -> List[TransformationHypothesis]:
        """Generate transformation hypotheses based on analyses."""
        hypotheses = []

        # Check for consistent transformations across examples
        first_comparison = analyses[0]["comparison"]

        # 1. Simple geometric transformations
        if first_comparison.get("likely_transformation"):
            transform_type = first_comparison["likely_transformation"]

            if transform_type == "horizontal_flip":
                hypotheses.append(
                    TransformationHypothesis(
                        name="horizontal_flip",
                        transform_fn=lambda g: np.flip(g, axis=1),
                        confidence=0.8,
                        evidence=["Grid comparison shows horizontal flip"],
                    )
                )
            elif transform_type == "vertical_flip":
                hypotheses.append(
                    TransformationHypothesis(
                        name="vertical_flip",
                        transform_fn=lambda g: np.flip(g, axis=0),
                        confidence=0.8,
                        evidence=["Grid comparison shows vertical flip"],
                    )
                )
            elif transform_type == "2x_scaling":
                hypotheses.append(
                    TransformationHypothesis(
                        name="2x_scaling",
                        transform_fn=self._scale_2x,
                        confidence=0.8,
                        evidence=["Output is 2x the size of input"],
                    )
                )
            elif transform_type == "rotation_90":
                hypotheses.append(
                    TransformationHypothesis(
                        name="rotation_90",
                        transform_fn=lambda g: np.rot90(g, -1),
                        confidence=0.8,
                        evidence=["Grid comparison shows 90-degree rotation"],
                    )
                )

        # 2. Object-based transformations
        hypotheses.extend(self._generate_object_hypotheses(analyses))

        # 3. Pattern-based transformations
        hypotheses.extend(self._generate_pattern_hypotheses(analyses))

        # 4. Color-based transformations
        hypotheses.extend(self._generate_color_hypotheses(analyses))

        # 5. Compositional transformations
        hypotheses.extend(self._generate_compositional_hypotheses(analyses))

        return hypotheses

    def _generate_object_hypotheses(
        self, analyses: List[Dict]
    ) -> List[TransformationHypothesis]:
        """Generate hypotheses based on object analysis."""
        hypotheses = []

        # Check if object count changes
        for analysis in analyses:
            input_count = analysis["input"]["counting"]["total_objects"]
            output_count = analysis["output"]["counting"]["total_objects"]

            if output_count > input_count:
                # Objects are being duplicated or added
                hypotheses.append(
                    TransformationHypothesis(
                        name="object_duplication",
                        transform_fn=self._duplicate_objects,
                        confidence=0.5,
                        evidence=[
                            f"Object count increases: {input_count} -> {output_count}"
                        ],
                    )
                )
            elif output_count < input_count:
                # Objects are being removed or merged
                hypotheses.append(
                    TransformationHypothesis(
                        name="object_removal",
                        transform_fn=self._remove_small_objects,
                        confidence=0.5,
                        evidence=[
                            f"Object count decreases: {input_count} -> {output_count}"
                        ],
                    )
                )

        # Check for object rearrangement
        if self._check_object_rearrangement(analyses):
            hypotheses.append(
                TransformationHypothesis(
                    name="object_rearrangement",
                    transform_fn=self._rearrange_objects,
                    confidence=0.6,
                    evidence=["Objects appear to be rearranged"],
                )
            )

        return hypotheses

    def _generate_pattern_hypotheses(
        self, analyses: List[Dict]
    ) -> List[TransformationHypothesis]:
        """Generate hypotheses based on pattern analysis."""
        hypotheses = []

        # Check for pattern completion
        for analysis in analyses:
            input_patterns = analysis["input"]["patterns"]
            output_patterns = analysis["output"]["patterns"]

            if len(output_patterns) > len(input_patterns):
                hypotheses.append(
                    TransformationHypothesis(
                        name="pattern_completion",
                        transform_fn=self._complete_pattern,
                        confidence=0.4,
                        evidence=["Output has more patterns than input"],
                    )
                )

        # Check for symmetry enforcement
        for analysis in analyses:
            input_sym = analysis["input"]["symmetry"]
            output_sym = analysis["output"]["symmetry"]

            if any(output_sym.values()) and not any(input_sym.values()):
                hypotheses.append(
                    TransformationHypothesis(
                        name="symmetry_enforcement",
                        transform_fn=self._enforce_symmetry,
                        confidence=0.5,
                        evidence=["Output has symmetry that input lacks"],
                    )
                )

        return hypotheses

    def _generate_color_hypotheses(
        self, analyses: List[Dict]
    ) -> List[TransformationHypothesis]:
        """Generate hypotheses based on color analysis."""
        hypotheses = []

        # Check for color mapping
        first_comparison = analyses[0]["comparison"]
        if first_comparison.get("color_changes"):
            color_map = first_comparison["color_changes"]

            hypotheses.append(
                TransformationHypothesis(
                    name="color_mapping",
                    transform_fn=lambda g: self._apply_color_map(g, color_map),
                    confidence=0.7,
                    evidence=[f"Color mapping detected: {color_map}"],
                )
            )

        # Check for color filling
        for analysis in analyses:
            input_colors = set(analysis["input"]["counting"]["by_color"].keys())
            output_colors = set(analysis["output"]["counting"]["by_color"].keys())

            new_colors = output_colors - input_colors
            if new_colors:
                hypotheses.append(
                    TransformationHypothesis(
                        name="color_filling",
                        transform_fn=self._fill_with_color,
                        confidence=0.4,
                        evidence=[f"New colors appear: {new_colors}"],
                    )
                )

        return hypotheses

    def _generate_compositional_hypotheses(
        self, analyses: List[Dict]
    ) -> List[TransformationHypothesis]:
        """Generate compositional transformation hypotheses."""
        hypotheses = []

        # Try combining simple transformations
        simple_transforms = []

        first_comparison = analyses[0]["comparison"]
        if first_comparison.get("size_change"):
            ratio = first_comparison["size_change"]["ratio"]
            if ratio == (2, 2):
                simple_transforms.append(("scale", self._scale_2x))

        # Check for flip + other transform
        for analysis in analyses:
            if analysis["output"]["symmetry"]["horizontal"]:
                simple_transforms.append(("flip", lambda g: np.flip(g, axis=1)))

        # Generate compositions
        if len(simple_transforms) >= 2:
            for t1, t2 in itertools.combinations(simple_transforms, 2):
                hypotheses.append(
                    TransformationHypothesis(
                        name=f"composite_{t1[0]}_{t2[0]}",
                        transform_fn=lambda g: t2[1](t1[1](g)),
                        confidence=0.3,
                        evidence=[f"Composition of {t1[0]} and {t2[0]}"],
                    )
                )

        return hypotheses

    def _test_hypotheses(
        self,
        hypotheses: List[TransformationHypothesis],
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> List[TransformationHypothesis]:
        """Test and rank hypotheses on training examples."""

        for hypothesis in hypotheses:
            correct_count = 0
            total_count = len(train_examples)

            for input_grid, expected_output in train_examples:
                try:
                    predicted_output = hypothesis.transform_fn(input_grid)

                    # Check if prediction matches expected
                    if np.array_equal(predicted_output, expected_output):
                        correct_count += 1
                except Exception:
                    # Transform failed
                    pass

            # Update confidence based on accuracy
            accuracy = correct_count / total_count if total_count > 0 else 0
            hypothesis.confidence *= accuracy

        # Sort by confidence
        return sorted(hypotheses, key=lambda h: h.confidence, reverse=True)

    def _create_adapted_rules(
        self,
        best_hypothesis: Optional[TransformationHypothesis],
        initial_rules: Optional[ARCRule],
    ) -> ARCRule:
        """Create adapted rules based on best hypothesis."""

        if not initial_rules:
            initial_rules = ARCRule(transformations=[])

        adapted_rules = ARCRule(
            transformations=initial_rules.transformations.copy(),
            object_detection=initial_rules.object_detection,
            composition_order=initial_rules.composition_order.copy()
            if initial_rules.composition_order
            else None,
        )

        if best_hypothesis:
            # Add the transformation from best hypothesis
            from arc_grid_extractor import GridTransformation

            adapted_rules.transformations.append(
                GridTransformation(
                    rule_type=best_hypothesis.name,
                    parameters={"confidence": best_hypothesis.confidence},
                )
            )

        return adapted_rules

    def _collect_discovered_patterns(
        self, analyses: List[Dict], best_hypothesis: Optional[TransformationHypothesis]
    ) -> List[str]:
        """Collect all discovered patterns."""
        patterns = []

        # Collect patterns from analyses
        for analysis in analyses:
            for pattern in analysis["input"]["patterns"]:
                patterns.append(f"input_{pattern.pattern_type}")
            for pattern in analysis["output"]["patterns"]:
                patterns.append(f"output_{pattern.pattern_type}")

        # Add hypothesis name if found
        if best_hypothesis:
            patterns.append(f"transformation_{best_hypothesis.name}")

        # Add object-based patterns
        for analysis in analyses:
            if analysis["input"]["topology"]["holes"] > 0:
                patterns.append("has_holes")
            if analysis["input"]["spatial"]:
                patterns.append("spatial_relationships")

        return list(set(patterns))  # Remove duplicates

    # Transformation helper functions

    def _scale_2x(self, grid: np.ndarray) -> np.ndarray:
        """Scale grid by 2x."""
        scaled = np.zeros((grid.shape[0] * 2, grid.shape[1] * 2), dtype=grid.dtype)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                scaled[i * 2 : i * 2 + 2, j * 2 : j * 2 + 2] = grid[i, j]
        return scaled

    def _duplicate_objects(self, grid: np.ndarray) -> np.ndarray:
        """Duplicate objects in the grid."""
        return self.object_manipulator.duplicate_objects(grid, "double")

    def _remove_small_objects(self, grid: np.ndarray) -> np.ndarray:
        """Remove small objects from the grid."""
        return self.object_manipulator.remove_small_objects(grid, threshold=3)

    def _rearrange_objects(self, grid: np.ndarray) -> np.ndarray:
        """Rearrange objects in the grid."""
        return self.object_manipulator.rearrange_objects(grid, "sort_by_size")

    def _complete_pattern(self, grid: np.ndarray) -> np.ndarray:
        """Complete pattern in the grid."""
        return self.object_manipulator.complete_pattern(grid)

    def _enforce_symmetry(self, grid: np.ndarray) -> np.ndarray:
        """Enforce symmetry by mirroring half of the grid."""
        result = grid.copy()
        mid = grid.shape[1] // 2

        # Mirror left half to right
        for i in range(grid.shape[0]):
            for j in range(mid, grid.shape[1]):
                mirror_j = grid.shape[1] - 1 - j
                if mirror_j >= 0:
                    result[i, j] = grid[i, mirror_j]

        return result

    def _apply_color_map(
        self, grid: np.ndarray, color_map: Dict[int, int]
    ) -> np.ndarray:
        """Apply color mapping."""
        result = grid.copy()
        for old_color, new_color in color_map.items():
            result[grid == old_color] = new_color
        return result

    def _fill_with_color(self, grid: np.ndarray) -> np.ndarray:
        """Fill empty spaces with the most common color."""
        # Find most common non-zero color
        unique, counts = np.unique(grid[grid != 0], return_counts=True)
        if len(unique) > 0:
            most_common_color = unique[np.argmax(counts)]
            return self.object_manipulator.fill_with_color(
                grid, most_common_color, "background"
            )
        return grid

    def _check_object_rearrangement(self, analyses: List[Dict]) -> bool:
        """Check if objects are being rearranged."""
        for analysis in analyses:
            input_objects = analysis["input"]["objects"]
            output_objects = analysis["output"]["objects"]

            # Simple check: same number but different positions
            if len(input_objects) == len(output_objects) and len(input_objects) > 0:
                input_centers = [obj.center for obj in input_objects]
                output_centers = [obj.center for obj in output_objects]

                # Check if centers have moved
                if input_centers != output_centers:
                    return True

        return False


def test_enhanced_tta():
    """Test the enhanced TTA adapter."""
    print("Testing Enhanced TTA Adapter")
    print("=" * 50)

    # Create test examples (horizontal flip)
    input1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    output1 = np.array([[3, 2, 1], [6, 5, 4], [9, 8, 7]])

    input2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    output2 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    train_examples = [(input1, output1), (input2, output2)]

    # Test adaptation
    adapter = EnhancedARCTestTimeAdapter()
    result = adapter.adapt(train_examples)

    print(f"Best hypothesis: {result.best_hypothesis.name}")
    print(f"Confidence: {result.best_hypothesis.confidence:.2f}")
    print(f"Evidence: {result.best_hypothesis.evidence}")
    print(f"\nDiscovered patterns: {result.discovered_patterns}")
    print(f"Total hypotheses tested: {result.adaptation_steps}")

    # Test the transformation
    test_input = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    predicted = result.best_hypothesis.transform_fn(test_input)

    print(f"\nTest input:\n{test_input}")
    print(f"Predicted output:\n{predicted}")


if __name__ == "__main__":
    test_enhanced_tta()
