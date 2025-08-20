#!/usr/bin/env python3
"""ARC-Specific Test-Time Adaptation.

Different from our previous physics TTA - this adapts DISCRETE RULES not continuous parameters.
Key insight: ARC requires discovering the specific transformation logic for each unique task.

Learning from past TTA work:
- Avoid JAX/TF compatibility issues by using pure numpy for rule extraction
- Focus on discrete rule refinement, not continuous parameter adaptation
- Use explicit extraction advantages instead of fighting gradient computation
"""

from utils.imports import setup_project_paths

setup_project_paths()

import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from arc_grid_extractor import ARCGridExtractor, ARCRule, GridTransformation
from neural_perception import NeuralPerceptionModule


@dataclass
class AdaptationResult:
    """Result of test-time adaptation."""

    refined_rules: ARCRule
    confidence: float
    adaptation_steps: int
    discovered_patterns: List[str]


class ARCTestTimeAdapter:
    """Test-time adaptation specifically for ARC tasks.

    Key differences from our previous physics TTA:
    1. Adapts discrete transformation rules, not continuous parameters
    2. Uses combinatorial search, not gradient descent
    3. Leverages explicit extraction instead of neural adaptation

    This is more like program synthesis than parameter optimization!
    """

    def __init__(self):
        self.extractor = ARCGridExtractor()
        self.perception = NeuralPerceptionModule()

        # Rule refinement strategies
        self.refinement_strategies = [
            self._refine_by_augmentation,
            self._refine_by_composition,
            self._refine_by_hypothesis_testing,
            self._refine_by_pattern_search,
        ]

    def adapt(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        initial_rules: Optional[ARCRule] = None,
        max_steps: int = 10,
    ) -> AdaptationResult:
        """Adapt transformation rules to specific ARC task.

        Unlike our physics TTA which did gradient steps, this:
        1. Generates rule hypotheses
        2. Tests them on examples
        3. Refines based on errors
        4. Discovers task-specific patterns

        Args:
            examples: Task demonstration pairs
            initial_rules: Starting rules (if any)
            max_steps: Maximum adaptation iterations

        Returns:
            Refined rules with confidence
        """
        # Start with initial extraction if no rules provided
        if initial_rules is None:
            initial_rules = self.extractor.extract_rules(examples)

        best_rules = initial_rules
        best_score = self._evaluate_rules(best_rules, examples)
        discovered_patterns = []

        for step in range(max_steps):
            # Try each refinement strategy
            for strategy in self.refinement_strategies:
                refined_rules, patterns = strategy(best_rules, examples)
                score = self._evaluate_rules(refined_rules, examples)

                # Always collect discovered patterns (not just on improvement)
                if patterns:
                    discovered_patterns.extend(patterns)

                if score > best_score:
                    best_rules = refined_rules
                    best_score = score

                    # Early stopping if perfect
                    if best_score >= 0.99:
                        return AdaptationResult(
                            refined_rules=best_rules,
                            confidence=best_score,
                            adaptation_steps=step + 1,
                            discovered_patterns=list(
                                set(discovered_patterns)
                            ),  # Deduplicate
                        )

        return AdaptationResult(
            refined_rules=best_rules,
            confidence=best_score,
            adaptation_steps=max_steps,
            discovered_patterns=list(set(discovered_patterns)),  # Deduplicate
        )

    def _refine_by_augmentation(
        self, rules: ARCRule, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[ARCRule, List[str]]:
        """Refine rules by augmenting examples.

        Different from physics augmentation (noise, scaling) -
        ARC augmentation preserves discrete patterns.
        """
        augmented_examples = []
        patterns = []

        for input_grid, output_grid in examples:
            # Color permutation (preserves structure)
            color_perm = self._generate_color_permutation(input_grid)
            aug_input = self._apply_color_permutation(input_grid, color_perm)
            aug_output = self._apply_color_permutation(output_grid, color_perm)
            augmented_examples.append((aug_input, aug_output))

            # Rotation (if output is also rotated)
            for k in [1, 2, 3]:
                rot_input = np.rot90(input_grid, k)
                rot_output = np.rot90(output_grid, k)
                augmented_examples.append((rot_input, rot_output))

            patterns.append("augmentation_refinement")

        # Re-extract with more examples
        all_examples = examples + augmented_examples
        refined_rules = self.extractor.extract_rules(all_examples)

        return refined_rules, patterns

    def _refine_by_composition(
        self, rules: ARCRule, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[ARCRule, List[str]]:
        """Try composing existing transformations in different orders."""
        if len(rules.transformations) < 2:
            return rules, []

        best_order = rules.composition_order or list(range(len(rules.transformations)))
        best_score = self._evaluate_rules(rules, examples)
        patterns = []

        # Try different orderings
        for perm in itertools.permutations(range(len(rules.transformations))):
            test_rules = ARCRule(
                transformations=rules.transformations,
                object_detection=rules.object_detection,
                composition_order=list(perm),
            )

            score = self._evaluate_rules(test_rules, examples)
            if score > best_score:
                best_score = score
                best_order = list(perm)
                patterns.append(f"reordered_to_{perm}")

        refined_rules = ARCRule(
            transformations=rules.transformations,
            object_detection=rules.object_detection,
            composition_order=best_order,
        )

        return refined_rules, patterns

    def _refine_by_hypothesis_testing(
        self, rules: ARCRule, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[ARCRule, List[str]]:
        """Generate and test transformation hypotheses.

        This is the key difference from continuous TTA -
        we're testing discrete hypotheses, not optimizing parameters.
        """
        hypotheses = []
        patterns = []

        # Analyze what current rules get wrong
        errors = self._analyze_errors(rules, examples)

        # Generate hypotheses based on errors
        if errors["color_errors"] > 0:
            # Try different color mappings
            hypothesis = self._hypothesize_color_mapping(examples)
            if hypothesis:
                hypotheses.append(hypothesis)
                patterns.append("color_hypothesis")

        if errors["position_errors"] > 0:
            # Try different spatial transformations
            hypothesis = self._hypothesize_spatial_transform(examples)
            if hypothesis:
                hypotheses.append(hypothesis)
                patterns.append("spatial_hypothesis")

        if errors["pattern_errors"] > 0:
            # Try pattern-based rules
            hypothesis = self._hypothesize_pattern_rule(examples)
            if hypothesis:
                hypotheses.append(hypothesis)
                patterns.append("pattern_hypothesis")

        # Test hypotheses and keep best
        best_rules = rules
        best_score = self._evaluate_rules(rules, examples)

        for hypothesis in hypotheses:
            test_rules = self._merge_hypothesis(rules, hypothesis)
            score = self._evaluate_rules(test_rules, examples)
            if score > best_score:
                best_rules = test_rules
                best_score = score

        return best_rules, patterns

    def _refine_by_pattern_search(
        self, rules: ARCRule, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[ARCRule, List[str]]:
        """Search for task-specific patterns using perception module."""
        patterns_found = []

        # Use neural perception to find patterns
        all_patterns = []
        all_detected_types = set()
        for input_grid, output_grid in examples:
            input_patterns = self.perception.detect_spatial_patterns(input_grid)
            output_patterns = self.perception.detect_spatial_patterns(output_grid)
            all_patterns.append((input_patterns, output_patterns))

            # Collect all pattern types seen
            for p in input_patterns:
                all_detected_types.add(f"input_{p.pattern_type}")
            for p in output_patterns:
                all_detected_types.add(f"output_{p.pattern_type}")

        # Look for consistent pattern transformations
        consistent_transforms = self._find_consistent_patterns(all_patterns)

        if consistent_transforms:
            # Create new transformations based on patterns
            new_transformations = []
            for pattern_type, transform in consistent_transforms.items():
                new_trans = GridTransformation(
                    rule_type=f"pattern_{pattern_type}",
                    parameters=transform,
                    scope="global",
                )
                new_transformations.append(new_trans)
                patterns_found.append(f"discovered_{pattern_type}")

            # Merge with existing rules
            refined_rules = ARCRule(
                transformations=rules.transformations + new_transformations,
                object_detection=rules.object_detection,
                composition_order=None,
            )
        else:
            refined_rules = rules
            # Still report what patterns were seen even if no consistent transforms
            if all_detected_types:
                patterns_found.append(f"detected_{len(all_detected_types)}_patterns")

        return refined_rules, patterns_found

    def _evaluate_rules(
        self, rules: ARCRule, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> float:
        """Evaluate how well rules work on examples."""
        if not examples:
            return 0.0

        correct = 0
        total = 0

        for input_grid, expected_output in examples:
            try:
                predicted = self.extractor.apply_rules(input_grid, rules)
                if np.array_equal(predicted, expected_output):
                    correct += 1
            except Exception:
                pass  # Rule application failed
            total += 1

        return correct / total if total > 0 else 0.0

    def _analyze_errors(
        self, rules: ARCRule, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, int]:
        """Analyze what types of errors the rules make."""
        errors = {
            "color_errors": 0,
            "position_errors": 0,
            "pattern_errors": 0,
            "size_errors": 0,
        }

        for input_grid, expected_output in examples:
            try:
                predicted = self.extractor.apply_rules(input_grid, rules)

                # Analyze differences
                if predicted.shape != expected_output.shape:
                    errors["size_errors"] += 1
                else:
                    # Color differences
                    color_diff = np.sum(predicted != expected_output)
                    if color_diff > 0:
                        errors["color_errors"] += color_diff

                    # Position differences (simplified)
                    if not np.array_equal(predicted, expected_output):
                        errors["position_errors"] += 1

            except Exception:
                errors["pattern_errors"] += 1

        return errors

    def _generate_color_permutation(self, grid: np.ndarray) -> Dict[int, int]:
        """Generate a color permutation for augmentation."""
        unique_colors = np.unique(grid)
        permuted = np.random.permutation(unique_colors)
        return {int(c): int(p) for c, p in zip(unique_colors, permuted)}

    def _apply_color_permutation(
        self, grid: np.ndarray, perm: Dict[int, int]
    ) -> np.ndarray:
        """Apply color permutation to grid."""
        result = grid.copy()
        for old_color, new_color in perm.items():
            result[grid == old_color] = new_color
        return result

    def _hypothesize_color_mapping(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[GridTransformation]:
        """Generate color mapping hypothesis."""
        # Extract consistent color mappings across examples
        mappings = []
        for input_grid, output_grid in examples:
            if input_grid.shape == output_grid.shape:
                mapping = {}
                for color in np.unique(input_grid):
                    mask = input_grid == color
                    output_colors = output_grid[mask]
                    if len(np.unique(output_colors)) == 1:
                        mapping[int(color)] = int(output_colors[0])
                mappings.append(mapping)

        # Find consistent mapping
        if mappings and all(m == mappings[0] for m in mappings):
            return GridTransformation(
                rule_type="color_mapping",
                parameters={"mapping": mappings[0]},
                scope="global",
            )
        return None

    def _hypothesize_spatial_transform(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[GridTransformation]:
        """Generate spatial transformation hypothesis."""
        # Check for consistent translations
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue

                valid = True
                for input_grid, output_grid in examples:
                    if input_grid.shape != output_grid.shape:
                        valid = False
                        break

                    shifted = np.roll(input_grid, (dy, dx), axis=(0, 1))
                    if not np.array_equal(shifted, output_grid):
                        valid = False
                        break

                if valid:
                    return GridTransformation(
                        rule_type="translation",
                        parameters={"dx": dx, "dy": dy},
                        scope="global",
                    )
        return None

    def _hypothesize_pattern_rule(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[GridTransformation]:
        """Generate pattern-based rule hypothesis."""
        # Simplified: check for fill patterns
        for input_grid, output_grid in examples:
            if 0 in input_grid:
                zero_mask = input_grid == 0
                filled_values = output_grid[zero_mask]
                if len(np.unique(filled_values)) == 1:
                    fill_color = int(filled_values[0])
                    return GridTransformation(
                        rule_type="pattern_fill",
                        parameters={"pattern": "solid", "color": fill_color},
                        condition="where_zero",
                        scope="global",
                    )
        return None

    def _merge_hypothesis(
        self, base_rules: ARCRule, hypothesis: GridTransformation
    ) -> ARCRule:
        """Merge a hypothesis into existing rules."""
        # Replace or add transformation
        transformations = []
        replaced = False

        for trans in base_rules.transformations:
            if trans.rule_type == hypothesis.rule_type:
                transformations.append(hypothesis)
                replaced = True
            else:
                transformations.append(trans)

        if not replaced:
            transformations.append(hypothesis)

        return ARCRule(
            transformations=transformations,
            object_detection=base_rules.object_detection,
            composition_order=None,
        )

    def _find_consistent_patterns(
        self, pattern_pairs: List[Tuple[List, List]]
    ) -> Dict[str, Any]:
        """Find consistent pattern transformations across examples.

        Enhanced to detect concrete ARC patterns beyond just symmetry.
        """
        consistent = {}

        # Check if symmetry is consistently added/removed
        has_input_symmetry = []
        has_output_symmetry = []

        for input_patterns, output_patterns in pattern_pairs:
            has_input_symmetry.append(
                any(p.pattern_type.endswith("symmetry") for p in input_patterns)
            )
            has_output_symmetry.append(
                any(p.pattern_type.endswith("symmetry") for p in output_patterns)
            )

        # If consistently adding symmetry
        if all(not i and o for i, o in zip(has_input_symmetry, has_output_symmetry)):
            consistent["add_symmetry"] = {"operation": "create_symmetry"}

        # Check for progression patterns
        if self._has_progression_pattern(pattern_pairs):
            consistent["progression"] = {"operation": "apply_progression"}

        # Check for alternation patterns
        if self._has_alternation_pattern(pattern_pairs):
            consistent["alternation"] = {"operation": "apply_alternation"}

        # Check for periodicity patterns
        if self._has_periodicity_pattern(pattern_pairs):
            consistent["periodicity"] = {"operation": "apply_period"}

        # Check for repetition patterns
        if self._has_repetition_pattern(pattern_pairs):
            consistent["repetition"] = {"operation": "apply_repetition"}

        return consistent

    def _has_progression_pattern(self, pattern_pairs: List[Tuple[List, List]]) -> bool:
        """Check if outputs show arithmetic or geometric progression."""
        for input_patterns, output_patterns in pattern_pairs:
            # Check for increasing counts or sizes
            for out_p in output_patterns:
                if (
                    hasattr(out_p, "pattern_type")
                    and "progression" in out_p.pattern_type
                ):
                    return True
        return False

    def _has_alternation_pattern(self, pattern_pairs: List[Tuple[List, List]]) -> bool:
        """Check if outputs show alternating patterns (A-B-A-B)."""
        for input_patterns, output_patterns in pattern_pairs:
            for out_p in output_patterns:
                if (
                    hasattr(out_p, "pattern_type")
                    and "alternation" in out_p.pattern_type
                ):
                    return True
        return False

    def _has_periodicity_pattern(self, pattern_pairs: List[Tuple[List, List]]) -> bool:
        """Check if outputs show periodic/cyclic patterns."""
        for input_patterns, output_patterns in pattern_pairs:
            for out_p in output_patterns:
                if hasattr(out_p, "pattern_type") and (
                    "periodic" in out_p.pattern_type or "cyclic" in out_p.pattern_type
                ):
                    return True
        return False

    def _has_repetition_pattern(self, pattern_pairs: List[Tuple[List, List]]) -> bool:
        """Check if outputs show repetition of input elements."""
        for input_patterns, output_patterns in pattern_pairs:
            for out_p in output_patterns:
                if (
                    hasattr(out_p, "pattern_type")
                    and "repetition" in out_p.pattern_type
                ):
                    return True
        return False


def test_arc_tta():
    """Test ARC-specific test-time adaptation."""
    adapter = ARCTestTimeAdapter()

    print("=" * 70)
    print("ARC TEST-TIME ADAPTATION TEST")
    print("=" * 70)
    print("\nKey difference from physics TTA:")
    print("- Physics: Adapt continuous parameters (gravity value)")
    print("- ARC: Adapt discrete rules (transformation type)")
    print("-" * 70)

    # Test case: Color mapping that needs refinement
    print("\nTest 1: Refining Color Mapping")
    examples = [
        (
            np.array([[1, 2, 0], [2, 1, 0], [0, 0, 3]]),
            np.array([[2, 1, 0], [1, 2, 0], [0, 0, 3]]),  # Swap 1 and 2, keep rest
        ),
        (
            np.array([[1, 1, 3], [2, 2, 3], [3, 3, 3]]),
            np.array([[2, 2, 3], [1, 1, 3], [3, 3, 3]]),  # Same rule
        ),
    ]

    # Initial extraction
    initial_rules = adapter.extractor.extract_rules(examples)
    initial_score = adapter._evaluate_rules(initial_rules, examples)
    print(f"Initial extraction score: {initial_score:.2f}")

    # Adapt
    result = adapter.adapt(examples, initial_rules, max_steps=5)
    print(f"After adaptation score: {result.confidence:.2f}")
    print(f"Adaptation steps: {result.adaptation_steps}")
    print(f"Discovered patterns: {result.discovered_patterns}")

    # Test case 2: Pattern that needs hypothesis testing
    print("\n" + "-" * 70)
    print("Test 2: Discovering Hidden Pattern")
    examples = [
        (
            np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]]),
            np.array([[1, 2, 1], [2, 2, 2], [1, 2, 1]]),  # Fill 0s with 2
        )
    ]

    result = adapter.adapt(examples, max_steps=5)
    print(f"Confidence after adaptation: {result.confidence:.2f}")
    print(f"Discovered: {result.discovered_patterns}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("ARC TTA is about RULE DISCOVERY, not parameter optimization.")
    print("This is why explicit extraction + adaptation is so powerful!")
    print("=" * 70)


if __name__ == "__main__":
    test_arc_tta()
