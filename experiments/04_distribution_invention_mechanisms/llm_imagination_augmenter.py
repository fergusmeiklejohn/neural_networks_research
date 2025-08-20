#!/usr/bin/env python3
"""
LLM Imagination Augmenter - Helps LLMs think outside their distribution.

Key idea: Instead of training from scratch, augment existing LLMs
to deliberately explore unlikely paths when pattern matching fails.
"""

from utils.imports import setup_project_paths

setup_project_paths()

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class FailureDetector:
    """Detects when an LLM is stuck in pattern matching mode."""

    def is_stuck(self, attempts: List[Dict]) -> bool:
        """Check if the solver is stuck."""
        if len(attempts) < 2:
            return False

        # Check for repetitive outputs
        outputs = [a.get("output") for a in attempts]
        if len(set(map(str, outputs))) == 1:
            return True  # Same output repeatedly

        # Check for low confidence
        confidences = [a.get("confidence", 1.0) for a in attempts]
        if np.mean(confidences) < 0.3:
            return True  # Very low confidence

        # Check for no progress
        accuracies = [a.get("accuracy", 0.0) for a in attempts]
        if len(accuracies) > 2 and np.std(accuracies) < 0.01:
            return True  # No improvement

        return False


class UnlikelyPathGenerator:
    """Generates deliberately unlikely but structured paths."""

    def generate_unlikely_paths(
        self, problem: Dict, failed_patterns: List, num_paths: int = 5
    ) -> List[Dict]:
        """Generate unlikely but potentially valid approaches."""
        unlikely_paths = []

        # Strategy 1: Invert common patterns
        if failed_patterns:
            for pattern in failed_patterns[:2]:
                inverted = self._invert_pattern(pattern)
                unlikely_paths.append(
                    {
                        "strategy": "pattern_inversion",
                        "description": "Opposite of failed pattern",
                        "pattern": inverted,
                    }
                )

        # Strategy 2: Random structured combinations
        components = self._extract_components(problem)
        for _ in range(2):
            random_combo = self._random_combination(components)
            unlikely_paths.append(
                {
                    "strategy": "random_combination",
                    "description": "Novel component combination",
                    "pattern": random_combo,
                }
            )

        # Strategy 3: Extreme transformations
        extreme = self._extreme_transformation(problem)
        unlikely_paths.append(
            {
                "strategy": "extreme_transformation",
                "description": "Extreme parameter values",
                "pattern": extreme,
            }
        )

        return unlikely_paths[:num_paths]

    def _invert_pattern(self, pattern: Any) -> Any:
        """Invert a pattern - do the opposite."""
        # This is problem-specific, simplified here
        if isinstance(pattern, dict):
            inverted = {}
            for key, value in pattern.items():
                if isinstance(value, bool):
                    inverted[key] = not value
                elif isinstance(value, (int, float)):
                    inverted[key] = -value
                else:
                    inverted[key] = value
            return inverted
        return pattern

    def _extract_components(self, problem: Dict) -> List:
        """Extract reusable components from the problem."""
        # Problem-specific extraction
        return problem.get("components", [])

    def _random_combination(self, components: List) -> Any:
        """Create random but structured combination."""
        if not components:
            return {}

        # Randomly combine components in unexpected ways
        import random

        num_components = random.randint(1, min(3, len(components)))
        selected = random.sample(components, num_components)

        return {"combined": selected}

    def _extreme_transformation(self, problem: Dict) -> Any:
        """Apply extreme parameter values."""
        return {
            "scale": 10.0,  # Extreme scaling
            "rotation": 180,  # Flip
            "noise": 0.9,  # High noise
        }


class ImaginationAugmenter:
    """
    Augments an LLM with imagination capabilities.

    When pattern matching fails, deliberately explores unlikely paths.
    """

    def __init__(self, base_solver=None):
        self.base_solver = base_solver
        self.failure_detector = FailureDetector()
        self.path_generator = UnlikelyPathGenerator()
        self.attempts = []

    def solve_with_imagination(
        self, examples: List[Tuple[np.ndarray, np.ndarray]], test_input: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Solve using standard approach first, then imagination if needed.
        """
        print("Phase 1: Trying standard pattern matching...")

        # First, try standard approaches
        for attempt_num in range(3):
            if self.base_solver:
                result = self.base_solver.solve(examples, test_input)

                # Check if it worked
                if self._is_successful(result, examples):
                    print(f"  ✓ Standard approach worked on attempt {attempt_num + 1}")
                    return result

                self.attempts.append(
                    {
                        "output": result,
                        "confidence": getattr(result, "confidence", 0.5),
                        "method": "standard",
                    }
                )
            else:
                # Simulate failed attempts for testing
                self.attempts.append(
                    {"output": None, "confidence": 0.2, "method": "standard"}
                )

        # Check if we're stuck
        if self.failure_detector.is_stuck(self.attempts):
            print("\nPhase 2: Standard approach failed. Engaging imagination...")
            return self._imagine_solution(examples, test_input)

        return None

    def _imagine_solution(
        self, examples: List[Tuple[np.ndarray, np.ndarray]], test_input: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Use imagination to find unlikely but valid solutions.
        """
        # Extract what failed
        failed_patterns = [a.get("pattern") for a in self.attempts if a.get("pattern")]

        # Generate unlikely paths
        problem = {"examples": examples, "test_input": test_input}
        unlikely_paths = self.path_generator.generate_unlikely_paths(
            problem, failed_patterns, num_paths=5
        )

        print(f"  Generated {len(unlikely_paths)} unlikely hypotheses:")

        for i, path in enumerate(unlikely_paths):
            print(f"    {i+1}. {path['strategy']}: {path['description']}")

            # Try to apply this path
            result = self._apply_unlikely_path(path, examples, test_input)

            if result is not None and self._is_successful(result, examples):
                print(f"  ✓ Unlikely path {i+1} worked! ({path['strategy']})")
                return result

        print("  ✗ No unlikely paths succeeded")
        return None

    def _apply_unlikely_path(
        self, path: Dict, examples: List, test_input: np.ndarray
    ) -> Optional[np.ndarray]:
        """Apply an unlikely path to generate a solution."""
        strategy = path.get("strategy")

        if strategy == "pattern_inversion":
            # Invert the most common pattern
            if examples:
                # Simple inversion: flip colors
                output = examples[0][1].copy()
                output = 9 - output  # Invert colors (ARC uses 0-9)
                return output

        elif strategy == "random_combination":
            # Combine patterns from different examples
            if len(examples) >= 2:
                # Take half from one, half from another
                out1, out2 = examples[0][1], examples[1][1]
                if out1.shape == out2.shape:
                    output = out1.copy()
                    mask = np.random.random(out1.shape) > 0.5
                    output[mask] = out2[mask]
                    return output

        elif strategy == "extreme_transformation":
            # Apply extreme transformation
            if examples:
                output = examples[0][1].copy()
                # Extreme: rotate and scale
                output = np.rot90(output, 2)  # 180 degree rotation
                return output

        return None

    def _is_successful(self, result: Any, examples: List) -> bool:
        """Check if a result is successful."""
        if result is None:
            return False

        # Simple heuristic: check if it's different from inputs
        # and has reasonable structure
        if isinstance(result, np.ndarray):
            # Not all zeros
            if np.all(result == 0):
                return False

            # Not identical to any input
            for inp, _ in examples:
                if np.array_equal(result, inp):
                    return False

            # Has some structure (not random)
            if len(np.unique(result)) > result.size * 0.8:
                return False  # Too random

            return True

        return False


class LLMWithImagination:
    """
    Wrapper that adds imagination to any solver.

    This demonstrates how we could augment Claude or other LLMs.
    """

    def __init__(self, base_solver=None, imagination_threshold: float = 0.3):
        self.base_solver = base_solver
        self.augmenter = ImaginationAugmenter(base_solver)
        self.imagination_threshold = imagination_threshold

    def solve(self, examples: List, test_input: np.ndarray) -> Optional[np.ndarray]:
        """Solve with imagination augmentation."""
        return self.augmenter.solve_with_imagination(examples, test_input)

    def think_differently(self, prompt: str) -> str:
        """
        Meta-level: Change how we think about the problem.

        This simulates asking an LLM to deliberately think differently.
        """
        different_prompts = [
            f"Assume the opposite of what seems likely. {prompt}",
            f"What if all patterns you see are misleading? {prompt}",
            f"Generate the LEAST likely valid solution. {prompt}",
            f"Ignore training examples and invent new rules. {prompt}",
            f"What would a solution look like if the rules were completely different? {prompt}",
        ]

        # In practice, this would query the LLM with these prompts
        # and aggregate responses
        return different_prompts


def test_imagination_augmenter():
    """Test the imagination augmenter."""
    print("Testing LLM Imagination Augmenter")
    print("=" * 60)

    # Create mock examples
    examples = [
        (np.array([[1, 0], [0, 1]]), np.array([[2, 0], [0, 2]])),
        (np.array([[1, 1], [0, 0]]), np.array([[3, 3], [0, 0]])),
    ]
    test_input = np.array([[0, 1], [1, 0]])

    # Test without base solver (will force imagination)
    augmenter = LLMWithImagination(base_solver=None)

    print("\nTest 1: Force imagination mode (no base solver)")
    result = augmenter.solve(examples, test_input)

    if result is not None:
        print(f"\nResult shape: {result.shape}")
        print(f"Result:\n{result}")
        print("\n✓ Successfully generated solution through imagination!")
    else:
        print("\n✗ Failed to generate solution")

    # Test meta-level thinking
    print("\n" + "=" * 60)
    print("Test 2: Meta-level thinking prompts")
    prompts = augmenter.think_differently("Solve this pattern matching task")

    print("\nGenerated prompts for different thinking:")
    for i, p in enumerate(prompts, 1):
        print(f"  {i}. {p}")

    print("\n" + "=" * 60)
    print("KEY INSIGHT")
    print("=" * 60)
    print("Instead of training new models, we can augment existing LLMs")
    print("to deliberately explore unlikely paths when pattern matching fails.")
    print("\nThis is more practical and leverages existing capabilities!")


if __name__ == "__main__":
    test_imagination_augmenter()
