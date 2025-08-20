"""Imagination Benchmark Suite for Testing True Distribution Invention.

This benchmark tests whether systems can truly imagine patterns outside their
training distribution, not just interpolate within it. It includes:

1. Pattern Discovery - Find operations not in training
2. Rule Combination - Compose rules in novel ways
3. Cross-Domain Analogy - Transfer principles across domains
4. Counterfactual Reasoning - Imagine impossible scenarios
5. Creative Problem Solving - Find genuinely novel solutions
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cosine


@dataclass
class ImaginationTask:
    """Represents a task that requires imagination to solve."""

    task_id: str
    category: str  # 'pattern_discovery', 'rule_combination', etc.
    train_examples: List[Tuple[np.ndarray, np.ndarray]]
    test_examples: List[Tuple[np.ndarray, np.ndarray]]
    required_insight: str  # What needs to be discovered
    valid_solutions: List[str]  # Multiple solutions may be valid
    difficulty: int  # 1-5 scale

    def evaluate_solution(self, predicted: np.ndarray, expected: np.ndarray) -> float:
        """Evaluate how well a solution matches expected output."""
        if predicted.shape != expected.shape:
            return 0.0

        # Exact match
        if np.array_equal(predicted, expected):
            return 1.0

        # Partial credit for close solutions
        correct_elements = np.sum(predicted == expected)
        total_elements = predicted.size
        return correct_elements / total_elements


class PatternDiscoveryTasks:
    """Tasks that require discovering new patterns not in training."""

    @staticmethod
    def create_shear_task() -> ImaginationTask:
        """Train on rotate/flip/scale, test requires shear transformation."""
        train_examples = []

        # Training: rotation
        for _ in range(2):
            inp = np.array([[1, 2, 0], [3, 4, 0], [0, 0, 0]])
            out = np.rot90(inp)
            train_examples.append((inp, out))

        # Training: flip
        for _ in range(2):
            inp = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            out = np.flipud(inp)
            train_examples.append((inp, out))

        # Training: scale
        for _ in range(2):
            inp = np.array([[1, 2], [3, 4]])
            out = np.repeat(np.repeat(inp, 2, axis=0), 2, axis=1)
            train_examples.append((inp, out))

        # Test: shear transformation (NOT in training)
        test_examples = []
        for i in range(3):
            inp = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            # Simple shear: shift each row by its index
            out = np.zeros_like(inp)
            for row in range(inp.shape[0]):
                for col in range(inp.shape[1]):
                    new_col = (col + row) % inp.shape[1]
                    out[row, new_col] = inp[row, col]
            test_examples.append((inp, out))

        return ImaginationTask(
            task_id="pattern_shear",
            category="pattern_discovery",
            train_examples=train_examples,
            test_examples=test_examples,
            required_insight="Shear transformation (diagonal shift)",
            valid_solutions=["shear", "diagonal_shift", "row_dependent_translate"],
            difficulty=3,
        )

    @staticmethod
    def create_spiral_task() -> ImaginationTask:
        """Train on linear patterns, test requires spiral pattern."""
        train_examples = []

        # Training: horizontal lines
        for i in range(3):
            inp = np.zeros((5, 5))
            inp[i, :] = 1
            out = np.roll(inp, 1, axis=0)  # Shift down
            train_examples.append((inp, out))

        # Training: vertical lines
        for i in range(3):
            inp = np.zeros((5, 5))
            inp[:, i] = 1
            out = np.roll(inp, 1, axis=1)  # Shift right
            train_examples.append((inp, out))

        # Test: spiral pattern (NOT in training)
        test_examples = []
        for _ in range(2):
            inp = np.zeros((5, 5))
            inp[2, 2] = 1  # Center point

            # Create spiral outward
            out = np.zeros((5, 5))
            spiral_coords = [
                (2, 2),
                (2, 3),
                (3, 3),
                (3, 2),
                (3, 1),
                (2, 1),
                (1, 1),
                (1, 2),
                (1, 3),
                (1, 4),
            ]
            for i, (r, c) in enumerate(spiral_coords[:5]):
                out[r, c] = 1

            test_examples.append((inp, out))

        return ImaginationTask(
            task_id="pattern_spiral",
            category="pattern_discovery",
            train_examples=train_examples,
            test_examples=test_examples,
            required_insight="Spiral pattern generation",
            valid_solutions=["spiral", "circular_expansion", "radial_growth"],
            difficulty=4,
        )


class RuleCombinationTasks:
    """Tasks requiring novel combination of separately learned rules."""

    @staticmethod
    def create_color_size_combo() -> ImaginationTask:
        """Train on color and size changes separately, test needs both."""
        train_examples = []

        # Training: color change only
        for _ in range(3):
            inp = np.array([[1, 1], [1, 1]])
            out = np.array([[2, 2], [2, 2]])  # Red to blue
            train_examples.append((inp, out))

        # Training: size change only
        for _ in range(3):
            inp = np.array([[3, 3], [3, 3]])
            out = np.array(
                [[3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]]
            )  # Double size
            train_examples.append((inp, out))

        # Test: BOTH color and size change
        test_examples = []
        for _ in range(3):
            inp = np.array([[1, 1], [1, 1]])
            out = np.array(
                [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]
            )  # Red to blue AND double size
            test_examples.append((inp, out))

        return ImaginationTask(
            task_id="rule_color_size",
            category="rule_combination",
            train_examples=train_examples,
            test_examples=test_examples,
            required_insight="Combine color change with size change",
            valid_solutions=["color_and_scale", "dual_transformation"],
            difficulty=2,
        )

    @staticmethod
    def create_conditional_combo() -> ImaginationTask:
        """Train on conditions separately, test needs combined logic."""
        train_examples = []

        # Training: if center is 1, fill corners
        for _ in range(2):
            inp = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
            out = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
            train_examples.append((inp, out))

        # Training: if edge sum > 2, fill center
        for _ in range(2):
            inp = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]])
            out = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]])
            train_examples.append((inp, out))

        # Test: BOTH conditions apply
        test_examples = []
        inp = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]])
        out = np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]])  # Both patterns applied
        test_examples.append((inp, out))

        return ImaginationTask(
            task_id="rule_conditional",
            category="rule_combination",
            train_examples=train_examples,
            test_examples=test_examples,
            required_insight="Apply multiple conditional rules",
            valid_solutions=["multi_condition", "combined_logic"],
            difficulty=3,
        )


class CrossDomainTasks:
    """Tasks requiring transfer of principles across domains."""

    @staticmethod
    def create_2d_to_color_rotation() -> ImaginationTask:
        """Train on 2D spatial rotation, test on color wheel rotation."""
        train_examples = []

        # Training: 2D rotation
        for _ in range(4):
            inp = np.array([[1, 2, 0], [3, 4, 0], [0, 0, 0]])
            out = np.rot90(inp)
            train_examples.append((inp, out))

        # Test: Color rotation (applying rotation concept to color space)
        test_examples = []
        # Color wheel: 0=empty, 1=red, 2=green, 3=blue, 4=yellow
        # Rotation in color space: 1→2→3→4→1
        for _ in range(3):
            inp = np.array([[1, 2, 3], [4, 0, 1], [2, 3, 4]])
            out = np.array(
                [[2, 3, 4], [1, 0, 2], [3, 4, 1]]
            )  # Each color rotated to next
            test_examples.append((inp, out))

        return ImaginationTask(
            task_id="cross_2d_to_color",
            category="cross_domain",
            train_examples=train_examples,
            test_examples=test_examples,
            required_insight="Apply rotation concept to color domain",
            valid_solutions=["color_rotation", "domain_transfer_rotation"],
            difficulty=4,
        )

    @staticmethod
    def create_symmetry_transfer() -> ImaginationTask:
        """Train on spatial symmetry, test on value symmetry."""
        train_examples = []

        # Training: spatial mirror symmetry
        for _ in range(3):
            inp = np.array([[1, 2, 0], [3, 4, 0], [5, 6, 0]])
            out = np.array([[1, 2, 1], [3, 4, 3], [5, 6, 5]])  # Mirror right side
            train_examples.append((inp, out))

        # Test: value symmetry (make values symmetric around median)
        test_examples = []
        for _ in range(2):
            inp = np.array([[1, 9, 2], [8, 5, 3], [7, 4, 6]])
            # Make symmetric: if val < 5, mirror with val > 5
            out = np.array(
                [[1, 9, 2], [8, 5, 3], [7, 3, 7]]
            )  # Values balanced around 5
            test_examples.append((inp, out))

        return ImaginationTask(
            task_id="cross_symmetry",
            category="cross_domain",
            train_examples=train_examples,
            test_examples=test_examples,
            required_insight="Transfer symmetry concept from space to values",
            valid_solutions=["value_symmetry", "abstract_symmetry"],
            difficulty=5,
        )


class CounterfactualTasks:
    """Tasks requiring imagination of impossible/counterfactual scenarios."""

    @staticmethod
    def create_reverse_gravity() -> ImaginationTask:
        """Train on falling objects, test on rising objects."""
        train_examples = []

        # Training: objects fall down
        for start_row in range(3):
            inp = np.zeros((5, 3))
            inp[start_row, 1] = 1  # Object at top

            out = np.zeros((5, 3))
            out[min(start_row + 2, 4), 1] = 1  # Falls 2 positions
            train_examples.append((inp, out))

        # Test: objects rise up (reverse gravity)
        test_examples = []
        for start_row in [4, 3, 2]:
            inp = np.zeros((5, 3))
            inp[start_row, 1] = 1  # Object at bottom

            out = np.zeros((5, 3))
            out[max(start_row - 2, 0), 1] = 1  # Rises 2 positions
            test_examples.append((inp, out))

        return ImaginationTask(
            task_id="counterfactual_gravity",
            category="counterfactual",
            train_examples=train_examples,
            test_examples=test_examples,
            required_insight="Reverse the direction of gravity",
            valid_solutions=["reverse_gravity", "anti_gravity", "upward_force"],
            difficulty=2,
        )

    @staticmethod
    def create_negative_counting() -> ImaginationTask:
        """Train on incrementing, test on negative/imaginary counting."""
        train_examples = []

        # Training: count objects and increment
        for n in [1, 2, 3]:
            inp = np.zeros((3, 3))
            inp[:n, 0] = 1  # n objects

            out = np.zeros((3, 3))
            out[: n + 1, 1] = 1  # n+1 objects
            train_examples.append((inp, out))

        # Test: negative counting (remove more than exists)
        test_examples = []
        inp = np.zeros((3, 3))
        inp[0, 0] = 1  # 1 object

        out = np.zeros((3, 3))
        out[:, 2] = -1  # Negative objects (impossible but imaginable)
        test_examples.append((inp, out))

        return ImaginationTask(
            task_id="counterfactual_negative",
            category="counterfactual",
            train_examples=train_examples,
            test_examples=test_examples,
            required_insight="Imagine negative quantities",
            valid_solutions=["negative_count", "imaginary_objects"],
            difficulty=4,
        )


class CreativeProblemTasks:
    """Tasks with multiple valid creative solutions."""

    @staticmethod
    def create_sort_without_compare() -> ImaginationTask:
        """Sort values without using comparison operations."""
        train_examples = []

        # Training: show sorted patterns (but not HOW to sort)
        for _ in range(3):
            inp = np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]])
            out = np.array([[1, 1, 2], [3, 4, 5], [5, 6, 9]])  # Sorted
            train_examples.append((inp, out))

        # Test: need to find creative sorting method
        test_examples = []
        inp = np.array([[8, 3, 6], [1, 9, 2], [7, 4, 5]])
        out = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        test_examples.append((inp, out))

        return ImaginationTask(
            task_id="creative_sort",
            category="creative",
            train_examples=train_examples,
            test_examples=test_examples,
            required_insight="Sort without explicit comparisons",
            valid_solutions=[
                "counting_sort",
                "bucket_sort",
                "radix_sort",
                "histogram_sort",
            ],
            difficulty=3,
        )

    @staticmethod
    def create_path_without_search() -> ImaginationTask:
        """Find path without search algorithms."""
        train_examples = []

        # Training: paths from A to B
        for _ in range(2):
            inp = np.zeros((5, 5))
            inp[0, 0] = 1  # Start
            inp[4, 4] = 2  # End

            out = np.zeros((5, 5))
            # Diagonal path
            for i in range(5):
                out[i, i] = 3
            out[0, 0] = 1
            out[4, 4] = 2
            train_examples.append((inp, out))

        # Test: find creative path solution
        test_examples = []
        inp = np.zeros((5, 5))
        inp[0, 0] = 1
        inp[4, 0] = 2  # Different end position

        out = np.zeros((5, 5))
        # Creative: use gravity/flow metaphor
        for i in range(5):
            out[i, 0] = 3
        out[0, 0] = 1
        out[4, 0] = 2
        test_examples.append((inp, out))

        return ImaginationTask(
            task_id="creative_path",
            category="creative",
            train_examples=train_examples,
            test_examples=test_examples,
            required_insight="Find path using creative metaphor",
            valid_solutions=["flow_path", "gravity_path", "potential_field"],
            difficulty=4,
        )


class ImaginationBenchmark:
    """Main benchmark for testing imagination and distribution invention."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.tasks: Dict[str, List[ImaginationTask]] = {
            "pattern_discovery": [],
            "rule_combination": [],
            "cross_domain": [],
            "counterfactual": [],
            "creative": [],
        }

        # Initialize tasks
        self._create_all_tasks()

    def _create_all_tasks(self):
        """Create all benchmark tasks."""
        # Pattern discovery
        self.tasks["pattern_discovery"].append(
            PatternDiscoveryTasks.create_shear_task()
        )
        self.tasks["pattern_discovery"].append(
            PatternDiscoveryTasks.create_spiral_task()
        )

        # Rule combination
        self.tasks["rule_combination"].append(
            RuleCombinationTasks.create_color_size_combo()
        )
        self.tasks["rule_combination"].append(
            RuleCombinationTasks.create_conditional_combo()
        )

        # Cross-domain
        self.tasks["cross_domain"].append(
            CrossDomainTasks.create_2d_to_color_rotation()
        )
        self.tasks["cross_domain"].append(CrossDomainTasks.create_symmetry_transfer())

        # Counterfactual
        self.tasks["counterfactual"].append(
            CounterfactualTasks.create_reverse_gravity()
        )
        self.tasks["counterfactual"].append(
            CounterfactualTasks.create_negative_counting()
        )

        # Creative
        self.tasks["creative"].append(
            CreativeProblemTasks.create_sort_without_compare()
        )
        self.tasks["creative"].append(CreativeProblemTasks.create_path_without_search())

        if self.verbose:
            total_tasks = sum(len(tasks) for tasks in self.tasks.values())
            print(
                f"Created {total_tasks} imagination tasks across {len(self.tasks)} categories"
            )

    def evaluate_model(self, model, model_name: str = "unnamed") -> Dict:
        """Evaluate a model on all imagination tasks."""
        results = {
            "model": model_name,
            "overall_score": 0.0,
            "category_scores": {},
            "task_results": [],
        }

        for category, tasks in self.tasks.items():
            category_scores = []

            for task in tasks:
                if self.verbose:
                    print(f"\nEvaluating {model_name} on {task.task_id}")

                # Get model prediction
                prediction = self._get_model_prediction(model, task)

                # Evaluate
                score = 0.0
                if prediction is not None:
                    test_inp, test_out = task.test_examples[0]
                    score = task.evaluate_solution(prediction, test_out)

                category_scores.append(score)

                # Calculate novelty
                novelty = self._calculate_novelty(task, prediction)

                task_result = {
                    "task_id": task.task_id,
                    "category": category,
                    "score": score,
                    "novelty": novelty,
                    "difficulty": task.difficulty,
                }
                results["task_results"].append(task_result)

                if self.verbose:
                    print(f"  Score: {score:.2f}, Novelty: {novelty:.2f}")

            # Average category score
            results["category_scores"][category] = (
                np.mean(category_scores) if category_scores else 0.0
            )

        # Overall score
        all_scores = [r["score"] for r in results["task_results"]]
        results["overall_score"] = np.mean(all_scores) if all_scores else 0.0

        # Imagination score (combines accuracy and novelty)
        novelty_scores = [r["novelty"] for r in results["task_results"]]
        avg_novelty = np.mean(novelty_scores) if novelty_scores else 0.0
        results["imagination_score"] = results["overall_score"] * avg_novelty

        return results

    def _get_model_prediction(
        self, model, task: ImaginationTask
    ) -> Optional[np.ndarray]:
        """Get model's prediction for a task."""
        try:
            # Different model types
            if hasattr(model, "wake_phase"):
                # Wake-Sleep learner
                model.wake_phase(
                    [{"id": f"{task.task_id}_train", "examples": task.train_examples}]
                )

                # Try to solve test
                test_inp = task.test_examples[0][0]
                solution = model._solve_task([(test_inp, test_inp)])
                if solution:
                    return solution.execute(test_inp)

            elif hasattr(model, "synthesize"):
                # Program synthesizer
                programs = model.synthesize(task.train_examples, max_programs=5)
                if programs:
                    test_inp = task.test_examples[0][0]
                    return programs[0].execute(test_inp)

            elif hasattr(model, "learn_pattern"):
                # Few-shot learner
                hypothesis = model.learn_pattern(task.train_examples)
                if hypothesis:
                    test_inp = task.test_examples[0][0]
                    return hypothesis.test(test_inp)

            elif callable(model):
                # Generic function
                return model(task.train_examples, task.test_examples[0][0])

        except Exception as e:
            if self.verbose:
                print(f"  Error getting prediction: {e}")

        return None

    def _calculate_novelty(
        self, task: ImaginationTask, prediction: Optional[np.ndarray]
    ) -> float:
        """Calculate how novel/imaginative a solution is."""
        if prediction is None:
            return 0.0

        # Compare prediction to training outputs
        train_outputs = [out for _, out in task.train_examples]

        # Flatten for comparison
        pred_flat = prediction.flatten()

        novelty_scores = []
        for train_out in train_outputs:
            train_flat = train_out.flatten()

            # Resize if needed
            min_len = min(len(pred_flat), len(train_flat))
            if min_len > 0:
                similarity = 1.0 - cosine(pred_flat[:min_len], train_flat[:min_len])
                novelty_scores.append(1.0 - similarity)

        # Return minimum novelty (maximum similarity to training)
        return min(novelty_scores) if novelty_scores else 1.0

    def create_baseline(self) -> Callable:
        """Create a simple baseline that memorizes training patterns."""

        def baseline(train_examples, test_input):
            # Find most similar training input
            best_match_idx = 0
            best_similarity = -1

            for i, (train_inp, _) in enumerate(train_examples):
                if train_inp.shape == test_input.shape:
                    sim = np.sum(train_inp == test_input) / test_input.size
                    if sim > best_similarity:
                        best_similarity = sim
                        best_match_idx = i

            # Return corresponding output
            return train_examples[best_match_idx][1]

        return baseline

    def compare_models(self, models: Dict[str, Any]):
        """Compare multiple models on the benchmark."""
        try:
            import pandas as pd
        except ImportError:
            pd = None

        comparison_data = []

        for model_name, model in models.items():
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Evaluating: {model_name}")
                print(f"{'='*60}")

            results = self.evaluate_model(model, model_name)

            row = {
                "Model": model_name,
                "Overall": f"{results['overall_score']:.1%}",
                "Imagination": f"{results['imagination_score']:.1%}",
            }

            # Add category scores
            for category in self.tasks.keys():
                score = results["category_scores"].get(category, 0.0)
                row[category.replace("_", " ").title()] = f"{score:.1%}"

            comparison_data.append(row)

        if pd is not None:
            df = pd.DataFrame(comparison_data)

            if self.verbose:
                print(f"\n{'='*60}")
                print("COMPARISON RESULTS")
                print(f"{'='*60}")
                print(df.to_string(index=False))

            return df
        else:
            # Fallback without pandas
            if self.verbose:
                print(f"\n{'='*60}")
                print("COMPARISON RESULTS")
                print(f"{'='*60}")
                for row in comparison_data:
                    print(f"\n{row['Model']}:")
                    for key, value in row.items():
                        if key != "Model":
                            print(f"  {key}: {value}")

            return comparison_data

    def save_results(self, results: Dict, path: Path):
        """Save evaluation results to file."""
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        if self.verbose:
            print(f"Results saved to {path}")
