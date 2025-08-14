#!/usr/bin/env python3
"""Enhanced ARC Solver V7 Fixed - with smart tiling support.

This version fixes the tiling issues found in V7 by:
1. Using SmartTilePattern for learning-based tiling
2. Trying multiple tiling strategies
3. Properly returning transformed outputs
"""

from utils.imports import setup_project_paths

setup_project_paths()

import time
from typing import List, Optional, Tuple

import numpy as np
from arc_dsl_enhanced import EnhancedDSLLibrary
from enhanced_arc_solver_v3 import ARCSolution, SizeChangeInfo
from enhanced_arc_tta import EnhancedARCTestTimeAdapter
from enhanced_perception_v2 import EnhancedPerceptionV2
from enhanced_program_synthesis import EnhancedProgramSynthesizer
from enhanced_tiling_primitives import AlternatingRowTile, SmartTilePattern
from learnable_pattern_modifier import LearnablePatternModifier
from object_manipulation import ObjectManipulator
from position_dependent_modifier import PositionDependentModifier


class EnhancedARCSolverV7Fixed:
    """Fixed V7 solver with smart tiling support."""

    def __init__(
        self,
        use_synthesis: bool = True,
        synthesis_timeout: float = 8.0,
        synthesis_confidence_threshold: float = 0.75,
        use_position_learning: bool = True,
        confidence_threshold: float = 0.85,
    ):
        """Initialize the fixed V7 solver."""
        self.perception = EnhancedPerceptionV2()
        self.object_manipulator = ObjectManipulator()
        self.tta_adapter = EnhancedARCTestTimeAdapter()
        self.use_synthesis = use_synthesis
        self.synthesizer = (
            EnhancedProgramSynthesizer(
                beam_width=30,
                max_depth=4,
                use_perception_hints=True,
                timeout=synthesis_timeout,
            )
            if use_synthesis
            else None
        )
        self.synthesis_timeout = synthesis_timeout
        self.synthesis_confidence_threshold = synthesis_confidence_threshold
        self.use_position_learning = use_position_learning
        self.confidence_threshold = confidence_threshold
        self.dsl_library = EnhancedDSLLibrary()
        self.position_modifier = PositionDependentModifier()
        self.simple_modifier = LearnablePatternModifier()

        # Add smart tiling
        self.smart_tile = SmartTilePattern()
        self.alt_tile = AlternatingRowTile()

    def detect_size_change(
        self, train_examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> SizeChangeInfo:
        """Detect if the task involves size changes."""
        size_info = SizeChangeInfo(has_size_change=False)

        if not train_examples:
            return size_info

        # Analyze size ratios
        height_ratios = []
        width_ratios = []

        for input_grid, output_grid in train_examples:
            in_h, in_w = input_grid.shape
            out_h, out_w = output_grid.shape

            if in_h > 0 and in_w > 0:
                height_ratios.append(out_h / in_h)
                width_ratios.append(out_w / in_w)

        if not height_ratios:
            return size_info

        # Check if sizes change consistently
        avg_h_ratio = sum(height_ratios) / len(height_ratios)
        avg_w_ratio = sum(width_ratios) / len(width_ratios)

        # Check for size changes
        if abs(avg_h_ratio - 1.0) > 0.1 or abs(avg_w_ratio - 1.0) > 0.1:
            size_info.has_size_change = True
            size_info.scale_factor = (avg_h_ratio, avg_w_ratio)

            # Check if it's an exact multiple (likely tiling)
            if (
                avg_h_ratio == int(avg_h_ratio)
                and avg_w_ratio == int(avg_w_ratio)
                and avg_h_ratio == avg_w_ratio
            ):
                size_info.is_exact_multiple = True
                scale = int(avg_h_ratio)

                if scale > 1 and scale in [2, 3, 4]:
                    size_info.transformation_type = "tiling"
                elif scale > 1:
                    size_info.transformation_type = "scaling"
                elif scale < 1:
                    size_info.transformation_type = "cropping"

        return size_info

    def try_smart_tiling(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        size_info: SizeChangeInfo,
    ) -> Optional[ARCSolution]:
        """Try smart tiling that learns from examples."""

        if not size_info.has_size_change:
            return None

        # Try SmartTilePattern first
        try:
            output = self.smart_tile.learn_and_apply(train_examples, test_input)
            if output is not None:
                # Validate on training examples
                validation_score = self._validate_on_examples(
                    train_examples, self.smart_tile.learn_and_apply
                )

                if validation_score > 0.8:
                    return ARCSolution(
                        output,
                        validation_score,
                        "smart_tiling",
                    )
        except Exception:
            pass

        # Try AlternatingRowTile for 3x3 patterns
        if size_info.scale_factor and size_info.scale_factor[0] == 3.0:
            try:
                output = self.alt_tile.execute(test_input)

                # Validate on training examples
                validation_score = self._validate_alternating_tile(train_examples)

                if validation_score > 0.8:
                    return ARCSolution(
                        output,
                        validation_score,
                        "alternating_tiling",
                    )
            except Exception:
                pass

        # Try standard tiling with modifications
        if size_info.scale_factor:
            scale = int(size_info.scale_factor[0])

            # Get standard tile
            try:
                tile_primitive = self.dsl_library.get_primitive(
                    "tile_pattern", scale=scale
                )
                base_output = tile_primitive.execute(test_input)

                # Learn modifications if needed
                if self.use_position_learning:
                    learning_examples = []
                    for input_grid, expected_output in train_examples:
                        base = tile_primitive.execute(input_grid)
                        if base.shape == expected_output.shape:
                            learning_examples.append(
                                (input_grid, base, expected_output)
                            )

                    if learning_examples:
                        position_rules = (
                            self.position_modifier.learn_from_multiple_examples(
                                learning_examples, tile_size=test_input.shape
                            )
                        )

                        if position_rules:
                            modified_output = (
                                self.position_modifier.apply_position_rules(
                                    base_output,
                                    position_rules,
                                    tile_size=test_input.shape,
                                )
                            )

                            validation_score = self._validate_position_pattern(
                                train_examples,
                                tile_primitive.execute,
                                position_rules,
                                test_input.shape,
                            )

                            if validation_score > 0.5:
                                return ARCSolution(
                                    modified_output,
                                    validation_score,
                                    "position_tiling",
                                )

                # Return base tiling if no modifications needed
                return ARCSolution(
                    base_output,
                    0.5,
                    "simple_tiling",
                )

            except Exception:
                pass

        return None

    def _validate_on_examples(self, train_examples, transform_func):
        """Validate a transformation function on training examples."""
        if not train_examples:
            return 0.0

        correct = 0
        total = 0

        for input_grid, expected_output in train_examples:
            try:
                predicted = transform_func(train_examples, input_grid)
                if predicted is not None and np.array_equal(predicted, expected_output):
                    correct += 1
            except:
                pass
            total += 1

        return correct / total if total > 0 else 0.0

    def _validate_alternating_tile(self, train_examples):
        """Validate alternating tile pattern."""
        if not train_examples:
            return 0.0

        correct = 0
        total = 0

        for input_grid, expected_output in train_examples:
            try:
                predicted = self.alt_tile.execute(input_grid)
                if np.array_equal(predicted, expected_output):
                    correct += 1
            except:
                pass
            total += 1

        return correct / total if total > 0 else 0.0

    def _validate_position_pattern(
        self, train_examples, base_transform, position_rules, tile_size
    ):
        """Validate position-dependent pattern."""
        if not train_examples:
            return 0.0

        correct = 0
        total = 0

        for input_grid, expected_output in train_examples:
            try:
                base = base_transform(input_grid)
                if base.shape == expected_output.shape:
                    modified = self.position_modifier.apply_position_rules(
                        base, position_rules, tile_size=tile_size
                    )
                    if np.array_equal(modified, expected_output):
                        correct += 1
            except:
                pass
            total += 1

        return correct / total if total > 0 else 0.0

    def solve(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> ARCSolution:
        """Solve an ARC task with smart tiling support."""
        start_time = time.time()

        # Detect task characteristics
        size_info = self.detect_size_change(train_examples)

        if size_info.has_size_change:
            print(
                f"  Size change detected: {size_info.transformation_type} (scale: {size_info.scale_factor})"
            )

        # Track best solution
        best_solution = None
        best_confidence = 0.0

        # 1. Try smart tiling FIRST if size change detected
        if size_info.has_size_change:
            print("  Trying smart tiling...")
            solution = self.try_smart_tiling(train_examples, test_input, size_info)
            if solution and solution.confidence > best_confidence:
                best_solution = solution
                best_confidence = solution.confidence
                print(f"  Smart tiling: {best_confidence:.2f} confidence")

                # If high confidence, return early
                if best_confidence >= 0.9:
                    best_solution.time_taken = time.time() - start_time
                    return best_solution

        # 2. Try synthesis if enabled
        if self.use_synthesis and best_confidence < self.synthesis_confidence_threshold:
            print(f"  Trying synthesis (confidence: {best_confidence:.2f})...")
            perception_analysis = self.perception.analyze(train_examples)

            try:
                self.synthesizer.timeout = self.synthesis_timeout * 0.5
                program = self.synthesizer.synthesize(
                    train_examples, perception_analysis
                )

                if program:
                    output = program.execute(test_input)
                    validation_score = self._validate_program(program, train_examples)

                    if validation_score > best_confidence:
                        best_solution = ARCSolution(
                            output,
                            validation_score,
                            "synthesis",
                            program=program,
                        )
                        best_confidence = validation_score
                        print(f"  Synthesis: {best_confidence:.2f} confidence")
            except Exception:
                pass

        # 3. Use TTA as fallback
        if best_solution is None or best_confidence < 0.3:
            print(f"  Falling back to TTA (best so far: {best_confidence:.2f})")
            try:
                tta_output = self.tta_adapter.adapt(train_examples, test_input)
                best_solution = ARCSolution(
                    tta_output,
                    0.2,
                    "tta",
                )
            except:
                # Last resort: return input unchanged
                best_solution = ARCSolution(
                    test_input,
                    0.0,
                    "unchanged",
                )

        # Set time and return
        best_solution.time_taken = time.time() - start_time
        return best_solution

    def _validate_program(self, program, train_examples):
        """Validate a synthesized program."""
        if not train_examples:
            return 0.0

        correct = 0
        total = 0

        for input_grid, expected_output in train_examples:
            try:
                predicted = program.execute(input_grid)
                if np.array_equal(predicted, expected_output):
                    correct += 1
            except:
                pass
            total += 1

        return correct / total if total > 0 else 0.0
