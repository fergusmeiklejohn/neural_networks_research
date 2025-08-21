"""Abstract Principle Extractor (APE) for cross-domain transfer.

This module extracts abstract, transferable principles from discovered patterns
and enables their application across different domains. It's the key to solving
cross-domain transfer tasks where we currently have 0% success.

Key Innovation: Represent transformations as abstract operations that can be
mapped to different representational spaces (spatial, color, symbolic, etc.).
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from hypothesis_generator import Hypothesis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Domain(Enum):
    """Different representational domains."""

    SPATIAL_2D = "spatial_2d"  # Grid positions
    COLOR = "color"  # Color values
    SIZE = "size"  # Object sizes
    SHAPE = "shape"  # Object shapes
    SYMBOLIC = "symbolic"  # Abstract symbols
    NUMERIC = "numeric"  # Numbers
    TEMPORAL = "temporal"  # Time-based


class AbstractOperation(Enum):
    """Abstract operations that can apply across domains."""

    ROTATE = "rotate"  # Cyclic permutation
    REFLECT = "reflect"  # Mirror symmetry
    TRANSLATE = "translate"  # Shift/offset
    SCALE = "scale"  # Magnify/shrink
    INVERT = "invert"  # Reverse/negate
    PERMUTE = "permute"  # Reorder
    COMPOSE = "compose"  # Combine operations
    CONDITIONAL = "conditional"  # If-then logic
    MAP = "map"  # Element-wise transformation
    REDUCE = "reduce"  # Aggregation


@dataclass
class AbstractPrinciple:
    """Represents an abstract principle that can transfer across domains."""

    name: str
    operation: AbstractOperation
    parameters: Dict[str, Any]
    source_domain: Domain
    description: str
    invariants: List[str]  # Properties that must hold

    def apply_to_domain(self, target_domain: Domain) -> Callable:
        """Generate a concrete function for the target domain."""
        if self.operation == AbstractOperation.ROTATE:
            return self._create_rotation_for_domain(target_domain)
        elif self.operation == AbstractOperation.REFLECT:
            return self._create_reflection_for_domain(target_domain)
        elif self.operation == AbstractOperation.TRANSLATE:
            return self._create_translation_for_domain(target_domain)
        elif self.operation == AbstractOperation.INVERT:
            return self._create_inversion_for_domain(target_domain)
        else:
            raise NotImplementedError(f"Operation {self.operation} not yet implemented")

    def _create_rotation_for_domain(self, domain: Domain) -> Callable:
        """Create rotation function for specific domain."""
        angle = self.parameters.get("angle", 90)

        if domain == Domain.SPATIAL_2D:
            # Standard 2D rotation
            def rotate_spatial(grid: np.ndarray) -> np.ndarray:
                rotations = int(angle / 90) % 4
                return np.rot90(grid, rotations)

            return rotate_spatial

        elif domain == Domain.COLOR:
            # Rotate through color space (e.g., RGB -> GBR -> BRG -> RGB)
            def rotate_color(grid: np.ndarray) -> np.ndarray:
                result = grid.copy()
                # Assuming values 0-9 represent different colors
                color_cycle = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                shift = int(angle / 36) % 10  # 360 degrees / 10 colors = 36 degrees per color

                for i in range(10):
                    mask = grid == color_cycle[i]
                    new_color = color_cycle[(i + shift) % 10]
                    result[mask] = new_color

                return result

            return rotate_color

        elif domain == Domain.SYMBOLIC:
            # Rotate through symbol sequence
            def rotate_symbolic(grid: np.ndarray) -> np.ndarray:
                unique_symbols = np.unique(grid)
                if len(unique_symbols) <= 1:
                    return grid

                result = grid.copy()
                n_symbols = len(unique_symbols)
                shift = int(angle / (360 / n_symbols)) % n_symbols

                for i, symbol in enumerate(unique_symbols):
                    mask = grid == symbol
                    new_symbol = unique_symbols[(i + shift) % n_symbols]
                    result[mask] = new_symbol

                return result

            return rotate_symbolic

        else:
            raise ValueError(f"Rotation not defined for domain {domain}")

    def _create_reflection_for_domain(self, domain: Domain) -> Callable:
        """Create reflection function for specific domain."""
        axis = self.parameters.get("axis", "horizontal")

        if domain == Domain.SPATIAL_2D:
            # Standard spatial reflection
            def reflect_spatial(grid: np.ndarray) -> np.ndarray:
                if axis == "horizontal":
                    return np.flipud(grid)
                elif axis == "vertical":
                    return np.fliplr(grid)
                elif axis == "diagonal":
                    return grid.T
                else:
                    return grid

            return reflect_spatial

        elif domain == Domain.COLOR:
            # Reflect in color space (e.g., complementary colors)
            def reflect_color(grid: np.ndarray) -> np.ndarray:
                result = grid.copy()
                # Simple complementary: 9-x for values 0-9
                max_val = 9
                return max_val - grid

            return reflect_color

        elif domain == Domain.NUMERIC:
            # Numeric reflection (negation)
            def reflect_numeric(grid: np.ndarray) -> np.ndarray:
                return -grid

            return reflect_numeric

        else:
            raise ValueError(f"Reflection not defined for domain {domain}")

    def _create_translation_for_domain(self, domain: Domain) -> Callable:
        """Create translation function for specific domain."""
        offset = self.parameters.get("offset", 1)

        if domain == Domain.SPATIAL_2D:
            # Spatial shift
            def translate_spatial(grid: np.ndarray) -> np.ndarray:
                result = np.zeros_like(grid)
                h, w = grid.shape
                for i in range(h):
                    for j in range(w):
                        new_j = (j + offset) % w
                        result[i, new_j] = grid[i, j]
                return result

            return translate_spatial

        elif domain == Domain.COLOR:
            # Color shift (hue shift)
            def translate_color(grid: np.ndarray) -> np.ndarray:
                return (grid + offset) % 10  # Assuming 0-9 color range

            return translate_color

        elif domain == Domain.NUMERIC:
            # Numeric offset
            def translate_numeric(grid: np.ndarray) -> np.ndarray:
                return grid + offset

            return translate_numeric

        else:
            raise ValueError(f"Translation not defined for domain {domain}")

    def _create_inversion_for_domain(self, domain: Domain) -> Callable:
        """Create inversion function for specific domain."""

        if domain == Domain.SPATIAL_2D:
            # Spatial inversion (180 degree rotation)
            def invert_spatial(grid: np.ndarray) -> np.ndarray:
                return np.rot90(grid, 2)

            return invert_spatial

        elif domain == Domain.COLOR:
            # Color inversion
            def invert_color(grid: np.ndarray) -> np.ndarray:
                return 9 - grid  # Assuming 0-9 range

            return invert_color

        elif domain == Domain.NUMERIC:
            # Numeric inversion
            def invert_numeric(grid: np.ndarray) -> np.ndarray:
                return -grid

            return invert_numeric

        else:
            raise ValueError(f"Inversion not defined for domain {domain}")


class AbstractPrincipleExtractor:
    """Extracts abstract principles from concrete transformations."""

    def __init__(self):
        self.extracted_principles: List[AbstractPrinciple] = []
        self.domain_mappings: Dict[Tuple[Domain, Domain], List[Callable]] = {}

    def extract_principle(
        self, hypothesis: Hypothesis, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[AbstractPrinciple]:
        """Extract abstract principle from a discovered hypothesis."""
        logger.info(f"Extracting principle from {hypothesis.transform_type}")

        # Analyze the transformation
        principle = self._analyze_transformation(hypothesis, examples)

        if principle:
            self.extracted_principles.append(principle)
            logger.info(f"Extracted principle: {principle.name}")

        return principle

    def _analyze_transformation(
        self, hypothesis: Hypothesis, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[AbstractPrinciple]:
        """Analyze transformation to identify abstract principle."""
        
        # First check hypothesis type hints
        if hypothesis.transform_type:
            type_lower = hypothesis.transform_type.lower()
            
            # Direct type matching
            if "shear" in type_lower or "diagonal" in type_lower:
                return self._extract_shear_principle(hypothesis, examples)
            elif "spiral" in type_lower:
                return self._extract_spiral_principle(hypothesis, examples)
            elif "rotate" in type_lower or "rotation" in type_lower:
                if self._verify_rotation(hypothesis, examples):
                    return self._extract_rotation_principle(hypothesis, examples)
            elif "reflect" in type_lower or "flip" in type_lower:
                if self._verify_reflection(hypothesis, examples):
                    return self._extract_reflection_principle(hypothesis, examples)
            elif "shift" in type_lower or "translate" in type_lower:
                return self._extract_translation_principle(hypothesis, examples)
            elif "scale" in type_lower:
                return self._extract_scale_principle(hypothesis, examples)
            elif "invert" in type_lower or "reverse" in type_lower:
                return self._extract_inversion_principle(hypothesis, examples)
        
        # Fallback to pattern analysis if no type hint
        # Check for shear first (often misidentified as rotation)
        if self._is_shear(hypothesis, examples):
            return self._extract_shear_principle(hypothesis, examples)
        
        # Check for rotation pattern
        if self._is_rotation(hypothesis, examples):
            return self._extract_rotation_principle(hypothesis, examples)

        # Check for reflection/symmetry
        if self._is_reflection(hypothesis, examples):
            return self._extract_reflection_principle(hypothesis, examples)

        # Check for translation/shift
        if self._is_translation(hypothesis, examples):
            return self._extract_translation_principle(hypothesis, examples)

        # Check for inversion
        if self._is_inversion(hypothesis, examples):
            return self._extract_inversion_principle(hypothesis, examples)

        # Default: unknown principle
        logger.warning(f"Could not identify principle for {hypothesis.transform_type}")
        return None

    def _is_shear(
        self, hypothesis: Hypothesis, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> bool:
        """Check if transformation is a shear."""
        if not examples:
            return False
            
        # Check for row-dependent column shift pattern
        for inp, out in examples:
            if inp.shape != out.shape:
                continue
                
            h, w = inp.shape
            is_shear = True
            
            # Check if each row is shifted by its index
            for row in range(h):
                # Find the shift amount for this row
                for shift in range(w):
                    shifted = np.roll(inp[row], shift)
                    if np.array_equal(shifted, out[row]):
                        # Check if shift matches row index
                        if shift != row % w:
                            is_shear = False
                        break
                        
            if is_shear:
                return True
                
        return False
    
    def _verify_rotation(
        self, hypothesis: Hypothesis, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> bool:
        """Verify that transformation is actually a rotation."""
        if not examples:
            return False
            
        # True rotation should match np.rot90 for some k
        for inp, out in examples:
            for k in range(4):
                if np.array_equal(np.rot90(inp, k), out):
                    return True
        return False
    
    def _verify_reflection(
        self, hypothesis: Hypothesis, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> bool:
        """Verify that transformation is actually a reflection."""
        if not examples:
            return False
            
        # Check standard reflections
        for inp, out in examples:
            if np.array_equal(np.flipud(inp), out):  # Horizontal reflection
                return True
            if np.array_equal(np.fliplr(inp), out):  # Vertical reflection
                return True
            if np.array_equal(inp.T, out):  # Diagonal reflection
                return True
        return False

    def _is_rotation(
        self, hypothesis: Hypothesis, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> bool:
        """Check if transformation is a rotation."""
        # Check if applying multiple times returns to original
        if not examples:
            return False

        input_grid = examples[0][0]
        transformed = input_grid.copy()

        # Apply transformation up to 4 times
        for i in range(4):
            transformed = hypothesis.apply(transformed)
            if np.array_equal(transformed, input_grid):
                # Also verify it's a true rotation
                return self._verify_rotation(hypothesis, examples)

        return False

    def _is_reflection(
        self, hypothesis: Hypothesis, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> bool:
        """Check if transformation is a reflection."""
        if not examples:
            return False

        # Reflection is self-inverse (applying twice returns original)
        input_grid = examples[0][0]
        once = hypothesis.apply(input_grid)
        twice = hypothesis.apply(once)

        return np.array_equal(twice, input_grid) and not np.array_equal(once, input_grid)

    def _is_translation(
        self, hypothesis: Hypothesis, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> bool:
        """Check if transformation is a translation/shift."""
        # Check for consistent offset pattern
        if "shift" in hypothesis.transform_type.lower() or "translate" in hypothesis.transform_type.lower():
            return True

        # Check if non-zero elements maintain relative positions
        if examples:
            inp, out = examples[0]
            if np.sum(inp != 0) == np.sum(out != 0):
                # Same number of non-zero elements, might be translation
                return True

        return False

    def _is_inversion(
        self, hypothesis: Hypothesis, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> bool:
        """Check if transformation is an inversion."""
        if not examples:
            return False

        # Check if values are inverted/negated
        inp, out = examples[0]

        # Check for value inversion (e.g., max - value)
        if inp.size > 0:
            max_val = np.max(inp)
            if max_val > 0:
                inverted = max_val - inp
                if np.array_equal(out, inverted):
                    return True

        return False

    def _extract_rotation_principle(
        self, hypothesis: Hypothesis, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> AbstractPrinciple:
        """Extract rotation principle."""
        # Determine rotation angle
        angle = 90  # Default
        if "180" in str(hypothesis.parameters):
            angle = 180
        elif "270" in str(hypothesis.parameters):
            angle = 270

        return AbstractPrinciple(
            name=f"rotation_{angle}",
            operation=AbstractOperation.ROTATE,
            parameters={"angle": angle},
            source_domain=Domain.SPATIAL_2D,
            description=f"Rotation by {angle} degrees",
            invariants=["preserves_element_count", "cyclic_operation"],
        )

    def _extract_reflection_principle(
        self, hypothesis: Hypothesis, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> AbstractPrinciple:
        """Extract reflection principle."""
        # Determine reflection axis
        axis = "horizontal"  # Default
        if "vertical" in str(hypothesis.transform_type).lower():
            axis = "vertical"
        elif "diagonal" in str(hypothesis.transform_type).lower():
            axis = "diagonal"

        return AbstractPrinciple(
            name=f"reflection_{axis}",
            operation=AbstractOperation.REFLECT,
            parameters={"axis": axis},
            source_domain=Domain.SPATIAL_2D,
            description=f"Reflection across {axis} axis",
            invariants=["self_inverse", "preserves_element_count"],
        )

    def _extract_translation_principle(
        self, hypothesis: Hypothesis, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> AbstractPrinciple:
        """Extract translation principle."""
        # Determine offset
        offset = hypothesis.parameters.get("shift", 1)
        if "factor" in hypothesis.parameters:
            offset = hypothesis.parameters["factor"]

        return AbstractPrinciple(
            name=f"translation_{offset}",
            operation=AbstractOperation.TRANSLATE,
            parameters={"offset": offset},
            source_domain=Domain.SPATIAL_2D,
            description=f"Translation by {offset} units",
            invariants=["preserves_element_values", "cyclic_on_boundaries"],
        )

    def _extract_inversion_principle(
        self, hypothesis: Hypothesis, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> AbstractPrinciple:
        """Extract inversion principle."""
        return AbstractPrinciple(
            name="inversion",
            operation=AbstractOperation.INVERT,
            parameters={},
            source_domain=Domain.NUMERIC,
            description="Value inversion",
            invariants=["self_inverse", "preserves_structure"],
        )
    
    def _extract_shear_principle(
        self, hypothesis: Hypothesis, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> AbstractPrinciple:
        """Extract shear principle."""
        # Determine shear parameters
        x_shear = hypothesis.parameters.get("x_shear", 1)
        y_shear = hypothesis.parameters.get("y_shear", 0)
        
        # For diagonal shift, x_shear is typically 1 (shift by row index)
        if "diagonal" in hypothesis.transform_type.lower():
            x_shear = 1
            
        return AbstractPrinciple(
            name=f"shear_{x_shear}_{y_shear}",
            operation=AbstractOperation.TRANSLATE,  # Shear is position-dependent translation
            parameters={"x_shear": x_shear, "y_shear": y_shear, "position_dependent": True},
            source_domain=Domain.SPATIAL_2D,
            description=f"Shear transformation (row-dependent shift by {x_shear})",
            invariants=["preserves_elements", "position_dependent_transform"],
        )
    
    def _extract_spiral_principle(
        self, hypothesis: Hypothesis, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> AbstractPrinciple:
        """Extract spiral principle."""
        clockwise = hypothesis.parameters.get("clockwise", True)
        
        return AbstractPrinciple(
            name=f"spiral_{'cw' if clockwise else 'ccw'}",
            operation=AbstractOperation.PERMUTE,
            parameters={"pattern": "spiral", "clockwise": clockwise},
            source_domain=Domain.SPATIAL_2D,
            description=f"Spiral permutation ({'clockwise' if clockwise else 'counter-clockwise'})",
            invariants=["preserves_all_elements", "spatial_reordering"],
        )
    
    def _extract_scale_principle(
        self, hypothesis: Hypothesis, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> AbstractPrinciple:
        """Extract scale principle."""
        x_scale = hypothesis.parameters.get("x_scale", 1.0)
        y_scale = hypothesis.parameters.get("y_scale", 1.0)
        
        return AbstractPrinciple(
            name=f"scale_{x_scale}_{y_scale}",
            operation=AbstractOperation.SCALE,
            parameters={"x_scale": x_scale, "y_scale": y_scale},
            source_domain=Domain.SPATIAL_2D,
            description=f"Scaling by ({x_scale}, {y_scale})",
            invariants=["may_change_size", "preserves_relative_positions"],
        )

    def transfer_principle(
        self,
        principle: AbstractPrinciple,
        target_domain: Domain,
        examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Optional[Callable]:
        """Transfer an abstract principle to a new domain."""
        logger.info(f"Transferring {principle.name} from {principle.source_domain} to {target_domain}")

        try:
            # Generate domain-specific function
            transform_fn = principle.apply_to_domain(target_domain)

            # Validate on examples if provided
            if examples:
                inp, expected = examples[0]
                result = transform_fn(inp)

                # Check if it produces reasonable output
                if result.shape == expected.shape:
                    similarity = np.sum(result == expected) / result.size
                    if similarity > 0.5:  # At least 50% match
                        logger.info(f"Successfully transferred with {similarity:.1%} accuracy")
                        return transform_fn

            return transform_fn

        except Exception as e:
            logger.warning(f"Failed to transfer principle: {e}")
            return None

    def identify_domain(self, data: np.ndarray) -> Domain:
        """Identify the domain of given data."""
        # Simple heuristics
        unique_values = np.unique(data)

        if len(unique_values) <= 10 and np.all(unique_values >= 0) and np.all(unique_values < 10):
            # Likely color or symbolic (0-9 range)
            if len(data.shape) == 2:
                return Domain.COLOR  # 2D grid of colors
            else:
                return Domain.SYMBOLIC

        elif data.dtype in [np.float32, np.float64]:
            return Domain.NUMERIC

        else:
            # Default to spatial
            return Domain.SPATIAL_2D

    def find_cross_domain_mapping(
        self,
        source_examples: List[Tuple[np.ndarray, np.ndarray]],
        target_examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Optional[Callable]:
        """Find mapping between different domains."""
        if not source_examples or not target_examples:
            return None

        # Identify domains
        source_domain = self.identify_domain(source_examples[0][0])
        target_domain = self.identify_domain(target_examples[0][0])

        logger.info(f"Mapping from {source_domain} to {target_domain}")

        # Look for common abstract principle
        for principle in self.extracted_principles:
            if principle.source_domain == source_domain:
                # Try to apply to target domain
                mapped_fn = self.transfer_principle(principle, target_domain, target_examples)
                if mapped_fn:
                    return mapped_fn

        return None

    def compose_principles(
        self, principles: List[AbstractPrinciple]
    ) -> Callable:
        """Compose multiple principles into a single transformation."""

        def composed_transform(grid: np.ndarray) -> np.ndarray:
            result = grid.copy()
            for principle in principles:
                # Apply each principle in sequence
                domain = self.identify_domain(result)
                transform_fn = principle.apply_to_domain(domain)
                result = transform_fn(result)
            return result

        return composed_transform

    def explain_principle(self, principle: AbstractPrinciple) -> str:
        """Generate human-readable explanation of a principle."""
        explanation = f"Principle: {principle.name}\n"
        explanation += f"Type: {principle.operation.value}\n"
        explanation += f"Description: {principle.description}\n"
        explanation += f"Originally from: {principle.source_domain.value} domain\n"

        if principle.invariants:
            explanation += "Invariant properties:\n"
            for inv in principle.invariants:
                explanation += f"  - {inv}\n"

        if principle.parameters:
            explanation += "Parameters:\n"
            for key, value in principle.parameters.items():
                explanation += f"  - {key}: {value}\n"

        return explanation