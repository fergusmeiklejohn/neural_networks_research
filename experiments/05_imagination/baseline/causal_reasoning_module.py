"""Causal Reasoning Module for Understanding Why Transformations Work.

This module builds causal graphs to understand the underlying principles
of transformations, enabling transfer to novel situations.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from few_shot_pattern_learner import Hypothesis
from scipy import stats


@dataclass
class Invariant:
    """Represents something that doesn't change during transformation."""

    name: str
    invariant_type: str  # 'spatial', 'color', 'structural', 'count', 'shape'
    description: str
    value: Any
    confidence: float


@dataclass
class CausalRelation:
    """Represents a causal relationship between features."""

    cause: str
    effect: str
    relation_type: str  # 'direct', 'mediated', 'conditional'
    strength: float
    condition: Optional[str] = None


@dataclass
class TransformationPrinciple:
    """A transferable principle extracted from transformations."""

    name: str
    description: str
    invariants: List[Invariant]
    causal_relations: List[CausalRelation]
    applicable_when: str
    confidence: float


class CausalReasoningModule:
    """Builds causal understanding of transformations."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.causal_graph = nx.DiGraph()
        self.invariants: List[Invariant] = []
        self.principles: List[TransformationPrinciple] = []

    def analyze_transformation(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        hypothesis: Optional[Hypothesis] = None,
    ) -> Dict:
        """Analyze why a transformation works."""

        if self.verbose:
            print("Analyzing transformation causally...")

        # 1. Detect invariants
        self.invariants = self._detect_invariants(examples)

        # 2. Build causal graph
        causal_relations = self._build_causal_graph(examples)

        # 3. Extract mechanisms
        mechanisms = self._extract_causal_mechanisms(examples, causal_relations)

        # 4. Generate counterfactuals
        counterfactuals = self._generate_counterfactuals(examples, mechanisms)

        # 5. Extract principles
        principle = self._extract_principle(
            self.invariants, causal_relations, mechanisms, hypothesis
        )

        if principle:
            self.principles.append(principle)

        return {
            "invariants": self.invariants,
            "causal_relations": causal_relations,
            "mechanisms": mechanisms,
            "counterfactuals": counterfactuals,
            "principle": principle,
        }

    def _detect_invariants(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Invariant]:
        """Detect what stays constant during transformation."""
        invariants = []

        # Check spatial invariants
        spatial_invariants = self._detect_spatial_invariants(examples)
        invariants.extend(spatial_invariants)

        # Check color invariants
        color_invariants = self._detect_color_invariants(examples)
        invariants.extend(color_invariants)

        # Check structural invariants
        structural_invariants = self._detect_structural_invariants(examples)
        invariants.extend(structural_invariants)

        # Check count invariants
        count_invariants = self._detect_count_invariants(examples)
        invariants.extend(count_invariants)

        # Check shape invariants
        shape_invariants = self._detect_shape_invariants(examples)
        invariants.extend(shape_invariants)

        return invariants

    def _detect_spatial_invariants(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Invariant]:
        """Detect spatial properties that remain constant."""
        invariants = []

        # Check if relative positions are preserved
        relative_preserved = True
        for inp, out in examples:
            if inp.shape != out.shape:
                relative_preserved = False
                break

            # Check if non-zero positions maintain relative ordering
            inp_positions = np.argwhere(inp != 0)
            out_positions = np.argwhere(out != 0)

            if len(inp_positions) == len(out_positions) and len(inp_positions) > 1:
                # Check relative distances
                inp_dists = np.linalg.norm(
                    inp_positions[:, None] - inp_positions, axis=2
                )
                out_dists = np.linalg.norm(
                    out_positions[:, None] - out_positions, axis=2
                )

                # Allow for rotations/reflections by checking if distance matrix is similar
                if not np.allclose(
                    np.sort(inp_dists.flatten()), np.sort(out_dists.flatten()), rtol=0.1
                ):
                    relative_preserved = False
                    break

        if relative_preserved:
            invariants.append(
                Invariant(
                    name="relative_positions",
                    invariant_type="spatial",
                    description="Relative positions between elements are preserved",
                    value=True,
                    confidence=0.9,
                )
            )

        # Check if center of mass is preserved
        com_preserved = True
        for inp, out in examples:
            if inp.shape == out.shape:
                inp_com = (
                    np.mean(np.argwhere(inp != 0), axis=0)
                    if np.any(inp != 0)
                    else [0, 0]
                )
                out_com = (
                    np.mean(np.argwhere(out != 0), axis=0)
                    if np.any(out != 0)
                    else [0, 0]
                )

                # Check if center of mass is similar (allowing for small shifts)
                if not np.allclose(inp_com, out_com, atol=2):
                    com_preserved = False
                    break

        if com_preserved:
            invariants.append(
                Invariant(
                    name="center_of_mass",
                    invariant_type="spatial",
                    description="Center of mass is approximately preserved",
                    value=True,
                    confidence=0.7,
                )
            )

        return invariants

    def _detect_color_invariants(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Invariant]:
        """Detect color properties that remain constant."""
        invariants = []

        # Check if number of unique colors is preserved
        unique_preserved = True
        for inp, out in examples:
            inp_unique = len(np.unique(inp))
            out_unique = len(np.unique(out))
            if inp_unique != out_unique:
                unique_preserved = False
                break

        if unique_preserved:
            invariants.append(
                Invariant(
                    name="unique_color_count",
                    invariant_type="color",
                    description="Number of unique colors is preserved",
                    value=True,
                    confidence=0.95,
                )
            )

        # Check if color mappings are consistent
        color_mapping_consistent = True
        color_map = {}

        for inp, out in examples:
            if inp.shape != out.shape:
                color_mapping_consistent = False
                break

            # Build color mapping for this example
            example_map = {}
            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    if inp[i, j] != 0:
                        if inp[i, j] in example_map:
                            if example_map[inp[i, j]] != out[i, j]:
                                # Inconsistent mapping within example
                                color_mapping_consistent = False
                                break
                        else:
                            example_map[inp[i, j]] = out[i, j]

            # Check consistency across examples
            if color_mapping_consistent:
                for in_color, out_color in example_map.items():
                    if in_color in color_map:
                        if color_map[in_color] != out_color:
                            color_mapping_consistent = False
                            break
                    else:
                        color_map[in_color] = out_color

        if color_mapping_consistent and color_map:
            invariants.append(
                Invariant(
                    name="consistent_color_mapping",
                    invariant_type="color",
                    description="Each input color maps to a consistent output color",
                    value=color_map,
                    confidence=0.9,
                )
            )

        return invariants

    def _detect_structural_invariants(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Invariant]:
        """Detect structural properties that remain constant."""
        invariants = []

        # Check if connectivity is preserved
        connectivity_preserved = True
        for inp, out in examples:
            if inp.shape != out.shape:
                connectivity_preserved = False
                break

            # Simple connectivity check: are adjacent non-zero cells still adjacent?
            inp_connected = self._get_connectivity_pattern(inp)
            out_connected = self._get_connectivity_pattern(out)

            # Allow for rotations/reflections
            if not self._connectivity_similar(inp_connected, out_connected):
                connectivity_preserved = False
                break

        if connectivity_preserved:
            invariants.append(
                Invariant(
                    name="connectivity",
                    invariant_type="structural",
                    description="Connectivity between elements is preserved",
                    value=True,
                    confidence=0.8,
                )
            )

        return invariants

    def _detect_count_invariants(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Invariant]:
        """Detect count properties that remain constant."""
        invariants = []

        # Check if total non-zero count is preserved
        count_preserved = True
        for inp, out in examples:
            inp_count = np.sum(inp != 0)
            out_count = np.sum(out != 0)
            if inp_count != out_count:
                count_preserved = False
                break

        if count_preserved:
            invariants.append(
                Invariant(
                    name="element_count",
                    invariant_type="count",
                    description="Total number of non-zero elements is preserved",
                    value=True,
                    confidence=0.95,
                )
            )

        return invariants

    def _detect_shape_invariants(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Invariant]:
        """Detect shape properties that remain constant."""
        invariants = []

        # Check if dimensions are preserved
        shape_preserved = True
        for inp, out in examples:
            if inp.shape != out.shape:
                shape_preserved = False
                break

        if shape_preserved:
            invariants.append(
                Invariant(
                    name="shape",
                    invariant_type="shape",
                    description="Array dimensions are preserved",
                    value=True,
                    confidence=1.0,
                )
            )

        # Check if aspect ratio is preserved (for different sizes)
        aspect_preserved = True
        base_aspect = None
        for inp, out in examples:
            inp_aspect = inp.shape[0] / inp.shape[1] if inp.shape[1] > 0 else 0
            out_aspect = out.shape[0] / out.shape[1] if out.shape[1] > 0 else 0

            if base_aspect is None:
                base_aspect = out_aspect / inp_aspect if inp_aspect > 0 else 1
            else:
                current_ratio = out_aspect / inp_aspect if inp_aspect > 0 else 1
                if not np.isclose(current_ratio, base_aspect, rtol=0.1):
                    aspect_preserved = False
                    break

        if aspect_preserved and base_aspect is not None:
            invariants.append(
                Invariant(
                    name="aspect_ratio",
                    invariant_type="shape",
                    description="Aspect ratio relationship is preserved",
                    value=base_aspect,
                    confidence=0.85,
                )
            )

        return invariants

    def _get_connectivity_pattern(
        self, arr: np.ndarray
    ) -> Set[Tuple[int, int, int, int]]:
        """Get connectivity pattern as set of adjacent non-zero pairs."""
        connections = set()
        h, w = arr.shape

        for i in range(h):
            for j in range(w):
                if arr[i, j] != 0:
                    # Check 4-connectivity
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w and arr[ni, nj] != 0:
                            # Store as sorted tuple to avoid duplicates
                            connections.add(tuple(sorted([(i, j), (ni, nj)])))

        return connections

    def _connectivity_similar(self, conn1: Set, conn2: Set) -> bool:
        """Check if two connectivity patterns are similar (allowing for transformations)."""
        if len(conn1) != len(conn2):
            return False

        # For simplicity, just check if same number of connections
        # In full implementation, would check for isomorphism
        return True

    def _build_causal_graph(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[CausalRelation]:
        """Build causal graph of transformation."""
        relations = []

        # Analyze feature dependencies
        features = self._extract_features(examples)

        # Test causal relationships between features
        for cause_feat in features:
            for effect_feat in features:
                if cause_feat != effect_feat:
                    strength = self._test_causal_relation(
                        examples, cause_feat, effect_feat
                    )
                    if strength > 0.5:  # Threshold for significant causation
                        relations.append(
                            CausalRelation(
                                cause=cause_feat,
                                effect=effect_feat,
                                relation_type="direct",
                                strength=strength,
                            )
                        )

        # Build graph for analysis
        self.causal_graph.clear()
        for rel in relations:
            self.causal_graph.add_edge(rel.cause, rel.effect, weight=rel.strength)

        # Detect mediated relations
        mediated = self._detect_mediated_relations(relations)
        relations.extend(mediated)

        return relations

    def _extract_features(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[str]:
        """Extract relevant features from examples."""
        features = []

        # Basic features present in all examples
        for inp, out in examples:
            if "position" not in features:
                features.append("position")
            if "color" not in features:
                features.append("color")
            if "shape" not in features and inp.shape != out.shape:
                features.append("shape")
            if "count" not in features:
                features.append("count")
            if "pattern" not in features:
                features.append("pattern")

        return features

    def _test_causal_relation(
        self, examples: List[Tuple[np.ndarray, np.ndarray]], cause: str, effect: str
    ) -> float:
        """Test strength of causal relation between features."""
        # Simplified causal testing
        # In full implementation, would use more sophisticated causal inference

        correlations = []
        for inp, out in examples:
            # Extract feature values
            cause_val = self._get_feature_value(inp, cause)
            effect_val = self._get_feature_value(out, effect)

            if cause_val is not None and effect_val is not None:
                # Simple correlation as proxy for causation
                if isinstance(cause_val, (int, float)) and isinstance(
                    effect_val, (int, float)
                ):
                    correlations.append((cause_val, effect_val))

        if len(correlations) > 1:
            cause_vals, effect_vals = zip(*correlations)
            correlation, _ = stats.pearsonr(cause_vals, effect_vals)
            return abs(correlation)

        # Default: check if cause changes imply effect changes
        cause_changes = False
        effect_changes = False

        for inp, out in examples:
            if not np.array_equal(inp, out):
                cause_changes = True
                if self._feature_changed(inp, out, effect):
                    effect_changes = True

        if cause_changes and effect_changes:
            return 0.7  # Moderate causal relation

        return 0.0

    def _get_feature_value(self, arr: np.ndarray, feature: str) -> Optional[Any]:
        """Extract feature value from array."""
        if feature == "position":
            # Return center of mass
            positions = np.argwhere(arr != 0)
            if len(positions) > 0:
                return tuple(np.mean(positions, axis=0))
            return None
        elif feature == "color":
            # Return dominant color
            nonzero = arr[arr != 0]
            if len(nonzero) > 0:
                return stats.mode(nonzero, keepdims=False)[0]
            return None
        elif feature == "shape":
            return arr.shape
        elif feature == "count":
            return np.sum(arr != 0)
        elif feature == "pattern":
            # Return simplified pattern hash
            return hash(arr.tobytes())

        return None

    def _feature_changed(self, inp: np.ndarray, out: np.ndarray, feature: str) -> bool:
        """Check if feature changed between input and output."""
        inp_val = self._get_feature_value(inp, feature)
        out_val = self._get_feature_value(out, feature)

        if inp_val is None or out_val is None:
            return False

        if isinstance(inp_val, np.ndarray):
            return not np.array_equal(inp_val, out_val)
        else:
            return inp_val != out_val

    def _detect_mediated_relations(
        self, direct_relations: List[CausalRelation]
    ) -> List[CausalRelation]:
        """Detect mediated causal relations (A → B → C)."""
        mediated = []

        # Build adjacency for easy lookup
        causes = defaultdict(list)
        effects = defaultdict(list)

        for rel in direct_relations:
            causes[rel.cause].append(rel)
            effects[rel.effect].append(rel)

        # Find chains
        for rel1 in direct_relations:
            # Find what rel1.effect causes
            for rel2 in causes.get(rel1.effect, []):
                # We have: rel1.cause → rel1.effect → rel2.effect
                if rel1.cause != rel2.effect:  # Avoid cycles
                    combined_strength = rel1.strength * rel2.strength
                    if combined_strength > 0.3:  # Threshold for mediated relation
                        mediated.append(
                            CausalRelation(
                                cause=rel1.cause,
                                effect=rel2.effect,
                                relation_type="mediated",
                                strength=combined_strength,
                                condition=f"via {rel1.effect}",
                            )
                        )

        return mediated

    def _extract_causal_mechanisms(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        relations: List[CausalRelation],
    ) -> Dict[str, Any]:
        """Extract the mechanisms behind the causal relations."""
        mechanisms = {"transformation_type": None, "mechanism": None, "parameters": {}}

        # Analyze what type of transformation this is
        inp, out = examples[0]

        # Check for spatial transformations
        if inp.shape == out.shape:
            # Rotation check
            for k in [1, 2, 3]:
                if np.array_equal(np.rot90(inp, k), out):
                    mechanisms["transformation_type"] = "rotation"
                    mechanisms["mechanism"] = "spatial_rotation"
                    mechanisms["parameters"]["degrees"] = k * 90
                    return mechanisms

            # Reflection check
            if np.array_equal(np.flipud(inp), out):
                mechanisms["transformation_type"] = "reflection"
                mechanisms["mechanism"] = "vertical_flip"
                return mechanisms

            if np.array_equal(np.fliplr(inp), out):
                mechanisms["transformation_type"] = "reflection"
                mechanisms["mechanism"] = "horizontal_flip"
                return mechanisms

        # Check for scaling
        if inp.shape != out.shape:
            h_ratio = out.shape[0] / inp.shape[0] if inp.shape[0] > 0 else 0
            w_ratio = out.shape[1] / inp.shape[1] if inp.shape[1] > 0 else 0

            if np.isclose(h_ratio, w_ratio):
                mechanisms["transformation_type"] = "scaling"
                mechanisms["mechanism"] = "uniform_scale"
                mechanisms["parameters"]["factor"] = h_ratio
                return mechanisms

        # Check for color transformations
        if set(np.unique(inp)) != set(np.unique(out)):
            mechanisms["transformation_type"] = "color_change"
            mechanisms["mechanism"] = "color_mapping"

            # Build color map
            color_map = {}
            for i in range(min(inp.shape[0], out.shape[0])):
                for j in range(min(inp.shape[1], out.shape[1])):
                    if inp[i, j] != 0:
                        color_map[int(inp[i, j])] = int(out[i, j])

            mechanisms["parameters"]["color_map"] = color_map
            return mechanisms

        # Default: pattern-based transformation
        mechanisms["transformation_type"] = "pattern"
        mechanisms["mechanism"] = "complex_pattern"

        return mechanisms

    def _generate_counterfactuals(
        self, examples: List[Tuple[np.ndarray, np.ndarray]], mechanisms: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate counterfactual scenarios to test understanding."""
        counterfactuals = []

        # Generate "what if" scenarios based on mechanisms
        if mechanisms["transformation_type"] == "rotation":
            degrees = mechanisms["parameters"].get("degrees", 90)
            counterfactuals.append(
                {
                    "scenario": f"What if we rotated by {degrees + 90} degrees instead?",
                    "prediction": "The output would be rotated an additional 90 degrees",
                    "testable": True,
                }
            )
            counterfactuals.append(
                {
                    "scenario": "What if the input had a different shape?",
                    "prediction": "Rotation would still apply, preserving relative positions",
                    "testable": True,
                }
            )

        elif mechanisms["transformation_type"] == "scaling":
            factor = mechanisms["parameters"].get("factor", 2)
            counterfactuals.append(
                {
                    "scenario": f"What if we scaled by {factor * 2} instead?",
                    "prediction": "The output would be twice as large",
                    "testable": True,
                }
            )
            counterfactuals.append(
                {
                    "scenario": "What if the input was already scaled?",
                    "prediction": "Scaling would compound, multiplying the factors",
                    "testable": True,
                }
            )

        elif mechanisms["transformation_type"] == "color_change":
            counterfactuals.append(
                {
                    "scenario": "What if we had additional colors in the input?",
                    "prediction": "New colors would follow the same mapping pattern",
                    "testable": True,
                }
            )
            counterfactuals.append(
                {
                    "scenario": "What if we applied the transformation twice?",
                    "prediction": "Colors might cycle or reach a fixed point",
                    "testable": True,
                }
            )

        # General counterfactuals
        counterfactuals.append(
            {
                "scenario": "What if we reversed the transformation?",
                "prediction": "We should recover the original input (if invertible)",
                "testable": True,
            }
        )

        return counterfactuals

    def _extract_principle(
        self,
        invariants: List[Invariant],
        relations: List[CausalRelation],
        mechanisms: Dict[str, Any],
        hypothesis: Optional[Hypothesis],
    ) -> Optional[TransformationPrinciple]:
        """Extract a transferable principle from the analysis."""

        if not mechanisms["mechanism"]:
            return None

        # Build principle based on mechanism type
        if mechanisms["transformation_type"] == "rotation":
            return TransformationPrinciple(
                name="rotation_principle",
                description="Rotation preserves relative positions while changing absolute positions",
                invariants=[
                    inv
                    for inv in invariants
                    if inv.invariant_type in ["spatial", "structural"]
                ],
                causal_relations=relations,
                applicable_when="Need to change orientation while preserving structure",
                confidence=0.9,
            )

        elif mechanisms["transformation_type"] == "scaling":
            return TransformationPrinciple(
                name="scaling_principle",
                description="Scaling preserves relative proportions while changing absolute size",
                invariants=[
                    inv
                    for inv in invariants
                    if inv.invariant_type in ["shape", "structural"]
                ],
                causal_relations=relations,
                applicable_when="Need to change size while preserving proportions",
                confidence=0.85,
            )

        elif mechanisms["transformation_type"] == "reflection":
            return TransformationPrinciple(
                name="reflection_principle",
                description="Reflection preserves distances but reverses orientation",
                invariants=[
                    inv for inv in invariants if inv.invariant_type == "spatial"
                ],
                causal_relations=relations,
                applicable_when="Need to mirror structure across an axis",
                confidence=0.9,
            )

        elif mechanisms["transformation_type"] == "color_change":
            return TransformationPrinciple(
                name="color_mapping_principle",
                description="Consistent color mapping preserves patterns while changing appearance",
                invariants=[
                    inv
                    for inv in invariants
                    if inv.invariant_type in ["structural", "count"]
                ],
                causal_relations=relations,
                applicable_when="Need to change colors while preserving pattern structure",
                confidence=0.8,
            )

        # Generic principle
        return TransformationPrinciple(
            name="generic_transformation",
            description=f"{mechanisms['transformation_type']} transformation with specific rules",
            invariants=invariants[:3] if len(invariants) > 3 else invariants,
            causal_relations=relations[:3] if len(relations) > 3 else relations,
            applicable_when="Complex pattern transformation needed",
            confidence=0.6,
        )

    def apply_principle(
        self, principle: TransformationPrinciple, new_input: np.ndarray
    ) -> Optional[np.ndarray]:
        """Apply a learned principle to new input."""

        if principle.name == "rotation_principle":
            # Apply rotation (default 90 degrees)
            return np.rot90(new_input)

        elif principle.name == "reflection_principle":
            # Apply reflection (default vertical)
            return np.flipud(new_input)

        elif principle.name == "scaling_principle":
            # Apply scaling (default 2x)
            return np.repeat(np.repeat(new_input, 2, axis=0), 2, axis=1)

        # For complex principles, would need more sophisticated application
        return None

    def explain_transformation(self, analysis: Dict) -> str:
        """Generate human-readable explanation of the transformation."""

        explanation = "CAUSAL ANALYSIS OF TRANSFORMATION\n"
        explanation += "=" * 50 + "\n\n"

        # Explain invariants
        if analysis["invariants"]:
            explanation += "What stays constant:\n"
            for inv in analysis["invariants"][:5]:  # Top 5 invariants
                explanation += (
                    f"  • {inv.description} (confidence: {inv.confidence:.1%})\n"
                )
            explanation += "\n"

        # Explain causal relations
        if analysis["causal_relations"]:
            explanation += "Causal relationships:\n"
            for rel in analysis["causal_relations"][:5]:  # Top 5 relations
                if rel.relation_type == "direct":
                    explanation += f"  • {rel.cause} → {rel.effect} (strength: {rel.strength:.2f})\n"
                elif rel.relation_type == "mediated":
                    explanation += f"  • {rel.cause} → {rel.effect} {rel.condition} (strength: {rel.strength:.2f})\n"
            explanation += "\n"

        # Explain mechanism
        if analysis["mechanisms"]:
            mech = analysis["mechanisms"]
            explanation += f"Transformation mechanism: {mech['transformation_type']}\n"
            if mech["mechanism"]:
                explanation += f"  Specific operation: {mech['mechanism']}\n"
            if mech["parameters"]:
                explanation += f"  Parameters: {mech['parameters']}\n"
            explanation += "\n"

        # Explain counterfactuals
        if analysis["counterfactuals"]:
            explanation += "Counterfactual reasoning:\n"
            for cf in analysis["counterfactuals"][:3]:  # Top 3 counterfactuals
                explanation += f"  • {cf['scenario']}\n"
                explanation += f"    → {cf['prediction']}\n"
            explanation += "\n"

        # Explain principle
        if analysis["principle"]:
            prin = analysis["principle"]
            explanation += f"Extracted principle: {prin.name}\n"
            explanation += f"  {prin.description}\n"
            explanation += f"  Applicable when: {prin.applicable_when}\n"
            explanation += f"  Confidence: {prin.confidence:.1%}\n"

        return explanation

    def transfer_knowledge(
        self,
        source_examples: List[Tuple[np.ndarray, np.ndarray]],
        target_examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, Any]:
        """Transfer knowledge from source to target domain."""

        # Analyze source
        source_analysis = self.analyze_transformation(source_examples)

        if not source_analysis["principle"]:
            return {"success": False, "reason": "No principle extracted from source"}

        principle = source_analysis["principle"]

        # Try to apply principle to target
        success_count = 0
        for inp, expected_out in target_examples:
            predicted_out = self.apply_principle(principle, inp)
            if predicted_out is not None and np.array_equal(
                predicted_out, expected_out
            ):
                success_count += 1

        transfer_rate = success_count / len(target_examples) if target_examples else 0

        return {
            "success": transfer_rate > 0.5,
            "transfer_rate": transfer_rate,
            "principle_used": principle.name,
            "source_invariants": len(source_analysis["invariants"]),
            "applicable_to_target": transfer_rate > 0.5,
        }
