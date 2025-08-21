"""Compositional Reasoning Module for multi-attribute rules.

This module enables reasoning about tasks that require understanding
relationships between multiple attributes (color, size, position, etc.)
and applying conditional logic (if-then rules).

Key Innovation: Represent transformations as compositions of attribute
modifications with logical conditions.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Attribute(Enum):
    """Different attributes that can be reasoned about."""
    
    COLOR = "color"
    SIZE = "size"
    POSITION = "position"
    SHAPE = "shape"
    COUNT = "count"
    ORIENTATION = "orientation"
    PATTERN = "pattern"
    VALUE = "value"


class LogicalOperator(Enum):
    """Logical operators for conditions."""
    
    AND = "and"
    OR = "or"
    NOT = "not"
    IF_THEN = "if_then"
    EQUALS = "equals"
    GREATER = "greater"
    LESS = "less"


@dataclass
class AttributeCondition:
    """Represents a condition on an attribute."""
    
    attribute: Attribute
    operator: LogicalOperator
    value: Any
    
    def evaluate(self, obj: Dict[str, Any]) -> bool:
        """Evaluate condition on an object."""
        if self.attribute.value not in obj:
            return False
            
        obj_value = obj[self.attribute.value]
        
        if self.operator == LogicalOperator.EQUALS:
            return obj_value == self.value
        elif self.operator == LogicalOperator.GREATER:
            return obj_value > self.value
        elif self.operator == LogicalOperator.LESS:
            return obj_value < self.value
        elif self.operator == LogicalOperator.NOT:
            return obj_value != self.value
        else:
            return False


@dataclass
class CompositeRule:
    """Represents a composite rule with conditions and actions."""
    
    name: str
    conditions: List[AttributeCondition]
    actions: List[Tuple[Attribute, Callable]]
    logical_op: LogicalOperator = LogicalOperator.AND
    
    def applies_to(self, obj: Dict[str, Any]) -> bool:
        """Check if rule applies to an object."""
        if not self.conditions:
            return True
            
        if self.logical_op == LogicalOperator.AND:
            return all(cond.evaluate(obj) for cond in self.conditions)
        elif self.logical_op == LogicalOperator.OR:
            return any(cond.evaluate(obj) for cond in self.conditions)
        else:
            return False
    
    def apply(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Apply rule to transform object."""
        if not self.applies_to(obj):
            return obj
            
        result = obj.copy()
        for attribute, action in self.actions:
            if attribute.value in result:
                result[attribute.value] = action(result[attribute.value])
        
        return result


class CompositionalReasoner:
    """Enables compositional reasoning about multi-attribute transformations."""
    
    def __init__(self):
        self.learned_rules: List[CompositeRule] = []
        self.attribute_extractors: Dict[Attribute, Callable] = {}
        self._setup_default_extractors()
    
    def _setup_default_extractors(self):
        """Set up default attribute extractors."""
        
        def extract_color(grid: np.ndarray, pos: Tuple[int, int]) -> int:
            """Extract color value at position."""
            return int(grid[pos[0], pos[1]])
        
        def extract_size(grid: np.ndarray, obj_mask: np.ndarray) -> int:
            """Extract size of object."""
            return int(np.sum(obj_mask))
        
        def extract_position(grid: np.ndarray, obj_mask: np.ndarray) -> Tuple[int, int]:
            """Extract center position of object."""
            coords = np.argwhere(obj_mask)
            if len(coords) > 0:
                return tuple(np.mean(coords, axis=0).astype(int))
            return (0, 0)
        
        def extract_count(grid: np.ndarray) -> int:
            """Count non-zero elements."""
            return int(np.sum(grid != 0))
        
        self.attribute_extractors[Attribute.COLOR] = extract_color
        self.attribute_extractors[Attribute.SIZE] = extract_size
        self.attribute_extractors[Attribute.POSITION] = extract_position
        self.attribute_extractors[Attribute.COUNT] = extract_count
    
    def extract_objects(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Extract objects and their attributes from a grid."""
        objects = []
        
        # Simple object detection: connected components
        visited = np.zeros_like(grid, dtype=bool)
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] != 0 and not visited[i, j]:
                    # Found new object
                    obj_mask = self._flood_fill(grid, i, j, grid[i, j], visited)
                    
                    obj = {
                        "mask": obj_mask,
                        "color": int(grid[i, j]),
                        "size": int(np.sum(obj_mask)),
                        "position": self.attribute_extractors[Attribute.POSITION](grid, obj_mask),
                        "value": int(grid[i, j]),
                    }
                    objects.append(obj)
        
        return objects
    
    def _flood_fill(
        self, grid: np.ndarray, start_i: int, start_j: int, 
        target_val: int, visited: np.ndarray
    ) -> np.ndarray:
        """Flood fill to find connected component."""
        mask = np.zeros_like(grid, dtype=bool)
        stack = [(start_i, start_j)]
        
        while stack:
            i, j = stack.pop()
            
            if (i < 0 or i >= grid.shape[0] or j < 0 or j >= grid.shape[1] or
                visited[i, j] or grid[i, j] != target_val):
                continue
            
            visited[i, j] = True
            mask[i, j] = True
            
            # Add neighbors
            stack.extend([(i+1, j), (i-1, j), (i, j+1), (i, j-1)])
        
        return mask
    
    def learn_rule_from_examples(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[CompositeRule]:
        """Learn a composite rule from examples."""
        
        if not examples:
            return None
        
        logger.info(f"Learning rule from {len(examples)} examples")
        
        # Extract objects from all examples
        all_transformations = []
        
        for inp, out in examples:
            input_objects = self.extract_objects(inp)
            output_objects = self.extract_objects(out)
            
            # Try to match input to output objects
            for in_obj in input_objects:
                # Find corresponding output object
                for out_obj in output_objects:
                    if self._objects_correspond(in_obj, out_obj):
                        all_transformations.append((in_obj, out_obj))
                        break
        
        if not all_transformations:
            logger.warning("Could not find object correspondences")
            return None
        
        # Analyze transformations to find patterns
        rule = self._analyze_transformations(all_transformations)
        
        if rule:
            self.learned_rules.append(rule)
            logger.info(f"Learned rule: {rule.name}")
        
        return rule
    
    def _objects_correspond(self, obj1: Dict, obj2: Dict) -> bool:
        """Check if two objects likely correspond."""
        # Simple heuristic: similar position or size
        pos1 = obj1.get("position", (0, 0))
        pos2 = obj2.get("position", (0, 0))
        
        dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        
        # Objects correspond if nearby or same size
        return dist < 3 or obj1.get("size") == obj2.get("size")
    
    def _analyze_transformations(
        self, transformations: List[Tuple[Dict, Dict]]
    ) -> Optional[CompositeRule]:
        """Analyze transformations to extract rule."""
        
        if not transformations:
            return None
        
        # Look for consistent patterns
        conditions = []
        actions = []
        
        # Check color-based rules
        color_changes = {}
        for in_obj, out_obj in transformations:
            in_color = in_obj.get("color")
            out_color = out_obj.get("color")
            
            if in_color is not None and out_color is not None:
                if in_color not in color_changes:
                    color_changes[in_color] = []
                color_changes[in_color].append(out_color)
        
        # If consistent color mapping exists
        for in_color, out_colors in color_changes.items():
            if len(set(out_colors)) == 1:  # All map to same color
                conditions.append(
                    AttributeCondition(
                        Attribute.COLOR,
                        LogicalOperator.EQUALS,
                        in_color
                    )
                )
                
                def color_transform(val, target=out_colors[0]):
                    return target
                
                actions.append((Attribute.COLOR, color_transform))
        
        # Check size-based rules
        size_patterns = []
        for in_obj, out_obj in transformations:
            in_size = in_obj.get("size", 0)
            out_size = out_obj.get("size", 0)
            
            if in_size > 0:
                size_ratio = out_size / in_size
                size_patterns.append(size_ratio)
        
        if size_patterns and len(set(size_patterns)) == 1:
            # Consistent size transformation
            ratio = size_patterns[0]
            
            def size_transform(val):
                return int(val * ratio)
            
            actions.append((Attribute.SIZE, size_transform))
        
        if conditions or actions:
            rule = CompositeRule(
                name=f"learned_rule_{len(self.learned_rules)}",
                conditions=conditions,
                actions=actions,
                logical_op=LogicalOperator.AND
            )
            return rule
        
        return None
    
    def apply_rule(self, rule: CompositeRule, grid: np.ndarray) -> np.ndarray:
        """Apply a composite rule to transform a grid."""
        
        # Extract objects
        objects = self.extract_objects(grid)
        
        # Create output grid
        output = np.zeros_like(grid)
        
        # Apply rule to each object
        for obj in objects:
            transformed = rule.apply(obj)
            
            # Place transformed object in output
            if "mask" in transformed:
                mask = transformed["mask"]
                color = transformed.get("color", obj.get("color", 1))
                output[mask] = color
        
        return output
    
    def discover_conditional_rule(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[CompositeRule]:
        """Discover if-then conditional rules."""
        
        logger.info("Discovering conditional rules...")
        
        # Look for patterns where certain conditions trigger specific actions
        conditional_patterns = []
        
        for inp, out in examples:
            input_objects = self.extract_objects(inp)
            output_objects = self.extract_objects(out)
            
            # Check for conditional transformations
            for in_obj in input_objects:
                # Find what attributes trigger what changes
                for attr in Attribute:
                    if attr.value in in_obj:
                        value = in_obj[attr.value]
                        
                        # See if this value consistently triggers a change
                        for out_obj in output_objects:
                            if self._objects_correspond(in_obj, out_obj):
                                # Record the condition and result
                                conditional_patterns.append({
                                    "condition_attr": attr,
                                    "condition_value": value,
                                    "changes": self._compute_changes(in_obj, out_obj)
                                })
        
        # Find consistent conditional patterns
        if conditional_patterns:
            # Group by condition
            grouped = {}
            for pattern in conditional_patterns:
                key = (pattern["condition_attr"], pattern["condition_value"])
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(pattern["changes"])
            
            # Find consistent rules
            for (attr, value), changes_list in grouped.items():
                if self._changes_consistent(changes_list):
                    # Create conditional rule
                    condition = AttributeCondition(attr, LogicalOperator.EQUALS, value)
                    actions = self._create_actions_from_changes(changes_list[0])
                    
                    if actions:
                        rule = CompositeRule(
                            name=f"conditional_{attr.value}_{value}",
                            conditions=[condition],
                            actions=actions,
                            logical_op=LogicalOperator.IF_THEN
                        )
                        
                        self.learned_rules.append(rule)
                        return rule
        
        return None
    
    def _compute_changes(self, obj1: Dict, obj2: Dict) -> Dict[str, Any]:
        """Compute what changed between two objects."""
        changes = {}
        
        for key in obj1:
            if key in obj2 and key != "mask":
                if obj1[key] != obj2[key]:
                    changes[key] = (obj1[key], obj2[key])
        
        return changes
    
    def _changes_consistent(self, changes_list: List[Dict]) -> bool:
        """Check if changes are consistent across examples."""
        if not changes_list:
            return False
        
        first = changes_list[0]
        for changes in changes_list[1:]:
            if changes != first:
                return False
        
        return True
    
    def _create_actions_from_changes(self, changes: Dict) -> List[Tuple[Attribute, Callable]]:
        """Create action functions from observed changes."""
        actions = []
        
        for key, (old_val, new_val) in changes.items():
            # Find corresponding attribute
            for attr in Attribute:
                if attr.value == key:
                    # Create transformation function
                    def transform(val, target=new_val):
                        return target
                    
                    actions.append((attr, transform))
                    break
        
        return actions
    
    def compose_rules(self, rules: List[CompositeRule]) -> CompositeRule:
        """Compose multiple rules into a single complex rule."""
        
        if not rules:
            return None
        
        if len(rules) == 1:
            return rules[0]
        
        # Combine conditions and actions
        all_conditions = []
        all_actions = []
        
        for rule in rules:
            all_conditions.extend(rule.conditions)
            all_actions.extend(rule.actions)
        
        composed = CompositeRule(
            name=f"composed_{'_'.join(r.name for r in rules)}",
            conditions=all_conditions,
            actions=all_actions,
            logical_op=LogicalOperator.AND
        )
        
        return composed
    
    def explain_rule(self, rule: CompositeRule) -> str:
        """Generate human-readable explanation of a rule."""
        
        explanation = f"Rule: {rule.name}\n"
        
        if rule.conditions:
            explanation += "IF "
            cond_strs = []
            for cond in rule.conditions:
                cond_strs.append(
                    f"{cond.attribute.value} {cond.operator.value} {cond.value}"
                )
            
            if rule.logical_op == LogicalOperator.AND:
                explanation += " AND ".join(cond_strs)
            else:
                explanation += " OR ".join(cond_strs)
            
            explanation += "\n"
        
        if rule.actions:
            explanation += "THEN "
            action_strs = []
            for attr, _ in rule.actions:
                action_strs.append(f"modify {attr.value}")
            
            explanation += ", ".join(action_strs)
            explanation += "\n"
        
        return explanation