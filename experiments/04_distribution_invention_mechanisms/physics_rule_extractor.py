#!/usr/bin/env python3
"""Physics Rule Extractor - Stage 1 of Two-Stage Physics Compiler.

Extracts explicit physics parameters and modifications from natural language commands,
similar to how we extract variable bindings. This enables distribution invention
for physics laws.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class PhysicsParameter:
    """Represents a physics parameter with temporal scope."""

    name: str  # "gravity", "friction", "elasticity", "damping"
    value: Union[float, str]  # 9.8 or "9.8 * sin(t)"
    unit: str  # "m/s²", "dimensionless", etc.
    context_start: int  # When this parameter becomes active
    context_end: Optional[int] = None  # When it stops (None = forever)


@dataclass
class PhysicsModification:
    """Represents a requested modification to physics."""

    parameter: str  # Which parameter to modify
    operation: str  # "set", "increase", "decrease", "multiply"
    value: Union[float, str]  # Target value or change amount
    scope: str  # "global", "temporal", "object-specific"
    temporal_condition: Optional[str] = None  # "for 2 seconds", "after 1s", etc.


class PhysicsRuleExtractor:
    """Extracts physics rules and modifications from commands.

    Similar to variable binding extraction, but for physics parameters.
    Key insight: "gravity = 9.8" is like "X means jump" at a higher level.
    """

    def __init__(self):
        # Standard physics parameters
        self.physics_params = {
            "gravity": {"unit": "m/s²", "default": 9.8, "aliases": ["g"]},
            "friction": {
                "unit": "",
                "default": 0.3,
                "aliases": ["mu", "friction coefficient"],
            },
            "elasticity": {
                "unit": "",
                "default": 0.8,
                "aliases": ["restitution", "bounciness"],
            },
            "damping": {
                "unit": "",
                "default": 0.99,
                "aliases": ["air resistance", "drag"],
            },
        }

        # Modification patterns
        self.set_patterns = [
            r"set (\w+) to ([\d.]+)",
            r"(\w+) = ([\d.]+)",
            r"make (\w+) ([\d.]+)",
            r"(\w+) is ([\d.]+)",
        ]

        self.increase_patterns = [
            r"increase (\w+) by ([\d.]+)%?",
            r"(\w+) \+= ([\d.]+)",
            r"double (\w+)",  # Special case
            r"triple (\w+)",  # Special case
        ]

        self.decrease_patterns = [
            r"decrease (\w+) by ([\d.]+)%?",
            r"reduce (\w+) by ([\d.]+)%?",
            r"(\w+) -= ([\d.]+)",
            r"half (\w+)",  # Special case
        ]

        self.temporal_patterns = [
            r"for ([\d.]+) seconds?",
            r"after ([\d.]+) seconds?",
            r"between ([\d.]+) and ([\d.]+) seconds?",
            r"every ([\d.]+) seconds?",
        ]

        # Physics scenarios (like compositional operators)
        self.scenarios = {
            "underwater": {
                "gravity": 7.0,  # Reduced effective gravity
                "damping": 0.5,  # High drag
                "friction": 0.8,  # Different friction
            },
            "space": {
                "gravity": 0.1,  # Near zero
                "friction": 0.0,  # No friction
                "damping": 1.0,  # No air resistance
            },
            "moon": {
                "gravity": 1.62,  # Moon gravity
                "friction": 0.3,  # Same friction
                "damping": 1.0,  # No atmosphere
            },
            "ice": {
                "friction": 0.05,  # Very low friction
                "elasticity": 0.95,  # High bounce
            },
        }

    def extract(
        self, command: str, current_state: Optional[Dict[str, float]] = None
    ) -> Tuple[List[PhysicsParameter], List[PhysicsModification]]:
        """Extract physics parameters and modifications from command.

        Args:
            command: Natural language command like "set gravity to 5 m/s²"
            current_state: Optional current physics state for relative modifications

        Returns:
            Tuple of (parameters, modifications)
        """
        command = command.lower()
        parameters = []
        modifications = []

        # Check for predefined scenarios first
        for scenario, params in self.scenarios.items():
            if scenario in command:
                for param_name, value in params.items():
                    modifications.append(
                        PhysicsModification(
                            parameter=param_name,
                            operation="set",
                            value=value,
                            scope="global",
                        )
                    )

        # Extract SET operations
        for pattern in self.set_patterns:
            matches = re.findall(pattern, command)
            for match in matches:
                param = self._normalize_parameter(match[0])
                if param:
                    value = (
                        float(match[1])
                        if match[1].replace(".", "").isdigit()
                        else match[1]
                    )
                    modifications.append(
                        PhysicsModification(
                            parameter=param,
                            operation="set",
                            value=value,
                            scope="global",
                        )
                    )

        # Extract INCREASE operations
        for pattern in self.increase_patterns:
            if "double" in pattern:
                match = re.search(pattern, command)
                if match:
                    param = self._normalize_parameter(match.group(1))
                    if param:
                        modifications.append(
                            PhysicsModification(
                                parameter=param,
                                operation="multiply",
                                value=2.0,
                                scope="global",
                            )
                        )
            elif "triple" in pattern:
                match = re.search(pattern, command)
                if match:
                    param = self._normalize_parameter(match.group(1))
                    if param:
                        modifications.append(
                            PhysicsModification(
                                parameter=param,
                                operation="multiply",
                                value=3.0,
                                scope="global",
                            )
                        )
            else:
                matches = re.findall(pattern, command)
                for match in matches:
                    param = self._normalize_parameter(match[0])
                    if param:
                        value = float(match[1].replace("%", ""))
                        is_percent = "%" in match[1]
                        modifications.append(
                            PhysicsModification(
                                parameter=param,
                                operation="increase",
                                value=value / 100 if is_percent else value,
                                scope="global",
                            )
                        )

        # Extract temporal conditions
        for mod in modifications:
            for pattern in self.temporal_patterns:
                match = re.search(pattern, command)
                if match:
                    mod.temporal_condition = match.group(0)
                    mod.scope = "temporal"

        # Convert modifications to parameters
        if current_state is None:
            current_state = {
                p: info["default"] for p, info in self.physics_params.items()
            }

        parameters = self._apply_modifications(current_state, modifications)

        return parameters, modifications

    def extract_time_varying(self, command: str) -> Optional[PhysicsParameter]:
        """Extract time-varying physics rules like 'gravity oscillates with period 2s'."""
        time_patterns = [
            (r"(\w+) oscillates with period ([\d.]+)", "sin(2*pi*t/{period})"),
            (r"(\w+) increases over time", "t"),
            (r"(\w+) decreases over time", "-t"),
            (r"(\w+) follows sin\(t\)", "sin(t)"),
            (r"(\w+) = (.*)", "{expr}"),  # Direct mathematical expression
        ]

        for pattern, template in time_patterns:
            match = re.search(pattern, command.lower())
            if match:
                param = self._normalize_parameter(match.group(1))
                if param:
                    if "period" in template:
                        period = float(match.group(2))
                        expr = template.format(period=period)
                    elif "{expr}" in template:
                        expr = match.group(2)
                    else:
                        expr = template

                    # Get default value for scaling
                    default = self.physics_params[param]["default"]
                    full_expr = f"{default} * ({expr})"

                    return PhysicsParameter(
                        name=param,
                        value=full_expr,
                        unit=self.physics_params[param]["unit"],
                        context_start=0,
                    )
        return None

    def _normalize_parameter(self, param_str: str) -> Optional[str]:
        """Normalize parameter names to standard form."""
        param_str = param_str.lower().strip()

        # Direct match
        if param_str in self.physics_params:
            return param_str

        # Check aliases
        for param, info in self.physics_params.items():
            if param_str in info["aliases"]:
                return param

        # Fuzzy matching for common variations
        if "grav" in param_str:
            return "gravity"
        elif "fric" in param_str or "mu" in param_str:
            return "friction"
        elif "elast" in param_str or "bounc" in param_str or "restit" in param_str:
            return "elasticity"
        elif "damp" in param_str or "drag" in param_str or "resist" in param_str:
            return "damping"

        return None

    def _apply_modifications(
        self, current_state: Dict[str, float], modifications: List[PhysicsModification]
    ) -> List[PhysicsParameter]:
        """Apply modifications to create parameter list."""
        # Start with current state
        params = []
        state = current_state.copy()

        for mod in modifications:
            if mod.parameter not in state:
                continue

            if mod.operation == "set":
                state[mod.parameter] = mod.value
            elif mod.operation == "increase":
                if isinstance(mod.value, float) and mod.value <= 1:  # Percentage
                    state[mod.parameter] *= 1 + mod.value
                else:
                    state[mod.parameter] += mod.value
            elif mod.operation == "decrease":
                if isinstance(mod.value, float) and mod.value <= 1:  # Percentage
                    state[mod.parameter] *= 1 - mod.value
                else:
                    state[mod.parameter] -= mod.value
            elif mod.operation == "multiply":
                state[mod.parameter] *= mod.value

        # Convert to parameter objects
        for param_name, value in state.items():
            params.append(
                PhysicsParameter(
                    name=param_name,
                    value=value,
                    unit=self.physics_params[param_name]["unit"],
                    context_start=0,
                )
            )

        return params


def test_physics_extractor():
    """Test the physics rule extractor with various commands."""
    extractor = PhysicsRuleExtractor()

    test_commands = [
        # Simple modifications
        "Set gravity to 5 m/s²",
        "Double the friction",
        "Reduce damping by 50%",
        "Make elasticity 0.95",
        # Scenarios
        "Simulate underwater physics",
        "Use moon gravity",
        "Make it like space",
        # Complex commands
        "Set gravity to 5 and double friction",
        "Underwater physics with half gravity",
        # Time-varying (for future)
        "Gravity oscillates with period 2s",
        "Increase gravity over time",
    ]

    print("Testing Physics Rule Extractor\n" + "=" * 50)

    for command in test_commands:
        print(f"\nCommand: '{command}'")

        params, mods = extractor.extract(command)

        print("Modifications:")
        for mod in mods:
            print(f"  - {mod.parameter}: {mod.operation} {mod.value}")

        print("Resulting parameters:")
        for param in params:
            print(f"  - {param.name} = {param.value} {param.unit}")

        # Also check time-varying
        time_param = extractor.extract_time_varying(command)
        if time_param:
            print(f"Time-varying: {time_param.name} = {time_param.value}")


if __name__ == "__main__":
    test_physics_extractor()
