#!/usr/bin/env python3
"""Multi-Force Physics Rule Extractor - Extended Stage 1 for compositional physics.

Extends the physics rule extractor to handle multiple force types including:
- Gravitational forces
- Magnetic forces (charged particles)
- Electric forces (static charges)
- Spring forces (Hooke's law)
- Custom forces with mathematical expressions

Key insight: Compositional physics (gravity + magnetic) is like compositional operators (AND + THEN).
"""

from utils.imports import setup_project_paths

setup_project_paths()

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class Force:
    """Represents a force with its parameters and scope."""

    force_type: str  # "gravity", "magnetic", "electric", "spring", "custom"
    parameters: Dict[str, Union[float, str]]  # Force-specific parameters
    active: bool = True  # Whether force is active
    temporal_scope: Optional[Tuple[float, float]] = None  # Time range when active
    object_scope: Optional[List[str]] = None  # Which objects affected


@dataclass
class PhysicsEnvironment:
    """Complete physics environment with multiple forces."""

    forces: List[Force]  # All active forces
    global_params: Dict[str, float]  # Temperature, pressure, etc.
    reference_frame: str = "inertial"  # inertial, rotating, accelerating
    dimensions: int = 2  # 2D or 3D physics


class MultiForcePhysicsExtractor:
    """Extracts multiple simultaneous forces and their interactions.

    Handles compositional physics like:
    - "Add gravity and magnetic field"
    - "Underwater physics with electric charges"
    - "Rotating frame with springs"
    """

    def __init__(self):
        # Force templates with default parameters
        self.force_templates = {
            "gravity": {
                "g": 9.8,  # m/s²
                "direction": [0, -1],  # Downward in 2D
            },
            "magnetic": {
                "B": 1.0,  # Tesla
                "direction": [0, 0, 1],  # Into page for 2D
                "charge_dependent": True,
            },
            "electric": {
                "E": 1000.0,  # V/m
                "direction": [1, 0],  # Horizontal field
                "charge_dependent": True,
            },
            "spring": {
                "k": 100.0,  # N/m
                "rest_length": 1.0,  # meters
                "damping": 0.1,
            },
            "friction": {
                "mu_static": 0.3,
                "mu_kinetic": 0.2,
                "surface_normal": [0, 1],
            },
            "drag": {
                "coefficient": 0.47,  # Sphere drag coefficient
                "fluid_density": 1.225,  # Air at sea level kg/m³
                "cross_section": 0.01,  # m²
            },
        }

        # Force combination patterns
        self.combination_patterns = [
            r"add (\w+) and (\w+)",
            r"combine (\w+) with (\w+)",
            r"(\w+) plus (\w+)",
            r"both (\w+) and (\w+)",
            r"(\w+) together with (\w+)",
        ]

        # Force modification patterns
        self.force_patterns = {
            "gravity": [
                r"gravity\s*=\s*([\d.]+)",
                r"set gravity to ([\d.]+)",
                r"gravitational field of ([\d.]+)",
                r"g\s*=\s*([\d.]+)",
            ],
            "magnetic": [
                r"magnetic field\s*=\s*([\d.]+)\s*(?:T|tesla)?",
                r"B\s*=\s*([\d.]+)",
                r"add magnetic field of ([\d.]+)",
                r"magnetism of ([\d.]+)",
            ],
            "electric": [
                r"electric field\s*=\s*([\d.]+)\s*(?:V/m)?",
                r"E\s*=\s*([\d.]+)",
                r"voltage gradient of ([\d.]+)",
                r"add electric field",
            ],
            "spring": [
                r"spring constant\s*=\s*([\d.]+)",
                r"k\s*=\s*([\d.]+)",
                r"stiffness of ([\d.]+)",
                r"add springs? with k\s*=\s*([\d.]+)",
            ],
        }

        # Preset environments with multiple forces
        self.environments = {
            "charged_particles": [
                Force("gravity", {"g": 9.8, "direction": [0, -1]}),
                Force("electric", {"E": 1000.0, "direction": [1, 0]}),
            ],
            "magnetic_pendulum": [
                Force("gravity", {"g": 9.8, "direction": [0, -1]}),
                Force("magnetic", {"B": 0.5, "direction": [0, 0, 1]}),
                Force("spring", {"k": 50.0, "rest_length": 1.0}),
            ],
            "plasma": [
                Force("electric", {"E": 5000.0, "direction": [1, 0]}),
                Force("magnetic", {"B": 2.0, "direction": [0, 0, 1]}),
            ],
            "underwater_magnets": [
                Force("gravity", {"g": 7.0, "direction": [0, -1]}),  # Buoyancy effect
                Force("drag", {"coefficient": 1.2, "fluid_density": 1000}),
                Force("magnetic", {"B": 1.5, "direction": [0, 0, 1]}),
            ],
            "space_tether": [
                Force("gravity", {"g": 0.01, "direction": [0, -1]}),  # Microgravity
                Force("spring", {"k": 200.0, "rest_length": 10.0}),
            ],
        }

        # Time-varying force patterns
        self.time_varying_patterns = [
            (r"(\w+) oscillates with period ([\d.]+)", "{base} * sin(2*pi*t/{period})"),
            (r"(\w+) increases linearly", "{base} * (1 + 0.1*t)"),
            (r"(\w+) decays exponentially", "{base} * exp(-0.1*t)"),
            (r"pulsing (\w+)", "{base} * (1 + 0.5*sin(2*pi*t))"),
            (r"rotating (\w+) field", "rotating"),  # Special case
        ]

    def extract(self, command: str) -> PhysicsEnvironment:
        """Extract complete physics environment from natural language.

        Args:
            command: Natural language like "Add gravity and magnetic field"

        Returns:
            PhysicsEnvironment with all forces and parameters
        """
        command_lower = command.lower()
        forces = []

        # Check for preset environments
        for env_name, env_forces in self.environments.items():
            if env_name.replace("_", " ") in command_lower:
                # Deep copy to avoid modifying templates
                import copy

                forces.extend([copy.deepcopy(f) for f in env_forces])

        # Extract individual forces from patterns
        for force_type, patterns in self.force_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, command_lower)
                if match:
                    force = self._create_force(force_type, match)
                    if force and not self._force_exists(forces, force_type):
                        forces.append(force)

        # Also check for simple mentions of force types
        force_keywords = {
            "magnetic": ["magnetic", "magnet"],
            "electric": ["electric"],
            "spring": ["spring"],
            "friction": ["friction"],
        }

        for force_type, keywords in force_keywords.items():
            for keyword in keywords:
                if keyword in command_lower and not self._force_exists(
                    forces, force_type
                ):
                    # Check it's not negated
                    if f"no {keyword}" not in command_lower:
                        forces.append(self._create_default_force(force_type))

        # Check for force combinations
        for pattern in self.combination_patterns:
            matches = re.findall(pattern, command_lower)
            for match in matches:
                force1 = self._normalize_force_name(match[0])
                force2 = self._normalize_force_name(match[1])

                if force1 and not self._force_exists(forces, force1):
                    forces.append(self._create_default_force(force1))
                if force2 and not self._force_exists(forces, force2):
                    forces.append(self._create_default_force(force2))

        # Handle special keywords that weren't caught
        if "negative gravity" in command_lower and not self._force_exists(
            forces, "gravity"
        ):
            force = Force("gravity", {"g": -9.8, "direction": [0, 1]})  # Upward
            forces.append(force)

        # Extract time-varying modifications
        forces = self._apply_time_varying(command_lower, forces)

        # Extract reference frame
        reference_frame = self._extract_reference_frame(command_lower)

        # If no forces extracted but gravity mentioned, add default gravity
        if not forces and "gravity" in command_lower:
            forces.append(self._create_default_force("gravity"))

        # Default to just gravity if nothing specified
        if not forces:
            forces.append(self._create_default_force("gravity"))

        return PhysicsEnvironment(
            forces=forces,
            global_params=self._extract_global_params(command_lower),
            reference_frame=reference_frame,
            dimensions=3 if "3d" in command_lower else 2,
        )

    def _create_force(self, force_type: str, match: re.Match) -> Optional[Force]:
        """Create a force from regex match."""
        if force_type not in self.force_templates:
            return None

        params = self.force_templates[force_type].copy()

        # Update with extracted value
        if match.groups():
            try:
                value = float(match.group(1))
                if force_type == "gravity":
                    params["g"] = value
                elif force_type == "magnetic":
                    params["B"] = value
                elif force_type == "electric":
                    params["E"] = value
                elif force_type == "spring":
                    params["k"] = value
            except (ValueError, IndexError):
                pass  # Keep default value if parsing fails

        return Force(force_type, params)

    def _create_default_force(self, force_type: str) -> Force:
        """Create force with default parameters."""
        if force_type in self.force_templates:
            return Force(force_type, self.force_templates[force_type].copy())
        return Force(force_type, {})

    def _force_exists(self, forces: List[Force], force_type: str) -> bool:
        """Check if a force type already exists in the list."""
        return any(f.force_type == force_type for f in forces)

    def _normalize_force_name(self, name: str) -> Optional[str]:
        """Normalize force name aliases."""
        name = name.lower().strip()
        aliases = {
            "gravity": ["gravity", "gravitational", "g", "weight"],
            "magnetic": ["magnetic", "magnetism", "b", "magnet"],
            "electric": ["electric", "electrical", "e", "voltage", "charge"],
            "spring": ["spring", "elastic", "hooke", "k"],
            "friction": ["friction", "frictional", "mu"],
            "drag": ["drag", "air resistance", "fluid resistance"],
        }

        for force_type, alias_list in aliases.items():
            if name in alias_list:
                return force_type
        return None

    def _apply_time_varying(self, command: str, forces: List[Force]) -> List[Force]:
        """Apply time-varying modifications to forces."""
        for pattern, template in self.time_varying_patterns:
            matches = re.findall(pattern, command)
            for match in matches:
                if isinstance(match, tuple):
                    force_name = self._normalize_force_name(match[0])
                    if force_name:
                        # Ensure the force exists
                        if not self._force_exists(forces, force_name):
                            forces.append(self._create_default_force(force_name))

                        for force in forces:
                            if force.force_type == force_name:
                                # Make the force time-varying
                                if "oscillates" in pattern:
                                    period = float(match[1])
                                    # Apply to main parameter
                                    param_key = {
                                        "gravity": "g",
                                        "magnetic": "B",
                                        "electric": "E",
                                        "spring": "k",
                                    }.get(force_name)
                                    if param_key and param_key in force.parameters:
                                        base = force.parameters[param_key]
                                        force.parameters[
                                            param_key
                                        ] = f"{base} * sin(2*pi*t/{period})"
                                elif "rotating" in template:
                                    # Special handling for rotating fields
                                    if "direction" in force.parameters:
                                        force.parameters["rotating"] = True
                                        force.parameters["rotation_freq"] = 1.0  # Hz
                                else:
                                    # Apply template to main parameter
                                    for key in ["g", "B", "E", "k"]:
                                        if key in force.parameters:
                                            base = force.parameters[key]
                                            force.parameters[key] = template.format(
                                                base=base
                                            )
                elif isinstance(match, str):
                    # Single group match
                    force_name = self._normalize_force_name(match)
                    if force_name and not self._force_exists(forces, force_name):
                        forces.append(self._create_default_force(force_name))

        # Handle "all forces oscillating"
        if "all forces oscillating" in command:
            period_match = re.search(r"period[s]?\s+(\d+)", command)
            base_period = 2.0
            for i, force in enumerate(forces):
                period = (
                    base_period * (i + 1)
                    if not period_match
                    else float(period_match.group(1))
                )
                param_key = {
                    "gravity": "g",
                    "magnetic": "B",
                    "electric": "E",
                    "spring": "k",
                }.get(force.force_type)
                if param_key and param_key in force.parameters:
                    base = force.parameters[param_key]
                    if isinstance(base, (int, float)):
                        force.parameters[param_key] = f"{base} * sin(2*pi*t/{period})"

        return forces

    def _extract_reference_frame(self, command: str) -> str:
        """Extract reference frame from command."""
        if "rotating" in command and "frame" in command:
            return "rotating"
        elif "accelerating" in command and "frame" in command:
            return "accelerating"
        elif "non-inertial" in command:
            return "non-inertial"
        return "inertial"

    def _extract_global_params(self, command: str) -> Dict[str, float]:
        """Extract global environment parameters."""
        params = {}

        # Temperature
        temp_match = re.search(r"temperature\s*=\s*([\d.]+)", command)
        if temp_match:
            params["temperature"] = float(temp_match.group(1))

        # Pressure
        pressure_match = re.search(r"pressure\s*=\s*([\d.]+)", command)
        if pressure_match:
            params["pressure"] = float(pressure_match.group(1))

        return params

    def describe_environment(self, env: PhysicsEnvironment) -> str:
        """Generate human-readable description of physics environment."""
        desc = []

        desc.append(
            f"Physics Environment ({env.dimensions}D, {env.reference_frame} frame):"
        )
        desc.append(f"Active Forces ({len(env.forces)}):")

        for force in env.forces:
            force_desc = f"  - {force.force_type.capitalize()}:"
            for param, value in force.parameters.items():
                if param != "direction":
                    force_desc += f" {param}={value}"
            desc.append(force_desc)

        if env.global_params:
            desc.append("Global Parameters:")
            for param, value in env.global_params.items():
                desc.append(f"  - {param}: {value}")

        return "\n".join(desc)


def test_multi_force_extraction():
    """Test the multi-force physics extractor."""
    extractor = MultiForcePhysicsExtractor()

    test_cases = [
        # Simple cases
        "Set gravity to 5",
        "Add magnetic field of 2 Tesla",
        # Compositional cases
        "Add gravity and magnetic field",
        "Combine gravity with electric field",
        "Underwater physics with magnetic field",
        # Complex scenarios
        "Charged particles environment",
        "Magnetic pendulum setup",
        "Space tether with springs",
        # Time-varying
        "Gravity oscillates with period 2s",
        "Add gravity and make magnetic field oscillate with period 1s",
        "Pulsing electric field with constant gravity",
        # Reference frames
        "Rotating frame with gravity",
        "Non-inertial frame with springs",
        # Extreme combinations
        "Add gravity = 25, magnetic field = 3T, and springs with k = 500",
        "Underwater magnets with oscillating gravity",
    ]

    print("Multi-Force Physics Extraction Tests")
    print("=" * 60)

    for i, command in enumerate(test_cases, 1):
        print(f"\nTest {i}: '{command}'")
        env = extractor.extract(command)
        print(extractor.describe_environment(env))
        print("-" * 40)

    # Test TRUE OOD scenarios
    print("\n" + "=" * 60)
    print("TRUE OOD Physics Tests (Compositional)")
    print("=" * 60)

    ood_tests = [
        "Gravity = 100 with magnetic field = 10T",  # Extreme parameters
        "Negative gravity with attractive electric field",  # Causal reversal
        "All forces oscillating with different periods",  # Complex time-varying
        "5D physics with magnetic and electric fields",  # Novel dimensions
    ]

    for i, command in enumerate(ood_tests, 1):
        print(f"\nOOD Test {i}: '{command}'")
        env = extractor.extract(command)
        print(extractor.describe_environment(env))
        print("-" * 40)


if __name__ == "__main__":
    test_multi_force_extraction()
