#!/usr/bin/env python3
"""Mathematical Operator Extractor - Explicit extraction for mathematical rules.

Extends our Two-Stage Compiler approach to mathematical operations, demonstrating
that distribution invention works for abstract mathematical structures.

Key insight: "Make multiplication non-commutative" is like "X means jump"
but for mathematical operators.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Union


@dataclass
class MathOperator:
    """Represents a mathematical operator with its properties."""

    symbol: str  # "+", "*", "⊕", "⊗", etc.
    name: str  # "addition", "multiplication", "custom_op"
    properties: Dict[str, Union[bool, str, float]]  # commutative, associative, etc.
    operation: Optional[str] = None  # Mathematical expression like "a*b + b*a"
    identity: Optional[float] = None  # Identity element
    inverse: Optional[str] = None  # Inverse operation


@dataclass
class MathEnvironment:
    """Complete mathematical environment with modified operators."""

    operators: Dict[str, MathOperator]  # Symbol -> Operator mapping
    dimension: int = 2  # Working in R^n
    field: str = "real"  # real, complex, quaternion, custom
    constraints: List[str] = None  # Additional constraints


class MathematicalOperatorExtractor:
    """Extracts mathematical operator modifications from natural language.

    Handles commands like:
    - "Make multiplication non-commutative"
    - "Define ⊕ as rotation by 90 degrees"
    - "Addition distributes over multiplication"
    - "Create group with generators a, b where a²=b²=1"
    """

    def __init__(self):
        # Standard operators with default properties
        self.standard_operators = {
            "+": MathOperator(
                symbol="+",
                name="addition",
                properties={
                    "commutative": True,
                    "associative": True,
                    "distributive_over": None,
                },
                operation="a + b",
                identity=0,
                inverse="-",
            ),
            "*": MathOperator(
                symbol="*",
                name="multiplication",
                properties={
                    "commutative": True,
                    "associative": True,
                    "distributive_over": "+",
                },
                operation="a * b",
                identity=1,
                inverse="/",
            ),
            "-": MathOperator(
                symbol="-",
                name="subtraction",
                properties={
                    "commutative": False,
                    "associative": False,
                },
                operation="a - b",
            ),
            "/": MathOperator(
                symbol="/",
                name="division",
                properties={
                    "commutative": False,
                    "associative": False,
                },
                operation="a / b",
            ),
        }

        # Property modification patterns
        self.property_patterns = {
            "commutative": [
                r"make (\w+) non-?commutative",
                r"(\w+) doesn't commute",
                r"(\w+) is non-?commutative",
                r"ab ≠ ba for (\w+)",
            ],
            "associative": [
                r"make (\w+) non-?associative",
                r"(\w+) is not associative",
                r"\(ab\)c ≠ a\(bc\) for (\w+)",
            ],
            "distributive": [
                r"(\w+) distributes over (\w+)",
                r"make (\w+) distributive",
                r"(\w+) is distributive over (\w+)",
            ],
        }

        # Custom operator definition patterns
        self.definition_patterns = [
            r"define ([⊕⊗⊙◇△▽★☆]) as (.*)",
            r"let ([⊕⊗⊙◇△▽★☆]) be (.*)",
            r"([⊕⊗⊙◇△▽★☆]) means (.*)",
            r"new operator ([⊕⊗⊙◇△▽★☆]) defined as (.*)",
        ]

        # Mathematical structure patterns
        self.structure_patterns = {
            "group": r"create group with (.*)",
            "ring": r"create ring with (.*)",
            "field": r"create field with (.*)",
            "quaternions": r"use quaternion multiplication",
            "matrices": r"use (\d+)x(\d+) matrix multiplication",
            "modular": r"work modulo (\d+)",
        }

        # Preset mathematical environments
        self.environments = {
            "quaternions": MathEnvironment(
                operators={
                    "*": MathOperator(
                        symbol="*",
                        name="quaternion_multiplication",
                        properties={
                            "commutative": False,
                            "associative": True,
                        },
                        operation="quaternion_mult(a, b)",
                    )
                },
                dimension=4,
                field="quaternion",
            ),
            "matrices_2x2": MathEnvironment(
                operators={
                    "*": MathOperator(
                        symbol="*",
                        name="matrix_multiplication",
                        properties={
                            "commutative": False,
                            "associative": True,
                        },
                        operation="matrix_mult(a, b)",
                    )
                },
                dimension=4,
                field="real",
            ),
            "modular_arithmetic": MathEnvironment(
                operators={
                    "+": MathOperator(
                        symbol="+",
                        name="modular_addition",
                        properties={
                            "commutative": True,
                            "associative": True,
                        },
                        operation="(a + b) % n",
                    ),
                    "*": MathOperator(
                        symbol="*",
                        name="modular_multiplication",
                        properties={
                            "commutative": True,
                            "associative": True,
                        },
                        operation="(a * b) % n",
                    ),
                },
                field="modular",
            ),
        }

    def extract(self, command: str) -> MathEnvironment:
        """Extract mathematical environment from natural language command.

        Args:
            command: Natural language like "Make multiplication non-commutative"

        Returns:
            MathEnvironment with modified operators
        """
        command_lower = command.lower()

        # Start with standard operators
        import copy

        operators = copy.deepcopy(self.standard_operators)

        # Check for preset environments
        for env_name, env in self.environments.items():
            if env_name.replace("_", " ") in command_lower:
                return env

        # Extract property modifications
        for prop_name, patterns in self.property_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, command_lower)
                if match:
                    op_name = self._normalize_operator_name(match.group(1))
                    if op_name in operators:
                        if prop_name == "commutative":
                            operators[op_name].properties["commutative"] = False
                            # Update operation to reflect non-commutativity
                            if op_name == "*":
                                operators[op_name].operation = "a*b (≠ b*a)"
                        elif prop_name == "associative":
                            operators[op_name].properties["associative"] = False
                        elif prop_name == "distributive" and len(match.groups()) > 1:
                            over_op = self._normalize_operator_name(match.group(2))
                            operators[op_name].properties["distributive_over"] = over_op

        # Extract custom operator definitions
        for pattern in self.definition_patterns:
            matches = re.findall(pattern, command)
            for match in matches:
                symbol = match[0]
                definition = match[1]
                operators[symbol] = self._create_custom_operator(symbol, definition)

        # Handle specific mathematical structures
        dimension = 2
        field = "real"

        # Quaternions
        if "quaternion" in command_lower:
            operators["*"] = MathOperator(
                symbol="*",
                name="quaternion_multiplication",
                properties={"commutative": False, "associative": True},
                operation="q1*q2 (Hamilton product)",
            )
            dimension = 4
            field = "quaternion"

        # Matrix multiplication
        matrix_match = re.search(r"(\d+)x(\d+) matrix", command_lower)
        if matrix_match or "matrix multiplication" in command_lower:
            operators["*"] = MathOperator(
                symbol="*",
                name="matrix_multiplication",
                properties={"commutative": False, "associative": True},
                operation="A @ B (matrix product)",
            )
            if matrix_match:
                n = int(matrix_match.group(1))
                dimension = n * n

        # Modular arithmetic
        mod_match = re.search(r"modulo (\d+)", command_lower)
        if mod_match:
            n = int(mod_match.group(1))
            for op in ["+", "*"]:
                if op in operators:
                    operators[op].operation = f"({operators[op].operation}) mod {n}"
            field = f"Z/{n}Z"

        # Cross product (3D specific)
        if "cross product" in command_lower or "×" in command:
            operators["×"] = MathOperator(
                symbol="×",
                name="cross_product",
                properties={
                    "commutative": False,
                    "associative": False,
                    "anti_commutative": True,
                },
                operation="a × b = -b × a",
            )
            dimension = 3

        # Extract constraints
        constraints = self._extract_constraints(command)

        return MathEnvironment(
            operators=operators,
            dimension=dimension,
            field=field,
            constraints=constraints,
        )

    def _normalize_operator_name(self, name: str) -> str:
        """Normalize operator name to symbol."""
        name = name.lower().strip()
        mappings = {
            "addition": "+",
            "add": "+",
            "plus": "+",
            "multiplication": "*",
            "multiply": "*",
            "times": "*",
            "product": "*",
            "subtraction": "-",
            "subtract": "-",
            "minus": "-",
            "division": "/",
            "divide": "/",
        }
        return mappings.get(name, name)

    def _create_custom_operator(self, symbol: str, definition: str) -> MathOperator:
        """Create a custom operator from definition."""
        # Parse common patterns
        properties = {
            "commutative": "commutative" in definition,
            "associative": "associative" in definition,
        }

        # Common operations
        if "rotation" in definition:
            angle_match = re.search(r"(\d+)\s*degrees?", definition)
            angle = int(angle_match.group(1)) if angle_match else 90
            operation = f"rotate(a, {angle}°)"
        elif "average" in definition:
            operation = "(a + b) / 2"
        elif "maximum" in definition or "max" in definition:
            operation = "max(a, b)"
        elif "minimum" in definition or "min" in definition:
            operation = "min(a, b)"
        elif "composition" in definition:
            operation = "compose(a, b)"
        else:
            operation = definition

        return MathOperator(
            symbol=symbol,
            name=f"custom_{symbol}",
            properties=properties,
            operation=operation,
        )

    def _extract_constraints(self, command: str) -> List[str]:
        """Extract mathematical constraints from command."""
        constraints = []

        # Generator relations
        gen_match = re.search(r"generators? (\w+(?:,\s*\w+)*) where (.*)", command)
        if gen_match:
            constraints.append(f"Generators: {gen_match.group(1)}")
            constraints.append(f"Relations: {gen_match.group(2)}")

        # Specific constraints
        if "nilpotent" in command:
            constraints.append("Nilpotent elements exist")
        if "idempotent" in command:
            constraints.append("Idempotent elements exist")
        if "zero divisors" in command:
            constraints.append("Has zero divisors")

        return constraints if constraints else None

    def describe_environment(self, env: MathEnvironment) -> str:
        """Generate human-readable description of mathematical environment."""
        desc = []
        desc.append(
            f"Mathematical Environment ({env.field} field, dim={env.dimension}):"
        )
        desc.append("\nOperators:")

        for symbol, op in env.operators.items():
            props = []
            if not op.properties.get("commutative", True):
                props.append("non-commutative")
            if not op.properties.get("associative", True):
                props.append("non-associative")
            if op.properties.get("distributive_over"):
                props.append(f"distributes over {op.properties['distributive_over']}")

            props_str = f" ({', '.join(props)})" if props else ""
            desc.append(f"  {symbol} ({op.name}){props_str}: {op.operation}")

        if env.constraints:
            desc.append("\nConstraints:")
            for constraint in env.constraints:
                desc.append(f"  - {constraint}")

        return "\n".join(desc)


def test_mathematical_extraction():
    """Test mathematical operator extraction."""
    extractor = MathematicalOperatorExtractor()

    test_cases = [
        # Basic modifications
        "Make multiplication non-commutative",
        "Addition is not associative",
        "Multiplication distributes over addition",
        # Custom operators
        "Define ⊕ as rotation by 90 degrees",
        "Let ⊗ be the average of two numbers",
        "New operator ◇ defined as maximum",
        # Mathematical structures
        "Use quaternion multiplication",
        "Work with 3x3 matrix multiplication",
        "Work modulo 7",
        # Complex scenarios
        "Non-commutative multiplication with identity 1",
        "Create group with generators a, b where a²=b²=(ab)³=1",
        "Define ⊕ as composition, make it associative but not commutative",
        # TRUE OOD scenarios
        "Make addition non-commutative and multiplication commutative",
        "All operations are non-associative",
        "Define ⊕ as a*b - b*a (Lie bracket)",
    ]

    print("Mathematical Operator Extraction Tests")
    print("=" * 60)

    for i, command in enumerate(test_cases, 1):
        print(f"\nTest {i}: '{command}'")
        env = extractor.extract(command)
        print(extractor.describe_environment(env))
        print("-" * 40)


def demonstrate_true_ood_math():
    """Demonstrate TRUE OOD mathematical operations."""
    extractor = MathematicalOperatorExtractor()

    print("\n" + "=" * 60)
    print("TRUE OOD Mathematical Operations")
    print("=" * 60)

    ood_commands = [
        "Make all operations non-commutative and non-associative",
        "Define ⊕ as a²+b², ⊗ as 2ab, both non-commutative",
        "Create a ring where multiplication doesn't distribute over addition",
        "Work in 5-dimensional space with quaternion-like multiplication",
    ]

    print("\nThese mathematical structures would be impossible for neural networks")
    print("to handle because they violate fundamental algebraic assumptions:\n")

    for command in ood_commands:
        env = extractor.extract(command)
        print(f"\nCommand: '{command}'")
        print("Result:")

        # Show key violations
        violations = []
        for symbol, op in env.operators.items():
            if not op.properties.get("commutative", True):
                violations.append(f"{symbol} is non-commutative")
            if not op.properties.get("associative", True):
                violations.append(f"{symbol} is non-associative")

        if violations:
            print("  Violations of standard algebra:")
            for v in violations:
                print(f"    - {v}")

        print(f"  Working in: {env.field} field, dimension {env.dimension}")


if __name__ == "__main__":
    test_mathematical_extraction()
    demonstrate_true_ood_math()
