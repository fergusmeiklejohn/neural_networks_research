#!/usr/bin/env python3
"""Test time-varying physics extraction fix."""

from utils.imports import setup_project_paths

setup_project_paths()

from physics_rule_extractor import PhysicsRuleExtractor


def test_time_varying_extraction():
    """Test that time-varying physics commands are properly extracted."""
    extractor = PhysicsRuleExtractor()

    test_commands = [
        "gravity oscillates with period 2s",
        "gravity increases over time",
        "set gravity to 5 and make it oscillate with period 3s",
        "underwater physics with gravity oscillating with period 1s",
        "friction decreases over time",
        "gravity follows sin(t)",
    ]

    print("Testing Time-Varying Physics Extraction")
    print("=" * 60)

    for command in test_commands:
        print(f"\nCommand: '{command}'")

        # Extract physics
        parameters, modifications = extractor.extract(command)

        # Check for time-varying
        time_varying_found = False
        for param in parameters:
            if isinstance(param.value, str) and (
                "sin" in param.value or "t" in param.value
            ):
                time_varying_found = True
                print(f"✅ Time-varying {param.name}: {param.value}")

        # Also check modifications
        for mod in modifications:
            if mod.operation == "time_varying":
                print(f"   Modification tracked: {mod.parameter} = {mod.value}")

        if not time_varying_found:
            print("❌ No time-varying physics extracted")

        # Show all parameters
        print("   All parameters:")
        for param in parameters:
            print(f"   - {param.name} = {param.value} {param.unit}")


def test_two_stage_compiler():
    """Test the full Two-Stage Compiler with time-varying physics."""
    from two_stage_physics_compiler import TwoStagePhysicsCompiler

    print("\n\nTesting Two-Stage Compiler with Time-Varying Physics")
    print("=" * 60)

    model = TwoStagePhysicsCompiler()

    test_cases = [
        {
            "command": "gravity oscillates with period 2s",
            "expected": "9.8 * (sin(2*pi*t/2.0))",
        },
        {
            "command": "set gravity to 5 m/s²",
            "expected": "5.0",
        },
        {
            "command": "gravity increases over time",
            "expected": "9.8 * (t)",
        },
    ]

    for test in test_cases:
        print(f"\nCommand: '{test['command']}'")
        print(f"Expected: {test['expected']}")

        # Extract physics context
        context = model.extract_physics_rules(test["command"])

        # Check at different timesteps
        for t in [0.0, 1.0, 2.0]:
            params = context.get_active_parameters(t)
            print(f"  t={t}: gravity = {params.get('gravity', 'N/A')}")


if __name__ == "__main__":
    test_time_varying_extraction()
    test_two_stage_compiler()
