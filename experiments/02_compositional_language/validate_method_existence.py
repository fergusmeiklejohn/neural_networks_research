#!/usr/bin/env python3
"""
Enhanced validation that checks method existence at runtime.
This would have caught the ModificationGenerator.load_modifications() error.
"""

import ast
import os
import sys
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"


def validate_script_methods(script_path: str):
    """Validate that all method calls in a script actually exist on their objects"""
    print(f"🔍 Validating method calls in {script_path}...")
    print("=" * 60)

    # Parse the script to find method calls
    with open(script_path, "r") as f:
        tree = ast.parse(f.read())

    # Find all method calls in the script
    method_calls = []

    class MethodCallVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute):
                # This is a method call like obj.method()
                if isinstance(node.func.value, ast.Name):
                    obj_name = node.func.value.id
                    method_name = node.func.attr
                    method_calls.append((obj_name, method_name, node.lineno))
            self.generic_visit(node)

    visitor = MethodCallVisitor()
    visitor.visit(tree)

    print(f"Found {len(method_calls)} method calls to validate\n")

    # Now validate each method exists
    errors = []
    validated = []

    # Import the modules used in the script
    script_dir = Path(script_path).parent
    sys.path.insert(0, str(script_dir))

    # Key objects to check
    validation_map = {
        "tokenizer": {
            "class": "SCANTokenizer",
            "module": "train_progressive_curriculum",
            "methods_to_check": [
                "save",
                "load",
                "save_vocabulary",
                "load_vocabulary",
                "build_vocabulary",
            ],
        },
        "mod_generator": {
            "class": "ModificationGenerator",
            "module": "modification_generator",
            "methods_to_check": [
                "load_modifications",
                "save_modifications",
                "generate_all_modifications",
            ],
        },
        "model": {
            "class": "create_model",
            "module": "models",
            "methods_to_check": [
                "save_weights",
                "load_weights",
                "compile",
                "fit",
                "evaluate",
            ],
        },
        "loader": {
            "class": "SCANDataLoader",
            "module": "scan_data_loader",
            "methods_to_check": ["load_processed_splits", "save_processed_data"],
        },
    }

    # Check each object type
    for obj_name, config in validation_map.items():
        print(f"\n📦 Checking {obj_name} ({config['class']})...")

        try:
            # Import the module
            module = __import__(config["module"])

            # Get the class or function
            if hasattr(module, config["class"]):
                cls = getattr(module, config["class"])

                # For actual classes, instantiate and check methods
                if obj_name in ["tokenizer", "mod_generator", "loader"]:
                    if obj_name == "tokenizer":
                        instance = cls()
                    elif obj_name == "mod_generator":
                        instance = cls(data_dir="data")
                    elif obj_name == "loader":
                        instance = cls(data_dir="data")

                    # Check each method
                    available_methods = [
                        m for m in dir(instance) if not m.startswith("_")
                    ]
                    print(f"  Available methods: {available_methods}")

                    for method in config["methods_to_check"]:
                        if hasattr(instance, method):
                            validated.append(f"{obj_name}.{method}()")
                            print(f"  ✓ {method}() exists")
                        else:
                            errors.append(f"{obj_name}.{method}() - Method not found!")
                            print(f"  ❌ {method}() NOT FOUND!")

                # For model, just check it's callable
                elif obj_name == "model":
                    if callable(cls):
                        print(f"  ✓ create_model is callable")
                        # We can't check model methods without creating it
                        print(f"  ℹ️  Model methods will be available after creation")
            else:
                errors.append(f"{config['class']} not found in {config['module']}")

        except ImportError as e:
            errors.append(f"Cannot import {config['module']}: {e}")
        except Exception as e:
            errors.append(f"Error checking {obj_name}: {e}")

    # Check specific method calls from the script
    print(f"\n🔎 Validating specific method calls from script...")
    script_specific_errors = []

    for obj_name, method_name, line_no in method_calls:
        if obj_name in validation_map:
            # We already checked these above
            if (
                f"{obj_name}.{method_name}()" not in validated
                and f"{obj_name}.{method_name}() - Method not found!" in errors
            ):
                script_specific_errors.append(
                    f"Line {line_no}: {obj_name}.{method_name}()"
                )

    # Report results
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    if errors:
        print("\n❌ ERRORS FOUND:")
        for error in errors:
            print(f"  • {error}")

        if script_specific_errors:
            print("\n❌ SCRIPT-SPECIFIC ERRORS:")
            for error in script_specific_errors:
                print(f"  • {error}")

        print("\n🔧 SUGGESTED FIXES:")
        for error in errors:
            if "tokenizer.save()" in error:
                print("  • Change tokenizer.save() to tokenizer.save_vocabulary()")
            elif "tokenizer.load()" in error:
                print("  • Change tokenizer.load() to tokenizer.load_vocabulary()")
            elif "load_modifications" in error:
                print(
                    "  • Ensure ModificationGenerator has load_modifications() method"
                )
                print("  • Check if modification_generator.py is up to date")
    else:
        print("\n✅ All method calls validated successfully!")

    return len(errors) == 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_method_existence.py <script.py>")
        sys.exit(1)

    script_path = sys.argv[1]
    if not Path(script_path).exists():
        print(f"Error: {script_path} not found")
        sys.exit(1)

    success = validate_script_methods(script_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
