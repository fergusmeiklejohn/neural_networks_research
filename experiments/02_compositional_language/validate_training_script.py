#!/usr/bin/env python3
"""
Comprehensive validation for training scripts before cloud deployment.
Catches method existence errors, import issues, and common pitfalls.
"""

import ast
import importlib
import os
import re
import sys
from pathlib import Path


class ValidationError(Exception):
    """Raised when validation fails"""


class TrainingScriptValidator:
    """Validate training scripts for common issues"""

    def __init__(self, script_path: str):
        self.script_path = Path(script_path)
        self.script_dir = self.script_path.parent
        self.issues = []
        self.warnings = []

    def validate(self) -> bool:
        """Run all validation checks"""
        print(f"🔍 Validating {self.script_path.name}")
        print("=" * 60)

        # Read script content
        with open(self.script_path, "r") as f:
            self.content = f.read()

        # Parse AST
        try:
            self.tree = ast.parse(self.content)
        except SyntaxError as e:
            self.issues.append(f"Syntax Error: {e}")
            return False

        # Run validation checks
        self._check_imports()
        self._check_method_calls()
        self._check_file_operations()
        self._check_common_pitfalls()
        self._check_error_handling()
        self._check_gpu_memory()

        # Report results
        self._report_results()

        return len(self.issues) == 0

    def _check_imports(self):
        """Validate all imports exist and are accessible"""
        print("\n1️⃣ Checking imports...")

        # Change to script directory for relative imports
        original_dir = os.getcwd()
        os.chdir(self.script_dir)

        try:
            for node in ast.walk(self.tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        try:
                            importlib.import_module(alias.name)
                        except ImportError:
                            self.issues.append(
                                f"Import Error: Cannot import '{alias.name}'"
                            )

                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module or ""
                    try:
                        module = importlib.import_module(module_name)
                        for alias in node.names:
                            if not hasattr(module, alias.name):
                                self.issues.append(
                                    f"Import Error: Cannot import '{alias.name}' from '{module_name}'"
                                )
                    except ImportError:
                        self.issues.append(
                            f"Import Error: Cannot import module '{module_name}'"
                        )

            if not self.issues:
                print("✓ All imports valid")

        finally:
            os.chdir(original_dir)

    def _check_method_calls(self):
        """Validate method calls on objects"""
        print("\n2️⃣ Checking method calls...")

        # Common patterns to check
        patterns = [
            (
                r"(\w+)\.load_modifications\(\)",
                "ModificationGenerator",
                "modification_generator",
            ),
            (r"tokenizer\.save\(", "SCANTokenizer", "train_progressive_curriculum"),
            (r"model\.save_weights\(", None, None),  # Keras method, should exist
        ]

        for pattern, expected_class, module_name in patterns:
            matches = re.findall(pattern, self.content)
            if matches and expected_class:
                # Try to validate the method exists
                try:
                    # Import the module
                    if module_name:
                        original_dir = os.getcwd()
                        os.chdir(self.script_dir)
                        try:
                            module = importlib.import_module(module_name)
                            cls = getattr(module, expected_class)

                            # Check if method exists
                            method_name = (
                                pattern.split("\\.")[-1]
                                .replace("\\(", "")
                                .replace(")", "")
                            )
                            if not hasattr(cls, method_name):
                                self.issues.append(
                                    f"Method Error: {expected_class}.{method_name}() doesn't exist"
                                )
                            else:
                                print(f"✓ {expected_class}.{method_name}() exists")
                        finally:
                            os.chdir(original_dir)
                except Exception as e:
                    self.warnings.append(
                        f"Could not validate {expected_class}.{method_name}: {e}"
                    )

    def _check_file_operations(self):
        """Check file paths and operations"""
        print("\n3️⃣ Checking file operations...")

        # Check for hardcoded paths
        hardcoded_paths = re.findall(
            r'["\']\/(?:home|Users)\/[^"\']+["\']', self.content
        )
        if hardcoded_paths:
            self.issues.append(f"Hardcoded paths found: {hardcoded_paths}")

        # Check for proper path handling
        if "Path(" not in self.content and "os.path" not in self.content:
            self.warnings.append("No Path or os.path usage - may have path issues")

        # Check for pickle operations
        if "pickle.load" in self.content:
            if "rb" not in self.content:
                self.issues.append("pickle.load without 'rb' mode")
            print("✓ Uses pickle.load with binary mode")

        # Check for storage directory validation
        if "/storage" in self.content:
            if "os.path.exists('/storage')" not in self.content:
                self.warnings.append("Uses /storage but doesn't check if it exists")
            else:
                print("✓ Validates /storage existence")

    def _check_common_pitfalls(self):
        """Check for common issues"""
        print("\n4️⃣ Checking common pitfalls...")

        # Tokenizer save method
        if "tokenizer.save(" in self.content:
            self.issues.append(
                "Uses tokenizer.save() - should be tokenizer.save_vocabulary()"
            )

        # Mixed precision issues
        if "mixed_float16" in self.content:
            self.warnings.append("Uses mixed precision - may cause NaN issues")

        # Memory settings
        if (
            "TF_GPU_ALLOCATOR" not in self.content
            and "tensorflow" in self.content.lower()
        ):
            self.warnings.append("No TF_GPU_ALLOCATOR setting - may have memory issues")

        # Batch size
        batch_sizes = re.findall(r"batch_size\s*=\s*(\d+)", self.content)
        for size in batch_sizes:
            if int(size) > 64:
                self.warnings.append(f"Large batch size ({size}) may cause OOM")

    def _check_error_handling(self):
        """Check for proper error handling"""
        print("\n5️⃣ Checking error handling...")

        try_count = self.content.count("try:")
        self.content.count("except")

        if try_count == 0:
            self.warnings.append("No try/except blocks for error handling")
        else:
            print(f"✓ Found {try_count} try/except blocks")

        # Check for main guard
        if "__main__" not in self.content:
            self.warnings.append("No if __name__ == '__main__' guard")

    def _check_gpu_memory(self):
        """Check for GPU memory settings"""
        print("\n6️⃣ Checking GPU settings...")

        if "CUDA_VISIBLE_DEVICES" in self.content:
            print("✓ Sets CUDA_VISIBLE_DEVICES")

        if "gpu_options" in self.content or "memory_growth" in self.content:
            print("✓ Configures GPU memory")
        elif "tensorflow" in self.content.lower():
            self.warnings.append("No explicit GPU memory configuration")

    def _report_results(self):
        """Report validation results"""
        print("\n" + "=" * 60)

        if self.issues:
            print("❌ VALIDATION FAILED")
            print("\n🚨 Critical Issues (must fix):")
            for issue in self.issues:
                print(f"  - {issue}")

        if self.warnings:
            print("\n⚠️  Warnings (should review):")
            for warning in self.warnings:
                print(f"  - {warning}")

        if not self.issues and not self.warnings:
            print("✅ All validation checks passed!")

        # Provide specific recommendations
        if self.issues or self.warnings:
            print("\n📋 Recommendations:")

            if any("load_modifications" in issue for issue in self.issues):
                print("  1. Add load_modifications() method to ModificationGenerator")
                print("     or use direct pickle loading like other scripts")

            if any("tokenizer.save" in issue for issue in self.issues):
                print("  2. Replace tokenizer.save() with tokenizer.save_vocabulary()")

            if any("/storage" in warning for warning in self.warnings):
                print("  3. Add validation: if os.path.exists('/storage'):")

            if any("memory" in warning for warning in self.warnings):
                print("  4. Add GPU memory configuration for TensorFlow")


def main():
    """Main validation entry point"""
    if len(sys.argv) < 2:
        print("Usage: python validate_training_script.py <script.py>")
        sys.exit(1)

    script_path = sys.argv[1]
    if not os.path.exists(script_path):
        print(f"Error: Script not found: {script_path}")
        sys.exit(1)

    validator = TrainingScriptValidator(script_path)
    success = validator.validate()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
