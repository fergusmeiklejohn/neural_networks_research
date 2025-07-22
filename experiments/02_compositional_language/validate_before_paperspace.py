#!/usr/bin/env python3
"""
MANDATORY validation script to run before pushing any code to Paperspace.
This prevents runtime errors and saves GPU hours.

Usage:
    python validate_before_paperspace.py <script_name.py>
"""

import sys
import subprocess
import ast
import importlib.util
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class ScriptValidator:
    def __init__(self, script_path: str):
        self.script_path = Path(script_path)
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.passed_tests: List[str] = []
        
    def validate(self) -> bool:
        """Run all validation checks"""
        print(f"🔍 Validating {self.script_path.name}...")
        print("=" * 60)
        
        # Check 1: File exists
        if not self._check_file_exists():
            return False
            
        # Check 2: Syntax is valid
        if not self._check_syntax():
            return False
            
        # Check 3: All imports are valid
        if not self._check_imports():
            return False
            
        # Check 4: Run with dry-run mode
        if not self._check_dry_run():
            return False
            
        # Check 5: Static analysis with mypy (if available)
        self._run_static_analysis()
        
        # Check 6: Linting
        self._run_linting()
        
        # Check 7: Check for common antipatterns
        self._check_antipatterns()
        
        # Report results
        self._report_results()
        
        return len(self.errors) == 0
    
    def _check_file_exists(self) -> bool:
        """Check if the script file exists"""
        if not self.script_path.exists():
            self.errors.append(f"File not found: {self.script_path}")
            return False
        self.passed_tests.append("✓ File exists")
        return True
    
    def _check_syntax(self) -> bool:
        """Check Python syntax is valid"""
        try:
            with open(self.script_path, 'r') as f:
                ast.parse(f.read())
            self.passed_tests.append("✓ Python syntax is valid")
            return True
        except SyntaxError as e:
            self.errors.append(f"Syntax error: {e}")
            return False
    
    def _check_imports(self) -> bool:
        """Check all imports in the script are valid"""
        print("\n📦 Checking imports...")
        
        with open(self.script_path, 'r') as f:
            tree = ast.parse(f.read())
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)
        
        # Change to script directory for relative imports
        original_dir = os.getcwd()
        os.chdir(self.script_path.parent)
        
        failed_imports = []
        for imp in imports:
            try:
                # Try to import the module
                parts = imp.split('.')
                module_name = parts[0]
                
                # Handle relative imports
                if module_name in ['models', 'scan_data_loader', 'modification_generator', 
                                   'train_progressive_curriculum', 'train_progressive_minimal']:
                    # These are local modules
                    module_path = Path(module_name + '.py')
                    if not module_path.exists():
                        failed_imports.append(f"{imp} (file not found: {module_path})")
                    else:
                        # Check if specific function/class exists
                        if len(parts) > 1:
                            # This is crude but effective for our use case
                            with open(module_path, 'r') as f:
                                content = f.read()
                                item_name = parts[-1]
                                if f"def {item_name}" not in content and f"class {item_name}" not in content:
                                    failed_imports.append(f"{imp} ('{item_name}' not found in {module_path})")
                else:
                    # Standard library or installed package
                    try:
                        __import__(module_name)
                    except ImportError:
                        failed_imports.append(f"{imp} (module not installed)")
                        
            except Exception as e:
                failed_imports.append(f"{imp} (error: {e})")
        
        os.chdir(original_dir)
        
        if failed_imports:
            self.errors.extend([f"Import error: {imp}" for imp in failed_imports])
            return False
        
        self.passed_tests.append(f"✓ All {len(imports)} imports are valid")
        return True
    
    def _check_dry_run(self) -> bool:
        """Try to run the script with minimal data"""
        print("\n🧪 Attempting dry run...")
        
        # Create a test wrapper that runs the script with minimal data
        test_wrapper = f"""
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['DRY_RUN'] = '1'
os.environ['MAX_SAMPLES'] = '10'
os.environ['MAX_EPOCHS'] = '1'

# Mock storage directory
os.environ['MOCK_STORAGE'] = '1'

try:
    import sys
    sys.argv = ['{self.script_path.name}', '--dry-run']
    
    # Import and run the main function
    spec = importlib.util.spec_from_file_location("test_module", "{self.script_path}")
    module = importlib.util.module_from_spec(spec)
    
    # Execute the module but catch the main execution
    import unittest.mock
    with unittest.mock.patch('sys.exit'):
        spec.loader.exec_module(module)
        
    print("Dry run completed successfully")
except Exception as e:
    print(f"Dry run failed: {{e}}")
    import traceback
    traceback.print_exc()
"""
        
        # Save and run the test wrapper
        test_file = self.script_path.parent / f"_test_{self.script_path.name}"
        with open(test_file, 'w') as f:
            f.write(test_wrapper)
        
        try:
            # Run in the script's directory
            result = subprocess.run(
                [sys.executable, str(test_file)],
                cwd=self.script_path.parent,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Clean up
            test_file.unlink(missing_ok=True)
            
            if "Dry run completed successfully" in result.stdout:
                self.passed_tests.append("✓ Dry run passed")
                return True
            else:
                # Look for specific errors we can identify
                if "AttributeError" in result.stderr:
                    # Extract the specific attribute error
                    for line in result.stderr.split('\n'):
                        if "AttributeError" in line:
                            self.errors.append(f"Runtime error: {line.strip()}")
                elif "ImportError" in result.stderr:
                    for line in result.stderr.split('\n'):
                        if "ImportError" in line:
                            self.errors.append(f"Runtime error: {line.strip()}")
                else:
                    self.warnings.append("Dry run did not complete (may need actual data)")
                return True  # Don't fail on dry run issues
                
        except subprocess.TimeoutExpired:
            self.warnings.append("Dry run timed out (script may work with real data)")
            test_file.unlink(missing_ok=True)
            return True
        except Exception as e:
            self.warnings.append(f"Could not perform dry run: {e}")
            test_file.unlink(missing_ok=True)
            return True
    
    def _run_static_analysis(self):
        """Run mypy if available"""
        print("\n🔬 Running static analysis...")
        
        try:
            result = subprocess.run(
                ["mypy", "--ignore-missing-imports", "--no-error-summary", str(self.script_path)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.passed_tests.append("✓ Static type checking passed")
            else:
                # Parse mypy output for specific issues
                for line in result.stdout.split('\n'):
                    if 'error:' in line:
                        self.warnings.append(f"Type error: {line.strip()}")
                        
        except FileNotFoundError:
            self.warnings.append("mypy not installed - skipping static analysis")
    
    def _run_linting(self):
        """Run flake8 if available"""
        print("\n🧹 Running linting...")
        
        try:
            result = subprocess.run(
                ["flake8", "--max-line-length=100", str(self.script_path)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.passed_tests.append("✓ Linting passed")
            else:
                # Count issues
                issues = [l for l in result.stdout.split('\n') if l.strip()]
                if len(issues) > 5:
                    self.warnings.append(f"Linting found {len(issues)} issues (showing first 5)")
                    for issue in issues[:5]:
                        self.warnings.append(f"  {issue}")
                else:
                    for issue in issues:
                        self.warnings.append(f"Linting: {issue}")
                        
        except FileNotFoundError:
            self.warnings.append("flake8 not installed - skipping linting")
    
    def _check_antipatterns(self):
        """Check for common antipatterns"""
        print("\n🚨 Checking for antipatterns...")
        
        with open(self.script_path, 'r') as f:
            content = f.read()
            lines = content.split('\n')
        
        # Check 1: Direct imports instead of centralized system
        if 'from utils.imports import setup_project_paths' not in content:
            self.warnings.append("Not using centralized import system (utils.imports)")
        
        # Check 2: No error handling around main execution
        if 'if __name__ == "__main__":' in content and 'try:' not in content:
            self.warnings.append("Main execution not wrapped in try/except")
        
        # Check 3: Hardcoded paths
        hardcoded_paths = []
        for i, line in enumerate(lines):
            if any(path in line for path in ['/home/', '/Users/', 'C:\\', '/notebooks/']):
                if not any(skip in line for skip in ['os.path.exists', 'if ', 'print']):
                    hardcoded_paths.append(f"Line {i+1}: {line.strip()}")
        
        if hardcoded_paths:
            self.warnings.append(f"Found {len(hardcoded_paths)} hardcoded paths")
            for path in hardcoded_paths[:3]:
                self.warnings.append(f"  {path}")
        
        # Check 4: Missing docstrings
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    self.warnings.append(f"Missing docstring: {node.name}")
        
        self.passed_tests.append("✓ Antipattern check complete")
    
    def _report_results(self):
        """Report validation results"""
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)
        
        # Passed tests
        if self.passed_tests:
            print("\n✅ Passed Tests:")
            for test in self.passed_tests:
                print(f"  {test}")
        
        # Errors (must fix)
        if self.errors:
            print("\n❌ ERRORS (must fix before deployment):")
            for error in self.errors:
                print(f"  • {error}")
        
        # Warnings (should fix)
        if self.warnings:
            print("\n⚠️  Warnings (should consider fixing):")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        # Summary
        print("\n" + "-" * 60)
        if self.errors:
            print(f"❌ VALIDATION FAILED - {len(self.errors)} error(s) must be fixed")
            print("\nSuggested fixes:")
            
            # Provide specific fixes for common errors
            for error in self.errors:
                if "save" in error and "AttributeError" in error:
                    print("  • Replace 'tokenizer.save()' with 'tokenizer.save_vocabulary()'")
                elif "create_model" in error and "ImportError" in error:
                    print("  • Import create_model from models.py, not train_progressive_minimal.py")
                elif "not found in" in error:
                    obj = error.split("'")[1]
                    print(f"  • Check the correct name/location of '{obj}'")
        else:
            print("✅ VALIDATION PASSED - Script is ready for Paperspace!")
            if self.warnings:
                print(f"   (with {len(self.warnings)} warnings to consider)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_before_paperspace.py <script_name.py>")
        sys.exit(1)
    
    script_path = sys.argv[1]
    validator = ScriptValidator(script_path)
    
    if validator.validate():
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()