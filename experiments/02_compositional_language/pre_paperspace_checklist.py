#!/usr/bin/env python3
"""
Comprehensive pre-Paperspace deployment checklist.
This ensures ALL code is committed and pushed before deployment.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple

def run_command(cmd: List[str]) -> Tuple[bool, str]:
    """Run a command and return success status and output"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)

def check_git_status() -> Tuple[bool, List[str]]:
    """Check if there are uncommitted changes"""
    print("\n🔍 Checking Git Status...")
    print("-" * 40)
    
    issues = []
    
    # Check for uncommitted changes
    success, output = run_command(["git", "status", "--porcelain"])
    if success and output.strip():
        print("❌ Uncommitted changes found:")
        for line in output.strip().split('\n'):
            print(f"   {line}")
            if line.strip():
                issues.append(f"Uncommitted: {line}")
    else:
        print("✓ Working directory clean")
    
    # Check if we're up to date with origin
    success, output = run_command(["git", "fetch", "origin"])
    if success:
        # Check if local is behind origin
        success, output = run_command(["git", "rev-list", "--count", "HEAD..origin/production"])
        if success and output.strip() != "0":
            commits_behind = output.strip()
            issues.append(f"Local is {commits_behind} commits behind origin/production")
            print(f"❌ Local branch is {commits_behind} commits behind origin")
    
    # Check current branch
    success, output = run_command(["git", "branch", "--show-current"])
    if success:
        current_branch = output.strip()
        print(f"ℹ️  Current branch: {current_branch}")
        if current_branch != "production":
            print(f"⚠️  Warning: Not on production branch (on {current_branch})")
    
    return len(issues) == 0, issues

def check_validation_passes(script_path: str) -> Tuple[bool, List[str]]:
    """Run all validation scripts"""
    print("\n🧪 Running Validation Scripts...")
    print("-" * 40)
    
    issues = []
    validators = [
        ("validate_before_paperspace.py", "Static validation"),
        ("test_script_locally.py", "Runtime testing"),
        ("validate_method_existence.py", "Method existence check")
    ]
    
    for validator, description in validators:
        if Path(validator).exists():
            print(f"\nRunning {description}...")
            success, output = run_command([sys.executable, validator, script_path])
            if success:
                print(f"✓ {description} passed")
            else:
                print(f"❌ {description} failed")
                # Extract key errors from output
                for line in output.split('\n'):
                    if 'ERROR' in line or 'AttributeError' in line or 'ImportError' in line:
                        issues.append(f"{validator}: {line.strip()}")
        else:
            print(f"⚠️  {validator} not found - skipping")
    
    return len(issues) == 0, issues

def check_critical_files_exist() -> Tuple[bool, List[str]]:
    """Check that critical files exist and are not empty"""
    print("\n📁 Checking Critical Files...")
    print("-" * 40)
    
    issues = []
    critical_files = [
        "models.py",
        "scan_data_loader.py",
        "modification_generator.py",
        "train_progressive_curriculum.py",
        "paperspace_train_with_safeguards.py"
    ]
    
    for file in critical_files:
        path = Path(file)
        if not path.exists():
            issues.append(f"Missing file: {file}")
            print(f"❌ Missing: {file}")
        elif path.stat().st_size == 0:
            issues.append(f"Empty file: {file}")
            print(f"❌ Empty: {file}")
        else:
            print(f"✓ {file} ({path.stat().st_size:,} bytes)")
    
    return len(issues) == 0, issues

def check_data_generation() -> Tuple[bool, List[str]]:
    """Check if data has been generated"""
    print("\n💾 Checking Data Generation...")
    print("-" * 40)
    
    issues = []
    data_files = [
        "data/processed/train.pkl",
        "data/processed/modification_pairs.pkl"
    ]
    
    for file in data_files:
        path = Path(file)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✓ {file} ({size_mb:.1f} MB)")
        else:
            # Try to generate if missing
            print(f"⚠️  {file} not found")
            if "train.pkl" in file:
                print("   Run: python scan_data_loader.py")
            elif "modification_pairs.pkl" in file:
                print("   Run: python modification_generator.py")
            issues.append(f"Missing data: {file}")
    
    return len(issues) == 0, issues

def suggest_fixes(all_issues: List[str]):
    """Suggest fixes for common issues"""
    print("\n🔧 Suggested Fixes:")
    print("-" * 40)
    
    # Group issues by type
    uncommitted = [i for i in all_issues if i.startswith("Uncommitted:")]
    missing_files = [i for i in all_issues if "Missing" in i]
    validation_errors = [i for i in all_issues if "Error" in i or "AttributeError" in i]
    
    if uncommitted:
        print("\n1. Commit your changes:")
        print("   git add -A")
        print("   git commit -m 'Your commit message'")
        print("   git push origin your-branch")
    
    if missing_files:
        print("\n2. Generate missing data:")
        if any("train.pkl" in i for i in missing_files):
            print("   python scan_data_loader.py")
        if any("modification_pairs.pkl" in i for i in missing_files):
            print("   python modification_generator.py")
    
    if validation_errors:
        print("\n3. Fix validation errors:")
        for error in validation_errors:
            if "save()" in error:
                print("   - Change tokenizer.save() to tokenizer.save_vocabulary()")
            elif "load_modifications" in error:
                print("   - Ensure modification_generator.py is committed and pushed")
            elif "ImportError" in error:
                print("   - Check import statements match actual module structure")

def main():
    print("=" * 60)
    print("PRE-PAPERSPACE DEPLOYMENT CHECKLIST")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("\nUsage: python pre_paperspace_checklist.py <script_to_deploy.py>")
        sys.exit(1)
    
    script_path = sys.argv[1]
    if not Path(script_path).exists():
        print(f"\n❌ Error: {script_path} not found")
        sys.exit(1)
    
    all_issues = []
    all_passed = True
    
    # Run all checks
    checks = [
        ("Git Status", check_git_status),
        ("Critical Files", check_critical_files_exist),
        ("Data Generation", check_data_generation),
        ("Validation", lambda: check_validation_passes(script_path))
    ]
    
    for name, check_func in checks:
        passed, issues = check_func()
        all_issues.extend(issues)
        if not passed:
            all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    print("DEPLOYMENT READINESS SUMMARY")
    print("=" * 60)
    
    if all_passed:
        print("\n✅ ALL CHECKS PASSED - Ready for Paperspace deployment!")
        print("\nNext steps:")
        print("1. On Paperspace: git pull origin production")
        print(f"2. Run: python {script_path}")
    else:
        print(f"\n❌ FAILED - {len(all_issues)} issues must be resolved")
        print("\nIssues found:")
        for i, issue in enumerate(all_issues, 1):
            print(f"{i}. {issue}")
        
        suggest_fixes(all_issues)
        
        print("\n⚠️  DO NOT DEPLOY until all issues are resolved!")
    
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()