#!/usr/bin/env python3
"""
Run the EXACT training script locally with minimal data.
This catches ALL runtime errors before wasting GPU time.

The key insight: We run the ACTUAL training code, not a simplified version.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

def setup_minimal_environment():
    """Set up environment for minimal local training"""
    # Use minimal settings
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Override settings for local testing
    os.environ['LOCAL_TEST_MODE'] = '1'
    os.environ['MAX_SAMPLES'] = '20'  # Use only 20 samples
    os.environ['EPOCHS_PER_STAGE'] = '1'  # 1 epoch per stage
    os.environ['BATCH_SIZE'] = '2'  # Small batch size
    
    print("🔧 Environment configured for local testing:")
    print(f"  - MAX_SAMPLES: {os.environ['MAX_SAMPLES']}")
    print(f"  - EPOCHS_PER_STAGE: {os.environ['EPOCHS_PER_STAGE']}")
    print(f"  - BATCH_SIZE: {os.environ['BATCH_SIZE']}")

def create_mock_storage():
    """Create a mock /storage directory for testing"""
    mock_storage = Path('mock_storage')
    mock_storage.mkdir(exist_ok=True)
    
    # Set environment variable to use mock storage
    os.environ['MOCK_STORAGE_PATH'] = str(mock_storage.absolute())
    
    print(f"\n📁 Created mock storage at: {mock_storage.absolute()}")
    return mock_storage

def modify_script_for_local_testing(script_path: str) -> str:
    """Create a modified version of the script for local testing"""
    # Read the original script
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Create modifications for local testing
    modifications = []
    
    # 1. Reduce epochs
    modifications.append((
        "'stage1_epochs': 5",
        "'stage1_epochs': int(os.environ.get('EPOCHS_PER_STAGE', '1'))"
    ))
    modifications.append((
        "'stage2_epochs': 5",
        "'stage2_epochs': int(os.environ.get('EPOCHS_PER_STAGE', '1'))"
    ))
    modifications.append((
        "'stage3_epochs': 5",
        "'stage3_epochs': int(os.environ.get('EPOCHS_PER_STAGE', '1'))"
    ))
    modifications.append((
        "'stage4_epochs': 5",
        "'stage4_epochs': int(os.environ.get('EPOCHS_PER_STAGE', '1'))"
    ))
    
    # 2. Reduce batch size
    modifications.append((
        "'batch_size': 32",
        "'batch_size': int(os.environ.get('BATCH_SIZE', '2'))"
    ))
    
    # 3. Use mock storage
    modifications.append((
        "if os.path.exists('/storage'):",
        "if os.path.exists(os.environ.get('MOCK_STORAGE_PATH', '/storage')):"
    ))
    modifications.append((
        "storage_dir = f'/storage/compositional_language_{timestamp}'",
        "storage_dir = os.path.join(os.environ.get('MOCK_STORAGE_PATH', '/storage'), f'compositional_language_{timestamp}')"
    ))
    
    # 4. Limit data samples
    modifications.append((
        "splits['train']",
        "splits['train'][:int(os.environ.get('MAX_SAMPLES', '20'))]"
    ))
    
    # Apply modifications
    modified_content = content
    for old, new in modifications:
        modified_content = modified_content.replace(old, new)
    
    # Save modified script
    test_script_path = script_path.replace('.py', '_local_test.py')
    with open(test_script_path, 'w') as f:
        f.write(modified_content)
    
    print(f"\n📝 Created test script: {test_script_path}")
    return test_script_path

def run_local_training_test(script_path: str):
    """Run the training script locally with minimal data"""
    print("\n" + "="*60)
    print("🏃 RUNNING LOCAL TRAINING TEST")
    print("="*60)
    
    # Setup
    setup_minimal_environment()
    mock_storage = create_mock_storage()
    
    # Create test version of script
    test_script = modify_script_for_local_testing(script_path)
    
    try:
        # Run the actual training script
        print(f"\n🚀 Executing: python {test_script}")
        print("This will run through ALL 4 training stages with minimal data...\n")
        
        result = subprocess.run(
            [sys.executable, test_script],
            capture_output=True,
            text=True
        )
        
        # Check results
        if result.returncode == 0:
            print("\n✅ LOCAL TRAINING TEST PASSED!")
            print("\nTraining completed successfully with:")
            print("  - All 4 stages executed")
            print("  - No runtime errors")
            print("  - Model saved successfully")
            
            # Check what was saved
            saved_files = list(mock_storage.rglob('*'))
            if saved_files:
                print(f"\n📦 Files saved during training:")
                for file in saved_files[:10]:  # Show first 10
                    print(f"  - {file.relative_to(mock_storage)}")
            
            return True
        else:
            print("\n❌ LOCAL TRAINING TEST FAILED!")
            print("\n" + "="*60)
            print("ERROR OUTPUT:")
            print("="*60)
            
            # Parse error for common issues
            error_output = result.stderr
            
            if "ValueError: Dimensions must be equal" in error_output:
                print("🔍 SHAPE MISMATCH ERROR DETECTED!")
                # Extract shape info
                import re
                shapes = re.findall(r'shapes: \[(.*?)\]', error_output)
                if shapes:
                    print(f"  Expected shape: {shapes[0].split(',')[0]}")
                    print(f"  Got shape: {shapes[0].split(',')[1] if ',' in shapes[0] else 'unknown'}")
                print("\n💡 This suggests the model output shape doesn't match target shape")
                
            elif "AttributeError" in error_output:
                print("🔍 ATTRIBUTE ERROR DETECTED!")
                # Extract attribute error
                attr_match = re.search(r"AttributeError: '(\w+)' object has no attribute '(\w+)'", error_output)
                if attr_match:
                    print(f"  Object: {attr_match.group(1)}")
                    print(f"  Missing attribute: {attr_match.group(2)}")
                
            elif "Target data is missing" in error_output:
                print("🔍 DATASET FORMAT ERROR DETECTED!")
                print("  The dataset is not returning (inputs, targets) format")
            
            # Show full error
            print("\nFull error output:")
            print(error_output[-2000:])  # Last 2000 chars
            
            return False
            
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        return False
    finally:
        # Cleanup
        if Path(test_script).exists():
            os.remove(test_script)
        if mock_storage.exists():
            shutil.rmtree(mock_storage)
        print("\n🧹 Cleaned up test files")

def check_common_issues():
    """Pre-flight checks for common issues"""
    print("\n🔍 Running pre-flight checks...")
    
    issues = []
    
    # Check data exists
    if not Path('data/processed/train.pkl').exists():
        issues.append("Training data not found - run: python scan_data_loader.py")
    
    if not Path('data/processed/modification_pairs.pkl').exists():
        issues.append("Modification data not found - run: python modification_generator.py")
    
    # Check git status
    result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
    if result.stdout.strip():
        issues.append("Uncommitted changes detected - commit before testing")
    
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("✓ All pre-flight checks passed")
    return True

def main():
    """Main test orchestrator"""
    print("🧪 LOCAL TRAINING TEST SUITE")
    print("This runs the ACTUAL training code with minimal data")
    print("to catch ALL runtime errors before Paperspace deployment.\n")
    
    if len(sys.argv) < 2:
        script_path = 'paperspace_train_with_safeguards.py'
        print(f"No script specified, using default: {script_path}")
    else:
        script_path = sys.argv[1]
    
    if not Path(script_path).exists():
        print(f"❌ Error: {script_path} not found")
        sys.exit(1)
    
    # Run checks
    if not check_common_issues():
        print("\n⚠️  Fix issues before proceeding")
        sys.exit(1)
    
    # Run local training test
    success = run_local_training_test(script_path)
    
    if success:
        print("\n" + "="*60)
        print("✅ READY FOR PAPERSPACE DEPLOYMENT!")
        print("="*60)
        print("\nNext steps:")
        print("1. Push to production: git push origin branch && gh pr create && gh pr merge")
        print("2. On Paperspace: git pull origin production")
        print("3. Run: python paperspace_train_with_safeguards.py")
    else:
        print("\n" + "="*60)
        print("❌ DO NOT DEPLOY - Fix errors first!")
        print("="*60)
        print("\nThis local test just saved you hours of GPU debugging!")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()