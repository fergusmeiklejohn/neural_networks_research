#!/usr/bin/env python3
"""Test modification generator to debug Stage 2 failure"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from modification_generator import ModificationGenerator
from scan_data_loader import SCANDataLoader

def test_modifications():
    """Test modification loading and format"""
    print("Testing Modification Generator...")
    print("-" * 40)
    
    # Load modifications
    mod_gen = ModificationGenerator()
    try:
        mods = mod_gen.load_modifications()
        print(f"✓ Loaded {len(mods)} modifications")
    except Exception as e:
        print(f"✗ Failed to load modifications: {e}")
        return False
    
    # Check format
    if not mods:
        print("✗ No modifications loaded!")
        return False
    
    # Analyze first few modifications
    print("\nFirst 3 modifications:")
    for i, mod in enumerate(mods[:3]):
        print(f"\nModification {i+1}:")
        print(f"  Keys: {list(mod.keys())}")
        print(f"  Command: {mod.get('command', 'MISSING')}")
        print(f"  Original action: {mod.get('original_action', 'MISSING')}")
        print(f"  Modification: {mod.get('modification', 'MISSING')}")
        print(f"  Modified action: {mod.get('modified_action', 'MISSING')}")
        
        # Check required fields
        required = ['command', 'original_action', 'modification', 'modified_action']
        missing = [field for field in required if field not in mod]
        if missing:
            print(f"  ⚠️  Missing fields: {missing}")
    
    # Test filtering modifications with 'modification' key
    print("\nTesting modification filtering:")
    filtered = [m for m in mods if m['modification']]
    print(f"  Modifications with non-empty 'modification': {len(filtered)}")
    
    # Test mixing with base data
    print("\nTesting data mixing (as in Stage 2):")
    loader = SCANDataLoader()
    splits = loader.load_processed_splits()
    base_data = splits['train']
    
    # Simulate Stage 2 data creation
    stage_data = base_data[:int(len(base_data) * 0.7)]  # 70% base
    stage_data.extend([m for m in mods[:100] if m['modification']])  # modifications
    print(f"  Base data samples: {len(base_data[:int(len(base_data) * 0.7)])}")
    print(f"  Modification samples: {len([m for m in mods[:100] if m['modification']])}")
    print(f"  Total stage data: {len(stage_data)}")
    
    return True

if __name__ == "__main__":
    success = test_modifications()
    if success:
        print("\n✓ Modification test passed!")
    else:
        print("\n✗ Modification test failed!")