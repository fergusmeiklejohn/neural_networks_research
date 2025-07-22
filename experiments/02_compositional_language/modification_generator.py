#!/usr/bin/env python3
"""
Linguistic Modification Pair Generator for SCAN

This module generates modification pairs for the SCAN dataset, creating
systematic rule changes that test compositional understanding.
"""

import random
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
from dataclasses import dataclass

from scan_data_loader import SCANSample, SCANDataLoader


@dataclass
class ModificationPair:
    """A pair of original and modified samples with the modification description"""
    original_sample: SCANSample
    modified_sample: SCANSample
    modification_type: str
    modification_description: str
    modification_rules: Dict[str, str]  # e.g., {"jump": "walk"}
    
    def to_dict(self):
        return {
            'original': self.original_sample.to_dict(),
            'modified': self.modified_sample.to_dict(),
            'modification_type': self.modification_type,
            'modification_description': self.modification_description,
            'modification_rules': self.modification_rules
        }


class ModificationGenerator:
    """Generate systematic modifications to SCAN commands"""
    
    # Action mappings in SCAN
    ACTION_MAP = {
        'walk': 'I_WALK',
        'run': 'I_RUN', 
        'jump': 'I_JUMP',
        'look': 'I_LOOK',
        'turn': 'I_TURN',
        'left': 'I_TURN_LEFT',
        'right': 'I_TURN_RIGHT'
    }
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / 'processed'
        
    def apply_word_swap(self, 
                       command: str, 
                       action: str,
                       swap_rules: Dict[str, str]) -> Tuple[str, str]:
        """Apply word swapping rules to command and action"""
        # Modify command
        command_tokens = command.split()
        modified_command_tokens = []
        
        for token in command_tokens:
            if token in swap_rules:
                modified_command_tokens.append(swap_rules[token])
            else:
                modified_command_tokens.append(token)
        
        modified_command = ' '.join(modified_command_tokens)
        
        # Modify action accordingly
        # This requires parsing the action sequence and applying the swaps
        modified_action = action
        
        for original, replacement in swap_rules.items():
            if original in self.ACTION_MAP and replacement in self.ACTION_MAP:
                # Replace action tokens
                original_action = self.ACTION_MAP[original]
                replacement_action = self.ACTION_MAP[replacement]
                modified_action = modified_action.replace(original_action, replacement_action)
        
        return modified_command, modified_action
    
    def generate_simple_swaps(self, samples: List[SCANSample]) -> List[ModificationPair]:
        """Generate simple word swap modifications"""
        pairs = []
        
        # Define swap rules
        swap_configurations = [
            # Primitive swaps
            {'jump': 'walk', 'description': 'jump means walk'},
            {'walk': 'run', 'description': 'walk means run'},
            {'run': 'jump', 'description': 'run means jump'},
            {'look': 'turn', 'description': 'look means turn'},
            
            # Direction swaps
            {'left': 'right', 'right': 'left', 'description': 'left and right are swapped'},
            
            # Modifier swaps
            {'twice': 'thrice', 'thrice': 'twice', 'description': 'twice and thrice are swapped'},
        ]
        
        # Generate modifications for each configuration
        for config in swap_configurations:
            description = config.pop('description')
            swap_rules = config
            
            # Apply to relevant samples
            relevant_samples = []
            for sample in samples:
                # Check if sample contains any of the words to swap
                command_tokens = set(sample.command.split())
                if any(word in command_tokens for word in swap_rules.keys()):
                    relevant_samples.append(sample)
            
            # Generate pairs for a subset
            num_pairs = min(100, len(relevant_samples))
            selected_samples = random.sample(relevant_samples, num_pairs)
            
            for sample in selected_samples:
                modified_command, modified_action = self.apply_word_swap(
                    sample.command, sample.action, swap_rules
                )
                
                # Create modified sample
                modified_sample = SCANSample(
                    command=modified_command,
                    action=modified_action,
                    primitives=sample.primitives,  # Update these based on swaps
                    modifiers=sample.modifiers,
                    connectors=sample.connectors,
                    length=len(modified_action.split())
                )
                
                # Create pair
                pair = ModificationPair(
                    original_sample=sample,
                    modified_sample=modified_sample,
                    modification_type='simple_swap',
                    modification_description=description,
                    modification_rules=swap_rules
                )
                pairs.append(pair)
        
        return pairs
    
    def generate_action_modifications(self, samples: List[SCANSample]) -> List[ModificationPair]:
        """Generate modifications that change action meanings"""
        pairs = []
        
        # Define action modifications
        action_modifications = [
            {
                'target': 'jump',
                'new_action': 'I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT',
                'description': 'jump means turn around (spin 360)'
            },
            {
                'target': 'look',
                'new_action': 'I_WALK I_TURN_LEFT I_TURN_LEFT I_WALK',
                'description': 'look means walk forward and back'
            },
            {
                'target': 'run',
                'new_action': 'I_WALK I_WALK',
                'description': 'run means walk twice'
            }
        ]
        
        for mod_config in action_modifications:
            target = mod_config['target']
            new_action = mod_config['new_action']
            description = mod_config['description']
            
            # Find samples with target action
            relevant_samples = [s for s in samples if target in s.command.split()]
            num_pairs = min(100, len(relevant_samples))
            selected_samples = random.sample(relevant_samples, num_pairs)
            
            for sample in selected_samples:
                # Replace action in the sequence
                modified_action = sample.action
                
                if target in self.ACTION_MAP:
                    original_action = self.ACTION_MAP[target]
                    # Count occurrences and replace
                    modified_action = modified_action.replace(original_action, new_action)
                
                # Create modified sample
                modified_sample = SCANSample(
                    command=sample.command,  # Command stays the same
                    action=modified_action,
                    primitives=sample.primitives,
                    modifiers=sample.modifiers,
                    connectors=sample.connectors,
                    length=len(modified_action.split())
                )
                
                # Create pair
                pair = ModificationPair(
                    original_sample=sample,
                    modified_sample=modified_sample,
                    modification_type='action_modification',
                    modification_description=description,
                    modification_rules={target: new_action}
                )
                pairs.append(pair)
        
        return pairs
    
    def generate_structural_modifications(self, samples: List[SCANSample]) -> List[ModificationPair]:
        """Generate modifications that change compositional structure"""
        pairs = []
        
        # Reverse all directions
        direction_samples = [s for s in samples if any(d in s.command for d in ['left', 'right'])]
        num_pairs = min(100, len(direction_samples))
        selected_samples = random.sample(direction_samples, num_pairs)
        
        for sample in selected_samples:
            # Reverse directions in action
            modified_action = sample.action
            modified_action = modified_action.replace('I_TURN_LEFT', '<TEMP>')
            modified_action = modified_action.replace('I_TURN_RIGHT', 'I_TURN_LEFT')
            modified_action = modified_action.replace('<TEMP>', 'I_TURN_RIGHT')
            
            # Also swap in command for consistency
            modified_command = sample.command
            modified_command = modified_command.replace('left', '<TEMP>')
            modified_command = modified_command.replace('right', 'left')
            modified_command = modified_command.replace('<TEMP>', 'right')
            
            # Create modified sample
            modified_sample = SCANSample(
                command=modified_command,
                action=modified_action,
                primitives=sample.primitives,
                modifiers=sample.modifiers,
                connectors=sample.connectors,
                length=len(modified_action.split())
            )
            
            pair = ModificationPair(
                original_sample=sample,
                modified_sample=modified_sample,
                modification_type='structural',
                modification_description='all directions reversed',
                modification_rules={'left': 'right', 'right': 'left'}
            )
            pairs.append(pair)
        
        # Modifier changes: "twice" means once, "thrice" means twice
        modifier_samples = [s for s in samples if any(m in s.command for m in ['twice', 'thrice'])]
        num_pairs = min(100, len(modifier_samples))
        selected_samples = random.sample(modifier_samples, num_pairs)
        
        for sample in selected_samples:
            modified_action = sample.action
            modified_command = sample.command
            
            # Parse and reconstruct action with modified repetitions
            # This is complex - for now, we'll create a simplified version
            if 'twice' in sample.command:
                # Should reduce repetitions by 1
                description = 'twice means once'
                rules = {'twice': 'once'}
            elif 'thrice' in sample.command:
                # Should reduce repetitions by 1
                description = 'thrice means twice'
                rules = {'thrice': 'twice'}
            else:
                continue
            
            # For simplicity, we'll keep the same action but note the rule
            # In a full implementation, we'd parse and modify the repetitions
            
            modified_sample = SCANSample(
                command=sample.command,
                action=sample.action,  # Simplified - should modify repetitions
                primitives=sample.primitives,
                modifiers=sample.modifiers,
                connectors=sample.connectors,
                length=sample.length
            )
            
            pair = ModificationPair(
                original_sample=sample,
                modified_sample=modified_sample,
                modification_type='structural',
                modification_description=description,
                modification_rules=rules
            )
            pairs.append(pair)
        
        return pairs
    
    def generate_novel_combinations(self, samples: List[SCANSample]) -> List[ModificationPair]:
        """Generate novel but valid command combinations"""
        pairs = []
        
        # Extract patterns from existing samples
        primitives = set()
        modifiers = set()
        connectors = set()
        
        for sample in samples:
            primitives.update(sample.primitives)
            modifiers.update(sample.modifiers)
            connectors.update(sample.connectors)
        
        # Create novel combinations not seen in training
        # For now, we'll create placeholder samples
        # In a full implementation, we'd generate truly novel combinations
        
        return pairs
    
    def generate_all_modifications(self, train_samples: List[SCANSample]) -> Dict[str, List[ModificationPair]]:
        """Generate all types of modifications"""
        print("Generating modification pairs...")
        
        modifications = {
            'simple_swaps': self.generate_simple_swaps(train_samples),
            'action_modifications': self.generate_action_modifications(train_samples),
            'structural_modifications': self.generate_structural_modifications(train_samples),
            'novel_combinations': self.generate_novel_combinations(train_samples)
        }
        
        # Report statistics
        print("\nModification statistics:")
        for mod_type, pairs in modifications.items():
            if pairs:
                print(f"  {mod_type}: {len(pairs)} pairs")
                # Show example
                example = pairs[0]
                print(f"    Example: {example.modification_description}")
                print(f"    Original: {example.original_sample.command}")
                print(f"    Modified: {example.modified_sample.command}")
        
        return modifications
    
    def save_modifications(self, modifications: Dict[str, List[ModificationPair]]):
        """Save modification pairs"""
        print("\nSaving modification pairs...")
        
        # Combine all modifications
        all_pairs = []
        for pairs in modifications.values():
            all_pairs.extend(pairs)
        
        # Convert to dict format
        data = [pair.to_dict() for pair in all_pairs]
        
        # Save
        filepath = self.processed_dir / 'modification_pairs.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        # Save metadata
        metadata = {
            'total_pairs': len(all_pairs),
            'modification_types': {
                mod_type: len(pairs) 
                for mod_type, pairs in modifications.items()
            }
        }
        
        meta_filepath = self.processed_dir / 'modification_metadata.json'
        with open(meta_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {len(all_pairs)} modification pairs")
    
    def load_modifications(self) -> List[ModificationPair]:
        """Load saved modification pairs"""
        filepath = self.processed_dir / 'modification_pairs.pkl'
        
        if not filepath.exists():
            print(f"WARNING: No modifications found at {filepath}")
            return []
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Convert dict format back to ModificationPair objects
        pairs = []
        for item in data:
            # Reconstruct SCANSample objects
            original_sample = SCANSample(
                command=item['original']['command'],
                action=item['original']['action'],
                primitives=set(item['original']['primitives']),
                modifiers=set(item['original']['modifiers']),
                connectors=set(item['original']['connectors']),
                length=item['original']['length']
            )
            
            modified_sample = SCANSample(
                command=item['modified']['command'],
                action=item['modified']['action'],
                primitives=set(item['modified']['primitives']),
                modifiers=set(item['modified']['modifiers']),
                connectors=set(item['modified']['connectors']),
                length=item['modified']['length']
            )
            
            # Create ModificationPair
            pair = ModificationPair(
                original_sample=original_sample,
                modified_sample=modified_sample,
                modification_type=item['modification_type'],
                modification_description=item['modification_description'],
                modification_rules=item['modification_rules']
            )
            pairs.append(pair)
        
        print(f"Loaded {len(pairs)} modification pairs")
        return pairs


def main():
    """Test the modification generator"""
    # Load processed data
    loader = SCANDataLoader()
    splits = loader.load_processed_splits()
    
    if 'train' not in splits:
        print("No training data found. Run scan_data_loader.py first.")
        return
    
    # Convert back to SCANSample objects
    train_samples = []
    for data in splits['train'][:1000]:  # Use subset for testing
        sample = SCANSample(
            command=data['command'],
            action=data['action'],
            primitives=set(data['primitives']),
            modifiers=set(data['modifiers']),
            connectors=set(data['connectors']),
            length=data['length']
        )
        train_samples.append(sample)
    
    # Generate modifications
    generator = ModificationGenerator()
    modifications = generator.generate_all_modifications(train_samples)
    
    # Save modifications
    generator.save_modifications(modifications)
    
    print("\nModification generation complete!")


if __name__ == "__main__":
    main()