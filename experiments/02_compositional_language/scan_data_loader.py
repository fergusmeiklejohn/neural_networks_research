#!/usr/bin/env python3
"""
SCAN Dataset Loader with Proper Train/Test Isolation

This module handles loading and preprocessing the SCAN dataset for compositional
language learning experiments. It ensures proper data isolation to prevent
leakage between train and test sets.

The SCAN dataset tests compositional generalization in seq2seq models by
mapping natural language commands to action sequences.
"""

import os
import json
import pickle
import random
import requests
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass


@dataclass
class SCANSample:
    """Single SCAN dataset sample"""
    command: str
    action: str
    primitives: Set[str]  # Primitive words in command
    modifiers: Set[str]   # Modifier words (twice, thrice, etc.)
    connectors: Set[str]  # Connectors (and, after)
    length: int          # Number of tokens in action
    
    def to_dict(self):
        return {
            'command': self.command,
            'action': self.action,
            'primitives': list(self.primitives),
            'modifiers': list(self.modifiers),
            'connectors': list(self.connectors),
            'length': self.length
        }


class SCANDataLoader:
    """Loads and preprocesses SCAN dataset with proper data isolation"""
    
    # SCAN primitive actions and modifiers
    PRIMITIVES = {'walk', 'run', 'jump', 'look', 'turn'}
    DIRECTIONS = {'left', 'right', 'around'}
    MODIFIERS = {'twice', 'thrice', 'opposite'}
    CONNECTORS = {'and', 'after'}
    
    # SCAN GitHub URLs
    SCAN_URLS = {
        'simple': 'https://raw.githubusercontent.com/brendenlake/SCAN/master/simple_split/tasks_train_simple.txt',
        'simple_test': 'https://raw.githubusercontent.com/brendenlake/SCAN/master/simple_split/tasks_test_simple.txt',
        'length': 'https://raw.githubusercontent.com/brendenlake/SCAN/master/length_split/tasks_train_length.txt',
        'length_test': 'https://raw.githubusercontent.com/brendenlake/SCAN/master/length_split/tasks_test_length.txt',
        'addprim_jump': 'https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/tasks_train_addprim_jump.txt',
        'addprim_jump_test': 'https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/tasks_test_addprim_jump.txt',
    }
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.samples: List[SCANSample] = []
        self.vocab_command: Set[str] = set()
        self.vocab_action: Set[str] = set()
        
    def download_scan_data(self):
        """Download SCAN dataset files"""
        print("Downloading SCAN dataset...")
        
        for split_name, url in self.SCAN_URLS.items():
            filepath = self.raw_dir / f"{split_name}.txt"
            
            if filepath.exists():
                print(f"  {split_name} already exists, skipping...")
                continue
                
            print(f"  Downloading {split_name}...")
            response = requests.get(url)
            response.raise_for_status()
            
            with open(filepath, 'w') as f:
                f.write(response.text)
                
        print("Download complete!")
        
    def parse_scan_file(self, filepath: Path) -> List[SCANSample]:
        """Parse a SCAN dataset file"""
        samples = []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                # SCAN format: "IN: command OUT: action"
                if line.startswith('IN: ') and ' OUT: ' in line:
                    parts = line.split(' OUT: ')
                    command = parts[0].replace('IN: ', '').strip()
                    action = parts[1].strip()
                    
                    # Extract components
                    command_tokens = command.lower().split()
                    primitives = set(t for t in command_tokens if t in self.PRIMITIVES)
                    directions = set(t for t in command_tokens if t in self.DIRECTIONS)
                    modifiers = set(t for t in command_tokens if t in self.MODIFIERS)
                    connectors = set(t for t in command_tokens if t in self.CONNECTORS)
                    
                    # Combine primitives and directions
                    primitives.update(directions)
                    
                    # Create sample
                    sample = SCANSample(
                        command=command,
                        action=action,
                        primitives=primitives,
                        modifiers=modifiers,
                        connectors=connectors,
                        length=len(action.split())
                    )
                    samples.append(sample)
                    
                    # Update vocabularies
                    self.vocab_command.update(command.lower().split())
                    self.vocab_action.update(action.split())
                    
        return samples
    
    def load_all_data(self) -> Dict[str, List[SCANSample]]:
        """Load all SCAN splits"""
        print("Loading SCAN data...")
        
        # Download if needed
        self.download_scan_data()
        
        # Load each split
        all_data = {}
        for split_name in self.SCAN_URLS.keys():
            filepath = self.raw_dir / f"{split_name}.txt"
            if filepath.exists():
                samples = self.parse_scan_file(filepath)
                all_data[split_name] = samples
                print(f"  Loaded {split_name}: {len(samples)} samples")
                
        self.samples = [s for samples in all_data.values() for s in samples]
        print(f"\nTotal samples loaded: {len(self.samples)}")
        print(f"Command vocabulary size: {len(self.vocab_command)}")
        print(f"Action vocabulary size: {len(self.vocab_action)}")
        
        return all_data
    
    def create_isolated_splits(self, 
                              train_ratio: float = 0.7,
                              val_ratio: float = 0.15,
                              seed: int = 42) -> Dict[str, List[SCANSample]]:
        """
        Create train/val/test splits with proper isolation.
        
        Isolation strategies:
        1. Interpolation: Same primitive combinations as training
        2. Primitive extrapolation: Unseen primitive in specific contexts
        3. Length extrapolation: Longer sequences than training
        4. Modifier extrapolation: Unseen modifier combinations
        """
        random.seed(seed)
        np.random.seed(seed)
        
        print("\nCreating isolated data splits...")
        
        # Group samples by characteristics
        samples_by_primitives = defaultdict(list)
        samples_by_length = defaultdict(list)
        samples_by_modifiers = defaultdict(list)
        
        for sample in self.samples:
            # Group by primitive combination
            prim_key = tuple(sorted(sample.primitives))
            samples_by_primitives[prim_key].append(sample)
            
            # Group by length
            length_bucket = sample.length // 10  # Bucket by 10s
            samples_by_length[length_bucket].append(sample)
            
            # Group by modifier combination
            mod_key = tuple(sorted(sample.modifiers))
            samples_by_modifiers[mod_key].append(sample)
        
        # Create splits
        splits = {
            'train': [],
            'val_interpolation': [],     # Same distribution as train
            'val_extrapolation': [],     # Different distribution
            'test_interpolation': [],    # Same distribution as train
            'test_primitive_extrap': [], # New primitive combinations
            'test_length_extrap': [],    # Longer sequences
            'test_modifier_extrap': []   # New modifier combinations
        }
        
        # 1. Reserve some primitive combinations for extrapolation testing
        all_prim_combos = list(samples_by_primitives.keys())
        random.shuffle(all_prim_combos)
        
        num_test_combos = max(1, len(all_prim_combos) // 10)  # 10% for testing
        test_prim_combos = set(all_prim_combos[:num_test_combos])
        train_prim_combos = set(all_prim_combos[num_test_combos:])
        
        # 2. Assign samples with test primitive combinations
        for prim_combo in test_prim_combos:
            splits['test_primitive_extrap'].extend(samples_by_primitives[prim_combo])
        
        # 3. From training primitive combinations, create train/val/test splits
        train_samples = []
        for prim_combo in train_prim_combos:
            train_samples.extend(samples_by_primitives[prim_combo])
        
        random.shuffle(train_samples)
        n_train = int(len(train_samples) * train_ratio)
        n_val = int(len(train_samples) * val_ratio)
        
        splits['train'] = train_samples[:n_train]
        splits['val_interpolation'] = train_samples[n_train:n_train + n_val]
        splits['test_interpolation'] = train_samples[n_train + n_val:]
        
        # 4. Create length extrapolation test set
        max_train_length = max(s.length for s in splits['train'])
        for sample in self.samples:
            if sample.length > max_train_length and len(splits['test_length_extrap']) < 100:
                splits['test_length_extrap'].append(sample)
        
        # 5. Create modifier extrapolation test set
        train_modifier_combos = set()
        for s in splits['train']:
            train_modifier_combos.add(tuple(sorted(s.modifiers)))
        
        for sample in self.samples:
            mod_combo = tuple(sorted(sample.modifiers))
            if mod_combo not in train_modifier_combos and len(splits['test_modifier_extrap']) < 100:
                splits['test_modifier_extrap'].append(sample)
        
        # 6. Create general extrapolation validation set
        # Mix of different extrapolation types
        extrap_samples = (splits['test_primitive_extrap'][:20] + 
                         splits['test_length_extrap'][:20] + 
                         splits['test_modifier_extrap'][:20])
        random.shuffle(extrap_samples)
        splits['val_extrapolation'] = extrap_samples[:50]
        
        # Report statistics
        print("\nData split statistics:")
        for split_name, samples in splits.items():
            if samples:
                avg_length = np.mean([s.length for s in samples])
                print(f"  {split_name}: {len(samples)} samples, avg length: {avg_length:.1f}")
        
        # Verify isolation
        train_primitives = set()
        for s in splits['train']:
            train_primitives.update(s.primitives)
        
        test_extrap_primitives = set()
        for s in splits['test_primitive_extrap']:
            test_extrap_primitives.update(s.primitives)
        
        print(f"\nIsolation verification:")
        print(f"  Train primitives: {train_primitives}")
        print(f"  Test extrapolation has new combinations: {len(splits['test_primitive_extrap']) > 0}")
        
        return splits
    
    def save_processed_data(self, splits: Dict[str, List[SCANSample]]):
        """Save processed data splits"""
        print("\nSaving processed data...")
        
        for split_name, samples in splits.items():
            if not samples:
                continue
                
            # Convert to dict format
            data = [s.to_dict() for s in samples]
            
            # Save as pickle
            filepath = self.processed_dir / f"{split_name}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            # Save metadata
            metadata = {
                'num_samples': len(samples),
                'avg_command_length': np.mean([len(s.command.split()) for s in samples]),
                'avg_action_length': np.mean([s.length for s in samples]),
                'unique_primitives': list(set(p for s in samples for p in s.primitives)),
                'unique_modifiers': list(set(m for s in samples for m in s.modifiers))
            }
            
            meta_filepath = self.processed_dir / f"{split_name}_metadata.json"
            with open(meta_filepath, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Save vocabularies
        vocab_data = {
            'command_vocab': sorted(list(self.vocab_command)),
            'action_vocab': sorted(list(self.vocab_action)),
            'primitives': sorted(list(self.PRIMITIVES)),
            'directions': sorted(list(self.DIRECTIONS)),
            'modifiers': sorted(list(self.MODIFIERS)),
            'connectors': sorted(list(self.CONNECTORS))
        }
        
        with open(self.processed_dir / 'vocabulary.json', 'w') as f:
            json.dump(vocab_data, f, indent=2)
        
        print("Data saved successfully!")
    
    def load_processed_splits(self) -> Dict[str, List[Dict]]:
        """Load previously processed splits"""
        splits = {}
        
        for filepath in self.processed_dir.glob("*.pkl"):
            if filepath.stem.endswith('_metadata'):
                continue
                
            with open(filepath, 'rb') as f:
                splits[filepath.stem] = pickle.load(f)
                
        return splits


def analyze_compositional_patterns(samples: List[SCANSample]):
    """Analyze compositional patterns in SCAN data"""
    print("\nAnalyzing compositional patterns...")
    
    # Pattern statistics
    pattern_counts = Counter()
    primitive_counts = Counter()
    modifier_counts = Counter()
    length_distribution = Counter()
    
    for sample in samples:
        # Count patterns
        if sample.modifiers:
            pattern_counts['has_modifiers'] += 1
        if sample.connectors:
            pattern_counts['has_connectors'] += 1
        if len(sample.primitives) > 1:
            pattern_counts['multiple_primitives'] += 1
            
        # Count individual elements
        for p in sample.primitives:
            primitive_counts[p] += 1
        for m in sample.modifiers:
            modifier_counts[m] += 1
            
        # Length distribution
        length_distribution[sample.length // 5 * 5] += 1  # Bucket by 5s
    
    print(f"\nPattern distribution:")
    for pattern, count in pattern_counts.most_common():
        print(f"  {pattern}: {count} ({count/len(samples)*100:.1f}%)")
    
    print(f"\nPrimitive distribution:")
    for prim, count in primitive_counts.most_common():
        print(f"  {prim}: {count}")
    
    print(f"\nModifier distribution:")
    for mod, count in modifier_counts.most_common():
        print(f"  {mod}: {count}")
    
    print(f"\nAction length distribution:")
    for length, count in sorted(length_distribution.items()):
        print(f"  {length}-{length+4}: {count}")


def main():
    """Test the SCAN data loader"""
    loader = SCANDataLoader()
    
    # Load all data
    all_data = loader.load_all_data()
    
    # Analyze patterns
    analyze_compositional_patterns(loader.samples)
    
    # Create isolated splits
    splits = loader.create_isolated_splits()
    
    # Save processed data
    loader.save_processed_data(splits)
    
    # Test loading processed data
    print("\nTesting processed data loading...")
    loaded_splits = loader.load_processed_splits()
    for split_name, data in loaded_splits.items():
        print(f"  Loaded {split_name}: {len(data)} samples")
    
    # Show example samples
    print("\nExample samples:")
    for split_name in ['train', 'test_primitive_extrap', 'test_length_extrap']:
        if split_name in splits and splits[split_name]:
            sample = splits[split_name][0]
            print(f"\n{split_name}:")
            print(f"  Command: {sample.command}")
            print(f"  Action: {sample.action}")
            print(f"  Primitives: {sample.primitives}")
            print(f"  Modifiers: {sample.modifiers}")


if __name__ == "__main__":
    main()