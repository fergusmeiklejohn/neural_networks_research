#!/usr/bin/env python3
"""
SCAN Dataset Preparation with Rule Modifications
Works both locally (for testing) and on Paperspace (for full training)
"""

import os
import sys
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import centralized utilities
from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
from utils.paths import get_data_path, get_output_path


class SCANDataGenerator:
    """Generate and prepare SCAN data with rule modifications"""
    
    def __init__(self, subset_size: Optional[int] = None, seed: int = 42):
        """
        Initialize SCAN data generator
        
        Args:
            subset_size: If provided, only use this many examples (for testing)
            seed: Random seed for reproducibility
        """
        self.subset_size = subset_size
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Define SCAN primitives and rules
        self.primitives = {
            'walk': 'WALK',
            'run': 'RUN', 
            'jump': 'JUMP',
            'look': 'LOOK',
            'turn left': 'TURN_LEFT',
            'turn right': 'TURN_RIGHT'
        }
        
        self.modifiers = {
            'twice': 2,
            'thrice': 3,
            'left': 'TURN_LEFT',
            'right': 'TURN_RIGHT',
            'around left': ['TURN_LEFT', 'TURN_LEFT', 'TURN_LEFT', 'TURN_LEFT'],
            'around right': ['TURN_RIGHT', 'TURN_RIGHT', 'TURN_RIGHT', 'TURN_RIGHT']
        }
        
        self.connectives = ['and', 'after']
        
        # Build vocabulary
        self.build_vocabulary()
    
    def build_vocabulary(self):
        """Build vocabularies for input commands and output actions"""
        # Input vocabulary
        input_words = set(['<PAD>', '<START>', '<END>', '<UNK>'])
        input_words.update(self.primitives.keys())
        input_words.update(self.modifiers.keys())
        input_words.update(self.connectives)
        input_words.update(['opposite', 'around'])  # Additional words
        
        self.input_vocab = {word: i for i, word in enumerate(sorted(input_words))}
        self.input_vocab_inv = {i: word for word, i in self.input_vocab.items()}
        
        # Output vocabulary
        output_words = set(['<PAD>', '<START>', '<END>', '<UNK>'])
        output_words.update(self.primitives.values())
        output_words.update(['TURN_LEFT', 'TURN_RIGHT'])
        
        self.output_vocab = {word: i for i, word in enumerate(sorted(output_words))}
        self.output_vocab_inv = {i: word for word, i in self.output_vocab.items()}
        
        print(f"Input vocabulary size: {len(self.input_vocab)}")
        print(f"Output vocabulary size: {len(self.output_vocab)}")
    
    def generate_simple_commands(self, n: int = 1000) -> List[Tuple[str, str]]:
        """Generate simple SCAN commands"""
        commands = []
        
        # Single actions
        for action, output in self.primitives.items():
            commands.append((action, output))
        
        # Actions with repetition
        for action, output in self.primitives.items():
            if action not in ['turn left', 'turn right']:
                commands.append((f"{action} twice", f"{output} {output}"))
                commands.append((f"{action} thrice", f"{output} {output} {output}"))
        
        # Actions with direction
        for action, output in list(self.primitives.items())[:4]:  # walk, run, jump, look
            commands.append((f"{action} left", f"TURN_LEFT {output}"))
            commands.append((f"{action} right", f"TURN_RIGHT {output}"))
        
        # Around commands
        for action, output in list(self.primitives.items())[:2]:  # walk, run
            commands.append((
                f"{action} around left",
                f"TURN_LEFT {output} TURN_LEFT {output} TURN_LEFT {output} TURN_LEFT {output}"
            ))
            commands.append((
                f"{action} around right", 
                f"TURN_RIGHT {output} TURN_RIGHT {output} TURN_RIGHT {output} TURN_RIGHT {output}"
            ))
        
        # Compound commands with 'and'
        actions = list(self.primitives.items())[:4]
        for i, (act1, out1) in enumerate(actions):
            for act2, out2 in actions[i+1:]:
                commands.append((f"{act1} and {act2}", f"{out1} {out2}"))
        
        # Compound commands with 'after'
        for i, (act1, out1) in enumerate(actions):
            for act2, out2 in actions[i+1:]:
                commands.append((f"{act1} after {act2}", f"{out2} {out1}"))
        
        # If we need more examples, generate more complex ones
        if len(commands) < n:
            # Three-part commands
            for _ in range(n - len(commands)):
                acts = random.sample(actions, 3)
                connector1 = random.choice(['and', 'after'])
                connector2 = random.choice(['and', 'after'])
                
                cmd = f"{acts[0][0]} {connector1} {acts[1][0]} {connector2} {acts[2][0]}"
                
                if connector1 == 'and' and connector2 == 'and':
                    out = f"{acts[0][1]} {acts[1][1]} {acts[2][1]}"
                elif connector1 == 'after' and connector2 == 'and':
                    out = f"{acts[1][1]} {acts[0][1]} {acts[2][1]}"
                elif connector1 == 'and' and connector2 == 'after':
                    out = f"{acts[2][1]} {acts[0][1]} {acts[1][1]}"
                else:  # after after
                    out = f"{acts[2][1]} {acts[1][1]} {acts[0][1]}"
                
                commands.append((cmd, out))
        
        # Shuffle and limit if needed
        random.shuffle(commands)
        if self.subset_size:
            commands = commands[:self.subset_size]
        
        return commands
    
    def create_rule_modifications(self, commands: List[Tuple[str, str]]) -> Dict[str, List[Tuple[str, str]]]:
        """Create different rule modifications"""
        modifications = {}
        
        # 1. Word swap: jump <-> walk
        modifications['jump_walk_swap'] = []
        for cmd, out in commands:
            new_cmd = cmd.replace('jump', 'TEMP').replace('walk', 'jump').replace('TEMP', 'walk')
            new_out = out.replace('JUMP', 'TEMP').replace('WALK', 'JUMP').replace('TEMP', 'WALK')
            modifications['jump_walk_swap'].append((new_cmd, new_out))
        
        # 2. Direction reversal: left <-> right
        modifications['direction_reversal'] = []
        for cmd, out in commands:
            new_cmd = cmd.replace('left', 'TEMP').replace('right', 'left').replace('TEMP', 'right')
            new_out = out.replace('TURN_LEFT', 'TEMP').replace('TURN_RIGHT', 'TURN_LEFT').replace('TEMP', 'TURN_RIGHT')
            modifications['direction_reversal'].append((new_cmd, new_out))
        
        # 3. Action modification: jump -> turn around
        modifications['jump_to_turn_around'] = []
        for cmd, out in commands:
            if 'jump' in cmd:
                new_cmd = cmd.replace('jump', 'turn around')
                new_out = out.replace('JUMP', 'TURN_LEFT TURN_LEFT')
                modifications['jump_to_turn_around'].append((new_cmd, new_out))
            else:
                modifications['jump_to_turn_around'].append((cmd, out))
        
        # 4. Modifier change: twice -> thrice
        modifications['twice_to_thrice'] = []
        for cmd, out in commands:
            if 'twice' in cmd:
                new_cmd = cmd.replace('twice', 'thrice')
                # Count occurrences of each action in output
                for action in ['WALK', 'RUN', 'JUMP', 'LOOK']:
                    if action in out:
                        count = out.count(action)
                        if count == 2:  # This was a 'twice' command
                            new_out = out.replace(f"{action} {action}", f"{action} {action} {action}")
                            break
                else:
                    new_out = out
                modifications['twice_to_thrice'].append((new_cmd, new_out))
            else:
                modifications['twice_to_thrice'].append((cmd, out))
        
        # 5. Connective change: and -> after
        modifications['and_to_after'] = []
        for cmd, out in commands:
            if ' and ' in cmd:
                new_cmd = cmd.replace(' and ', ' after ')
                # Reverse the order of actions
                parts = out.split()
                if len(parts) == 2:
                    new_out = f"{parts[1]} {parts[0]}"
                else:
                    new_out = out  # Complex case, keep as is
                modifications['and_to_after'].append((new_cmd, new_out))
            else:
                modifications['and_to_after'].append((cmd, out))
        
        return modifications
    
    def create_train_val_test_splits(self, commands: List[Tuple[str, str]], 
                                   ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
        """Create train/val/test splits ensuring compositional isolation"""
        # Sort commands by complexity (length)
        commands_sorted = sorted(commands, key=lambda x: len(x[0].split()))
        
        # Split by complexity to ensure test has novel compositions
        n = len(commands_sorted)
        simple = commands_sorted[:n//3]
        medium = commands_sorted[n//3:2*n//3]
        complex = commands_sorted[2*n//3:]
        
        # Create splits
        train = []
        val = []
        test = []
        
        # Add simple commands to train
        train.extend(simple[:int(len(simple)*0.8)])
        val.extend(simple[int(len(simple)*0.8):int(len(simple)*0.9)])
        test.extend(simple[int(len(simple)*0.9):])
        
        # Add medium commands
        train.extend(medium[:int(len(medium)*0.7)])
        val.extend(medium[int(len(medium)*0.7):int(len(medium)*0.85)])
        test.extend(medium[int(len(medium)*0.85):])
        
        # Complex commands mostly in test (compositional generalization)
        train.extend(complex[:int(len(complex)*0.3)])
        val.extend(complex[int(len(complex)*0.3):int(len(complex)*0.5)])
        test.extend(complex[int(len(complex)*0.5):])
        
        # Shuffle within each split
        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)
        
        return train, val, test
    
    def encode_data(self, data: List[Tuple[str, str]], max_length: int = 20):
        """Encode commands and actions to sequences"""
        encoded_inputs = []
        encoded_outputs = []
        
        for cmd, out in data:
            # Encode input
            input_tokens = ['<START>'] + cmd.split() + ['<END>']
            input_ids = [self.input_vocab.get(token, self.input_vocab['<UNK>']) 
                        for token in input_tokens]
            
            # Pad or truncate
            if len(input_ids) < max_length:
                input_ids += [self.input_vocab['<PAD>']] * (max_length - len(input_ids))
            else:
                input_ids = input_ids[:max_length]
            
            # Encode output
            output_tokens = ['<START>'] + out.split() + ['<END>']
            output_ids = [self.output_vocab.get(token, self.output_vocab['<UNK>']) 
                         for token in output_tokens]
            
            # Pad or truncate
            if len(output_ids) < max_length:
                output_ids += [self.output_vocab['<PAD>']] * (max_length - len(output_ids))
            else:
                output_ids = output_ids[:max_length]
            
            encoded_inputs.append(input_ids)
            encoded_outputs.append(output_ids)
        
        return np.array(encoded_inputs), np.array(encoded_outputs)
    
    def save_data(self, output_dir: str):
        """Generate and save all data"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating SCAN data (subset_size={self.subset_size})...")
        
        # Generate base commands
        commands = self.generate_simple_commands(n=self.subset_size or 5000)
        print(f"Generated {len(commands)} base commands")
        
        # Create splits
        train, val, test = self.create_train_val_test_splits(commands)
        print(f"Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        # Create modifications
        modifications = self.create_rule_modifications(commands)
        print(f"Created {len(modifications)} rule modification types")
        
        # Encode data
        train_x, train_y = self.encode_data(train)
        val_x, val_y = self.encode_data(val)
        test_x, test_y = self.encode_data(test)
        
        # Save everything
        data_dict = {
            'train': {'x': train_x.tolist(), 'y': train_y.tolist(), 'raw': train},
            'val': {'x': val_x.tolist(), 'y': val_y.tolist(), 'raw': val},
            'test': {'x': test_x.tolist(), 'y': test_y.tolist(), 'raw': test},
            'modifications': {name: data for name, data in modifications.items()},
            'vocab': {
                'input': self.input_vocab,
                'output': self.output_vocab
            },
            'metadata': {
                'total_examples': len(commands),
                'max_length': 20,
                'seed': self.seed,
                'generated_at': datetime.now().isoformat()
            }
        }
        
        # Save as JSON
        output_file = output_path / 'scan_data.json'
        with open(output_file, 'w') as f:
            json.dump(data_dict, f, indent=2)
        
        print(f"Data saved to: {output_file}")
        
        # Save vocabularies separately for easy access
        vocab_file = output_path / 'vocabularies.json'
        with open(vocab_file, 'w') as f:
            json.dump({
                'input_vocab': self.input_vocab,
                'output_vocab': self.output_vocab,
                'input_size': len(self.input_vocab),
                'output_size': len(self.output_vocab)
            }, f, indent=2)
        
        # Save sample data for verification
        samples_file = output_path / 'samples.txt'
        with open(samples_file, 'w') as f:
            f.write("=== Sample Training Data ===\n")
            for i, (cmd, out) in enumerate(train[:10]):
                f.write(f"{i+1}. Input: {cmd}\n")
                f.write(f"   Output: {out}\n\n")
            
            f.write("\n=== Sample Test Data (Complex) ===\n")
            complex_test = [t for t in test if len(t[0].split()) > 3][:5]
            for i, (cmd, out) in enumerate(complex_test):
                f.write(f"{i+1}. Input: {cmd}\n")
                f.write(f"   Output: {out}\n\n")
            
            f.write("\n=== Sample Modifications ===\n")
            for mod_name, mod_data in modifications.items():
                f.write(f"\n{mod_name}:\n")
                for i, (cmd, out) in enumerate(mod_data[:3]):
                    f.write(f"{i+1}. Input: {cmd}\n")
                    f.write(f"   Output: {out}\n\n")
        
        return data_dict


def main():
    """Main function for both local testing and full generation"""
    import argparse
    parser = argparse.ArgumentParser(description='Prepare SCAN data')
    parser.add_argument('--subset', type=int, default=None,
                       help='Generate only N examples (for testing)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: auto-detect)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()
    
    # Set up environment
    config = setup_environment()
    
    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = get_data_path('experiments/02_compositional_language/data')
    
    print(f"Output directory: {output_dir}")
    
    # Generate data
    generator = SCANDataGenerator(subset_size=args.subset, seed=args.seed)
    data = generator.save_data(output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("DATA GENERATION COMPLETE")
    print("="*50)
    print(f"Total examples: {data['metadata']['total_examples']}")
    print(f"Train size: {len(data['train']['x'])}")
    print(f"Val size: {len(data['val']['x'])}")
    print(f"Test size: {len(data['test']['x'])}")
    print(f"Modifications: {list(data['modifications'].keys())}")
    print(f"Input vocab size: {len(data['vocab']['input'])}")
    print(f"Output vocab size: {len(data['vocab']['output'])}")


if __name__ == "__main__":
    main()