"""
Quick test of sequence planning functionality.
"""

from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
import mlx.core as mx
import numpy as np

from train_sequential_planning import SequencePlanner, VOCAB

config = setup_environment()

# Ensure 'then' is in vocabulary
if 'then' not in VOCAB:
    VOCAB['then'] = len(VOCAB)

# Create sequence planner
planner = SequencePlanner()

# Test patterns
test_patterns = [
    "X means jump do X then Y means walk do Y",
    "Z means turn do Z twice then X means run do X", 
    "Y means walk do Y then Z means turn do Z thrice",
    "X means jump do X",  # No 'then'
]

print("Testing Sequence Planning:")
print("=" * 50)

for pattern in test_patterns:
    print(f"\nPattern: {pattern}")
    
    # Tokenize
    tokens = pattern.split()
    token_ids = [VOCAB.get(token, VOCAB['<PAD>']) for token in tokens]
    token_array = mx.array(token_ids)
    
    # Parse segments
    segments = planner.parse_sequence(token_array)
    
    print(f"Segments: {segments}")
    
    # Show each segment
    for i, (start, end) in enumerate(segments):
        segment_tokens = tokens[start:end]
        print(f"  Segment {i+1}: {' '.join(segment_tokens)}")

print("\n" + "=" * 50)
print("Sequence planning test complete!")