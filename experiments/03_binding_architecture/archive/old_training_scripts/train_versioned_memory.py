#!/usr/bin/env python3
"""Implementation of versioned memory for variable rebinding.

This extends the sequential planning model to support updating variable bindings
over time, enabling patterns like:
- "X means jump do X then X means walk do X"
- "X is 5 recall X now X is 10 recall X"
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import os
import sys

# Add project root
sys.path.append(os.path.abspath('../..'))

# Import existing components
from train_sequential_action_positions import (
    VOCAB, ACTIONS, ActionPositionTracker,
    SequentialModelWithActionTracking
)
from train_sequential_planning_fixed import (
    BindingAttentionFixed as BindingAttention,
    SequencePlanner
)
from train_binding_curriculum import generate_stage3_data
from mlx_model_io import save_model_simple


class VersionedMemory(nn.Module):
    """Memory module that supports versioning for rebinding variables."""
    
    def __init__(self, num_slots: int = 4, embed_dim: int = 128, max_versions: int = 3):
        super().__init__()
        self.num_slots = num_slots
        self.embed_dim = embed_dim
        self.max_versions = max_versions
        
        # Learnable slot keys
        self.slot_keys = mx.random.normal((num_slots, embed_dim))
        
        # Version tracking - stores (slot_id, version_id, timestamp)
        self.version_encoder = nn.Linear(embed_dim + 1, embed_dim)  # +1 for timestamp
        
    def create_version_timestamp(self, position: int, max_position: int = 100) -> float:
        """Create normalized timestamp based on position in sequence."""
        return position / max_position
        
    def bind_versioned(self, 
                      var_embed: mx.array, 
                      value_embed: mx.array,
                      binding_scores: mx.array,
                      position: int,
                      current_memory: Dict) -> Dict:
        """Bind a variable to a value with versioning support.
        
        Args:
            var_embed: Variable embedding (batch, embed_dim)
            value_embed: Value embedding to bind (batch, embed_dim)
            binding_scores: Attention scores for slots (batch, num_slots)
            position: Position in sequence (for timestamp)
            current_memory: Current memory state dict
            
        Returns:
            Updated memory state with new version
        """
        batch_size = var_embed.shape[0]
        timestamp = self.create_version_timestamp(position)
        
        # Get slot assignment
        slot_id = mx.argmax(binding_scores, axis=1)  # (batch,)
        
        # Create versioned value with timestamp
        timestamp_embed = mx.ones((batch_size, 1)) * timestamp
        value_with_time = mx.concatenate([value_embed, timestamp_embed], axis=1)
        versioned_value = self.version_encoder(value_with_time)
        
        # Update memory
        new_memory = current_memory.copy()
        
        for b in range(batch_size):
            slot_idx = int(slot_id[b])
            
            # Initialize slot if needed
            if slot_idx not in new_memory:
                new_memory[slot_idx] = {
                    'versions': [],
                    'current_version': -1
                }
            
            # Add new version
            new_memory[slot_idx]['versions'].append({
                'value': versioned_value[b],
                'timestamp': timestamp,
                'position': position
            })
            
            # Update current version pointer
            new_memory[slot_idx]['current_version'] = len(new_memory[slot_idx]['versions']) - 1
            
            # Limit versions if needed
            if len(new_memory[slot_idx]['versions']) > self.max_versions:
                new_memory[slot_idx]['versions'].pop(0)
                new_memory[slot_idx]['current_version'] = min(
                    new_memory[slot_idx]['current_version'],
                    self.max_versions - 1
                )
        
        return new_memory
    
    def retrieve_versioned(self, 
                          binding_scores: mx.array,
                          memory_state: Dict,
                          position: int) -> mx.array:
        """Retrieve values using versioned memory.
        
        Args:
            binding_scores: Attention scores for slots (batch, num_slots)
            memory_state: Current memory state
            position: Current position (for recency)
            
        Returns:
            Retrieved values (batch, embed_dim)
        """
        batch_size = binding_scores.shape[0]
        
        # Build value matrix from current versions
        value_matrix = mx.zeros((batch_size, self.num_slots, self.embed_dim))
        
        for slot_idx in range(self.num_slots):
            if slot_idx in memory_state and memory_state[slot_idx]['versions']:
                # Get most recent version
                current_ver = memory_state[slot_idx]['current_version']
                if 0 <= current_ver < len(memory_state[slot_idx]['versions']):
                    value = memory_state[slot_idx]['versions'][current_ver]['value']
                    # Broadcast to batch
                    value_matrix[:, slot_idx, :] = value
        
        # Weighted retrieval
        retrieved = mx.sum(
            binding_scores[:, :, None] * value_matrix,
            axis=1
        )
        
        return retrieved


class VersionedMemoryModel(nn.Module):
    """Model with versioned memory for variable rebinding."""
    
    def __init__(self,
                 vocab_size: int = len(VOCAB),
                 num_actions: int = len(ACTIONS),
                 embed_dim: int = 128,
                 num_slots: int = 4,
                 num_heads: int = 8,
                 max_versions: int = 3):
        super().__init__()
        
        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Sequence planning
        self.sequence_planner = SequencePlanner()
        
        # Binding attention
        self.binder = BindingAttention(embed_dim, num_heads)
        
        # Versioned memory
        self.memory = VersionedMemory(num_slots, embed_dim, max_versions)
        
        # Value encoder for binding
        self.value_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_actions)
        )
        
        # Recognition decoder
        self.recognition_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2)  # Binary: is_variable or not
        )
        
        # Context encoder
        self.context_encoder = nn.LSTM(embed_dim, embed_dim)
        
        self.num_slots = num_slots
        self.embed_dim = embed_dim
    
    def detect_binding_pattern(self, tokens: List[int], position: int) -> Tuple[bool, Optional[int], Optional[int]]:
        """Detect if current position is a binding pattern.
        
        Returns:
            (is_binding, var_position, value_position)
        """
        if position + 2 < len(tokens):
            # Check for "X means ACTION" pattern
            if tokens[position + 1] == VOCAB.get('means', -1):
                return True, position, position + 2
            # Check for "X is VALUE" pattern
            if tokens[position + 1] == VOCAB.get('is', -1):
                return True, position, position + 2
        return False, None, None
    
    def process_segment_versioned(self,
                                 command_ids: mx.array,
                                 segment: List[int],
                                 memory_state: Dict,
                                 position_offset: int,
                                 stage: str) -> Dict:
        """Process a segment with versioned memory."""
        batch_size = command_ids.shape[0]
        segment_tokens = command_ids[:, segment[0]:segment[1]]
        
        # Embed tokens
        embeddings = self.token_embed(segment_tokens)
        
        # Encode context
        context, _ = self.context_encoder(embeddings)
        
        # Action position tracking
        tracker = ActionPositionTracker(VOCAB)
        segment_list = [int(segment_tokens[0, i]) for i in range(segment_tokens.shape[1])]
        action_positions = tracker.find_action_positions(segment_list)
        
        # Process each position
        recognition_outputs = []
        action_outputs = []
        action_output_positions = []
        
        for t in range(segment_tokens.shape[1]):
            current_embed = context[:, t, :]
            global_position = position_offset + t
            
            # Check for binding pattern
            is_binding, var_pos, val_pos = self.detect_binding_pattern(
                segment_list, t
            )
            
            if is_binding and var_pos is not None and val_pos is not None:
                # Handle binding
                var_embed = embeddings[:, var_pos, :]
                val_embed = embeddings[:, val_pos, :]
                
                # Get binding scores
                binding_scores = self.binder(
                    var_embed,
                    self.memory.slot_keys,
                    self.memory.slot_keys
                )
                
                # Encode value
                encoded_value = self.value_encoder(val_embed)
                
                # Update memory with version
                memory_state = self.memory.bind_versioned(
                    var_embed,
                    encoded_value,
                    binding_scores,
                    global_position,
                    memory_state
                )
                
                # Recognition output
                if stage == "recognition":
                    recognition_outputs.append(current_embed)
            
            # Check for action positions
            elif t in action_positions['positions']:
                # Retrieve from versioned memory
                binding_scores = self.binder(
                    current_embed,
                    self.memory.slot_keys,
                    self.memory.slot_keys
                )
                
                retrieved = self.memory.retrieve_versioned(
                    binding_scores,
                    memory_state,
                    global_position
                )
                
                # Handle repetitions
                repeat_count = action_positions['repeat_counts'].get(t, 1)
                for _ in range(repeat_count):
                    action_outputs.append(retrieved)
                    action_output_positions.append(global_position)
        
        return {
            'recognition_outputs': recognition_outputs,
            'action_outputs': action_outputs,
            'action_positions': action_output_positions,
            'memory_state': memory_state
        }
    
    def __call__(self, command_ids: mx.array, stage: str = "full_binding") -> Dict:
        """Forward pass with versioned memory."""
        batch_size = command_ids.shape[0]
        
        # Parse sequence
        segments = self.sequence_planner.parse_sequence(command_ids)
        
        # Initialize versioned memory
        memory_state = {}
        
        # Process segments
        all_recognition_outputs = []
        all_action_outputs = []
        all_action_positions = []
        position_offset = 0
        
        for segment in segments:
            results = self.process_segment_versioned(
                command_ids,
                segment,
                memory_state,
                position_offset,
                stage
            )
            
            all_recognition_outputs.extend(results['recognition_outputs'])
            all_action_outputs.extend(results['action_outputs'])
            all_action_positions.extend(results['action_positions'])
            
            memory_state = results['memory_state']
            position_offset = segment[1]
        
        # Prepare outputs
        outputs = {
            'memory_state': memory_state,
            'segments': segments,
            'action_positions': all_action_positions
        }
        
        if stage == "recognition" and all_recognition_outputs:
            outputs_stacked = mx.stack(all_recognition_outputs, axis=1)
            recognition_logits = self.recognition_decoder(outputs_stacked)
            outputs['recognition_logits'] = recognition_logits
        
        if stage != "recognition" and all_action_outputs:
            action_stacked = mx.stack(all_action_outputs, axis=1)
            action_logits = self.action_decoder(action_stacked)
            outputs['action_logits'] = action_logits
            outputs['num_actions'] = len(all_action_outputs)
        
        return outputs


def generate_rebinding_data(num_samples: int = 100) -> List[Dict]:
    """Generate training data with variable rebinding patterns."""
    data = []
    variables = ['X', 'Y', 'Z', 'W']
    actions = list(ACTIONS.keys())[1:]  # Skip PAD
    
    for _ in range(num_samples):
        # Choose variable and actions
        var = np.random.choice(variables)
        action1 = np.random.choice(actions)
        action2 = np.random.choice([a for a in actions if a != action1])
        
        # Create rebinding pattern
        pattern_type = np.random.choice(['basic', 'temporal', 'sequential'])
        
        if pattern_type == 'basic':
            # Basic rebinding: "X means jump do X now X means walk do X"
            command = f"{var} means {action1.lower()} do {var} now {var} means {action2.lower()} do {var}"
            expected = [action1, action2]
        
        elif pattern_type == 'temporal':
            # Rebinding with temporal: "X means jump do X twice then X means walk do X"
            command = f"{var} means {action1.lower()} do {var} twice then {var} means {action2.lower()} do {var}"
            expected = [action1, action1, action2]
        
        else:  # sequential
            # Multiple rebinding: "X means jump do X then X means walk do X then X means turn do X"
            action3 = np.random.choice([a for a in actions if a not in [action1, action2]])
            command = f"{var} means {action1.lower()} do {var} then {var} means {action2.lower()} do {var} then {var} means {action3.lower()} do {var}"
            expected = [action1, action2, action3]
        
        data.append({
            'command': command,
            'expected': expected,
            'pattern_type': pattern_type
        })
    
    return data


def test_versioned_memory():
    """Test the versioned memory implementation."""
    print("Testing versioned memory model...")
    
    # Create model
    model = VersionedMemoryModel()
    
    # Test patterns
    test_patterns = [
        {
            'command': "X means jump do X then X means walk do X",
            'expected': ['JUMP', 'WALK']
        },
        {
            'command': "Y means turn do Y now Y means run do Y",
            'expected': ['TURN', 'RUN']
        },
        {
            'command': "Z means jump do Z twice then Z means walk do Z",
            'expected': ['JUMP', 'JUMP', 'WALK']
        }
    ]
    
    for pattern in test_patterns:
        print(f"\nTesting: {pattern['command']}")
        
        # Tokenize
        tokens = pattern['command'].split()
        token_ids = mx.array([[VOCAB.get(t, VOCAB['<PAD>']) for t in tokens]])
        
        # Run model
        outputs = model(token_ids, stage="full_binding")
        
        print(f"Expected: {pattern['expected']}")
        print(f"Memory state: {len(outputs.get('memory_state', {}))} slots used")
        print(f"Action positions: {outputs.get('action_positions', [])}")
        
        if 'action_logits' in outputs:
            predictions = mx.argmax(outputs['action_logits'], axis=-1)
            print(f"Predictions shape: {predictions.shape}")
    
    print("\n✓ Versioned memory test complete!")


def main():
    """Train model with versioned memory on rebinding tasks."""
    print("Training versioned memory model for variable rebinding...")
    
    # Create model
    model = VersionedMemoryModel()
    optimizer = optim.Adam(learning_rate=1e-3)
    
    # Generate training data
    print("\nGenerating rebinding training data...")
    train_data = generate_rebinding_data(200)
    
    # Training loop would go here
    # For now, just save the model architecture
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('models', exist_ok=True)
    save_model_simple(f"models/versioned_memory_{timestamp}.pkl", model)
    
    print(f"\nVersioned memory model saved!")
    
    # Test on example patterns
    print("\nTesting on rebinding patterns:")
    test_versioned_memory()


if __name__ == "__main__":
    main()