"""
Train variable binding model with proper action position tracking.
This fixes the output interpretation issue by only generating predictions at action positions.
"""

from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
from utils.paths import get_data_path, get_output_path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import json
import os
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm
from datetime import datetime

from train_binding_curriculum import (
    generate_stage1_data, generate_stage2_data, generate_stage3_data,
    VOCAB, ACTIONS
)
from train_temporal_curriculum import TemporalActionBuffer
from train_sequential_planning_fixed import (
    SequencePlanner, BindingAttentionFixed, create_sequential_dataset
)

config = setup_environment()


class ActionPositionTracker:
    """Track positions where actions should be generated."""
    
    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.do_token = vocab.get('do', -1)
        self.var_tokens = {vocab.get('X', -1), vocab.get('Y', -1), vocab.get('Z', -1)}
        self.twice_token = vocab.get('twice', -1)
        self.thrice_token = vocab.get('thrice', -1)
    
    def find_action_positions(self, tokens: mx.array) -> List[Tuple[int, int]]:
        """Find positions where actions should be generated.
        
        Returns list of (position, repeat_count) tuples.
        """
        action_positions = []
        
        # Convert to numpy for easier manipulation
        if hasattr(tokens, 'numpy'):
            tokens_np = tokens.numpy()
        else:
            tokens_np = np.array(tokens)
        
        # Handle batch dimension
        if len(tokens_np.shape) > 1:
            tokens_np = tokens_np[0]
        
        i = 0
        while i < len(tokens_np):
            # Look for "do VARIABLE" pattern
            if tokens_np[i] == self.do_token and i + 1 < len(tokens_np):
                if tokens_np[i + 1] in self.var_tokens:
                    # Check for temporal modifier
                    repeat_count = 1
                    if i + 2 < len(tokens_np):
                        if tokens_np[i + 2] == self.twice_token:
                            repeat_count = 2
                        elif tokens_np[i + 2] == self.thrice_token:
                            repeat_count = 3
                    
                    action_positions.append((i + 1, repeat_count))
                    i += 2  # Skip past the variable
                else:
                    i += 1
            else:
                i += 1
        
        return action_positions


class SequentialModelWithActionTracking(nn.Module):
    """Variable binding model that only outputs at action positions."""
    
    def __init__(self, vocab_size: int, num_actions: int, embed_dim: int = 256, 
                 num_slots: int = 4, num_heads: int = 8, mlp_hidden_dim: int = 512):
        super().__init__()
        
        # Core components
        self.vocab_size = vocab_size
        self.num_actions = num_actions
        self.embed_dim = embed_dim
        self.num_slots = num_slots
        
        # Embeddings and encoding
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(50, embed_dim)
        
        # Sequence planning and action tracking
        self.sequence_planner = SequencePlanner()
        self.action_tracker = ActionPositionTracker(VOCAB)
        
        # Memory components
        self.slot_keys = mx.random.normal((num_slots, embed_dim))
        self.binder = BindingAttentionFixed(embed_dim, num_heads)
        
        # Prediction heads
        self.recognition_decoder = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, vocab_size)
        )
        
        self.action_decoder = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, num_actions)
        )
    
    def process_segment_with_tracking(self, command_ids: mx.array, segment: Tuple[int, int], 
                                    slot_values: mx.array, bindings: Dict, stage: str) -> Dict:
        """Process a segment and track action positions."""
        # Extract segment tokens
        segment_tokens = command_ids[:, segment[0]:segment[1]]
        segment_length = segment[1] - segment[0]
        
        # Find action positions in this segment
        action_positions = self.action_tracker.find_action_positions(segment_tokens)
        
        # Storage for outputs
        recognition_outputs = []
        action_outputs = []
        action_output_positions = []
        
        # Process each token
        for t in range(segment_length):
            # Get embeddings
            token_embed = self.token_embeddings(segment_tokens[:, t:t+1])
            pos_embed = self.position_embeddings(mx.array([t]))
            current_embed = token_embed + pos_embed
            
            # Check for storage pattern
            is_storage = False
            if t + 2 < segment_length:
                is_means = segment_tokens[:, t+1] == VOCAB['means']
                is_is = segment_tokens[:, t+1] == VOCAB['is']
                is_storage = mx.any(is_means | is_is)
            
            # Compute binding scores
            binding_scores, raw_scores = self.binder(
                current_embed, 
                self.slot_keys,
                training=(stage != "recognition")
            )
            
            # Update bindings if storage pattern
            if is_storage:
                var_id = segment_tokens[0, t].item()
                slot_idx = mx.argmax(binding_scores[0]).item()
                bindings[var_id] = slot_idx
                
                # Store value in slot
                if t + 2 < segment_length:
                    value_embed = self.token_embeddings(segment_tokens[:, t+2:t+3])
                    mask = mx.arange(self.num_slots) == slot_idx
                    mask = mask[None, :, None]
                    
                    if len(value_embed.shape) == 3:
                        value_embed = value_embed.squeeze(1)
                    
                    value_broadcast = mx.zeros_like(slot_values)
                    value_broadcast = mx.where(
                        mask,
                        value_embed[:, None, :],
                        value_broadcast
                    )
                    
                    slot_values = mx.where(mask, value_broadcast, slot_values)
            
            # Handle recognition stage
            if stage == "recognition":
                if current_embed.shape[1] == 1:
                    output = current_embed.squeeze(1)
                else:
                    output = current_embed[:, 0, :]
                recognition_outputs.append(output)
            
            # Check if this is an action position
            for pos, repeat_count in action_positions:
                if t == pos:
                    # Retrieve value for this variable
                    var_id = segment_tokens[0, t].item()
                    if var_id in bindings:
                        slot_idx = bindings[var_id]
                        one_hot_binding = mx.zeros((1, self.num_slots))
                        one_hot_binding = mx.where(
                            mx.arange(self.num_slots) == slot_idx,
                            1.0,
                            one_hot_binding
                        )
                        retrieved = (one_hot_binding[:, None, :] @ slot_values).squeeze(1)
                        
                        # Add outputs for each repetition
                        for _ in range(repeat_count):
                            action_outputs.append(retrieved)
                            action_output_positions.append(segment[0] + t)
        
        return {
            'recognition_outputs': recognition_outputs,
            'action_outputs': action_outputs,
            'action_positions': action_output_positions,
            'slot_values': slot_values,
            'bindings': bindings
        }
    
    def __call__(self, command_ids: mx.array, action_ids: Optional[mx.array] = None, 
                 stage: str = "recognition") -> Dict:
        batch_size = command_ids.shape[0]
        
        # Parse sequence into segments
        segments = self.sequence_planner.parse_sequence(command_ids)
        
        # Initialize memory
        slot_values = mx.zeros((batch_size, self.num_slots, self.embed_dim))
        bindings = {}
        
        # Process each segment
        all_recognition_outputs = []
        all_action_outputs = []
        all_action_positions = []
        
        for segment in segments:
            segment_results = self.process_segment_with_tracking(
                command_ids, segment, slot_values, bindings, stage
            )
            
            all_recognition_outputs.extend(segment_results['recognition_outputs'])
            all_action_outputs.extend(segment_results['action_outputs'])
            all_action_positions.extend(segment_results['action_positions'])
            
            # Update memory state
            slot_values = segment_results['slot_values']
            bindings = segment_results['bindings']
        
        # Prepare outputs based on stage
        outputs = {
            'bindings': bindings,
            'slot_values': slot_values,
            'segments': segments,
            'action_positions': all_action_positions
        }
        
        if stage == "recognition" and all_recognition_outputs:
            # Stack recognition outputs
            outputs_stacked = mx.stack(all_recognition_outputs, axis=1)
            recognition_logits = self.recognition_decoder(outputs_stacked)
            outputs['recognition_logits'] = recognition_logits
        
        if stage != "recognition" and all_action_outputs:
            # Stack action outputs
            action_stacked = mx.stack(all_action_outputs, axis=1)
            action_logits = self.action_decoder(action_stacked)
            outputs['action_logits'] = action_logits
            outputs['num_actions'] = len(all_action_outputs)
        
        return outputs


def extract_action_predictions_improved(model, command: str) -> List[str]:
    """Extract action predictions using the improved model."""
    # Tokenize command
    command_tokens = command.split()
    command_ids = mx.array([[VOCAB.get(token, VOCAB['<PAD>']) for token in command_tokens]])
    
    # Get model outputs
    outputs = model(command_ids, stage="full_binding")
    
    if 'action_logits' not in outputs or outputs.get('num_actions', 0) == 0:
        return []
    
    # Get predictions - now only at action positions
    action_logits = outputs['action_logits']
    predictions = mx.argmax(action_logits, axis=-1)
    
    # Convert predictions to action names
    predicted_actions = []
    predictions_np = np.array(predictions)
    
    for j in range(predictions_np.shape[1]):
        pred_id = int(predictions_np[0, j])
        for action_name, action_id in ACTIONS.items():
            if action_id == pred_id and action_name != '<PAD>':
                predicted_actions.append(action_name)
                break
    
    return predicted_actions


def train_step_improved(model: SequentialModelWithActionTracking, batch: Dict, 
                       optimizer: optim.Optimizer, stage: str) -> Dict:
    """Training step for improved model."""
    
    def loss_fn(model, command_ids, action_ids, stage):
        # Forward pass
        outputs = model(command_ids, stage=stage)
        
        # Compute loss based on stage
        if stage == "recognition" and 'recognition_logits' in outputs:
            # Recognition loss (same as before)
            var_mask = mx.logical_or(
                command_ids == VOCAB['X'],
                mx.logical_or(command_ids == VOCAB['Y'], command_ids == VOCAB['Z'])
            )
            
            if mx.any(var_mask):
                recog_logits = outputs['recognition_logits']
                
                # Ensure compatible shapes
                batch_size, cmd_len = command_ids.shape
                pred_len = recog_logits.shape[1]
                min_len = min(cmd_len, pred_len)
                
                recog_logits_truncated = recog_logits[:, :min_len, :]
                command_ids_truncated = command_ids[:, :min_len]
                var_mask_truncated = var_mask[:, :min_len]
                
                # Compute masked loss
                logits_flat = recog_logits_truncated.reshape(-1, recog_logits_truncated.shape[-1])
                targets_flat = command_ids_truncated.reshape(-1)
                mask_flat = var_mask_truncated.reshape(-1)
                
                all_losses = nn.losses.cross_entropy(
                    logits_flat,
                    targets_flat,
                    reduction='none'
                )
                
                masked_losses = all_losses * mask_flat
                num_valid = mx.sum(mask_flat)
                
                if num_valid > 0:
                    loss = mx.sum(masked_losses) / num_valid
                else:
                    loss = mx.array(0.0)
            else:
                loss = mx.array(0.0)
                
        elif 'action_logits' in outputs and outputs.get('num_actions', 0) > 0:
            # Action prediction loss - now only for actual action positions
            action_logits = outputs['action_logits']
            num_actions = outputs['num_actions']
            
            # Extract only the relevant action targets
            # This is simplified since we now only output at action positions
            action_targets = action_ids[:, :num_actions]
            
            # Mask out padding
            mask = action_targets != ACTIONS['<PAD>']
            
            if mx.any(mask):
                logits_flat = action_logits.reshape(-1, action_logits.shape[-1])
                targets_flat = action_targets.reshape(-1)
                mask_flat = mask.reshape(-1)
                
                all_losses = nn.losses.cross_entropy(
                    logits_flat,
                    targets_flat,
                    reduction='none'
                )
                
                masked_losses = all_losses * mask_flat
                num_valid = mx.sum(mask_flat)
                
                if num_valid > 0:
                    loss = mx.sum(masked_losses) / num_valid
                else:
                    loss = mx.array(0.0)
            else:
                loss = mx.array(0.0)
        else:
            loss = mx.array(0.0)
        
        return loss
    
    # Compute loss and gradients
    loss_and_grad_fn = mx.value_and_grad(loss_fn)
    loss, grads = loss_and_grad_fn(
        model, batch['command_ids'], batch['action_ids'], stage
    )
    
    # Update model
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    
    return {
        'loss': loss.item() if hasattr(loss, 'item') else float(loss),
        'stage': stage
    }


def test_improved_model():
    """Test the improved model with action position tracking."""
    print("Testing improved model with action position tracking...")
    
    # Initialize improved model
    model = SequentialModelWithActionTracking(
        vocab_size=len(VOCAB),
        num_actions=len(ACTIONS),
        embed_dim=128,
        num_slots=4,
        num_heads=4,
        mlp_hidden_dim=256
    )
    
    # Test patterns
    test_cases = [
        ("X means jump do X", ["JUMP"]),
        ("X means jump do X twice", ["JUMP", "JUMP"]),
        ("X means jump do X then Y means walk do Y", ["JUMP", "WALK"]),
        ("X means jump do X twice then Y means walk do Y", ["JUMP", "JUMP", "WALK"]),
        ("Z means turn do Z then X means run do X thrice", ["TURN", "RUN", "RUN", "RUN"]),
    ]
    
    print("\n" + "="*70)
    print("TESTING ACTION POSITION TRACKING")
    print("="*70)
    
    for command, expected in test_cases:
        command_tokens = command.split()
        command_ids = mx.array([[VOCAB.get(token, VOCAB['<PAD>']) for token in command_tokens]])
        
        # Get outputs
        outputs = model(command_ids, stage="full_binding")
        
        print(f"\nCommand: {command}")
        print(f"Expected actions: {expected}")
        print(f"Action positions identified: {outputs.get('action_positions', [])}")
        print(f"Number of actions: {outputs.get('num_actions', 0)}")
        
        if 'action_logits' in outputs:
            shape = outputs['action_logits'].shape
            print(f"Action logits shape: {shape}")
            print(f"✓ Correct shape!" if shape[1] == len(expected) else "✗ Shape mismatch!")
    
    print("\n" + "="*70)


def main():
    """Train the improved model."""
    # First test the model structure
    test_improved_model()
    
    # Configuration
    num_epochs = 10
    batch_size = 16
    learning_rate = 0.001
    num_samples = 500
    eval_interval = 2
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = create_sequential_dataset(num_samples=num_samples)
    
    # Initialize improved model
    model = SequentialModelWithActionTracking(
        vocab_size=len(VOCAB),
        num_actions=len(ACTIONS),
        embed_dim=128,
        num_slots=4,
        num_heads=4,
        mlp_hidden_dim=256
    )
    
    # Initialize optimizer
    optimizer = optim.Adam(learning_rate=learning_rate)
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        # Create random batches
        indices = np.random.permutation(len(dataset['commands']))
        
        for i in tqdm(range(0, len(indices), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_indices = indices[i:i+batch_size]
            
            # Prepare batch
            batch_commands = [dataset['commands'][idx] for idx in batch_indices]
            batch_actions = [dataset['actions'][idx] for idx in batch_indices]
            
            # Convert to token IDs
            max_cmd_len = max(len(cmd.split()) for cmd in batch_commands)
            max_act_len = max(len(acts) for acts in batch_actions)
            
            command_ids = []
            action_ids = []
            
            for cmd, acts in zip(batch_commands, batch_actions):
                # Tokenize command
                tokens = cmd.split()
                cmd_ids = [VOCAB.get(token, VOCAB['<PAD>']) for token in tokens]
                cmd_ids += [VOCAB['<PAD>']] * (max_cmd_len - len(cmd_ids))
                command_ids.append(cmd_ids)
                
                # Tokenize actions
                act_ids = [ACTIONS.get(act, ACTIONS['<PAD>']) for act in acts]
                act_ids += [ACTIONS['<PAD>']] * (max_act_len - len(act_ids))
                action_ids.append(act_ids)
            
            batch = {
                'command_ids': mx.array(command_ids),
                'action_ids': mx.array(action_ids)
            }
            
            # Train on both stages
            stage = "full_binding" if epoch > 2 else "recognition"
            
            # Train step
            step_results = train_step_improved(model, batch, optimizer, stage)
            epoch_losses.append(step_results['loss'])
        
        # Print epoch summary
        avg_loss = np.mean(epoch_losses)
        print(f"\nEpoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Evaluate periodically
        if (epoch + 1) % eval_interval == 0:
            print("\nEvaluating...")
            correct = 0
            total = 0
            
            for i in range(min(100, len(dataset['commands']))):
                command = dataset['commands'][i]
                expected_actions = dataset['actions'][i]
                
                predicted = extract_action_predictions_improved(model, command)
                
                if predicted == expected_actions:
                    correct += 1
                total += 1
            
            accuracy = correct / total if total > 0 else 0
            print(f"Accuracy: {accuracy*100:.1f}% ({correct}/{total})")
    
    # Final evaluation with detailed output
    print("\n" + "="*50)
    print("FINAL EVALUATION - IMPROVED MODEL")
    print("="*50)
    
    test_patterns = [
        "X means jump do X",
        "X means jump do X then Y means walk do Y", 
        "Z means turn do Z twice then X means run do X",
        "X means jump do X twice then Y means walk do Y then Z means turn do Z thrice",
    ]
    
    for pattern in test_patterns:
        predicted = extract_action_predictions_improved(model, pattern)
        print(f"\nCommand: {pattern}")
        print(f"Predicted: {predicted}")
    
    # Save improved model using pickle for reliability
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = get_output_path('models')
    os.makedirs(output_dir, exist_ok=True)
    
    # Import save function
    from mlx_model_io import save_model_simple
    
    model_path = os.path.join(output_dir, f'sequential_improved_{timestamp}.pkl')
    save_model_simple(model_path, model)
    print(f"\nImproved model saved to: {model_path}")


if __name__ == "__main__":
    main()