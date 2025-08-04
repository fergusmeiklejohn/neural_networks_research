#!/usr/bin/env python3
"""
Dynamic Memory Architecture for Variable Binding

Key innovation: slot_values are dynamically set based on input,
not static learnable parameters.
"""

import argparse
import time
from typing import Dict, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# Actions mapping
ACTIONS = {"WALK": 0, "JUMP": 1, "TURN": 2, "LOOK": 3, "RUN": 4, "<PAD>": 5}

# Vocabulary - extended for curriculum stages
VOCAB = {
    "<PAD>": 0,
    "X": 1,
    "Y": 2,
    "Z": 3,
    "means": 4,
    "do": 5,
    "walk": 6,
    "jump": 7,
    "turn": 8,
    "look": 9,
    "run": 10,
    "twice": 11,
    "thrice": 12,
    "what": 13,
    "is": 14,
    "recall": 15,
}


class DynamicMemoryModel(nn.Module):
    """Variable binding model with dynamic memory storage"""

    def __init__(
        self,
        vocab_size: int = len(VOCAB),
        num_actions: int = len(ACTIONS),
        embed_dim: int = 128,
        num_slots: int = 4,
        num_heads: int = 8,
        initial_temperature: float = 1.0,
    ):
        super().__init__()

        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # Binding attention mechanism
        self.binder = BindingAttention(embed_dim, num_heads, initial_temperature)

        # Memory slots - keys are learnable, values are dynamic
        self.slot_keys = mx.random.normal((num_slots, embed_dim))

        # Value extractor - extracts value from context
        self.value_extractor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )

        # Pattern detector - identifies storage patterns
        self.pattern_detector = nn.Linear(
            embed_dim * 3, 2
        )  # Binary: is_storage_pattern

        # Action decoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_actions),
        )

    def detect_storage_patterns(
        self, word_embeds: mx.array, command_ids: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """
        Detect patterns like "X is jump" or "X means jump"
        Returns:
            storage_mask: (batch, seq_len) - True where storage happens
            value_positions: (batch, seq_len) - Position of value to store
        """
        batch_size, seq_len = command_ids.shape

        # Simple pattern detection based on tokens
        # Look for "is" or "means" tokens
        is_tokens = (command_ids == VOCAB["is"]) | (command_ids == VOCAB["means"])

        # Storage happens at the position before "is/means"
        storage_mask = mx.zeros((batch_size, seq_len))
        value_positions = mx.zeros((batch_size, seq_len), dtype=mx.int32)

        for i in range(seq_len - 2):
            # If we see "VAR is/means VALUE", mark VAR position for storage
            var_is_variable = mx.logical_or(
                command_ids[:, i] == VOCAB["X"],
                mx.logical_or(
                    command_ids[:, i] == VOCAB["Y"], command_ids[:, i] == VOCAB["Z"]
                ),
            )
            next_is_storage = is_tokens[:, i + 1]

            # Mark positions where storage should happen
            should_store = var_is_variable & next_is_storage
            storage_mask = mx.where(
                should_store[:, None] & (mx.arange(seq_len) == i), 1.0, storage_mask
            )

            # Mark value positions (position after is/means)
            value_positions = mx.where(
                should_store[:, None] & (mx.arange(seq_len) == i),
                i + 2,  # Value is 2 positions after variable
                value_positions,
            )

        return storage_mask, value_positions

    def detect_action_positions(self, command_ids: mx.array) -> mx.array:
        """
        Detect positions where actions should be predicted
        Returns mask of shape (batch, seq_len) indicating action positions
        """
        batch_size, seq_len = command_ids.shape
        action_mask = mx.zeros((batch_size, seq_len))

        # Actions occur after "do" and when seeing variables after "do"
        for b in range(batch_size):
            seen_do = False
            for i in range(seq_len):
                token = command_ids[b, i].item()

                if token == VOCAB["do"]:
                    seen_do = True
                elif seen_do and token in [VOCAB["X"], VOCAB["Y"], VOCAB["Z"]]:
                    # Mark this position for action prediction
                    action_mask = mx.where(
                        (mx.arange(batch_size)[:, None] == b)
                        & (mx.arange(seq_len) == i),
                        1.0,
                        action_mask,
                    )

        return action_mask

    def __call__(
        self, command_ids: mx.array, stage: str = "full", training: bool = True
    ) -> Dict[str, mx.array]:
        """
        Forward pass with dynamic memory updates
        """
        batch_size, seq_len = command_ids.shape

        # Embed tokens
        word_embeds = self.token_embed(command_ids)  # (batch, seq_len, embed_dim)

        # Detect storage patterns
        storage_mask, value_positions = self.detect_storage_patterns(
            word_embeds, command_ids
        )

        # Initialize dynamic slot values
        slot_values = mx.zeros(
            (batch_size, self.slot_keys.shape[0], self.slot_keys.shape[1])
        )

        # Process sequence with dynamic updates
        outputs = []

        for t in range(seq_len):
            # Get current token embedding
            current_embed = word_embeds[:, t : t + 1, :]  # (batch, 1, embed_dim)

            # Get bindings for current token
            bindings, binding_scores = self.binder(
                current_embed, self.slot_keys, training=training
            )  # bindings: (batch, 1), binding_scores: (batch, 1, num_slots)

            # Check if this position stores a value
            stores_value = storage_mask[:, t : t + 1]  # (batch, 1)

            if mx.sum(stores_value) > 0:
                # Extract value embeddings for positions that store
                batch_indices = mx.arange(batch_size)
                value_pos = value_positions[:, t]  # Position of value to store

                # Get value embeddings
                value_embeds = word_embeds[
                    batch_indices, value_pos
                ]  # (batch, embed_dim)
                processed_values = self.value_extractor(
                    value_embeds
                )  # (batch, embed_dim)

                # Update slot values using soft attention
                # binding_scores: (batch, 1, num_slots)
                # processed_values: (batch, embed_dim)
                # We need to update slot_values using soft assignment
                update_weights = (
                    binding_scores.squeeze(1) * stores_value
                )  # (batch, num_slots)
                updates = (
                    update_weights[:, :, None] * processed_values[:, None, :]
                )  # (batch, num_slots, embed_dim)
                slot_values = slot_values + updates

            # Retrieve values using current bindings
            # binding_scores: (batch, 1, num_slots)
            # slot_values: (batch, num_slots, embed_dim)
            # Compute weighted sum: scores @ values
            retrieved = (binding_scores @ slot_values).squeeze(1)  # (batch, embed_dim)

            # Decode to action
            if stage == "recognition":
                # For stage 1, we don't use retrieval
                output = current_embed.squeeze(1)
            else:
                # For stages 2 and 3, use retrieved values
                output = retrieved

            outputs.append(output)

        # Stack outputs
        outputs = mx.stack(outputs, axis=1)  # (batch, seq_len, embed_dim)

        # Final decoding
        if stage == "recognition":
            # Use a special head for recognition
            recognition_logits = self.decoder(outputs)  # Reuse decoder for simplicity
            return {
                "recognition_logits": recognition_logits,
                "bindings": bindings,
                "slot_values": slot_values,
                "storage_mask": storage_mask,
            }
        else:
            action_logits = self.decoder(outputs)
            return {
                "action_logits": action_logits,
                "bindings": bindings,
                "slot_values": slot_values,
                "storage_mask": storage_mask,
            }


class BindingAttention(nn.Module):
    """Attention mechanism for binding variables to slots"""

    def __init__(
        self, embed_dim: int = 128, num_heads: int = 8, initial_temperature: float = 1.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.temperature = initial_temperature

    def gumbel_softmax(
        self, logits: mx.array, temperature: float, hard: bool = False
    ) -> mx.array:
        """Gumbel-Softmax for differentiable discrete sampling"""
        shape = tuple(logits.shape)
        gumbel_noise = -mx.log(-mx.log(mx.random.uniform(shape=shape) + 1e-8) + 1e-8)
        y = logits + gumbel_noise
        y = mx.softmax(y / temperature, axis=-1)

        if hard:
            # Straight-through estimator
            y_hard = mx.zeros_like(y)
            indices = mx.argmax(y, axis=-1, keepdims=True)
            ones = mx.ones(indices.shape)
            y_hard = mx.put_along_axis(y_hard, indices, ones, axis=-1)
            y = mx.stop_gradient(y_hard - y) + y

        return y

    def __call__(
        self, words: mx.array, slot_keys: mx.array, training: bool = True
    ) -> Tuple[mx.array, mx.array]:
        """
        Compute bindings between words and slots
        """
        batch_size, seq_len, embed_dim = words.shape
        num_slots = slot_keys.shape[0]

        # Project to multi-head format
        Q = self.q_proj(words).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        Q = Q.transpose(0, 2, 1, 3)  # (batch, heads, seq_len, head_dim)

        K = self.k_proj(slot_keys).reshape(num_slots, self.num_heads, self.head_dim)
        K = mx.broadcast_to(
            K[None, :, :, :], (batch_size, num_slots, self.num_heads, self.head_dim)
        )
        K = K.transpose(0, 2, 1, 3)  # (batch, heads, num_slots, head_dim)

        # Compute attention scores
        scores = (
            Q @ K.transpose(0, 1, 3, 2)
        ) * self.scale  # (batch, heads, seq_len, num_slots)
        scores = mx.mean(scores, axis=1)  # (batch, seq_len, num_slots)

        # Apply Gumbel-Softmax for differentiable selection
        soft_scores = self.gumbel_softmax(scores, self.temperature, hard=training)

        # Get hard bindings for analysis
        bindings = mx.argmax(scores, axis=-1)  # (batch, seq_len)

        return bindings, soft_scores


# Import data generation functions from curriculum training
def generate_stage2_data(
    batch_size: int = 32, max_examples: int = 10000
) -> Dict[str, mx.array]:
    """
    Stage 2: Direct Retrieval
    Pattern: "X is jump recall X" -> JUMP
    """
    commands = []
    labels = []

    variables = ["X", "Y", "Z"]
    actions = ["walk", "jump", "turn", "look", "run"]

    for _ in range(min(batch_size, max_examples)):
        var = np.random.choice(variables)
        action = np.random.choice(actions)

        # Build command: "X is jump recall X"
        cmd = [var, "is", action, "recall", var]
        label = [ACTIONS[action.upper()]]

        # Convert to ids
        cmd_ids = [VOCAB.get(token, VOCAB["<PAD>"]) for token in cmd]
        commands.append(cmd_ids)
        labels.append(label)

    # Pad sequences
    max_cmd_len = max(len(cmd) for cmd in commands)
    padded_commands = []
    padded_labels = []

    for cmd, label in zip(commands, labels):
        padded_cmd = cmd + [VOCAB["<PAD>"]] * (max_cmd_len - len(cmd))
        padded_label = label + [ACTIONS["<PAD>"]] * (1 - len(label))
        padded_commands.append(padded_cmd)
        padded_labels.append(padded_label)

    return {
        "command": mx.array(padded_commands, dtype=mx.int32),
        "labels": mx.array(padded_labels, dtype=mx.int32),
    }


def train_step(model, batch, stage, optimizer):
    """Single training step"""

    def loss_fn(model):
        outputs = model(batch["command"], stage=stage, training=True)

        # For stage 2, we only care about the last token prediction
        logits = outputs["action_logits"]
        labels = batch["labels"]

        # Find last non-pad position for each sequence
        mask = batch["command"] != VOCAB["<PAD>"]
        last_positions = mx.sum(mask, axis=1) - 1
        batch_indices = mx.arange(logits.shape[0])

        # Get logits at last positions
        last_logits = logits[batch_indices, last_positions]  # (batch, num_actions)
        # Labels are already single actions for stage 2
        loss = mx.mean(nn.losses.cross_entropy(last_logits, labels.squeeze()))

        return loss

    loss_and_grad_fn = mx.value_and_grad(loss_fn)
    loss, grads = loss_and_grad_fn(model)
    optimizer.update(model, grads)
    mx.eval(loss)

    return loss.item()


def evaluate_stage2(model, num_eval: int = 100) -> float:
    """Evaluate performance on Stage 2"""
    model.eval()
    correct = 0
    total = 0

    for _ in range(num_eval // 32):
        batch = generate_stage2_data(32)
        outputs = model(batch["command"], stage="retrieval", training=False)

        # Check action predictions
        logits = outputs["action_logits"]

        # Get prediction at last position
        mask = batch["command"] != VOCAB["<PAD>"]
        last_positions = mx.sum(mask, axis=1) - 1
        batch_indices = mx.arange(logits.shape[0])
        last_logits = logits[batch_indices, last_positions]

        predictions = mx.argmax(last_logits, axis=-1)
        labels = batch["labels"].squeeze()

        matches = predictions == labels
        correct += mx.sum(matches).item()
        total += labels.shape[0]

    return correct / total if total > 0 else 0.0


def test_specific_examples(model):
    """Test model on specific Stage 2 examples"""
    print("\nTesting specific examples:")
    print("=" * 50)

    test_cases = [
        ("X is jump recall X", "JUMP"),
        ("Y is walk recall Y", "WALK"),
        ("Z is turn recall Z", "TURN"),
        ("X is run recall X", "RUN"),
        ("Y is look recall Y", "LOOK"),
    ]

    id_to_action = {v: k for k, v in ACTIONS.items()}

    successes = 0

    for cmd_str, expected in test_cases:
        # Encode command
        tokens = cmd_str.split()
        cmd_ids = [VOCAB.get(token, VOCAB["<PAD>"]) for token in tokens]
        cmd_batch = mx.array([cmd_ids], dtype=mx.int32)

        # Get prediction
        outputs = model(cmd_batch, stage="retrieval", training=False)
        logits = outputs["action_logits"]

        # Get prediction at last position
        last_pos = len(cmd_ids) - 1
        pred_logits = logits[0, last_pos]
        pred_id = mx.argmax(pred_logits).item()
        predicted = id_to_action.get(pred_id, "?")

        # Check bindings and storage
        bindings = outputs["bindings"][0].tolist()
        storage_mask = outputs["storage_mask"][0].tolist()

        print(f"\nCommand: {cmd_str}")
        print(f"  Expected: {expected}")
        print(f"  Predicted: {predicted}")
        print(f"  Bindings: {bindings}")
        print(
            f"  Storage positions: {[i for i, m in enumerate(storage_mask) if m > 0]}"
        )

        # Print slot values for debugging
        slot_values = outputs["slot_values"][0]  # (num_slots, embed_dim)
        slot_norms = mx.sqrt(mx.sum(slot_values * slot_values, axis=1))
        print(f"  Slot value norms: {slot_norms.tolist()}")

        success = predicted == expected
        print(f"  Success: {'✓' if success else '✗'}")
        if success:
            successes += 1

    print(
        f"\nSuccess rate: {successes}/{len(test_cases)} = {successes/len(test_cases)*100:.1f}%"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Dynamic memory model for variable binding"
    )
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_slots", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # Initialize model
    model = DynamicMemoryModel(
        vocab_size=len(VOCAB),
        num_actions=len(ACTIONS),
        embed_dim=args.embed_dim,
        num_slots=args.num_slots,
        num_heads=args.num_heads,
        initial_temperature=1.0,
    )
    mx.eval(model.parameters())

    # Initialize optimizer
    optimizer = optim.Adam(learning_rate=args.lr)

    # Temperature annealing
    initial_temperature = 1.0
    min_temperature = 0.1
    temperature_decay = 0.95

    print("Training Dynamic Memory Model on Stage 2")
    print("=" * 60)

    best_accuracy = 0.0

    for epoch in range(args.epochs):
        # Update temperature
        current_temp = max(
            min_temperature, initial_temperature * (temperature_decay**epoch)
        )
        model.binder.temperature = current_temp

        # Training
        epoch_loss = 0.0
        num_batches = 100

        start_time = time.time()
        for _ in range(num_batches):
            batch = generate_stage2_data(args.batch_size)
            loss = train_step(model, batch, "retrieval", optimizer)
            epoch_loss += loss

        avg_loss = epoch_loss / num_batches

        # Evaluation
        accuracy = evaluate_stage2(model)

        print(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}, "
            f"Temperature: {current_temp:.3f}, Time: {time.time()-start_time:.2f}s"
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"  New best accuracy: {accuracy:.2%}")

    print(f"\nFinal accuracy: {best_accuracy:.2%}")

    # Test specific examples
    test_specific_examples(model)


if __name__ == "__main__":
    main()
