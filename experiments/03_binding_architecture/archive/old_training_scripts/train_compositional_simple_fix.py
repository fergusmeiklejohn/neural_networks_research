#!/usr/bin/env python3
"""
Simple fix: Add explicit binding memory to help the model learn associations.
Instead of trying to learn bindings implicitly, we'll provide them as additional input.
"""

import logging
from datetime import datetime
from typing import Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define vocabulary and actions
VOCAB = {
    "<PAD>": 0,
    "<SOS>": 1,
    "<EOS>": 2,
    "jump": 3,
    "walk": 4,
    "turn": 5,
    "run": 6,
    "means": 7,
    "do": 8,
    "twice": 9,
    "thrice": 10,
    "X": 11,
    "Y": 12,
    "Z": 13,
    "W": 14,
    "and": 15,
    "then": 16,
    "while": 17,
    "or": 18,
    "true": 19,
}

ACTIONS = {"<PAD>": 0, "JUMP": 1, "WALK": 2, "TURN": 3, "RUN": 4}

# Import the fixed parser
from compositional_operators_fixed import (
    CompositionalExecutor,
    CompositionalParser,
)


class SimpleBindingModel(nn.Module):
    """Simple model that uses explicit binding information."""

    def __init__(self, vocab_size: int, num_actions: int, embed_dim: int = 128):
        super().__init__()

        # Token embeddings
        self.embed = nn.Embedding(vocab_size, embed_dim)

        # Variable embeddings (for X, Y, Z, W)
        self.var_embed = nn.Embedding(4, embed_dim)

        # Action embeddings
        self.action_embed = nn.Embedding(num_actions, embed_dim)

        # Simple MLP to predict action given variable and binding context
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_actions),
        )

        # Components
        self.parser = CompositionalParser(VOCAB)
        self.executor = CompositionalExecutor(self, VOCAB)

    def __call__(self, inputs: Dict[str, mx.array]) -> mx.array:
        """Forward pass with explicit binding handling."""
        command_ids = inputs["command"]

        # Parse and get bindings
        parse_tree, bindings = self.parser.parse_with_bindings(command_ids)

        # Execute with custom segment processor
        outputs = self.executor.execute(parse_tree, command_ids, bindings, "full")

        if outputs:
            return mx.stack(outputs)
        else:
            return mx.zeros((1, len(ACTIONS)))

    def process_segment_versioned(
        self,
        command_ids: mx.array,
        segment: Tuple[int, int],
        bindings: Dict[str, str],
        stage: str,
    ) -> List[mx.array]:
        """Process segment with explicit binding lookup."""
        start, end = segment
        segment_tokens = command_ids[start:end]

        outputs = []

        for i, token_id in enumerate(segment_tokens):
            token_str = None
            for word, idx in VOCAB.items():
                if idx == int(token_id):
                    token_str = word
                    break

            # Skip operators
            if token_str in ["and", "then", "or", "while", "do", "true", "means"]:
                continue

            # For variables, use the binding
            if token_str in ["X", "Y", "Z", "W"]:
                if token_str in bindings:
                    # Get variable embedding
                    var_idx = ["X", "Y", "Z", "W"].index(token_str)
                    var_emb = self.var_embed(mx.array([var_idx]))[0]

                    # Get the bound action
                    bound_action = bindings[token_str]
                    action_idx = ["jump", "walk", "turn", "run"].index(bound_action)

                    # Create context from the binding
                    action_emb = self.action_embed(mx.array([action_idx + 1]))[
                        0
                    ]  # +1 for PAD

                    # Combine and predict
                    combined = mx.concatenate([var_emb, action_emb])
                    logits = self.predictor(combined)
                    outputs.append(logits)

            # For direct actions
            elif token_str in ["jump", "walk", "turn", "run"]:
                # Create a simple prediction
                action_idx = ["jump", "walk", "turn", "run"].index(token_str)

                # Use action embedding directly
                action_emb = self.action_embed(mx.array([action_idx + 1]))[0]
                dummy_var = mx.zeros_like(action_emb)

                combined = mx.concatenate([dummy_var, action_emb])
                logits = self.predictor(combined)
                outputs.append(logits)

        return outputs


def generate_focused_data(num_samples: int = 1000) -> List[Tuple[str, List[str]]]:
    """Generate training data focused on binding patterns."""
    data = []
    actions = ["jump", "walk", "turn", "run"]

    for _ in range(num_samples):
        # Random actions for X and Y
        action_x = np.random.choice(actions)
        action_y = np.random.choice(actions)

        # Pattern type
        pattern = np.random.choice(
            ["simple_and", "reversed_and", "simple_then", "complex_then", "with_twice"]
        )

        if pattern == "simple_and":
            command = f"X means {action_x} Y means {action_y} do X and Y"
            expected = [action_x.upper(), action_y.upper()]

        elif pattern == "reversed_and":
            command = f"X means {action_x} Y means {action_y} do Y and X"
            expected = [action_y.upper(), action_x.upper()]

        elif pattern == "simple_then":
            command = f"X means {action_x} Y means {action_y} do X then Y"
            expected = [action_x.upper(), action_y.upper()]

        elif pattern == "complex_then":
            command = f"X means {action_x} Y means {action_y} do X and Y then X"
            expected = [action_x.upper(), action_y.upper(), action_x.upper()]

        else:  # with_twice
            command = f"X means {action_x} do X twice and Y means {action_y} do Y"
            expected = [action_x.upper(), action_x.upper(), action_y.upper()]

        data.append((command, expected))

    return data


def train_simple_model(model, train_data, num_epochs=30, learning_rate=0.001):
    """Train with simple objective."""
    optimizer = optim.Adam(learning_rate=learning_rate)

    def loss_fn(model, tokens, expected):
        inputs = {"command": tokens}
        outputs = model(inputs)

        loss = 0.0
        for i, exp in enumerate(expected):
            if i < outputs.shape[0]:
                target = mx.array([ACTIONS[exp]])
                loss += mx.mean(nn.losses.cross_entropy(outputs[i : i + 1], target))

        return loss / max(len(expected), 1)

    loss_and_grad = mx.value_and_grad(loss_fn)

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        indices = np.random.permutation(len(train_data))

        for idx in indices:
            command, expected = train_data[idx]
            tokens = mx.array([VOCAB.get(w, 0) for w in command.split()])

            loss, grads = loss_and_grad(model, tokens, expected)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            total_loss += float(loss)

            # Accuracy
            outputs = model({"command": tokens})
            for i, exp in enumerate(expected):
                if i < outputs.shape[0]:
                    pred = mx.argmax(outputs[i])
                    if int(pred) == ACTIONS[exp]:
                        correct += 1
                total += 1

        acc = correct / total if total > 0 else 0
        avg_loss = total_loss / len(train_data)

        if epoch % 5 == 0:
            logger.info(
                f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}, Acc: {acc:.2%}"
            )

    return model


def test_simple_model(model):
    """Test on standard cases."""
    test_cases = [
        ("X means jump Y means walk do X and Y", ["JUMP", "WALK"]),
        ("X means run Y means turn do Y and X", ["TURN", "RUN"]),
        ("X means jump Y means walk do X then Y", ["JUMP", "WALK"]),
        ("X means walk Y means jump do X and Y then X", ["WALK", "JUMP", "WALK"]),
        ("X means jump do X then Y means walk do Y and X", ["JUMP", "WALK", "JUMP"]),
        ("X means jump do X twice and Y means walk do Y", ["JUMP", "JUMP", "WALK"]),
    ]

    correct = 0

    for command, expected in test_cases:
        tokens = mx.array([VOCAB.get(w, 0) for w in command.split()])
        outputs = model({"command": tokens})

        predicted = []
        for i in range(outputs.shape[0]):
            pred = mx.argmax(outputs[i])
            for name, idx in ACTIONS.items():
                if idx == int(pred):
                    predicted.append(name)
                    break

        is_correct = predicted == expected
        logger.info(f"\nCmd: {command}")
        logger.info(f"Exp: {expected}")
        logger.info(f"Got: {predicted}")
        logger.info(f"OK: {is_correct}")

        if is_correct:
            correct += 1

    acc = correct / len(test_cases)
    logger.info(f"\nAccuracy: {acc:.2%}")
    return acc


def main():
    """Main function."""
    model = SimpleBindingModel(len(VOCAB), len(ACTIONS))

    # Generate focused training data
    train_data = generate_focused_data(2000)
    logger.info(f"Generated {len(train_data)} samples")

    # Train
    model = train_simple_model(model, train_data, num_epochs=30)

    # Test
    accuracy = test_simple_model(model)

    if accuracy > 0.8:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        params = {
            k: v for k, v in model.parameters().items() if isinstance(v, mx.array)
        }
        mx.savez(f"simple_binding_model_{timestamp}.npz", **params)
        logger.info("Model saved!")

    return model


if __name__ == "__main__":
    main()
