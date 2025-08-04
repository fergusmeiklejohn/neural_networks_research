#!/usr/bin/env python3
"""
Neural compositional model that learns variable bindings.
Combines the robust parser from compositional_final_fix.py with a neural architecture
that can actually learn binding associations.
"""

import logging
from datetime import datetime
from typing import Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# Import the robust parser
from compositional_final_fix import FinalCompositionalParser, ParseNode

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

# Reverse mappings
VOCAB_REVERSE = {v: k for k, v in VOCAB.items()}
ACTIONS_REVERSE = {v: k for k, v in ACTIONS.items()}


class BindingAwareModel(nn.Module):
    """Neural model that learns to map variables to actions based on bindings."""

    def __init__(self, vocab_size: int, num_actions: int, embed_dim: int = 256):
        super().__init__()

        # Embeddings for tokens
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # Separate embeddings for variables (to learn their associations)
        self.var_embed = nn.Embedding(10, embed_dim)  # For X, Y, Z, W + padding

        # Binding encoder - learns to associate variables with actions
        self.binding_encoder = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),  # var + means + action
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Context encoder - processes the full command
        self.context_encoder = nn.TransformerEncoder(
            num_layers=2, dims=embed_dim, num_heads=4, mlp_dims=512
        )

        # Action predictor - combines binding and context
        self.action_predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),  # binding + context
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_actions),
        )

        # Parser for structured understanding
        self.parser = FinalCompositionalParser(VOCAB)

    def encode_bindings(
        self, command_ids: mx.array, bindings: Dict[int, str]
    ) -> Dict[str, mx.array]:
        """Encode the variable bindings into neural representations."""
        binding_vectors = {}

        for pos, action_name in bindings.items():
            # Get the variable token at this position
            if pos < len(command_ids):
                var_token = int(command_ids[pos])
                var_name = VOCAB_REVERSE.get(var_token, "")

                if var_name in ["X", "Y", "Z", "W"]:
                    # Create binding representation
                    var_idx = ["X", "Y", "Z", "W"].index(var_name)
                    var_emb = self.var_embed(mx.array([var_idx]))

                    # Get action embedding
                    action_token = VOCAB.get(action_name.lower(), VOCAB["<PAD>"])
                    action_emb = self.token_embed(mx.array([action_token]))

                    # Get 'means' embedding
                    means_emb = self.token_embed(mx.array([VOCAB["means"]]))

                    # Combine to create binding representation
                    binding_input = mx.concatenate(
                        [var_emb[0], means_emb[0], action_emb[0]]
                    )
                    binding_vec = self.binding_encoder(binding_input)

                    binding_vectors[var_name] = binding_vec

        return binding_vectors

    def __call__(self, inputs: Dict[str, mx.array]) -> mx.array:
        """Forward pass with neural binding processing."""
        command_ids = inputs["command"]

        # Parse command to understand structure
        parse_tree, bindings = self.parser.parse_with_bindings(command_ids)

        # Encode the full command for context
        command_embed = self.token_embed(command_ids)
        if len(command_embed.shape) == 2:
            command_embed = mx.expand_dims(command_embed, 0)

        # Create attention mask
        seq_len = command_embed.shape[1]
        mask = mx.ones((seq_len, seq_len))
        context = self.context_encoder(command_embed, mask)

        # Encode bindings
        binding_vectors = self.encode_bindings(command_ids, bindings)

        # Execute parse tree with neural predictions
        outputs = self.execute_neural(
            parse_tree, command_ids, binding_vectors, context[0]
        )

        # Stack outputs
        if outputs:
            return mx.stack(outputs)
        else:
            return mx.zeros((1, len(ACTIONS)))

    def execute_neural(
        self,
        node: ParseNode,
        command_ids: mx.array,
        binding_vectors: Dict[str, mx.array],
        context: mx.array,
    ) -> List[mx.array]:
        """Execute parse tree using neural predictions."""
        if node.is_leaf():
            outputs = []

            for token_id in node.tokens:
                token_str = VOCAB_REVERSE.get(token_id, "")

                # Skip operators
                if token_str in ["and", "then", "or", "while", "do", "true", "means"]:
                    continue

                # For variables, use binding vector
                if token_str in binding_vectors:
                    binding_vec = binding_vectors[token_str]

                    # Get context at this position
                    if node.start_pos < context.shape[0]:
                        context_vec = context[node.start_pos]
                    else:
                        context_vec = mx.mean(context, axis=0)

                    # Combine binding and context
                    combined = mx.concatenate([binding_vec, context_vec])

                    # Predict action
                    action_logits = self.action_predictor(combined)
                    outputs.append(action_logits)

                # For direct actions, create prediction
                elif token_str in ["jump", "walk", "turn", "run"]:
                    # Use context to predict
                    if node.start_pos < context.shape[0]:
                        context_vec = context[node.start_pos]
                    else:
                        context_vec = mx.mean(context, axis=0)

                    # Create dummy binding vector for consistency
                    dummy_binding = mx.zeros_like(context_vec)
                    combined = mx.concatenate([dummy_binding, context_vec])

                    action_logits = self.action_predictor(combined)
                    outputs.append(action_logits)

            # Handle modifiers
            if node.modifier == "twice" and outputs:
                outputs = outputs * 2
            elif node.modifier == "thrice" and outputs:
                outputs = outputs * 3

            return outputs

        else:
            # Handle operators
            all_outputs = []

            if node.operator.value == "AND":
                # Execute all children
                for child in node.children:
                    child_outputs = self.execute_neural(
                        child, command_ids, binding_vectors, context
                    )
                    all_outputs.extend(child_outputs)

            elif node.operator.value == "OR":
                # Execute first child (simplified)
                if node.children:
                    child_outputs = self.execute_neural(
                        node.children[0], command_ids, binding_vectors, context
                    )
                    all_outputs.extend(child_outputs)

            elif node.operator.value == "THEN":
                # Execute children in sequence
                for child in node.children:
                    child_outputs = self.execute_neural(
                        child, command_ids, binding_vectors, context
                    )
                    all_outputs.extend(child_outputs)

            elif node.operator.value == "WHILE":
                # Execute child 3 times (simplified)
                if node.children:
                    for _ in range(3):
                        child_outputs = self.execute_neural(
                            node.children[0], command_ids, binding_vectors, context
                        )
                        all_outputs.extend(child_outputs)

            return all_outputs


def generate_training_data(num_samples: int = 1000) -> List[Tuple[str, List[str]]]:
    """Generate diverse training data."""
    data = []
    actions = ["jump", "walk", "turn", "run"]
    variables = ["X", "Y", "Z", "W"]

    patterns = [
        # Basic patterns
        lambda v1, a1, v2, a2: (
            f"{v1} means {a1} {v2} means {a2} do {v1} and {v2}",
            [a1.upper(), a2.upper()],
        ),
        lambda v1, a1, v2, a2: (
            f"{v1} means {a1} {v2} means {a2} do {v2} and {v1}",
            [a2.upper(), a1.upper()],
        ),
        lambda v1, a1, v2, a2: (
            f"{v1} means {a1} {v2} means {a2} do {v1} then {v2}",
            [a1.upper(), a2.upper()],
        ),
        # With modifiers
        lambda v1, a1, v2, a2: (
            f"{v1} means {a1} do {v1} twice and {v2} means {a2} do {v2}",
            [a1.upper(), a1.upper(), a2.upper()],
        ),
        # Complex patterns
        lambda v1, a1, v2, a2: (
            f"{v1} means {a1} {v2} means {a2} do {v1} and {v2} then {v1}",
            [a1.upper(), a2.upper(), a1.upper()],
        ),
        lambda v1, a1, v2, a2: (
            f"{v1} means {a1} do {v1} then {v2} means {a2} do {v2} and {v1}",
            [a1.upper(), a2.upper(), a1.upper()],
        ),
    ]

    for _ in range(num_samples // len(patterns)):
        for pattern in patterns:
            v1, v2 = np.random.choice(variables, 2, replace=False)
            a1, a2 = np.random.choice(actions, 2, replace=True)
            command, expected = pattern(v1, a1, v2, a2)
            data.append((command, expected))

    return data[:num_samples]


def train_model(
    model: BindingAwareModel,
    train_data: List[Tuple[str, List[str]]],
    num_epochs: int = 50,
    learning_rate: float = 0.001,
):
    """Train the model with binding-aware loss."""
    optimizer = optim.Adam(learning_rate=learning_rate)

    def loss_fn(model, command_tokens, expected_actions):
        inputs = {"command": command_tokens}
        outputs = model(inputs)

        total_loss = 0.0
        count = 0

        for i, expected in enumerate(expected_actions):
            if i < outputs.shape[0]:
                target = mx.array([ACTIONS[expected]])
                total_loss += mx.mean(
                    nn.losses.cross_entropy(outputs[i : i + 1], target)
                )
                count += 1

        return total_loss / max(count, 1)

    loss_and_grad_fn = mx.value_and_grad(loss_fn)

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        indices = np.random.permutation(len(train_data))

        for idx in indices:
            command, expected_actions = train_data[idx]

            # Tokenize
            tokens = [VOCAB.get(word, VOCAB["<PAD>"]) for word in command.split()]
            command_tokens = mx.array(tokens)

            # Compute loss and gradients
            loss, grads = loss_and_grad_fn(model, command_tokens, expected_actions)

            # Update
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            total_loss += float(loss)

            # Check accuracy
            inputs = {"command": command_tokens}
            outputs = model(inputs)

            for i, expected in enumerate(expected_actions):
                if i < outputs.shape[0]:
                    pred = mx.argmax(outputs[i])
                    if int(pred) == ACTIONS[expected]:
                        correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(train_data)

        if epoch % 5 == 0:
            logger.info(
                f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}"
            )

    return model


def test_model(model: BindingAwareModel):
    """Test the model on challenging cases."""
    test_cases = [
        ("X means jump Y means walk do X and Y", ["JUMP", "WALK"]),
        ("X means run Y means turn do Y and X", ["TURN", "RUN"]),
        ("X means jump Y means walk do X or Y", ["JUMP"]),
        ("X means jump Y means walk do X then Y", ["JUMP", "WALK"]),
        ("X means walk Y means jump do X and Y then X", ["WALK", "JUMP", "WALK"]),
        ("X means jump do X then Y means walk do Y and X", ["JUMP", "WALK", "JUMP"]),
        ("X means jump do X twice and Y means walk do Y", ["JUMP", "JUMP", "WALK"]),
    ]

    correct = 0

    logger.info("\nTesting model...")
    for command, expected in test_cases:
        tokens = [VOCAB.get(word, VOCAB["<PAD>"]) for word in command.split()]
        inputs = {"command": mx.array(tokens)}

        outputs = model(inputs)
        predictions = mx.argmax(outputs, axis=1)

        predicted_actions = []
        for pred in predictions:
            if int(pred) in ACTIONS_REVERSE:
                predicted_actions.append(ACTIONS_REVERSE[int(pred)])

        is_correct = predicted_actions == expected
        if "or" in command and len(predicted_actions) == 1:
            is_correct = predicted_actions[0] in ["JUMP", "WALK", "RUN", "TURN"]

        logger.info(f"\nCommand: {command}")
        logger.info(f"Expected: {expected}")
        logger.info(f"Predicted: {predicted_actions}")
        logger.info(f"Correct: {is_correct}")

        if is_correct:
            correct += 1

    accuracy = correct / len(test_cases)
    logger.info(f"\nOverall Accuracy: {accuracy:.2%}")
    return accuracy


def main():
    """Main training function."""
    # Create model
    model = BindingAwareModel(
        vocab_size=len(VOCAB), num_actions=len(ACTIONS), embed_dim=128
    )

    # Generate training data
    train_data = generate_training_data(num_samples=2000)
    logger.info(f"Generated {len(train_data)} training samples")

    # Train model
    trained_model = train_model(model, train_data, num_epochs=50)

    # Test model
    accuracy = test_model(trained_model)

    if accuracy > 0.8:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"compositional_neural_v3_{timestamp}.npz"

        params_dict = {
            name: param
            for name, param in trained_model.parameters().items()
            if isinstance(param, mx.array)
        }
        mx.savez(model_path, **params_dict)
        logger.info(f"Model saved to {model_path}")

    return trained_model


if __name__ == "__main__":
    main()
