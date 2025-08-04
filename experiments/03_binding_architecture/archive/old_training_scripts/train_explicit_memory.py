"""Train model with explicit memory read/write operations"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


class ExplicitMemoryModel(nn.Module):
    """Model that explicitly writes to and reads from memory slots"""

    def __init__(self, vocab_size=20, embed_dim=64, num_slots=4, num_actions=5):
        super().__init__()

        self.num_slots = num_slots
        self.embed_dim = embed_dim

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Memory operations
        self.write_key = nn.Linear(embed_dim, embed_dim)
        self.write_value = nn.Linear(embed_dim, embed_dim)
        self.read_key = nn.Linear(embed_dim, embed_dim)

        # Initial memory (learnable)
        self.init_memory = mx.zeros((num_slots, embed_dim))

        # Action decoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(), nn.Linear(128, num_actions)
        )

    def __call__(self, words, training=True):
        batch_size, seq_len = words.shape
        embeds = self.embedding(words)

        # Initialize memory for each batch
        memory = mx.broadcast_to(
            self.init_memory[None, :, :], (batch_size, self.num_slots, self.embed_dim)
        )

        outputs = []
        all_write_weights = []
        all_read_weights = []

        # Process sequence step by step
        for t in range(seq_len):
            word_embed = embeds[:, t, :]  # (batch, embed_dim)

            # Decide whether to write or read based on position
            # Simple heuristic: write at positions 0-2, read at positions 3+
            if t < 3:
                # WRITE operation
                write_k = self.write_key(word_embed)  # (batch, embed_dim)
                write_v = self.write_value(word_embed)  # (batch, embed_dim)

                # Compute write weights (which slot to write to)
                write_scores = mx.sum(
                    memory * write_k[:, None, :], axis=2
                )  # (batch, num_slots)
                write_weights = mx.softmax(write_scores, axis=1)
                all_write_weights.append(write_weights)

                # Write to memory (soft write using weights)
                write_weights_exp = write_weights[:, :, None]  # (batch, num_slots, 1)
                write_v_exp = write_v[:, None, :]  # (batch, 1, embed_dim)

                # Update memory
                memory = memory + write_weights_exp * write_v_exp

                # No output during write phase
                outputs.append(mx.zeros((batch_size, 5)))

            else:
                # READ operation
                read_k = self.read_key(word_embed)  # (batch, embed_dim)

                # Compute read weights
                read_scores = mx.sum(
                    memory * read_k[:, None, :], axis=2
                )  # (batch, num_slots)
                read_weights = mx.softmax(
                    read_scores / 0.5, axis=1
                )  # Lower temperature for sharper attention
                all_read_weights.append(read_weights)

                # Read from memory
                read_weights_exp = read_weights[:, :, None]  # (batch, num_slots, 1)
                read_value = mx.sum(
                    memory * read_weights_exp, axis=1
                )  # (batch, embed_dim)

                # Decode to action
                action_logits = self.decoder(read_value)
                outputs.append(action_logits)

        # Stack outputs
        outputs = mx.stack(outputs, axis=1)  # (batch, seq_len, num_actions)

        return outputs, all_write_weights, all_read_weights


def generate_memory_task_batch(batch_size=32):
    """Generate tasks that require explicit memory usage

    Pattern: "X is ACTION Y is ACTION2 recall X recall Y"
    Where X, Y are variable names and ACTION is what they should do
    """
    # Vocabulary
    vocab = {
        "<PAD>": 0,
        "X": 1,
        "Y": 2,
        "A": 3,
        "B": 4,
        "is": 5,
        "recall": 6,
        "jump": 7,
        "walk": 8,
        "turn": 9,
        "run": 10,
    }

    actions = {"jump": 1, "walk": 2, "turn": 3, "run": 4}

    sentences = []
    labels = []

    for _ in range(batch_size):
        # Choose two different variables and actions
        vars = np.random.choice(["X", "Y", "A", "B"], size=2, replace=False)
        acts = np.random.choice(["jump", "walk", "turn", "run"], size=2, replace=False)

        # Build sentence: "X is jump Y is walk recall X recall Y"
        sentence = [
            vocab[vars[0]],
            vocab["is"],
            vocab[acts[0]],
            vocab[vars[1]],
            vocab["is"],
            vocab[acts[1]],
            vocab["recall"],
            vocab[vars[0]],
            vocab["recall"],
            vocab[vars[1]],
        ]

        # Labels (only for recall positions)
        label = [0, 0, 0, 0, 0, 0, 0, actions[acts[0]], 0, actions[acts[1]]]

        sentences.append(sentence)
        labels.append(label)

    return mx.array(np.array(sentences, dtype=np.int32)), mx.array(
        np.array(labels, dtype=np.int32)
    )


def train_explicit_memory():
    """Train the explicit memory model"""
    model = ExplicitMemoryModel()
    optimizer = optim.Adam(learning_rate=0.003)

    print("Training explicit memory model...")
    print("Pattern: 'X is jump Y is walk recall X recall Y'")
    print()

    for epoch in range(50):
        total_loss = 0
        total_correct = 0
        total_predictions = 0

        for _ in range(50):  # 50 batches per epoch
            sentences, labels = generate_memory_task_batch(32)

            def loss_fn(model):
                outputs, write_weights, read_weights = model(sentences)

                # Compute loss only for positions where we expect an action
                mask = labels > 0

                if mx.any(mask):
                    # Get predictions and labels for non-zero positions
                    outputs_masked = outputs * mask[:, :, None]
                    outputs_flat = outputs_masked.reshape(-1, 5)
                    labels_flat = labels.reshape(-1)

                    # Only compute loss where label > 0
                    valid_mask = labels_flat > 0
                    if mx.any(valid_mask):
                        # MLX doesn't have advanced indexing, so use multiplication
                        ce_losses = nn.losses.cross_entropy(outputs_flat, labels_flat)
                        ce_loss = mx.sum(ce_losses * valid_mask) / mx.sum(valid_mask)
                    else:
                        ce_loss = mx.array(0.0)
                else:
                    ce_loss = mx.array(0.0)

                # Regularization: encourage using different slots for different variables
                if write_weights:
                    # Entropy of write weights (want high entropy = using different slots)
                    write_entropy = 0
                    for w in write_weights:
                        write_entropy += mx.mean(-mx.sum(w * mx.log(w + 1e-8), axis=1))
                    write_entropy = write_entropy / len(write_weights)

                    # Total loss
                    total_loss = (
                        ce_loss - 0.01 * write_entropy
                    )  # Negative because we want high entropy
                else:
                    total_loss = ce_loss

                return total_loss, ce_loss

            # Compute gradients
            def grad_fn(model):
                loss, _ = loss_fn(model)
                return loss

            value_and_grad_fn = mx.value_and_grad(grad_fn)
            loss, grads = value_and_grad_fn(model)
            optimizer.update(model, grads)

            # Evaluate
            outputs, _, _ = model(sentences, training=False)
            preds = mx.argmax(outputs, axis=-1)

            # Only check accuracy where we expect predictions
            mask = labels > 0
            if mx.any(mask):
                correct = mx.sum((preds == labels) * mask)
                total_predictions_batch = mx.sum(mask)

                total_correct += correct.item()
                total_predictions += total_predictions_batch.item()

            total_loss += loss.item()

        if epoch % 5 == 0:
            accuracy = total_correct / total_predictions if total_predictions > 0 else 0
            print(f"Epoch {epoch}: Loss={total_loss/50:.3f}, Accuracy={accuracy:.1%}")

    # Test the model
    print("\n=== Testing ===")

    # Manual test cases
    vocab = {
        "<PAD>": 0,
        "X": 1,
        "Y": 2,
        "A": 3,
        "B": 4,
        "is": 5,
        "recall": 6,
        "jump": 7,
        "walk": 8,
        "turn": 9,
        "run": 10,
    }

    action_names = {0: "PAD", 1: "jump", 2: "walk", 3: "turn", 4: "run"}

    test_cases = [
        # X is jump, Y is walk, recall X, recall Y
        ([1, 5, 7, 2, 5, 8, 6, 1, 6, 2], "X is jump Y is walk recall X recall Y"),
        # A is turn, B is run, recall A, recall B
        ([3, 5, 9, 4, 5, 10, 6, 3, 6, 4], "A is turn B is run recall A recall B"),
    ]

    for sentence, description in test_cases:
        sentence_mx = mx.array([sentence], dtype=mx.int32)
        outputs, write_weights, read_weights = model(sentence_mx, training=False)

        preds = mx.argmax(outputs[0], axis=-1).tolist()

        print(f"\n{description}")
        print(f"Predictions: {[action_names[p] for p in preds]}")

        # Show memory operations
        if write_weights:
            print("\nWrite operations:")
            for i, w in enumerate(write_weights):
                slot = mx.argmax(w[0]).item()
                confidence = mx.max(w[0]).item()
                word_idx = [0, 2, 4][i]  # Positions where writes happen
                word = [k for k, v in vocab.items() if v == sentence[word_idx]][0]
                print(
                    f"  Position {word_idx} ({word}): Write to slot {slot} (confidence: {confidence:.2f})"
                )

        if read_weights:
            print("\nRead operations:")
            for i, r in enumerate(read_weights):
                slot = mx.argmax(r[0]).item()
                confidence = mx.max(r[0]).item()
                read_pos = 7 + i * 2  # Positions 7 and 9
                word = [k for k, v in vocab.items() if v == sentence[read_pos]][0]
                print(
                    f"  Position {read_pos} ({word}): Read from slot {slot} (confidence: {confidence:.2f})"
                )


if __name__ == "__main__":
    train_explicit_memory()
