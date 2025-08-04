"""Analyze what the model is actually learning"""

import mlx.core as mx
import numpy as np
from dereferencing_tasks import DereferencingTaskGenerator
from train_binding_mlx_proper import ProperBindingModel, generate_batch_from_dataset


def analyze_model_behavior():
    """Analyze what patterns the model has learned"""

    # Load model
    generator = DereferencingTaskGenerator()
    model = ProperBindingModel(
        vocab_size=len(generator.word_to_id),
        num_actions=len(generator.action_to_id),
        embed_dim=128,
        hidden_dim=256,
        num_slots=10,
        num_heads=8,
    )

    # Load weights
    weights = np.load("proper_binding_model.npz", allow_pickle=True)

    # Flatten model parameters for easier updating
    def update_params(model_dict, loaded_dict, prefix=""):
        for key, value in model_dict.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                update_params(value, loaded_dict, full_key)
            elif full_key in loaded_dict:
                model_dict[key] = mx.array(loaded_dict[full_key])

    # Update model with loaded weights
    model_params = dict(model.parameters())
    update_params(model_params, weights)

    print("=== Model Behavior Analysis ===\n")

    # Test 1: What actions does it predict most?
    print("1. Action Distribution Analysis")
    action_counts = {action: 0 for action in generator.action_to_id.keys()}
    total_predictions = 0

    for _ in range(10):
        batch = generate_batch_from_dataset(generator, 32)
        outputs = model(batch["command"], training=False)
        predictions = mx.argmax(outputs["action_logits"], axis=-1)

        for pred_batch in predictions:
            for pred in pred_batch.tolist():
                if pred < len(generator.id_to_action):
                    action_counts[generator.id_to_action[pred]] += 1
                    total_predictions += 1

    print("  Most common predictions:")
    for action, count in sorted(
        action_counts.items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(f"    {action}: {count/total_predictions*100:.1f}%")

    # Test 2: Does it respond to any patterns?
    print("\n2. Pattern Response Analysis")
    test_patterns = [
        ("X means jump do X", ["JUMP"]),
        ("Y means walk do Y", ["WALK"]),
        ("A means turn do A", ["TURN"]),
        ("B means run do B", ["RUN"]),
        ("do jump", ["JUMP"]),
        ("do walk", ["WALK"]),
        ("jump", ["JUMP"]),
        ("walk", ["WALK"]),
    ]

    for command_str, expected in test_patterns:
        command = command_str.split()
        command_encoded = generator.encode_words(command)
        command_array = mx.array(command_encoded)[None, :]

        outputs = model(command_array, training=False)
        predictions = mx.argmax(outputs["action_logits"], axis=-1)[0]

        predicted_actions = []
        for pred in predictions.tolist():
            if pred < len(generator.id_to_action):
                predicted_actions.append(generator.id_to_action[pred])

        match = "✓" if predicted_actions[: len(expected)] == expected else "✗"
        print(f"  {command_str:30} -> {predicted_actions[0]:10} {match}")

    # Test 3: Analyze binding patterns
    print("\n3. Binding Pattern Analysis")

    # Simple binding task
    command = "X means jump do X".split()
    command_encoded = generator.encode_words(command)
    command_array = mx.array(command_encoded)[None, :]

    outputs = model(command_array, training=False)
    bindings = outputs["bindings"][0].tolist()
    binding_scores = outputs["binding_scores"][0]

    print(f"  Command: {' '.join(command)}")
    print(f"  Word -> Slot bindings:")
    for i, (word, slot) in enumerate(zip(command, bindings)):
        max_score = mx.max(binding_scores[i]).item()
        print(f"    {word:6} -> slot {slot} (confidence: {max_score:.3f})")

    # Check if X gets consistent binding
    print("\n4. Variable Consistency Check")
    x_slots = []
    for _ in range(5):
        outputs = model(command_array, training=False)
        bindings = outputs["bindings"][0].tolist()
        # Find slots for 'X' (appears at positions 0 and 4)
        x_slots.append((bindings[0], bindings[4]))

    print(f"  X binding consistency across 5 runs:")
    for i, (first_x, second_x) in enumerate(x_slots):
        consistent = "✓" if first_x == second_x else "✗"
        print(
            f"    Run {i+1}: X@pos0 -> slot {first_x}, X@pos4 -> slot {second_x} {consistent}"
        )

    # Test 5: Check attention entropy
    print("\n5. Attention Entropy Analysis")
    total_entropy = 0
    num_samples = 0

    for _ in range(10):
        batch = generate_batch_from_dataset(generator, 16)
        outputs = model(batch["command"], training=False)
        binding_scores = outputs["binding_scores"]

        # Calculate entropy for each attention distribution
        for i in range(binding_scores.shape[0]):
            for j in range(binding_scores.shape[1]):
                probs = binding_scores[i, j]
                entropy = -mx.sum(probs * mx.log(probs + 1e-8))
                total_entropy += entropy.item()
                num_samples += 1

    avg_entropy = total_entropy / num_samples
    max_entropy = np.log(10)  # 10 slots
    print(f"  Average attention entropy: {avg_entropy:.3f} (max: {max_entropy:.3f})")
    print(f"  Entropy ratio: {avg_entropy/max_entropy:.2%}")


if __name__ == "__main__":
    analyze_model_behavior()
