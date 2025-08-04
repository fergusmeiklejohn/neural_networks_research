"""Test gradient flow through variable binding architecture"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from dereferencing_tasks import DereferencingTaskGenerator
from train_binding_mlx_proper import (
    ProperBindingModel,
    generate_batch_from_dataset,
    train_step,
)


def analyze_gradients(model, generator):
    """Analyze gradient flow through the model"""
    print("=== Gradient Flow Analysis ===\n")

    # Create simple test batch
    batch = generate_batch_from_dataset(generator, 4)

    # Setup optimizer and loss
    optimizer = optim.Adam(learning_rate=0.001)
    loss_fn = nn.losses.cross_entropy

    # Get initial parameters (flatten nested structure)
    initial_params = {}

    def flatten_params(params, prefix="", target_dict=None):
        if target_dict is None:
            target_dict = initial_params
        for name, param in params.items():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(param, dict):
                flatten_params(param, full_name, target_dict)
            elif hasattr(param, "shape"):  # It's an array
                target_dict[full_name] = mx.array(param)

    flatten_params(dict(model.parameters()))

    # Run training step
    loss, outputs, grad_norms = train_step(model, batch, loss_fn, optimizer)

    print(f"Loss: {loss.item():.4f}")
    print(f"\nBinding scores shape: {outputs['binding_scores'].shape}")
    print(f"Bindings shape: {outputs['bindings'].shape}")

    # Analyze gradient norms
    print("\n=== Gradient Norms ===")

    # Group by component
    components = {"embedding": [], "memory": [], "binder": [], "executor": []}

    for name, norm in grad_norms.items():
        for comp in components:
            if comp in name:
                components[comp].append((name, norm))
                break

    # Print by component
    for comp, grads in components.items():
        if grads:
            print(f"\n{comp.upper()}:")
            for name, norm in grads:
                print(f"  {name}: {norm:.6f}")

    # Check if parameters changed
    print("\n=== Parameter Changes ===")
    changed_count = 0
    total_count = 0

    current_params = {}
    flatten_params(dict(model.parameters()), target_dict=current_params)

    for name in initial_params:
        if name in current_params:
            diff = mx.mean(mx.abs(current_params[name] - initial_params[name]))
            total_count += 1
            if diff.item() > 1e-6:
                changed_count += 1
                if changed_count <= 5:  # Show first 5
                    print(f"  {name}: changed by {diff.item():.6f}")

    print(f"\nTotal parameters changed: {changed_count}/{total_count}")

    # Analyze binding attention patterns
    print("\n=== Binding Analysis ===")
    binding_scores = outputs["binding_scores"]
    bindings = outputs["bindings"]

    # Check if bindings are diverse
    bindings_flat = bindings.flatten().tolist()
    unique_bindings = list(set(bindings_flat))
    print(f"Unique slot indices used: {sorted(unique_bindings)}")

    # Check attention entropy
    avg_entropy = 0
    for i in range(binding_scores.shape[0]):
        for j in range(binding_scores.shape[1]):
            probs = binding_scores[i, j]
            entropy = -mx.sum(probs * mx.log(probs + 1e-8))
            avg_entropy += entropy.item()
    avg_entropy /= binding_scores.shape[0] * binding_scores.shape[1]
    print(f"Average attention entropy: {avg_entropy:.4f}")

    # Check temperature effect
    print(f"\nCurrent temperature: {model.binder.temperature}")

    # Visualize one example
    print("\n=== Example Binding Pattern ===")
    example_idx = 0
    example_command = batch["command"][example_idx]
    example_scores = binding_scores[example_idx]
    example_bindings = bindings[example_idx]

    # Decode command
    words = []
    for idx in example_command.tolist():
        if idx < len(generator.id_to_word):
            words.append(generator.id_to_word[idx])

    print(f"Command: {' '.join(words[:10])}")  # First 10 words
    print(f"Bindings: {example_bindings[:10].tolist()}")
    print(f"Max attention scores: {mx.max(example_scores, axis=1)[:10].tolist()}")


def main():
    # Initialize generator and model
    generator = DereferencingTaskGenerator()
    model = ProperBindingModel(
        vocab_size=len(generator.word_to_id),
        num_actions=len(generator.action_to_id),
        embed_dim=128,
        hidden_dim=256,
        num_slots=10,
        num_heads=8,
    )

    # Test with different temperatures
    temperatures = [5.0, 2.0, 1.0, 0.5]

    for temp in temperatures:
        print(f"\n{'='*60}")
        print(f"Testing with temperature = {temp}")
        print(f"{'='*60}")

        model.binder.temperature = temp
        analyze_gradients(model, generator)


if __name__ == "__main__":
    main()
