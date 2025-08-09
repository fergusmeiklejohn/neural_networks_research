#!/usr/bin/env python3
"""
Train the minimal imagination network on an ARC task.

Key experiment: Can a neural network learn to imagine solutions
that weren't in its training distribution?
"""

from utils.imports import setup_project_paths

setup_project_paths()


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from minimal_imagination_network import ImaginationLoss, MinimalImaginationNetwork
from torch.utils.data import DataLoader, Dataset


class ARCImaginationDataset(Dataset):
    """Dataset for training imagination on ARC tasks."""

    def __init__(self, examples, augment=True):
        self.examples = examples
        self.augment = augment

    def __len__(self):
        return len(self.examples) * (10 if self.augment else 1)

    def __getitem__(self, idx):
        # Get base example
        base_idx = idx % len(self.examples)
        inp, out = self.examples[base_idx]

        # Flatten for neural network
        inp_flat = torch.tensor(inp.flatten(), dtype=torch.float32)
        out_flat = torch.tensor(out.flatten(), dtype=torch.float32)

        # Normalize
        inp_flat = inp_flat / 9.0  # ARC uses colors 0-9
        out_flat = out_flat / 9.0

        if self.augment and idx >= len(self.examples):
            # Add noise to create variations
            noise_scale = 0.1 * (idx // len(self.examples)) / 10
            inp_flat = inp_flat + torch.randn_like(inp_flat) * noise_scale
            inp_flat = torch.clamp(inp_flat, 0, 1)

        return inp_flat, out_flat


def create_imagination_required_task():
    """
    Create a task where imagination is REQUIRED.

    Training: Show patterns A, B, C
    Test: Requires pattern D that's different but related
    """
    train_examples = []

    # Pattern A: Diagonal line
    inp1 = np.zeros((5, 5))
    out1 = np.eye(5) * 2
    train_examples.append((inp1, out1))

    # Pattern B: Vertical lines
    inp2 = np.zeros((5, 5))
    out2 = np.zeros((5, 5))
    out2[:, 0] = 3
    out2[:, 4] = 3
    train_examples.append((inp2, out2))

    # Pattern C: Horizontal lines
    inp3 = np.zeros((5, 5))
    out3 = np.zeros((5, 5))
    out3[0, :] = 4
    out3[4, :] = 4
    train_examples.append((inp3, out3))

    # Test: Anti-diagonal (NOT in training!)
    test_input = np.zeros((5, 5))
    test_output = np.fliplr(np.eye(5)) * 5

    return train_examples, (test_input, test_output)


def train_imagination_network(epochs=100):
    """Train the imagination network."""
    print("Training Imagination Network")
    print("=" * 60)

    # Create task
    train_examples, (test_input, test_output) = create_imagination_required_task()

    print(f"Training patterns: {len(train_examples)}")
    print("Test requires: Anti-diagonal (NOT in training patterns!)")
    print()

    # Create dataset and dataloader
    dataset = ARCImaginationDataset(train_examples, augment=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Create model
    input_dim = 25  # 5x5 grid flattened
    model = MinimalImaginationNetwork(input_dim=input_dim, hidden_dim=32)

    # Loss and optimizer
    loss_fn = ImaginationLoss(diversity_weight=0.2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        diversity_scores = []

        for batch_idx, (inp, target) in enumerate(dataloader):
            optimizer.zero_grad()

            # Forward pass with imagination
            output, hypotheses, success_scores, best_idx = model.imagine_and_test(inp)

            # Compute loss
            loss, loss_dict = loss_fn(output, target, hypotheses, success_scores)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            diversity_scores.append(loss_dict["diversity_bonus"])

        if epoch % 20 == 0:
            avg_loss = total_loss / len(dataloader)
            avg_diversity = np.mean(diversity_scores)
            print(
                f"Epoch {epoch:3d}: Loss={avg_loss:.4f}, Diversity={avg_diversity:.4f}"
            )

    print("\n" + "=" * 60)
    print("Testing on pattern that requires imagination...")

    # Test on the anti-diagonal pattern
    model.eval()
    with torch.no_grad():
        test_inp = torch.tensor(test_input.flatten(), dtype=torch.float32) / 9.0
        test_inp = test_inp.unsqueeze(0)

        output, hypotheses, scores, best_idx = model.imagine_and_test(test_inp)

        print(f"\nGenerated {hypotheses.shape[1]} hypotheses")
        print(f"Success scores: {scores.squeeze()}")
        print(f"Selected hypothesis: {best_idx.item()}")

        # Check accuracy
        output_np = output.squeeze().numpy() * 9.0
        output_grid = output_np.reshape(5, 5)

        accuracy = np.mean(np.abs(output_grid - test_output) < 0.5)
        print(f"\nAccuracy on imagination-required pattern: {accuracy:.1%}")

        if accuracy > 0.8:
            print("✅ Network successfully imagined a pattern not in training!")
        else:
            print("❌ Network struggled to imagine the novel pattern")

        # Analyze hypothesis diversity
        hypotheses_np = hypotheses.squeeze().numpy()
        pairwise_distances = []
        for i in range(hypotheses_np.shape[0]):
            for j in range(i + 1, hypotheses_np.shape[0]):
                dist = np.linalg.norm(hypotheses_np[i] - hypotheses_np[j])
                pairwise_distances.append(dist)

        avg_distance = np.mean(pairwise_distances)
        print(f"\nHypothesis diversity (avg pairwise distance): {avg_distance:.4f}")

        # Compare with traditional network
        print("\n" + "=" * 60)
        print("Comparing with traditional network (no imagination)...")

        # Simple feedforward network
        traditional = nn.Sequential(
            nn.Linear(25, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 25),
        )

        # Train traditional network
        trad_optimizer = optim.Adam(traditional.parameters(), lr=0.001)
        trad_loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            for inp, target in dataloader:
                trad_optimizer.zero_grad()
                output = traditional(inp)
                loss = trad_loss_fn(output, target)
                loss.backward()
                trad_optimizer.step()

        # Test traditional network
        traditional.eval()
        with torch.no_grad():
            trad_output = traditional(test_inp)
            trad_output_np = trad_output.squeeze().numpy() * 9.0
            trad_output_grid = trad_output_np.reshape(5, 5)

            trad_accuracy = np.mean(np.abs(trad_output_grid - test_output) < 0.5)
            print(f"Traditional network accuracy: {trad_accuracy:.1%}")

        print(
            f"\nImprovement from imagination: {(accuracy - trad_accuracy)*100:.1f} percentage points"
        )

        if accuracy > trad_accuracy:
            print("✅ Imagination network outperformed traditional network!")
        else:
            print("❌ Traditional network performed better (more work needed)")


if __name__ == "__main__":
    train_imagination_network(epochs=100)
