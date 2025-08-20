#!/usr/bin/env python3
"""
Minimal Imagination Network - A first attempt at neural imagination.

Core idea: Start with the simplest possible mechanism that can:
1. Generate multiple diverse hypotheses
2. Test them internally
3. Select based on empirical success

This is exploratory - we don't know if it will work!
"""

from utils.imports import setup_project_paths

setup_project_paths()


import torch
import torch.nn as nn
import torch.nn.functional as F


class HypothesisGenerator(nn.Module):
    """
    Generates diverse hypotheses from hints.

    Key innovation: Uses multiple heads with different biases
    to generate diverse possibilities, not just likely ones.
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads

        # Each head has different initialization to encourage diversity
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.5 * (i / num_heads)),  # Varying dropout
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for i in range(num_heads)
            ]
        )

        # Add noise at different scales
        self.noise_scales = torch.linspace(0.0, 2.0, num_heads)

    def forward(self, hints: torch.Tensor, num_hypotheses: int = 10):
        """Generate diverse hypotheses from hints."""
        hints.shape[0]
        hypotheses = []

        for i, head in enumerate(self.heads):
            # Generate base hypothesis
            h = head(hints)

            # Add scaled noise for diversity
            if self.training:
                noise = torch.randn_like(h) * self.noise_scales[i]
                h = h + noise

            hypotheses.append(h)

        # Stack all hypotheses
        return torch.stack(hypotheses, dim=1)  # [batch, num_heads, hidden_dim]


class InternalWorldModel(nn.Module):
    """
    Tests hypotheses internally without executing them.

    This is the key challenge - how to know if something will work
    without actually trying it? We approximate this with a learned model.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()

        # Learn to predict if hypothesis will work given context
        self.success_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),  # hypothesis + context
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # Success probability
        )

        # Learn to predict what the hypothesis would produce
        self.outcome_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256), nn.ReLU(), nn.Linear(256, hidden_dim)
        )

    def forward(self, hypothesis: torch.Tensor, context: torch.Tensor):
        """Test hypothesis in context."""
        # Concatenate hypothesis with context
        combined = torch.cat([hypothesis, context], dim=-1)

        # Predict success probability
        success_prob = self.success_predictor(combined)

        # Predict outcome
        predicted_outcome = self.outcome_predictor(combined)

        return success_prob, predicted_outcome


class EmpiricalSelector(nn.Module):
    """
    Selects best hypothesis based on empirical testing, not similarity.

    Key: This should NOT be trained to match training distribution!
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()

        # Learn to rank based on empirical success
        self.ranker = nn.Sequential(
            nn.Linear(hidden_dim + 1, 64),  # hypothesis + success score
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, hypotheses: torch.Tensor, success_scores: torch.Tensor):
        """Select best hypothesis based on empirical success."""
        batch_size, num_hypotheses, hidden_dim = hypotheses.shape

        # Combine hypotheses with their success scores
        scores_expanded = success_scores.unsqueeze(-1)
        combined = torch.cat([hypotheses, scores_expanded], dim=-1)

        # Rank each hypothesis
        rankings = self.ranker(combined).squeeze(-1)

        # Select best (highest ranking)
        best_idx = torch.argmax(rankings, dim=1)

        # Extract best hypothesis for each batch
        best_hypotheses = hypotheses[torch.arange(batch_size), best_idx]

        return best_hypotheses, best_idx


class MinimalImaginationNetwork(nn.Module):
    """
    A minimal network that can imagine solutions.

    This is our first attempt at building a network that can
    think outside its training distribution.
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 128):
        super().__init__()

        # Extract hints from input (not deterministic patterns!)
        self.hint_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Tanh(),  # Bounded hints
        )

        # Generate diverse hypotheses
        self.hypothesis_generator = HypothesisGenerator(64, hidden_dim)

        # Test hypotheses internally
        self.world_model = InternalWorldModel(hidden_dim)

        # Select based on empirical success
        self.selector = EmpiricalSelector(hidden_dim)

        # Decode hypothesis to output
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Linear(256, input_dim)
        )

    def forward(self, x: torch.Tensor, return_all_hypotheses: bool = False):
        """
        Forward pass with imagination.

        Args:
            x: Input tensor
            return_all_hypotheses: If True, return all generated hypotheses
                                  for analysis
        """
        # Extract hints (not rules!)
        hints = self.hint_extractor(x)

        # Generate multiple diverse hypotheses
        hypotheses = self.hypothesis_generator(hints)

        # Test each hypothesis internally
        batch_size, num_hypotheses, hidden_dim = hypotheses.shape
        success_scores = []

        # Create context from hints
        context = hints.unsqueeze(1).expand(-1, num_hypotheses, -1)
        context = F.pad(context, (0, hidden_dim - context.shape[-1]))

        for i in range(num_hypotheses):
            success, _ = self.world_model(hypotheses[:, i], context[:, i])
            success_scores.append(success)

        success_scores = torch.stack(success_scores, dim=1).squeeze(-1)

        # Select best hypothesis based on empirical success
        best_hypothesis, best_idx = self.selector(hypotheses, success_scores)

        # Decode to output
        output = self.decoder(best_hypothesis)

        if return_all_hypotheses:
            return output, hypotheses, success_scores, best_idx
        return output

    def imagine_and_test(self, x: torch.Tensor):
        """
        Explicit imagination and testing for analysis.

        Returns all hypotheses and their scores for inspection.
        """
        return self.forward(x, return_all_hypotheses=True)


class ImaginationLoss(nn.Module):
    """
    Custom loss for training imagination.

    Key principles:
    1. Reward diversity of hypotheses
    2. Reward empirical success
    3. DON'T penalize deviation from training distribution
    """

    def __init__(self, diversity_weight: float = 0.1):
        super().__init__()
        self.diversity_weight = diversity_weight

    def forward(self, output, target, hypotheses, success_scores):
        """
        Compute imagination-aware loss.

        Args:
            output: Network output
            target: Target output
            hypotheses: All generated hypotheses
            success_scores: Success scores for each hypothesis
        """
        # Standard reconstruction loss (but weighted down)
        reconstruction_loss = F.mse_loss(output, target) * 0.5

        # Diversity bonus - reward different hypotheses
        # Compute pairwise distances between hypotheses
        batch_size, num_hyp, hidden_dim = hypotheses.shape
        hypotheses_flat = hypotheses.view(batch_size, -1)

        # Simple diversity: variance of hypotheses
        diversity = torch.var(hypotheses_flat, dim=1).mean()
        diversity_bonus = -diversity * self.diversity_weight  # Negative for bonus

        # Success alignment - reward if high-scoring hypothesis was correct
        # This is tricky: we want to learn to recognize success without
        # just memorizing training patterns

        # Total loss
        total_loss = reconstruction_loss + diversity_bonus

        return total_loss, {
            "reconstruction": reconstruction_loss.item(),
            "diversity_bonus": -diversity_bonus.item(),
            "total": total_loss.item(),
        }


def test_minimal_imagination():
    """Test the minimal imagination network."""
    print("Testing Minimal Imagination Network")
    print("=" * 60)

    # Create simple test data
    batch_size = 4
    input_dim = 784
    x = torch.randn(batch_size, input_dim)
    target = torch.randn(batch_size, input_dim)

    # Create network
    model = MinimalImaginationNetwork(input_dim=input_dim)

    # Test forward pass
    output, hypotheses, scores, best_idx = model.imagine_and_test(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Generated {hypotheses.shape[1]} hypotheses")
    print(f"Success scores: {scores}")
    print(f"Selected hypothesis: {best_idx}")

    # Test loss
    loss_fn = ImaginationLoss()
    loss, loss_dict = loss_fn(output, target, hypotheses, scores)

    print(f"\nLoss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")

    print("\n" + "=" * 60)
    print("Key insights from this architecture:")
    print("1. Multiple hypothesis generation with diversity mechanisms")
    print("2. Internal testing without execution")
    print("3. Selection based on predicted success, not similarity")
    print("4. Loss that rewards imagination, not just accuracy")
    print("\nThis is just a starting point - much experimentation needed!")


if __name__ == "__main__":
    test_minimal_imagination()
