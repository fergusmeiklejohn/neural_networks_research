#!/usr/bin/env python3
"""Neural Program Ranker for guiding program synthesis.

Uses a transformer architecture to predict which programs are likely to solve
ARC tasks based on input-output examples.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class ProgramEncoder(nn.Module):
    """Encode programs into vector representations."""

    def __init__(self, vocab_size: int = 100, embed_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_encoding = nn.Parameter(torch.randn(1, 100, embed_dim))

    def forward(self, program_tokens: torch.Tensor) -> torch.Tensor:
        """Encode program tokens.

        Args:
            program_tokens: [batch_size, seq_len] tensor of token IDs

        Returns:
            [batch_size, seq_len, embed_dim] tensor
        """
        seq_len = program_tokens.size(1)
        embeddings = self.embedding(program_tokens)
        embeddings += self.position_encoding[:, :seq_len, :]
        return embeddings


class GridEncoder(nn.Module):
    """Encode grids (input/output examples) into vector representations."""

    def __init__(self, max_grid_size: int = 30, hidden_dim: int = 128):
        super().__init__()
        # Convolutional layers to process grids
        self.conv1 = nn.Conv2d(11, 32, kernel_size=3, padding=1)  # 11 colors (0-10)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, hidden_dim, kernel_size=3, padding=1)

        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # Fixed size output
        self.flatten = nn.Flatten()
        self.projection = nn.Linear(hidden_dim * 16, hidden_dim)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """Encode a grid.

        Args:
            grid: [batch_size, height, width] tensor of color indices

        Returns:
            [batch_size, hidden_dim] tensor
        """
        grid.size(0)

        # Convert to one-hot encoding
        grid_one_hot = F.one_hot(grid.long(), num_classes=11)  # [B, H, W, 11]
        grid_one_hot = grid_one_hot.permute(0, 3, 1, 2).float()  # [B, 11, H, W]

        # Apply convolutions
        x = F.relu(self.conv1(grid_one_hot))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Pool and flatten
        x = self.pool(x)
        x = self.flatten(x)
        x = self.projection(x)

        return x


class NeuralProgramRanker(nn.Module):
    """Transformer-based model to rank programs for ARC tasks."""

    def __init__(
        self,
        vocab_size: int = 100,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.program_encoder = ProgramEncoder(vocab_size, hidden_dim)
        self.grid_encoder = GridEncoder(hidden_dim=hidden_dim)

        # Transformer for processing programs and examples together
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layers
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        program_tokens: torch.Tensor,
        input_grids: List[torch.Tensor],
        output_grids: List[torch.Tensor],
    ) -> torch.Tensor:
        """Rank programs based on examples.

        Args:
            program_tokens: [batch_size, seq_len] program token IDs
            input_grids: List of [height, width] input grids
            output_grids: List of [height, width] output grids

        Returns:
            [batch_size] tensor of scores
        """
        batch_size = program_tokens.size(0)

        # Encode program
        program_embeddings = self.program_encoder(
            program_tokens
        )  # [B, seq_len, hidden_dim]

        # Encode all input-output pairs
        example_embeddings = []
        for inp, out in zip(input_grids, output_grids):
            # Add batch dimension if needed
            if inp.dim() == 2:
                inp = inp.unsqueeze(0).expand(batch_size, -1, -1)
                out = out.unsqueeze(0).expand(batch_size, -1, -1)

            inp_enc = self.grid_encoder(inp)  # [B, hidden_dim]
            out_enc = self.grid_encoder(out)  # [B, hidden_dim]

            # Combine input and output
            example_emb = (inp_enc + out_enc) / 2  # Simple average
            example_embeddings.append(example_emb.unsqueeze(1))  # [B, 1, hidden_dim]

        # Concatenate all embeddings
        example_embeddings = torch.cat(
            example_embeddings, dim=1
        )  # [B, num_examples, hidden_dim]
        all_embeddings = torch.cat(
            [program_embeddings, example_embeddings], dim=1
        )  # [B, seq_len + num_examples, hidden_dim]

        # Apply transformer
        transformed = self.transformer(
            all_embeddings
        )  # [B, seq_len + num_examples, hidden_dim]

        # Pool over sequence dimension
        pooled = transformed.mean(dim=1)  # [B, hidden_dim]

        # Generate score
        scores = self.score_head(pooled).squeeze(-1)  # [B]

        return scores


class ProgramTokenizer:
    """Tokenize programs for neural processing."""

    def __init__(self):
        self.vocab = {
            "<PAD>": 0,
            "<START>": 1,
            "<END>": 2,
            "move": 3,
            "rotate": 4,
            "flip_h": 5,
            "flip_v": 6,
            "set_color": 7,
            "fill_rectangle": 8,
            "extract_objects": 9,
            "sequence": 10,
            "loop": 11,
            "tile_pattern": 12,
            "draw_border": 13,
        }
        # Add number tokens
        for i in range(-10, 11):
            self.vocab[str(i)] = len(self.vocab)

        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, program_str: str, max_len: int = 50) -> torch.Tensor:
        """Convert program string to token IDs."""
        # Simple tokenization based on program string representation
        tokens = ["<START>"]

        # Extract primitive names and parameters from string
        parts = (
            program_str.replace("(", " ").replace(")", " ").replace(",", " ").split()
        )

        for part in parts:
            if part in self.vocab:
                tokens.append(part)
            elif part.lstrip("-").isdigit():
                # Number
                tokens.append(part)
            # Ignore other parts for simplicity

        tokens.append("<END>")

        # Convert to IDs
        token_ids = []
        for token in tokens[:max_len]:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab.get("<PAD>", 0))

        # Pad to max_len
        while len(token_ids) < max_len:
            token_ids.append(self.vocab["<PAD>"])

        return torch.tensor(token_ids[:max_len])


class ProgramRankingDataset(Dataset):
    """Dataset for training the program ranker."""

    def __init__(self, data_path: Optional[Path] = None):
        self.data = []
        self.tokenizer = ProgramTokenizer()

        if data_path and data_path.exists():
            self.load_data(data_path)

    def load_data(self, data_path: Path):
        """Load training data from file."""
        with open(data_path, "r") as f:
            raw_data = json.load(f)

        for item in raw_data:
            self.data.append(
                {
                    "program": item["program"],
                    "input_grids": [np.array(g) for g in item["input_grids"]],
                    "output_grids": [np.array(g) for g in item["output_grids"]],
                    "score": item["score"],  # 1.0 if program solves task, 0.0 otherwise
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize program
        program_tokens = self.tokenizer.tokenize(item["program"])

        # Convert grids to tensors
        input_grids = [torch.tensor(g) for g in item["input_grids"]]
        output_grids = [torch.tensor(g) for g in item["output_grids"]]

        return {
            "program_tokens": program_tokens,
            "input_grids": input_grids,
            "output_grids": output_grids,
            "score": torch.tensor(item["score"], dtype=torch.float),
        }


def collate_fn(batch):
    """Custom collate function for variable-sized grids."""
    program_tokens = torch.stack([item["program_tokens"] for item in batch])
    scores = torch.stack([item["score"] for item in batch])

    # For grids, we'll pad them to the same size within the batch
    all_input_grids = []
    all_output_grids = []

    max_examples = max(len(item["input_grids"]) for item in batch)

    for i in range(max_examples):
        batch_inputs = []
        batch_outputs = []

        for item in batch:
            if i < len(item["input_grids"]):
                batch_inputs.append(item["input_grids"][i])
                batch_outputs.append(item["output_grids"][i])
            else:
                # Use the first example as padding
                batch_inputs.append(item["input_grids"][0])
                batch_outputs.append(item["output_grids"][0])

        # Pad grids to same size
        max_h = max(g.size(0) for g in batch_inputs)
        max_w = max(g.size(1) for g in batch_inputs)

        padded_inputs = []
        padded_outputs = []

        for inp, out in zip(batch_inputs, batch_outputs):
            # Pad with zeros
            pad_h = max_h - inp.size(0)
            pad_w = max_w - inp.size(1)

            if pad_h > 0 or pad_w > 0:
                inp = F.pad(inp, (0, pad_w, 0, pad_h), value=0)
                out = F.pad(out, (0, pad_w, 0, pad_h), value=0)

            padded_inputs.append(inp)
            padded_outputs.append(out)

        all_input_grids.append(torch.stack(padded_inputs))
        all_output_grids.append(torch.stack(padded_outputs))

    return {
        "program_tokens": program_tokens,
        "input_grids": all_input_grids,
        "output_grids": all_output_grids,
        "scores": scores,
    }


class ProgramRankerTrainer:
    """Trainer for the neural program ranker."""

    def __init__(self, model: NeuralProgramRanker, learning_rate: float = 1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            self.optimizer.zero_grad()

            # Forward pass
            scores = self.model(
                batch["program_tokens"], batch["input_grids"], batch["output_grids"]
            )

            # Compute loss
            loss = self.criterion(scores, batch["scores"])

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                scores = self.model(
                    batch["program_tokens"], batch["input_grids"], batch["output_grids"]
                )
                loss = self.criterion(scores, batch["scores"])
                total_loss += loss.item()

                # Collect predictions
                preds = torch.sigmoid(scores) > 0.5
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch["scores"].cpu().numpy())

        # Calculate metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        accuracy = np.mean(all_preds == all_targets)
        precision = np.sum((all_preds == 1) & (all_targets == 1)) / max(
            np.sum(all_preds == 1), 1
        )
        recall = np.sum((all_preds == 1) & (all_targets == 1)) / max(
            np.sum(all_targets == 1), 1
        )

        return {
            "loss": total_loss / len(dataloader),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        }


def test_neural_ranker():
    """Test the neural program ranker."""
    print("Testing Neural Program Ranker...")

    # Create model
    model = NeuralProgramRanker(
        vocab_size=50, hidden_dim=128, num_heads=4, num_layers=2
    )

    # Create dummy data
    tokenizer = ProgramTokenizer()
    program_tokens = tokenizer.tokenize("rotate 90").unsqueeze(0)  # Add batch dimension

    input_grid = torch.tensor([[1, 2], [3, 4]])
    output_grid = torch.tensor([[2, 4], [1, 3]])

    # Forward pass
    score = model(program_tokens, [input_grid], [output_grid])

    print(f"Program: rotate 90")
    print(f"Score: {torch.sigmoid(score).item():.3f}")

    # Test with multiple examples
    input_grids = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]])]

    output_grids = [torch.tensor([[2, 4], [1, 3]]), torch.tensor([[6, 8], [5, 7]])]

    score = model(program_tokens, input_grids, output_grids)
    print(f"Score with 2 examples: {torch.sigmoid(score).item():.3f}")

    print("\nModel architecture:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test training loop with synthetic data
    print("\nTesting training loop...")

    # Create synthetic dataset
    dataset = ProgramRankingDataset()
    for i in range(10):
        dataset.data.append(
            {
                "program": "rotate 90" if i % 2 == 0 else "flip_h",
                "input_grids": [np.random.randint(0, 5, (3, 3)) for _ in range(2)],
                "output_grids": [np.random.randint(0, 5, (3, 3)) for _ in range(2)],
                "score": 1.0 if i % 2 == 0 else 0.0,
            }
        )

    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    # Train for one epoch
    trainer = ProgramRankerTrainer(model)
    loss = trainer.train_epoch(dataloader)
    print(f"Training loss: {loss:.4f}")

    # Evaluate
    metrics = trainer.evaluate(dataloader)
    print(f"Evaluation metrics: {metrics}")


if __name__ == "__main__":
    test_neural_ranker()
