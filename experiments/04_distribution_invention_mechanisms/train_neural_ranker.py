#!/usr/bin/env python3
"""Train the neural program ranker using successful synthesis results.

This script:
1. Loads successful programs from evaluation
2. Generates synthetic variations
3. Trains the ranker with contrastive learning
"""

from utils.imports import setup_project_paths

setup_project_paths()

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from enhanced_compositional_dsl import EnhancedCompositionalDSL
from neural_program_ranker import NeuralProgramRanker, ProgramTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ProgramRankingDataset(Dataset):
    """Dataset for training program ranker."""

    def __init__(
        self,
        successful_programs: List[Dict],
        dsl: EnhancedCompositionalDSL,
        num_negatives: int = 5,
    ):
        """Initialize with successful programs.

        Args:
            successful_programs: List of dicts with task_id, program, examples
            dsl: DSL for generating negative examples
            num_negatives: Number of negative programs per positive
        """
        self.data = []
        self.tokenizer = ProgramTokenizer()
        self.dsl = dsl

        print(
            f"Generating training data from {len(successful_programs)} successful programs..."
        )

        for prog_info in successful_programs:
            task_id = prog_info["task_id"]
            program_str = prog_info["program"]
            examples = prog_info["examples"]

            if not examples:
                continue

            # Positive example (correct program)
            inp, out = examples[0]

            # Tokenize program (returns tensor, convert to list)
            program_tokens = self.tokenizer.tokenize(program_str).tolist()

            # Add positive example
            self.data.append(
                {
                    "input_grid": inp,
                    "output_grid": out,
                    "program_tokens": program_tokens,
                    "label": 1.0,  # Positive
                    "task_id": task_id,
                }
            )

            # Generate negative examples
            for _ in range(num_negatives):
                neg_program = self._generate_negative_program(program_str)
                neg_tokens = self.tokenizer.tokenize(neg_program).tolist()

                self.data.append(
                    {
                        "input_grid": inp,
                        "output_grid": out,
                        "program_tokens": neg_tokens,
                        "label": 0.0,  # Negative
                        "task_id": task_id,
                    }
                )

        print(f"Generated {len(self.data)} training examples")

    def _generate_negative_program(self, correct_program: str) -> str:
        """Generate a plausible but incorrect program."""
        # Strategy 1: Change parameters
        if "FillInterior" in correct_program:
            # Change colors
            colors = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            boundary = random.choice(colors)
            fill = random.choice(colors)
            return f"FillInterior(boundary={boundary}, fill={fill})"

        elif "Rotate" in correct_program:
            # Wrong rotation
            angles = [90, 180, 270]
            return f"Rotate({random.choice(angles)})"

        elif "FlipH" in correct_program:
            return "FlipV()"

        elif "FlipV" in correct_program:
            return "FlipH()"

        elif "SetColor" in correct_program:
            # Wrong color mapping
            from_color = random.randint(1, 9)
            to_color = random.randint(1, 9)
            return f"SetColor({from_color} -> {to_color})"

        # Strategy 2: Random primitive
        primitives = [
            "Rotate(90)",
            "Rotate(180)",
            "Rotate(270)",
            "FlipH()",
            "FlipV()",
            "SetColor(1 -> 2)",
            "SetColor(2 -> 3)",
            "TilePattern(2, 2)",
            "TilePattern(3, 3)",
            "DrawBorder(color=5)",
            "ExtractObjects()",
            "CropToContent()",
        ]

        return random.choice(primitives)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Pad grids to consistent size (30x30)
        inp = item["input_grid"]
        out = item["output_grid"]

        inp_padded = np.zeros((30, 30), dtype=np.float32)
        out_padded = np.zeros((30, 30), dtype=np.float32)

        h, w = inp.shape
        inp_padded[:h, :w] = inp / 9.0  # Normalize colors

        h, w = out.shape
        out_padded[:h, :w] = out / 9.0

        # Pad program tokens
        tokens = item["program_tokens"]
        max_len = 50
        if len(tokens) < max_len:
            tokens = tokens + [0] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]

        return {
            "input_grid": torch.tensor(inp_padded, dtype=torch.float32),
            "output_grid": torch.tensor(out_padded, dtype=torch.float32),
            "program_tokens": torch.LongTensor(tokens),
            "label": torch.tensor(item["label"], dtype=torch.float32),
        }


def load_successful_programs() -> List[Dict]:
    """Load successful programs from evaluation results."""
    programs = []

    # Load from synthesis evaluation
    results_file = Path("../../synthesis_evaluation_results.json")
    if results_file.exists():
        with open(results_file, "r") as f:
            data = json.load(f)

        # Extract successful programs
        for task_group in ["known_tasks", "random_sample"]:
            for task in data.get(task_group, []):
                if task.get("solved", False) and task.get("program"):
                    # Load task examples
                    task_id = task["task_id"]
                    examples = load_task_examples(task_id)

                    programs.append(
                        {
                            "task_id": task_id,
                            "program": task["program"],
                            "examples": examples,
                        }
                    )

    # Add some manually successful programs for diversity
    manual_programs = [
        {
            "task_id": "manual_rotate",
            "program": "Rotate(180)",
            "examples": [(np.array([[1, 2], [3, 4]]), np.array([[4, 3], [2, 1]]))],
        },
        {
            "task_id": "manual_flip",
            "program": "FlipH()",
            "examples": [(np.array([[1, 2], [3, 4]]), np.array([[2, 1], [4, 3]]))],
        },
        {
            "task_id": "manual_color",
            "program": "SetColor(1 -> 5)",
            "examples": [(np.array([[1, 0], [0, 1]]), np.array([[5, 0], [0, 5]]))],
        },
    ]

    programs.extend(manual_programs)

    print(f"Loaded {len(programs)} successful programs")
    return programs


def load_task_examples(task_id: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Load examples for a specific task."""
    # Try training directory first
    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")
    task_file = data_dir / f"{task_id}.json"

    if not task_file.exists():
        # Try evaluation directory
        data_dir = Path("data/arc_agi_official/ARC-AGI/data/evaluation")
        task_file = data_dir / f"{task_id}.json"

    if not task_file.exists():
        return []

    with open(task_file, "r") as f:
        task = json.load(f)

    examples = [(np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]]

    return examples


def train_ranker(
    model: NeuralProgramRanker,
    dataset: ProgramRankingDataset,
    num_epochs: int = 10,
    lr: float = 1e-3,
):
    """Train the neural program ranker."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    print(f"\nTraining on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress_bar:
            # Move to device
            input_grids = batch["input_grid"].to(device)
            output_grids = batch["output_grid"].to(device)
            program_tokens = batch["program_tokens"].to(device)
            labels = batch["label"].to(device)

            # Forward pass (correct order: program_tokens, input_grids, output_grids)
            scores = model(program_tokens, input_grids, output_grids)
            loss = criterion(scores.squeeze(), labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            total_loss += loss.item()
            predictions = (torch.sigmoid(scores.squeeze()) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            accuracy = correct / total * 100
            progress_bar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{accuracy:.1f}%"}
            )

        # Epoch summary
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total * 100
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.1f}%")

    return model


def evaluate_ranker(model: NeuralProgramRanker, dataset: ProgramRankingDataset):
    """Evaluate the trained ranker."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Separate positive and negative examples
    positives = [i for i, item in enumerate(dataset.data) if item["label"] == 1.0]
    negatives = [i for i, item in enumerate(dataset.data) if item["label"] == 0.0]

    print(
        f"\nEvaluating on {len(positives)} positive and {len(negatives)} negative examples"
    )

    with torch.no_grad():
        # Evaluate on positives
        pos_scores = []
        for idx in positives[:20]:  # Sample 20
            item = dataset[idx]
            inp = item["input_grid"].unsqueeze(0).to(device)
            out = item["output_grid"].unsqueeze(0).to(device)
            prog = item["program_tokens"].unsqueeze(0).to(device)

            score = torch.sigmoid(model(prog, inp, out)).item()
            pos_scores.append(score)

        # Evaluate on negatives
        neg_scores = []
        for idx in negatives[:20]:  # Sample 20
            item = dataset[idx]
            inp = item["input_grid"].unsqueeze(0).to(device)
            out = item["output_grid"].unsqueeze(0).to(device)
            prog = item["program_tokens"].unsqueeze(0).to(device)

            score = torch.sigmoid(model(prog, inp, out)).item()
            neg_scores.append(score)

    print(
        f"Positive scores: mean={np.mean(pos_scores):.3f}, std={np.std(pos_scores):.3f}"
    )
    print(
        f"Negative scores: mean={np.mean(neg_scores):.3f}, std={np.std(neg_scores):.3f}"
    )

    # Check discrimination
    threshold = 0.5
    pos_correct = sum(s > threshold for s in pos_scores)
    neg_correct = sum(s <= threshold for s in neg_scores)

    print(
        f"Positive accuracy: {pos_correct}/{len(pos_scores)} = {pos_correct/len(pos_scores)*100:.1f}%"
    )
    print(
        f"Negative accuracy: {neg_correct}/{len(neg_scores)} = {neg_correct/len(neg_scores)*100:.1f}%"
    )


def main():
    """Main training function."""
    print("Training Neural Program Ranker")
    print("=" * 60)

    # Load successful programs
    programs = load_successful_programs()

    if len(programs) < 2:
        print("⚠️ Not enough successful programs for training.")
        print("Generating synthetic training data...")

        # Generate synthetic successful programs
        programs = generate_synthetic_programs()

    # Create DSL
    dsl = EnhancedCompositionalDSL()

    # Create dataset
    dataset = ProgramRankingDataset(programs, dsl, num_negatives=5)

    # Create model
    model = NeuralProgramRanker(
        vocab_size=100, hidden_dim=256, num_heads=8, num_layers=4
    )

    # Train
    print("\nStarting training...")
    trained_model = train_ranker(model, dataset, num_epochs=10)

    # Evaluate
    evaluate_ranker(trained_model, dataset)

    # Save model
    output_path = Path("trained_neural_ranker.pt")
    torch.save(
        {
            "model_state_dict": trained_model.state_dict(),
            "vocab_size": 100,
            "hidden_dim": 256,
            "num_heads": 8,
            "num_layers": 4,
        },
        output_path,
    )

    print(f"\n✅ Model saved to {output_path}")


def generate_synthetic_programs() -> List[Dict]:
    """Generate synthetic training programs."""
    programs = []

    # Rotation tasks
    for angle in [90, 180, 270]:
        grid = np.random.randint(0, 5, (5, 5))
        rotated = np.rot90(grid, angle // 90)
        programs.append(
            {
                "task_id": f"synthetic_rotate_{angle}",
                "program": f"Rotate({angle})",
                "examples": [(grid, rotated)],
            }
        )

    # Flip tasks
    for i in range(5):
        grid = np.random.randint(0, 5, (5, 5))
        programs.append(
            {
                "task_id": f"synthetic_fliph_{i}",
                "program": "FlipH()",
                "examples": [(grid, np.fliplr(grid))],
            }
        )
        programs.append(
            {
                "task_id": f"synthetic_flipv_{i}",
                "program": "FlipV()",
                "examples": [(grid, np.flipud(grid))],
            }
        )

    # Color change tasks
    for i in range(10):
        grid = np.random.randint(0, 5, (5, 5))
        from_color = np.random.randint(1, 5)
        to_color = np.random.randint(1, 5)

        output = grid.copy()
        output[grid == from_color] = to_color

        programs.append(
            {
                "task_id": f"synthetic_color_{i}",
                "program": f"SetColor({from_color} -> {to_color})",
                "examples": [(grid, output)],
            }
        )

    # Fill interior tasks
    for i in range(5):
        # Create a boundary
        grid = np.zeros((7, 7), dtype=int)
        grid[1:6, 1] = 3  # Left
        grid[1:6, 5] = 3  # Right
        grid[1, 1:6] = 3  # Top
        grid[5, 1:6] = 3  # Bottom

        output = grid.copy()
        output[2:5, 2:5] = 4  # Fill interior

        programs.append(
            {
                "task_id": f"synthetic_fill_{i}",
                "program": "FillInterior(boundary=3, fill=4)",
                "examples": [(grid, output)],
            }
        )

    print(f"Generated {len(programs)} synthetic programs")
    return programs


if __name__ == "__main__":
    main()
