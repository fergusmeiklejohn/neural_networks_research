#!/usr/bin/env python3
"""
Create visualizations for the Evaluation Illusion blog post.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Set style for clean, professional plots
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 12

# Create output directory
output_dir = Path("blog_visuals")
output_dir.mkdir(exist_ok=True)


def create_evaluation_gap_chart():
    """Create the main chart showing the evaluation illusion."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Left plot: The Illusion
    categories = ["Standard\nValidation"]
    values = [84.3]
    colors = ["#2ecc71"]  # Green for "success"

    bars1 = ax1.bar(categories, values, color=colors, width=0.6)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Accuracy (%)", fontsize=14)
    ax1.set_title("What We Think We Have", fontsize=16, fontweight="bold")

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    # Right plot: The Reality
    categories2 = [
        "Base\nExamples",
        "Walk→Skip\nModification",
        "Jump→Hop\nModification",
        "Look→Scan\nModification",
        "Novel\nCompositions",
    ]
    values2 = [0, 0, 0, 0, 0]
    colors2 = ["#e74c3c"] * 5  # Red for failure

    bars2 = ax2.bar(categories2, values2, color=colors2, width=0.6)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Accuracy (%)", fontsize=14)
    ax2.set_title("What We Actually Have", fontsize=16, fontweight="bold")
    ax2.set_xticklabels(categories2, rotation=15, ha="right")

    # Add value labels
    for bar in bars2:
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            5,
            "0%",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    # Main title
    fig.suptitle(
        "The Evaluation Illusion in Neural Networks",
        fontsize=20,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / "evaluation_illusion_gap.png",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()


def create_evaluation_methodology_diagram():
    """Create a diagram showing standard vs proper evaluation."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Remove axes
    ax1.axis("off")
    ax2.axis("off")

    # Standard Evaluation (top)
    ax1.text(
        0.5,
        0.9,
        "Standard Evaluation",
        fontsize=18,
        fontweight="bold",
        ha="center",
        transform=ax1.transAxes,
    )

    # Draw mixed dataset box
    mixed_box = mpatches.FancyBboxPatch(
        (0.1, 0.3),
        0.8,
        0.4,
        boxstyle="round,pad=0.1",
        facecolor="#f39c12",
        alpha=0.7,
        edgecolor="black",
        linewidth=2,
    )
    ax1.add_patch(mixed_box)

    ax1.text(
        0.5,
        0.5,
        'Mixed Validation Set\n\n50% Base Examples:\n"walk" → "WALK"\n\n50% Modifications:\n"quickly walk" → "FAST_WALK"',
        ha="center",
        va="center",
        transform=ax1.transAxes,
        fontsize=12,
    )

    ax1.text(
        0.5,
        0.1,
        "Result: 84.3% (Misleading!)",
        fontsize=14,
        fontweight="bold",
        ha="center",
        transform=ax1.transAxes,
        color="#e74c3c",
    )

    # Proper Evaluation (bottom)
    ax2.text(
        0.5,
        0.9,
        "Proper Evaluation",
        fontsize=18,
        fontweight="bold",
        ha="center",
        transform=ax2.transAxes,
    )

    # Draw separate test set boxes
    box_width = 0.15
    box_height = 0.3
    x_positions = [0.1, 0.3, 0.5, 0.7]

    test_sets = [
        ("Base\nOnly", "#3498db"),
        ("Walk→Skip\nOnly", "#9b59b6"),
        ("Jump→Hop\nOnly", "#1abc9c"),
        ("Novel\nCompositions", "#e67e22"),
    ]

    for i, (label, color) in enumerate(test_sets):
        box = mpatches.FancyBboxPatch(
            (x_positions[i], 0.4),
            box_width,
            box_height,
            boxstyle="round,pad=0.05",
            facecolor=color,
            alpha=0.7,
            edgecolor="black",
            linewidth=2,
        )
        ax2.add_patch(box)
        ax2.text(
            x_positions[i] + box_width / 2,
            0.55,
            label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
        ax2.text(
            x_positions[i] + box_width / 2,
            0.3,
            "0%",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#e74c3c",
        )

    ax2.text(
        0.5,
        0.1,
        "Result: Complete failure exposed!",
        fontsize=14,
        fontweight="bold",
        ha="center",
        transform=ax2.transAxes,
        color="#27ae60",
    )

    plt.suptitle(
        "Standard vs Proper Evaluation Methodology", fontsize=20, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "evaluation_methodology.png",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()


def create_model_complexity_comparison():
    """Create a comparison of model complexity vs performance."""
    fig, ax = plt.subplots(figsize=(10, 8))

    models = [
        "Simple\nLSTM\n(267K params)",
        "Complex V1\n(2.3M params)",
        "Complex V2\n(4.1M params)",
    ]
    standard_val = [25.5, 84.3, 84.3]
    proper_val = [0, 0, 0]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        standard_val,
        width,
        label="Standard Validation",
        color="#3498db",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        proper_val,
        width,
        label="Proper Validation",
        color="#e74c3c",
        alpha=0.8,
    )

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.set_xlabel("Model Architecture", fontsize=14)
    ax.set_title("Model Complexity vs True Performance", fontsize=18, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 100)

    # Add insight text
    ax.text(
        0.5,
        0.95,
        "More complexity ≠ Better understanding",
        transform=ax.transAxes,
        ha="center",
        fontsize=14,
        style="italic",
        color="#7f8c8d",
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / "model_complexity_comparison.png",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()


def create_illusion_flow_diagram():
    """Create a flow diagram showing how the illusion happens."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")

    # Define box positions and connections
    boxes = [
        # Step 1
        {
            "pos": (0.2, 0.8),
            "text": "Train on mixed data:\nBase + Modifications",
            "color": "#3498db",
        },
        # Step 2
        {
            "pos": (0.2, 0.6),
            "text": "Validate on same\ndistribution",
            "color": "#3498db",
        },
        # Step 3
        {"pos": (0.2, 0.4), "text": "See high accuracy\n(84.3%)", "color": "#2ecc71"},
        # Step 4
        {"pos": (0.2, 0.2), "text": "Declare success!", "color": "#f39c12"},
        # Reality branch
        {"pos": (0.6, 0.6), "text": "Test specific\ncapabilities", "color": "#9b59b6"},
        {
            "pos": (0.6, 0.4),
            "text": "Discover 0%\non modifications",
            "color": "#e74c3c",
        },
        {
            "pos": (0.6, 0.2),
            "text": "Realize model\nmemorized patterns",
            "color": "#7f8c8d",
        },
    ]

    # Draw boxes
    for box in boxes:
        fancy_box = mpatches.FancyBboxPatch(
            (box["pos"][0] - 0.08, box["pos"][1] - 0.04),
            0.16,
            0.08,
            boxstyle="round,pad=0.01",
            facecolor=box["color"],
            alpha=0.7,
            edgecolor="black",
            linewidth=2,
        )
        ax.add_patch(fancy_box)
        ax.text(
            box["pos"][0],
            box["pos"][1],
            box["text"],
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    # Draw arrows
    arrows = [
        # Main flow
        ((0.2, 0.76), (0.2, 0.64)),
        ((0.2, 0.56), (0.2, 0.44)),
        ((0.2, 0.36), (0.2, 0.24)),
        # Branch to reality
        ((0.28, 0.6), (0.52, 0.6)),
        ((0.6, 0.56), (0.6, 0.44)),
        ((0.6, 0.36), (0.6, 0.24)),
    ]

    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="->", lw=2, color="black"),
        )

    # Add labels
    ax.text(0.2, 0.9, "The Illusion Path", fontsize=16, fontweight="bold", ha="center")
    ax.text(0.6, 0.7, "The Reality Check", fontsize=16, fontweight="bold", ha="center")

    # Add main title
    ax.text(
        0.4,
        0.95,
        "How the Evaluation Illusion Happens",
        fontsize=20,
        fontweight="bold",
        ha="center",
    )

    ax.set_xlim(0, 0.8)
    ax.set_ylim(0.1, 1.0)

    plt.tight_layout()
    plt.savefig(
        output_dir / "illusion_flow_diagram.png", bbox_inches="tight", facecolor="white"
    )
    plt.close()


def main():
    """Create all visualizations."""
    print("Creating visualization 1: Evaluation gap chart...")
    create_evaluation_gap_chart()

    print("Creating visualization 2: Evaluation methodology diagram...")
    create_evaluation_methodology_diagram()

    print("Creating visualization 3: Model complexity comparison...")
    create_model_complexity_comparison()

    print("Creating visualization 4: Illusion flow diagram...")
    create_illusion_flow_diagram()

    print(f"\nAll visualizations saved to: {output_dir.absolute()}")
    print("\nFiles created:")
    for file in sorted(output_dir.glob("*.png")):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
