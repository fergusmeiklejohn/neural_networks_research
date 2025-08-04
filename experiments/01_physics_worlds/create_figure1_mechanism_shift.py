"""Create Figure 1: Mechanism vs Statistical Shift Diagram."""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

# Title
ax.text(
    5, 9.5, "Distribution Shift Taxonomy", fontsize=20, fontweight="bold", ha="center"
)

# Define colors
level1_color = "#E8F5E9"  # Light green
level2_color = "#FFF3E0"  # Light orange
level3_color = "#FFEBEE"  # Light red
arrow_color = "#333333"
text_color = "#212121"

# Level 1: Surface Variations
level1_box = FancyBboxPatch(
    (0.5, 6.5),
    2.8,
    2.3,
    boxstyle="round,pad=0.1",
    facecolor=level1_color,
    edgecolor="#4CAF50",
    linewidth=2,
)
ax.add_patch(level1_box)

ax.text(
    1.9, 8.3, "Level 1: Surface Variations", fontsize=14, fontweight="bold", ha="center"
)
ax.text(
    1.9,
    7.9,
    "Same computation,\ndifferent appearance",
    fontsize=11,
    ha="center",
    style="italic",
)

# Examples for Level 1
ax.text(
    1.9,
    7.3,
    "• Image noise/blur\n• Style transfer\n• Sensor variations",
    fontsize=10,
    ha="center",
    va="top",
)

# Performance box for Level 1
perf1_box = Rectangle(
    (0.7, 6.6), 2.4, 0.5, facecolor="white", edgecolor="#4CAF50", linewidth=1
)
ax.add_patch(perf1_box)
ax.text(
    1.9,
    6.85,
    "TTA: ✓ Improves (-13%)",
    fontsize=10,
    ha="center",
    color="#2E7D32",
    fontweight="bold",
)

# Level 2: Statistical Shifts
level2_box = FancyBboxPatch(
    (3.6, 6.5),
    2.8,
    2.3,
    boxstyle="round,pad=0.1",
    facecolor=level2_color,
    edgecolor="#FF9800",
    linewidth=2,
)
ax.add_patch(level2_box)

ax.text(
    5.0, 8.3, "Level 2: Statistical Shifts", fontsize=14, fontweight="bold", ha="center"
)
ax.text(
    5.0,
    7.9,
    "Same process,\ndifferent statistics",
    fontsize=11,
    ha="center",
    style="italic",
)

# Examples for Level 2
ax.text(
    5.0,
    7.3,
    "• Object frequencies\n• Demographic shifts\n• Seasonal variations",
    fontsize=10,
    ha="center",
    va="top",
)

# Performance box for Level 2
perf2_box = Rectangle(
    (3.8, 6.6), 2.4, 0.5, facecolor="white", edgecolor="#FF9800", linewidth=1
)
ax.add_patch(perf2_box)
ax.text(
    5.0,
    6.85,
    "TTA: ~ Marginal (-6%)",
    fontsize=10,
    ha="center",
    color="#E65100",
    fontweight="bold",
)

# Level 3: Mechanism Changes
level3_box = FancyBboxPatch(
    (6.7, 6.5),
    2.8,
    2.3,
    boxstyle="round,pad=0.1",
    facecolor=level3_color,
    edgecolor="#F44336",
    linewidth=2,
)
ax.add_patch(level3_box)

ax.text(
    8.1, 8.3, "Level 3: Mechanism Changes", fontsize=14, fontweight="bold", ha="center"
)
ax.text(8.1, 7.9, "New computation\nrequired", fontsize=11, ha="center", style="italic")

# Examples for Level 3
ax.text(
    8.1,
    7.3,
    "• g → g(t)\n• L → L(t)\n• Rule changes",
    fontsize=10,
    ha="center",
    va="top",
)

# Performance box for Level 3
perf3_box = Rectangle(
    (6.9, 6.6), 2.4, 0.5, facecolor="white", edgecolor="#F44336", linewidth=1
)
ax.add_patch(perf3_box)
ax.text(
    8.1,
    6.85,
    "TTA: ✗ Degrades (238%)",
    fontsize=10,
    ha="center",
    color="#B71C1C",
    fontweight="bold",
)

# Arrows showing progression
arrow1 = FancyArrowPatch(
    (3.3, 7.65),
    (3.6, 7.65),
    connectionstyle="arc3",
    arrowstyle="->",
    mutation_scale=20,
    linewidth=2,
    color=arrow_color,
)
ax.add_patch(arrow1)

arrow2 = FancyArrowPatch(
    (6.4, 7.65),
    (6.7, 7.65),
    connectionstyle="arc3",
    arrowstyle="->",
    mutation_scale=20,
    linewidth=2,
    color=arrow_color,
)
ax.add_patch(arrow2)

# Physics examples section
physics_box = FancyBboxPatch(
    (1, 3.5),
    8,
    2.5,
    boxstyle="round,pad=0.1",
    facecolor="#F5F5F5",
    edgecolor="#616161",
    linewidth=2,
)
ax.add_patch(physics_box)

ax.text(5, 5.7, "Physics System Examples", fontsize=16, fontweight="bold", ha="center")

# Two-ball system
ball_box = FancyBboxPatch(
    (1.5, 4.3),
    3,
    1,
    boxstyle="round,pad=0.05",
    facecolor="white",
    edgecolor="#9E9E9E",
    linewidth=1,
)
ax.add_patch(ball_box)

ax.text(3, 5.1, "Two-Ball System", fontsize=12, fontweight="bold", ha="center")
ax.text(3, 4.7, "Training: g = 9.8", fontsize=10, ha="center")
ax.text(
    3, 4.5, "Test: g(t) = 9.8 + 2sin(0.1t)", fontsize=10, ha="center", color="#B71C1C"
)

# Pendulum system
pend_box = FancyBboxPatch(
    (5.5, 4.3),
    3,
    1,
    boxstyle="round,pad=0.05",
    facecolor="white",
    edgecolor="#9E9E9E",
    linewidth=1,
)
ax.add_patch(pend_box)

ax.text(7, 5.1, "Pendulum System", fontsize=12, fontweight="bold", ha="center")
ax.text(7, 4.7, "Training: L = L₀", fontsize=10, ha="center")
ax.text(
    7,
    4.5,
    "Test: L(t) = L₀(1 + 0.2sin(0.1t))",
    fontsize=10,
    ha="center",
    color="#B71C1C",
)

# Key insight box
insight_box = FancyBboxPatch(
    (1, 0.5),
    8,
    2.5,
    boxstyle="round,pad=0.1",
    facecolor="#FFF9C4",
    edgecolor="#F57C00",
    linewidth=2,
)
ax.add_patch(insight_box)

ax.text(5, 2.7, "Key Insight", fontsize=16, fontweight="bold", ha="center")
ax.text(
    5,
    2.2,
    "Test-Time Adaptation (TTA) methods optimize self-supervised objectives\n"
    "that align with true error for Level 1-2 shifts but become misaligned\n"
    "for Level 3 shifts, causing systematic degradation.",
    fontsize=12,
    ha="center",
    va="top",
)

# Gradient alignment visualization
ax.text(2.5, 1.5, "Gradient Alignment:", fontsize=11, fontweight="bold")
ax.text(2.5, 1.2, "Level 1-2: cos(θ) > 0", fontsize=10, color="#2E7D32")
ax.text(2.5, 0.9, "Level 3: cos(θ) < 0", fontsize=10, color="#B71C1C")

# Performance summary
ax.text(6.5, 1.5, "Performance Impact:", fontsize=11, fontweight="bold")
ax.text(6.5, 1.2, "Two-ball: 238% worse", fontsize=10, color="#B71C1C")
ax.text(6.5, 0.9, "Pendulum: 12-18x worse", fontsize=10, color="#B71C1C")

plt.tight_layout()
plt.savefig("figure1_mechanism_shift_taxonomy.pdf", dpi=300, bbox_inches="tight")
plt.savefig("figure1_mechanism_shift_taxonomy.png", dpi=300, bbox_inches="tight")
plt.close()  # Close without showing

print("Figure 1 saved as:")
print("- figure1_mechanism_shift_taxonomy.pdf")
print("- figure1_mechanism_shift_taxonomy.png")
