"""
Analyze why PINN failed compared to baselines.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
with open('../../outputs/pinn_scaled_20250714_063917_results/final_results.json', 'r') as f:
    pinn_results = json.load(f)

with open('outputs/baseline_results/all_baselines_results.json', 'r') as f:
    baseline_results = json.load(f)

# Create comprehensive analysis plot
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('PINN Failure Analysis: Why Physics-Informed Models Struggled', fontsize=16)

# 1. MSE Comparison
ax = axes[0, 0]
conditions = ['Earth', 'Moon', 'Jupiter']

# PINN results
pinn_mse = [
    pinn_results['final_comparison']['pinn']['earth'],
    pinn_results['final_comparison']['pinn']['moon'],
    pinn_results['final_comparison']['pinn']['jupiter']
]

# Best baseline for each condition
best_baseline_mse = []
for cond in ['earth', 'moon', 'jupiter']:
    best = min([baseline_results['baselines'][b]['final_metrics'][f'{cond}_mse'] 
                for b in baseline_results['baselines']])
    best_baseline_mse.append(best)

x = np.arange(len(conditions))
width = 0.35

bars1 = ax.bar(x - width/2, best_baseline_mse, width, label='Best Baseline', color='#2E7D32')
bars2 = ax.bar(x + width/2, pinn_mse, width, label='PINN', color='#D32F2F')

ax.set_ylabel('MSE')
ax.set_title('MSE Comparison: PINN vs Best Baseline')
ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.legend()
ax.set_yscale('log')  # Log scale to show large differences

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}' if height < 1 else f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

# 2. Training Progression
ax = axes[0, 1]
stages = ['Stage 1\n(Earth)', 'Stage 2\n(+Mars+Moon)', 'Stage 3\n(+Jupiter)']
jupiter_mse_progression = [
    pinn_results['stages'][0]['test_results']['Jupiter']['mse'],
    pinn_results['stages'][1]['test_results']['Jupiter']['mse'],
    pinn_results['stages'][2]['test_results']['Jupiter']['mse']
]

ax.plot(stages, jupiter_mse_progression, 'o-', linewidth=2, markersize=10, label='PINN')
ax.axhline(y=0.766, color='green', linestyle='--', label='Best Baseline (GraphExtrap)')
ax.set_ylabel('Jupiter MSE')
ax.set_title('PINN Learning Progression on Jupiter Gravity')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Scale Comparison
ax = axes[1, 0]
models = ['ERM+Aug', 'GFlowNet', 'GraphExtrap', 'MAML', 'PINN']
jupiter_mses = [
    baseline_results['baselines']['ERM+Aug']['final_metrics']['jupiter_mse'],
    baseline_results['baselines']['GFlowNet']['final_metrics']['jupiter_mse'],
    baseline_results['baselines']['GraphExtrap']['final_metrics']['jupiter_mse'],
    baseline_results['baselines']['MAML']['final_metrics']['jupiter_mse'],
    pinn_results['final_comparison']['pinn']['jupiter']
]

colors = ['#1976D2', '#388E3C', '#D32F2F', '#F57C00', '#7B1FA2']
bars = ax.bar(models, jupiter_mses, color=colors)
ax.set_ylabel('Jupiter MSE')
ax.set_title('All Models: Jupiter Gravity Performance')
ax.set_yscale('log')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}' if height < 1 else f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

# 4. Key Insights
ax = axes[1, 1]
ax.axis('off')

insights = """
KEY FINDINGS:

1. CATASTROPHIC FAILURE
   • PINN: 880.88 MSE on Jupiter
   • Best Baseline: 0.766 MSE
   • PINN is 1,150x WORSE!

2. PHYSICS LOSSES DIDN'T HELP
   • Gravity prediction error: 24.8 m/s²
   • Model predicts Earth gravity for Jupiter
   • Physics constraints ineffective

3. PROGRESSIVE TRAINING HELPED SLIGHTLY
   • Stage 1: 1220.65 MSE
   • Stage 3: 880.88 MSE
   • 28% improvement, but still terrible

4. CRITICAL INSIGHT
   • Even with physics knowledge,
     model fails at extrapolation
   • Suggests architectural or
     optimization issues
   • Physics losses may be poorly scaled

5. HYPOTHESIS
   • MSE dominates physics losses
   • Model ignores physics constraints
   • Need better loss balancing or
     different architecture
"""

ax.text(0.05, 0.95, insights, transform=ax.transAxes, 
        fontsize=11, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('outputs/baseline_results/pinn_failure_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a focused comparison plot
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate ratio of PINN to best baseline
ratio = pinn_results['final_comparison']['pinn']['jupiter'] / 0.766

ax.text(0.5, 0.5, f"PINN CATASTROPHIC FAILURE\n\n"
        f"Jupiter MSE:\n"
        f"PINN: {pinn_results['final_comparison']['pinn']['jupiter']:.2f}\n"
        f"Best Baseline: 0.766\n\n"
        f"PINN is {ratio:.0f}x WORSE than baseline!\n\n"
        f"This suggests fundamental issues with:\n"
        f"• Architecture design\n"
        f"• Loss function balance\n"
        f"• Optimization strategy\n\n"
        f"Physics knowledge alone isn't enough.",
        ha='center', va='center', fontsize=14,
        bbox=dict(boxstyle='round,pad=1', facecolor='#FFCDD2', edgecolor='#D32F2F', linewidth=2))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.tight_layout()
plt.savefig('outputs/baseline_results/pinn_failure_summary.png', dpi=300, bbox_inches='tight')
plt.close()

print("Analysis complete! Saved visualizations to outputs/baseline_results/")
print(f"\nPINN performed {ratio:.0f}x WORSE than the best baseline on Jupiter gravity!")
print("This is a critical finding - physics-informed approaches need careful implementation.")