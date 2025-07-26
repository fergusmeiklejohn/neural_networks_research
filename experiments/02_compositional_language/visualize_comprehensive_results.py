#!/usr/bin/env python3
"""
Visualize the comprehensive experiment results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
results_path = Path('comprehensive_results_20250723_060233/final_results.json')
with open(results_path, 'r') as f:
    results = json.load(f)

# Create figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Compositional Language Experiment Results', fontsize=16)

# Colors for each stage
stage_colors = ['blue', 'orange', 'green', 'red']

# Plot each experiment
for idx, (exp_name, exp_data) in enumerate(results['experiments'].items()):
    # Select subplot
    ax = [ax1, ax2, ax3, ax4][idx]
    
    # Extract data
    all_losses = []
    all_accuracies = []
    stage_boundaries = [0]
    
    for stage in exp_data['stages']:
        losses = [h['loss'] for h in stage['history']]
        accuracies = [h['accuracy'] for h in stage['history']]
        all_losses.extend(losses)
        all_accuracies.extend(accuracies)
        stage_boundaries.append(len(all_losses))
    
    # Plot
    epochs = range(1, len(all_losses) + 1)
    
    # Plot loss (primary y-axis)
    color = 'tab:red'
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss', color=color)
    ax.plot(epochs, all_losses, color=color, marker='o', label='Loss')
    ax.tick_params(axis='y', labelcolor=color)
    
    # Plot accuracy (secondary y-axis)
    ax2_twin = ax.twinx()
    color = 'tab:blue'
    ax2_twin.set_ylabel('Accuracy', color=color)
    ax2_twin.plot(epochs, all_accuracies, color=color, marker='s', label='Accuracy')
    ax2_twin.tick_params(axis='y', labelcolor=color)
    ax2_twin.set_ylim(0, 1)
    
    # Add stage boundaries
    for i, boundary in enumerate(stage_boundaries[1:-1]):
        ax.axvline(x=boundary + 0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add validation accuracy line
    val_acc = exp_data['stages'][0]['val_accuracy']
    ax2_twin.axhline(y=val_acc, color='green', linestyle=':', alpha=0.7, 
                     label=f'Val Acc: {val_acc:.3f}')
    
    # Title and formatting
    model_type = exp_data['model_version']
    training_type = 'Mixed' if exp_data['mixed_training'] else 'Standard'
    ax.set_title(f'{exp_name} ({model_type} + {training_type} Training)')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax2_twin.legend(loc='lower left')

plt.tight_layout()
plt.savefig('comprehensive_results_visualization.png', dpi=150, bbox_inches='tight')
plt.close()

# Create comparison bar chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Extract comparison data
experiments = list(results['comparisons'].keys())
stage1_accs = [results['comparisons'][exp]['stage1_accuracy'] for exp in experiments]
final_accs = [results['comparisons'][exp]['final_accuracy'] for exp in experiments]
degradations = [results['comparisons'][exp]['degradation_percent'] for exp in experiments]

# Bar positions
x = np.arange(len(experiments))
width = 0.35

# Accuracy comparison
ax1.bar(x - width/2, stage1_accs, width, label='Stage 1', color='blue', alpha=0.7)
ax1.bar(x + width/2, final_accs, width, label='Stage 4', color='red', alpha=0.7)
ax1.set_xlabel('Experiment')
ax1.set_ylabel('Validation Accuracy')
ax1.set_title('Validation Accuracy: Stage 1 vs Stage 4')
ax1.set_xticks(x)
ax1.set_xticklabels(experiments, rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)

# Add value labels
for i, (s1, s4) in enumerate(zip(stage1_accs, final_accs)):
    ax1.text(i - width/2, s1 + 0.01, f'{s1:.3f}', ha='center', va='bottom', fontsize=8)
    ax1.text(i + width/2, s4 + 0.01, f'{s4:.3f}', ha='center', va='bottom', fontsize=8)

# Training dynamics
ax2.set_title('Training Loss Progression')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')

for exp_name, exp_data in results['experiments'].items():
    all_losses = []
    for stage in exp_data['stages']:
        losses = [h['loss'] for h in stage['history']]
        all_losses.extend(losses)
    
    epochs = range(1, len(all_losses) + 1)
    label = f"{exp_data['model_version']} {'Mixed' if exp_data['mixed_training'] else 'Std'}"
    ax2.plot(epochs, all_losses, marker='o', label=label, alpha=0.7)

ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comprehensive_results_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("Visualizations saved:")
print("- comprehensive_results_visualization.png")
print("- comprehensive_results_comparison.png")

# Print summary statistics
print("\n=== SUMMARY STATISTICS ===")
print("\nValidation Accuracies:")
for exp_name, comp in results['comparisons'].items():
    print(f"{exp_name:15} - Stage 1: {comp['stage1_accuracy']:.3f}, Final: {comp['final_accuracy']:.3f}")

print("\nTraining Dynamics:")
for exp_name, exp_data in results['experiments'].items():
    stage1_loss = exp_data['stages'][0]['history'][-1]['loss']
    final_loss = exp_data['stages'][-1]['history'][-1]['loss']
    loss_increase = (final_loss - stage1_loss) / stage1_loss * 100
    
    stage1_acc = exp_data['stages'][0]['history'][-1]['accuracy']
    final_acc = exp_data['stages'][-1]['history'][-1]['accuracy']
    acc_decrease = (stage1_acc - final_acc) / stage1_acc * 100
    
    print(f"\n{exp_name}:")
    print(f"  Loss: {stage1_loss:.3f} → {final_loss:.3f} ({loss_increase:+.1f}%)")
    print(f"  Acc:  {stage1_acc:.3f} → {final_acc:.3f} ({acc_decrease:+.1f}% degradation)")