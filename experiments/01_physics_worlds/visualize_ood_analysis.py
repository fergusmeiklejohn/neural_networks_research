"""
Create visualization summarizing the OOD analysis findings.
"""

import os
os.environ['KERAS_BACKEND'] = 'jax'

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def create_ood_summary_plot():
    """Create a summary visualization of the OOD analysis."""
    
    # Data from the analysis
    models = ['ERM+Aug', 'GFlowNet', 'GraphExtrap', 'MAML']
    
    # All models show same pattern (using raw input space)
    interpolation_pct = [91.7, 91.7, 91.7, 91.7]
    near_extrap_pct = [8.3, 8.3, 8.3, 8.3]
    far_extrap_pct = [0.0, 0.0, 0.0, 0.0]
    
    # Model performance on "far-OOD" (from baseline results)
    baseline_results = {
        'ERM+Aug': 1.1284,
        'GFlowNet': 0.8500,
        'GraphExtrap': 0.7663,
        'MAML': 0.8228
    }
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 8))
    
    # Subplot 1: Stacked bar chart of representation analysis
    ax1 = plt.subplot(2, 2, 1)
    
    x = np.arange(len(models))
    width = 0.6
    
    p1 = ax1.bar(x, interpolation_pct, width, label='Interpolation', color='#2ecc71')
    p2 = ax1.bar(x, near_extrap_pct, width, bottom=interpolation_pct, 
                 label='Near-extrapolation', color='#f39c12')
    p3 = ax1.bar(x, far_extrap_pct, width, 
                 bottom=np.array(interpolation_pct) + np.array(near_extrap_pct),
                 label='Far-extrapolation', color='#e74c3c')
    
    ax1.set_ylabel('Percentage of Test Samples')
    ax1.set_title('Jupiter Gravity Samples: Actually Interpolation!', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45)
    ax1.legend()
    ax1.set_ylim(0, 100)
    
    # Add percentage labels
    for i, model in enumerate(models):
        ax1.text(i, interpolation_pct[i]/2, f'{interpolation_pct[i]:.1f}%', 
                ha='center', va='center', fontweight='bold')
    
    # Subplot 2: Performance degradation
    ax2 = plt.subplot(2, 2, 2)
    
    mse_values = [baseline_results[m] for m in models]
    colors = ['#3498db', '#9b59b6', '#1abc9c', '#34495e']
    
    bars = ax2.bar(x, mse_values, width, color=colors, alpha=0.7)
    ax2.set_ylabel('MSE on Jupiter Gravity')
    ax2.set_title('Poor Performance Despite Being "In-Distribution"', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45)
    ax2.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Good performance')
    ax2.set_ylim(0, 1.2)
    
    # Add value labels
    for i, (model, mse) in enumerate(zip(models, mse_values)):
        ax2.text(i, mse + 0.02, f'{mse:.3f}', ha='center', va='bottom')
    
    # Subplot 3: Key insight text
    ax3 = plt.subplot(2, 1, 2)
    ax3.axis('off')
    
    insight_text = """
    KEY INSIGHTS:
    
    1. 91.7% of "far-OOD" Jupiter gravity samples are actually INTERPOLATION in state space
    
    2. Models fail not because data is out-of-distribution, but because they haven't learned
       the causal relationship between gravity parameter and trajectory dynamics
    
    3. This validates our approach: We need models that can understand and modify physics rules,
       not just interpolate between observed states
    
    4. Standard deep learning approaches achieve <12% of their in-distribution performance
       on these "interpolation" samples - showing the limitation of pattern matching
    
    CONCLUSION: True extrapolation requires understanding causal structure, not just statistical patterns
    """
    
    ax3.text(0.1, 0.5, insight_text, fontsize=12, 
            verticalalignment='center', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.suptitle('OOD Analysis Reveals: Jupiter Gravity is NOT Out-of-Distribution!', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_path = Path("outputs/baseline_results/ood_analysis_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Summary plot saved to {output_path}")
    
    # Also create a simple comparison plot
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: What we thought
    ax1.pie([10, 30, 60], labels=['In-dist', 'Near-OOD', 'Far-OOD'], 
            colors=['#2ecc71', '#f39c12', '#e74c3c'],
            autopct='%1.0f%%', startangle=90)
    ax1.set_title("What We Thought:\nJupiter = Far-OOD", fontsize=14, fontweight='bold')
    
    # Right: Reality
    ax2.pie([91.7, 8.3, 0], labels=['Interpolation', 'Near-extrap', 'Far-extrap'],
            colors=['#2ecc71', '#f39c12', '#e74c3c'],
            autopct='%1.0f%%', startangle=90)
    ax2.set_title("Reality:\nJupiter = Interpolation!", fontsize=14, fontweight='bold')
    
    plt.suptitle('The OOD Illusion: Jupiter Gravity Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path2 = Path("outputs/baseline_results/ood_illusion.png")
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"OOD illusion plot saved to {output_path2}")

if __name__ == "__main__":
    create_ood_summary_plot()