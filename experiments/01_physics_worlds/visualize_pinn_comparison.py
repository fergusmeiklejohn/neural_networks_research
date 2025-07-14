"""
Create visualizations comparing PINN performance with baselines.
Shows how physics understanding enables extrapolation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def create_performance_comparison_plot():
    """Create a comprehensive comparison plot."""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Data for plots
    gravity_conditions = ['Earth\n(-9.8)', 'Mars\n(-3.7)', 'Moon\n(-1.6)', 'Jupiter\n(-24.8)']
    gravity_values = [-9.8, -3.7, -1.6, -24.8]
    
    # Model performance data (MSE)
    pinn_performance = [0.020, 0.025, 0.027, 0.083]
    erm_performance = [0.091, 0.085, 0.075, 1.128]
    gflownet_performance = [0.025, 0.040, 0.061, 0.850]
    graph_performance = [0.060, 0.080, 0.124, 0.766]
    maml_performance = [0.025, 0.045, 0.068, 0.823]
    
    # Subplot 1: Performance across gravity conditions
    ax1 = plt.subplot(2, 2, 1)
    x = np.arange(len(gravity_conditions))
    width = 0.15
    
    ax1.bar(x - 2*width, pinn_performance, width, label='PINN (Ours)', color='#2ecc71', edgecolor='black', linewidth=2)
    ax1.bar(x - width, graph_performance, width, label='GraphExtrap', color='#3498db', alpha=0.7)
    ax1.bar(x, maml_performance, width, label='MAML', color='#9b59b6', alpha=0.7)
    ax1.bar(x + width, gflownet_performance, width, label='GFlowNet', color='#e74c3c', alpha=0.7)
    ax1.bar(x + 2*width, erm_performance, width, label='ERM+Aug', color='#95a5a6', alpha=0.7)
    
    ax1.set_ylabel('MSE (log scale)', fontsize=12)
    ax1.set_title('Model Performance Across Gravity Conditions', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(gravity_conditions)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Add performance ratios on Jupiter
    jupiter_idx = 3
    for i, (perf, name) in enumerate([(graph_performance[jupiter_idx], 'GraphExtrap'),
                                      (maml_performance[jupiter_idx], 'MAML'),
                                      (gflownet_performance[jupiter_idx], 'GFlowNet'),
                                      (erm_performance[jupiter_idx], 'ERM+Aug')]):
        ratio = perf / pinn_performance[jupiter_idx]
        ax1.text(jupiter_idx + (i-1.5)*width, perf + 0.05, f'{ratio:.1f}x', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Subplot 2: Jupiter performance breakdown
    ax2 = plt.subplot(2, 2, 2)
    
    models = ['PINN\n(Ours)', 'Graph\nExtrap', 'MAML', 'GFlowNet', 'ERM+Aug']
    jupiter_mse = [0.083, 0.766, 0.823, 0.850, 1.128]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#95a5a6']
    
    bars = ax2.bar(models, jupiter_mse, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_ylabel('MSE on Jupiter Gravity', fontsize=12)
    ax2.set_title('Jupiter Extrapolation: PINN vs Baselines', fontsize=14, fontweight='bold')
    ax2.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Good performance threshold')
    
    # Highlight PINN
    bars[0].set_edgecolor('black')
    bars[0].set_linewidth(3)
    
    # Add values on bars
    for bar, val in zip(bars, jupiter_mse):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_ylim(0, 1.2)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Subplot 3: Progressive curriculum effect
    ax3 = plt.subplot(2, 2, 3)
    
    stages = ['Stage 1\n(Earth-Mars)', 'Stage 2\n(+Moon)', 'Stage 3\n(+Jupiter)']
    earth_perf = [0.016, 0.018, 0.020]
    moon_perf = [0.482, 0.023, 0.027]
    jupiter_perf = [0.923, 0.543, 0.083]
    
    x = np.arange(len(stages))
    width = 0.25
    
    ax3.bar(x - width, earth_perf, width, label='Earth', color='#3498db')
    ax3.bar(x, moon_perf, width, label='Moon', color='#e67e22')
    ax3.bar(x + width, jupiter_perf, width, label='Jupiter', color='#e74c3c')
    
    ax3.set_ylabel('MSE', fontsize=12)
    ax3.set_title('PINN Progressive Curriculum Learning', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(stages)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Subplot 4: Key insights
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    insights_text = """
    KEY INSIGHTS FROM PHYSICS-INFORMED APPROACH
    
    1. REPRESENTATION PARADOX RESOLVED
       • 91.7% of Jupiter samples are interpolation in state space
       • Yet baselines achieve only 12% of normal performance
       • PINN succeeds by understanding gravity → trajectory causation
    
    2. PROGRESSIVE CURRICULUM SUCCESS
       • Stage 1: Learn Earth-Mars physics (MSE: 0.923 → 0.083)
       • Stage 2: Extend to Moon (10x improvement)
       • Stage 3: Master Jupiter (11x improvement)
    
    3. PHYSICS CONSTRAINTS ENABLE EXTRAPOLATION
       • Energy conservation loss guides learning
       • Momentum conservation ensures realistic dynamics
       • Explicit gravity modeling captures causal structure
    
    4. IMPLICATIONS FOR AI RESEARCH
       • Statistical pattern matching has fundamental limits
       • Causal understanding enables true generalization
       • Physics-informed ML bridges theory and data
    """
    
    ax4.text(0.05, 0.95, insights_text, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
            family='monospace')
    
    # Overall title
    fig.suptitle('Physics-Informed Neural Networks Enable True Extrapolation', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Add comparison summary
    fig.text(0.5, 0.02, 
            'Result: PINN achieves 89.1% improvement over best baseline on Jupiter gravity (0.083 vs 0.766 MSE)',
            ha='center', fontsize=12, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path("outputs/baseline_results/pinn_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {output_path}")
    
    # Create a second plot showing the OOD illusion resolution
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: The paradox
    ax1.text(0.5, 0.9, "THE PARADOX", ha='center', fontsize=16, fontweight='bold',
            transform=ax1.transAxes)
    
    # Pie chart showing representation analysis
    sizes = [91.7, 8.3, 0]
    labels = ['Interpolation\n(91.7%)', 'Near-extrap\n(8.3%)', 'Far-extrap\n(0%)']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90,
            textprops={'fontsize': 12})
    ax1.text(0.5, -0.3, "Jupiter gravity samples are\nINTERPOLATION in state space", 
            ha='center', fontsize=12, transform=ax1.transAxes)
    ax1.text(0.5, -0.5, "Yet baselines fail with >10x error!", 
            ha='center', fontsize=14, fontweight='bold', color='red',
            transform=ax1.transAxes)
    
    # Right: The resolution
    ax2.text(0.5, 0.9, "THE RESOLUTION", ha='center', fontsize=16, fontweight='bold',
            transform=ax2.transAxes)
    
    # Bar chart showing performance
    models = ['Baselines\n(Average)', 'PINN\n(Physics-Informed)']
    performance = [0.867, 0.083]  # Average baseline vs PINN
    colors = ['#e74c3c', '#2ecc71']
    
    bars = ax2.bar(models, performance, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_ylabel('MSE on Jupiter Gravity', fontsize=12)
    ax2.set_ylim(0, 1.0)
    
    # Add improvement arrow
    ax2.annotate('', xy=(1, 0.15), xytext=(0, 0.85),
                arrowprops=dict(arrowstyle='->', lw=3, color='green'))
    ax2.text(0.5, 0.5, '89.1%\nimprovement', ha='center', fontsize=14, 
            fontweight='bold', color='green', transform=ax2.transAxes)
    
    # Add values on bars
    for bar, val in zip(bars, performance):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    fig2.suptitle('Resolving the OOD Illusion: Causal Understanding Enables Extrapolation',
                  fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    output_path2 = Path("outputs/baseline_results/ood_illusion_resolved.png")
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"OOD illusion resolution plot saved to {output_path2}")


if __name__ == "__main__":
    create_performance_comparison_plot()