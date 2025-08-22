"""Analyze ARC test results to understand performance and identify improvements.

This module provides comprehensive analysis of test results including:
- Performance visualization
- Failure pattern analysis
- Overfitting detection
- Primitive effectiveness analysis
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


class ARCPerformanceAnalyzer:
    """Comprehensive analyzer for ARC test results."""
    
    def __init__(self, results_file: str):
        """Load results from JSON file."""
        with open(results_file, 'r') as f:
            self.data = json.load(f)
        
        self.results = self.data['detailed_results']
        self.metadata = self.data['metadata']
        self.overall = self.data['overall_performance']
        self.by_type = self.data['performance_by_type']
        self.overfitting = self.data['overfitting_analysis']
    
    def generate_report(self, output_file: str = "performance_report.md"):
        """Generate comprehensive markdown report."""
        
        report = []
        report.append("# ARC Performance Analysis Report")
        report.append(f"\n*Generated: {datetime.now().isoformat()}*")
        report.append(f"\n*Dataset: {self.metadata['total_tasks']} tasks*")
        
        # Executive Summary
        report.append("\n## Executive Summary")
        report.append(f"\n- **Overall Success Rate**: {self.overall['solve_rate']:.1%}")
        report.append(f"- **Tasks Solved**: {self.overall['tasks_solved']}/{self.metadata['total_tasks']}")
        report.append(f"- **Average Accuracy**: {self.overall['average_accuracy']:.1%}")
        report.append(f"- **Perfect Solutions**: {self.overall['perfect_solutions']}")
        report.append(f"- **Average Time per Task**: {self.metadata['avg_time_per_task']:.2f}s")
        
        # Key Findings
        report.append("\n## Key Findings")
        
        # Check for overfitting
        perf_change = self.overfitting['performance_change']
        if abs(perf_change) < 0.05:
            report.append(f"\n✅ **No significant overfitting detected** (performance change: {perf_change*100:+.1f}%)")
        elif perf_change > 0:
            report.append(f"\n✅ **Performance improves with experience** (improvement: {perf_change*100:+.1f}%)")
        else:
            report.append(f"\n⚠️ **Potential overfitting detected** (performance drop: {perf_change*100:.1f}%)")
        
        # Best performing categories
        best_categories = sorted(self.by_type.items(), 
                               key=lambda x: x[1]['solve_rate'], 
                               reverse=True)[:3]
        
        report.append("\n### Strongest Task Categories:")
        for cat, stats in best_categories:
            report.append(f"- **{cat}**: {stats['solve_rate']:.1%} success rate "
                         f"({stats['solved']}/{stats['total']} tasks)")
        
        # Worst performing categories
        worst_categories = sorted(self.by_type.items(), 
                                key=lambda x: x[1]['solve_rate'])[:3]
        
        report.append("\n### Areas Needing Improvement:")
        for cat, stats in worst_categories:
            report.append(f"- **{cat}**: {stats['solve_rate']:.1%} success rate "
                         f"({stats['solved']}/{stats['total']} tasks)")
        
        # Detailed Performance Analysis
        report.append("\n## Detailed Performance Analysis")
        
        report.append("\n### Performance by Task Type")
        report.append("\n| Task Type | Total | Solved | Success Rate | Avg Accuracy |")
        report.append("|-----------|-------|--------|--------------|--------------|")
        
        for task_type in sorted(self.by_type.keys()):
            stats = self.by_type[task_type]
            report.append(f"| {task_type:15s} | {stats['total']:5d} | {stats['solved']:6d} | "
                         f"{stats['solve_rate']*100:11.1f}% | {stats['avg_accuracy']*100:11.1f}% |")
        
        # Overfitting Analysis
        report.append("\n### Overfitting Analysis")
        report.append(f"\n- **First Half Performance**: {self.overfitting['first_half_rate']:.1%} "
                     f"({self.overfitting['first_half_solved']} tasks)")
        report.append(f"- **Second Half Performance**: {self.overfitting['second_half_rate']:.1%} "
                     f"({self.overfitting['second_half_solved']} tasks)")
        report.append(f"- **Performance Delta**: {self.overfitting['performance_change']*100:+.1f}%")
        
        # Top Primitives
        if 'top_primitives' in self.data and self.data['top_primitives']:
            report.append("\n### Most Effective Primitives")
            report.append("\n| Rank | Primitive | Uses |")
            report.append("|------|-----------|------|")
            
            for i, (primitive, count) in enumerate(self.data['top_primitives'][:15], 1):
                # Truncate long primitive names
                prim_name = primitive if len(primitive) <= 50 else primitive[:47] + "..."
                report.append(f"| {i:4d} | {prim_name:50s} | {count:4d} |")
        
        # Failure Analysis
        report.append("\n## Failure Pattern Analysis")
        
        failed_tasks = [r for r in self.results if not r['success']]
        if failed_tasks:
            # Group failures by type
            failure_types = {}
            for task in failed_tasks:
                task_type = task['task_type']
                if task_type not in failure_types:
                    failure_types[task_type] = []
                failure_types[task_type].append(task)
            
            report.append(f"\n### Failed Tasks: {len(failed_tasks)}/{self.metadata['total_tasks']}")
            
            for task_type, tasks in sorted(failure_types.items(), 
                                          key=lambda x: -len(x[1]))[:5]:
                report.append(f"\n**{task_type}** ({len(tasks)} failures):")
                # Show first few task IDs
                task_ids = [t['task_id'] for t in tasks[:5]]
                report.append(f"  Examples: {', '.join(task_ids)}")
        
        # Recommendations
        report.append("\n## Recommendations")
        
        recommendations = self._generate_recommendations()
        for rec in recommendations:
            report.append(f"- {rec}")
        
        # Save report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Report saved to: {output_file}")
        return '\n'.join(report)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        
        recommendations = []
        
        # Check overall performance
        if self.overall['solve_rate'] < 0.4:
            recommendations.append("**Critical**: Overall solve rate below 40% - consider adding more fundamental primitives")
        
        # Check for specific weak areas
        for task_type, stats in self.by_type.items():
            if stats['total'] > 10 and stats['solve_rate'] < 0.2:
                recommendations.append(f"**Focus Area**: {task_type} tasks have very low success rate ({stats['solve_rate']:.1%}) - analyze failed examples")
        
        # Check overfitting
        if self.overfitting['performance_change'] < -0.1:
            recommendations.append("**Overfitting Concern**: Performance drops significantly on later tasks - improve generalization")
        elif self.overfitting['performance_change'] > 0.1:
            recommendations.append("**Positive Learning**: System improves with experience - compound learning is effective")
        
        # Check for missing capabilities
        if 'resize' in self.by_type and self.by_type['resize']['solve_rate'] < 0.3:
            recommendations.append("**Missing Capability**: Resize operations need improvement - add more flexible grid manipulation")
        
        if 'color_mapping' in self.by_type and self.by_type['color_mapping']['solve_rate'] < 0.3:
            recommendations.append("**Missing Capability**: Color mapping weak - add more sophisticated color transformation primitives")
        
        # Check time efficiency
        if self.metadata['avg_time_per_task'] > 8:
            recommendations.append("**Performance**: Average solve time is high - optimize search strategies")
        
        return recommendations
    
    def visualize_performance(self, output_dir: str = "results/visualizations"):
        """Create performance visualizations."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Performance by Task Type
        fig, ax = plt.subplots(figsize=(10, 6))
        
        types = list(self.by_type.keys())
        solve_rates = [self.by_type[t]['solve_rate'] * 100 for t in types]
        colors = ['green' if r > 50 else 'orange' if r > 30 else 'red' for r in solve_rates]
        
        bars = ax.bar(types, solve_rates, color=colors)
        ax.set_xlabel('Task Type')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Performance by Task Type')
        ax.set_ylim(0, 100)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path / 'performance_by_type.png', dpi=150)
        plt.close()
        
        # 2. Learning Curve (if we have sequential results)
        if len(self.results) > 20:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Calculate rolling success rate
            window_size = 50
            success_values = [1 if r['success'] else 0 for r in self.results]
            
            rolling_avg = []
            for i in range(len(success_values)):
                start = max(0, i - window_size + 1)
                window = success_values[start:i+1]
                rolling_avg.append(sum(window) / len(window) * 100)
            
            ax.plot(rolling_avg, linewidth=2, color='blue', alpha=0.7)
            ax.fill_between(range(len(rolling_avg)), rolling_avg, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(range(len(rolling_avg)), rolling_avg, 1)
            p = np.poly1d(z)
            ax.plot(range(len(rolling_avg)), p(range(len(rolling_avg))), 
                   "r--", alpha=0.5, label=f'Trend: {z[0]:.3f}')
            
            ax.set_xlabel('Task Number')
            ax.set_ylabel('Success Rate (%, rolling average)')
            ax.set_title(f'Learning Curve (window={window_size})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / 'learning_curve.png', dpi=150)
            plt.close()
        
        # 3. Accuracy Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        
        accuracies = [r['accuracy'] for r in self.results if r['accuracy'] >= 0]
        
        ax.hist(accuracies, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(x=0.9, color='green', linestyle='--', label='Success Threshold (90%)')
        ax.axvline(x=np.mean(accuracies), color='red', linestyle='--', 
                  label=f'Mean ({np.mean(accuracies):.1%})')
        
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Number of Tasks')
        ax.set_title('Accuracy Distribution')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'accuracy_distribution.png', dpi=150)
        plt.close()
        
        print(f"Visualizations saved to: {output_path}")
    
    def analyze_failure_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in failed tasks."""
        
        failed_tasks = [r for r in self.results if not r['success']]
        
        if not failed_tasks:
            return {'message': 'No failures to analyze!'}
        
        # Analyze characteristics
        patterns = {
            'total_failures': len(failed_tasks),
            'failure_rate': len(failed_tasks) / len(self.results),
            'by_type': {},
            'common_characteristics': {},
            'zero_accuracy_tasks': []
        }
        
        # Group by type
        for task in failed_tasks:
            task_type = task['task_type']
            if task_type not in patterns['by_type']:
                patterns['by_type'][task_type] = {
                    'count': 0,
                    'task_ids': [],
                    'avg_accuracy': []
                }
            
            patterns['by_type'][task_type]['count'] += 1
            patterns['by_type'][task_type]['task_ids'].append(task['task_id'])
            if task['accuracy'] >= 0:
                patterns['by_type'][task_type]['avg_accuracy'].append(task['accuracy'])
            
            # Track complete failures
            if task['accuracy'] == 0:
                patterns['zero_accuracy_tasks'].append(task['task_id'])
        
        # Calculate averages
        for task_type in patterns['by_type']:
            accs = patterns['by_type'][task_type]['avg_accuracy']
            patterns['by_type'][task_type]['avg_accuracy'] = np.mean(accs) if accs else 0
        
        return patterns
    
    def export_for_further_analysis(self, output_file: str = "failed_tasks_for_analysis.json"):
        """Export failed tasks for manual analysis."""
        
        failed_tasks = [r for r in self.results if not r['success']]
        
        # Group by accuracy ranges
        complete_failures = [t for t in failed_tasks if t['accuracy'] == 0]
        partial_failures = [t for t in failed_tasks if 0 < t['accuracy'] < 0.5]
        near_misses = [t for t in failed_tasks if 0.5 <= t['accuracy'] < 0.9]
        
        export_data = {
            'summary': {
                'total_failures': len(failed_tasks),
                'complete_failures': len(complete_failures),
                'partial_failures': len(partial_failures),
                'near_misses': len(near_misses)
            },
            'complete_failures': [t['task_id'] for t in complete_failures],
            'partial_failures': [(t['task_id'], t['accuracy']) for t in partial_failures],
            'near_misses': [(t['task_id'], t['accuracy']) for t in near_misses],
            'by_type': {}
        }
        
        # Organize by type
        for task in failed_tasks:
            task_type = task['task_type']
            if task_type not in export_data['by_type']:
                export_data['by_type'][task_type] = []
            export_data['by_type'][task_type].append({
                'id': task['task_id'],
                'accuracy': task['accuracy']
            })
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Failed tasks exported to: {output_file}")
        return export_data


def analyze_latest_results():
    """Analyze the most recent test results."""
    
    # Find the latest results file
    results_dir = Path("results")
    if not results_dir.exists():
        print("No results directory found")
        return
    
    result_files = list(results_dir.glob("full_test_results_*.json"))
    if not result_files:
        print("No result files found")
        return
    
    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    print(f"Analyzing: {latest_file}")
    
    # Run analysis
    analyzer = ARCPerformanceAnalyzer(str(latest_file))
    
    # Generate report
    report = analyzer.generate_report("results/performance_report.md")
    
    # Create visualizations
    analyzer.visualize_performance()
    
    # Analyze failure patterns
    failures = analyzer.analyze_failure_patterns()
    print(f"\nFailure Analysis:")
    print(f"  Total failures: {failures['total_failures']}")
    print(f"  Failure rate: {failures['failure_rate']:.1%}")
    
    # Export for further analysis
    analyzer.export_for_further_analysis()
    
    print("\nAnalysis complete!")
    print("  - Report: results/performance_report.md")
    print("  - Visualizations: results/visualizations/")
    print("  - Failed tasks: failed_tasks_for_analysis.json")


if __name__ == "__main__":
    analyze_latest_results()