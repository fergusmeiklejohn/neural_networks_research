"""Meta-Learning System - Learns which strategies work for which patterns.

This module implements a meta-learning layer that learns from experience to:
1. Predict the best strategy for a given task
2. Learn from failures to improve future attempts
3. Adapt strategies based on task context
4. Extract meta-patterns from successful inventions

Key Innovation: Instead of fixed strategies, we learn how to choose and adapt strategies.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import json
from pathlib import Path
import time


@dataclass
class TaskFeatures:
    """Features extracted from a task for strategy selection."""
    
    # Shape features
    input_shapes: List[Tuple[int, int]]
    output_shapes: List[Tuple[int, int]]
    shape_change: str  # 'same', 'smaller', 'larger', 'different'
    
    # Color features
    num_input_colors: int
    num_output_colors: int
    new_colors_appear: bool
    colors_disappear: bool
    
    # Pattern features
    has_repeating_pattern: bool
    has_symmetry: bool
    has_objects: bool
    has_grid_structure: bool
    
    # Transformation hints
    likely_geometric: bool  # Rotation, reflection, etc.
    likely_extraction: bool  # Extracting subregion
    likely_construction: bool  # Building new pattern
    likely_coloring: bool  # Color-based transformation
    
    # Complexity
    input_complexity: float  # Entropy-based measure
    output_complexity: float
    example_consistency: float  # How similar are the examples
    
    def to_vector(self) -> np.ndarray:
        """Convert features to numerical vector for ML models."""
        features = []
        
        # Shape features
        avg_input_h = np.mean([s[0] for s in self.input_shapes])
        avg_input_w = np.mean([s[1] for s in self.input_shapes])
        avg_output_h = np.mean([s[0] for s in self.output_shapes])
        avg_output_w = np.mean([s[1] for s in self.output_shapes])
        
        features.extend([avg_input_h, avg_input_w, avg_output_h, avg_output_w])
        
        # Shape change encoding
        shape_change_map = {'same': 0, 'smaller': 1, 'larger': 2, 'different': 3}
        features.append(shape_change_map.get(self.shape_change, 3))
        
        # Color features
        features.extend([
            self.num_input_colors,
            self.num_output_colors,
            1.0 if self.new_colors_appear else 0.0,
            1.0 if self.colors_disappear else 0.0
        ])
        
        # Pattern features
        features.extend([
            1.0 if self.has_repeating_pattern else 0.0,
            1.0 if self.has_symmetry else 0.0,
            1.0 if self.has_objects else 0.0,
            1.0 if self.has_grid_structure else 0.0
        ])
        
        # Transformation hints
        features.extend([
            1.0 if self.likely_geometric else 0.0,
            1.0 if self.likely_extraction else 0.0,
            1.0 if self.likely_construction else 0.0,
            1.0 if self.likely_coloring else 0.0
        ])
        
        # Complexity
        features.extend([
            self.input_complexity,
            self.output_complexity,
            self.example_consistency
        ])
        
        return np.array(features, dtype=np.float32)


@dataclass
class StrategyOutcome:
    """Records the outcome of applying a strategy to a task."""
    task_id: str
    task_features: TaskFeatures
    strategy_name: str
    success: bool
    accuracy: float
    time_taken: float
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    invention_created: Optional[str] = None


@dataclass 
class StrategyKnowledge:
    """Knowledge about a strategy's performance."""
    name: str
    total_attempts: int = 0
    successes: int = 0
    failures: int = 0
    avg_accuracy: float = 0.0
    avg_time: float = 0.0
    
    # Task types it works well for
    successful_features: List[np.ndarray] = field(default_factory=list)
    failed_features: List[np.ndarray] = field(default_factory=list)
    
    # Error patterns
    error_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_attempts == 0:
            return 0.0
        return self.successes / self.total_attempts
    
    def predict_success(self, task_features: TaskFeatures) -> float:
        """Predict likelihood of success for given task features."""
        if not self.successful_features and not self.failed_features:
            return 0.5  # No data, neutral prediction
        
        feature_vec = task_features.to_vector()
        
        # Simple nearest neighbor approach
        if self.successful_features:
            min_success_dist = min(
                np.linalg.norm(feature_vec - sf) 
                for sf in self.successful_features
            )
        else:
            min_success_dist = float('inf')
        
        if self.failed_features:
            min_fail_dist = min(
                np.linalg.norm(feature_vec - ff)
                for ff in self.failed_features
            )
        else:
            min_fail_dist = float('inf')
        
        # Convert distances to probability
        if min_success_dist == float('inf') and min_fail_dist == float('inf'):
            return 0.5
        
        # Use softmax-like scoring
        success_score = np.exp(-min_success_dist / 10.0)
        fail_score = np.exp(-min_fail_dist / 10.0)
        
        return success_score / (success_score + fail_score)


class MetaLearner:
    """Meta-learning system that learns from experience to improve strategy selection."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize the meta-learner.
        
        Args:
            storage_path: Path for persistent storage of learned knowledge
        """
        self.storage_path = storage_path or Path("meta_learning_knowledge.json")
        
        # Strategy knowledge base
        self.strategies: Dict[str, StrategyKnowledge] = {}
        
        # Task memory
        self.task_outcomes: List[StrategyOutcome] = []
        
        # Learning parameters
        self.exploration_rate = 0.2  # Probability of trying non-optimal strategy
        self.min_confidence = 0.3  # Minimum confidence to use a strategy
        
        # Meta-patterns discovered
        self.meta_patterns: List[Dict[str, Any]] = []
        
        # Load existing knowledge
        self.load()
    
    def extract_task_features(self, 
                             examples: List[Tuple[np.ndarray, np.ndarray]]) -> TaskFeatures:
        """Extract features from task examples."""
        
        if not examples:
            raise ValueError("No examples provided")
        
        input_shapes = []
        output_shapes = []
        all_input_colors = set()
        all_output_colors = set()
        
        for inp, out in examples:
            input_shapes.append(inp.shape)
            output_shapes.append(out.shape)
            all_input_colors.update(np.unique(inp))
            all_output_colors.update(np.unique(out))
        
        # Determine shape change
        if all(i == o for i, o in zip(input_shapes, output_shapes)):
            shape_change = 'same'
        elif all(i[0] * i[1] > o[0] * o[1] for i, o in zip(input_shapes, output_shapes)):
            shape_change = 'smaller'
        elif all(i[0] * i[1] < o[0] * o[1] for i, o in zip(input_shapes, output_shapes)):
            shape_change = 'larger'
        else:
            shape_change = 'different'
        
        # Check for patterns
        first_input = examples[0][0]
        has_repeating = self._check_repeating_pattern(first_input)
        has_symmetry = self._check_symmetry(first_input)
        has_objects = self._check_objects(first_input)
        has_grid = self._check_grid_structure(first_input)
        
        # Transformation hints
        likely_geometric = self._is_likely_geometric(examples)
        likely_extraction = shape_change == 'smaller'
        likely_construction = shape_change == 'larger'
        likely_coloring = len(all_output_colors - all_input_colors) > 0
        
        # Complexity measures
        input_complexity = self._calculate_complexity(first_input)
        output_complexity = self._calculate_complexity(examples[0][1])
        example_consistency = self._calculate_consistency(examples)
        
        return TaskFeatures(
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            shape_change=shape_change,
            num_input_colors=len(all_input_colors),
            num_output_colors=len(all_output_colors),
            new_colors_appear=len(all_output_colors - all_input_colors) > 0,
            colors_disappear=len(all_input_colors - all_output_colors) > 0,
            has_repeating_pattern=has_repeating,
            has_symmetry=has_symmetry,
            has_objects=has_objects,
            has_grid_structure=has_grid,
            likely_geometric=likely_geometric,
            likely_extraction=likely_extraction,
            likely_construction=likely_construction,
            likely_coloring=likely_coloring,
            input_complexity=input_complexity,
            output_complexity=output_complexity,
            example_consistency=example_consistency
        )
    
    def predict_best_strategy(self, 
                             task_features: TaskFeatures,
                             available_strategies: List[str]) -> List[Tuple[str, float]]:
        """Predict best strategies for a task.
        
        Returns:
            List of (strategy_name, confidence) tuples, sorted by confidence
        """
        predictions = []
        
        for strategy in available_strategies:
            if strategy not in self.strategies:
                # Unknown strategy, give neutral score
                predictions.append((strategy, 0.5))
            else:
                knowledge = self.strategies[strategy]
                confidence = knowledge.predict_success(task_features)
                
                # Boost confidence based on success rate
                confidence *= (0.5 + 0.5 * knowledge.success_rate())
                
                predictions.append((strategy, confidence))
        
        # Sort by confidence
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Apply exploration (occasionally try less confident strategies)
        if np.random.random() < self.exploration_rate and len(predictions) > 1:
            # Swap first with random other
            idx = np.random.randint(1, len(predictions))
            predictions[0], predictions[idx] = predictions[idx], predictions[0]
        
        return predictions
    
    def learn_from_outcome(self, outcome: StrategyOutcome):
        """Learn from the outcome of a strategy application."""
        
        # Store outcome
        self.task_outcomes.append(outcome)
        
        # Update strategy knowledge
        if outcome.strategy_name not in self.strategies:
            self.strategies[outcome.strategy_name] = StrategyKnowledge(outcome.strategy_name)
        
        knowledge = self.strategies[outcome.strategy_name]
        knowledge.total_attempts += 1
        
        if outcome.success:
            knowledge.successes += 1
            knowledge.successful_features.append(outcome.task_features.to_vector())
            
            # Keep only recent successful features (memory limit)
            if len(knowledge.successful_features) > 100:
                knowledge.successful_features = knowledge.successful_features[-100:]
        else:
            knowledge.failures += 1
            knowledge.failed_features.append(outcome.task_features.to_vector())
            
            # Track error types
            if outcome.error_type:
                knowledge.error_counts[outcome.error_type] += 1
            
            # Keep only recent failed features
            if len(knowledge.failed_features) > 100:
                knowledge.failed_features = knowledge.failed_features[-100:]
        
        # Update averages
        knowledge.avg_accuracy = (
            knowledge.avg_accuracy * (knowledge.total_attempts - 1) + outcome.accuracy
        ) / knowledge.total_attempts
        
        knowledge.avg_time = (
            knowledge.avg_time * (knowledge.total_attempts - 1) + outcome.time_taken
        ) / knowledge.total_attempts
        
        # Extract meta-patterns periodically
        if len(self.task_outcomes) % 10 == 0:
            self._extract_meta_patterns()
    
    def learn_from_failure(self, 
                          task_id: str,
                          task_features: TaskFeatures,
                          attempted_strategies: List[str],
                          errors: List[str]) -> Dict[str, Any]:
        """Analyze failure and extract lessons.
        
        Returns:
            Dictionary of insights learned from the failure
        """
        insights = {
            'task_id': task_id,
            'failed_strategies': attempted_strategies,
            'error_patterns': [],
            'suggested_improvements': []
        }
        
        # Analyze error patterns
        for error in errors:
            if 'index' in error and 'out of bounds' in error:
                insights['error_patterns'].append('index_out_of_bounds')
                insights['suggested_improvements'].append('Add bounds checking')
            elif 'broadcast' in error:
                insights['error_patterns'].append('shape_mismatch')
                insights['suggested_improvements'].append('Handle variable shapes')
            elif 'missing' in error and 'argument' in error:
                insights['error_patterns'].append('api_mismatch')
                insights['suggested_improvements'].append('Fix API calls')
        
        # Check if task type is completely new
        feature_vec = task_features.to_vector()
        all_known_features = []
        
        for knowledge in self.strategies.values():
            all_known_features.extend(knowledge.successful_features)
            all_known_features.extend(knowledge.failed_features)
        
        if all_known_features:
            min_dist = min(np.linalg.norm(feature_vec - kf) for kf in all_known_features)
            if min_dist > 20:  # Threshold for "very different"
                insights['error_patterns'].append('novel_task_type')
                insights['suggested_improvements'].append('Need new strategy type')
        
        # Store failure for future learning
        for strategy in attempted_strategies:
            outcome = StrategyOutcome(
                task_id=task_id,
                task_features=task_features,
                strategy_name=strategy,
                success=False,
                accuracy=0.0,
                time_taken=0.0,
                error_type=insights['error_patterns'][0] if insights['error_patterns'] else 'unknown'
            )
            self.learn_from_outcome(outcome)
        
        return insights
    
    def adapt_strategy(self, 
                      base_strategy: str,
                      task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt a strategy based on task context and past experience.
        
        Returns:
            Dictionary of adaptations to apply
        """
        adaptations = {
            'strategy': base_strategy,
            'modifications': []
        }
        
        if base_strategy not in self.strategies:
            return adaptations
        
        knowledge = self.strategies[base_strategy]
        
        # Check common error patterns for this strategy
        if 'index_out_of_bounds' in knowledge.error_counts:
            if knowledge.error_counts['index_out_of_bounds'] > 2:
                adaptations['modifications'].append({
                    'type': 'add_bounds_check',
                    'reason': 'Frequent index errors'
                })
        
        if 'shape_mismatch' in knowledge.error_counts:
            if knowledge.error_counts['shape_mismatch'] > 2:
                adaptations['modifications'].append({
                    'type': 'flexible_shapes',
                    'reason': 'Frequent shape mismatches'
                })
        
        # Check if strategy typically needs more time
        if knowledge.avg_time > 5.0:
            adaptations['modifications'].append({
                'type': 'increase_timeout',
                'value': knowledge.avg_time * 1.5,
                'reason': 'Strategy typically needs more time'
            })
        
        return adaptations
    
    def extract_meta_patterns(self) -> List[Dict[str, Any]]:
        """Extract high-level patterns from successful inventions.
        
        Returns:
            List of discovered meta-patterns
        """
        return self._extract_meta_patterns()
    
    def _extract_meta_patterns(self) -> List[Dict[str, Any]]:
        """Internal method to extract meta-patterns."""
        
        patterns = []
        
        # Pattern 1: Strategy success correlation with task features
        for strategy_name, knowledge in self.strategies.items():
            if knowledge.successes > 5:  # Enough data
                # Find common features in successful cases
                if knowledge.successful_features:
                    avg_features = np.mean(knowledge.successful_features, axis=0)
                    
                    pattern = {
                        'type': 'strategy_affinity',
                        'strategy': strategy_name,
                        'success_rate': knowledge.success_rate(),
                        'typical_features': avg_features.tolist(),
                        'confidence': min(knowledge.successes / 10, 1.0)
                    }
                    patterns.append(pattern)
        
        # Pattern 2: Task type clusters
        if len(self.task_outcomes) > 20:
            successful_outcomes = [o for o in self.task_outcomes if o.success]
            
            if successful_outcomes:
                # Group by strategy
                by_strategy = defaultdict(list)
                for outcome in successful_outcomes:
                    by_strategy[outcome.strategy_name].append(outcome.task_features.to_vector())
                
                for strategy, features in by_strategy.items():
                    if len(features) > 3:
                        pattern = {
                            'type': 'task_cluster',
                            'strategy': strategy,
                            'num_tasks': len(features),
                            'center': np.mean(features, axis=0).tolist()
                        }
                        patterns.append(pattern)
        
        self.meta_patterns = patterns
        return patterns
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of what has been learned."""
        
        summary = {
            'total_tasks_seen': len(self.task_outcomes),
            'strategies_learned': len(self.strategies),
            'meta_patterns_discovered': len(self.meta_patterns),
            'overall_success_rate': 0.0,
            'strategy_performance': {},
            'top_errors': {}
        }
        
        # Calculate overall success rate
        if self.task_outcomes:
            successes = sum(1 for o in self.task_outcomes if o.success)
            summary['overall_success_rate'] = successes / len(self.task_outcomes)
        
        # Strategy performance
        for name, knowledge in self.strategies.items():
            summary['strategy_performance'][name] = {
                'success_rate': knowledge.success_rate(),
                'attempts': knowledge.total_attempts,
                'avg_accuracy': knowledge.avg_accuracy,
                'avg_time': knowledge.avg_time
            }
        
        # Top errors across all strategies
        all_errors = defaultdict(int)
        for knowledge in self.strategies.values():
            for error, count in knowledge.error_counts.items():
                all_errors[error] += count
        
        summary['top_errors'] = dict(sorted(all_errors.items(), 
                                           key=lambda x: x[1], 
                                           reverse=True)[:5])
        
        return summary
    
    def _check_repeating_pattern(self, grid: np.ndarray) -> bool:
        """Check if grid has repeating pattern."""
        h, w = grid.shape
        
        # Check for 2x2, 3x3 patterns
        for size in [2, 3]:
            if h >= size * 2 and w >= size * 2:
                pattern = grid[:size, :size]
                
                repeats = True
                for i in range(0, h - size + 1, size):
                    for j in range(0, w - size + 1, size):
                        if not np.array_equal(grid[i:i+size, j:j+size], pattern):
                            repeats = False
                            break
                    if not repeats:
                        break
                
                if repeats:
                    return True
        
        return False
    
    def _check_symmetry(self, grid: np.ndarray) -> bool:
        """Check for symmetry in the grid."""
        # Horizontal symmetry
        if np.array_equal(grid, np.flip(grid, axis=1)):
            return True
        # Vertical symmetry
        if np.array_equal(grid, np.flip(grid, axis=0)):
            return True
        # Diagonal symmetry (if square)
        if grid.shape[0] == grid.shape[1]:
            if np.array_equal(grid, grid.T):
                return True
        return False
    
    def _check_objects(self, grid: np.ndarray) -> bool:
        """Check if grid contains distinct objects."""
        unique_colors = [c for c in np.unique(grid) if c != 0]
        return len(unique_colors) > 1 and len(unique_colors) < grid.size / 4
    
    def _check_grid_structure(self, grid: np.ndarray) -> bool:
        """Check if grid has regular structure."""
        # Simple check: are values arranged in regular intervals?
        h, w = grid.shape
        
        # Check for regular spacing in rows
        for row in grid:
            unique = np.unique(row)
            if len(unique) > 2 and len(unique) < w / 2:
                # Check if positions are regular
                for val in unique:
                    if val != 0:
                        positions = np.where(row == val)[0]
                        if len(positions) > 2:
                            diffs = np.diff(positions)
                            if np.all(diffs == diffs[0]):
                                return True
        
        return False
    
    def _is_likely_geometric(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """Check if transformation is likely geometric."""
        if not examples:
            return False
        
        inp, out = examples[0]
        
        # Check for rotations
        if inp.shape == out.shape:
            for k in [1, 2, 3]:
                if np.array_equal(np.rot90(inp, k), out):
                    return True
        
        # Check for flips
        if np.array_equal(np.flip(inp, axis=0), out):
            return True
        if np.array_equal(np.flip(inp, axis=1), out):
            return True
        
        return False
    
    def _calculate_complexity(self, grid: np.ndarray) -> float:
        """Calculate complexity of a grid (entropy-based)."""
        unique, counts = np.unique(grid, return_counts=True)
        probabilities = counts / grid.size
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def _calculate_consistency(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Calculate how consistent examples are with each other."""
        if len(examples) < 2:
            return 1.0
        
        # Compare output shapes
        output_shapes = [out.shape for _, out in examples]
        if len(set(output_shapes)) == 1:
            shape_consistency = 1.0
        else:
            shape_consistency = 0.5
        
        # Compare color usage
        color_sets = [set(np.unique(out)) for _, out in examples]
        if all(cs == color_sets[0] for cs in color_sets):
            color_consistency = 1.0
        else:
            # Calculate Jaccard similarity
            intersection = set.intersection(*color_sets) if color_sets else set()
            union = set.union(*color_sets) if color_sets else set()
            color_consistency = len(intersection) / len(union) if union else 0.0
        
        return (shape_consistency + color_consistency) / 2
    
    def save(self):
        """Save learned knowledge to disk."""
        try:
            save_data = {
                'strategies': {},
                'task_outcomes': [],
                'meta_patterns': self.meta_patterns,
                'exploration_rate': self.exploration_rate
            }
            
            # Save strategy knowledge
            for name, knowledge in self.strategies.items():
                save_data['strategies'][name] = {
                    'name': name,
                    'total_attempts': knowledge.total_attempts,
                    'successes': knowledge.successes,
                    'failures': knowledge.failures,
                    'avg_accuracy': knowledge.avg_accuracy,
                    'avg_time': knowledge.avg_time,
                    'error_counts': dict(knowledge.error_counts),
                    'successful_features': [f.tolist() for f in knowledge.successful_features[-50:]],
                    'failed_features': [f.tolist() for f in knowledge.failed_features[-50:]]
                }
            
            # Save recent task outcomes
            for outcome in self.task_outcomes[-100:]:  # Keep last 100
                save_data['task_outcomes'].append({
                    'task_id': outcome.task_id,
                    'strategy_name': outcome.strategy_name,
                    'success': outcome.success,
                    'accuracy': outcome.accuracy,
                    'time_taken': outcome.time_taken,
                    'error_type': outcome.error_type
                })
            
            with open(self.storage_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            print(f"Saved meta-learning knowledge to {self.storage_path}")
            
        except Exception as e:
            print(f"Error saving meta-learner: {e}")
    
    def load(self):
        """Load learned knowledge from disk."""
        try:
            if not self.storage_path.exists():
                return
            
            with open(self.storage_path, 'r') as f:
                save_data = json.load(f)
            
            # Load strategy knowledge
            for name, data in save_data.get('strategies', {}).items():
                knowledge = StrategyKnowledge(
                    name=name,
                    total_attempts=data['total_attempts'],
                    successes=data['successes'],
                    failures=data['failures'],
                    avg_accuracy=data['avg_accuracy'],
                    avg_time=data['avg_time']
                )
                
                knowledge.error_counts = defaultdict(int, data.get('error_counts', {}))
                knowledge.successful_features = [np.array(f) for f in data.get('successful_features', [])]
                knowledge.failed_features = [np.array(f) for f in data.get('failed_features', [])]
                
                self.strategies[name] = knowledge
            
            # Load meta patterns
            self.meta_patterns = save_data.get('meta_patterns', [])
            self.exploration_rate = save_data.get('exploration_rate', 0.2)
            
            print(f"Loaded meta-learning knowledge from {self.storage_path}")
            print(f"  Strategies: {len(self.strategies)}")
            print(f"  Meta-patterns: {len(self.meta_patterns)}")
            
        except Exception as e:
            print(f"Error loading meta-learner: {e}")