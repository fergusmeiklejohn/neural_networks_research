"""
Invention Composer

Composes simple inventions into complex solutions through various composition strategies.
This enables solving complex tasks by combining simpler solutions.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Any, Union
from dataclasses import dataclass
import inspect
from copy import deepcopy


@dataclass
class ComposedInvention:
    """A complex invention composed from simpler ones."""
    name: str
    components: List[Any]  # List of inventions or functions
    composition_type: str  # 'sequential', 'parallel', 'conditional', 'iterative'
    parameters: Dict[str, Any]
    function: Callable
    description: str
    score: float = 0.0


class InventionComposer:
    """Compose simple inventions into complex solutions."""
    
    def __init__(self):
        self.composition_strategies = {
            'sequential': self.sequential_composition,
            'parallel': self.parallel_composition,
            'conditional': self.conditional_composition,
            'iterative': self.iterative_composition,
            'hierarchical': self.hierarchical_composition
        }
        
        self.composition_history = []
    
    def compose(self,
               inventions: List[Any],
               composition_type: str = 'sequential',
               **kwargs) -> ComposedInvention:
        """
        Compose multiple inventions into a complex solution.
        
        Args:
            inventions: List of inventions or functions to compose
            composition_type: Type of composition
            **kwargs: Additional parameters for composition
            
        Returns:
            ComposedInvention with the composed function
        """
        if composition_type not in self.composition_strategies:
            raise ValueError(f"Unknown composition type: {composition_type}")
        
        composer = self.composition_strategies[composition_type]
        composed_function = composer(inventions, **kwargs)
        
        # Create composed invention
        result = ComposedInvention(
            name=f"{composition_type}_composition",
            components=inventions,
            composition_type=composition_type,
            parameters=kwargs,
            function=composed_function,
            description=self._generate_description(inventions, composition_type)
        )
        
        self.composition_history.append(result)
        return result
    
    def sequential_composition(self, 
                              inventions: List[Any],
                              **kwargs) -> Callable:
        """
        Chain inventions: output of one feeds into the next.
        A → B → C
        
        Args:
            inventions: List of functions to chain
            
        Returns:
            Composed function
        """
        def composed_function(grid: np.ndarray) -> np.ndarray:
            result = grid.copy()
            
            for invention in inventions:
                # Handle different invention types
                if callable(invention):
                    result = invention(result)
                elif hasattr(invention, 'function'):
                    result = invention.function(result)
                elif hasattr(invention, '__call__'):
                    result = invention(result)
                else:
                    # Skip non-callable inventions
                    continue
                    
                # Ensure result is valid
                if result is None:
                    return grid  # Return original if composition fails
                    
            return result
        
        return composed_function
    
    def parallel_composition(self,
                           inventions: List[Any],
                           regions: Optional[List[Tuple[int, int, int, int]]] = None,
                           merge_strategy: str = 'overlay',
                           **kwargs) -> Callable:
        """
        Apply different inventions to different parts of the grid.
        
        Args:
            inventions: List of functions to apply
            regions: Optional list of (r1, c1, r2, c2) regions for each invention
            merge_strategy: How to merge results ('overlay', 'max', 'sum', 'replace')
            
        Returns:
            Composed function
        """
        def composed_function(grid: np.ndarray) -> np.ndarray:
            h, w = grid.shape
            results = []
            
            for i, invention in enumerate(inventions):
                # Get function
                if callable(invention):
                    func = invention
                elif hasattr(invention, 'function'):
                    func = invention.function
                else:
                    continue
                
                # Apply to region or whole grid
                if regions and i < len(regions):
                    r1, c1, r2, c2 = regions[i]
                    # Clip bounds
                    r1 = max(0, min(r1, h - 1))
                    c1 = max(0, min(c1, w - 1))
                    r2 = max(0, min(r2, h - 1))
                    c2 = max(0, min(c2, w - 1))
                    
                    # Extract and process region
                    region = grid[r1:r2+1, c1:c2+1]
                    processed = func(region)
                    
                    # Create full-size result
                    result = np.zeros_like(grid)
                    if processed is not None:
                        result[r1:r2+1, c1:c2+1] = processed
                    results.append(result)
                else:
                    # Apply to whole grid
                    result = func(grid)
                    if result is not None:
                        results.append(result)
            
            # Merge results
            if not results:
                return grid
            
            return self._merge_results(results, grid, merge_strategy)
        
        return composed_function
    
    def conditional_composition(self,
                              condition: Union[Callable, str],
                              if_true: Any,
                              if_false: Any,
                              **kwargs) -> Callable:
        """
        Apply different inventions based on condition.
        
        Args:
            condition: Function that returns bool or string condition type
            if_true: Invention to apply if condition is true
            if_false: Invention to apply if condition is false
            
        Returns:
            Composed function
        """
        def composed_function(grid: np.ndarray) -> np.ndarray:
            # Evaluate condition
            if callable(condition):
                cond_result = condition(grid)
            elif isinstance(condition, str):
                cond_result = self._evaluate_string_condition(condition, grid)
            else:
                cond_result = bool(condition)
            
            # Apply appropriate invention
            if cond_result:
                invention = if_true
            else:
                invention = if_false
            
            # Apply invention
            if callable(invention):
                return invention(grid)
            elif hasattr(invention, 'function'):
                return invention.function(grid)
            else:
                return grid
        
        return composed_function
    
    def iterative_composition(self,
                            invention: Any,
                            max_iterations: int = 10,
                            stop_condition: Optional[Callable] = None,
                            **kwargs) -> Callable:
        """
        Apply invention iteratively until condition is met.
        
        Args:
            invention: Function to apply iteratively
            max_iterations: Maximum number of iterations
            stop_condition: Optional function to check if should stop
            
        Returns:
            Composed function
        """
        def composed_function(grid: np.ndarray) -> np.ndarray:
            result = grid.copy()
            
            # Get function
            if callable(invention):
                func = invention
            elif hasattr(invention, 'function'):
                func = invention.function
            else:
                return grid
            
            for i in range(max_iterations):
                prev_result = result.copy()
                result = func(result)
                
                if result is None:
                    return prev_result
                
                # Check stop condition
                if stop_condition and stop_condition(result, prev_result, i):
                    break
                
                # Check if converged
                if np.array_equal(result, prev_result):
                    break
            
            return result
        
        return composed_function
    
    def hierarchical_composition(self,
                               inventions: List[Any],
                               hierarchy: Dict[str, Any],
                               **kwargs) -> Callable:
        """
        Apply inventions in a hierarchical structure.
        
        Args:
            inventions: List of inventions
            hierarchy: Dictionary describing hierarchical structure
            
        Returns:
            Composed function
        """
        def composed_function(grid: np.ndarray) -> np.ndarray:
            # Parse hierarchy and apply inventions
            return self._apply_hierarchy(grid, inventions, hierarchy)
        
        return composed_function
    
    def _apply_hierarchy(self,
                        grid: np.ndarray,
                        inventions: List[Any],
                        hierarchy: Dict[str, Any]) -> np.ndarray:
        """Apply inventions according to hierarchical structure."""
        result = grid.copy()
        
        # Extract hierarchy type
        h_type = hierarchy.get('type', 'sequential')
        
        if h_type == 'sequential':
            # Apply in sequence
            for item in hierarchy.get('sequence', []):
                if isinstance(item, int) and item < len(inventions):
                    inv = inventions[item]
                    if callable(inv):
                        result = inv(result)
                    elif hasattr(inv, 'function'):
                        result = inv.function(result)
                elif isinstance(item, dict):
                    # Nested hierarchy
                    result = self._apply_hierarchy(result, inventions, item)
        
        elif h_type == 'parallel':
            # Apply in parallel and merge
            results = []
            for item in hierarchy.get('parallel', []):
                if isinstance(item, int) and item < len(inventions):
                    inv = inventions[item]
                    if callable(inv):
                        results.append(inv(grid))
                    elif hasattr(inv, 'function'):
                        results.append(inv.function(grid))
            
            if results:
                result = self._merge_results(results, grid, 
                                           hierarchy.get('merge', 'overlay'))
        
        elif h_type == 'conditional':
            # Conditional application
            cond_idx = hierarchy.get('condition_idx', 0)
            true_idx = hierarchy.get('true_idx', 1)
            false_idx = hierarchy.get('false_idx', 2)
            
            if cond_idx < len(inventions):
                cond = inventions[cond_idx]
                # Evaluate condition
                if callable(cond):
                    cond_result = cond(grid)
                    # If it returns a grid, check if it's different
                    if isinstance(cond_result, np.ndarray):
                        cond_result = not np.array_equal(cond_result, grid)
                else:
                    cond_result = False
                
                # Apply appropriate branch
                idx = true_idx if cond_result else false_idx
                if idx < len(inventions):
                    inv = inventions[idx]
                    if callable(inv):
                        result = inv(grid)
                    elif hasattr(inv, 'function'):
                        result = inv.function(grid)
        
        return result
    
    def _merge_results(self,
                      results: List[np.ndarray],
                      original: np.ndarray,
                      strategy: str) -> np.ndarray:
        """Merge multiple results according to strategy."""
        if not results:
            return original
        
        if strategy == 'overlay':
            # Later results overlay earlier ones (non-zero values)
            merged = results[0].copy()
            for result in results[1:]:
                mask = result != 0
                merged[mask] = result[mask]
            return merged
        
        elif strategy == 'max':
            # Take maximum value at each position
            return np.maximum.reduce(results)
        
        elif strategy == 'sum':
            # Sum all results
            return np.sum(results, axis=0)
        
        elif strategy == 'replace':
            # Last result replaces all
            return results[-1]
        
        elif strategy == 'vote':
            # Majority voting at each position
            stacked = np.stack(results)
            # Mode calculation
            from scipy import stats
            mode_result = stats.mode(stacked, axis=0, keepdims=False)
            return mode_result.mode
        
        else:
            # Default to overlay
            return self._merge_results(results, original, 'overlay')
    
    def _evaluate_string_condition(self, condition: str, grid: np.ndarray) -> bool:
        """Evaluate string-based conditions."""
        if condition == 'has_color':
            return np.any(grid != 0)
        elif condition == 'is_empty':
            return np.all(grid == 0)
        elif condition == 'has_pattern':
            # Check for repeated patterns
            unique = len(np.unique(grid))
            return unique > 2
        elif condition == 'is_symmetric':
            return (np.array_equal(grid, np.fliplr(grid)) or 
                   np.array_equal(grid, np.flipud(grid)))
        else:
            return False
    
    def _generate_description(self, 
                            inventions: List[Any],
                            composition_type: str) -> str:
        """Generate human-readable description of composition."""
        inv_names = []
        for inv in inventions:
            if hasattr(inv, 'name'):
                inv_names.append(inv.name)
            elif hasattr(inv, '__name__'):
                inv_names.append(inv.__name__)
            else:
                inv_names.append('function')
        
        if composition_type == 'sequential':
            return f"Sequential: {' → '.join(inv_names)}"
        elif composition_type == 'parallel':
            return f"Parallel: {' || '.join(inv_names)}"
        elif composition_type == 'conditional':
            return f"Conditional: if condition then {inv_names[0]} else {inv_names[1] if len(inv_names) > 1 else 'none'}"
        elif composition_type == 'iterative':
            return f"Iterative: {inv_names[0]} (repeated)"
        else:
            return f"{composition_type}: {', '.join(inv_names)}"
    
    def suggest_composition(self,
                          inventions: List[Any],
                          examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[ComposedInvention]:
        """
        Suggest best composition strategy based on examples.
        
        Args:
            inventions: Available inventions
            examples: Input-output examples
            
        Returns:
            Best composed invention or None
        """
        best_composition = None
        best_score = 0
        
        # Try different composition strategies
        for comp_type in ['sequential', 'parallel', 'conditional']:
            try:
                # Create composition
                if comp_type == 'conditional' and len(inventions) >= 2:
                    # Need at least 2 inventions for conditional
                    composed = self.compose(
                        inventions[:2],
                        comp_type,
                        condition='has_pattern',
                        if_true=inventions[0],
                        if_false=inventions[1] if len(inventions) > 1 else inventions[0]
                    )
                else:
                    composed = self.compose(inventions, comp_type)
                
                # Test on examples
                score = self._evaluate_composition(composed, examples)
                
                if score > best_score:
                    best_score = score
                    best_composition = composed
                    best_composition.score = score
                    
            except Exception:
                continue
        
        return best_composition
    
    def _evaluate_composition(self,
                            composition: ComposedInvention,
                            examples: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Evaluate how well a composition performs on examples."""
        if not examples:
            return 0.0
        
        total_score = 0.0
        
        for input_grid, expected_output in examples:
            try:
                result = composition.function(input_grid)
                
                if result is None:
                    continue
                
                # Calculate similarity
                if result.shape == expected_output.shape:
                    matches = np.sum(result == expected_output)
                    total = result.size
                    score = matches / total if total > 0 else 0
                else:
                    score = 0
                
                total_score += score
                
            except Exception:
                continue
        
        return total_score / len(examples)
    
    def learn_composition_patterns(self,
                                 successful_compositions: List[ComposedInvention]) -> Dict[str, Any]:
        """
        Learn patterns from successful compositions.
        
        Args:
            successful_compositions: List of compositions that worked well
            
        Returns:
            Dictionary of learned patterns
        """
        patterns = {
            'common_sequences': [],
            'effective_combinations': [],
            'composition_types': {},
            'parameter_patterns': {}
        }
        
        # Analyze composition types
        type_counts = {}
        for comp in successful_compositions:
            comp_type = comp.composition_type
            type_counts[comp_type] = type_counts.get(comp_type, 0) + 1
            
            # Track parameters for each type
            if comp_type not in patterns['parameter_patterns']:
                patterns['parameter_patterns'][comp_type] = []
            patterns['parameter_patterns'][comp_type].append(comp.parameters)
        
        patterns['composition_types'] = type_counts
        
        # Find common sequences
        sequences = []
        for comp in successful_compositions:
            if comp.composition_type == 'sequential':
                # Extract sequence of component names
                seq = []
                for component in comp.components:
                    if hasattr(component, 'name'):
                        seq.append(component.name)
                    elif hasattr(component, '__name__'):
                        seq.append(component.__name__)
                sequences.append(tuple(seq))
        
        # Find most common sequences
        from collections import Counter
        seq_counts = Counter(sequences)
        patterns['common_sequences'] = seq_counts.most_common(5)
        
        return patterns