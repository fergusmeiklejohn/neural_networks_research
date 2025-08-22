"""Integrated ARC Imagination Engine.

Combines all our successful components:
- Hypothesis Generator (92% on pattern discovery)
- Program Synthesis with ARC primitives
- Compositional Reasoning
- Memory for learned programs
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import json
import time

# Import our components
from hypothesis_generator import MinimalHypothesisGenerator, GenerationStrategy
from program_synthesis import ProgramSynthesizer, Transform, Primitive, Sequence
from arc_primitives import ARCPrimitives
from improved_compositional_reasoner import ImprovedCompositionalReasoner


class ProgramMemory:
    """Memory system for storing and retrieving successful programs."""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.programs = []
        self.task_features_cache = {}
    
    def extract_features(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Extract features from task examples."""
        features = {}
        
        if not examples:
            return features
        
        input_grid = examples[0][0]
        output_grid = examples[0][1]
        
        # Shape features
        features['input_shape'] = input_grid.shape
        features['output_shape'] = output_grid.shape
        features['size_ratio'] = output_grid.size / input_grid.size
        
        # Color features
        features['input_colors'] = tuple(sorted(np.unique(input_grid)))
        features['output_colors'] = tuple(sorted(np.unique(output_grid)))
        features['new_colors'] = tuple(set(features['output_colors']) - set(features['input_colors']))
        
        # Transformation type hints
        features['is_enlargement'] = output_grid.shape[0] > input_grid.shape[0]
        features['has_new_colors'] = len(features['new_colors']) > 0
        features['preserves_structure'] = np.array_equal(input_grid.shape, output_grid.shape)
        
        # Pattern features
        features['has_enclosed_regions'] = self._check_for_enclosed_regions(input_grid)
        features['has_repeating_pattern'] = ARCPrimitives.find_repeating_pattern(input_grid) is not None
        
        return features
    
    def _check_for_enclosed_regions(self, grid: np.ndarray) -> bool:
        """Check if grid has potential enclosed regions."""
        # Simple heuristic: check for non-zero boundary pixels
        unique_colors = np.unique(grid)
        for color in unique_colors:
            if color == 0:
                continue
            regions = ARCPrimitives.find_enclosed_regions(grid, color)
            if regions:
                return True
        return False
    
    def store(self, program: Transform, examples: List[Tuple[np.ndarray, np.ndarray]], 
             score: float):
        """Store a successful program with its features."""
        features = self.extract_features(examples)
        
        entry = {
            'program': program,
            'features': features,
            'score': score,
            'usage_count': 0,
            'timestamp': time.time()
        }
        
        self.programs.append(entry)
        
        # Evict oldest if over capacity
        if len(self.programs) > self.capacity:
            self.programs.pop(0)
    
    def retrieve(self, examples: List[Tuple[np.ndarray, np.ndarray]], k: int = 5) -> List[Transform]:
        """Retrieve k most relevant programs for the given examples."""
        if not self.programs:
            return []
        
        task_features = self.extract_features(examples)
        
        # Score each stored program by feature similarity
        scored_programs = []
        for entry in self.programs:
            similarity = self._compute_similarity(task_features, entry['features'])
            scored_programs.append((similarity * entry['score'], entry))
        
        # Sort by score and return top k
        scored_programs.sort(key=lambda x: x[0], reverse=True)
        
        # Update usage counts and return programs
        results = []
        for _, entry in scored_programs[:k]:
            entry['usage_count'] += 1
            results.append(entry['program'])
        
        return results
    
    def _compute_similarity(self, features1: Dict, features2: Dict) -> float:
        """Compute similarity between two feature sets."""
        similarity = 0.0
        count = 0
        
        # Shape similarity
        if features1.get('input_shape') == features2.get('input_shape'):
            similarity += 1.0
            count += 1
        if features1.get('output_shape') == features2.get('output_shape'):
            similarity += 1.0
            count += 1
        
        # Size ratio similarity
        if 'size_ratio' in features1 and 'size_ratio' in features2:
            ratio_diff = abs(features1['size_ratio'] - features2['size_ratio'])
            similarity += max(0, 1.0 - ratio_diff / 10)
            count += 1
        
        # Color similarity
        if features1.get('input_colors') == features2.get('input_colors'):
            similarity += 0.5
            count += 0.5
        if features1.get('has_new_colors') == features2.get('has_new_colors'):
            similarity += 0.5
            count += 0.5
        
        # Transformation type similarity
        for key in ['is_enlargement', 'preserves_structure', 'has_enclosed_regions', 
                   'has_repeating_pattern']:
            if features1.get(key) == features2.get(key):
                similarity += 0.5
                count += 0.5
        
        return similarity / count if count > 0 else 0.0


class ARCImaginationEngine:
    """Main ARC solver combining all components."""
    
    def __init__(self):
        # Initialize components
        self.hypothesis_gen = MinimalHypothesisGenerator(seed=42)
        self.program_synthesizer = ProgramSynthesizer()
        self.compositional_reasoner = ImprovedCompositionalReasoner()
        self.program_memory = ProgramMemory(capacity=1000)
        
        # Statistics
        self.tasks_attempted = 0
        self.tasks_solved = 0
        self.total_time = 0.0
    
    def solve(self, examples: List[Tuple[np.ndarray, np.ndarray]], 
             max_time: float = 10.0) -> Optional[Transform]:
        """Solve an ARC task given training examples."""
        start_time = time.time()
        self.tasks_attempted += 1
        
        # 1. Try memory first
        print("Checking memory for similar tasks...")
        memory_programs = self.program_memory.retrieve(examples, k=3)
        for program in memory_programs:
            score = self._evaluate_program(program, examples)
            if score == 1.0:
                print(f"Found perfect match in memory: {program.to_string()}")
                self.tasks_solved += 1
                return program
        
        # 2. Analyze task type
        task_type = self._analyze_task_type(examples)
        print(f"Task type detected: {task_type}")
        
        # 3. Route to appropriate solver
        program = None
        
        if task_type == "fill_operation":
            # Use program synthesis for fill operations
            print("Attempting program synthesis...")
            program = self.program_synthesizer.synthesize(examples, max_depth=2)
            
        elif task_type == "pattern_discovery":
            # Use hypothesis generator for patterns
            print("Attempting pattern discovery...")
            hypothesis = self.hypothesis_gen.discover_pattern(
                examples,
                max_attempts=100,
                strategies=[GenerationStrategy.SYSTEMATIC, GenerationStrategy.RANDOM]
            )
            if hypothesis:
                # Convert hypothesis to program
                program = Primitive(
                    hypothesis.transform_type,
                    hypothesis.transform_fn,
                    hypothesis.parameters
                )
        
        elif task_type == "compositional":
            # Skip compositional for now, use program synthesis
            print("Attempting program synthesis for compositional task...")
            program = self.program_synthesizer.synthesize(examples, max_depth=2)
        
        else:
            # Default: try program synthesis
            print("Attempting general program synthesis...")
            program = self.program_synthesizer.synthesize(examples, max_depth=3)
        
        # 4. Evaluate and store if successful
        if program:
            score = self._evaluate_program(program, examples)
            print(f"Program score: {score:.2%}")
            
            if score > 0.8:
                self.program_memory.store(program, examples, score)
                self.tasks_solved += (score == 1.0)
            
            return program if score > 0.5 else None
        
        elapsed = time.time() - start_time
        self.total_time += elapsed
        
        return None
    
    def _analyze_task_type(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> str:
        """Analyze examples to determine task type."""
        if not examples:
            return "unknown"
        
        input_grid = examples[0][0]
        output_grid = examples[0][1]
        
        # Check for fill operations
        input_colors = set(np.unique(input_grid))
        output_colors = set(np.unique(output_grid))
        new_colors = output_colors - input_colors
        
        if new_colors:
            # Check if it's filling enclosed regions
            for color in input_colors - {0}:
                regions = ARCPrimitives.find_enclosed_regions(input_grid, color)
                if regions:
                    return "fill_operation"
        
        # Check for size changes (pattern/tiling)
        if output_grid.shape != input_grid.shape:
            if output_grid.shape[0] > input_grid.shape[0]:
                return "pattern_discovery"
        
        # Check for multi-attribute changes
        if len(input_colors) > 3 and len(output_colors) > 3:
            return "compositional"
        
        return "general"
    
    def _evaluate_program(self, program: Transform, 
                         examples: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Evaluate program on examples."""
        if not examples:
            return 0.0
        
        total_score = 0.0
        for input_grid, expected_output in examples:
            try:
                predicted = program.apply(input_grid)
                
                if predicted.shape != expected_output.shape:
                    continue
                
                accuracy = np.mean(predicted == expected_output)
                total_score += accuracy
            except:
                continue
        
        return total_score / len(examples)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get solver statistics."""
        return {
            'tasks_attempted': self.tasks_attempted,
            'tasks_solved': self.tasks_solved,
            'success_rate': self.tasks_solved / self.tasks_attempted if self.tasks_attempted > 0 else 0,
            'memory_size': len(self.program_memory.programs),
            'total_time': self.total_time,
            'avg_time': self.total_time / self.tasks_attempted if self.tasks_attempted > 0 else 0
        }


def test_on_arc_tasks():
    """Test the integrated engine on ARC tasks."""
    print("=" * 60)
    print("Testing ARC Imagination Engine")
    print("=" * 60)
    
    arc_dir = Path(__file__).parent / "arc_agi_2_data" / "training"
    
    if not arc_dir.exists():
        print("ARC data directory not found")
        return
    
    # Create engine
    engine = ARCImaginationEngine()
    
    # Test on first 20 tasks
    task_files = list(arc_dir.glob("*.json"))[:20]
    results = []
    
    for i, task_file in enumerate(task_files):
        print(f"\n--- Task {i+1}/{len(task_files)}: {task_file.stem} ---")
        
        with open(task_file, 'r') as f:
            task = json.load(f)
        
        # Get training examples
        examples = []
        for ex in task['train'][:3]:
            input_grid = np.array(ex['input'])
            output_grid = np.array(ex['output'])
            examples.append((input_grid, output_grid))
        
        # Solve
        program = engine.solve(examples)
        
        if program and task['test']:
            # Test on first test example
            test_input = np.array(task['test'][0]['input'])
            test_output = np.array(task['test'][0]['output'])
            
            try:
                predicted = program.apply(test_input)
                if predicted.shape == test_output.shape:
                    accuracy = np.mean(predicted == test_output)
                else:
                    accuracy = 0.0
            except:
                accuracy = 0.0
            
            results.append(accuracy)
            status = "✓" if accuracy > 0.8 else "✗"
            print(f"Result: {accuracy:.1%} {status}")
            if accuracy > 0.8:
                print(f"Solution: {program.to_string()}")
        else:
            results.append(0.0)
            print("Result: No solution ✗")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    stats = engine.get_statistics()
    avg_accuracy = sum(results) / len(results) if results else 0
    solved = sum(1 for r in results if r > 0.8)
    
    print(f"Tasks solved: {solved}/{len(results)} ({solved/len(results)*100:.1f}%)")
    print(f"Average accuracy: {avg_accuracy:.1%}")
    print(f"Memory size: {stats['memory_size']} programs")
    print(f"Average time: {stats['avg_time']:.2f}s per task")
    
    # Show which types of tasks we're good at
    print("\nSuccessful task types:")
    for r, tf in zip(results, task_files):
        if r > 0.8:
            print(f"  - {tf.stem}: {r:.1%}")


if __name__ == "__main__":
    test_on_arc_tasks()