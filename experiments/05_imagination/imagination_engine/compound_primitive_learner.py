"""Learn compound primitives from successful programs.

This module automatically creates new primitives from successful program
sequences, enabling the system to learn and reuse complex patterns.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Callable
from dataclasses import dataclass
import json
from pathlib import Path

from program_synthesis_v2 import Transform, Primitive, Sequence


@dataclass
class CompoundPrimitive:
    """A learned compound primitive created from successful programs."""
    name: str
    description: str
    component_sequence: List[Transform]
    task_signatures: List[Dict[str, Any]]  # Features of tasks it solved
    success_count: int
    
    def to_primitive(self) -> Primitive:
        """Convert to a standard Primitive for use in synthesis."""
        def apply_compound(grid: np.ndarray) -> np.ndarray:
            result = grid
            for transform in self.component_sequence:
                result = transform.apply(result)
            return result
        
        return Primitive(
            name=self.name,
            func=apply_compound,
            params={"components": len(self.component_sequence)}
        )
    
    def matches_task(self, task_features: Dict[str, Any]) -> float:
        """Calculate similarity between task and this primitive's signatures."""
        if not self.task_signatures:
            return 0.0
        
        best_match = 0.0
        for signature in self.task_signatures:
            match_score = 0.0
            match_count = 0
            
            for key in ['input_shape', 'output_shape', 'input_colors', 'output_colors']:
                if key in signature and key in task_features:
                    if signature[key] == task_features[key]:
                        match_score += 1.0
                    match_count += 1
            
            if match_count > 0:
                best_match = max(best_match, match_score / match_count)
        
        return best_match


class CompoundPrimitiveLearner:
    """Learn and manage compound primitives from successful programs."""
    
    def __init__(self, storage_path: str = "learned_compounds.json"):
        self.storage_path = Path(storage_path)
        self.compounds: List[CompoundPrimitive] = []
        self.load_compounds()
    
    def learn_from_solution(self, 
                          solution: Transform,
                          task_features: Dict[str, Any],
                          accuracy: float) -> Optional[CompoundPrimitive]:
        """Learn a compound primitive from a successful solution."""
        
        # Only learn from high-accuracy solutions
        if accuracy < 0.95:
            return None
        
        # Only learn from sequences (single primitives are already known)
        if not isinstance(solution, Sequence):
            return None
        
        # Check if sequence is non-trivial
        if len(solution.transforms) < 2:
            return None
        
        # Create a name for the compound
        components_str = "_then_".join(
            t.name if hasattr(t, 'name') else t.__class__.__name__
            for t in solution.transforms[:3]  # First 3 components
        )
        name = f"compound_{components_str}"
        
        # Check if we already have a similar compound
        for existing in self.compounds:
            if existing.name == name:
                # Update existing compound with new task signature
                existing.task_signatures.append(task_features)
                existing.success_count += 1
                self.save_compounds()
                return existing
        
        # Create new compound primitive
        compound = CompoundPrimitive(
            name=name,
            description=f"Learned sequence: {solution.to_string()}",
            component_sequence=solution.transforms,
            task_signatures=[task_features],
            success_count=1
        )
        
        self.compounds.append(compound)
        self.save_compounds()
        
        print(f"Learned new compound: {name}")
        return compound
    
    def get_relevant_compounds(self, 
                              task_features: Dict[str, Any],
                              top_k: int = 5) -> List[CompoundPrimitive]:
        """Get compound primitives relevant to a task."""
        
        # Score all compounds
        scored = []
        for compound in self.compounds:
            score = compound.matches_task(task_features)
            # Boost score by success count (popular compounds are useful)
            score *= (1 + np.log1p(compound.success_count) * 0.1)
            scored.append((score, compound))
        
        # Sort by score and return top k
        scored.sort(key=lambda x: -x[0])
        return [compound for _, compound in scored[:top_k] if _ > 0]
    
    def get_all_as_primitives(self) -> List[Primitive]:
        """Get all compound primitives as standard Primitives."""
        return [c.to_primitive() for c in self.compounds]
    
    def save_compounds(self):
        """Save learned compounds to disk."""
        def convert_to_native(obj):
            """Convert numpy types to native Python types."""
            import numpy as np
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_native(item) for item in obj)
            else:
                return obj
        
        data = []
        for compound in self.compounds:
            # Serialize the compound (excluding actual transform functions)
            data.append({
                'name': compound.name,
                'description': compound.description,
                'component_names': [
                    t.name if hasattr(t, 'name') else t.__class__.__name__
                    for t in compound.component_sequence
                ],
                'task_signatures': convert_to_native(compound.task_signatures),
                'success_count': int(compound.success_count)
            })
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_compounds(self):
        """Load learned compounds from disk."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            # Note: We can't fully reconstruct the transforms from JSON,
            # so this is primarily for tracking what's been learned.
            # In practice, compounds would be reconstructed from their
            # component names when the system restarts.
            
            print(f"Loaded {len(data)} learned compounds from {self.storage_path}")
            
        except Exception as e:
            print(f"Could not load compounds: {e}")
    
    def analyze_compounds(self) -> Dict[str, Any]:
        """Analyze the learned compounds for insights."""
        
        if not self.compounds:
            return {'total': 0}
        
        # Collect statistics
        stats = {
            'total': len(self.compounds),
            'total_uses': sum(c.success_count for c in self.compounds),
            'avg_components': np.mean([len(c.component_sequence) for c in self.compounds]),
            'most_successful': None,
            'most_components': None,
            'component_frequency': {}
        }
        
        # Find most successful
        most_successful = max(self.compounds, key=lambda c: c.success_count)
        stats['most_successful'] = {
            'name': most_successful.name,
            'uses': most_successful.success_count,
            'description': most_successful.description
        }
        
        # Find most complex
        most_complex = max(self.compounds, key=lambda c: len(c.component_sequence))
        stats['most_components'] = {
            'name': most_complex.name,
            'components': len(most_complex.component_sequence),
            'description': most_complex.description
        }
        
        # Count component frequencies
        for compound in self.compounds:
            for transform in compound.component_sequence:
                name = transform.name if hasattr(transform, 'name') else transform.__class__.__name__
                stats['component_frequency'][name] = stats['component_frequency'].get(name, 0) + 1
        
        return stats


def demonstrate_compound_learning():
    """Demonstrate compound primitive learning."""
    
    from program_synthesis_v2 import EnhancedProgramSynthesizer
    from arc_data_loader import load_arc_training_data
    
    print("Demonstrating Compound Primitive Learning")
    print("=" * 50)
    
    # Initialize learner
    learner = CompoundPrimitiveLearner("demo_compounds.json")
    synthesizer = EnhancedProgramSynthesizer()
    
    # Load some tasks
    tasks = load_arc_training_data(max_tasks=10)
    
    for i, task in enumerate(tasks):
        print(f"\nTask {i+1}: {task['id']}")
        
        # Get examples
        examples = [
            (np.array(ex['input']), np.array(ex['output']))
            for ex in task['train']
        ]
        
        # Extract task features
        task_features = {
            'input_shape': examples[0][0].shape,
            'output_shape': examples[0][1].shape,
            'input_colors': tuple(sorted(np.unique(examples[0][0]))),
            'output_colors': tuple(sorted(np.unique(examples[0][1])))
        }
        
        # Check for relevant compounds
        relevant = learner.get_relevant_compounds(task_features, top_k=3)
        if relevant:
            print(f"  Found {len(relevant)} relevant compounds:")
            for compound in relevant:
                print(f"    - {compound.name} (used {compound.success_count} times)")
        
        # Try synthesis
        solution = synthesizer.synthesize(examples, max_time=3.0)
        
        if solution:
            # Test accuracy
            test = task['test'][0] if task['test'] else None
            if test:
                inp = np.array(test['input'])
                out = np.array(test['output']) if 'output' in test else None
                
                if out is not None:
                    pred = solution.apply(inp)
                    if pred.shape == out.shape:
                        accuracy = np.mean(pred == out)
                        print(f"  Solution accuracy: {accuracy:.1%}")
                        
                        # Learn from the solution if successful
                        compound = learner.learn_from_solution(solution, task_features, accuracy)
                        if compound:
                            print(f"  ✓ Learned compound: {compound.name}")
    
    # Analyze what was learned
    print("\n" + "=" * 50)
    print("COMPOUND LEARNING ANALYSIS")
    print("=" * 50)
    
    stats = learner.analyze_compounds()
    print(f"\nTotal compounds learned: {stats['total']}")
    print(f"Total successful uses: {stats['total_uses']}")
    
    if stats.get('most_successful'):
        print(f"\nMost successful compound:")
        print(f"  {stats['most_successful']['name']}")
        print(f"  Used {stats['most_successful']['uses']} times")
    
    if stats.get('component_frequency'):
        print(f"\nMost common components:")
        for comp, count in sorted(stats['component_frequency'].items(), 
                                 key=lambda x: -x[1])[:5]:
            print(f"  {comp}: {count} times")


if __name__ == "__main__":
    demonstrate_compound_learning()