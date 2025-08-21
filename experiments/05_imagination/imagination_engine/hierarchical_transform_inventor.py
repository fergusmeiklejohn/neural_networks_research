"""Hierarchical Transform Inventor (HTI) - HRM-inspired architecture for learning transforms.

This module implements a hierarchical reasoning system that learns to invent new transforms
through a two-module architecture inspired by the Hierarchical Reasoning Model (HRM).

Key components:
1. High-level planner: Abstract reasoning about transform strategies
2. Low-level executor: Concrete pixel-level transformations
3. Adaptive computation: Dynamic reasoning depth based on task complexity
4. Transform memory: Learned library of discovered transforms
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TransformConcept:
    """Abstract representation of a transform concept."""
    
    name: str
    embedding: np.ndarray
    properties: Dict[str, Any]
    
    def similarity(self, other: 'TransformConcept') -> float:
        """Compute similarity between concepts."""
        return float(np.dot(self.embedding, other.embedding) / 
                    (np.linalg.norm(self.embedding) * np.linalg.norm(other.embedding)))


@dataclass
class ExecutionState:
    """State of the low-level executor."""
    
    grid_state: np.ndarray
    step_count: int
    accumulated_transform: np.ndarray
    confidence: float


@dataclass
class PlannerState:
    """State of the high-level planner."""
    
    abstract_plan: List[TransformConcept]
    reasoning_depth: int
    task_encoding: np.ndarray
    confidence: float


class HighLevelPlanner:
    """High-level module for abstract transform planning.
    
    Operates at a slow timescale, generating abstract strategies for transformation.
    """
    
    def __init__(self, hidden_dim: int = 512, max_steps: int = 8, n_concepts: int = 64):
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        self.n_concepts = n_concepts
        
        # Initialize concept library with basic transforms
        self.concept_library = self._initialize_concepts()
        
        # Learnable parameters (simplified - in practice would be neural network)
        self.concept_embeddings = np.random.randn(n_concepts, hidden_dim) * 0.1
        self.reasoning_weights = np.random.randn(hidden_dim, hidden_dim) * 0.01
        
        # State tracking
        self.current_state = None
        self.reasoning_trace = []
    
    def _initialize_concepts(self) -> List[TransformConcept]:
        """Initialize basic transform concepts."""
        concepts = []
        
        # Basic geometric transforms
        basic_transforms = [
            ("translate", {"type": "spatial", "complexity": 1}),
            ("rotate", {"type": "spatial", "complexity": 2}),
            ("scale", {"type": "spatial", "complexity": 2}),
            ("shear", {"type": "spatial", "complexity": 3}),
            ("flip", {"type": "spatial", "complexity": 1}),
        ]
        
        # Abstract operations
        abstract_ops = [
            ("compose", {"type": "meta", "complexity": 4}),
            ("iterate", {"type": "meta", "complexity": 3}),
            ("conditional", {"type": "meta", "complexity": 5}),
            ("inverse", {"type": "meta", "complexity": 2}),
        ]
        
        # Pattern-based concepts
        patterns = [
            ("periodic", {"type": "pattern", "complexity": 3}),
            ("symmetric", {"type": "pattern", "complexity": 2}),
            ("progressive", {"type": "pattern", "complexity": 3}),
            ("recursive", {"type": "pattern", "complexity": 4}),
        ]
        
        all_concepts = basic_transforms + abstract_ops + patterns
        
        for i, (name, props) in enumerate(all_concepts):
            embedding = np.random.randn(self.hidden_dim) * 0.1
            embedding[i % self.hidden_dim] = 1.0  # Unique signature
            concepts.append(TransformConcept(name, embedding, props))
        
        return concepts
    
    def initialize(self, task_encoding: np.ndarray) -> PlannerState:
        """Initialize planner state for a new task."""
        self.current_state = PlannerState(
            abstract_plan=[],
            reasoning_depth=0,
            task_encoding=task_encoding,
            confidence=0.0
        )
        self.reasoning_trace = []
        return self.current_state
    
    def reason_step(self, state: PlannerState) -> List[TransformConcept]:
        """Perform one step of abstract reasoning."""
        logger.debug(f"Planner reasoning step {state.reasoning_depth}")
        
        # Compute attention over concept library based on task
        task_concept_scores = []
        for concept in self.concept_library:
            score = np.dot(state.task_encoding, concept.embedding)
            task_concept_scores.append((score, concept))
        
        # Sort by relevance
        task_concept_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Select top concepts for this reasoning step
        n_select = min(3, len(task_concept_scores))
        selected_concepts = [c for _, c in task_concept_scores[:n_select]]
        
        # Check for composition opportunities
        if state.reasoning_depth > 0 and len(state.abstract_plan) > 0:
            # Consider composing with previous concepts
            compose_concept = next((c for c in self.concept_library if c.name == "compose"), None)
            if compose_concept and np.random.random() > 0.5:
                selected_concepts.append(compose_concept)
        
        # Update state
        state.abstract_plan.extend(selected_concepts)
        state.reasoning_depth += 1
        state.confidence = min(0.95, state.confidence + 0.2)
        
        # Record in trace
        self.reasoning_trace.append({
            'depth': state.reasoning_depth,
            'concepts': [c.name for c in selected_concepts],
            'confidence': state.confidence
        })
        
        return selected_concepts
    
    def update(self, state: PlannerState, execution_summary: Dict) -> PlannerState:
        """Update planner state based on execution results."""
        # Adjust confidence based on execution success
        execution_score = execution_summary.get('score', 0.0)
        state.confidence = 0.9 * state.confidence + 0.1 * execution_score
        
        # If execution failed, try alternative concepts
        if execution_score < 0.3:
            # Remove last concept and try different approach
            if len(state.abstract_plan) > 0:
                state.abstract_plan.pop()
                logger.info("Planner backtracking due to low execution score")
        
        return state


class LowLevelExecutor:
    """Low-level module for concrete transform execution.
    
    Operates at a fast timescale, implementing the abstract plans from the planner.
    """
    
    def __init__(self, hidden_dim: int = 256, exec_steps: int = 32, n_primitives: int = 128):
        self.hidden_dim = hidden_dim
        self.exec_steps = exec_steps
        self.n_primitives = n_primitives
        
        # Primitive transform operations
        self.primitives = self._initialize_primitives()
        
        # Learnable parameters
        self.primitive_weights = np.random.randn(n_primitives, hidden_dim) * 0.01
        self.execution_weights = np.random.randn(hidden_dim, hidden_dim) * 0.01
        
        # Execution state
        self.current_state = None
        self.execution_trace = []
        self.steps_since_reset = 0
    
    def _initialize_primitives(self) -> Dict[str, Callable]:
        """Initialize primitive transform operations."""
        primitives = {}
        
        # Pixel-level operations
        primitives['shift_right'] = lambda g: np.roll(g, 1, axis=1)
        primitives['shift_left'] = lambda g: np.roll(g, -1, axis=1)
        primitives['shift_up'] = lambda g: np.roll(g, -1, axis=0)
        primitives['shift_down'] = lambda g: np.roll(g, 1, axis=0)
        
        # Diagonal shifts (useful for shear)
        primitives['shift_diag_ur'] = lambda g: np.roll(np.roll(g, -1, axis=0), 1, axis=1)
        primitives['shift_diag_dr'] = lambda g: np.roll(np.roll(g, 1, axis=0), 1, axis=1)
        
        # Row/column operations
        for i in range(5):
            primitives[f'shift_row_{i}'] = lambda g, row=i: self._shift_specific_row(g, row, 1)
            primitives[f'shift_col_{i}'] = lambda g, col=i: self._shift_specific_col(g, col, 1)
        
        # Conditional operations
        primitives['shift_if_nonzero'] = lambda g: self._conditional_shift(g)
        
        # Scaling operations
        primitives['expand_2x'] = lambda g: np.repeat(np.repeat(g, 2, axis=0), 2, axis=1)
        primitives['contract_2x'] = lambda g: g[::2, ::2]
        
        # Rotation primitives
        primitives['rotate_90'] = lambda g: np.rot90(g)
        primitives['rotate_180'] = lambda g: np.rot90(g, 2)
        primitives['rotate_270'] = lambda g: np.rot90(g, 3)
        
        # Flip primitives
        primitives['flip_h'] = lambda g: np.fliplr(g)
        primitives['flip_v'] = lambda g: np.flipud(g)
        
        # Identity (no-op)
        primitives['identity'] = lambda g: g
        
        return primitives
    
    def _shift_specific_row(self, grid: np.ndarray, row: int, amount: int) -> np.ndarray:
        """Shift a specific row."""
        result = grid.copy()
        if 0 <= row < grid.shape[0]:
            result[row] = np.roll(result[row], amount)
        return result
    
    def _shift_specific_col(self, grid: np.ndarray, col: int, amount: int) -> np.ndarray:
        """Shift a specific column."""
        result = grid.copy()
        if 0 <= col < grid.shape[1]:
            result[:, col] = np.roll(result[:, col], amount)
        return result
    
    def _conditional_shift(self, grid: np.ndarray) -> np.ndarray:
        """Shift only non-zero elements."""
        result = grid.copy()
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] != 0:
                    # Shift right if non-zero
                    new_j = (j + 1) % grid.shape[1]
                    result[i, j] = 0
                    result[i, new_j] = grid[i, j]
        return result
    
    def reset(self, initial_grid: np.ndarray) -> ExecutionState:
        """Reset executor for new execution phase."""
        self.current_state = ExecutionState(
            grid_state=initial_grid.copy(),
            step_count=0,
            accumulated_transform=initial_grid.copy(),
            confidence=0.0
        )
        self.execution_trace = []
        self.steps_since_reset = 0
        return self.current_state
    
    def execute(self, concepts: List[TransformConcept], state: ExecutionState) -> np.ndarray:
        """Execute concrete transforms based on abstract concepts."""
        logger.debug(f"Executor step {state.step_count}")
        
        # Map abstract concepts to primitive operations
        selected_primitives = self._concept_to_primitives(concepts)
        
        # Apply primitives sequentially
        current_grid = state.grid_state.copy()
        for primitive_name in selected_primitives:
            if primitive_name in self.primitives:
                try:
                    current_grid = self.primitives[primitive_name](current_grid)
                    logger.debug(f"Applied primitive: {primitive_name}")
                except Exception as e:
                    logger.warning(f"Failed to apply {primitive_name}: {e}")
        
        # Update state
        state.grid_state = current_grid
        state.step_count += 1
        state.accumulated_transform = current_grid
        state.confidence = min(0.95, state.confidence + 0.1)
        
        # Record execution
        self.execution_trace.append({
            'step': state.step_count,
            'primitives': selected_primitives,
            'confidence': state.confidence
        })
        
        self.steps_since_reset += 1
        
        return current_grid
    
    def _concept_to_primitives(self, concepts: List[TransformConcept]) -> List[str]:
        """Map abstract concepts to concrete primitives."""
        primitives = []
        
        for concept in concepts:
            if concept.name == "translate":
                # Random translation
                primitives.extend(['shift_right', 'shift_up'])
            elif concept.name == "rotate":
                primitives.append('rotate_90')
            elif concept.name == "scale":
                primitives.append('expand_2x')
            elif concept.name == "shear":
                # Shear is row-dependent shift
                for i in range(5):
                    primitives.append(f'shift_row_{i}')
            elif concept.name == "flip":
                primitives.append('flip_h')
            elif concept.name == "compose":
                # Composition means applying multiple
                primitives.extend(['rotate_90', 'shift_right'])
            elif concept.name == "periodic":
                # Periodic pattern
                primitives.extend(['shift_right', 'shift_left'] * 2)
        
        # Limit number of primitives per execution
        return primitives[:5]
    
    def get_summary(self) -> Dict:
        """Get execution summary for planner update."""
        if not self.execution_trace:
            return {'score': 0.0, 'steps': 0}
        
        # Simple scoring based on execution success
        # In practice, this would evaluate against task objectives
        avg_confidence = np.mean([t['confidence'] for t in self.execution_trace])
        
        return {
            'score': avg_confidence,
            'steps': self.steps_since_reset,
            'primitives_used': len(set(p for t in self.execution_trace for p in t['primitives']))
        }
    
    def update(self, score: float):
        """Update executor based on performance feedback."""
        # In full implementation, would update primitive weights
        logger.debug(f"Executor received score: {score}")


class HierarchicalTransformInventor:
    """Main HTI system combining planner and executor with adaptive computation."""
    
    def __init__(self):
        # Core modules
        self.planner = HighLevelPlanner(hidden_dim=512, max_steps=8, n_concepts=64)
        self.executor = LowLevelExecutor(hidden_dim=256, exec_steps=32, n_primitives=128)
        
        # Adaptive computation parameters
        self.max_reasoning_cycles = 16
        self.confidence_threshold = 0.8
        
        # Task encoding
        self.encoding_dim = 512
        
        logger.info("HTI initialized with hierarchical architecture")
    
    def encode_task(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """Encode task examples into vector representation."""
        # Simple encoding: statistics of input/output differences
        features = []
        
        for inp, out in examples[:3]:  # Use first 3 examples
            # Shape features
            features.append(inp.shape[0] / 10.0)
            features.append(inp.shape[1] / 10.0)
            
            # Count features
            features.append(np.count_nonzero(inp) / inp.size)
            features.append(np.count_nonzero(out) / out.size)
            
            # Difference features
            if inp.shape == out.shape:
                diff = out - inp
                features.append(np.mean(diff))
                features.append(np.std(diff))
            else:
                features.extend([0, 0])
            
            # Position features
            if np.any(inp):
                y_coords, x_coords = np.where(inp != 0)
                features.append(np.mean(y_coords) / inp.shape[0])
                features.append(np.mean(x_coords) / inp.shape[1])
            else:
                features.extend([0, 0])
        
        # Pad or truncate to encoding dimension
        feature_array = np.array(features)
        if len(feature_array) < self.encoding_dim:
            feature_array = np.pad(feature_array, (0, self.encoding_dim - len(feature_array)))
        else:
            feature_array = feature_array[:self.encoding_dim]
        
        return feature_array
    
    def invent_transform(
        self, 
        examples: List[Tuple[np.ndarray, np.ndarray]],
        max_cycles: Optional[int] = None
    ) -> Tuple[Callable, Dict]:
        """Invent a transform through hierarchical reasoning."""
        logger.info("Starting transform invention process")
        
        # Encode the task
        task_encoding = self.encode_task(examples)
        
        # Initialize planner
        planner_state = self.planner.initialize(task_encoding)
        
        # Initialize tracking
        max_cycles = max_cycles or self.max_reasoning_cycles
        best_transform = None
        best_score = 0.0
        reasoning_trace = []
        
        # Reasoning cycles
        for cycle in range(max_cycles):
            logger.info(f"Reasoning cycle {cycle + 1}/{max_cycles}")
            
            # High-level planning
            concepts = self.planner.reason_step(planner_state)
            
            # Low-level execution phase
            test_inp = examples[0][0]
            executor_state = self.executor.reset(test_inp)
            
            # Multiple execution steps per planning cycle
            for exec_step in range(self.executor.exec_steps // 4):
                transformed = self.executor.execute(concepts, executor_state)
                
                # Evaluate transform
                score = self.evaluate_transform(transformed, examples[0][1])
                
                if score > best_score:
                    best_score = score
                    best_transform = self.create_transform_function(
                        self.executor.execution_trace
                    )
                    logger.info(f"New best score: {best_score:.2%}")
                
                # Update executor
                self.executor.update(score)
                
                # Check if we should halt this execution phase
                if score > 0.95 or exec_step > 10:
                    break
            
            # Update planner with execution results
            execution_summary = self.executor.get_summary()
            planner_state = self.planner.update(planner_state, execution_summary)
            
            # Record reasoning cycle
            reasoning_trace.append({
                'cycle': cycle,
                'concepts': [c.name for c in concepts],
                'best_score': best_score,
                'execution_steps': self.executor.steps_since_reset
            })
            
            # Check stopping condition
            if planner_state.confidence > self.confidence_threshold and best_score > 0.9:
                logger.info(f"Stopping early - confidence: {planner_state.confidence:.2%}")
                break
        
        # Create final transform
        if best_transform is None:
            # Fallback to identity
            best_transform = lambda x: x
        
        return best_transform, {
            'score': best_score,
            'cycles': cycle + 1,
            'reasoning_trace': reasoning_trace,
            'planner_trace': self.planner.reasoning_trace,
            'executor_trace': self.executor.execution_trace
        }
    
    def evaluate_transform(self, output: np.ndarray, expected: np.ndarray) -> float:
        """Evaluate how well a transform matches expected output."""
        if output.shape != expected.shape:
            return 0.0
        
        # Exact match bonus
        if np.array_equal(output, expected):
            return 1.0
        
        # Partial credit for close matches
        correct_pixels = np.sum(output == expected)
        total_pixels = output.size
        
        return correct_pixels / total_pixels
    
    def create_transform_function(self, execution_trace: List[Dict]) -> Callable:
        """Create a reusable transform function from execution trace."""
        # Extract primitive sequence from trace
        primitive_sequence = []
        for trace_item in execution_trace:
            primitive_sequence.extend(trace_item['primitives'])
        
        def transform(grid: np.ndarray) -> np.ndarray:
            """Apply learned transform sequence."""
            result = grid.copy()
            for primitive_name in primitive_sequence:
                if primitive_name in self.executor.primitives:
                    try:
                        result = self.executor.primitives[primitive_name](result)
                    except:
                        pass
            return result
        
        return transform
    
    def learn_from_success(self, transform: Callable, task_encoding: np.ndarray, score: float):
        """Update HTI parameters based on successful transform discovery."""
        # In full implementation, would update planner and executor weights
        logger.info(f"Learning from success with score {score:.2%}")
        
        # Could store in transform memory here
        # self.memory.add(transform, task_encoding, score)


def test_hti_on_shear():
    """Test HTI on the shear transformation task."""
    print("\n" + "=" * 60)
    print("TESTING HTI ON SHEAR TRANSFORMATION")
    print("=" * 60)
    
    # Create HTI
    hti = HierarchicalTransformInventor()
    
    # Create shear examples
    examples = []
    for i in range(3):
        # Input: vertical line
        inp = np.zeros((5, 5))
        inp[:, 2] = 1
        
        # Output: sheared line
        out = np.zeros((5, 5))
        for row in range(5):
            col = 2 + row  # Shear right
            if col < 5:
                out[row, col] = 1
        
        examples.append((inp, out))
    
    print("\nExample transformation (shear):")
    print("Input:")
    print(examples[0][0])
    print("\nExpected output:")
    print(examples[0][1])
    
    # Invent transform
    transform, info = hti.invent_transform(examples, max_cycles=5)
    
    # Test the transform
    test_input = examples[0][0]
    predicted = transform(test_input)
    
    print("\nPredicted output:")
    print(predicted)
    
    print(f"\nInvention summary:")
    print(f"  Score: {info['score']:.2%}")
    print(f"  Reasoning cycles: {info['cycles']}")
    print(f"  Concepts explored: {set(c for t in info['reasoning_trace'] for c in t['concepts'])}")
    
    # Check if it discovered shear
    if np.array_equal(predicted, examples[0][1]):
        print("\n✅ HTI successfully discovered shear transformation!")
        return True
    else:
        print("\n❌ HTI did not fully discover shear (but may have found partial solution)")
        return False


if __name__ == "__main__":
    # Test the HTI
    success = test_hti_on_shear()
    
    if success:
        print("\n🎉 Hierarchical Transform Inventor is working!")
    else:
        print("\n📝 HTI needs further training/tuning")