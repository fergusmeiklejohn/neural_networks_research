"""Imagination Engine V3 - Integrated system with invention memory.

This is the complete integration of all our components:
- Primitive Inventor: Creates new primitives on-the-fly
- Invention Memory: Stores and retrieves successful inventions
- Invention Strategies: Advanced pattern understanding
- Program Synthesis: Falls back to fixed primitives when needed
- Hypothesis Generator: Systematic exploration

Key Innovation: The system learns from experience, building a growing library
of invented primitives that can be reused and adapted.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import time
import logging

# Import all our components
from primitive_inventor import PrimitiveInventor, InventedPrimitive
from invention_strategies import InventionStrategies
from invention_memory import InventionMemory
from hypothesis_generator import MinimalHypothesisGenerator, GenerationStrategy
from program_synthesis import ProgramSynthesizer, Transform
from arc_primitives import ARCPrimitives
from improved_compositional_reasoner import ImprovedCompositionalReasoner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Solution:
    """Represents a solution to an ARC task."""
    predictions: List[np.ndarray]
    strategy_used: str
    invention_used: Optional[str] = None
    new_invention: Optional[str] = None
    accuracy: float = 0.0
    solving_time: float = 0.0
    operation_count: int = 0
    
    
class ImaginationEngineV3:
    """Integrated imagination engine with learning capabilities."""
    
    def __init__(
        self,
        memory_path: Optional[Path] = None,
        memory_capacity: int = 1000,
        enable_learning: bool = True,
        verbose: bool = True
    ):
        """Initialize the imagination engine.
        
        Args:
            memory_path: Path for persistent memory storage
            memory_capacity: Maximum number of inventions to remember
            enable_learning: Whether to learn from successful solutions
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        self.enable_learning = enable_learning
        
        # Initialize components
        self._log("Initializing Imagination Engine V3...")
        
        # Invention components
        self.primitive_inventor = PrimitiveInventor(max_program_length=20)
        self.invention_strategies = InventionStrategies()
        self.invention_memory = InventionMemory(
            capacity=memory_capacity,
            storage_path=memory_path or Path("imagination_v3_memory.json")
        )
        
        # Fallback components
        self.hypothesis_generator = MinimalHypothesisGenerator()
        self.program_synthesizer = ProgramSynthesizer()
        self.compositional_reasoner = ImprovedCompositionalReasoner()
        
        # Statistics
        self.total_tasks_solved = 0
        self.memory_hits = 0
        self.adaptations = 0
        self.new_inventions = 0
        
        self._log(f"Engine initialized with {len(self.invention_memory.inventions)} stored inventions")
    
    def solve(
        self,
        task: Dict[str, Any],
        timeout: float = 30.0
    ) -> Solution:
        """Solve an ARC task using the integrated system.
        
        Args:
            task: ARC task with 'train' and 'test' examples
            timeout: Maximum time to spend solving
            
        Returns:
            Solution object with predictions and metadata
        """
        start_time = time.time()
        
        # Extract examples
        train_examples = [(np.array(ex['input']), np.array(ex['output'])) 
                         for ex in task.get('train', [])]
        test_inputs = [np.array(ex['input']) for ex in task.get('test', [])]
        
        if not train_examples:
            return Solution(predictions=[], strategy_used="no_examples", solving_time=0)
        
        self._log(f"\nSolving task with {len(train_examples)} training examples...")
        
        # Phase 1: Check memory for similar inventions
        solution = self._try_memory_retrieval(train_examples, test_inputs, timeout)
        if solution and solution.accuracy > 0.8:
            self.memory_hits += 1
            self._log(f"✓ Solved using stored invention: {solution.invention_used}")
            return solution
        
        # Phase 2: Try to adapt retrieved inventions
        if solution and solution.accuracy > 0.5:
            adapted = self._try_adaptation(train_examples, test_inputs, solution, timeout)
            if adapted and adapted.accuracy > 0.8:
                self.adaptations += 1
                self._log(f"✓ Solved by adapting: {adapted.invention_used}")
                return adapted
        
        # Phase 3: Invent new primitive
        solution = self._try_invention(train_examples, test_inputs, timeout)
        if solution and solution.accuracy > 0.8:
            self.new_inventions += 1
            self._log(f"✓ Solved with new invention: {solution.new_invention}")
            
            # Store successful invention if learning is enabled
            if self.enable_learning and solution.new_invention:
                self._store_invention(train_examples, solution)
            
            return solution
        
        # Phase 4: Fallback to hypothesis generator
        solution = self._try_hypothesis_generator(train_examples, test_inputs, timeout)
        if solution and solution.accuracy > 0.5:
            self._log(f"✓ Solved using hypothesis generator: {solution.strategy_used}")
            return solution
        
        # Phase 5: Fallback to program synthesis
        solution = self._try_program_synthesis(train_examples, test_inputs, timeout)
        if solution:
            self._log(f"✓ Solved using program synthesis")
            return solution
        
        # Failed to solve
        self._log("✗ Failed to solve task")
        return Solution(
            predictions=[test_inputs[0] for _ in test_inputs],  # Return unchanged
            strategy_used="failed",
            solving_time=time.time() - start_time
        )
    
    def _try_memory_retrieval(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_inputs: List[np.ndarray],
        timeout: float
    ) -> Optional[Solution]:
        """Try to solve using stored inventions."""
        
        self._log("  Phase 1: Checking memory for similar inventions...")
        
        retrieved = self.invention_memory.retrieve(train_examples, k=5)
        
        if not retrieved:
            self._log("    No similar inventions found")
            return None
        
        self._log(f"    Found {len(retrieved)} candidate inventions")
        
        # Try each retrieved invention
        for invention, function in retrieved:
            try:
                # Test on training examples
                train_accuracy = self._evaluate_function(function, train_examples)
                
                if train_accuracy > 0.8:
                    # Apply to test inputs
                    predictions = [function(inp) for inp in test_inputs]
                    
                    # Update usage statistics
                    self.invention_memory.update_success(invention.invention_id, True)
                    
                    return Solution(
                        predictions=predictions,
                        strategy_used="memory_retrieval",
                        invention_used=invention.name,
                        accuracy=train_accuracy,
                        operation_count=invention.operation_count
                    )
                    
            except Exception as e:
                self._log(f"    Error applying {invention.name}: {e}")
                continue
        
        return None
    
    def _try_adaptation(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_inputs: List[np.ndarray],
        base_solution: Solution,
        timeout: float
    ) -> Optional[Solution]:
        """Try to adapt a partially successful solution."""
        
        self._log("  Phase 2: Attempting to adapt retrieved invention...")
        
        # TODO: Implement adaptation strategies
        # For now, return None to proceed to invention
        return None
    
    def _try_invention(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_inputs: List[np.ndarray],
        timeout: float
    ) -> Optional[Solution]:
        """Try to invent a new primitive."""
        
        self._log("  Phase 3: Inventing new primitive...")
        
        start_time = time.time()
        
        # Try different invention strategies
        strategies = [
            ("geometric_reasoning", self.invention_strategies.geometric_reasoning),
            ("pattern_decomposition", self.invention_strategies.pattern_decomposition),
            ("abstraction_discovery", self.invention_strategies.abstraction_discovery),
            ("trace", lambda ex: self.primitive_inventor.invent_primitive(ex, "trace")),
            ("search", lambda ex: self.primitive_inventor.invent_primitive(ex, "search"))
        ]
        
        best_solution = None
        best_accuracy = 0.0
        
        for strategy_name, strategy_func in strategies:
            if time.time() - start_time > timeout * 0.8:  # Use 80% of timeout
                break
            
            try:
                self._log(f"    Trying {strategy_name}...")
                
                # Invent primitive
                invented = strategy_func(train_examples)
                
                if invented:
                    # Evaluate on training data
                    train_accuracy = self._evaluate_function(invented.function, train_examples)
                    
                    if train_accuracy > best_accuracy:
                        # Apply to test inputs
                        predictions = [invented.function(inp) for inp in test_inputs]
                        
                        best_solution = Solution(
                            predictions=predictions,
                            strategy_used=f"invention_{strategy_name}",
                            new_invention=invented.name if hasattr(invented, 'name') else invented.program,
                            accuracy=train_accuracy,
                            solving_time=time.time() - start_time,
                            operation_count=len(invented.atomic_sequence)
                        )
                        best_accuracy = train_accuracy
                        
                        # Store metadata for later storage
                        best_solution._invented = invented
                        best_solution._strategy = strategy_name
                        
                        if train_accuracy >= 1.0:
                            self._log(f"    ✓ Perfect solution found with {strategy_name}")
                            return best_solution
                            
            except Exception as e:
                self._log(f"    Error with {strategy_name}: {e}")
                continue
        
        return best_solution
    
    def _try_hypothesis_generator(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_inputs: List[np.ndarray],
        timeout: float
    ) -> Optional[Solution]:
        """Try using the hypothesis generator."""
        
        self._log("  Phase 4: Using hypothesis generator...")
        
        try:
            # Generate hypotheses
            hypotheses = self.hypothesis_generator.generate_hypotheses(
                train_examples,
                n_hypotheses=20,
                strategy=GenerationStrategy.SYSTEMATIC
            )
            
            if hypotheses:
                best_hyp = hypotheses[0]
                predictions = [best_hyp.transform_fn(inp) for inp in test_inputs]
                
                return Solution(
                    predictions=predictions,
                    strategy_used="hypothesis_generator",
                    accuracy=best_hyp.score
                )
                
        except Exception as e:
            self._log(f"    Error with hypothesis generator: {e}")
        
        return None
    
    def _try_program_synthesis(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_inputs: List[np.ndarray],
        timeout: float
    ) -> Optional[Solution]:
        """Try using program synthesis with fixed primitives."""
        
        self._log("  Phase 5: Using program synthesis...")
        
        try:
            # Synthesize program
            program = self.program_synthesizer.synthesize(train_examples)
            
            if program:
                predictions = [program.apply(inp) for inp in test_inputs]
                
                return Solution(
                    predictions=predictions,
                    strategy_used="program_synthesis",
                    accuracy=program.score if hasattr(program, 'score') else 0.5
                )
                
        except Exception as e:
            self._log(f"    Error with program synthesis: {e}")
        
        return None
    
    def _store_invention(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        solution: Solution
    ):
        """Store a successful invention in memory."""
        
        if not hasattr(solution, '_invented'):
            return
        
        invented = solution._invented
        strategy = solution._strategy
        
        try:
            inv_id = self.invention_memory.store(
                name=solution.new_invention or "unnamed",
                program_description=invented.program if hasattr(invented, 'program') else str(invented),
                atomic_sequence=invented.atomic_sequence,
                function=invented.function,
                examples=train_examples,
                accuracy=solution.accuracy,
                invention_time=solution.solving_time,
                strategy_used=strategy,
                generalization_score=0.7  # Could be computed more sophisticatedly
            )
            
            self._log(f"    Stored invention with ID: {inv_id}")
            
        except Exception as e:
            self._log(f"    Error storing invention: {e}")
    
    def _evaluate_function(
        self,
        function: Callable,
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> float:
        """Evaluate a function on examples."""
        
        if not examples:
            return 0.0
        
        correct = 0
        for inp, expected in examples:
            try:
                predicted = function(inp)
                if np.array_equal(predicted, expected):
                    correct += 1
            except:
                pass
        
        return correct / len(examples)
    
    def _get_arc_primitives(self) -> List[Transform]:
        """Get ARC primitives for program synthesis."""
        
        primitives = []
        
        # Add some basic ARC primitives
        primitive_funcs = [
            ("transpose", ARCPrimitives.transpose),
            ("rotate_90", ARCPrimitives.rotate_90),
            ("flip_horizontal", ARCPrimitives.flip_horizontal),
            ("flip_vertical", ARCPrimitives.flip_vertical),
        ]
        
        for name, func in primitive_funcs:
            from program_synthesis import Primitive
            primitives.append(Primitive(name, func))
        
        return primitives
    
    def _log(self, message: str):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        
        memory_stats = self.invention_memory.get_statistics()
        
        return {
            "engine": {
                "total_tasks_solved": self.total_tasks_solved,
                "memory_hits": self.memory_hits,
                "adaptations": self.adaptations,
                "new_inventions": self.new_inventions,
                "memory_hit_rate": self.memory_hits / max(self.total_tasks_solved, 1),
                "invention_rate": self.new_inventions / max(self.total_tasks_solved, 1)
            },
            "memory": memory_stats
        }
    
    def save_memory(self):
        """Save the invention memory to disk."""
        self.invention_memory.save()
        self._log(f"Memory saved with {len(self.invention_memory.inventions)} inventions")
    
    def load_memory(self):
        """Load the invention memory from disk."""
        self.invention_memory.load()
        self._log(f"Memory loaded with {len(self.invention_memory.inventions)} inventions")