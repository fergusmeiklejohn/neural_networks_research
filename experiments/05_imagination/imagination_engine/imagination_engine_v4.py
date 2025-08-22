"""Imagination Engine V4 - With Meta-Learning Capabilities.

This version adds meta-learning to learn from experience:
- Learns which strategies work for which task types
- Adapts strategies based on past failures
- Extracts meta-patterns from successful inventions
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
from meta_learner import MetaLearner, TaskFeatures, StrategyOutcome
from hypothesis_generator import MinimalHypothesisGenerator, GenerationStrategy
from program_synthesis import ProgramSynthesizer
from arc_primitives import ARCPrimitives
from improved_compositional_reasoner import ImprovedCompositionalReasoner
from region_extraction_learner import RegionExtractionLearner
from invention_composer import InventionComposer

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
    meta_learning_applied: bool = False
    

class ImaginationEngineV4:
    """Imagination Engine with meta-learning capabilities."""
    
    def __init__(
        self,
        memory_path: Optional[Path] = None,
        meta_learning_path: Optional[Path] = None,
        memory_capacity: int = 1000,
        enable_learning: bool = True,
        enable_meta_learning: bool = True,
        verbose: bool = True
    ):
        """Initialize the imagination engine with meta-learning.
        
        Args:
            memory_path: Path for invention memory storage
            meta_learning_path: Path for meta-learning knowledge
            memory_capacity: Maximum number of inventions to remember
            enable_learning: Whether to learn from successful solutions
            enable_meta_learning: Whether to use meta-learning for strategy selection
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        self.enable_learning = enable_learning
        self.enable_meta_learning = enable_meta_learning
        
        # Initialize components
        self._log("Initializing Imagination Engine V4 with Meta-Learning...")
        
        # Invention components
        self.primitive_inventor = PrimitiveInventor(max_program_length=20)
        self.invention_strategies = InventionStrategies()
        self.invention_memory = InventionMemory(
            capacity=memory_capacity,
            storage_path=memory_path or Path("imagination_v4_memory.json")
        )
        
        # Meta-learning component
        self.meta_learner = MetaLearner(
            storage_path=meta_learning_path or Path("meta_learning_knowledge.json")
        )
        
        # Fallback components
        self.hypothesis_generator = MinimalHypothesisGenerator()
        self.program_synthesizer = ProgramSynthesizer()
        self.compositional_reasoner = ImprovedCompositionalReasoner()
        
        # New components
        self.region_extractor = RegionExtractionLearner()
        self.invention_composer = InventionComposer()
        
        # Statistics
        self.total_tasks_solved = 0
        self.memory_hits = 0
        self.adaptations = 0
        self.new_inventions = 0
        self.meta_learning_successes = 0
        
        self._log(f"Engine initialized:")
        self._log(f"  Inventions in memory: {len(self.invention_memory.inventions)}")
        self._log(f"  Strategies learned: {len(self.meta_learner.strategies)}")
    
    def solve(
        self,
        task: Dict[str, Any],
        timeout: float = 30.0
    ) -> Solution:
        """Solve an ARC task using meta-learning for strategy selection.
        
        Args:
            task: ARC task with 'train' and 'test' examples
            timeout: Maximum time to spend solving
            
        Returns:
            Solution object with predictions and metadata
        """
        start_time = time.time()
        task_id = task.get('id', f'task_{self.total_tasks_solved}')
        
        # Extract examples
        train_examples = [(np.array(ex['input']), np.array(ex['output'])) 
                         for ex in task.get('train', [])]
        test_inputs = [np.array(ex['input']) for ex in task.get('test', [])]
        
        if not train_examples:
            return Solution(predictions=[], strategy_used="no_examples", solving_time=0)
        
        self._log(f"\nSolving task {task_id} with {len(train_examples)} training examples...")
        
        # Extract task features for meta-learning
        task_features = self.meta_learner.extract_task_features(train_examples)
        
        # Phase 0: Meta-learning strategy selection
        if self.enable_meta_learning:
            solution = self._try_with_meta_learning(
                task_id, train_examples, test_inputs, task_features, timeout
            )
            if solution and solution.accuracy > 0.8:
                self.meta_learning_successes += 1
                return solution
        
        # Phase 1: Check memory for similar inventions
        solution = self._try_memory_retrieval(train_examples, test_inputs, timeout)
        if solution and solution.accuracy > 0.8:
            self.memory_hits += 1
            self._record_outcome(task_id, task_features, solution)
            return solution
        
        # Phase 2: Try to adapt retrieved inventions
        if solution and solution.accuracy > 0.5:
            adapted = self._try_adaptation(train_examples, test_inputs, solution, timeout)
            if adapted and adapted.accuracy > 0.8:
                self.adaptations += 1
                self._record_outcome(task_id, task_features, adapted)
                return adapted
        
        # Phase 3: Invent new primitive (with all strategies)
        solution = self._try_all_invention_strategies(
            task_id, train_examples, test_inputs, task_features, timeout
        )
        if solution and solution.accuracy > 0.8:
            self.new_inventions += 1
            
            # Store successful invention
            if self.enable_learning and solution.new_invention:
                self._store_invention(train_examples, solution)
            
            self._record_outcome(task_id, task_features, solution)
            return solution
        
        # Phase 4: Fallback strategies
        solution = self._try_fallback_strategies(train_examples, test_inputs, timeout)
        
        # Learn from failure if all strategies failed
        if not solution or solution.accuracy < 0.5:
            if self.enable_meta_learning:
                attempted = ['invention', 'hypothesis', 'synthesis']
                errors = ['No successful strategy found']
                insights = self.meta_learner.learn_from_failure(
                    task_id, task_features, attempted, errors
                )
                self._log(f"  Learned from failure: {insights['suggested_improvements']}")
        
        # Record outcome for meta-learning
        if solution:
            self._record_outcome(task_id, task_features, solution)
        
        return solution or Solution(
            predictions=[test_inputs[0] for _ in test_inputs],
            strategy_used="failed",
            solving_time=time.time() - start_time
        )
    
    def _try_with_meta_learning(
        self,
        task_id: str,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_inputs: List[np.ndarray],
        task_features: TaskFeatures,
        timeout: float
    ) -> Optional[Solution]:
        """Try solving using meta-learned strategy selection."""
        
        self._log("  Phase 0: Meta-learning strategy selection...")
        
        # Get strategy predictions
        available_strategies = [
            "geometric_reasoning",
            "pattern_decomposition", 
            "abstraction_discovery",
            "multi_object_coordination",
            "conditional_transformation",
            "recursive_patterns",
            "boundary_operations",
            "symmetry_operations",
            "counting_arithmetic",
            "pattern_completion",
            "grid_subdivision",
            "color_mapping",
            "trace",
            "search",
            "region_extraction"
        ]
        
        predictions = self.meta_learner.predict_best_strategy(task_features, available_strategies)
        
        if not predictions:
            return None
        
        # Try top predicted strategies
        for strategy_name, confidence in predictions[:3]:  # Try top 3
            if confidence < self.meta_learner.min_confidence:
                continue
            
            self._log(f"    Trying {strategy_name} (confidence: {confidence:.2f})")
            
            # Get adaptations for this strategy
            adaptations = self.meta_learner.adapt_strategy(strategy_name, {'task_features': task_features})
            
            # Apply strategy with adaptations
            solution = self._apply_strategy_with_adaptations(
                strategy_name, train_examples, test_inputs, adaptations, timeout
            )
            
            if solution and solution.accuracy > 0.8:
                solution.meta_learning_applied = True
                self._log(f"    ✓ Meta-learning success with {strategy_name}")
                return solution
        
        return None
    
    def _apply_strategy_with_adaptations(
        self,
        strategy_name: str,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_inputs: List[np.ndarray],
        adaptations: Dict[str, Any],
        timeout: float
    ) -> Optional[Solution]:
        """Apply a specific strategy with adaptations."""
        
        start_time = time.time()
        
        # Apply timeout adaptation if suggested
        for mod in adaptations.get('modifications', []):
            if mod['type'] == 'increase_timeout':
                timeout = mod['value']
        
        try:
            # Map strategy name to function
            if strategy_name == "geometric_reasoning":
                invented = self.invention_strategies.geometric_reasoning(train_examples)
            elif strategy_name == "pattern_decomposition":
                invented = self.invention_strategies.pattern_decomposition(train_examples)
            elif strategy_name == "abstraction_discovery":
                invented = self.invention_strategies.abstraction_discovery(train_examples)
            elif strategy_name == "trace":
                invented = self.primitive_inventor.invent_primitive(train_examples, "trace")
            elif strategy_name == "search":
                invented = self.primitive_inventor.invent_primitive(train_examples, "search")
            elif strategy_name == "region_extraction":
                invented = self._apply_region_extraction(train_examples)
            elif strategy_name == "multi_object_coordination":
                invented = self.invention_strategies.multi_object_coordination(train_examples)
            elif strategy_name == "conditional_transformation":
                invented = self.invention_strategies.conditional_transformation(train_examples)
            elif strategy_name == "recursive_patterns":
                invented = self.invention_strategies.recursive_patterns(train_examples)
            elif strategy_name == "boundary_operations":
                invented = self.invention_strategies.boundary_operations(train_examples)
            else:
                return None
            
            if invented:
                # Evaluate
                train_accuracy = self._evaluate_function(invented.function, train_examples)
                
                if train_accuracy > 0.8:
                    predictions = [invented.function(inp) for inp in test_inputs]
                    
                    return Solution(
                        predictions=predictions,
                        strategy_used=f"meta_{strategy_name}",
                        new_invention=invented.name if hasattr(invented, 'name') else invented.program,
                        accuracy=train_accuracy,
                        solving_time=time.time() - start_time,
                        operation_count=len(invented.atomic_sequence),
                        meta_learning_applied=True
                    )
                    
        except Exception as e:
            self._log(f"    Error with {strategy_name}: {e}")
        
        return None
    
    def _try_all_invention_strategies(
        self,
        task_id: str,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_inputs: List[np.ndarray],
        task_features: TaskFeatures,
        timeout: float
    ) -> Optional[Solution]:
        """Try all invention strategies, learning from outcomes."""
        
        self._log("  Phase 3: Trying all invention strategies...")
        
        start_time = time.time()
        
        strategies = [
            ("geometric_reasoning", self.invention_strategies.geometric_reasoning),
            ("pattern_decomposition", self.invention_strategies.pattern_decomposition),
            ("abstraction_discovery", self.invention_strategies.abstraction_discovery),
            ("multi_object_coordination", self.invention_strategies.multi_object_coordination),
            ("conditional_transformation", self.invention_strategies.conditional_transformation),
            ("recursive_patterns", self.invention_strategies.recursive_patterns),
            ("boundary_operations", self.invention_strategies.boundary_operations),
            ("symmetry_operations", self.invention_strategies.symmetry_operations),
            ("counting_arithmetic", self.invention_strategies.counting_arithmetic),
            ("pattern_completion", self.invention_strategies.pattern_completion),
            ("grid_subdivision", self.invention_strategies.grid_subdivision),
            ("color_mapping", self.invention_strategies.color_mapping),
            ("trace", lambda ex: self.primitive_inventor.invent_primitive(ex, "trace")),
            ("search", lambda ex: self.primitive_inventor.invent_primitive(ex, "search"))
        ]
        
        best_solution = None
        best_accuracy = 0.0
        partial_solutions = []  # Collect partial solutions for composition
        
        for strategy_name, strategy_func in strategies:
            if time.time() - start_time > timeout * 0.8:
                break
            
            strategy_start = time.time()
            
            try:
                self._log(f"    Trying {strategy_name}...")
                
                invented = strategy_func(train_examples)
                
                if invented:
                    train_accuracy = self._evaluate_function(invented.function, train_examples)
                    strategy_time = time.time() - strategy_start
                    
                    # Enhanced debug logging
                    self._log(f"      Strategy {strategy_name}: accuracy={train_accuracy:.3f}, time={strategy_time:.2f}s")
                    if hasattr(invented, 'program'):
                        self._log(f"      Created program: {invented.program[:100]}...")
                    
                    # Record outcome for meta-learning
                    outcome = StrategyOutcome(
                        task_id=task_id,
                        task_features=task_features,
                        strategy_name=strategy_name,
                        success=train_accuracy > 0.8,
                        accuracy=train_accuracy,
                        time_taken=strategy_time,
                        invention_created=invented.name if hasattr(invented, 'name') else invented.program
                    )
                    self.meta_learner.learn_from_outcome(outcome)
                    
                    # Collect partial solutions for composition
                    if train_accuracy > 0.3:  # Collect moderately successful attempts
                        partial_solutions.append({
                            'invented': invented,
                            'accuracy': train_accuracy,
                            'strategy_name': strategy_name
                        })
                        self._log(f"      Collected partial solution: {strategy_name} ({train_accuracy:.3f})")
                    
                    if train_accuracy > best_accuracy:
                        predictions = [invented.function(inp) for inp in test_inputs]
                        
                        best_solution = Solution(
                            predictions=predictions,
                            strategy_used=f"invention_{strategy_name}",
                            new_invention=invented.name if hasattr(invented, 'name') else invented.program,
                            accuracy=train_accuracy,
                            solving_time=time.time() - start_time,
                            operation_count=len(invented.atomic_sequence)
                        )
                        best_solution._invented = invented
                        best_solution._strategy = strategy_name
                        best_accuracy = train_accuracy
                        
                        if train_accuracy >= 1.0:
                            self._log(f"    ✓ Perfect solution with {strategy_name}")
                            return best_solution
                else:
                    strategy_time = time.time() - strategy_start
                    self._log(f"      Strategy {strategy_name}: no invention created (time={strategy_time:.2f}s)")
                    
                    # Record failure for meta-learning
                    outcome = StrategyOutcome(
                        task_id=task_id,
                        task_features=task_features,
                        strategy_name=strategy_name,
                        success=False,
                        accuracy=0.0,
                        time_taken=strategy_time,
                        error_type="no_invention"
                    )
                    self.meta_learner.learn_from_outcome(outcome)
                    
            except Exception as e:
                self._log(f"    Error with {strategy_name}: {e}")
                
                # Record error for meta-learning
                outcome = StrategyOutcome(
                    task_id=task_id,
                    task_features=task_features,
                    strategy_name=strategy_name,
                    success=False,
                    accuracy=0.0,
                    time_taken=time.time() - strategy_start,
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
                self.meta_learner.learn_from_outcome(outcome)
        
        # Try composition if we have multiple partial solutions but no perfect solution
        if partial_solutions and (not best_solution or best_solution.accuracy < 0.8):
            self._log(f"  Trying composition with {len(partial_solutions)} partial solutions...")
            composition_result = self._try_composition(partial_solutions, train_examples, test_inputs)
            if composition_result and (not best_solution or composition_result.accuracy > best_solution.accuracy):
                self._log(f"    ✓ Composition improved accuracy to {composition_result.accuracy:.3f}")
                return composition_result
        
        return best_solution
    
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
        
        for invention, function in retrieved:
            try:
                train_accuracy = self._evaluate_function(function, train_examples)
                
                if train_accuracy > 0.8:
                    predictions = [function(inp) for inp in test_inputs]
                    
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
        return None
    
    def _try_fallback_strategies(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_inputs: List[np.ndarray],
        timeout: float
    ) -> Optional[Solution]:
        """Try fallback strategies (hypothesis generator, program synthesis)."""
        
        # Try hypothesis generator
        self._log("  Phase 4: Using hypothesis generator...")
        
        try:
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
        
        # Try program synthesis
        self._log("  Phase 5: Using program synthesis...")
        
        try:
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
                generalization_score=0.7
            )
            
            self._log(f"    Stored invention with ID: {inv_id}")
            
        except Exception as e:
            self._log(f"    Error storing invention: {e}")
    
    def _record_outcome(
        self,
        task_id: str,
        task_features: TaskFeatures,
        solution: Solution
    ):
        """Record task outcome for meta-learning."""
        
        if not self.enable_meta_learning:
            return
        
        outcome = StrategyOutcome(
            task_id=task_id,
            task_features=task_features,
            strategy_name=solution.strategy_used,
            success=solution.accuracy > 0.8,
            accuracy=solution.accuracy,
            time_taken=solution.solving_time,
            invention_created=solution.new_invention
        )
        
        self.meta_learner.learn_from_outcome(outcome)
    
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
    
    def _apply_region_extraction(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[InventedPrimitive]:
        """Apply region extraction strategy."""
        
        if not train_examples:
            return None
        
        # Try to learn extraction rules from examples
        # For each example, try to identify regions that transform
        for inp, out in train_examples:
            # Auto-detect markers
            markers = self.region_extractor._detect_markers(inp)
            
            # Try to extract region
            region = self.region_extractor.extract_marked_region(inp, markers)
            
            if region is not None:
                # Try to find transformation for the region
                # This is simplified - in practice would be more sophisticated
                def region_transform(grid):
                    result = grid.copy()
                    # Apply some transformation to the extracted region
                    extracted = self.region_extractor.extract_marked_region(grid)
                    if extracted is not None:
                        # Simple example: fill the region
                        h, w = extracted.shape
                        for r in range(h):
                            for c in range(w):
                                if extracted[r, c] != 0:
                                    result[r, c] = extracted[r, c] + 1
                    return result
                
                return InventedPrimitive(
                    name="region_extraction",
                    program="Extract and transform marked regions",
                    function=region_transform,
                    atomic_sequence=["detect_markers", "extract_region", "transform"],
                    score=0.5,
                    invention_time=0.0
                )
        
        return None
    
    def _try_composition(
        self,
        partial_solutions: List[Any],
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_inputs: List[np.ndarray]
    ) -> Optional[Solution]:
        """Try composing partial solutions into a complete solution."""
        
        if not partial_solutions or len(partial_solutions) < 2:
            return None
        
        # Try different composition strategies
        best_composition = self.invention_composer.suggest_composition(
            partial_solutions,
            train_examples
        )
        
        if best_composition and best_composition.score > 0.5:
            # Test on test inputs
            predictions = []
            for inp in test_inputs:
                try:
                    pred = best_composition.function(inp)
                    predictions.append(pred)
                except:
                    predictions.append(inp)  # Fallback to input
            
            return Solution(
                predictions=predictions,
                strategy_used="composition",
                accuracy=best_composition.score,
                solving_time=0.0
            )
        
        return None
    
    def _log(self, message: str):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics including meta-learning insights."""
        
        memory_stats = self.invention_memory.get_statistics()
        meta_stats = self.meta_learner.get_learning_summary()
        
        return {
            "engine": {
                "total_tasks_solved": self.total_tasks_solved,
                "memory_hits": self.memory_hits,
                "adaptations": self.adaptations,
                "new_inventions": self.new_inventions,
                "meta_learning_successes": self.meta_learning_successes,
                "memory_hit_rate": self.memory_hits / max(self.total_tasks_solved, 1),
                "invention_rate": self.new_inventions / max(self.total_tasks_solved, 1),
                "meta_learning_rate": self.meta_learning_successes / max(self.total_tasks_solved, 1)
            },
            "memory": memory_stats,
            "meta_learning": meta_stats
        }
    
    def save_all(self):
        """Save both invention memory and meta-learning knowledge."""
        self.invention_memory.save()
        self.meta_learner.save()
        self._log("Saved invention memory and meta-learning knowledge")
    
    def load_all(self):
        """Load both invention memory and meta-learning knowledge."""
        self.invention_memory.load()
        self.meta_learner.load()
        self._log("Loaded invention memory and meta-learning knowledge")