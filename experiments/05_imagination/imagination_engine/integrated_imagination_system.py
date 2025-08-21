"""Integrated Imagination System combining all components.

This system integrates:
1. Hypothesis Generator - Discovers novel patterns
2. Abstract Principle Extractor - Enables cross-domain transfer
3. Compositional Reasoner - Handles multi-attribute rules
4. Pattern Grammar Learner - Extracts atomic operations
5. Causal Reasoning - Understands WHY patterns work
6. Program Synthesis - Generates readable solutions

The integration creates a complete imagination pipeline that can tackle
all types of imagination tasks.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

# Import our imagination components
from abstract_principle_extractor import (
    AbstractPrincipleExtractor,
    Domain,
)
from compositional_reasoner import (
    CompositionalReasoner,
)
from improved_compositional_reasoner import (
    ImprovedCompositionalReasoner,
)
from hypothesis_generator import (
    GenerationStrategy,
    Hypothesis,
    MinimalHypothesisGenerator,
)

# Import baseline components - using stub implementations for now
# from baseline.causal_reasoning_module import CausalReasoner
# from baseline.few_shot_pattern_learner import FewShotLearner
# from baseline.pattern_grammar_learner import PatternGrammarLearner
# from baseline.program_synthesis_natural_priors import ProgramSynthesizer

# Stub implementations for baseline components
class CausalReasoner:
    def analyze(self, examples):
        return None
    def apply_model(self, model, inp):
        return None
    def explain_model(self, model):
        return "Causal model explanation"

class FewShotLearner:
    def learn(self, examples):
        return None
    def apply_pattern(self, pattern, inp):
        return None

class PatternGrammarLearner:
    def extract_operations(self, examples):
        return []
    def learn_grammar(self, examples):
        return None
    def apply_grammar(self, grammar, inp):
        return None

class ProgramSynthesizer:
    def synthesize(self, examples, max_attempts=1000):
        return None
    def execute(self, program, inp):
        return None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ImaginationResult:
    """Result from the imagination system."""
    
    success: bool
    score: float
    method_used: str
    hypothesis: Optional[Hypothesis] = None
    principle: Optional[Any] = None
    rule: Optional[Any] = None
    program: Optional[str] = None
    explanation: str = ""


class IntegratedImaginationSystem:
    """Complete imagination system with all components integrated."""
    
    def __init__(self, verbose: bool = True):
        """Initialize all components."""
        self.verbose = verbose
        
        # Core imagination components
        self.hypothesis_generator = MinimalHypothesisGenerator(seed=42)
        self.principle_extractor = AbstractPrincipleExtractor()
        self.compositional_reasoner = CompositionalReasoner()
        self.improved_compositional = ImprovedCompositionalReasoner()
        
        # Baseline reasoning components
        self.pattern_grammar = PatternGrammarLearner()
        self.few_shot_learner = FewShotLearner()
        self.causal_reasoner = CausalReasoner()
        self.program_synthesizer = ProgramSynthesizer()
        
        # Track performance
        self.results_history = []
        
        if self.verbose:
            logger.info("Integrated Imagination System initialized")
    
    def imagine(
        self,
        task_examples: List[Tuple[np.ndarray, np.ndarray]],
        task_category: Optional[str] = None,
        max_attempts: int = 1000
    ) -> ImaginationResult:
        """Main imagination method that tries all strategies.
        
        Args:
            task_examples: Input-output examples
            task_category: Optional hint about task type
            max_attempts: Maximum attempts for hypothesis generation
            
        Returns:
            ImaginationResult with best solution found
        """
        
        if self.verbose:
            logger.info(f"Starting imagination on {len(task_examples)} examples")
            if task_category:
                logger.info(f"Task category hint: {task_category}")
        
        # Try strategies in order of expected success
        strategies = self._select_strategies(task_category)
        
        best_result = ImaginationResult(
            success=False,
            score=0.0,
            method_used="none",
            explanation="No successful strategy found"
        )
        
        for strategy_name, strategy_fn in strategies:
            if self.verbose:
                logger.info(f"Trying strategy: {strategy_name}")
            
            try:
                result = strategy_fn(task_examples, max_attempts)
                
                if result.score > best_result.score:
                    best_result = result
                    
                    if self.verbose:
                        logger.info(f"New best: {strategy_name} (score: {result.score:.1%})")
                    
                    # Early stopping on perfect solution
                    if result.score >= 1.0:
                        if self.verbose:
                            logger.info(f"Perfect solution found with {strategy_name}!")
                        break
                        
            except Exception as e:
                logger.warning(f"Strategy {strategy_name} failed: {e}")
                continue
        
        self.results_history.append(best_result)
        return best_result
    
    def _select_strategies(self, task_category: Optional[str]) -> List[Tuple[str, Any]]:
        """Select strategies based on task category."""
        
        # Default strategy order
        strategies = []
        
        if task_category == "pattern_discovery":
            # Hypothesis generation first for geometric patterns
            strategies.append(("hypothesis_generation", self._try_hypothesis_generation))
            strategies.append(("pattern_grammar", self._try_pattern_grammar))
            
        elif task_category == "rule_combination":
            # Compositional reasoning first for multi-attribute
            strategies.append(("compositional_reasoning", self._try_compositional))
            strategies.append(("causal_reasoning", self._try_causal))
            
        elif task_category == "cross_domain":
            # Principle extraction first for transfer
            strategies.append(("principle_extraction", self._try_principle_extraction))
            strategies.append(("few_shot_transfer", self._try_few_shot))
            
        elif task_category == "counterfactual":
            # Causal reasoning for understanding inversions
            strategies.append(("causal_reasoning", self._try_causal))
            strategies.append(("hypothesis_generation", self._try_hypothesis_generation))
            
        elif task_category == "creative":
            # Program synthesis for novel algorithms
            strategies.append(("program_synthesis", self._try_program_synthesis))
            strategies.append(("hypothesis_generation", self._try_hypothesis_generation))
            
        else:
            # Try all strategies in general order
            strategies = [
                ("hypothesis_generation", self._try_hypothesis_generation),
                ("compositional_reasoning", self._try_compositional),
                ("principle_extraction", self._try_principle_extraction),
                ("pattern_grammar", self._try_pattern_grammar),
                ("causal_reasoning", self._try_causal),
                ("few_shot_learning", self._try_few_shot),
                ("program_synthesis", self._try_program_synthesis),
            ]
        
        return strategies
    
    def _try_hypothesis_generation(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        max_attempts: int
    ) -> ImaginationResult:
        """Try hypothesis generation strategy."""
        
        # Try different generation strategies
        best_hypothesis = None
        best_score = 0.0
        
        for strategy in [GenerationStrategy.SYSTEMATIC, GenerationStrategy.RANDOM,
                         GenerationStrategy.COMPOSITIONAL, GenerationStrategy.CONSTRAINT_RELAXATION]:
            
            hypothesis = self.hypothesis_generator.discover_pattern(
                examples,
                max_attempts=max_attempts // 4,  # Split attempts
                strategies=[strategy]
            )
            
            if hypothesis:
                score = self.hypothesis_generator.test_hypothesis(hypothesis, examples)
                
                if score > best_score:
                    best_score = score
                    best_hypothesis = hypothesis
                    
                    if score >= 1.0:
                        break
        
        if best_hypothesis:
            return ImaginationResult(
                success=best_score > 0.5,
                score=best_score,
                method_used="hypothesis_generation",
                hypothesis=best_hypothesis,
                explanation=f"Discovered {best_hypothesis.transform_type} pattern"
            )
        
        return ImaginationResult(
            success=False,
            score=0.0,
            method_used="hypothesis_generation",
            explanation="No pattern discovered"
        )
    
    def _try_compositional(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        max_attempts: int
    ) -> ImaginationResult:
        """Try compositional reasoning strategy."""
        
        # First try improved compositional reasoner
        if len(examples) > 0:
            # Try to solve as combination task
            test_input = examples[0][0] if examples else None
            
            if test_input is not None:
                # Use improved reasoner
                result = self.improved_compositional.solve_combination_task(
                    examples[:-1] if len(examples) > 1 else examples,
                    test_input
                )
                
                # Test the result
                score = 0.0
                for inp, out in examples:
                    predicted = self.improved_compositional.solve_combination_task(
                        examples[:-1] if len(examples) > 1 else examples,
                        inp
                    )
                    if predicted.shape == out.shape:
                        score += np.sum(predicted == out) / out.size
                
                score = score / len(examples) if examples else 0.0
                
                if score > 0.5:
                    return ImaginationResult(
                        success=True,
                        score=score,
                        method_used="improved_compositional",
                        explanation="Learned and combined transformations"
                    )
        
        # Fallback to original compositional reasoner
        rule = self.compositional_reasoner.learn_rule_from_examples(examples)
        
        if not rule:
            # Try conditional rule discovery
            rule = self.compositional_reasoner.discover_conditional_rule(examples)
        
        if rule:
            # Test the rule
            score = 0.0
            for inp, out in examples:
                predicted = self.compositional_reasoner.apply_rule(rule, inp)
                if predicted.shape == out.shape:
                    score += np.sum(predicted == out) / out.size
            
            score = score / len(examples) if examples else 0.0
            
            return ImaginationResult(
                success=score > 0.5,
                score=score,
                method_used="compositional_reasoning",
                rule=rule,
                explanation=self.compositional_reasoner.explain_rule(rule)
            )
        
        return ImaginationResult(
            success=False,
            score=0.0,
            method_used="compositional_reasoning",
            explanation="No compositional rule found"
        )
    
    def _try_principle_extraction(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        max_attempts: int
    ) -> ImaginationResult:
        """Try principle extraction and cross-domain transfer."""
        
        # First discover a pattern
        hypothesis = self.hypothesis_generator.discover_pattern(
            examples[:2] if len(examples) > 2 else examples,
            max_attempts=max_attempts // 2
        )
        
        if hypothesis:
            # Extract principle
            principle = self.principle_extractor.extract_principle(hypothesis, examples)
            
            if principle:
                # Try to apply principle
                target_domain = self.principle_extractor.identify_domain(examples[0][0])
                
                transform_fn = self.principle_extractor.transfer_principle(
                    principle, target_domain, examples
                )
                
                if transform_fn:
                    # Test the transfer
                    score = 0.0
                    for inp, out in examples:
                        predicted = transform_fn(inp)
                        if predicted.shape == out.shape:
                            score += np.sum(predicted == out) / out.size
                    
                    score = score / len(examples) if examples else 0.0
                    
                    return ImaginationResult(
                        success=score > 0.5,
                        score=score,
                        method_used="principle_extraction",
                        principle=principle,
                        explanation=self.principle_extractor.explain_principle(principle)
                    )
        
        return ImaginationResult(
            success=False,
            score=0.0,
            method_used="principle_extraction",
            explanation="No transferable principle found"
        )
    
    def _try_pattern_grammar(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        max_attempts: int
    ) -> ImaginationResult:
        """Try pattern grammar learning."""
        
        # Extract atomic operations
        operations = self.pattern_grammar.extract_operations(examples)
        
        if operations:
            # Learn composition
            grammar = self.pattern_grammar.learn_grammar(examples)
            
            if grammar:
                # Apply learned grammar
                score = 0.0
                for inp, out in examples:
                    predicted = self.pattern_grammar.apply_grammar(grammar, inp)
                    if predicted is not None and predicted.shape == out.shape:
                        score += np.sum(predicted == out) / out.size
                
                score = score / len(examples) if examples else 0.0
                
                if score > 0:
                    return ImaginationResult(
                        success=score > 0.5,
                        score=score,
                        method_used="pattern_grammar",
                        explanation=f"Learned grammar with {len(operations)} operations"
                    )
        
        return ImaginationResult(
            success=False,
            score=0.0,
            method_used="pattern_grammar",
            explanation="No pattern grammar learned"
        )
    
    def _try_causal(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        max_attempts: int
    ) -> ImaginationResult:
        """Try causal reasoning."""
        
        # Analyze causal structure
        causal_model = self.causal_reasoner.analyze(examples)
        
        if causal_model:
            # Apply causal model
            score = 0.0
            for inp, out in examples:
                predicted = self.causal_reasoner.apply_model(causal_model, inp)
                if predicted is not None and predicted.shape == out.shape:
                    score += np.sum(predicted == out) / out.size
            
            score = score / len(examples) if examples else 0.0
            
            if score > 0:
                explanation = self.causal_reasoner.explain_model(causal_model)
                
                return ImaginationResult(
                    success=score > 0.5,
                    score=score,
                    method_used="causal_reasoning",
                    explanation=explanation
                )
        
        return ImaginationResult(
            success=False,
            score=0.0,
            method_used="causal_reasoning",
            explanation="No causal model found"
        )
    
    def _try_few_shot(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        max_attempts: int
    ) -> ImaginationResult:
        """Try few-shot learning."""
        
        # Learn from few examples
        pattern = self.few_shot_learner.learn(examples[:3] if len(examples) > 3 else examples)
        
        if pattern:
            # Test on remaining examples
            score = 0.0
            test_examples = examples[3:] if len(examples) > 3 else examples
            
            for inp, out in test_examples:
                predicted = self.few_shot_learner.apply_pattern(pattern, inp)
                if predicted is not None and predicted.shape == out.shape:
                    score += np.sum(predicted == out) / out.size
            
            score = score / len(test_examples) if test_examples else 0.0
            
            if score > 0:
                return ImaginationResult(
                    success=score > 0.5,
                    score=score,
                    method_used="few_shot_learning",
                    explanation=f"Learned pattern from {len(examples)} examples"
                )
        
        return ImaginationResult(
            success=False,
            score=0.0,
            method_used="few_shot_learning",
            explanation="No pattern learned from few examples"
        )
    
    def _try_program_synthesis(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        max_attempts: int
    ) -> ImaginationResult:
        """Try program synthesis."""
        
        # Synthesize program
        program = self.program_synthesizer.synthesize(examples, max_attempts=max_attempts)
        
        if program:
            # Test program
            score = 0.0
            for inp, out in examples:
                predicted = self.program_synthesizer.execute(program, inp)
                if predicted is not None and predicted.shape == out.shape:
                    score += np.sum(predicted == out) / out.size
            
            score = score / len(examples) if examples else 0.0
            
            if score > 0:
                return ImaginationResult(
                    success=score > 0.5,
                    score=score,
                    method_used="program_synthesis",
                    program=str(program),
                    explanation=f"Synthesized program: {program.name if hasattr(program, 'name') else 'unnamed'}"
                )
        
        return ImaginationResult(
            success=False,
            score=0.0,
            method_used="program_synthesis",
            explanation="No program synthesized"
        )
    
    def explain_solution(self, result: ImaginationResult) -> str:
        """Generate detailed explanation of the solution."""
        
        explanation = f"Method: {result.method_used}\n"
        explanation += f"Success: {'Yes' if result.success else 'No'}\n"
        explanation += f"Score: {result.score:.1%}\n\n"
        
        if result.explanation:
            explanation += f"Details:\n{result.explanation}\n"
        
        if result.hypothesis:
            explanation += f"\nHypothesis Type: {result.hypothesis.transform_type}\n"
            explanation += f"Parameters: {result.hypothesis.parameters}\n"
        
        if result.principle:
            explanation += f"\nPrinciple: {result.principle.name}\n"
            explanation += f"Operation: {result.principle.operation}\n"
        
        if result.rule:
            explanation += f"\nRule: {result.rule.name if hasattr(result.rule, 'name') else 'unnamed'}\n"
        
        if result.program:
            explanation += f"\nProgram:\n{result.program}\n"
        
        return explanation
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        
        if not self.results_history:
            return {
                "total_attempts": 0,
                "success_rate": 0.0,
                "average_score": 0.0,
                "best_method": "none"
            }
        
        successful = [r for r in self.results_history if r.success]
        
        # Count method usage
        method_counts = {}
        method_scores = {}
        
        for result in self.results_history:
            method = result.method_used
            if method not in method_counts:
                method_counts[method] = 0
                method_scores[method] = []
            method_counts[method] += 1
            method_scores[method].append(result.score)
        
        # Find best method
        best_method = "none"
        best_avg = 0.0
        
        for method, scores in method_scores.items():
            avg = np.mean(scores)
            if avg > best_avg:
                best_avg = avg
                best_method = method
        
        return {
            "total_attempts": len(self.results_history),
            "success_rate": len(successful) / len(self.results_history),
            "average_score": np.mean([r.score for r in self.results_history]),
            "best_method": best_method,
            "method_performance": {
                method: {
                    "count": method_counts[method],
                    "avg_score": np.mean(method_scores[method])
                }
                for method in method_counts
            }
        }