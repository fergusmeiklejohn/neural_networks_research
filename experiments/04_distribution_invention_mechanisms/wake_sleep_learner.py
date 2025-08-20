"""Wake-Sleep Learning System for Self-Improvement.

Inspired by DreamCoder, this system alternates between:
1. Wake: Solving real tasks and storing experience
2. Sleep: Training on synthetic tasks generated from learned principles
3. Dream: Creative exploration without correctness constraints
4. Consolidation: Extracting abstractions and building libraries

The system progressively builds expertise by creating domain-specific
abstractions and learning to reuse them effectively.
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from causal_reasoning_module import CausalReasoningModule, TransformationPrinciple
from few_shot_pattern_learner import FewShotPatternLearner
from pattern_grammar_learner import PatternGrammarLearner
from program_synthesis_natural_priors import (
    ProgramNode,
    ProgramSynthesizer,
    SynthesizedProgram,
)


@dataclass
class Experience:
    """Represents a single learning experience."""

    task_id: str
    examples: List[Tuple[np.ndarray, np.ndarray]]
    solution: SynthesizedProgram
    principle: Optional[TransformationPrinciple]
    score: float
    timestamp: datetime
    iteration: int
    source: str  # 'wake', 'sleep', 'dream'

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "score": self.score,
            "iteration": self.iteration,
            "source": self.source,
            "principle": self.principle.name if self.principle else None,
            "program_complexity": self.solution.complexity,
        }


@dataclass
class ProgramAbstraction:
    """Represents a reusable program component."""

    name: str
    program: ProgramNode
    frequency: int  # How often it's been used
    examples: List[str]  # Task IDs where it was useful
    description: str

    def __hash__(self):
        return hash(self.name)


class ExperienceBuffer:
    """Stores and manages learning experiences."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.experiences: List[Experience] = []
        self.by_principle: Dict[str, List[Experience]] = defaultdict(list)
        self.by_score: List[Experience] = []  # Sorted by score

    def add(self, experience: Experience):
        """Add new experience to buffer."""
        self.experiences.append(experience)

        # Organize by principle
        if experience.principle:
            self.by_principle[experience.principle.name].append(experience)

        # Maintain sorted list by score
        self.by_score.append(experience)
        self.by_score.sort(key=lambda e: -e.score)

        # Remove oldest if buffer is full
        if len(self.experiences) > self.max_size:
            self.experiences.pop(0)

    def sample(self, n: int, strategy: str = "random") -> List[Experience]:
        """Sample experiences from buffer."""
        if not self.experiences:
            return []

        if strategy == "random":
            n = min(n, len(self.experiences))
            indices = np.random.choice(len(self.experiences), n, replace=False)
            return [self.experiences[i] for i in indices]

        elif strategy == "best":
            return self.by_score[:n]

        elif strategy == "recent":
            return self.experiences[-n:]

        elif strategy == "prioritized":
            # Sample with probability proportional to score
            scores = np.array([e.score for e in self.experiences])
            probs = scores / scores.sum()
            n = min(n, len(self.experiences))
            indices = np.random.choice(len(self.experiences), n, p=probs, replace=False)
            return [self.experiences[i] for i in indices]

        return []

    def get_by_principle(self, principle_name: str) -> List[Experience]:
        """Get all experiences with a specific principle."""
        return self.by_principle.get(principle_name, [])


class ProgramLibrary:
    """Manages a library of reusable program components."""

    def __init__(self):
        self.abstractions: Dict[str, ProgramAbstraction] = {}
        self.hierarchy: Dict[str, Set[str]] = defaultdict(set)  # parent -> children

    def add_abstraction(self, abstraction: ProgramAbstraction):
        """Add new abstraction to library."""
        self.abstractions[abstraction.name] = abstraction

    def find_applicable(
        self, task_examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[ProgramAbstraction]:
        """Find abstractions that might apply to given task."""
        applicable = []

        for name, abstraction in self.abstractions.items():
            # Simple heuristic: abstractions used frequently are more likely to apply
            if abstraction.frequency > 2:
                applicable.append(abstraction)

        # Sort by frequency
        applicable.sort(key=lambda a: -a.frequency)
        return applicable[:5]  # Return top 5

    def update_usage(self, abstraction_name: str, task_id: str):
        """Update usage statistics for an abstraction."""
        if abstraction_name in self.abstractions:
            self.abstractions[abstraction_name].frequency += 1
            self.abstractions[abstraction_name].examples.append(task_id)


class WakeSleepLearner:
    """Main Wake-Sleep learning system."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

        # Core components
        self.grammar_learner = PatternGrammarLearner(verbose=False)
        self.few_shot_learner = FewShotPatternLearner(self.grammar_learner)
        self.causal_module = CausalReasoningModule(verbose=False)
        self.synthesizer = ProgramSynthesizer(
            self.grammar_learner, self.causal_module, verbose=False
        )

        # Learning infrastructure
        self.experience_buffer = ExperienceBuffer()
        self.program_library = ProgramLibrary()

        # Statistics
        self.iteration = 0
        self.total_tasks_solved = 0
        self.wake_success_rate = []
        self.sleep_success_rate = []

    def wake_phase(self, tasks: List[Dict]) -> Dict:
        """
        Wake phase: Solve real tasks and store experience.

        Args:
            tasks: List of task dictionaries with 'id' and 'examples'

        Returns:
            Statistics about wake phase performance
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"WAKE PHASE - Iteration {self.iteration}")
            print(f"{'='*60}")

        solved = 0
        total = len(tasks)

        for task in tasks:
            task_id = task.get("id", f"task_{self.iteration}_{solved}")
            examples = task["examples"]

            if self.verbose:
                print(f"\nSolving task: {task_id}")

            # Try to solve with current knowledge
            solution = self._solve_task(examples, use_library=True)

            if solution and solution.score > 0.5:
                # Successful solution
                solved += 1

                # Extract principle
                causal_analysis = self.causal_module.analyze_transformation(examples)
                principle = causal_analysis.get("principle")

                # Store experience
                experience = Experience(
                    task_id=task_id,
                    examples=examples,
                    solution=solution,
                    principle=principle,
                    score=solution.score,
                    timestamp=datetime.now(),
                    iteration=self.iteration,
                    source="wake",
                )
                self.experience_buffer.add(experience)

                if self.verbose:
                    print(f"  ✓ Solved with score {solution.score:.2f}")
                    if principle:
                        print(f"  Principle: {principle.name}")

        success_rate = solved / total if total > 0 else 0
        self.wake_success_rate.append(success_rate)

        if self.verbose:
            print(
                f"\nWake phase complete: {solved}/{total} tasks solved ({success_rate:.1%})"
            )

        return {"solved": solved, "total": total, "success_rate": success_rate}

    def sleep_phase(self, num_synthetic: int = 10) -> Dict:
        """
        Sleep phase: Generate and train on synthetic tasks.

        Args:
            num_synthetic: Number of synthetic tasks to generate

        Returns:
            Statistics about sleep phase performance
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"SLEEP PHASE - Iteration {self.iteration}")
            print(f"{'='*60}")

        # Sample experiences to learn from
        experiences = self.experience_buffer.sample(
            min(5, len(self.experience_buffer.experiences)), strategy="prioritized"
        )

        if not experiences:
            if self.verbose:
                print("No experiences to learn from yet")
            return {"generated": 0, "solved": 0}

        synthetic_tasks = []

        # Generate synthetic tasks based on learned principles
        for exp in experiences:
            if exp.principle:
                # Create variations of the principle
                for i in range(num_synthetic // len(experiences)):
                    synthetic_task = self._generate_synthetic_task(exp.principle)
                    if synthetic_task:
                        synthetic_tasks.append(synthetic_task)

        if self.verbose:
            print(f"Generated {len(synthetic_tasks)} synthetic tasks")

        # Try to solve synthetic tasks
        solved = 0
        for task in synthetic_tasks:
            solution = self._solve_task(task["examples"], use_library=True)

            if solution and solution.score > 0.5:
                solved += 1

                # Store synthetic experience
                experience = Experience(
                    task_id=task["id"],
                    examples=task["examples"],
                    solution=solution,
                    principle=task.get("principle"),
                    score=solution.score,
                    timestamp=datetime.now(),
                    iteration=self.iteration,
                    source="sleep",
                )
                self.experience_buffer.add(experience)

        success_rate = solved / len(synthetic_tasks) if synthetic_tasks else 0
        self.sleep_success_rate.append(success_rate)

        if self.verbose:
            print(
                f"Sleep phase complete: {solved}/{len(synthetic_tasks)} synthetic tasks solved"
            )

        return {
            "generated": len(synthetic_tasks),
            "solved": solved,
            "success_rate": success_rate,
        }

    def dream_phase(self, num_dreams: int = 5) -> Dict:
        """
        Dream phase: Explore creative combinations without correctness constraints.

        Args:
            num_dreams: Number of dream explorations

        Returns:
            Statistics about dream phase discoveries
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"DREAM PHASE - Iteration {self.iteration}")
            print(f"{'='*60}")

        discoveries = []

        # Sample diverse experiences
        experiences = self.experience_buffer.sample(
            min(3, len(self.experience_buffer.experiences)), strategy="random"
        )

        for i in range(num_dreams):
            if len(experiences) >= 2:
                # Combine principles from different experiences
                exp1, exp2 = np.random.choice(experiences, 2, replace=False)

                if exp1.solution and exp2.solution:
                    # Create novel composition
                    novel_program = self._compose_programs(
                        exp1.solution.root, exp2.solution.root
                    )

                    # Test on a random task
                    test_exp = np.random.choice(experiences)
                    try:
                        result = SynthesizedProgram(
                            root=novel_program,
                            score=0,
                            complexity=novel_program.complexity(),
                            respects_invariants=True,
                            follows_causality=False,
                            description="Dream composition",
                        ).execute(test_exp.examples[0][0])

                        # Even if it doesn't match expected output, it might be interesting
                        if result is not None:
                            discoveries.append(
                                {
                                    "program": novel_program,
                                    "source_tasks": [exp1.task_id, exp2.task_id],
                                    "complexity": novel_program.complexity(),
                                }
                            )

                            if self.verbose:
                                print(f"  Dream {i+1}: Novel composition discovered")
                    except:
                        pass  # Dreams can fail, that's okay

        if self.verbose:
            print(
                f"Dream phase complete: {len(discoveries)} novel compositions explored"
            )

        return {"num_dreams": num_dreams, "discoveries": len(discoveries)}

    def consolidate(self) -> Dict:
        """
        Consolidation phase: Extract abstractions and update library.

        Returns:
            Statistics about consolidation
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"CONSOLIDATION PHASE - Iteration {self.iteration}")
            print(f"{'='*60}")

        # Get recent successful experiences
        recent_experiences = [
            exp for exp in self.experience_buffer.experiences[-20:] if exp.score > 0.7
        ]

        if len(recent_experiences) < 2:
            if self.verbose:
                print("Not enough successful experiences to consolidate")
            return {"abstractions_created": 0}

        # Find common patterns
        abstractions_created = 0
        program_nodes = [exp.solution.root for exp in recent_experiences]

        # Simple abstraction: find repeated operations
        operation_counts = defaultdict(int)
        for node in program_nodes:
            self._count_operations(node, operation_counts)

        # Create abstractions for frequent patterns
        for operation, count in operation_counts.items():
            if count >= 3:  # Used at least 3 times
                abstraction_name = f"learned_{operation}_{self.iteration}"

                # Create simple abstraction (in real system, would be more sophisticated)
                abstraction = ProgramAbstraction(
                    name=abstraction_name,
                    program=ProgramNode(operation, "atomic"),
                    frequency=count,
                    examples=[exp.task_id for exp in recent_experiences[:count]],
                    description=f"Learned pattern: {operation}",
                )

                self.program_library.add_abstraction(abstraction)
                abstractions_created += 1

                if self.verbose:
                    print(
                        f"  Created abstraction: {abstraction_name} (used {count} times)"
                    )

        # Update natural priors based on experience
        if abstractions_created > 0:
            self._update_priors()

        if self.verbose:
            print(
                f"Consolidation complete: {abstractions_created} abstractions created"
            )
            print(f"Library size: {len(self.program_library.abstractions)}")

        return {
            "abstractions_created": abstractions_created,
            "library_size": len(self.program_library.abstractions),
        }

    def run_iteration(self, real_tasks: List[Dict]) -> Dict:
        """
        Run one complete wake-sleep-dream-consolidate iteration.

        Args:
            real_tasks: Real tasks to solve in wake phase

        Returns:
            Combined statistics from all phases
        """
        self.iteration += 1

        if self.verbose:
            print(f"\n{'#'*70}")
            print(f"# WAKE-SLEEP ITERATION {self.iteration}")
            print(f"{'#'*70}")

        stats = {}

        # Wake phase
        wake_stats = self.wake_phase(real_tasks)
        stats["wake"] = wake_stats

        # Sleep phase
        sleep_stats = self.sleep_phase(num_synthetic=10)
        stats["sleep"] = sleep_stats

        # Dream phase
        dream_stats = self.dream_phase(num_dreams=5)
        stats["dream"] = dream_stats

        # Consolidation
        consolidation_stats = self.consolidate()
        stats["consolidation"] = consolidation_stats

        # Overall statistics
        self.total_tasks_solved += wake_stats["solved"]
        stats["total_solved"] = self.total_tasks_solved
        stats["iteration"] = self.iteration

        if self.verbose:
            print(f"\n{'#'*70}")
            print(f"# ITERATION {self.iteration} COMPLETE")
            print(f"# Total tasks solved: {self.total_tasks_solved}")
            print(f"# Library size: {len(self.program_library.abstractions)}")
            print(f"# Wake success rate: {wake_stats['success_rate']:.1%}")
            print(f"{'#'*70}")

        return stats

    def _solve_task(
        self, examples: List[Tuple[np.ndarray, np.ndarray]], use_library: bool = True
    ) -> Optional[SynthesizedProgram]:
        """Solve a task using current knowledge."""
        # First try with library components if available
        if use_library and self.program_library.abstractions:
            self.program_library.find_applicable(examples)
            # In real implementation, would use library components in synthesis

        # Use standard synthesis pipeline
        programs = self.synthesizer.synthesize(examples, max_programs=5)

        if programs:
            return programs[0]  # Return best program
        return None

    def _generate_synthetic_task(
        self, principle: TransformationPrinciple
    ) -> Optional[Dict]:
        """Generate a synthetic task based on a principle."""
        # Create random input
        size = np.random.randint(3, 6)
        num_examples = 3

        examples = []
        for i in range(num_examples):
            inp = np.random.randint(0, 3, (size, size))

            # Apply transformation based on principle
            if "rotation" in principle.name.lower():
                out = np.rot90(inp, np.random.randint(1, 4))
            elif "scaling" in principle.name.lower():
                factor = np.random.choice([2, 3])
                out = np.repeat(np.repeat(inp, factor, axis=0), factor, axis=1)
            elif "flip" in principle.name.lower():
                out = np.flipud(inp) if np.random.random() > 0.5 else np.fliplr(inp)
            else:
                # Default: some transformation
                out = np.rot90(inp)

            examples.append((inp, out))

        return {
            "id": f"synthetic_{self.iteration}_{len(self.experience_buffer.experiences)}",
            "examples": examples,
            "principle": principle,
        }

    def _compose_programs(self, prog1: ProgramNode, prog2: ProgramNode) -> ProgramNode:
        """Create a novel composition of two programs."""
        # Simple composition strategies
        strategies = ["sequential", "conditional", "parallel"]
        strategy = np.random.choice(strategies)

        if strategy == "sequential":
            return ProgramNode(
                operation="sequence", node_type="composite", children=[prog1, prog2]
            )
        elif strategy == "conditional":
            return ProgramNode(
                operation="conditional",
                node_type="conditional",
                children=[prog1, prog2],
                parameters={"condition": "has_pattern"},
            )
        else:  # parallel
            return ProgramNode(
                operation="parallel", node_type="composite", children=[prog1, prog2]
            )

    def _count_operations(self, node: ProgramNode, counts: Dict):
        """Recursively count operations in a program tree."""
        counts[node.operation] += 1
        for child in node.children:
            self._count_operations(child, counts)

    def _update_priors(self):
        """Update synthesizer priors based on learned patterns."""
        # Adjust weights based on what worked
        total_experiences = len(self.experience_buffer.experiences)
        if total_experiences > 10:
            # Analyze successful programs
            successful = [
                e for e in self.experience_buffer.experiences if e.score > 0.8
            ]

            avg_complexity = np.mean([e.solution.complexity for e in successful])

            # If simple programs work well, increase simplicity weight
            if avg_complexity < 2:
                self.synthesizer.simplicity_weight = min(
                    0.5, self.synthesizer.simplicity_weight + 0.05
                )
            # If complex programs needed, decrease simplicity weight
            elif avg_complexity > 3:
                self.synthesizer.simplicity_weight = max(
                    0.1, self.synthesizer.simplicity_weight - 0.05
                )

    def save_state(self, path: Path):
        """Save learner state to disk."""
        state = {
            "iteration": self.iteration,
            "total_tasks_solved": self.total_tasks_solved,
            "wake_success_rate": self.wake_success_rate,
            "sleep_success_rate": self.sleep_success_rate,
            "experiences": [
                exp.to_dict() for exp in self.experience_buffer.experiences[-100:]
            ],
            "library_size": len(self.program_library.abstractions),
        }

        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)

        if self.verbose:
            print(f"State saved to {path}")

    def get_improvement_curve(self) -> Dict:
        """Get learning curve statistics."""
        return {
            "wake_success_rate": self.wake_success_rate,
            "sleep_success_rate": self.sleep_success_rate,
            "library_growth": [
                len(self.program_library.abstractions) for _ in range(self.iteration)
            ],
            "total_experiences": len(self.experience_buffer.experiences),
        }
