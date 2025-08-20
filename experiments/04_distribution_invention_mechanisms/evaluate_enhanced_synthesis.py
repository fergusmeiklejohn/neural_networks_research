#!/usr/bin/env python3
"""Evaluate the enhanced synthesis system with new DSL primitives.

This script:
1. Tests on previously failed tasks
2. Uses the enhanced DSL with 31 primitives
3. Leverages the trained neural ranker
4. Measures improvement over baseline
"""

from utils.imports import setup_project_paths

setup_project_paths()

import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from bidirectional_synthesis import BidirectionalSynthesizer
from enhanced_compositional_dsl import EnhancedCompositionalDSL
from neural_guided_synthesis import NeuralGuidedSynthesizer
from neural_program_ranker import NeuralProgramRanker
from tqdm import tqdm


def load_trained_ranker() -> Optional[NeuralProgramRanker]:
    """Load the trained neural program ranker."""
    model_path = Path("trained_neural_ranker.pt")

    if not model_path.exists():
        print("⚠️ Trained ranker not found, using untrained model")
        return NeuralProgramRanker(
            vocab_size=100, hidden_dim=256, num_heads=8, num_layers=4
        )

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")

    # Create model
    model = NeuralProgramRanker(
        vocab_size=checkpoint.get("vocab_size", 100),
        hidden_dim=checkpoint.get("hidden_dim", 256),
        num_heads=checkpoint.get("num_heads", 8),
        num_layers=checkpoint.get("num_layers", 4),
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("✅ Loaded trained neural ranker")
    return model


def load_arc_task(task_id: str) -> Optional[Dict]:
    """Load an ARC task by ID."""
    # Try training directory first
    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")
    task_file = data_dir / f"{task_id}.json"

    if not task_file.exists():
        # Try evaluation directory
        data_dir = Path("data/arc_agi_official/ARC-AGI/data/evaluation")
        task_file = data_dir / f"{task_id}.json"

    if not task_file.exists():
        return None

    with open(task_file, "r") as f:
        return json.load(f)


def evaluate_synthesis_on_task(
    task_id: str,
    synthesizers: Dict,
    dsl: EnhancedCompositionalDSL,
    verbose: bool = False,
) -> Dict:
    """Evaluate synthesis on a single task."""
    # Load task
    task = load_arc_task(task_id)
    if not task:
        return {"task_id": task_id, "error": "Task not found", "solved": False}

    # Convert to numpy arrays
    train_examples = [
        (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
    ]

    test_examples = [
        (np.array(ex["input"]), np.array(ex["output"])) for ex in task["test"]
    ]

    results = {
        "task_id": task_id,
        "num_train": len(train_examples),
        "num_test": len(test_examples),
    }

    # Try each synthesizer
    for synth_name, synthesizer in synthesizers.items():
        if verbose:
            print(f"  Trying {synth_name}...")

        start_time = time.time()

        try:
            # Synthesize program
            program = synthesizer.synthesize(train_examples)
            synthesis_time = time.time() - start_time

            if program:
                # Evaluate on train and test
                train_correct = 0
                for inp, expected in train_examples:
                    try:
                        result = dsl.execute_program(program, inp)
                        if np.array_equal(result, expected):
                            train_correct += 1
                    except:
                        pass

                test_correct = 0
                for inp, expected in test_examples:
                    try:
                        result = dsl.execute_program(program, inp)
                        if np.array_equal(result, expected):
                            test_correct += 1
                    except:
                        pass

                train_acc = train_correct / len(train_examples) if train_examples else 0
                test_acc = test_correct / len(test_examples) if test_examples else 0

                results[synth_name] = {
                    "program": str(program),
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "synthesis_time": synthesis_time,
                    "solved": test_acc == 1.0,
                }
            else:
                results[synth_name] = {
                    "program": None,
                    "train_acc": 0,
                    "test_acc": 0,
                    "synthesis_time": synthesis_time,
                    "solved": False,
                }

        except Exception as e:
            results[synth_name] = {
                "program": None,
                "error": str(e)[:100],
                "synthesis_time": time.time() - start_time,
                "solved": False,
            }

    # Overall solved
    results["solved"] = any(
        r.get("solved", False) for r in results.values() if isinstance(r, dict)
    )

    return results


def evaluate_on_failed_tasks():
    """Evaluate on previously failed tasks."""
    # Load list of failed tasks
    detailed_file = Path("failed_tasks_detailed.json")
    if detailed_file.exists():
        with open(detailed_file, "r") as f:
            data = json.load(f)
            failed_ids = data["failed_task_ids"]
    else:
        # Use default failed tasks
        failed_ids = [
            "ae3edfdc",
            "d406998b",
            "8403a5d5",
            "8e5a5113",
            "508bd3b6",
            "db93a21d",
            "484b58aa",
            "1a07d186",
        ]

    print(f"Evaluating on {len(failed_ids)} previously failed tasks")

    # Create enhanced DSL
    dsl = EnhancedCompositionalDSL()
    print(f"Using enhanced DSL with {len(dsl.all_primitives)} primitives")

    # Load neural ranker
    neural_ranker = load_trained_ranker()

    # Create synthesizers (reduced timeout for faster evaluation)
    synthesizers = {
        "bidirectional": BidirectionalSynthesizer(dsl, timeout=10.0),
        "neural_guided": NeuralGuidedSynthesizer(
            dsl,
            neural_ranker=neural_ranker,
            beam_width=30,
            timeout=10.0,
            neural_weight=0.3,
        ),
    }

    # Evaluate on each task
    all_results = []
    solved_count = 0

    # Test on subset for speed
    test_tasks = failed_ids[:5]  # Test first 5 for quick evaluation

    for task_id in tqdm(test_tasks, desc="Evaluating"):
        result = evaluate_synthesis_on_task(task_id, synthesizers, dsl, verbose=False)
        all_results.append(result)

        if result.get("solved", False):
            solved_count += 1
            print(f"  ✅ Solved {task_id}!")

    return all_results, solved_count


def compare_with_baseline():
    """Compare enhanced system with baseline results."""
    # Load baseline results
    baseline_file = Path("../../synthesis_evaluation_results.json")
    baseline_results = {}

    if baseline_file.exists():
        with open(baseline_file, "r") as f:
            data = json.load(f)

            # Extract baseline results
            for task_group in ["known_tasks", "random_sample"]:
                for task in data.get(task_group, []):
                    baseline_results[task["task_id"]] = {
                        "solved": task.get("solved", False),
                        "program": task.get("program"),
                        "test_acc": task.get("test_acc", 0),
                    }

    return baseline_results


def main():
    """Main evaluation function."""
    print("=" * 60)
    print("Enhanced Synthesis System Evaluation")
    print("=" * 60)

    # Load baseline for comparison
    baseline = compare_with_baseline()

    # Run evaluation
    results, solved_count = evaluate_on_failed_tasks()

    # Analysis
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    total_tasks = len(results)
    print(f"\nTasks evaluated: {total_tasks}")
    print(
        f"Tasks solved: {solved_count}/{total_tasks} ({solved_count/total_tasks*100:.1f}%)"
    )

    # Breakdown by synthesizer
    bidirectional_solved = sum(
        1 for r in results if r.get("bidirectional", {}).get("solved", False)
    )
    neural_solved = sum(
        1 for r in results if r.get("neural_guided", {}).get("solved", False)
    )

    print(f"\nBy synthesizer:")
    print(
        f"  Bidirectional: {bidirectional_solved}/{total_tasks} ({bidirectional_solved/total_tasks*100:.1f}%)"
    )
    print(
        f"  Neural-guided: {neural_solved}/{total_tasks} ({neural_solved/total_tasks*100:.1f}%)"
    )

    # Show newly solved tasks
    newly_solved = []
    for result in results:
        task_id = result["task_id"]
        if result.get("solved", False):
            if task_id not in baseline or not baseline[task_id]["solved"]:
                newly_solved.append(task_id)

    if newly_solved:
        print(f"\n🎉 Newly solved tasks ({len(newly_solved)}):")
        for task_id in newly_solved:
            # Find which synthesizer solved it
            result = next(r for r in results if r["task_id"] == task_id)
            if result.get("bidirectional", {}).get("solved"):
                program = result["bidirectional"]["program"]
                method = "bidirectional"
            else:
                program = result["neural_guided"]["program"]
                method = "neural_guided"

            print(f"  - {task_id}: {program} (via {method})")

    # Analyze which new primitives were used
    print("\n📊 Primitive usage in solutions:")
    primitive_usage = {}

    for result in results:
        if result.get("solved", False):
            for synth_name in ["bidirectional", "neural_guided"]:
                if result.get(synth_name, {}).get("solved"):
                    program = result[synth_name]["program"]

                    # Check for new primitives
                    new_primitives = [
                        "DrawLine",
                        "ConnectObjects",
                        "CountObjects",
                        "SortBySize",
                        "PartitionGrid",
                        "ConditionalFill",
                        "PropagatePattern",
                        "ExtractBoundaries",
                        "TraceBorder",
                    ]

                    for prim in new_primitives:
                        if prim in program:
                            primitive_usage[prim] = primitive_usage.get(prim, 0) + 1

    if primitive_usage:
        for prim, count in sorted(
            primitive_usage.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  - {prim}: used in {count} solution(s)")
    else:
        print("  No new primitives used in solutions")

    # Save results
    output_file = Path("enhanced_synthesis_results.json")
    with open(output_file, "w") as f:
        json.dump(
            {
                "total_tasks": total_tasks,
                "solved_count": solved_count,
                "accuracy": solved_count / total_tasks,
                "newly_solved": newly_solved,
                "primitive_usage": primitive_usage,
                "detailed_results": results,
            },
            f,
            indent=2,
        )

    print(f"\n💾 Results saved to {output_file}")

    # Final comparison
    if baseline:
        baseline_solved = sum(1 for b in baseline.values() if b["solved"])
        print(f"\n📈 Improvement over baseline:")
        print(f"  Baseline: {baseline_solved} tasks solved")
        print(f"  Enhanced: {solved_count} tasks solved")
        print(f"  Improvement: +{solved_count - baseline_solved} tasks")


if __name__ == "__main__":
    main()
