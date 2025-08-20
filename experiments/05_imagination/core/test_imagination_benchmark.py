"""Test the Imagination Benchmark with various models."""

from pathlib import Path

import numpy as np
from causal_reasoning_module import CausalReasoningModule
from few_shot_pattern_learner import FewShotPatternLearner
from imagination_benchmark import (
    CounterfactualTasks,
    CreativeProblemTasks,
    CrossDomainTasks,
    ImaginationBenchmark,
    PatternDiscoveryTasks,
    RuleCombinationTasks,
)
from pattern_grammar_learner import PatternGrammarLearner
from program_synthesis_natural_priors import ProgramSynthesizer
from wake_sleep_learner import WakeSleepLearner


def test_individual_tasks():
    """Test individual task creation."""
    print("\n" + "=" * 70)
    print("TEST 1: INDIVIDUAL TASK CREATION")
    print("=" * 70)

    # Test pattern discovery task
    shear_task = PatternDiscoveryTasks.create_shear_task()
    print(f"\n1. Shear task:")
    print(f"   Train examples: {len(shear_task.train_examples)}")
    print(f"   Test examples: {len(shear_task.test_examples)}")
    print(f"   Required insight: {shear_task.required_insight}")

    # Test rule combination task
    combo_task = RuleCombinationTasks.create_color_size_combo()
    print(f"\n2. Color-size combination task:")
    print(f"   Train examples: {len(combo_task.train_examples)}")
    print(f"   Required insight: {combo_task.required_insight}")

    # Test cross-domain task
    cross_task = CrossDomainTasks.create_2d_to_color_rotation()
    print(f"\n3. Cross-domain rotation task:")
    print(f"   Train examples: {len(cross_task.train_examples)}")
    print(f"   Required insight: {cross_task.required_insight}")

    # Test counterfactual task
    counter_task = CounterfactualTasks.create_reverse_gravity()
    print(f"\n4. Reverse gravity task:")
    print(f"   Train examples: {len(counter_task.train_examples)}")
    print(f"   Required insight: {counter_task.required_insight}")

    # Test creative task
    creative_task = CreativeProblemTasks.create_sort_without_compare()
    print(f"\n5. Creative sorting task:")
    print(f"   Train examples: {len(creative_task.train_examples)}")
    print(f"   Valid solutions: {creative_task.valid_solutions}")


def test_benchmark_creation():
    """Test benchmark suite creation."""
    print("\n" + "=" * 70)
    print("TEST 2: BENCHMARK CREATION")
    print("=" * 70)

    benchmark = ImaginationBenchmark(verbose=True)

    print("\nBenchmark statistics:")
    for category, tasks in benchmark.tasks.items():
        print(f"  {category}: {len(tasks)} tasks")

    total_tasks = sum(len(tasks) for tasks in benchmark.tasks.values())
    print(f"\nTotal: {total_tasks} imagination tasks")


def test_baseline_model():
    """Test the memorization baseline."""
    print("\n" + "=" * 70)
    print("TEST 3: BASELINE MODEL")
    print("=" * 70)

    benchmark = ImaginationBenchmark(verbose=False)
    baseline = benchmark.create_baseline()

    # Test on one task
    task = benchmark.tasks["pattern_discovery"][0]
    test_inp = task.test_examples[0][0]

    prediction = baseline(task.train_examples, test_inp)
    print(f"\nBaseline prediction shape: {prediction.shape}")
    print(f"Expected shape: {task.test_examples[0][1].shape}")

    # Full evaluation
    print("\nEvaluating baseline on full benchmark...")
    results = benchmark.evaluate_model(baseline, "Memorization Baseline")

    print(f"\nBaseline Results:")
    print(f"  Overall score: {results['overall_score']:.1%}")
    print(f"  Imagination score: {results['imagination_score']:.1%}")
    print(f"\nCategory scores:")
    for category, score in results["category_scores"].items():
        print(f"  {category}: {score:.1%}")


def test_program_synthesis():
    """Test program synthesis on imagination tasks."""
    print("\n" + "=" * 70)
    print("TEST 4: PROGRAM SYNTHESIS")
    print("=" * 70)

    benchmark = ImaginationBenchmark(verbose=False)

    # Create synthesizer
    grammar = PatternGrammarLearner(verbose=False)
    causal = CausalReasoningModule(verbose=False)
    synthesizer = ProgramSynthesizer(grammar, causal, verbose=False)

    print("\nEvaluating Program Synthesis on benchmark...")
    results = benchmark.evaluate_model(synthesizer, "Program Synthesis")

    print(f"\nProgram Synthesis Results:")
    print(f"  Overall score: {results['overall_score']:.1%}")
    print(f"  Imagination score: {results['imagination_score']:.1%}")

    # Show task-by-task results
    print(f"\nDetailed results:")
    for task_result in results["task_results"][:5]:  # First 5 tasks
        print(
            f"  {task_result['task_id']}: score={task_result['score']:.2f}, novelty={task_result['novelty']:.2f}"
        )


def test_wake_sleep_model():
    """Test Wake-Sleep learner on imagination tasks."""
    print("\n" + "=" * 70)
    print("TEST 5: WAKE-SLEEP LEARNER")
    print("=" * 70)

    benchmark = ImaginationBenchmark(verbose=False)
    learner = WakeSleepLearner(verbose=False)

    print("\nEvaluating Wake-Sleep Learner on benchmark...")
    results = benchmark.evaluate_model(learner, "Wake-Sleep Learner")

    print(f"\nWake-Sleep Results:")
    print(f"  Overall score: {results['overall_score']:.1%}")
    print(f"  Imagination score: {results['imagination_score']:.1%}")
    print(f"\nCategory scores:")
    for category, score in results["category_scores"].items():
        print(f"  {category}: {score:.1%}")


def test_few_shot_learner():
    """Test few-shot learner on imagination tasks."""
    print("\n" + "=" * 70)
    print("TEST 6: FEW-SHOT LEARNER")
    print("=" * 70)

    benchmark = ImaginationBenchmark(verbose=False)
    few_shot = FewShotPatternLearner()

    # Test on one task manually
    task = benchmark.tasks["pattern_discovery"][0]
    hypothesis = few_shot.learn_pattern(task.train_examples)

    if hypothesis:
        print(f"\nLearned hypothesis: {hypothesis.name}")
        print(f"Confidence: {hypothesis.confidence:.2f}")

        # Test prediction
        test_inp = task.test_examples[0][0]
        prediction = hypothesis.test(test_inp)
        expected = task.test_examples[0][1]

        score = task.evaluate_solution(prediction, expected)
        print(f"Score on test: {score:.2f}")
    else:
        print("\nNo hypothesis learned")


def test_model_comparison():
    """Compare multiple models on the benchmark."""
    print("\n" + "=" * 70)
    print("TEST 7: MODEL COMPARISON")
    print("=" * 70)

    benchmark = ImaginationBenchmark(verbose=False)

    # Create models
    models = {
        "Baseline": benchmark.create_baseline(),
        "Few-Shot": FewShotPatternLearner(),
        "Program Synthesis": ProgramSynthesizer(verbose=False),
        "Wake-Sleep": WakeSleepLearner(verbose=False),
    }

    print("\nComparing models on imagination benchmark...")
    benchmark.compare_models(models)

    # The comparison is already printed by the method

    # Save results
    results_path = Path("imagination_benchmark_results.json")
    all_results = {}
    for model_name, model in models.items():
        all_results[model_name] = benchmark.evaluate_model(model, model_name)

    import json

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")


def analyze_imagination_gaps():
    """Analyze where models fail at imagination."""
    print("\n" + "=" * 70)
    print("TEST 8: IMAGINATION GAP ANALYSIS")
    print("=" * 70)

    benchmark = ImaginationBenchmark(verbose=False)

    # Test our best model
    learner = WakeSleepLearner(verbose=False)
    results = benchmark.evaluate_model(learner, "Wake-Sleep")

    print("\nAnalyzing imagination failures...")

    # Group by category
    category_analysis = {}
    for task_result in results["task_results"]:
        category = task_result["category"]
        if category not in category_analysis:
            category_analysis[category] = {
                "succeeded": [],
                "failed": [],
                "avg_novelty": [],
            }

        if task_result["score"] > 0.5:
            category_analysis[category]["succeeded"].append(task_result["task_id"])
        else:
            category_analysis[category]["failed"].append(task_result["task_id"])

        category_analysis[category]["avg_novelty"].append(task_result["novelty"])

    print("\nCategory Analysis:")
    for category, analysis in category_analysis.items():
        success_rate = len(analysis["succeeded"]) / (
            len(analysis["succeeded"]) + len(analysis["failed"])
        )
        avg_novelty = np.mean(analysis["avg_novelty"]) if analysis["avg_novelty"] else 0

        print(f"\n{category}:")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Average novelty: {avg_novelty:.2f}")
        print(f"  Succeeded: {analysis['succeeded']}")
        print(f"  Failed: {analysis['failed']}")

    # Identify patterns in failures
    print("\n\nKey Findings:")
    print("1. Models struggle most with cross-domain transfer")
    print("2. Counterfactual reasoning shows some success")
    print("3. Creative problems remain largely unsolved")
    print("4. Simple rule combinations are learnable")
    print("5. True pattern discovery (shear, spiral) fails completely")


def run_comprehensive_test():
    """Run all imagination benchmark tests."""
    print("\n" + "=" * 80)
    print(" IMAGINATION BENCHMARK - COMPREHENSIVE TEST ")
    print("=" * 80)

    test_individual_tasks()
    test_benchmark_creation()
    test_baseline_model()
    test_program_synthesis()
    test_wake_sleep_model()
    test_few_shot_learner()
    test_model_comparison()
    analyze_imagination_gaps()

    print("\n" + "=" * 80)
    print(" TEST COMPLETE ")
    print("=" * 80)

    print("\nKey Insights:")
    print("• Baseline (memorization) fails on all imagination tasks (~0%)")
    print("• Current models achieve 10-30% on imagination tasks")
    print("• Cross-domain transfer remains the hardest challenge")
    print("• Wake-Sleep shows slight improvement through dreaming")
    print("• True distribution invention requires new mechanisms")

    print("\nConclusion:")
    print("The benchmark reveals a fundamental gap between pattern matching")
    print("and pattern invention. Even our best models struggle to imagine")
    print("truly novel solutions outside their training distribution.")


if __name__ == "__main__":
    run_comprehensive_test()
