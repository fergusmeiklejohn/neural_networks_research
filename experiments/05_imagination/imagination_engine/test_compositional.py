"""Test Compositional Reasoner on rule combination tasks.

This specifically targets the rule combination tasks where we had 0% success,
testing whether compositional reasoning can handle multi-attribute rules.
"""

import sys
from pathlib import Path

import numpy as np

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from compositional_reasoner import (
    Attribute,
    AttributeCondition,
    CompositeRule,
    CompositionalReasoner,
    LogicalOperator,
)
from core.imagination_benchmark import RuleCombinationTasks


def test_color_size_combo():
    """Test on color-size combination task."""
    print("\n" + "=" * 60)
    print("Testing Color-Size Combination")
    print("=" * 60)
    
    # Get the benchmark task
    task = RuleCombinationTasks.create_color_size_combo()
    
    print(f"Task: {task.task_id}")
    print(f"Required insight: {task.required_insight}")
    print(f"Examples: {len(task.test_examples)}")
    
    # Create compositional reasoner
    reasoner = CompositionalReasoner()
    
    # Learn rule from examples
    rule = reasoner.learn_rule_from_examples(task.test_examples[:2])
    
    if rule:
        print(f"\n✅ Learned rule: {rule.name}")
        print(reasoner.explain_rule(rule))
        
        # Test on remaining example
        if len(task.test_examples) > 2:
            test_input, test_output = task.test_examples[2]
            predicted = reasoner.apply_rule(rule, test_input)
            
            score = task.evaluate_solution(predicted, test_output)
            print(f"\nTest score: {score:.1%}")
            
            if score > 0.5:
                print("✅ Successfully learned color-size combination!")
                return True
            else:
                print("❌ Rule didn't generalize")
                print(f"Input:\n{test_input}")
                print(f"Expected:\n{test_output}")
                print(f"Predicted:\n{predicted}")
    else:
        print("❌ Failed to learn rule")
    
    return False


def test_conditional_combo():
    """Test on conditional combination task."""
    print("\n" + "=" * 60)
    print("Testing Conditional Combination")
    print("=" * 60)
    
    # Get the benchmark task
    task = RuleCombinationTasks.create_conditional_combo()
    
    print(f"Task: {task.task_id}")
    print(f"Required insight: {task.required_insight}")
    
    # Create compositional reasoner
    reasoner = CompositionalReasoner()
    
    # Try to discover conditional rule
    rule = reasoner.discover_conditional_rule(task.test_examples)
    
    if rule:
        print(f"\n✅ Discovered conditional rule: {rule.name}")
        print(reasoner.explain_rule(rule))
        
        # Test on examples
        total_score = 0
        for i, (inp, out) in enumerate(task.test_examples):
            predicted = reasoner.apply_rule(rule, inp)
            score = task.evaluate_solution(predicted, out)
            total_score += score
            print(f"Example {i+1} score: {score:.1%}")
        
        avg_score = total_score / len(task.test_examples)
        print(f"\nAverage score: {avg_score:.1%}")
        
        if avg_score > 0.5:
            print("✅ Successfully learned conditional rule!")
            return True
    else:
        print("❌ Failed to discover conditional rule")
    
    return False


def test_manual_rule_creation():
    """Test creating rules manually."""
    print("\n" + "=" * 60)
    print("Testing Manual Rule Creation")
    print("=" * 60)
    
    # Create a simple rule: If color=1, then change to color=2
    condition = AttributeCondition(
        Attribute.COLOR,
        LogicalOperator.EQUALS,
        1
    )
    
    def change_color(val):
        return 2
    
    rule = CompositeRule(
        name="color_1_to_2",
        conditions=[condition],
        actions=[(Attribute.COLOR, change_color)],
        logical_op=LogicalOperator.IF_THEN
    )
    
    # Test the rule
    test_obj = {
        "color": 1,
        "size": 3,
        "position": (2, 2)
    }
    
    transformed = rule.apply(test_obj)
    
    print(f"Original: {test_obj}")
    print(f"Transformed: {transformed}")
    
    if transformed["color"] == 2:
        print("✅ Manual rule works correctly!")
        return True
    else:
        print("❌ Manual rule failed")
        return False


def test_rule_composition():
    """Test composing multiple rules."""
    print("\n" + "=" * 60)
    print("Testing Rule Composition")
    print("=" * 60)
    
    reasoner = CompositionalReasoner()
    
    # Create two simple rules
    rule1 = CompositeRule(
        name="increase_size",
        conditions=[],
        actions=[(Attribute.SIZE, lambda x: x * 2)],
    )
    
    rule2 = CompositeRule(
        name="shift_color",
        conditions=[],
        actions=[(Attribute.COLOR, lambda x: (x + 1) % 10)],
    )
    
    # Compose them
    composed = reasoner.compose_rules([rule1, rule2])
    
    print(f"Composed rule: {composed.name}")
    print(reasoner.explain_rule(composed))
    
    # Test composition
    test_obj = {
        "color": 5,
        "size": 3,
    }
    
    transformed = composed.apply(test_obj)
    
    print(f"\nOriginal: {test_obj}")
    print(f"Transformed: {transformed}")
    
    if transformed["size"] == 6 and transformed["color"] == 6:
        print("✅ Rule composition works!")
        return True
    else:
        print("❌ Rule composition failed")
        return False


def test_object_extraction():
    """Test object extraction from grids."""
    print("\n" + "=" * 60)
    print("Testing Object Extraction")
    print("=" * 60)
    
    reasoner = CompositionalReasoner()
    
    # Create a simple grid with two objects
    grid = np.array([
        [1, 1, 0, 0, 0],
        [1, 1, 0, 3, 3],
        [0, 0, 0, 3, 3],
        [0, 2, 2, 0, 0],
        [0, 2, 2, 0, 0]
    ])
    
    objects = reasoner.extract_objects(grid)
    
    print(f"Grid:\n{grid}")
    print(f"\nFound {len(objects)} objects:")
    
    for i, obj in enumerate(objects):
        print(f"\nObject {i+1}:")
        print(f"  Color: {obj.get('color')}")
        print(f"  Size: {obj.get('size')}")
        print(f"  Position: {obj.get('position')}")
    
    if len(objects) == 3:
        print("\n✅ Object extraction works correctly!")
        return True
    else:
        print(f"\n❌ Expected 3 objects, found {len(objects)}")
        return False


def main():
    """Run all compositional reasoning tests."""
    print("=" * 60)
    print("COMPOSITIONAL REASONING TESTS")
    print("=" * 60)
    
    results = []
    
    # Test basic functionality
    print("\n🔬 Test 1: Object Extraction")
    results.append(("Object Extraction", test_object_extraction()))
    
    print("\n🔬 Test 2: Manual Rule Creation")
    results.append(("Manual Rule", test_manual_rule_creation()))
    
    print("\n🔬 Test 3: Rule Composition")
    results.append(("Rule Composition", test_rule_composition()))
    
    # Test on benchmark tasks
    print("\n🔬 Test 4: Color-Size Combination")
    results.append(("Color-Size Combo", test_color_size_combo()))
    
    print("\n🔬 Test 5: Conditional Combination")
    results.append(("Conditional Combo", test_conditional_combo()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    success_count = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {success_count}/{total} tests passed ({success_count/total:.0%})")
    
    if success_count >= 3:
        print("\n🎉 Compositional Reasoner shows promise!")
        print("   Basic functionality works, needs refinement for complex tasks")
    else:
        print("\n📝 Compositional reasoning needs more work")


if __name__ == "__main__":
    main()