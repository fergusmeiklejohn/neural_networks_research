"""Test the invention memory system."""

import numpy as np
from typing import List, Tuple
import time
from pathlib import Path

from invention_memory import InventionMemory, TaskSignature
from primitive_inventor import PrimitiveInventor
from invention_strategies import InventionStrategies


def create_test_examples():
    """Create test examples for memory system."""
    
    examples = []
    
    # Example 1: Simple increment
    input1 = np.array([[1, 2], [3, 4]])
    output1 = np.array([[2, 3], [4, 5]])
    examples.append(("increment", [(input1, output1)]))
    
    # Example 2: Color swap
    input2 = np.array([[1, 2, 1], [2, 1, 2]])
    output2 = np.array([[2, 1, 2], [1, 2, 1]])
    examples.append(("color_swap", [(input2, output2)]))
    
    # Example 3: Diagonal pattern
    input3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    output3 = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    examples.append(("diagonal", [(input3, output3)]))
    
    # Example 4: Border addition
    input4 = np.array([[1, 1], [1, 1]])
    output4 = np.array([[3, 3, 3, 3], [3, 1, 1, 3], [3, 1, 1, 3], [3, 3, 3, 3]])
    examples.append(("border", [(input4, output4)]))
    
    return examples


def test_memory_storage_and_retrieval():
    """Test basic storage and retrieval functionality."""
    
    print("=" * 70)
    print("TESTING MEMORY STORAGE AND RETRIEVAL")
    print("=" * 70)
    
    # Create memory system
    memory = InventionMemory(capacity=100, storage_path=Path("test_memory.json"))
    
    # Get test examples
    test_cases = create_test_examples()
    
    print("\n1. Storing inventions...")
    stored_ids = []
    
    for name, examples in test_cases:
        # Create a simple function for testing
        def test_func(grid):
            return grid + 1  # Simple test function
        
        # Store invention
        inv_id = memory.store(
            name=name,
            program_description=f"Test program for {name}",
            atomic_sequence=["increment_all"],
            function=test_func,
            examples=examples,
            accuracy=0.95,
            invention_time=0.1,
            strategy_used="test_strategy",
            generalization_score=0.8
        )
        
        stored_ids.append(inv_id)
        print(f"  Stored: {name} (ID: {inv_id})")
    
    print(f"\n  Total stored: {len(memory.inventions)}")
    
    print("\n2. Testing exact match retrieval...")
    
    # Try to retrieve exact match
    _, examples = test_cases[0]  # Get increment examples
    retrieved = memory.retrieve(examples, k=1)
    
    if retrieved:
        invention, func = retrieved[0]
        print(f"  ✓ Retrieved exact match: {invention.name}")
        print(f"    Usage count: {invention.usage_count}")
        print(f"    Cache hits: {memory.cache_hits}")
    else:
        print("  ✗ Failed to retrieve exact match")
    
    print("\n3. Testing similarity-based retrieval...")
    
    # Create similar but not exact example
    similar_input = np.array([[2, 3], [4, 5]])
    similar_output = np.array([[3, 4], [5, 6]])
    similar_examples = [(similar_input, similar_output)]
    
    retrieved = memory.retrieve(similar_examples, k=3, min_similarity=0.5)
    
    print(f"  Found {len(retrieved)} similar inventions:")
    for invention, _ in retrieved:
        print(f"    - {invention.name}: similarity={invention.similarity_to_task(memory.extract_task_signature(similar_examples)):.2f}")
    
    return memory


def test_with_real_inventions():
    """Test memory with real primitive inventions."""
    
    print("\n" + "=" * 70)
    print("TESTING WITH REAL PRIMITIVE INVENTIONS")
    print("=" * 70)
    
    # Create memory and inventor
    memory = InventionMemory(capacity=100, storage_path=Path("real_inventions.json"))
    inventor = PrimitiveInventor()
    strategies = InventionStrategies()
    
    # Test Case 1: Value mapping
    print("\n1. Inventing value mapping primitive...")
    input1 = np.array([[1, 2, 3], [4, 5, 6]])
    output1 = np.array([[2, 3, 4], [5, 6, 7]])
    examples1 = [(input1, output1)]
    
    start_time = time.time()
    invented = inventor.invent_primitive(examples1, strategy="trace")
    invention_time = time.time() - start_time
    
    if invented:
        print(f"  ✓ Invented: {invented.program}")
        
        # Store in memory
        inv_id = memory.store(
            name="increment_values",
            program_description=invented.program,
            atomic_sequence=invented.atomic_sequence,
            function=invented.function,
            examples=examples1,
            accuracy=invented.score,
            invention_time=invention_time,
            strategy_used="trace",
            generalization_score=0.9
        )
        print(f"  Stored with ID: {inv_id}")
    
    # Test Case 2: Pattern with geometric reasoning
    print("\n2. Inventing geometric pattern...")
    input2 = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    output2 = np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]])
    examples2 = [(input2, output2)]
    
    start_time = time.time()
    invented = strategies.geometric_reasoning(examples2)
    invention_time = time.time() - start_time
    
    if invented:
        print(f"  ✓ Invented: {invented.program}")
        
        inv_id = memory.store(
            name="cross_pattern",
            program_description=invented.program,
            atomic_sequence=invented.atomic_sequence,
            function=invented.function,
            examples=examples2,
            accuracy=invented.score,
            invention_time=invention_time,
            strategy_used="geometric_reasoning",
            generalization_score=0.85
        )
        print(f"  Stored with ID: {inv_id}")
    
    # Test retrieval
    print("\n3. Testing retrieval of stored inventions...")
    
    # Create a similar value mapping task
    similar_input = np.array([[10, 20], [30, 40]])
    similar_output = np.array([[11, 21], [31, 41]])
    similar_examples = [(similar_input, similar_output)]
    
    retrieved = memory.retrieve(similar_examples, k=2)
    
    if retrieved:
        print(f"  Found {len(retrieved)} relevant inventions:")
        for invention, func in retrieved:
            print(f"    - {invention.name} ({invention.strategy_used})")
            
            # Test if the function works
            result = func(similar_input)
            if np.array_equal(result, similar_output):
                print(f"      ✓ Function works correctly!")
                memory.update_success(invention.invention_id, True)
            else:
                print(f"      ✗ Function doesn't match expected output")
                memory.update_success(invention.invention_id, False)
    
    return memory


def test_memory_persistence():
    """Test saving and loading memory."""
    
    print("\n" + "=" * 70)
    print("TESTING MEMORY PERSISTENCE")
    print("=" * 70)
    
    # Create and populate memory
    print("\n1. Creating and saving memory...")
    memory1 = InventionMemory(storage_path=Path("persistence_test.json"))
    
    # Add some inventions
    test_cases = create_test_examples()
    for name, examples in test_cases[:2]:
        memory1.store(
            name=name,
            program_description=f"Program for {name}",
            atomic_sequence=[f"op_{name}"],
            function=lambda g: g + 1,
            examples=examples,
            accuracy=0.9,
            invention_time=0.1,
            strategy_used="test",
            generalization_score=0.8
        )
    
    stats_before = memory1.get_statistics()
    print(f"  Stored {stats_before['total_inventions']} inventions")
    
    # Save
    memory1.save()
    print("  Memory saved to disk")
    
    # Create new memory and load
    print("\n2. Loading memory from disk...")
    memory2 = InventionMemory(storage_path=Path("persistence_test.json"))
    
    stats_after = memory2.get_statistics()
    print(f"  Loaded {stats_after['total_inventions']} inventions")
    
    # Verify contents
    if stats_before['total_inventions'] == stats_after['total_inventions']:
        print("  ✓ Memory successfully persisted and restored!")
    else:
        print("  ✗ Memory persistence failed")
    
    # Clean up
    Path("persistence_test.json").unlink(missing_ok=True)
    Path("persistence_test.pkl").unlink(missing_ok=True)
    
    return memory2


def test_memory_statistics():
    """Test memory statistics and analysis."""
    
    print("\n" + "=" * 70)
    print("TESTING MEMORY STATISTICS")
    print("=" * 70)
    
    memory = test_with_real_inventions()
    
    print("\n4. Memory Statistics:")
    stats = memory.get_statistics()
    
    print(f"  Total inventions: {stats['total_inventions']}")
    print(f"  Total stored: {stats['total_stored']}")
    print(f"  Total retrieved: {stats['total_retrieved']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
    
    print("\n  By transformation type:")
    for trans_type, count in stats['by_type'].items():
        print(f"    - {trans_type}: {count}")
    
    print("\n  By strategy:")
    for strategy, count in stats['by_strategy'].items():
        print(f"    - {strategy}: {count}")
    
    print(f"\n  Average accuracy: {stats['avg_accuracy']:.1%}")
    print(f"  Average operation count: {stats['avg_operation_count']:.1f}")
    print(f"  Average generalization: {stats['avg_generalization']:.1%}")


if __name__ == "__main__":
    # Run all tests
    test_memory_storage_and_retrieval()
    test_with_real_inventions()
    test_memory_persistence()
    test_memory_statistics()
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)
    
    # Clean up test files
    for f in ["test_memory.json", "test_memory.pkl", "real_inventions.json", "real_inventions.pkl"]:
        Path(f).unlink(missing_ok=True)