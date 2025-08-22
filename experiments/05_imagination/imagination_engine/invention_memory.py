"""Invention Memory System - Stores and retrieves successful primitive inventions.

This module enables the system to learn from past inventions, building a growing
library of discovered primitives that can be reused and adapted for new tasks.

Key Features:
- Store inventions with rich metadata
- Retrieve similar inventions using multiple similarity metrics
- Track usage patterns and success rates
- Compose simpler inventions into complex ones
- Persistent storage for learned knowledge
"""

import json
import pickle
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Callable
import numpy as np
from collections import defaultdict
import hashlib


@dataclass
class TaskSignature:
    """Signature of a task for similarity matching."""
    input_shape: Tuple[int, int]
    output_shape: Tuple[int, int]
    input_colors: Tuple[int, ...]
    output_colors: Tuple[int, ...]
    transformation_type: str  # e.g., "geometric", "color_mapping", "pattern"
    has_objects: bool
    has_patterns: bool
    has_symmetry: bool
    
    def to_vector(self) -> np.ndarray:
        """Convert signature to numerical vector for similarity computation."""
        features = []
        
        # Shape features
        features.extend(self.input_shape)
        features.extend(self.output_shape)
        features.append(self.output_shape[0] / self.input_shape[0])  # height ratio
        features.append(self.output_shape[1] / self.input_shape[1])  # width ratio
        
        # Color features
        features.append(len(self.input_colors))
        features.append(len(self.output_colors))
        features.append(len(set(self.output_colors) - set(self.input_colors)))  # new colors
        
        # Transformation features (one-hot encoding)
        trans_types = ["geometric", "color_mapping", "pattern", "compositional", "other"]
        for t in trans_types:
            features.append(1.0 if self.transformation_type == t else 0.0)
        
        # Boolean features
        features.append(1.0 if self.has_objects else 0.0)
        features.append(1.0 if self.has_patterns else 0.0)
        features.append(1.0 if self.has_symmetry else 0.0)
        
        return np.array(features, dtype=np.float32)


@dataclass
class StoredInvention:
    """A stored primitive invention with metadata."""
    
    # Core invention data
    invention_id: str
    name: str
    program_description: str
    atomic_sequence: List[str]
    invention_time: float
    strategy_used: str  # e.g., "geometric_reasoning", "pattern_decomposition"
    
    # Task context
    task_signature: TaskSignature
    examples_hash: str  # Hash of examples for exact matching
    
    # Performance metrics
    accuracy: float
    operation_count: int
    generalization_score: float  # How well it generalizes to variations
    
    # Usage statistics
    usage_count: int = 0
    success_count: int = 0
    last_used: float = field(default_factory=time.time)
    
    # The actual function (stored separately due to pickling)
    _function: Optional[Callable] = field(default=None, repr=False)
    
    def similarity_to_task(self, other_signature: TaskSignature) -> float:
        """Compute similarity between this invention's task and another task."""
        vec1 = self.task_signature.to_vector()
        vec2 = other_signature.to_vector()
        
        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norm_product == 0:
            return 0.0
        
        return float(dot_product / norm_product)
    
    def efficiency_score(self) -> float:
        """Score based on efficiency (fewer operations is better)."""
        # Inverse of operation count, normalized
        return 1.0 / (1.0 + self.operation_count / 10.0)
    
    def overall_score(self, task_signature: Optional[TaskSignature] = None) -> float:
        """Compute overall score for ranking."""
        score = self.accuracy * 0.4  # Accuracy weight
        score += self.generalization_score * 0.3  # Generalization weight
        score += self.efficiency_score() * 0.2  # Efficiency weight
        score += min(self.success_count / max(self.usage_count, 1), 1.0) * 0.1  # Success rate
        
        if task_signature:
            score *= self.similarity_to_task(task_signature)
        
        return score


class InventionMemory:
    """Memory system for storing and retrieving primitive inventions."""
    
    def __init__(
        self, 
        capacity: int = 1000,
        similarity_threshold: float = 0.7,
        storage_path: Optional[Path] = None
    ):
        """Initialize the invention memory.
        
        Args:
            capacity: Maximum number of inventions to store
            similarity_threshold: Minimum similarity for retrieval
            storage_path: Path for persistent storage
        """
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self.storage_path = storage_path or Path("invention_memory.pkl")
        
        # Main storage
        self.inventions: List[StoredInvention] = []
        self.invention_functions: Dict[str, Callable] = {}  # Separate function storage
        
        # Indices for fast retrieval
        self.task_type_index: Dict[str, List[StoredInvention]] = defaultdict(list)
        self.strategy_index: Dict[str, List[StoredInvention]] = defaultdict(list)
        self.exact_match_index: Dict[str, StoredInvention] = {}  # examples_hash -> invention
        
        # Statistics
        self.total_stored = 0
        self.total_retrieved = 0
        self.cache_hits = 0
        self.adaptations = 0
        
        # Load existing memory if available
        self.load()
    
    def extract_task_signature(
        self, 
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> TaskSignature:
        """Extract task signature from examples."""
        if not examples:
            raise ValueError("No examples provided")
        
        input_grid = examples[0][0]
        output_grid = examples[0][1]
        
        # Collect all colors
        input_colors = set()
        output_colors = set()
        for inp, out in examples:
            input_colors.update(np.unique(inp))
            output_colors.update(np.unique(out))
        
        # Detect transformation type
        trans_type = self._detect_transformation_type(examples)
        
        # Detect features
        has_objects = self._has_distinct_objects(input_grid)
        has_patterns = self._has_repeating_patterns(input_grid)
        has_symmetry = self._has_symmetry(input_grid)
        
        return TaskSignature(
            input_shape=input_grid.shape,
            output_shape=output_grid.shape,
            input_colors=tuple(sorted(input_colors)),
            output_colors=tuple(sorted(output_colors)),
            transformation_type=trans_type,
            has_objects=has_objects,
            has_patterns=has_patterns,
            has_symmetry=has_symmetry
        )
    
    def _detect_transformation_type(
        self, 
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> str:
        """Detect the type of transformation."""
        input_grid = examples[0][0]
        output_grid = examples[0][1]
        
        # Check for size change
        if input_grid.shape != output_grid.shape:
            return "geometric"
        
        # Check for color changes
        if set(np.unique(input_grid)) != set(np.unique(output_grid)):
            return "color_mapping"
        
        # Check for pattern repetition
        if output_grid.size > input_grid.size * 2:
            return "pattern"
        
        # Default to compositional
        return "compositional"
    
    def _has_distinct_objects(self, grid: np.ndarray) -> bool:
        """Check if grid contains distinct objects."""
        from scipy.ndimage import label
        unique_colors = [c for c in np.unique(grid) if c != 0]
        
        for color in unique_colors:
            binary = (grid == color).astype(int)
            labeled, num_features = label(binary)
            if num_features > 0:
                return True
        return False
    
    def _has_repeating_patterns(self, grid: np.ndarray) -> bool:
        """Check for repeating patterns in the grid."""
        h, w = grid.shape
        
        # Check for simple repetitions (2x2, 3x3 patterns)
        for pattern_size in [2, 3]:
            if h >= pattern_size * 2 and w >= pattern_size * 2:
                pattern = grid[:pattern_size, :pattern_size]
                
                # Check if pattern repeats
                for i in range(0, h - pattern_size + 1, pattern_size):
                    for j in range(0, w - pattern_size + 1, pattern_size):
                        if not np.array_equal(grid[i:i+pattern_size, j:j+pattern_size], pattern):
                            break
                    else:
                        continue
                    break
                else:
                    return True
        
        return False
    
    def _has_symmetry(self, grid: np.ndarray) -> bool:
        """Check for symmetry in the grid."""
        # Horizontal symmetry
        if np.array_equal(grid, np.flip(grid, axis=1)):
            return True
        
        # Vertical symmetry
        if np.array_equal(grid, np.flip(grid, axis=0)):
            return True
        
        # Diagonal symmetry
        if grid.shape[0] == grid.shape[1]:
            if np.array_equal(grid, grid.T):
                return True
        
        return False
    
    def _compute_examples_hash(
        self, 
        examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> str:
        """Compute hash of examples for exact matching."""
        hasher = hashlib.md5()
        for inp, out in examples:
            hasher.update(inp.tobytes())
            hasher.update(out.tobytes())
        return hasher.hexdigest()
    
    def store(
        self,
        name: str,
        program_description: str,
        atomic_sequence: List[str],
        function: Callable,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        accuracy: float,
        invention_time: float,
        strategy_used: str,
        generalization_score: float = 0.5
    ) -> str:
        """Store a new invention.
        
        Returns:
            invention_id: Unique ID of the stored invention
        """
        # Generate ID
        invention_id = f"inv_{self.total_stored}_{int(time.time())}"
        
        # Extract task signature
        task_signature = self.extract_task_signature(examples)
        examples_hash = self._compute_examples_hash(examples)
        
        # Check for exact match
        if examples_hash in self.exact_match_index:
            existing = self.exact_match_index[examples_hash]
            # Update if this one is better
            if accuracy > existing.accuracy or len(atomic_sequence) < existing.operation_count:
                self._remove_invention(existing.invention_id)
            else:
                return existing.invention_id
        
        # Create stored invention
        invention = StoredInvention(
            invention_id=invention_id,
            name=name,
            program_description=program_description,
            atomic_sequence=atomic_sequence,
            invention_time=invention_time,
            strategy_used=strategy_used,
            task_signature=task_signature,
            examples_hash=examples_hash,
            accuracy=accuracy,
            operation_count=len(atomic_sequence),
            generalization_score=generalization_score
        )
        
        # Store
        self.inventions.append(invention)
        self.invention_functions[invention_id] = function
        
        # Update indices
        self.task_type_index[task_signature.transformation_type].append(invention)
        self.strategy_index[strategy_used].append(invention)
        self.exact_match_index[examples_hash] = invention
        
        self.total_stored += 1
        
        # Evict if over capacity (LRU)
        if len(self.inventions) > self.capacity:
            self._evict_lru()
        
        # Auto-save periodically
        if self.total_stored % 10 == 0:
            self.save()
        
        return invention_id
    
    def retrieve(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        k: int = 5,
        min_similarity: Optional[float] = None
    ) -> List[Tuple[StoredInvention, Callable]]:
        """Retrieve k most relevant inventions for the given examples.
        
        Returns:
            List of (invention, function) tuples sorted by relevance
        """
        if not self.inventions:
            return []
        
        min_similarity = min_similarity or self.similarity_threshold
        
        # Check for exact match first
        examples_hash = self._compute_examples_hash(examples)
        if examples_hash in self.exact_match_index:
            self.cache_hits += 1
            invention = self.exact_match_index[examples_hash]
            invention.usage_count += 1
            invention.last_used = time.time()
            func = self.invention_functions.get(invention.invention_id)
            if func:
                return [(invention, func)]
        
        # Extract task signature
        task_signature = self.extract_task_signature(examples)
        
        # Score all inventions
        candidates = []
        for invention in self.inventions:
            similarity = invention.similarity_to_task(task_signature)
            if similarity >= min_similarity:
                score = invention.overall_score(task_signature)
                candidates.append((score, invention))
        
        # Sort by score and return top k
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, invention in candidates[:k]:
            func = self.invention_functions.get(invention.invention_id)
            if func:
                invention.usage_count += 1
                invention.last_used = time.time()
                results.append((invention, func))
        
        self.total_retrieved += len(results)
        return results
    
    def update_success(self, invention_id: str, success: bool):
        """Update success statistics for an invention."""
        for inv in self.inventions:
            if inv.invention_id == invention_id:
                if success:
                    inv.success_count += 1
                break
    
    def _remove_invention(self, invention_id: str):
        """Remove an invention from memory."""
        # Remove from main list
        self.inventions = [inv for inv in self.inventions if inv.invention_id != invention_id]
        
        # Remove from indices
        for inv_list in self.task_type_index.values():
            inv_list[:] = [inv for inv in inv_list if inv.invention_id != invention_id]
        
        for inv_list in self.strategy_index.values():
            inv_list[:] = [inv for inv in inv_list if inv.invention_id != invention_id]
        
        # Remove function
        if invention_id in self.invention_functions:
            del self.invention_functions[invention_id]
    
    def _evict_lru(self):
        """Evict least recently used invention."""
        if not self.inventions:
            return
        
        # Find LRU invention
        lru_invention = min(self.inventions, key=lambda x: x.last_used)
        self._remove_invention(lru_invention.invention_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = {
            "total_inventions": len(self.inventions),
            "total_stored": self.total_stored,
            "total_retrieved": self.total_retrieved,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(self.total_retrieved, 1),
            "by_type": {},
            "by_strategy": {},
            "avg_accuracy": 0.0,
            "avg_operation_count": 0.0,
            "avg_generalization": 0.0
        }
        
        # Count by type
        for trans_type, invs in self.task_type_index.items():
            stats["by_type"][trans_type] = len(invs)
        
        # Count by strategy
        for strategy, invs in self.strategy_index.items():
            stats["by_strategy"][strategy] = len(invs)
        
        # Compute averages
        if self.inventions:
            stats["avg_accuracy"] = np.mean([inv.accuracy for inv in self.inventions])
            stats["avg_operation_count"] = np.mean([inv.operation_count for inv in self.inventions])
            stats["avg_generalization"] = np.mean([inv.generalization_score for inv in self.inventions])
        
        return stats
    
    def save(self):
        """Save memory to disk."""
        try:
            # Prepare data for saving (exclude functions)
            save_data = {
                "inventions": [asdict(inv) for inv in self.inventions],
                "statistics": {
                    "total_stored": self.total_stored,
                    "total_retrieved": self.total_retrieved,
                    "cache_hits": self.cache_hits
                }
            }
            
            # Save metadata as JSON
            json_path = self.storage_path.with_suffix(".json")
            with open(json_path, "w") as f:
                json.dump(save_data, f, indent=2, default=str)
            
            # Save functions separately as pickle
            if self.invention_functions:
                func_path = self.storage_path.with_suffix(".pkl")
                with open(func_path, "wb") as f:
                    pickle.dump(self.invention_functions, f)
            
            print(f"Saved {len(self.inventions)} inventions to {self.storage_path}")
            
        except Exception as e:
            print(f"Error saving memory: {e}")
    
    def load(self):
        """Load memory from disk."""
        try:
            json_path = self.storage_path.with_suffix(".json")
            if not json_path.exists():
                return
            
            # Load metadata
            with open(json_path, "r") as f:
                save_data = json.load(f)
            
            # Reconstruct inventions
            self.inventions = []
            for inv_dict in save_data["inventions"]:
                # Reconstruct TaskSignature
                sig_dict = inv_dict.pop("task_signature")
                task_signature = TaskSignature(**sig_dict)
                
                # Remove function field
                inv_dict.pop("_function", None)
                
                # Create invention
                invention = StoredInvention(
                    task_signature=task_signature,
                    **inv_dict
                )
                self.inventions.append(invention)
                
                # Update indices
                self.task_type_index[task_signature.transformation_type].append(invention)
                self.strategy_index[invention.strategy_used].append(invention)
                self.exact_match_index[invention.examples_hash] = invention
            
            # Load functions if available
            func_path = self.storage_path.with_suffix(".pkl")
            if func_path.exists():
                with open(func_path, "rb") as f:
                    self.invention_functions = pickle.load(f)
            
            # Restore statistics
            stats = save_data.get("statistics", {})
            self.total_stored = stats.get("total_stored", len(self.inventions))
            self.total_retrieved = stats.get("total_retrieved", 0)
            self.cache_hits = stats.get("cache_hits", 0)
            
            print(f"Loaded {len(self.inventions)} inventions from {self.storage_path}")
            
        except Exception as e:
            print(f"Error loading memory: {e}")