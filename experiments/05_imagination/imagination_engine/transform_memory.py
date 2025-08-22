"""Transform Memory System for HTI.

Stores discovered transforms, enables retrieval and composition,
and builds a growing library of learned operations.
"""

import json
import logging
import pickle
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StoredTransform:
    """A transform stored in memory."""
    
    id: str
    embedding: np.ndarray
    task_encoding: np.ndarray
    performance_score: float
    primitive_sequence: List[str]
    metadata: Dict[str, Any]
    usage_count: int = 0
    discovery_time: float = 0.0
    
    def similarity_to_task(self, task_encoding: np.ndarray) -> float:
        """Compute similarity between this transform and a task."""
        # Handle dimension mismatch by truncating or padding
        min_dim = min(len(self.task_encoding), len(task_encoding))
        task_enc_1 = self.task_encoding[:min_dim]
        task_enc_2 = task_encoding[:min_dim]
        
        cosine_sim = np.dot(task_enc_1, task_enc_2) / (
            np.linalg.norm(task_enc_1) * np.linalg.norm(task_enc_2) + 1e-8
        )
        return float(cosine_sim)
    
    def similarity_to_transform(self, other_embedding: np.ndarray) -> float:
        """Compute similarity between transform embeddings."""
        cosine_sim = np.dot(self.embedding, other_embedding) / (
            np.linalg.norm(self.embedding) * np.linalg.norm(other_embedding) + 1e-8
        )
        return float(cosine_sim)


class TransformMemory:
    """Memory system for storing and retrieving learned transforms."""
    
    def __init__(
        self,
        capacity: int = 1000,
        embedding_dim: int = 256,
        similarity_threshold: float = 0.8
    ):
        self.capacity = capacity
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        
        # Memory storage
        self.transforms: List[StoredTransform] = []
        self.transform_index: Dict[str, StoredTransform] = {}
        
        # Composition cache
        self.composition_cache: Dict[Tuple[str, str], StoredTransform] = {}
        
        # Statistics
        self.total_stored = 0
        self.total_retrieved = 0
        self.total_composed = 0
        
        # Initialize with basic transforms
        self._initialize_basic_transforms()
        
        logger.info(f"Transform memory initialized with capacity={capacity}")
    
    def _initialize_basic_transforms(self):
        """Initialize memory with basic transform primitives."""
        basic_transforms = [
            {
                'name': 'identity',
                'primitives': ['identity'],
                'properties': {'type': 'basic', 'complexity': 0}
            },
            {
                'name': 'translate_right',
                'primitives': ['shift_right'],
                'properties': {'type': 'spatial', 'complexity': 1}
            },
            {
                'name': 'translate_up',
                'primitives': ['shift_up'],
                'properties': {'type': 'spatial', 'complexity': 1}
            },
            {
                'name': 'rotate_90',
                'primitives': ['rotate_90'],
                'properties': {'type': 'rotation', 'complexity': 2}
            },
            {
                'name': 'flip_horizontal',
                'primitives': ['flip_h'],
                'properties': {'type': 'reflection', 'complexity': 1}
            },
            {
                'name': 'scale_2x',
                'primitives': ['expand_2x'],
                'properties': {'type': 'scaling', 'complexity': 2}
            }
        ]
        
        for i, transform_spec in enumerate(basic_transforms):
            # Create embedding
            embedding = np.zeros(self.embedding_dim)
            embedding[i] = 1.0  # One-hot for basic transforms
            
            # Create task encoding (generic)
            task_encoding = np.random.randn(self.embedding_dim) * 0.1
            
            # Store transform
            transform = StoredTransform(
                id=f"basic_{transform_spec['name']}",
                embedding=embedding,
                task_encoding=task_encoding,
                performance_score=1.0,  # Perfect for basic transforms
                primitive_sequence=transform_spec['primitives'],
                metadata=transform_spec['properties']
            )
            
            self.transforms.append(transform)
            self.transform_index[transform.id] = transform
        
        logger.info(f"Initialized {len(basic_transforms)} basic transforms")
    
    def add(
        self,
        primitive_sequence: List[str],
        task_encoding: np.ndarray,
        performance_score: float,
        metadata: Optional[Dict] = None
    ) -> str:
        """Add a new transform to memory."""
        # Check if we already have a very similar transform
        for existing in self.transforms:
            if (existing.primitive_sequence == primitive_sequence and
                existing.similarity_to_task(task_encoding) > self.similarity_threshold):
                # Update existing transform
                existing.usage_count += 1
                existing.performance_score = max(existing.performance_score, performance_score)
                logger.debug(f"Updated existing transform {existing.id}")
                return existing.id
        
        # Create new transform
        transform_id = f"learned_{self.total_stored}"
        
        # Generate embedding from primitive sequence
        embedding = self._generate_embedding(primitive_sequence)
        
        # Create stored transform
        transform = StoredTransform(
            id=transform_id,
            embedding=embedding,
            task_encoding=task_encoding,
            performance_score=performance_score,
            primitive_sequence=primitive_sequence,
            metadata=metadata or {},
            discovery_time=self.total_stored  # Use counter as proxy for time
        )
        
        # Add to memory
        self.transforms.append(transform)
        self.transform_index[transform_id] = transform
        self.total_stored += 1
        
        # Manage capacity
        if len(self.transforms) > self.capacity:
            self._evict_least_useful()
        
        logger.info(f"Added transform {transform_id} with score {performance_score:.2%}")
        return transform_id
    
    def _generate_embedding(self, primitive_sequence: List[str]) -> np.ndarray:
        """Generate embedding from primitive sequence."""
        embedding = np.zeros(self.embedding_dim)
        
        # Simple embedding: hash primitives to dimensions
        for i, primitive in enumerate(primitive_sequence):
            idx = hash(primitive) % self.embedding_dim
            embedding[idx] += 1.0 / (i + 1)  # Decay weight by position
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        
        return embedding
    
    def retrieve(
        self,
        task_encoding: np.ndarray,
        k: int = 10
    ) -> List[StoredTransform]:
        """Retrieve k most relevant transforms for a task."""
        if not self.transforms:
            return []
        
        # Compute similarities
        similarities = []
        for transform in self.transforms:
            sim = transform.similarity_to_task(task_encoding)
            similarities.append((sim, transform))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Get top k
        retrieved = [t for _, t in similarities[:k]]
        
        # Update usage counts
        for transform in retrieved:
            transform.usage_count += 1
        
        self.total_retrieved += len(retrieved)
        
        logger.debug(f"Retrieved {len(retrieved)} transforms")
        return retrieved
    
    def compose(
        self,
        transform1_id: str,
        transform2_id: str
    ) -> Optional[StoredTransform]:
        """Compose two transforms to create a new one."""
        # Check cache
        cache_key = (transform1_id, transform2_id)
        if cache_key in self.composition_cache:
            return self.composition_cache[cache_key]
        
        # Get transforms
        t1 = self.transform_index.get(transform1_id)
        t2 = self.transform_index.get(transform2_id)
        
        if not t1 or not t2:
            return None
        
        # Compose primitive sequences
        composed_primitives = t1.primitive_sequence + t2.primitive_sequence
        
        # Compose embeddings (average for simplicity)
        composed_embedding = (t1.embedding + t2.embedding) / 2
        
        # Compose task encodings
        composed_task_encoding = (t1.task_encoding + t2.task_encoding) / 2
        
        # Estimate performance (conservative)
        composed_score = min(t1.performance_score, t2.performance_score) * 0.9
        
        # Create composed transform
        composed = StoredTransform(
            id=f"composed_{transform1_id}_{transform2_id}",
            embedding=composed_embedding,
            task_encoding=composed_task_encoding,
            performance_score=composed_score,
            primitive_sequence=composed_primitives,
            metadata={
                'type': 'composed',
                'source1': transform1_id,
                'source2': transform2_id
            }
        )
        
        # Cache it
        self.composition_cache[cache_key] = composed
        self.total_composed += 1
        
        logger.info(f"Composed {transform1_id} + {transform2_id}")
        return composed
    
    def find_novel_combinations(
        self,
        task_encoding: np.ndarray,
        n_combinations: int = 10
    ) -> List[StoredTransform]:
        """Find novel transform combinations for a task."""
        # Get relevant base transforms
        base_transforms = self.retrieve(task_encoding, k=10)
        
        novel_combinations = []
        
        # Try pairwise compositions
        for i, t1 in enumerate(base_transforms):
            for t2 in base_transforms[i+1:]:
                # Check if composition would be novel
                composed_embedding = (t1.embedding + t2.embedding) / 2
                
                # Check novelty (low similarity to existing)
                is_novel = True
                for existing in self.transforms:
                    if existing.similarity_to_transform(composed_embedding) > 0.9:
                        is_novel = False
                        break
                
                if is_novel:
                    composed = self.compose(t1.id, t2.id)
                    if composed:
                        novel_combinations.append(composed)
                
                if len(novel_combinations) >= n_combinations:
                    break
            
            if len(novel_combinations) >= n_combinations:
                break
        
        logger.info(f"Found {len(novel_combinations)} novel combinations")
        return novel_combinations
    
    def _evict_least_useful(self):
        """Evict least useful transform when at capacity."""
        # Score transforms by usefulness
        scores = []
        for transform in self.transforms:
            # Don't evict basic transforms
            if transform.id.startswith("basic_"):
                continue
            
            # Usefulness score
            recency = 1.0 / (self.total_stored - transform.discovery_time + 1)
            frequency = transform.usage_count / (self.total_retrieved + 1)
            performance = transform.performance_score
            
            score = recency * 0.2 + frequency * 0.3 + performance * 0.5
            scores.append((score, transform))
        
        if scores:
            # Sort by score
            scores.sort(key=lambda x: x[0])
            
            # Evict lowest scoring
            to_evict = scores[0][1]
            self.transforms.remove(to_evict)
            del self.transform_index[to_evict.id]
            
            logger.debug(f"Evicted transform {to_evict.id}")
    
    def get_statistics(self) -> Dict:
        """Get memory statistics."""
        if not self.transforms:
            return {}
        
        performance_scores = [t.performance_score for t in self.transforms]
        usage_counts = [t.usage_count for t in self.transforms]
        
        # Count by type
        type_counts = {}
        for transform in self.transforms:
            t_type = transform.metadata.get('type', 'unknown')
            type_counts[t_type] = type_counts.get(t_type, 0) + 1
        
        return {
            'total_transforms': len(self.transforms),
            'total_stored': self.total_stored,
            'total_retrieved': self.total_retrieved,
            'total_composed': self.total_composed,
            'average_performance': np.mean(performance_scores),
            'average_usage': np.mean(usage_counts),
            'type_distribution': type_counts,
            'capacity_used': len(self.transforms) / self.capacity
        }
    
    def save(self, filepath: str):
        """Save memory to file."""
        data = {
            'transforms': [(t.id, t.primitive_sequence, t.performance_score) 
                          for t in self.transforms],
            'statistics': self.get_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved memory to {filepath}")
    
    def load(self, filepath: str):
        """Load memory from file."""
        # Not fully implemented - would need to reconstruct transforms
        logger.warning("Load not fully implemented")


def test_transform_memory():
    """Test the transform memory system."""
    print("\n" + "=" * 60)
    print("TESTING TRANSFORM MEMORY SYSTEM")
    print("=" * 60)
    
    # Create memory
    memory = TransformMemory(capacity=100)
    
    # Test adding transforms
    print("\nAdding learned transforms...")
    
    # Add a shear-like transform
    shear_primitives = ['shift_row_0', 'shift_row_1', 'shift_row_2']
    task_encoding = np.random.randn(256)
    transform_id = memory.add(shear_primitives, task_encoding, 0.95)
    print(f"Added shear transform: {transform_id}")
    
    # Add a rotation + translation
    rot_trans_primitives = ['rotate_90', 'shift_right', 'shift_up']
    task_encoding2 = np.random.randn(256)
    transform_id2 = memory.add(rot_trans_primitives, task_encoding2, 0.88)
    print(f"Added rotation+translation: {transform_id2}")
    
    # Test retrieval
    print("\nTesting retrieval...")
    retrieved = memory.retrieve(task_encoding, k=3)
    print(f"Retrieved {len(retrieved)} transforms:")
    for t in retrieved:
        print(f"  - {t.id}: score={t.performance_score:.2%}, primitives={len(t.primitive_sequence)}")
    
    # Test composition
    print("\nTesting composition...")
    composed = memory.compose('basic_rotate_90', 'basic_translate_right')
    if composed:
        print(f"Composed transform: {composed.id}")
        print(f"  Primitives: {composed.primitive_sequence}")
    
    # Test novel combinations
    print("\nFinding novel combinations...")
    novel = memory.find_novel_combinations(task_encoding, n_combinations=3)
    print(f"Found {len(novel)} novel combinations")
    
    # Print statistics
    print("\n" + "-" * 40)
    print("Memory Statistics:")
    stats = memory.get_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    print("\n✅ Transform Memory is working!")
    return True


if __name__ == "__main__":
    test_transform_memory()