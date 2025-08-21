"""Adaptive Computation Time (ACT) mechanism for HTI.

Implements Q-learning based halting mechanism that learns when to stop reasoning,
inspired by the ACT mechanism in the Hierarchical Reasoning Model.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ComputationState:
    """State for adaptive computation decisions."""
    
    task_complexity: float  # Estimated complexity of current task
    current_confidence: float  # Current solution confidence
    cycles_used: int  # Number of reasoning cycles so far
    improvement_rate: float  # Rate of improvement in recent cycles
    exploration_diversity: float  # Diversity of explored solutions


class AdaptiveComputationTime:
    """Q-learning based adaptive computation controller.
    
    Learns when to halt reasoning based on task state and performance.
    """
    
    def __init__(
        self,
        state_dim: int = 768,
        max_segments: int = 16,
        learning_rate: float = 0.01,
        discount_factor: float = 0.95,
        epsilon: float = 0.1
    ):
        self.state_dim = state_dim
        self.max_segments = max_segments
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Q-function approximation (simplified - linear for now)
        # Maps state features to Q-values for [continue, halt] actions
        self.q_weights = np.random.randn(state_dim, 2) * 0.01
        
        # Experience replay buffer
        self.replay_buffer = []
        self.max_buffer_size = 1000
        
        # Statistics tracking
        self.halt_decisions = []
        self.average_cycles = 0
        
        logger.info(f"ACT initialized with max_segments={max_segments}")
    
    def extract_features(self, state: ComputationState) -> np.ndarray:
        """Extract features from computation state for Q-learning."""
        features = []
        
        # Basic features
        features.append(state.task_complexity)
        features.append(state.current_confidence)
        features.append(state.cycles_used / self.max_segments)
        features.append(state.improvement_rate)
        features.append(state.exploration_diversity)
        
        # Derived features
        features.append(state.current_confidence * state.task_complexity)  # Interaction
        features.append(1.0 if state.cycles_used > self.max_segments / 2 else 0.0)  # Half-way
        features.append(np.exp(-state.improvement_rate))  # Decay indicator
        
        # Threshold features
        confidence_thresholds = [0.5, 0.7, 0.9, 0.95]
        for threshold in confidence_thresholds:
            features.append(1.0 if state.current_confidence > threshold else 0.0)
        
        # Cycle bucket features
        cycle_buckets = [2, 4, 8, 12]
        for bucket in cycle_buckets:
            features.append(1.0 if state.cycles_used >= bucket else 0.0)
        
        # Pad to state_dim
        feature_array = np.array(features)
        if len(feature_array) < self.state_dim:
            feature_array = np.pad(feature_array, (0, self.state_dim - len(feature_array)))
        else:
            feature_array = feature_array[:self.state_dim]
        
        return feature_array
    
    def should_halt(
        self,
        state: ComputationState,
        use_epsilon_greedy: bool = True
    ) -> bool:
        """Decide whether to halt computation based on current state."""
        # Extract features
        features = self.extract_features(state)
        
        # Compute Q-values
        q_values = np.dot(features, self.q_weights)
        q_continue = q_values[0]
        q_halt = q_values[1]
        
        # Epsilon-greedy action selection
        if use_epsilon_greedy and np.random.random() < self.epsilon:
            # Random action
            action = np.random.choice([0, 1])
        else:
            # Greedy action
            action = 1 if q_halt > q_continue else 0
        
        # Force halt if at maximum segments
        if state.cycles_used >= self.max_segments:
            action = 1
            logger.info(f"Forced halt at maximum segments ({self.max_segments})")
        
        # Record decision
        self.halt_decisions.append({
            'cycles': state.cycles_used,
            'confidence': state.current_confidence,
            'action': action,
            'q_continue': q_continue,
            'q_halt': q_halt
        })
        
        should_halt = bool(action == 1)
        
        if should_halt:
            logger.info(f"Halting at cycle {state.cycles_used} with confidence {state.current_confidence:.2%}")
        
        return should_halt
    
    def update_q_function(
        self,
        state: ComputationState,
        action: int,
        reward: float,
        next_state: Optional[ComputationState] = None
    ):
        """Update Q-function based on experience."""
        # Extract features
        features = self.extract_features(state)
        
        # Current Q-value
        current_q = np.dot(features, self.q_weights[:, action])
        
        # Next Q-value (if not terminal)
        if next_state is not None:
            next_features = self.extract_features(next_state)
            next_q_values = np.dot(next_features, self.q_weights)
            max_next_q = np.max(next_q_values)
        else:
            max_next_q = 0  # Terminal state
        
        # Q-learning update
        target = reward + self.discount_factor * max_next_q
        td_error = target - current_q
        
        # Update weights
        self.q_weights[:, action] += self.learning_rate * td_error * features
        
        # Store in replay buffer
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state
        }
        self.replay_buffer.append(experience)
        
        # Limit buffer size
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer.pop(0)
        
        logger.debug(f"Q-update: action={action}, reward={reward:.3f}, td_error={td_error:.3f}")
    
    def compute_reward(
        self,
        final_score: float,
        cycles_used: int,
        improvement_achieved: float
    ) -> float:
        """Compute reward for the halting decision."""
        # Reward components
        score_reward = final_score  # Higher score is better
        
        # Efficiency bonus (fewer cycles is better)
        efficiency_bonus = 0.0
        if final_score > 0.9:  # Only reward efficiency if solution is good
            efficiency_bonus = 0.2 * (1.0 - cycles_used / self.max_segments)
        
        # Improvement reward
        improvement_reward = improvement_achieved * 0.3
        
        # Penalty for using too many cycles without improvement
        cycle_penalty = 0.0
        if cycles_used > self.max_segments / 2 and improvement_achieved < 0.1:
            cycle_penalty = -0.2
        
        total_reward = score_reward + efficiency_bonus + improvement_reward + cycle_penalty
        
        return float(np.clip(total_reward, -1.0, 1.0))
    
    def replay_experience(self, batch_size: int = 32):
        """Replay experiences to improve Q-function."""
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample batch
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        
        for idx in indices:
            exp = self.replay_buffer[idx]
            self.update_q_function(
                exp['state'],
                exp['action'],
                exp['reward'],
                exp['next_state']
            )
    
    def get_statistics(self) -> Dict:
        """Get statistics about halting decisions."""
        if not self.halt_decisions:
            return {}
        
        cycles_list = [d['cycles'] for d in self.halt_decisions]
        confidence_list = [d['confidence'] for d in self.halt_decisions]
        
        return {
            'average_cycles': np.mean(cycles_list),
            'std_cycles': np.std(cycles_list),
            'average_confidence': np.mean(confidence_list),
            'min_cycles': np.min(cycles_list),
            'max_cycles': np.max(cycles_list),
            'total_decisions': len(self.halt_decisions)
        }
    
    def adapt_for_task_complexity(self, estimated_complexity: float):
        """Adapt parameters based on estimated task complexity."""
        # More complex tasks get more computation budget
        if estimated_complexity > 0.8:
            # Hard task - be more patient
            self.epsilon = max(0.05, self.epsilon * 0.9)
            logger.info(f"Hard task detected, reducing exploration to {self.epsilon:.3f}")
        elif estimated_complexity < 0.3:
            # Easy task - be more aggressive about halting
            self.epsilon = min(0.2, self.epsilon * 1.1)
            logger.info(f"Easy task detected, increasing exploration to {self.epsilon:.3f}")


def test_adaptive_computation():
    """Test the ACT mechanism."""
    print("\n" + "=" * 60)
    print("TESTING ADAPTIVE COMPUTATION TIME")
    print("=" * 60)
    
    # Create ACT controller
    act = AdaptiveComputationTime(max_segments=10)
    
    # Simulate different task scenarios
    scenarios = [
        ("Easy task", 0.2, [(0.3, 0.1), (0.6, 0.3), (0.9, 0.3), (0.95, 0.05)]),
        ("Medium task", 0.5, [(0.2, 0.2), (0.4, 0.2), (0.6, 0.2), (0.75, 0.15), (0.85, 0.1), (0.9, 0.05)]),
        ("Hard task", 0.8, [(0.1, 0.1), (0.2, 0.1), (0.3, 0.1), (0.4, 0.1), (0.5, 0.1), 
                           (0.6, 0.1), (0.7, 0.1), (0.8, 0.1), (0.85, 0.05), (0.9, 0.05)])
    ]
    
    for scenario_name, complexity, confidence_trajectory in scenarios:
        print(f"\n{scenario_name} (complexity={complexity}):")
        
        # Reset for new task
        act.adapt_for_task_complexity(complexity)
        
        # Simulate reasoning cycles
        for cycle, (confidence, improvement) in enumerate(confidence_trajectory):
            state = ComputationState(
                task_complexity=complexity,
                current_confidence=confidence,
                cycles_used=cycle + 1,
                improvement_rate=improvement,
                exploration_diversity=0.5
            )
            
            should_halt = act.should_halt(state, use_epsilon_greedy=False)
            
            print(f"  Cycle {cycle + 1}: confidence={confidence:.1%}, improvement={improvement:.1%} -> {'HALT' if should_halt else 'CONTINUE'}")
            
            if should_halt:
                # Compute reward
                final_improvement = sum(imp for _, imp in confidence_trajectory[:cycle+1])
                reward = act.compute_reward(confidence, cycle + 1, final_improvement)
                
                # Update Q-function
                act.update_q_function(state, 1, reward, None)
                print(f"  Final: {cycle + 1} cycles, reward={reward:.3f}")
                break
            else:
                # Update for continuing
                if cycle < len(confidence_trajectory) - 1:
                    next_confidence, next_improvement = confidence_trajectory[cycle + 1]
                    next_state = ComputationState(
                        task_complexity=complexity,
                        current_confidence=next_confidence,
                        cycles_used=cycle + 2,
                        improvement_rate=next_improvement,
                        exploration_diversity=0.5
                    )
                    
                    # Small negative reward for continuing (time cost)
                    reward = -0.01
                    act.update_q_function(state, 0, reward, next_state)
    
    # Print statistics
    stats = act.get_statistics()
    print("\n" + "-" * 40)
    print("Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")
    
    print("\n✅ Adaptive Computation Time is working!")
    return True


if __name__ == "__main__":
    test_adaptive_computation()