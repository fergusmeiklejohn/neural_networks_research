"""
PeTTA-inspired collapse detection for our physics TTA experiments.
Since we don't have their exact algorithm, we implement reasonable collapse metrics.
"""

import numpy as np
import keras
from keras import ops
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt

class CollapseDetector:
    """Monitor various metrics that might indicate model collapse during TTA"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.metrics_history = {
            'prediction_entropy': [],
            'prediction_variance': [],
            'parameter_change': [],
            'loss_trajectory': [],
            'gradient_norm': []
        }
        self.initial_params = None
        
    def compute_prediction_entropy(self, predictions: np.ndarray) -> float:
        """Compute entropy of predictions - low entropy might indicate collapse"""
        # For regression, we can look at the distribution of predictions
        pred_flat = predictions.flatten()
        hist, _ = np.histogram(pred_flat, bins=20)
        hist = hist / hist.sum() + 1e-8  # Normalize and avoid log(0)
        entropy = -np.sum(hist * np.log(hist))
        return entropy
    
    def compute_prediction_variance(self, predictions: np.ndarray) -> float:
        """Compute variance of predictions - low variance might indicate collapse to constant"""
        return np.var(predictions)
    
    def compute_parameter_drift(self, current_params: List[np.ndarray]) -> float:
        """Compute how far parameters have drifted from initial"""
        if self.initial_params is None:
            self.initial_params = [p.copy() for p in current_params]
            return 0.0
        
        total_drift = 0.0
        for p_init, p_curr in zip(self.initial_params, current_params):
            drift = np.mean((p_curr - p_init) ** 2)
            total_drift += drift
        return total_drift
    
    def detect_collapse(self, predictions: np.ndarray, 
                       current_params: List[np.ndarray],
                       loss: float,
                       gradients: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        """Check multiple indicators of collapse"""
        
        # Compute metrics
        entropy = self.compute_prediction_entropy(predictions)
        variance = self.compute_prediction_variance(predictions)
        param_drift = self.compute_parameter_drift(current_params)
        
        # Store history
        self.metrics_history['prediction_entropy'].append(entropy)
        self.metrics_history['prediction_variance'].append(variance)
        self.metrics_history['parameter_change'].append(param_drift)
        self.metrics_history['loss_trajectory'].append(loss)
        
        if gradients is not None:
            grad_norm = sum(np.mean(g**2) for g in gradients)
            self.metrics_history['gradient_norm'].append(grad_norm)
        
        # Detect collapse patterns
        collapse_indicators = {
            'low_entropy': False,
            'low_variance': False,
            'high_drift': False,
            'loss_plateau': False,
            'gradient_vanish': False
        }
        
        # Need enough history
        if len(self.metrics_history['prediction_entropy']) >= self.window_size:
            recent_entropy = self.metrics_history['prediction_entropy'][-self.window_size:]
            recent_variance = self.metrics_history['prediction_variance'][-self.window_size:]
            
            # Check for decreasing entropy (predictions becoming too similar)
            if np.mean(recent_entropy) < 0.5 * self.metrics_history['prediction_entropy'][0]:
                collapse_indicators['low_entropy'] = True
                
            # Check for vanishing variance (predictions converging to constant)
            if np.mean(recent_variance) < 0.1 * self.metrics_history['prediction_variance'][0]:
                collapse_indicators['low_variance'] = True
                
            # Check for excessive parameter drift
            if param_drift > 10.0:  # Threshold depends on model scale
                collapse_indicators['high_drift'] = True
                
            # Check for loss plateau (no improvement)
            recent_loss = self.metrics_history['loss_trajectory'][-self.window_size:]
            if np.std(recent_loss) < 0.01 * np.mean(recent_loss):
                collapse_indicators['loss_plateau'] = True
        
        # Overall collapse detection
        collapse_score = sum(collapse_indicators.values()) / len(collapse_indicators)
        is_collapsing = collapse_score >= 0.3  # If 30% of indicators suggest collapse
        
        return {
            'is_collapsing': is_collapsing,
            'collapse_score': collapse_score,
            'indicators': collapse_indicators,
            'current_metrics': {
                'entropy': entropy,
                'variance': variance,
                'param_drift': param_drift,
                'loss': loss
            }
        }
    
    def plot_collapse_metrics(self, save_path: Optional[str] = None):
        """Visualize collapse detection metrics over time"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Prediction entropy
        axes[0, 0].plot(self.metrics_history['prediction_entropy'])
        axes[0, 0].set_title('Prediction Entropy')
        axes[0, 0].set_xlabel('TTA Step')
        axes[0, 0].set_ylabel('Entropy')
        
        # Prediction variance
        axes[0, 1].plot(self.metrics_history['prediction_variance'])
        axes[0, 1].set_title('Prediction Variance')
        axes[0, 1].set_xlabel('TTA Step')
        axes[0, 1].set_ylabel('Variance')
        
        # Parameter drift
        axes[1, 0].plot(self.metrics_history['parameter_change'])
        axes[1, 0].set_title('Parameter Drift')
        axes[1, 0].set_xlabel('TTA Step')
        axes[1, 0].set_ylabel('L2 Distance from Initial')
        
        # Loss trajectory
        axes[1, 1].plot(self.metrics_history['loss_trajectory'])
        axes[1, 1].set_title('Loss Trajectory')
        axes[1, 1].set_xlabel('TTA Step')
        axes[1, 1].set_ylabel('Loss')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()


class PeTTAInspiredTTA:
    """TTA with collapse detection and intervention inspired by PeTTA"""
    
    def __init__(self, base_model: keras.Model, 
                 adaptation_type: str = 'prediction',
                 collapse_threshold: float = 0.3):
        self.base_model = base_model
        self.adaptation_type = adaptation_type
        self.collapse_threshold = collapse_threshold
        
        # Clone model for adaptation
        self.adapted_model = keras.models.clone_model(base_model)
        self.adapted_model.set_weights(base_model.get_weights())
        
        # Store checkpoints
        self.checkpoints = []
        self.save_checkpoint()
        
        # Initialize collapse detector
        self.collapse_detector = CollapseDetector()
        
        # Adaptation settings
        self.base_lr = 1e-4
        self.current_lr = self.base_lr
        self.tta_optimizer = keras.optimizers.Adam(learning_rate=self.current_lr)
        
    def save_checkpoint(self):
        """Save current model state"""
        weights = self.adapted_model.get_weights()
        self.checkpoints.append([w.copy() for w in weights])
        
    def restore_checkpoint(self, idx: int = -1):
        """Restore model to a previous checkpoint"""
        if self.checkpoints:
            self.adapted_model.set_weights(self.checkpoints[idx])
            
    def adjust_learning_rate(self, factor: float):
        """Adjust learning rate in response to collapse"""
        self.current_lr = self.base_lr * factor
        self.tta_optimizer.learning_rate.assign(self.current_lr)
        
    def compute_adaptation_loss(self, predictions):
        """Compute the adaptation loss based on type"""
        if self.adaptation_type == 'prediction':
            # Prediction consistency
            pred_mean = ops.mean(predictions, axis=1, keepdims=True)
            return ops.mean((predictions - pred_mean)**2)
        else:
            raise NotImplementedError(f"Adaptation type {self.adaptation_type} not implemented")
            
    def adapt_batch_with_monitoring(self, X_batch: np.ndarray, num_steps: int = 20):
        """Adapt with collapse detection and intervention"""
        
        adaptation_history = {
            'losses': [],
            'collapse_events': [],
            'interventions': []
        }
        
        # Create a simple loss function for adaptation
        def tta_loss_fn(y_true, y_pred):
            return self.compute_adaptation_loss(y_pred)
        
        # Compile model with TTA loss
        self.adapted_model.compile(
            optimizer=self.tta_optimizer,
            loss=tta_loss_fn
        )
        
        for step in range(num_steps):
            # Forward pass and compute loss
            predictions = self.adapted_model.predict(X_batch, verbose=0)
            
            # Create dummy targets (we don't use them in the loss)
            dummy_targets = predictions.copy()
            
            # Single adaptation step
            history = self.adapted_model.fit(
                X_batch, 
                dummy_targets,
                epochs=1,
                batch_size=len(X_batch),
                verbose=0
            )
            
            loss = history.history['loss'][0]
            
            # Check for collapse
            collapse_status = self.collapse_detector.detect_collapse(
                predictions,
                self.adapted_model.get_weights(),
                float(loss),
                None  # Skip gradients for simplicity
            )
            
            adaptation_history['losses'].append(float(loss))
            
            # Intervention if collapsing
            if collapse_status['is_collapsing']:
                print(f"Step {step}: Collapse detected! Score: {collapse_status['collapse_score']:.3f}")
                adaptation_history['collapse_events'].append(step)
                
                # Intervention strategies
                intervention = self.intervene_on_collapse(collapse_status)
                adaptation_history['interventions'].append({
                    'step': step,
                    'type': intervention
                })
                
                if intervention == 'stop':
                    print("Stopping adaptation due to collapse")
                    break
            
            # Periodic checkpointing
            if step % 5 == 0:
                self.save_checkpoint()
                    
            if step % 5 == 0:
                print(f"Step {step}: Loss = {float(loss):.6f}, "
                      f"Entropy = {collapse_status['current_metrics']['entropy']:.3f}, "
                      f"Collapse = {collapse_status['is_collapsing']}")
                      
        return adaptation_history
    
    def intervene_on_collapse(self, collapse_status: Dict) -> str:
        """Decide how to intervene when collapse detected"""
        indicators = collapse_status['indicators']
        
        if indicators['high_drift']:
            # Too much parameter change - restore earlier checkpoint
            print("Intervention: Restoring earlier checkpoint due to high drift")
            self.restore_checkpoint(-5)  # Go back 5 checkpoints
            self.adjust_learning_rate(0.1)  # Reduce learning rate
            return 'restore_and_reduce_lr'
            
        elif indicators['low_variance'] and indicators['low_entropy']:
            # Predictions collapsing to constant - stop adaptation
            print("Intervention: Stopping adaptation due to prediction collapse")
            return 'stop'
            
        elif indicators['loss_plateau']:
            # No progress - reduce learning rate
            print("Intervention: Reducing learning rate due to plateau")
            self.adjust_learning_rate(0.5)
            return 'reduce_lr'
            
        else:
            # Minor intervention - just reduce learning rate slightly
            self.adjust_learning_rate(0.8)
            return 'minor_lr_reduction'


def test_petta_on_pendulum():
    """Test PeTTA-inspired approach on pendulum data"""
    print("Testing PeTTA-inspired Collapse Detection on Pendulum")
    print("=" * 60)
    
    # This would load the pre-trained model and test data
    # For now, showing the structure
    
    # Example usage:
    # model = load_model("pendulum_base_model.keras")
    # petta_tta = PeTTAInspiredTTA(model, collapse_threshold=0.3)
    # history = petta_tta.adapt_batch_with_monitoring(X_test, num_steps=30)
    # petta_tta.collapse_detector.plot_collapse_metrics("collapse_metrics.png")
    
    print("Implementation complete - ready to test on actual data")


if __name__ == "__main__":
    test_petta_on_pendulum()