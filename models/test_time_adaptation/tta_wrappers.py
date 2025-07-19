"""Wrappers to add TTA capabilities to existing models."""

from typing import Any, Dict, Optional, Union
import keras
from keras import ops

from .tent import TENT, PhysicsTENT
from .ttt_physics import PhysicsTTT
from .regression_tta import RegressionTTA, PhysicsRegressionTTA
from .regression_tta_v2 import RegressionTTAV2, PhysicsRegressionTTAV2
from ..baseline_models import BaselineModel


class TTAWrapper:
    """Universal wrapper to add TTA to any model.
    
    This wrapper can apply different TTA methods to existing models
    without modifying their architecture.
    """
    
    def __init__(
        self,
        model: Union[keras.Model, BaselineModel],
        tta_method: str = 'tent',
        **tta_kwargs
    ):
        """Initialize TTA wrapper.
        
        Args:
            model: Base model to wrap
            tta_method: TTA method to use ('tent', 'physics_tent', 'ttt', 'regression', 'physics_regression',
                       'regression_v2', 'physics_regression_v2')
            **tta_kwargs: Arguments passed to TTA method
        """
        self.base_model = model
        self.tta_method = tta_method
        
        # Extract Keras model if using BaselineModel
        if hasattr(model, 'model'):
            keras_model = model.model
        else:
            keras_model = model
        
        # Create TTA adapter
        if tta_method == 'tent':
            self.tta_adapter = TENT(keras_model, **tta_kwargs)
        elif tta_method == 'physics_tent':
            self.tta_adapter = PhysicsTENT(keras_model, **tta_kwargs)
        elif tta_method == 'ttt':
            self.tta_adapter = PhysicsTTT(keras_model, **tta_kwargs)
        elif tta_method == 'regression':
            self.tta_adapter = RegressionTTA(keras_model, **tta_kwargs)
        elif tta_method == 'physics_regression':
            self.tta_adapter = PhysicsRegressionTTA(keras_model, **tta_kwargs)
        elif tta_method == 'regression_v2':
            self.tta_adapter = RegressionTTAV2(keras_model, **tta_kwargs)
        elif tta_method == 'physics_regression_v2':
            self.tta_adapter = PhysicsRegressionTTAV2(keras_model, **tta_kwargs)
        else:
            raise ValueError(f"Unknown TTA method: {tta_method}")
    
    def predict(self, x: Any, adapt: bool = True, **kwargs) -> Any:
        """Make predictions with optional TTA.
        
        Args:
            x: Input data
            adapt: Whether to perform test-time adaptation
            **kwargs: Additional arguments for prediction
            
        Returns:
            Predictions
        """
        if adapt:
            return self.tta_adapter.predict_and_adapt(x, **kwargs)
        else:
            if hasattr(self.base_model, 'predict'):
                return self.base_model.predict(x, **kwargs)
            else:
                return self.base_model(x, training=False)
    
    def reset(self):
        """Reset TTA adapter to original state."""
        self.tta_adapter.reset()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get TTA adaptation metrics."""
        return self.tta_adapter.get_metrics()


def create_tta_baseline(
    baseline_name: str,
    tta_method: str = 'tent',
    model_kwargs: Optional[Dict] = None,
    tta_kwargs: Optional[Dict] = None
) -> TTAWrapper:
    """Create a TTA-enhanced baseline model.
    
    Args:
        baseline_name: Name of baseline ('erm', 'gflownet', 'graph_extrap', 'maml')
        tta_method: TTA method to use
        model_kwargs: Arguments for baseline model
        tta_kwargs: Arguments for TTA method
        
    Returns:
        TTA-wrapped baseline model
    """
    from ..baseline_models import create_baseline_model
    
    # Create baseline model
    model_kwargs = model_kwargs or {}
    baseline = create_baseline_model(baseline_name, **model_kwargs)
    
    # Wrap with TTA
    tta_kwargs = tta_kwargs or {}
    return TTAWrapper(baseline, tta_method, **tta_kwargs)


class OnlinePhysicsAdapter:
    """Online adaptation for streaming physics data.
    
    This adapter maintains state across predictions and continuously
    adapts to changing physics.
    """
    
    def __init__(
        self,
        model: keras.Model,
        window_size: int = 10,
        adaptation_frequency: int = 5,
        tta_method: str = 'ttt'
    ):
        """Initialize online adapter.
        
        Args:
            model: Base physics model
            window_size: Size of observation window
            adaptation_frequency: Adapt every N observations
            tta_method: TTA method to use
        """
        self.model = model
        self.window_size = window_size
        self.adaptation_frequency = adaptation_frequency
        
        # Create TTA adapter
        if tta_method == 'ttt':
            self.tta_adapter = PhysicsTTT(
                model,
                reset_after_batch=False,  # Keep adaptation state
                adaptation_window=window_size
            )
        else:
            raise ValueError(f"Unsupported online TTA method: {tta_method}")
        
        # State tracking
        self.observation_buffer = []
        self.prediction_count = 0
        self.physics_estimates = []
    
    def predict_next(self, context: Any) -> Any:
        """Predict next state with online adaptation.
        
        Args:
            context: Current context/observation
            
        Returns:
            Next state prediction
        """
        # Add to buffer
        self.observation_buffer.append(context)
        if len(self.observation_buffer) > self.window_size:
            self.observation_buffer.pop(0)
        
        # Adapt if needed
        self.prediction_count += 1
        should_adapt = (
            self.prediction_count % self.adaptation_frequency == 0 and
            len(self.observation_buffer) >= self.window_size
        )
        
        if should_adapt:
            # Perform online adaptation
            adaptation_result = self.tta_adapter.adapt_online(context)
            if adaptation_result['estimated_physics']:
                self.physics_estimates.append(adaptation_result['estimated_physics'])
        
        # Make prediction
        prediction = self.tta_adapter.predict_next(
            ops.stack(self.observation_buffer),
            adapt=False  # Already adapted above
        )
        
        return prediction
    
    def get_physics_estimates(self) -> list:
        """Get estimated physics parameters over time."""
        return self.physics_estimates
    
    def reset(self):
        """Reset adapter state."""
        self.observation_buffer = []
        self.prediction_count = 0
        self.physics_estimates = []
        self.tta_adapter.reset()