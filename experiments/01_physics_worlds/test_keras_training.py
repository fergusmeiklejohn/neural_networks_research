import os
os.environ['KERAS_BACKEND'] = 'jax'

import keras
import numpy as np

# Create a simple model
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    keras.layers.Dense(1)
])

# Create dummy data
x = np.random.randn(100, 5).astype(np.float32)
y = np.random.randn(100, 1).astype(np.float32)

# Compile and fit
model.compile(optimizer='adam', loss='mse')
history = model.fit(x, y, epochs=2, batch_size=10, verbose=1)

print("\nTraining completed successfully!")
print(f"Final loss: {history.history['loss'][-1]:.4f}")