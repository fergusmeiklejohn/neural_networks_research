import os

os.environ["KERAS_BACKEND"] = "jax"

import keras

print(f"Keras version: {keras.__version__}")
print(f"Backend: {keras.backend.backend()}")
print("Available keras.backend methods:")
print([attr for attr in dir(keras.backend) if not attr.startswith("_")])
