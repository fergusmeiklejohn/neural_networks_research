# Fix for "Target data is missing" Error

## Problem
The model training was failing with "Target data is missing" error when calling `model.fit()` in `paperspace_train_with_safeguards.py`.

## Root Cause
The `create_dataset()` function was returning a dataset with a dictionary structure:
```python
{
    'command': commands,
    'action': actions,
    'has_modification': has_modification,
    'modification': modification_commands
}
```

However, `model.fit()` expects data in the format `(inputs, targets)` where:
- `inputs` is a dictionary with the model's expected input keys
- `targets` is the target data for computing the loss

## Solution
Updated `create_dataset()` to:

1. **Map the data to the correct format** using a `prepare_for_training` function that:
   - Creates an inputs dictionary with keys: `'command'`, `'target'`, `'modification'`
   - Uses teacher forcing by splitting the action sequence:
     - Input `target`: `action[:, :-1]` (all tokens except last)
     - Output targets: `action[:, 1:]` (all tokens except first)

2. **Updated related functions**:
   - `compute_accuracy()` now unpacks `(batch_inputs, batch_targets)`
   - Training loop in `train_progressive_curriculum()` updated to handle new format
   - Model now compiles automatically in `create_model()`

## Key Changes

### 1. create_dataset() in train_progressive_curriculum.py
```python
def prepare_for_training(data):
    action = data['action']
    inputs = {
        'command': data['command'],
        'target': action[:, :-1],  # Teacher forcing input
        'modification': data['modification']
    }
    targets = action[:, 1:]  # Shifted targets
    return inputs, targets

# Apply mapping
dataset = dataset.map(prepare_for_training, num_parallel_calls=tf.data.AUTOTUNE)
```

### 2. Model Compilation in models.py
```python
# Compile the model with standard settings
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### 3. Updated Training Loop
Changed from:
```python
for batch in dataset:
    inputs = {
        'command': batch['command'],
        'target': batch['action'][:, :-1]
    }
    targets = batch['action'][:, 1:]
```

To:
```python
for batch_inputs, batch_targets in dataset:
    outputs = model(batch_inputs, training=True)
    loss = loss_fn(batch_targets, outputs['logits'])
```

## Testing
Created `test_dataset_fix.py` to verify:
1. Dataset returns correct format
2. Model can successfully call `fit()` with the dataset
3. All required keys are present in the inputs dictionary

## Impact
This fix ensures that:
- The dataset format matches Keras expectations
- Model training can proceed without errors
- Teacher forcing is properly implemented
- The model is pre-compiled with appropriate loss and metrics