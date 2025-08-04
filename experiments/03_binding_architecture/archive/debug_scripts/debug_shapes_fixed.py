"""Debug shapes in MLX-compatible sequential model."""

import mlx.core as mx

# Simulate the shapes we expect
batch_size = 16
num_slots = 4
embed_dim = 128

# Create test data
binding_scores = mx.random.uniform(shape=(batch_size, num_slots))
slot_values = mx.random.uniform(shape=(batch_size, num_slots, embed_dim))

print("Initial shapes:")
print(f"binding_scores: {binding_scores.shape}")
print(f"slot_values: {slot_values.shape}")

# Current approach (that's failing)
print("\nCurrent approach:")
binding_scores_unsqueezed = binding_scores[:, None, :]
print(f"binding_scores_unsqueezed: {binding_scores_unsqueezed.shape}")

# Matrix multiplication
result = binding_scores_unsqueezed @ slot_values
print(f"result shape: {result.shape}")
print(f"Can squeeze axis 1? {result.shape[1] == 1}")

# If shape is wrong, let's try different approach
if result.shape[1] != 1:
    print("\nAlternative approach (transpose):")
    # We want: (batch, 1, num_slots) @ (batch, num_slots, embed_dim) = (batch, 1, embed_dim)
    # But we're getting: (batch, 1, num_slots) @ (batch, num_slots, embed_dim) = (batch, embed_dim, embed_dim)?
    
    # Let's check what's happening step by step
    print(f"binding_scores_unsqueezed: {binding_scores_unsqueezed.shape}")  # (batch, 1, num_slots)
    print(f"slot_values: {slot_values.shape}")  # (batch, num_slots, embed_dim)
    
    # For batch matrix multiply, the last two dimensions are used
    # (batch, 1, num_slots) @ (batch, num_slots, embed_dim) should give (batch, 1, embed_dim)
    
    # Try element-wise multiply and sum
    print("\nElement-wise approach:")
    # Expand binding_scores to match slot_values shape
    binding_expanded = binding_scores[:, :, None]  # (batch, num_slots, 1)
    print(f"binding_expanded: {binding_expanded.shape}")
    
    # Element-wise multiply and sum over slots
    weighted = slot_values * binding_expanded  # (batch, num_slots, embed_dim)
    print(f"weighted: {weighted.shape}")
    
    retrieved = mx.sum(weighted, axis=1)  # (batch, embed_dim)
    print(f"retrieved: {retrieved.shape}")