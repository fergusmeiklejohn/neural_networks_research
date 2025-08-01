"""Test simple binding patterns to understand what's being learned"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

class SimpleVariableBindingModel(nn.Module):
    """Minimal model to test variable binding concept"""
    
    def __init__(self, vocab_size=10, embed_dim=32, num_slots=3):
        super().__init__()
        
        # Simple components
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Slot memory (learnable)
        self.slot_memory = nn.Linear(embed_dim, embed_dim * num_slots)
        
        # Binding network
        self.bind_net = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_slots)
        )
        
        # Action decoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # 4 actions
        )
        
        self.num_slots = num_slots
        self.embed_dim = embed_dim
        
    def __call__(self, sentence):
        # sentence shape: (batch, seq_len)
        embeds = self.embedding(sentence)  # (batch, seq_len, embed_dim)
        
        # For each position, decide which slot to use
        binding_logits = self.bind_net(embeds)  # (batch, seq_len, num_slots)
        binding_probs = mx.softmax(binding_logits, axis=-1)
        
        # Get slot contents (same for all positions in a sentence)
        # Use first word embedding as "key" to generate slot contents
        key = embeds[:, 0, :]  # (batch, embed_dim)
        slot_contents = self.slot_memory(key)  # (batch, embed_dim * num_slots)
        slot_contents = slot_contents.reshape(-1, self.num_slots, self.embed_dim)
        
        # Retrieve content from slots using soft attention
        batch_size, seq_len = sentence.shape
        slot_contents_exp = slot_contents[:, None, :, :]  # (batch, 1, num_slots, embed_dim)
        slot_contents_exp = mx.broadcast_to(slot_contents_exp, 
                                           (batch_size, seq_len, self.num_slots, self.embed_dim))
        
        binding_probs_exp = binding_probs[:, :, :, None]  # (batch, seq_len, num_slots, 1)
        retrieved = mx.sum(slot_contents_exp * binding_probs_exp, axis=2)  # (batch, seq_len, embed_dim)
        
        # Decode to actions
        action_logits = self.decoder(retrieved)  # (batch, seq_len, 4)
        
        return action_logits, binding_probs

def generate_simple_data(batch_size=32):
    """Generate very simple binding data
    
    Pattern: [VAR, MEANS, ACTION, DO, VAR]
    VAR: 1=X, 2=Y
    ACTION: 3=jump, 4=walk
    MEANS: 5
    DO: 6
    """
    sentences = []
    labels = []
    
    for _ in range(batch_size):
        var = np.random.choice([1, 2])  # X or Y
        action = np.random.choice([3, 4])  # jump or walk
        
        sentence = [var, 5, action, 6, var]  # VAR means ACTION do VAR
        label = [action - 2]  # Convert to action index (1 or 2)
        
        sentences.append(sentence)
        labels.append(label)
    
    return mx.array(np.array(sentences, dtype=np.int32)), mx.array(np.array(labels, dtype=np.int32))

def train_simple_binding():
    """Train the simple model"""
    model = SimpleVariableBindingModel()
    optimizer = optim.Adam(learning_rate=0.01)
    
    print("Training simple variable binding model...")
    print("Pattern: X/Y means jump/walk, do X/Y")
    print()
    
    for epoch in range(100):
        total_loss = 0
        total_correct = 0
        total_count = 0
        
        for _ in range(20):  # 20 batches per epoch
            sentences, labels = generate_simple_data(32)
            
            def loss_fn(model):
                logits, bindings = model(sentences)
                
                # Only look at last position (where action should be predicted)
                last_logits = logits[:, -1, :]  # (batch, 4)
                last_labels = labels[:, 0]  # (batch,)
                
                # Cross entropy loss
                ce_loss = mx.mean(nn.losses.cross_entropy(last_logits, last_labels))
                
                # Binding consistency loss - same variable should bind to same slot
                first_binding = bindings[:, 0, :]  # Binding for first position
                last_binding = bindings[:, -1, :]  # Binding for last position
                
                # KL divergence to encourage consistency
                kl_loss = mx.mean(mx.sum(first_binding * mx.log(first_binding / (last_binding + 1e-8) + 1e-8), axis=1))
                
                total_loss = ce_loss + 0.1 * kl_loss
                
                return total_loss, (ce_loss, kl_loss, last_logits, last_labels)
            
            # Compute gradients
            def grad_fn(model):
                loss, aux = loss_fn(model)
                return loss
            
            value_and_grad_fn = mx.value_and_grad(grad_fn)
            loss, grads = value_and_grad_fn(model)
            
            # Update
            optimizer.update(model, grads)
            
            # Get metrics
            _, (ce_loss, kl_loss, last_logits, last_labels) = loss_fn(model)
            
            # Accuracy
            preds = mx.argmax(last_logits, axis=1)
            correct = mx.sum(preds == last_labels)
            
            total_loss += loss.item()
            total_correct += correct.item()
            total_count += len(sentences)
        
        if epoch % 10 == 0:
            accuracy = total_correct / total_count
            print(f"Epoch {epoch}: Loss={total_loss/20:.3f}, Accuracy={accuracy:.1%}")
    
    # Test the model
    print("\n=== Testing ===")
    test_cases = [
        ([1, 5, 3, 6, 1], "X means jump do X", 1),  # Should predict jump (1)
        ([1, 5, 4, 6, 1], "X means walk do X", 2),  # Should predict walk (2)
        ([2, 5, 3, 6, 2], "Y means jump do Y", 1),  # Should predict jump (1)
        ([2, 5, 4, 6, 2], "Y means walk do Y", 2),  # Should predict walk (2)
    ]
    
    for sentence, description, expected in test_cases:
        sentence_mx = mx.array([sentence])
        logits, bindings = model(sentence_mx)
        
        pred = mx.argmax(logits[0, -1, :]).item()
        binding_pattern = mx.argmax(bindings[0], axis=1).tolist()
        
        correct = "✓" if pred == expected else "✗"
        print(f"{description:20} -> {pred} (expected {expected}) {correct}")
        print(f"  Binding pattern: {binding_pattern}")
        print(f"  Consistent: {'✓' if binding_pattern[0] == binding_pattern[-1] else '✗'}")
        print()

if __name__ == "__main__":
    train_simple_binding()