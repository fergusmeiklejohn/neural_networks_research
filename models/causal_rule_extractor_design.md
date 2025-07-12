# CausalRuleExtractor Implementation Design

## Overview

The CausalRuleExtractor is a core component of our Distribution Inventor architecture. Based on reviewer feedback, we're implementing it using Causal Generative Neural Networks (CGNNs) with independence-guided encoding.

## Architecture Details

### 1. Base Architecture: CGNN-Inspired Design

Based on the CGNN paper, we incorporate Maximum Mean Discrepancy (MMD) loss and topological ordering:

```python
class CausalRuleExtractor(keras.Model):
    def __init__(self, 
                 n_causal_variables=32,
                 n_mechanisms=16,
                 hidden_dim=256,
                 n_heads=8,
                 mmd_kernel='rbf'):
        super().__init__()
        
        # Causal variable encoder
        self.variable_encoder = CausalVariableEncoder(
            n_variables=n_causal_variables,
            hidden_dim=hidden_dim
        )
        
        # Mechanism identifier with attention
        self.mechanism_attention = keras.layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=hidden_dim // n_heads
        )
        
        # Independence-guided separator
        self.independence_separator = IndependenceLoss()
        
        # Causal graph predictor
        self.graph_predictor = CausalGraphPredictor(
            n_variables=n_causal_variables
        )
        
        # MMD loss for distribution matching (from CGNN)
        self.mmd_loss = MaximumMeanDiscrepancy(kernel=mmd_kernel)
```

### 2. Key Components

#### 2.1 CausalVariableEncoder
Learns disentangled representations of causal variables:
```python
class CausalVariableEncoder(keras.layers.Layer):
    def __init__(self, n_variables, hidden_dim):
        self.encoders = [
            keras.Sequential([
                keras.layers.Dense(hidden_dim),
                keras.layers.LayerNormalization(),
                keras.layers.ReLU(),
                keras.layers.Dense(hidden_dim // 2)
            ]) for _ in range(n_variables)
        ]
        
    def call(self, inputs):
        # Encode each potential causal variable separately
        variables = []
        for i, encoder in enumerate(self.encoders):
            var_rep = encoder(inputs)
            variables.append(var_rep)
        return tf.stack(variables, axis=1)
```

#### 2.2 Independence Loss
Ensures causal variables are maximally independent:
```python
class IndependenceLoss(keras.layers.Layer):
    def compute_mutual_information(self, x, y):
        # Estimate MI using MINE or similar
        return estimated_mi
        
    def independence_loss(self, variables):
        # Minimize pairwise mutual information
        loss = 0
        for i in range(len(variables)):
            for j in range(i+1, len(variables)):
                loss += self.compute_mutual_information(
                    variables[i], variables[j]
                )
        return loss
```

#### 2.3 Mechanism Attention
Identifies which variables participate in which mechanisms:
```python
class MechanismIdentifier(keras.layers.Layer):
    def __init__(self, n_mechanisms, hidden_dim):
        self.mechanism_queries = self.add_weight(
            shape=(n_mechanisms, hidden_dim),
            initializer='glorot_uniform'
        )
        
    def call(self, causal_variables):
        # Attention to identify mechanism participation
        attention_weights = keras.layers.Attention()(
            [self.mechanism_queries, causal_variables]
        )
        return attention_weights
```

### 3. Training Strategy

#### 3.1 Multi-Task Learning
Train on multiple objectives simultaneously:
1. **Reconstruction**: Ensure extracted rules can reconstruct data
2. **Independence**: Maximize independence between causal variables
3. **Intervention Prediction**: Predict effects of interventions
4. **Counterfactual Generation**: Generate valid counterfactuals
5. **MMD Loss** (from CGNN): Minimize discrepancy between generated and observed distributions

#### 3.2 Progressive Curriculum
1. **Stage 1**: Learn basic variable disentanglement
2. **Stage 2**: Identify causal relationships
3. **Stage 3**: Extract modifiable mechanisms
4. **Stage 4**: Validate through interventions

### 4. Integration with Distribution Inventor

```python
class DistributionInventor(keras.Model):
    def __init__(self):
        # Extract causal rules from data
        self.rule_extractor = CausalRuleExtractor()
        
        # Modify specific rules
        self.rule_modifier = SelectiveRuleModifier()
        
        # Generate new distribution
        self.distribution_generator = DistributionGenerator()
        
    def extract_and_modify_rules(self, data, modification_request):
        # Extract current rules
        causal_vars, mechanisms = self.rule_extractor(data)
        
        # Apply requested modifications
        modified_rules = self.rule_modifier(
            causal_vars, mechanisms, modification_request
        )
        
        # Generate new distribution
        new_distribution = self.distribution_generator(modified_rules)
        
        return new_distribution
```

### 5. Evaluation Metrics

1. **Disentanglement Score**: MIG (Mutual Information Gap)
2. **Causal Discovery Accuracy**: Compare to ground truth graphs
3. **Intervention Accuracy**: Predict intervention effects
4. **Modification Fidelity**: Only specified rules change

### 6. Implementation Notes

- Use Keras 3 for multi-backend support
- Leverage JAX for efficient graph operations
- Implement custom losses as Keras layers
- Use tf.function for performance optimization

### 7. Fallback Options

If pure neural approach struggles:
1. **Hybrid Approach**: Combine with symbolic causal discovery
2. **Structured Priors**: Use domain knowledge to guide extraction
3. **Program Synthesis**: Extract rules as programs
4. **Meta-Learning**: Learn to extract rules from few examples

#### 2.4 Maximum Mean Discrepancy (from CGNN)
```python
class MaximumMeanDiscrepancy(keras.layers.Layer):
    def __init__(self, kernel='rbf'):
        super().__init__()
        self.kernel = kernel
        
    def compute_mmd(self, X, Y):
        """Compute MMD between distributions X and Y"""
        if self.kernel == 'rbf':
            # RBF kernel with multiple bandwidths
            XX = self.kernel_matrix(X, X)
            YY = self.kernel_matrix(Y, Y)
            XY = self.kernel_matrix(X, Y)
            
            # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
            mmd = tf.reduce_mean(XX) + tf.reduce_mean(YY) - 2*tf.reduce_mean(XY)
            return mmd
    
    def kernel_matrix(self, X, Y):
        """Compute RBF kernel matrix"""
        # Multiple bandwidth RBF kernel
        bandwidths = [0.01, 0.1, 1.0, 10.0, 100.0]
        kernel_val = 0
        for bw in bandwidths:
            kernel_val += tf.exp(-tf.square(X - Y) / (2 * bw**2))
        return kernel_val / len(bandwidths)
```

## References

- Goudet et al. (2018): "Learning Functional Causal Models with Generative Neural Networks"
- Bengio et al. (2019): "A Meta-Transfer Objective for Learning to Disentangle Causal Mechanisms"
- Schölkopf et al. (2021): "Toward Causal Representation Learning"