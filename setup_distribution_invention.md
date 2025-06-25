# Setting Up Your Mac for Distribution Invention Research

## 1. Create and Activate Conda Environment

```bash
# Create a new environment with Python 3.11 (best compatibility with latest libraries)
[X] conda create -n dist-invention python=3.11 -y

# Activate the environment
[x] conda activate dist-invention
```

## 2. Install Core Deep Learning Libraries

```bash
# [x] Install PyTorch with MPS (Metal Performance Shaders) support for Mac
conda install pytorch torchvision torchaudio -c pytorch -y

# [x] Install JAX with Metal support (experimental but improving)
pip install --upgrade pip
pip install --upgrade "jax[metal]"

# [x] Install TensorFlow with Metal acceleration (Apple Silicon approach)
# For Apple Silicon, we need to use pip for compatible versions
# First ensure we have compatible base dependencies
conda install -c conda-forge numpy scipy -y
# Install TensorFlow for Apple Silicon (these packages manage their own dependencies)
pip install tensorflow-macos tensorflow-metal

# [x] Install Keras 3 (multi-backend support)
pip install --upgrade keras

# [ ] If you encounter version conflicts, clean install TensorFlow:
# pip uninstall tensorflow tensorflow-macos tensorflow-metal keras -y
# pip install tensorflow-macos tensorflow-metal keras
```

## 3. Install Essential Scientific Computing Libraries

```bash
# [x] Install core scientific computing libraries
pip install numpy scipy pandas matplotlib seaborn
pip install scikit-learn scikit-image

# [x] Install Jupyter for notebooks
pip install jupyter notebook jupyterlab ipywidgets

# [x] Install progress bars and utilities
pip install tqdm rich typer click

# [x] Install experiment tracking tools
pip install wandb tensorboard mlflow

# [x] Install testing and code quality tools
pip install pytest pytest-cov black flake8 isort mypy
```

## 4. Install Specialized Libraries for Your Research

```bash
# [x] Install physics simulation libraries
pip install pymunk pygame gym

# [x] Install language task libraries
pip install transformers datasets tokenizers sentencepiece

# [x] Install image generation/manipulation libraries
pip install diffusers accelerate torchmetrics
pip install opencv-python pillow albumentations

# [x] Install symbolic reasoning libraries
pip install sympy networkx

# [x] Install probabilistic programming libraries
pip install pyro-ppl numpyro
```

## 5. Create Project Directory Structure

```bash
# [ ] Create directory structure
mkdir -p {experiments,models,configs,scripts,notebooks,tests,docs}
mkdir -p experiments/{01_physics_worlds,02_compositional_language,03_visual_concepts}
mkdir -p experiments/{04_abstract_reasoning,05_mathematical_extension,06_multimodal_transfer}
mkdir -p models/{core,utils,baselines}
mkdir -p data/{raw,processed,generated}
mkdir -p outputs/{checkpoints,logs,visualizations}

# [ ] Create initial files
touch README.md
touch requirements.txt
touch setup.py
touch .gitignore
```

## 6. Create Initial Configuration Files

### `.gitignore`

```bash
# [ ] Create .gitignore file
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.env

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Data and models
data/raw/*
data/processed/*
outputs/checkpoints/*
*.h5
*.pt
*.pth
*.keras
*.safetensors

# Logs
*.log
wandb/
mlruns/
logs/

# OS
.DS_Store
.idea/
.vscode/

# Keep directory structure
!data/raw/.gitkeep
!data/processed/.gitkeep
!outputs/checkpoints/.gitkeep
EOF

# [ ] Create .gitkeep files
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch outputs/checkpoints/.gitkeep
```

### `setup.py`

```python
# [ ] Create setup.py file
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="distribution-invention",
    version="0.1.0",
    author="Your Name",
    description="Neural networks that invent new distributions",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "keras>=3.0",
        "torch>=2.0",
        "jax[metal]",
        "transformers",
        "wandb",
    ],
)
EOF
```

### `requirements.txt`

```bash
# [ ] Generate requirements file
pip freeze > requirements.txt
```

## 7. Configure Keras Backend

```bash
# [ ] Create keras config directory if it doesn't exist
mkdir -p ~/.keras

# [ ] Set JAX as default backend (you can change this anytime)
cat > ~/.keras/keras.json << 'EOF'
{
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "jax",
    "image_data_format": "channels_last"
}
EOF
```

## 8. Create Development Tools Configuration

### VS Code Settings (if using VS Code)

```bash
# [ ] Create .vscode directory
mkdir -p .vscode

# [ ] Create VS Code settings
cat > .vscode/settings.json << 'EOF'
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "editor.formatOnSave": true,
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true
    }
}
EOF
```

### Pre-commit hooks (optional but recommended)

```bash
# [ ] Install pre-commit
pip install pre-commit

# [ ] Create pre-commit configuration
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=88', '--ignore=E203,W503']
EOF

# [ ] Install pre-commit hooks
pre-commit install
```

## 9. Test Your Setup

Create a test script to verify everything is working:

```bash
# [ ] Create test setup script
cat > test_setup.py << 'EOF'
#!/usr/bin/env python
"""Test script to verify the development environment setup."""

import sys
print(f"Python version: {sys.version}")
```

## 10. Create Your First Experiment Notebook

```bash
cat > notebooks/01_initial_exploration.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Distribution Invention: Initial Exploration\n",
    "Testing our setup and exploring initial ideas"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import keras\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from keras import layers, ops\n",
    "\n",
    "print(f\"Keras version: {keras.__version__}\")\n",
    "print(f\"Backend: {keras.backend.backend()}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Simple test: Can we create a layer that modifies its behavior?\n",
    "class SimpleDistributionModifier(layers.Layer):\n",
    "    def __init__(self, units=32, **kwargs):\n",
    "        super().__init__(**kwargs)\n",
    "        self.units = units\n",
    "        \n",
    "    def build(self, input_shape):\n",
    "        self.base_kernel = self.add_weight(\n",
    "            shape=(input_shape[-1], self.units),\n",
    "            initializer='random_normal',\n",
    "            trainable=True,\n",
    "            name='base_kernel'\n",
    "        )\n",
    "        self.modifier = self.add_weight(\n",
    "            shape=(input_shape[-1], self.units),\n",
    "            initializer='zeros',\n",
    "            trainable=True,\n",
    "            name='modifier'\n",
    "        )\n",
    "        \n",
    "    def call(self, inputs, modification_strength=0.0):\n",
    "        # Apply base transformation\n",
    "        base_output = ops.matmul(inputs, self.base_kernel)\n",
    "        \n",
    "        # Apply modification based on strength\n",
    "        modified_kernel = self.base_kernel + modification_strength * self.modifier\n",
    "        modified_output = ops.matmul(inputs, modified_kernel)\n",
    "        \n",
    "        return modified_output\n",
    "\n",
    "# Test the layer\n",
    "layer = SimpleDistributionModifier(16)\n",
    "test_input = ops.ones((2, 8))\n",
    "output_base = layer(test_input, modification_strength=0.0)\n",
    "output_modified = layer(test_input, modification_strength=1.0)\n",
    "\n",
    "print(f\"Base output shape: {output_base.shape}\")\n",
    "print(f\"Output difference: {ops.mean(ops.abs(output_modified - output_base))}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "dist-invention",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF
```

## 11. Install Performance Monitoring Tools

```bash
# [ ] Install Mac system monitoring tools
pip install psutil py-spy

# [ ] Install ML experiment tracking tools
pip install neptune-client comet-ml

# [ ] Install model profiling tools
pip install torchinfo keras-tuner
```

## 12. Set Up Environment Variables

```bash
# [ ] Create .env file for project
cat > .env << 'EOF'
# Keras backend (jax, tensorflow, or torch)
KERAS_BACKEND=jax

# Experiment tracking
WANDB_PROJECT=distribution-invention
WANDB_ENTITY=your_username

# Hardware settings
PYTORCH_ENABLE_MPS_FALLBACK=1

# Memory settings for JAX
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# Paths
DATA_DIR=./data
OUTPUT_DIR=./outputs
CHECKPOINT_DIR=./outputs/checkpoints
EOF

# [ ] Create a script to load environment variables
cat > scripts/setup_env.sh << 'EOF'
#!/bin/bash
# Source this file to set up environment variables
# Usage: source scripts/setup_env.sh

export $(grep -v '^#' .env | xargs)
echo "Environment variables loaded from .env"
EOF

# [ ] Make the script executable
chmod +x scripts/setup_env.sh
```

## 13. VS Code Setup for Development

VS Code provides an excellent environment for this research, combining
notebook-style interactive development with full IDE features.

### Install VS Code Extensions

```bash
# [ ] Install essential Python extensions
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code --install-extension ms-toolsai.jupyter-keymap
code --install-extension ms-toolsai.jupyter-renderers
code --install-extension ms-toolsai.vscode-jupyter-cell-tags
code --install-extension ms-toolsai.vscode-jupyter-slideshow
```

### Create Interactive Python Files

VS Code supports "Python Interactive" files (`.py` files with special cell
markers). Create an example:

```bash
cat > experiments/interactive_dev.py << 'EOF'
# %%
"""
Distribution Invention: Interactive Development
This file can be run cell-by-cell like a notebook in VS Code
"""

import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, ops

print(f"Keras version: {keras.__version__}")
print(f"Backend: {keras.backend.backend()}")

# %% [markdown]
# ## Test Simple Distribution Modifier
# Let's create a layer that can modify its behavior based on input

# %%
class SimpleDistributionModifier(layers.Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.base_kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True,
            name='base_kernel'
        )
        self.modifier = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='zeros',
            trainable=True,
            name='modifier'
        )
        
    def call(self, inputs, modification_strength=0.0):
        # Apply base transformation
        base_output = ops.matmul(inputs, self.base_kernel)
        
        # Apply modification based on strength
        modified_kernel = self.base_kernel + modification_strength * self.modifier
        modified_output = ops.matmul(inputs, modified_kernel)
        
        return modified_output

# %%
# Test the layer
layer = SimpleDistributionModifier(16)
test_input = ops.ones((2, 8))

# Test different modification strengths
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for i, strength in enumerate([0.0, 0.5, 1.0]):
    output = layer(test_input, modification_strength=strength)
    axes[i].imshow(output.numpy(), cmap='viridis')
    axes[i].set_title(f'Modification Strength: {strength}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Next: Implement Distribution Generator

# %%
class DistributionGenerator(keras.Model):
    """Early prototype of distribution generation model"""
    
    def __init__(self, latent_dim=64, num_rules=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_rules = num_rules
        
        # Rule encoder
        self.rule_encoder = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(latent_dim * num_rules),
            layers.Reshape((num_rules, latent_dim))
        ])
        
        # Rule modifier
        self.rule_modifier = layers.LSTM(latent_dim, return_sequences=True)
        
        # Distribution builder
        self.distribution_builder = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(64)
        ])
        
    def call(self, inputs, modification_mask=None):
        # Encode base rules
        rules = self.rule_encoder(inputs)
        
        # Apply modifications if mask provided
        if modification_mask is not None:
            modified_rules = self.rule_modifier(rules)
            # Selective modification based on mask
            rules = rules * (1 - modification_mask) + modified_rules * modification_mask
            
        # Build distribution parameters
        dist_params = self.distribution_builder(layers.Flatten()(rules))
        
        return dist_params, rules

# %%
# Test the distribution generator
model = DistributionGenerator()
test_input = keras.random.normal((4, 32))
modification_mask = ops.zeros((4, 10, 1))  # No modifications initially

dist_params, rules = model(test_input, modification_mask)
print(f"Distribution parameters shape: {dist_params.shape}")
print(f"Rules shape: {rules.shape}")
EOF
```

### Add Launch Configuration

```bash
cat > .vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": false,
            "env": {
                "KERAS_BACKEND": "jax"
            }
        },
        {
            "name": "Python: Train Model",
            "type": "python",
            "request": "launch",
            "module": "scripts.train",
            "args": ["--experiment", "physics_worlds"],
            "console": "integratedTerminal"
        }
    ]
}
EOF
```

### Add Tasks Configuration

```bash
cat > .vscode/tasks.json << 'EOF'
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Run Tests",
            "type": "shell",
            "command": "pytest tests/ -v",
            "group": {
                "kind": "test",
                "isDefault": true
            }
        },
        {
            "label": "Format Code",
            "type": "shell",
            "command": "black . && isort .",
            "group": "build"
        }
    ]
}
EOF
```

### VS Code Workflow Tips

**Running Interactive Python Files:**

- **Run Cell**: Click "Run Cell" or press `Shift+Enter` on any `# %%` cell
- **Run All Above**: Right-click → "Run All Above"
- **Interactive Window**: Opens automatically when you run cells

**Key Shortcuts:**

- `Shift+Enter`: Run current cell
- `Cmd+Enter`: Run current cell and stay
- `Option+Enter`: Run current cell and insert below
- `Cmd+Shift+P`: Command palette
- `Cmd+Shift+E`: Explorer
- `Cmd+Shift+F`: Search across files
- `Cmd+Shift+D`: Debug panel

**Best Practices:**

1. Use `.py` files with cells for main development (version control friendly)
2. Use `.ipynb` notebooks for final presentations/sharing
3. Create reusable components in proper module files
4. Write tests as you develop

## Quick Verification Checklist

Run these commands to verify everything is set up correctly:

```bash
# 1. Check Python version
python --version  # Should be 3.11.x

# 2. Check GPU/Metal availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# 3. Check Keras backend
python -c "import keras; print(f'Keras backend: {keras.backend.backend()}')"

# 4. Test JAX Metal
python -c "import jax; print(f'JAX devices: {jax.devices()}')"

# 5. List installed packages
pip list | grep -E "(keras|torch|jax|tensorflow)"

# 6. Open VS Code in project directory
code .
```

## Next Steps

1. **Select Python Interpreter in VS Code**:
   - Open Command Palette: `Cmd+Shift+P`
   - Type: "Python: Select Interpreter"
   - Choose: `dist-invention` conda environment

2. **Open the interactive development file** in VS Code and start experimenting
   with cells

3. **Set up Weights & Biases** (optional but recommended):
   ```bash
   wandb login
   ```

4. **Clone relevant datasets** for your first experiment:
   ```bash
   # Example: Download a physics dataset
   cd data/raw
   # Add your dataset downloads here
   ```

Your Mac is now fully set up for distribution invention research! The
environment supports all three major backends (JAX, PyTorch, TensorFlow) through
Keras 3, giving you maximum flexibility. VS Code provides an excellent
development environment combining the interactivity of notebooks with the power
of a full IDE.

### Configure VS Code for Your Project

Update `.vscode/settings.json` (already created above) with Jupyter-specific
settings:

```json
{
    // [ ] Set default Python interpreter path
    "python.defaultInterpreterPath": "~/miniconda3/envs/dist-invention/bin/python",

    // [ ] Activate environment in terminal
    "python.terminal.activateEnvironment": true,

    // [ ] Enable linting
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,

    // [ ] Set formatting provider
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,

    // [ ] Configure testing
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],

    // [ ] Configure Jupyter settings
    "jupyter.askForKernelRestart": false,
    "jupyter.interactiveWindow.textEditor.executeSelection": true,

    // [ ] Configure file exclusions
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".ipynb_checkpoints": true
    },

    // [ ] Configure Python-specific editor settings
    "[python]": {
        "editor.rulers": [88],
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        }
    },

    // [ ] Configure notebook cell toolbar location
    "notebook.cellToolbarLocation": {
        "default": "right",
        "jupyter-notebook": "left"
    }
}
```

### Create Interactive Python Files

```bash
# [ ] Create interactive Python exploration files
mkdir -p scripts/explorations
```
