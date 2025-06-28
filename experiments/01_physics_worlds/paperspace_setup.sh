#!/bin/bash
# Setup script for Paperspace GPU training

echo "Setting up Paperspace environment for progressive curriculum training..."

# Create output directories
mkdir -p /workspace/outputs/checkpoints
mkdir -p /workspace/outputs/logs
mkdir -p /workspace/outputs/metrics

# Install dependencies (if not already installed)
pip install tensorflow wandb tqdm

# Copy code to workspace (assuming code is in /storage)
if [ -d "/storage/neural_networks_research" ]; then
    echo "Copying code from persistent storage..."
    cp -r /storage/neural_networks_research /workspace/
else
    echo "Code directory not found in /storage. Please upload your code."
fi

# Login to wandb (you'll need to set your API key)
# wandb login YOUR_API_KEY_HERE

echo "Setup complete!"
echo ""
echo "To run the training:"
echo "cd /workspace/neural_networks_research/experiments/01_physics_worlds"
echo "python train_progressive_paperspace.py"
echo ""
echo "Monitor with wandb at: https://wandb.ai/YOUR_USERNAME/physics-worlds-extrapolation"