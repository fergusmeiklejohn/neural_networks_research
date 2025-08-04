#!/bin/bash
# Script to activate conda environment and run tests

echo "🚀 Activating dist-invention environment and running tests..."
echo "============================================================"

# Initialize conda (try multiple common locations)
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
elif [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then
    source /opt/miniconda3/etc/profile.d/conda.sh
else
    echo "⚠️  Could not find conda initialization script"
    echo "   Please manually run: source ~/miniconda3/etc/profile.d/conda.sh"
fi

# Activate environment
echo "Activating dist-invention environment..."
conda activate dist-invention

# Check if activation was successful
if [ "$CONDA_DEFAULT_ENV" = "dist-invention" ]; then
    echo "✅ Successfully activated dist-invention environment"
else
    echo "❌ Failed to activate dist-invention environment"
    echo "   Current environment: $CONDA_DEFAULT_ENV"
    echo "   Please manually run: conda activate dist-invention"
    exit 1
fi

# Check environment
echo "Checking environment..."
python check_environment.py

# If environment check passes, run tests
if [ $? -eq 0 ]; then
    echo ""
    echo "Environment check passed! Running physics experiment tests..."
    cd experiments/01_physics_worlds
    python simple_test.py
else
    echo ""
    echo "Environment check failed. Please fix the issues above before proceeding."
fi
