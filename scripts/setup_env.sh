#!/bin/bash
# Source this file to set up environment variables
# Usage: source scripts/setup_env.sh

export $(grep -v '^#' .env | xargs)
echo "Environment variables loaded from .env"
