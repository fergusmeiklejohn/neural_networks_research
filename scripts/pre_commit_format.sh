#!/bin/bash
# Quick format script to run before git add
# Usage: ./scripts/pre_commit_format.sh [files...]

set -e

# If no arguments, format everything
if [ $# -eq 0 ]; then
    echo "Formatting all Python files..."
    FILES="."
else
    echo "Formatting specified files..."
    FILES="$@"
fi

# Run formatters (order matters - black first, then isort, then autoflake)
black $FILES --line-length=88 2>/dev/null || true
isort $FILES --profile=black --line-length=88 2>/dev/null || true
autoflake --in-place --remove-all-unused-imports --remove-unused-variables --remove-duplicate-keys -r $FILES 2>/dev/null || true

# Run other formatting fixes
pre-commit run trailing-whitespace --files $FILES 2>/dev/null || true
pre-commit run end-of-file-fixer --files $FILES 2>/dev/null || true

echo "✅ Formatting complete!"
