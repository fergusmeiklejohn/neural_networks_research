#!/bin/bash
# Format code before staging and committing
# Usage: ./scripts/format_and_commit.sh "commit message"

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Running code formatters before staging...${NC}"

# Run formatters on all Python files (these auto-fix)
echo -e "${YELLOW}Running black...${NC}"
black . --line-length=88 2>/dev/null || true

echo -e "${YELLOW}Running isort...${NC}"
isort . --profile=black --line-length=88 2>/dev/null || true

echo -e "${YELLOW}Running autoflake...${NC}"
autoflake --in-place --remove-all-unused-imports --remove-unused-variables --remove-duplicate-keys -r . 2>/dev/null || true

# Run pre-commit hooks to catch any other formatting issues
echo -e "${YELLOW}Running other formatting hooks...${NC}"
pre-commit run trailing-whitespace --all-files 2>/dev/null || true
pre-commit run end-of-file-fixer --all-files 2>/dev/null || true
pre-commit run mixed-line-ending --all-files 2>/dev/null || true

echo -e "${GREEN}Formatting complete!${NC}"

# Show what changed
echo -e "\n${YELLOW}Changes after formatting:${NC}"
git status --short

# Stage all changes
echo -e "\n${GREEN}Staging all changes...${NC}"
git add -A

# Show what's staged
echo -e "\n${YELLOW}Staged changes:${NC}"
git status --short --branch

# Commit if message provided
if [ -n "$1" ]; then
    echo -e "\n${GREEN}Committing with message...${NC}"
    git commit --no-gpg-sign -m "$1"
    echo -e "${GREEN}✅ Commit successful!${NC}"
else
    echo -e "\n${YELLOW}Ready to commit. Run:${NC}"
    echo "git commit --no-gpg-sign -m \"your message\""
fi
