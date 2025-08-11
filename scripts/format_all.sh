#!/bin/bash

# Format all changed files without committing
# Usage: ./scripts/format_all.sh

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🎨 Formatting all changed files${NC}"
echo "================================"

# Get list of changed files
CHANGED_FILES=$(git diff --name-only --diff-filter=ACMR; git diff --cached --name-only --diff-filter=ACMR)
CHANGED_FILES=$(echo "$CHANGED_FILES" | sort -u | grep -v '^$' || true)

if [ -z "$CHANGED_FILES" ]; then
    echo "No changed files to format"
    exit 0
fi

# Separate by type
PY_FILES=$(echo "$CHANGED_FILES" | grep '\.py$' || true)
MD_FILES=$(echo "$CHANGED_FILES" | grep '\.md$' || true)
JSON_FILES=$(echo "$CHANGED_FILES" | grep '\.json$' || true)

# Format Python files
if [ ! -z "$PY_FILES" ]; then
    echo -e "\n${YELLOW}Python files:${NC}"
    echo "$PY_FILES" | sed 's/^/  /'
    
    echo -e "\n  Running black..."
    echo "$PY_FILES" | xargs black --line-length=88
    
    echo "  Running isort..."
    echo "$PY_FILES" | xargs isort --profile=black --line-length=88
    
    echo "  Running autoflake..."
    echo "$PY_FILES" | xargs autoflake --in-place --remove-all-unused-imports
fi

# Fix whitespace for all files
echo -e "\n${YELLOW}Fixing whitespace and EOF for all files...${NC}"
for file in $CHANGED_FILES; do
    if [ -f "$file" ]; then
        # Remove trailing whitespace
        sed -i.bak 's/[[:space:]]*$//' "$file" && rm "${file}.bak" 2>/dev/null || true
        
        # Ensure file ends with newline
        if [ -s "$file" ] && [ "$(tail -c 1 "$file" | wc -l)" -eq 0 ]; then
            echo "" >> "$file"
        fi
        echo "  ✓ $file"
    fi
done

echo -e "\n${GREEN}✅ Formatting complete!${NC}"
echo -e "${BLUE}💡 Now you can stage and commit without format changes:${NC}"
echo "  git add -A"
echo "  git commit --no-gpg-sign -m \"Your message\""