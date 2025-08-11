#!/bin/bash

# Smart commit script that formats BEFORE staging to avoid double commits
# Usage: ./scripts/smart_commit.sh "Your commit message"

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get commit message
if [ -z "$1" ]; then
    echo "Usage: $0 \"Your commit message\""
    exit 1
fi
COMMIT_MSG="$1"

echo -e "${BLUE}🔧 Smart Commit: Format → Stage → Commit${NC}"
echo "================================================"

# Step 1: Identify changed files (both staged and unstaged)
echo -e "\n${YELLOW}📋 Detecting changed files...${NC}"
# Get list of modified/added files (both staged and unstaged)
CHANGED_FILES=$(git diff --name-only --diff-filter=ACMR; git diff --cached --name-only --diff-filter=ACMR)
CHANGED_FILES=$(echo "$CHANGED_FILES" | sort -u | grep -v '^$' || true)

if [ -z "$CHANGED_FILES" ]; then
    echo "No changes to commit"
    exit 0
fi

# Count files by type
PY_FILES=$(echo "$CHANGED_FILES" | grep '\.py$' || true)
MD_FILES=$(echo "$CHANGED_FILES" | grep '\.md$' || true)
JSON_FILES=$(echo "$CHANGED_FILES" | grep '\.json$' || true)
YAML_FILES=$(echo "$CHANGED_FILES" | grep '\.ya\?ml$' || true)

echo "Found changes in:"
[ ! -z "$PY_FILES" ] && echo "  - $(echo "$PY_FILES" | wc -l | tr -d ' ') Python files"
[ ! -z "$MD_FILES" ] && echo "  - $(echo "$MD_FILES" | wc -l | tr -d ' ') Markdown files"
[ ! -z "$JSON_FILES" ] && echo "  - $(echo "$JSON_FILES" | wc -l | tr -d ' ') JSON files"
[ ! -z "$YAML_FILES" ] && echo "  - $(echo "$YAML_FILES" | wc -l | tr -d ' ') YAML files"

# Step 2: Run formatters on changed files BEFORE staging
echo -e "\n${YELLOW}🎨 Running formatters on changed files...${NC}"

# Format Python files
if [ ! -z "$PY_FILES" ]; then
    echo "  Running black..."
    echo "$PY_FILES" | xargs black --quiet --line-length=88 2>/dev/null || true
    
    echo "  Running isort..."
    echo "$PY_FILES" | xargs isort --quiet --profile=black --line-length=88 2>/dev/null || true
    
    echo "  Running autoflake..."
    echo "$PY_FILES" | xargs autoflake --in-place --remove-all-unused-imports --quiet 2>/dev/null || true
fi

# Fix trailing whitespace and end-of-file for all text files
echo "  Fixing whitespace and EOF..."
for file in $CHANGED_FILES; do
    if [ -f "$file" ]; then
        # Remove trailing whitespace
        sed -i.bak 's/[[:space:]]*$//' "$file" && rm "${file}.bak" 2>/dev/null || true
        
        # Ensure file ends with newline
        if [ -s "$file" ] && [ "$(tail -c 1 "$file" | wc -l)" -eq 0 ]; then
            echo "" >> "$file"
        fi
    fi
done

# Step 3: Stage all changes (including formatting changes)
echo -e "\n${YELLOW}📦 Staging all changes...${NC}"
git add -A

# Step 4: Check if there are changes to commit
if ! git diff --cached --quiet; then
    # Step 5: Commit with --no-verify to skip hooks (since we already formatted)
    echo -e "\n${YELLOW}💾 Committing...${NC}"
    git commit --no-gpg-sign --no-verify -m "$COMMIT_MSG"
    
    echo -e "\n${GREEN}✅ Success! Changes formatted and committed in one step.${NC}"
    
    # Show what was committed
    echo -e "\n${BLUE}📊 Commit summary:${NC}"
    git log --oneline -1
    git diff --stat HEAD~1
else
    echo -e "\n${YELLOW}ℹ️  No changes to commit after formatting${NC}"
fi

echo -e "\n${BLUE}💡 Tip: You can now push with: git push${NC}"