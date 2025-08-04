#!/bin/bash
# Automatic research diary update script

set -e  # Exit on error

# Get today's date
TODAY=$(date +%Y-%m-%d)
DIARY_FILE="research_diary/${TODAY}_research_diary.md"

# Function to create new diary entry
create_diary_entry() {
    echo "📝 Creating new research diary entry for $TODAY..."

    # Get recent git commits for context
    RECENT_COMMITS=$(git log --oneline -10 --pretty=format:"- %s")

    # Get list of modified files today
    MODIFIED_FILES=$(git diff --name-only HEAD~5 HEAD | grep -v "\.pyc" | head -20)

    cat > "$DIARY_FILE" << EOF
# Research Diary: $(date +"%B %d, %Y")

## Today's Focus: [TO BE FILLED]

### Summary
[Brief summary of today's work - TO BE FILLED]

### Key Accomplishments
[List key accomplishments - TO BE FILLED]

### Recent Git Activity
$RECENT_COMMITS

### Files Modified
$(echo "$MODIFIED_FILES" | sed 's/^/- /')

### Technical Details
[Technical insights and implementation details - TO BE FILLED]

### Challenges Encountered
[List any challenges or blockers - TO BE FILLED]

### Results and Metrics
[Quantitative results from experiments - TO BE FILLED]

### Next Steps (Actionable for Tomorrow)
1. **Immediate Priority**: [Specific task with file paths and commands]
2. **Secondary Tasks**: [Additional tasks with context]
3. **Open Questions**: [Questions to investigate with hypotheses]

### Key Code Changes
[Important code snippets or architectural changes - TO BE FILLED]

### Notes for Tomorrow
- Start from: [Specific file and line number]
- Run: [Exact commands to execute]
- Check: [Things to verify or test]
EOF

    echo "✅ Created new diary entry: $DIARY_FILE"
}

# Function to update existing diary entry
update_diary_entry() {
    echo "📝 Updating existing research diary entry for $TODAY..."

    # Create a backup
    cp "$DIARY_FILE" "${DIARY_FILE}.bak"

    # Get the latest experiment status
    EXPERIMENT_STATUS=""
    if [ -f "experiments/01_physics_worlds/CURRENT_STATUS.md" ]; then
        EXPERIMENT_STATUS=$(head -20 experiments/01_physics_worlds/CURRENT_STATUS.md | grep -E "^\*\*" | head -3)
    fi

    # Append update section
    cat >> "$DIARY_FILE" << EOF

## Update: $(date +"%H:%M")

### Current Status
$EXPERIMENT_STATUS

### Latest Activity
- Working directory: $(pwd)
- Active branch: $(git branch --show-current)
- Uncommitted changes: $(git status --porcelain | wc -l | tr -d ' ') files

### Auto-generated Reminders
- Remember to update CURRENT_STATUS.md if experiment state changed
- Consider running tests before major commits
- Document any new insights in appropriate analysis files
EOF

    echo "✅ Updated diary entry: $DIARY_FILE"
}

# Main logic
if [ -f "$DIARY_FILE" ]; then
    update_diary_entry
else
    create_diary_entry
fi

# Optional: Open in editor
read -p "Open diary in editor? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v code &> /dev/null; then
        code "$DIARY_FILE"
    elif command -v nano &> /dev/null; then
        nano "$DIARY_FILE"
    else
        vi "$DIARY_FILE"
    fi
fi
