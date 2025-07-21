#!/bin/bash
# Daily merge script - pushes current branch to origin, creates PR, and merges to production

set -e  # Exit on error

echo "🚀 Starting daily merge process..."

# 0. Update research diary (optional)
read -p "📝 Update research diary before merge? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ./scripts/update_research_diary.sh
fi

# 1. Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "❌ You have uncommitted changes. Please commit them first."
    exit 1
fi

# 1.5 Run pre-merge tests
echo "🧪 Running pre-merge tests..."
if ./scripts/pre_merge_tests.sh; then
    echo "✅ Pre-merge tests passed"
else
    echo "❌ Pre-merge tests failed. Check output above."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 2. Get current branch name
CURRENT_BRANCH=$(git branch --show-current)
echo "📌 Current branch: $CURRENT_BRANCH"

if [ "$CURRENT_BRANCH" == "production" ]; then
    echo "❌ You're on production branch. Switch to a feature branch first."
    exit 1
fi

# 3. Push to origin
echo "📤 Pushing to origin..."
git push -u origin "$CURRENT_BRANCH"

# 4. Generate commit summary for PR body
echo "📝 Generating PR description..."
COMMITS=$(git log origin/production.."$CURRENT_BRANCH" --oneline)
TODAY=$(date +%Y-%m-%d)

PR_BODY="## Daily Merge - $TODAY

### Commits included:
\`\`\`
$COMMITS
\`\`\`

### Recent work:
$(git log origin/production.."$CURRENT_BRANCH" --pretty=format:"- %s" | head -10)

🤖 Generated with [Claude Code](https://claude.ai/code)"

# 5. Create PR
echo "🔄 Creating pull request..."
PR_URL=$(gh pr create \
    --base production \
    --head "$CURRENT_BRANCH" \
    --title "Daily merge: $CURRENT_BRANCH → production ($TODAY)" \
    --body "$PR_BODY" \
    2>&1 | grep -E "https://github.com/.*/pull/[0-9]+")

if [ -z "$PR_URL" ]; then
    echo "❌ Failed to create PR"
    exit 1
fi

echo "✅ PR created: $PR_URL"

# 6. Extract PR number
PR_NUMBER=$(echo "$PR_URL" | grep -oE "[0-9]+$")

# 7. Merge PR
echo "🔀 Merging PR #$PR_NUMBER..."
gh pr merge "$PR_NUMBER" --merge --delete-branch=false

# 8. Update local branch
echo "📥 Updating local branches..."
git fetch origin
git pull origin "$CURRENT_BRANCH"

echo "✅ Daily merge complete!"
echo "📊 Summary:"
echo "  - Branch: $CURRENT_BRANCH"
echo "  - PR: #$PR_NUMBER"
echo "  - URL: $PR_URL"