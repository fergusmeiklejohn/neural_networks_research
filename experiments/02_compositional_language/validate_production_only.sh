#!/bin/bash
# The only validation that matters: Is your code in production?

if [ -n "$(git diff origin/production)" ]; then
    echo "❌ STOP! You have local changes not in production."
    echo ""
    echo "Push to production FIRST:"
    echo "  git add -A"
    echo "  git commit -m 'your message'"
    echo "  git push origin $(git branch --show-current)"
    echo "  gh pr create && gh pr merge"
    echo ""
    echo "THEN deploy to Paperspace."
    exit 1
else
    echo "✅ Local matches production - safe to deploy!"
    echo ""
    echo "On Paperspace, run:"
    echo "  git pull origin production"
    echo "  python paperspace_train_with_safeguards.py"
fi
