# Git Workflow

## Quick Reference

Before starting any new work:

```bash
# Check outstanding work on current branch
git status
git log --oneline -3
BRANCH=$(git branch --show-current)
BASE_BRANCH=$(gh repo view --json defaultBranchRef --jq '.defaultBranchRef.name')
git log --oneline "origin/$BASE_BRANCH..origin/$BRANCH"
gh pr list --head "$BRANCH" --state open --json number,url --jq '.[] | [.number, .url] | @tsv'

# Start new work from fresh branch off main
git checkout main && git pull origin main && git checkout -b <new-branch>
```

## Rules

- Start every new piece of work from a fresh branch off main
- If outstanding work exists (unmerged PR, unpushed commits), finish that work first
- origin/main is protected — all changes go through PRs
- All development must be done on a branch. origin/main is protected

## Making Commits

When creating commits, follow the existing commit style in the repository. Each commit should be a focused atomic unit — a single logical change that can be reviewed and understood in isolation. If you are making changes to multiple unrelated concerns, split them into separate commits.

## Updating a PR Description

The ``gh pr edit --body`` flag can silently fail (e.g., when the remote URL is stale after a repo rename). To reliably update a PR's body, write it to a file and use the API directly instead:

```bash
# Write the new body to a file
cat > /tmp/pr_body.md << 'EOF'
## Summary
...
EOF

# Update body
gh api "repos/$(gh repo view --json owner,name --jq '[.owner.login,.name] | join("/")')/pulls/$PR_NUMBER" \
  -X PATCH -F body=@/tmp/pr_body.md

# Update title (works with both methods)
gh api "repos/$(gh repo view --json owner,name --jq '[.owner.login,.name] | join("/")')/pulls/$PR_NUMBER" \
  -X PATCH -f title="New title here"
```

The ``-F body=@file`` form reliably sends the file contents as a string field. The ``-f`` flag is for short string fields. Use ``-F`` (capital) for file references with ``@`` and ``-f`` for inline values.