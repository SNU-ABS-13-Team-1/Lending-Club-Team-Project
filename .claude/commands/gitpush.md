---
allowed-tools: Bash(git add:*), Bash(git status:*), Bash(git commit:*), Bash(git push:*), Bash(git diff:*), Bash(git log:*), Bash(git branch:*)
description: Stage all changes, commit with an appropriate message, and push to origin in one go
---

## Context

- Current git status: !`git status`
- Current git diff (staged and unstaged changes): !`git diff HEAD`
- Current branch: !`git branch --show-current`
- Recent commits: !`git log --oneline -5`

## Your task

Based on the changes above:

1. Stage all relevant changes (respect `.gitignore`; do not force-add ignored files).
2. Write a commit message that summarizes the "why" of the changes, following this repo's existing commit message style.
3. Create the commit.
4. Push to `origin` on the current branch.
5. If there are no changes to commit, say so and stop — do not create an empty commit.

Do all of this in as few tool calls as possible. Do not use any other tools or take any other action.
