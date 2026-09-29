#!/usr/bin/env bash
# Deploy the checked-out commit to the Hugging Face Space as a snapshot commit
# on the local `space-deploy` branch, then push that branch to the Space.
#
# The Hub refuses any pushed history that ever carried a binary outside
# LFS/Xet, and this repository's history predates the wheel's LFS tracking,
# so the Space gets a chain of tree snapshots rather than the branch itself.
set -euo pipefail
cd "$(dirname "$0")/.."

SPACE_URL=${SPACE_URL:-https://huggingface.co/spaces/whitphx/artalk-realtime}
BRANCH=space-deploy

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "commit or stash your changes first; only the committed tree is deployed" >&2
  exit 1
fi
git remote get-url space >/dev/null 2>&1 || git remote add space "$SPACE_URL"

head=$(git rev-parse HEAD)
parent_args=()
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  parent_args=(-p "refs/heads/$BRANCH")
fi
commit=$(git commit-tree "$head^{tree}" "${parent_args[@]}" \
  -m "Deploy $(git log -1 --format='%h %s' "$head")")
git update-ref "refs/heads/$BRANCH" "$commit"
git push --force space "$BRANCH:main"
