#!/bin/sh

# Publishes a network: points the hidden ref refs/assets/net of a remote to a
# commit that holds only this file (named maskfish.nnue), apart from the
# history of the code. The ref is the only place of the network on the remote:
# scripts/net.sh fetches it (CI, tools/check_all.sh, the release), fishtest
# takes it as its default network, and a release (Actions > Release > Run
# workflow) ships it in the archive.
#
#   scripts/push_net.sh [file] [remote]
#
# file defaults to nets/maskfish.nnue, remote to origin. The file must be a
# network the engine reads: an HNN net, raw (.hnn from train/train.py) or
# compressed (tools/hnn/jqnz_tool z NET.hnn NET.nnue: a smaller file). Over ssh
# with a key that has a passphrase, only the push asks for it (once).
#
# The ref is not shown on GitHub (files, branches, tags) nor by
# `git branch -a`, and `git clone` does not fetch it; `git ls-remote` lists it.

set -e

root=$(cd "$(dirname "$0")/.." && pwd)
file="${1:-$root/nets/maskfish.nnue}"
remote="${2:-origin}"
ref="${MASKFISH_NET_REF:-refs/assets/net}"

m=$(head -c 8 "$file" 2> /dev/null | od -An -c | tr -d ' \n')
case $m in
  JQHNNUE1|JQHNNL01|JQHNNZ01) ;;
  *) echo "Error: $file is not a network (HNN: JQHNNUE1, JQHNNL01, JQHNNZ01)." >&2; exit 1 ;;
esac

# git runs in the repository, whatever the current directory
file="$(cd "$(dirname "$file")" && pwd)/$(basename "$file")"
cd "$root"

sha256() {
  if command -v sha256sum > /dev/null 2>&1; then sha256sum "$1"; else shasum -a 256 "$1"; fi |
    cut -d ' ' -f 1
}

blob=$(git hash-object -w "$file")

# git fetch that never prompts: no password (HTTPS), and ssh in batch mode, so
# that a key with a passphrase makes the fetch fail instead of asking for it
fetch_quiet() {
  if [ -z "$GIT_SSH" ]; then
    _ssh="${GIT_SSH_COMMAND:-$(git config core.sshCommand 2> /dev/null || echo ssh)}"
    GIT_SSH_COMMAND="$_ssh -o BatchMode=yes" GIT_TERMINAL_PROMPT=0 \
      git fetch --quiet --no-tags "$@" 2> /dev/null
  else
    GIT_TERMINAL_PROMPT=0 git fetch --quiet --no-tags "$@" 2> /dev/null
  fi
}

# Nothing to do when the remote already has this network
if fetch_quiet "$remote" "+$ref:$ref" &&
  [ "$(git rev-parse "$ref:maskfish.nnue" 2> /dev/null)" = "$blob" ]; then
  echo "$ref of $remote already holds this network ($blob)."
  exit 0
fi

tree=$(printf '100644 blob %s\tmaskfish.nnue\n' "$blob" | git mktree)
commit=$(git commit-tree "$tree" -m "maskfish.nnue" -m "sha256: $(sha256 "$file")")

git push --force "$remote" "$commit:$ref"
git update-ref "$ref" "$commit"
echo "Published $file as $ref of $remote ($commit)."
echo "To release it: Actions > Release > Run workflow."
