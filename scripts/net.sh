#!/bin/sh

# Gets the network of the build, src/maskfish.nnue (run from src/), from the
# hidden git ref refs/assets/net: a commit that holds only maskfish.nnue,
# published from net/maskfish.nnue by scripts/push_net.sh. The ref is the only
# source of the network: src/maskfish.nnue is a local copy of it, which the
# build embeds in the binary.
#
# The ref is fetched from the remote origin, then from the repository of
# Maskfish (when origin is a fork without the ref, or cannot be fetched
# without a prompt, e.g. over ssh with a key that is not in ssh-agent), at
# every run: the copy is replaced when the network of the ref changed. This
# needs a git clone of the source (`git clone` itself does not fetch this
# ref). When the ref cannot be fetched (no connection), an existing copy is
# kept.
#
# A failed fetch does not stop the build: the engine builds without a
# network, but then needs one at run time (EvalFile option).
#
# With --require (used by `make profile-build`, whose PGO bench runs the
# engine), a missing or invalid network is an error: the script exits with
# status 1 and says so, before anything is built.
#
# Environment variables:
#   MASKFISH_NET_REF   the ref (default refs/assets/net)
#   MASKFISH_NET_REPO  the repository fetched after origin (default
#                      https://github.com/nguyenhungitc/maskfish)

_filename="maskfish.nnue"
_ref="${MASKFISH_NET_REF:-refs/assets/net}"
_repo="${MASKFISH_NET_REPO:-https://github.com/nguyenhungitc/maskfish}"

# The engine reads zstd-compressed networks only: a valid file starts with the
# zstd magic number 28 b5 2f fd.
is_valid() {
  [ -s "$1" ] && [ "$(od -An -tx1 -N4 "$1" | tr -d ' \n')" = "28b52ffd" ]
}

# git fetch that never prompts: no password (HTTPS), and ssh in batch mode, so
# that a key with a passphrase that is not in ssh-agent makes the fetch fail
# instead of asking for it (a GIT_SSH program, e.g. plink, is left as it is)
fetch_quiet() {
  if [ -z "$GIT_SSH" ]; then
    _ssh="${GIT_SSH_COMMAND:-$(git config core.sshCommand 2> /dev/null || echo ssh)}"
    GIT_SSH_COMMAND="$_ssh -o BatchMode=yes" GIT_TERMINAL_PROMPT=0 \
      git fetch --quiet --no-tags "$@" 2> /dev/null
  else
    GIT_TERMINAL_PROMPT=0 git fetch --quiet --no-tags "$@" 2> /dev/null
  fi
}

# Fetches the ref (into the local ref of the same name) from $1 and writes its
# maskfish.nnue to $2
fetch_ref() {
  fetch_quiet "$1" "+$_ref:$_ref" &&
    git cat-file blob "$_ref:$_filename" > "$2" 2> /dev/null
}

note_missing() {
  echo "Note: no network. The engine builds without it, but then needs a Jieqi"
  echo "network at run time (EvalFile option). Publish one with"
  echo "scripts/push_net.sh (from net/$_filename), or set EvalFile to a file."
}

fetch_network() {
  if ! command -v git > /dev/null 2>&1 || ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Note: the source is not a git clone, cannot fetch $_filename from $_ref."
  else
    _tmp="$_filename.part"
    for _remote in origin "$_repo"; do
      rm -f "$_tmp"
      if fetch_ref "$_remote" "$_tmp" && is_valid "$_tmp"; then
        if cmp -s "$_tmp" "$_filename"; then
          rm -f "$_tmp"
          echo "Up to date $_filename ($_ref of $_remote)"
        else
          mv "$_tmp" "$_filename"
          echo "Fetched $_filename ($_ref of $_remote)"
        fi
        return
      fi
    done
    rm -f "$_tmp"
    echo "Note: could not fetch $_filename from $_ref (origin, $_repo)."
  fi

  if is_valid "$_filename"; then
    echo "Using the existing copy $_filename (it may be older than $_ref)."
  else
    note_missing
  fi
}

fetch_network

if [ "$1" = "--require" ] && ! is_valid "$_filename"; then
  echo "Error: profile-guided optimization runs a benchmark, which needs the"
  echo "network of $_ref (see above), or use \`make build\` (no PGO)."
  exit 1
fi
