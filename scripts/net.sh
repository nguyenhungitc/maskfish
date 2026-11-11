#!/bin/sh

# Gets the network of the engine, nets/maskfish.nnue, from the hidden git ref
# refs/assets/net: a commit that holds only maskfish.nnue, published by
# scripts/push_net.sh. The ref is the only source of the network: nets/ is a
# local copy of it (ignored by git), which the tests, the tools, fishtest and
# the release archive use. Run it from anywhere in a git clone of the
# repository.
#
# The network is an HNN net, raw (JQHNNUE1) or compressed (JQHNNL01 /
# JQHNNZ01): the engine recognises the format by its first bytes, whatever the
# file name.
#
# The ref is fetched from the remote origin, then from the repository of
# Maskfish (when origin is a fork without the ref, or cannot be fetched without
# a prompt, e.g. over ssh with a key that has a passphrase), at every run: the
# copy is replaced when the network of the ref changed. `git clone` itself does
# not fetch this ref. When the ref cannot be fetched (no connection), an
# existing copy is kept.
#
# With --require, a missing or invalid network is an error (exit status 1).
#
# Environment variables:
#   MASKFISH_NET_REF   the ref (default refs/assets/net)
#   MASKFISH_NET_REPO  the repository fetched after origin (default
#                      https://github.com/nguyenhungitc/maskfish)

_filename="maskfish.nnue"
_ref="${MASKFISH_NET_REF:-refs/assets/net}"
_repo="${MASKFISH_NET_REPO:-https://github.com/nguyenhungitc/maskfish}"

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root" || exit 1
mkdir -p nets
_dest="nets/$_filename"

# A network the engine reads: an HNN net (JQHNNUE1, JQHNNL01, JQHNNZ01)
is_valid() {
  [ -s "$1" ] || return 1
  _m=$(head -c 8 "$1" | od -An -c | tr -d ' \n')
  case $_m in
    JQHNNUE1|JQHNNL01|JQHNNZ01) return 0 ;;
  esac
  return 1
}

# git fetch that never prompts: no password (HTTPS), and ssh in batch mode
fetch_quiet() {
  if [ -z "$GIT_SSH" ]; then
    _ssh="${GIT_SSH_COMMAND:-$(git config core.sshCommand 2> /dev/null || echo ssh)}"
    GIT_SSH_COMMAND="$_ssh -o BatchMode=yes" GIT_TERMINAL_PROMPT=0 \
      git fetch --quiet --no-tags "$@" 2> /dev/null
  else
    GIT_TERMINAL_PROMPT=0 git fetch --quiet --no-tags "$@" 2> /dev/null
  fi
}

fetch_network() {
  if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Note: not a git clone, cannot fetch $_filename from $_ref."
    return
  fi
  _tmp="$_dest.part"
  for _remote in origin "$_repo"; do
    rm -f "$_tmp"
    if fetch_quiet "$_remote" "+$_ref:$_ref" && git cat-file blob "$_ref:$_filename" > "$_tmp" 2> /dev/null; then
      if ! is_valid "$_tmp"; then
        echo "Note: $_filename of $_ref of $_remote is not a network this engine reads (HNN: JQHNNUE1, JQHNNL01, JQHNNZ01)."
        continue
      fi
      if cmp -s "$_tmp" "$_dest"; then
        rm -f "$_tmp"
        echo "Up to date $_dest ($_ref of $_remote)"
      else
        mv "$_tmp" "$_dest"
        echo "Fetched $_dest ($_ref of $_remote)"
      fi
      return
    fi
  done
  rm -f "$_tmp"
  echo "Note: no network this engine reads on $_ref (origin, $_repo)."
}

fetch_network
if is_valid "$_dest"; then
  exit 0
fi
echo "No network in $_dest: publish one with scripts/push_net.sh, or set EvalFile to a file."
[ "$1" = "--require" ] && exit 1
exit 0
