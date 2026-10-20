#!/bin/sh

# Makes sure the default network is present in the current directory.
#
# Maskfish needs a Jieqi network (feature set HalfKAv2_hm with dark-piece and
# pool features). If src/maskfish.nnue does not exist, it is fetched from the
# hidden git ref refs/assets/net (a commit that holds only maskfish.nnue,
# published by scripts/push_net.sh) of the remote origin, then of the
# repository of Maskfish (when origin is a fork without the ref). This needs a
# git clone of the source: `git clone` itself does not fetch this ref.
#
# An existing file is kept as it is: delete it and run `make net` again to get
# the latest network.
#
# A failed fetch does not stop the build: the engine builds without a
# network, but then needs one at run time (EvalFile option). A network can also
# be trained with the trainer in the trainer/ directory (see trainer/README.md).
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

# Fetches the ref (into the local ref of the same name) from $1 and writes its
# maskfish.nnue to $2
fetch_ref() {
  GIT_TERMINAL_PROMPT=0 git fetch --quiet --no-tags "$1" "+$_ref:$_ref" 2> /dev/null &&
    git cat-file blob "$_ref:$_filename" > "$2" 2> /dev/null
}

note_missing() {
  echo "Note: $_filename not found. The engine builds without it, but needs a"
  echo "Jieqi network at run time (EvalFile option). Put a Jieqi network at"
  echo "src/$_filename (e.g. the one of a release archive), or train one with the"
  echo "trainer (see trainer/README.md)."
}

fetch_network() {
  if [ -f "$_filename" ]; then
    if is_valid "$_filename"; then
      echo "Exists $_filename"
    else
      echo "Warning: $_filename exists but is not a zstd-compressed network."
    fi
    return
  fi

  if ! command -v git > /dev/null 2>&1 || ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Note: the source is not a git clone, cannot fetch $_filename from $_ref."
    note_missing
    return
  fi

  _tmp="$_filename.part"
  for _remote in origin "$_repo"; do
    echo "Fetching $_filename from $_ref of $_remote ..."
    rm -f "$_tmp"
    if fetch_ref "$_remote" "$_tmp" && is_valid "$_tmp"; then
      mv "$_tmp" "$_filename"
      echo "Fetched $_filename ($_ref of $_remote)"
      return
    fi
  done
  rm -f "$_tmp"

  echo "Note: could not fetch $_filename from $_ref."
  note_missing
}

fetch_network

if [ "$1" = "--require" ] && ! is_valid "$_filename"; then
  echo "Error: profile-guided optimization runs a benchmark, which needs a valid"
  echo "Jieqi network at src/$_filename (see above). Put one there, or use"
  echo "\`make build\` (no PGO)."
  exit 1
fi
