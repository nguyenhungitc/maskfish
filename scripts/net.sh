#!/bin/sh

# Makes sure the default network is present in the current directory.
#
# Maskfish needs a Jieqi network (feature set HalfKAv2_hm with dark-piece and
# pool features). If src/maskfish.nnue does not exist, the network of the
# latest GitHub release of this repository is downloaded. An existing file is
# kept as it is: delete it and run `make net` again to get the latest network.
#
# A failed download does not stop the build: the engine builds without a
# network, but then needs one at run time (EvalFile option). A network can also
# be trained with the trainer in the trainer/ directory (see trainer/README.md).
#
# The download URL can be overridden with the MASKFISH_NET_URL environment
# variable (e.g. to pin a given release).

_filename="maskfish.nnue"
_url="${MASKFISH_NET_URL:-https://github.com/nguyenhungitc/maskfish/releases/latest/download/$_filename}"

# The engine reads zstd-compressed networks only: a valid file starts with the
# zstd magic number 28 b5 2f fd. This also rejects an HTML error page.
is_valid() {
  [ -s "$1" ] && [ "$(od -An -tx1 -N4 "$1" | tr -d ' \n')" = "28b52ffd" ]
}

download() {
  if command -v curl > /dev/null 2>&1; then
    curl -fsSL --retry 3 -o "$2" "$1"
  elif command -v wget > /dev/null 2>&1; then
    wget -q --tries=3 -O "$2" "$1"
  else
    echo "Note: neither curl nor wget is installed, cannot download $_filename."
    return 1
  fi
}

note_missing() {
  echo "Note: $_filename not found. The engine builds without it, but needs a"
  echo "Jieqi network at run time (EvalFile option). Download $_filename from"
  echo "https://github.com/nguyenhungitc/maskfish/releases/latest and put it in"
  echo "src/, or train one with the trainer (see trainer/README.md)."
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

  echo "Downloading $_filename from $_url ..."
  _tmp="$_filename.part"
  rm -f "$_tmp"

  if download "$_url" "$_tmp" && is_valid "$_tmp"; then
    mv "$_tmp" "$_filename"
    echo "Downloaded $_filename"
  else
    rm -f "$_tmp"
    echo "Note: could not download $_filename."
    note_missing
  fi
}

fetch_network
