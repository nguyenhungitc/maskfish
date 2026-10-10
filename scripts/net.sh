#!/bin/sh

# Downloads the default network into the current directory if it is missing.
# Note: this is Pikafish's xiangqi network (master-net), which this Jieqi
# engine cannot use (different feature set, see README.md). Put a Jieqi
# network at src/pikafish.nnue to skip the download.

wget_or_curl=$( (command -v wget > /dev/null 2>&1 && echo "wget -qO-") || \
                (command -v curl > /dev/null 2>&1 && echo "curl -fsSL"))


fetch_network() {
  _filename="pikafish.nnue"

  if [ -f "$_filename" ]; then
    echo "Exists $_filename, skipping download"
    return
  fi

  if [ -z "$wget_or_curl" ]; then
    >&2 printf "%s\n" "Neither wget or curl is installed." \
          "Install one of these tools to download NNUE files automatically."
    exit 1
  fi

  url="https://github.com/official-pikafish/Networks/releases/download/master-net/$_filename"
  echo "Downloading from $url ..."
  # Download to a temporary file, so that a failed download does not leave an
  # empty or partial network that would be taken for a valid one next time
  if $wget_or_curl "$url" > "$_filename.part" && [ -s "$_filename.part" ]; then
    mv "$_filename.part" "$_filename"
    echo "Successfully downloaded $_filename"
    echo "Warning: this xiangqi network is not compatible with this Jieqi engine (see README.md)"
  else
    rm -f "$_filename.part"
    echo "Failed to download $_filename from $url"
    return 1
  fi
}

fetch_network
