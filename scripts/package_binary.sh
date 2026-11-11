#!/bin/sh

#
# Packages a built binary for publication (used by .github/workflows/binaries.yml):
#
#   scripts/package_binary.sh <binary> <archive.tar|archive.zip>
#
# <archive> is a release archive, maskfish-ubuntu-x86-64-universal.tar or
# maskfish-ubuntu-arm64-universal.tar. It holds the binary (renamed
# maskfish, or maskfish.exe for a Windows binary), nets/maskfish.nnue when it
# exists (the network of refs/assets/net, fetched by scripts/net.sh; the engine
# loads it from its own directory), README.md, AUTHORS and
# Copying.txt. Run it from the root of the repository.
#

set -e

binary=$1
archive=$2

if [ -z "$binary" ] || [ -z "$archive" ]; then
  printf 'usage: %s <binary> <archive.tar|archive.zip>\n' "$0" >&2
  exit 1
fi

case $binary in
  *.exe) name='maskfish.exe' ;;
  *)     name='maskfish' ;;
esac

dir=$(mktemp -d)
trap 'rm -rf "$dir"' EXIT

cp "$binary" "$dir/$name"
chmod +x "$dir/$name"
cp README.md AUTHORS Copying.txt "$dir/"
if [ -f nets/maskfish.nnue ]; then
  cp nets/maskfish.nnue "$dir/"
else
  printf 'warning: nets/maskfish.nnue not found, the archive has no network\n' >&2
fi

case $archive in
  /*) out=$archive ;;
  *)  out=$(pwd)/$archive ;;
esac
rm -f "$out"
case $archive in
  *.tar) (cd "$dir" && tar -cf "$out" ./*) ;;
  *.zip) (cd "$dir" && zip -q -9 "$out" ./*) ;;
  *)
    printf 'unknown archive type: %s\n' "$archive" >&2
    exit 1
    ;;
esac

printf '%s:\n' "$archive"
case $archive in
  *.tar) tar -tvf "$out" ;;
  *.zip) unzip -l "$out" ;;
esac
