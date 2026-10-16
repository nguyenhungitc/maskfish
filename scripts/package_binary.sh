#!/bin/sh

#
# Packages a built binary for publication (used by .github/workflows/binaries.yml):
#
#   scripts/package_binary.sh <binary> <archive>
#
# <archive> ends with .tar or .zip and is named like the file that
# scripts/get_native_properties.sh gives for the system, e.g.
# maskfish-ubuntu-x86-64-universal.tar. It holds the binary (renamed
# maskfish, or maskfish.exe for a Windows binary), src/maskfish.nnue when it
# exists (the network the engine looks for next to the binary), README.md,
# Copying.txt and AUTHORS. Run it from the root of the repository.
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
cp README.md Copying.txt AUTHORS "$dir/"
if [ -f src/maskfish.nnue ]; then
  cp src/maskfish.nnue "$dir/"
else
  printf 'warning: src/maskfish.nnue not found, the archive has no network\n' >&2
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
