#!/bin/sh

# Checks that the default network is present in the current directory.
#
# Maskfish needs a Jieqi network (feature set HalfKAv2_hm with dark-piece and
# pool features). No compatible network is downloaded automatically: train one
# with the trainer in the trainer/ directory of this repository and put it at
# src/maskfish.nnue, or point the EvalFile UCI option to it at run time.

check_network() {
  _filename="maskfish.nnue"

  if [ -f "$_filename" ]; then
    echo "Exists $_filename"
    return
  fi

  echo "Note: $_filename not found. The engine builds without it, but needs a"
  echo "Jieqi network at run time (EvalFile option). Train one with"
  echo "the trainer in the trainer/ directory (see trainer/README.md)."
}

check_network
