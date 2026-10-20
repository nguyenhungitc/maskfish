# shellcheck shell=bash
# Network architecture of pipeline.sh and autotune.sh, from the configuration
# (scripts/pipeline.env):
#   NNUE_ARCH  default | att | threats | att-threats   (feature set)
#   META       0 | 1    meta features            (make meta=yes,    README 7.8)
#   BLEND      0 | 1    layer stack selection    (make lsblend=yes, README 7.7)
#   LAYERS     1 | 2    layer stacks             (make layers=2,    README 7.4)
#   CTX        0 | 1    inventory context        (make ctx=yes,     README 7.6)
#   PSQT       1 | 0    PSQT part of the network (make psqt=no,     README 7.5)
#   QUANT      1 | 2    quantization             (make quant=2,     README 7.12)
# Sets:
#   ARCH_FULL      architecture name of the trainer (jieqi_features.ARCHS)
#   FEATURE_ARCH   architecture of the prepared data (prepare_data.py --arch)
#   BIN_BASE       engine binary name in src/
#   MAKE_FLAGS     make options of the engine (array)
#   TRAIN_ARCH     train.py options of the architecture (array)
#   LP             file name prefix of this line of nets ("" for default)
#   PREP_DIR       directory of the prepared data (per feature set)
# Returns 1 (with a message on stderr) for an invalid configuration.

set_arch_vars() {
  : "${NNUE_ARCH:=default}"; : "${LAYERS:=1}"
  : "${META:=0}"; : "${BLEND:=0}"; : "${CTX:=0}"; : "${PSQT:=1}"; : "${QUANT:=1}"
  case "$NNUE_ARCH" in
    default)     BIN_BASE="maskfish";         MAKE_FLAGS=() ;;
    att)         BIN_BASE="maskfish-att";     MAKE_FLAGS=(nnue=att) ;;
    threats)     BIN_BASE="maskfish-thr";     MAKE_FLAGS=(threats=yes) ;;
    att-threats) BIN_BASE="maskfish-att-thr"; MAKE_FLAGS=(threats=yes nnue=att) ;;
    *) echo "Invalid NNUE_ARCH: $NNUE_ARCH (default|att|threats|att-threats)" >&2; return 1 ;;
  esac
  local v
  for v in META BLEND CTX PSQT; do
    [[ "${!v}" == 0 || "${!v}" == 1 ]] || { echo "$v must be 0 or 1" >&2; return 1; }
  done
  [[ "$LAYERS" == 1 || "$LAYERS" == 2 ]] || { echo "LAYERS must be 1 or 2" >&2; return 1; }
  [[ "$QUANT" == 1 || "$QUANT" == 2 ]] || { echo "QUANT must be 1 or 2" >&2; return 1; }

  # suffixes in the order of jieqi_features.SUFFIXES
  ARCH_FULL="$NNUE_ARCH"; TRAIN_ARCH=(--layers "$LAYERS")
  if (( META ));      then ARCH_FULL+="-meta";   BIN_BASE+="-meta";   MAKE_FLAGS+=(meta=yes); fi
  if (( BLEND ));     then ARCH_FULL+="-blend";  BIN_BASE+="-blend";  MAKE_FLAGS+=(lsblend=yes); fi
  FEATURE_ARCH="$ARCH_FULL"
  if (( LAYERS == 2 )); then ARCH_FULL+="-l2";   BIN_BASE+="-l2";     MAKE_FLAGS+=(layers=2); fi
  if (( CTX ));       then ARCH_FULL+="-ctx";    BIN_BASE+="-ctx";    MAKE_FLAGS+=(ctx=yes); TRAIN_ARCH+=(--ctx); fi
  if (( ! PSQT ));    then ARCH_FULL+="-nopsqt"; BIN_BASE+="-nopsqt"; MAKE_FLAGS+=(psqt=no); TRAIN_ARCH+=(--no-psqt); fi
  if (( QUANT == 2 )); then ARCH_FULL+="-q2";    BIN_BASE+="-q2";     MAKE_FLAGS+=(quant=2); TRAIN_ARCH+=(--quant2); fi

  # every architecture is its own line of nets; default keeps the README layout
  if [[ "$ARCH_FULL" == "default" ]]; then LP=""; else LP="${ARCH_FULL}-"; fi
  PREP_DIR="data/prep-${FEATURE_ARCH}"
  return 0
}
