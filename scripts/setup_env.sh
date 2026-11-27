#!/usr/bin/env bash
# =============================================================================
#  Maskfish — the build and test environment of pipeline.sh / autotune.sh
#
#  Checks what the training scripts need and installs what is missing:
#    system  a C++20 compiler, make, CMake ≥ 3.20, git, curl, Python ≥ 3.11 with
#            venv — with apt-get, dnf, yum, zypper, pacman or apk, as root or
#            with a password-less sudo
#    python  .venv/ with the packages of train/requirements.txt (numpy, PyTorch):
#            PyTorch with CUDA when an NVIDIA GPU is visible, else its CPU build;
#            a CPU build in .venv is replaced once a GPU is visible
#  A CMake older than 3.20 that the package manager cannot update is replaced by
#  the pip one in .venv/bin (pipeline.sh puts .venv/bin first on PATH then).
#  Nothing that is already there is changed.
#
#  Usage (any directory):
#     scripts/setup_env.sh              # check, install what is missing
#     scripts/setup_env.sh --check      # only check: exit 1 if something is missing
#     scripts/setup_env.sh --no-torch   # without PyTorch (build, matches, option tests)
#
#  Environment: PYTHON (this interpreter instead of .venv; packages are installed
#  into it only when it is a virtual environment), CXX (the compiler to check),
#  TORCH_INDEX_URL (pip index of PyTorch; default: PyPI with a GPU, the CPU
#  builds without one), PIP_OPTS (more pip options).
# =============================================================================
set -uo pipefail
if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4) )); then
  echo "bash >= 4.4 required (this is $BASH_VERSION)" >&2; exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 1

CHECK=0; TORCH=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)    CHECK=1; shift ;;
    --no-torch) TORCH=0; shift ;;
    -h|--help)  sed -n '2,/^# ====/p' "${BASH_SOURCE[0]}" | sed '$d;s/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $1 (--help)" >&2; exit 2 ;;
  esac
done

CMAKE_MIN=3.20; PY_MIN=3.11
VENV="$ROOT/.venv"
CXX_BIN="${CXX:-c++}"
CPU_INDEX=https://download.pytorch.org/whl/cpu
read -r -a PIP_EXTRA <<< "${PIP_OPTS:-}"
[[ -x "$VENV/bin/cmake" ]] && PATH="$VENV/bin:$PATH"   # a pip CMake made by an earlier run

say() { echo "setup: $*" >&2; }
MISSING=()   # what is still missing at the end
miss() { MISSING+=("$1"); }
TMP=$(mktemp -d) || exit 1
trap 'rm -rf "$TMP"' EXIT

# ----------------------------------------------------------------------------- package manager
PM=""
for p in apt-get dnf yum zypper pacman apk; do command -v "$p" > /dev/null && { PM=$p; break; }; done
SUDO=()
if (( EUID != 0 )) && command -v sudo > /dev/null && sudo -n true 2> /dev/null; then SUDO=(sudo -n); fi
can_install() { [[ -n "$PM" ]] && (( EUID == 0 || ${#SUDO[@]} )); }
APT_UPDATED=0
pkg_install() {   # pkg_install PACKAGE...
  (( $# )) || return 0
  if (( CHECK )); then say "would install ($PM): $*"; return 1; fi
  if [[ -z "$PM" ]]; then say "no known package manager: install by hand: $*"; return 1; fi
  if ! can_install; then say "not root and no password-less sudo: as root, install: $*"; return 1; fi
  say "installing ($PM): $*"
  case "$PM" in
    apt-get)
      if (( ! APT_UPDATED )); then
        "${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get update -qq >&2 || return 1
        APT_UPDATED=1
      fi
      "${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "$@" >&2 ;;
    dnf|yum) "${SUDO[@]}" "$PM" install -y "$@" >&2 ;;
    zypper)  "${SUDO[@]}" zypper --non-interactive install "$@" >&2 ;;
    pacman)  "${SUDO[@]}" pacman -S --needed --noconfirm "$@" >&2 ;;
    apk)     "${SUDO[@]}" apk add --no-cache "$@" >&2 ;;
  esac
}
pkg() {   # pkg WHAT → the package names of PM
  case "$PM:$1" in
    apt-get:cxx)              echo build-essential ;;
    dnf:cxx|yum:cxx|zypper:cxx) echo gcc-c++ ;;
    pacman:cxx)               echo gcc ;;
    apk:cxx)                  echo build-base ;;
    pacman:python)            echo python python-pip ;;
    apk:python)               echo python3 py3-pip ;;
    apt-get:python)           echo python3 python3-venv python3-pip ;;
    *:python)                 echo python3 python3-pip ;;
    apt-get:python3.*)        echo "$1 $1-venv" ;;
    *:python3.*)              echo "$1" ;;
    *)                        echo "$1" ;;
  esac
}

# ----------------------------------------------------------------------------- checks
ver_ge() { [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1)" == "$2" ]]; }   # ver_ge A B: A ≥ B
cxx_ok() {   # the compiler builds C++20 (the standard headers the engine uses)
  command -v "$CXX_BIN" > /dev/null || return 1
  printf '#include <bit>\n#include <charconv>\n#include <concepts>\nint main() { return std::popcount(1u) - 1; }\n' > "$TMP/t.cpp"
  "$CXX_BIN" -std=c++20 "$TMP/t.cpp" -o "$TMP/t" 2> /dev/null && "$TMP/t"
}
cmake_version() { cmake --version 2> /dev/null | head -n 1 | awk '{print $3}'; }
cmake_ok() { local v; v=$(cmake_version); [[ -n "$v" ]] && ver_ge "$v" "$CMAKE_MIN"; }
py_ok() {    # py_ok PYTHON: version ≥ PY_MIN and the venv module with ensurepip
  "$1" -c "import sys, venv, ensurepip; sys.exit(sys.version_info < tuple(map(int, '$PY_MIN'.split('.'))))" 2> /dev/null
}
base_python() {   # the newest Python ≥ PY_MIN with venv on PATH (empty if none)
  local p
  for p in python3 python3.14 python3.13 python3.12 python3.11; do
    command -v "$p" > /dev/null && py_ok "$p" && { command -v "$p"; return 0; }
  done
  return 0
}
has_gpu() { [[ -e /dev/nvidia0 ]] || nvidia-smi -L > /dev/null 2>&1; }

# ----------------------------------------------------------------------------- system
need=()
cxx_ok || need+=($(pkg cxx))
command -v make > /dev/null || need+=($(pkg make))
cmake_ok || need+=($(pkg cmake))
command -v git > /dev/null || need+=($(pkg git))
command -v curl > /dev/null || need+=($(pkg curl))
[[ -n "${PYTHON:-}" || -x "$VENV/bin/python" || -n "$(base_python)" ]] || need+=($(pkg python))
if (( ${#need[@]} )); then
  # shellcheck disable=SC2046
  pkg_install $(printf '%s\n' "${need[@]}" | awk '!seen[$0]++')
fi
# a distribution whose python3 is older than PY_MIN: a newer one from its packages
if [[ -z "${PYTHON:-}" && ! -x "$VENV/bin/python" && -z "$(base_python)" ]] && (( ! CHECK )) && can_install; then
  say "no Python ≥ $PY_MIN with venv on PATH: trying a newer one from the packages"
  for v in 3.13 3.12 3.11; do
    pkg_install $(pkg "python$v") 2> /dev/null && [[ -n "$(base_python)" ]] && break
  done
fi

cxx_ok || miss "a C++20 compiler ($CXX_BIN does not build C++20: g++ ≥ 10 or clang++ ≥ 12, set CXX)"
command -v make > /dev/null || miss "make"
command -v git > /dev/null || miss "git"
command -v curl > /dev/null || miss "curl"

# ----------------------------------------------------------------------------- python
PY=""; IN_VENV=0
pip() { "$PY" -m pip install -q --disable-pip-version-check "${PIP_EXTRA[@]}" "$@" >&2; }
if [[ -n "${PYTHON:-}" ]]; then
  PY=$(command -v "$PYTHON") || { miss "python: PYTHON=$PYTHON not found"; PY=""; }
  [[ -n "$PY" ]] && "$PY" -c 'import sys; sys.exit(sys.prefix == sys.base_prefix)' 2> /dev/null && IN_VENV=1
elif [[ -x "$VENV/bin/python" ]]; then
  PY="$VENV/bin/python"; IN_VENV=1
else
  base=$(base_python)
  if [[ -z "$base" ]]; then miss "Python ≥ $PY_MIN with venv"
  elif (( CHECK )); then miss "$VENV (would be made with $base)"
  else
    say "creating $VENV ($("$base" --version 2>&1))"
    if "$base" -m venv "$VENV" >&2 && [[ -x "$VENV/bin/python" ]]; then
      PY="$VENV/bin/python"; IN_VENV=1
      "$PY" -m pip install -q --disable-pip-version-check --upgrade pip >&2 || say "(pip could not be upgraded)"
    else
      miss "$VENV: python -m venv failed"
    fi
  fi
fi
if [[ -n "$PY" ]] && ! "$PY" -c "import sys; sys.exit(sys.version_info < tuple(map(int, '$PY_MIN'.split('.'))))"; then
  miss "Python ≥ $PY_MIN: $PY is $("$PY" --version 2>&1) (remove $VENV to make it again, or set PYTHON)"
fi

py_has() { [[ -n "$PY" ]] && "$PY" -c "import $1" 2> /dev/null; }
py_install() {   # py_install WHAT ARGS...: pip into PY when it is a virtual environment
  local what=$1; shift
  if [[ -z "$PY" ]]; then return 1; fi
  if (( CHECK )); then say "would install into $PY: $what"; return 1; fi
  if (( ! IN_VENV )); then say "$PY is not a virtual environment: install $what into it by hand"; return 1; fi
  say "installing into $PY: $what"
  pip "$@"
}
TORCH_INFO=""   # "VERSION BUILT-WITH-CUDA CUDA-USABLE" (importing torch takes seconds: once, and after a change)
torch_probe() {
  TORCH_INFO=$("$PY" -c 'import torch; print(torch.__version__, int(bool(torch.version.cuda)), int(torch.cuda.is_available()))' 2> /dev/null)
}
if [[ -n "$PY" ]]; then
  if (( TORCH )); then
    torch_probe
    if [[ -z "$TORCH_INFO" ]]; then
      idx=(); [[ -n "${TORCH_INDEX_URL:-}" ]] && idx=(--index-url "$TORCH_INDEX_URL")
      [[ -z "${TORCH_INDEX_URL:-}" ]] && ! has_gpu && idx=(--index-url "$CPU_INDEX")
      py_install "PyTorch${idx[1]:+ from ${idx[1]}}" torch "${idx[@]}" && torch_probe
    elif [[ -z "${TORCH_INDEX_URL:-}" ]] && has_gpu && (( IN_VENV )) && [[ "$(cut -d' ' -f2 <<< "$TORCH_INFO")" == 0 ]]; then
      # made on a machine without a GPU (or a container started without one)
      if (( CHECK )); then say "would replace the CPU build of PyTorch in $PY by the CUDA one"
      else
        say "a GPU is visible: replacing the CPU build of PyTorch by the CUDA one"
        "$PY" -m pip uninstall -y -q torch >&2 && pip torch
        torch_probe
      fi
    fi
    [[ -n "$TORCH_INFO" ]] || miss "PyTorch in $PY (pip install -r train/requirements.txt)"
    py_has numpy || py_install "train/requirements.txt" -r train/requirements.txt
  else
    py_has numpy || py_install numpy numpy
  fi
  py_has numpy || miss "numpy in $PY"
fi

# ----------------------------------------------------------------------------- cmake
if ! cmake_ok; then
  if [[ "$PY" == "$VENV/bin/python" ]] && (( ! CHECK )); then
    say "CMake ${CMAKE_MIN}+ from pip into $VENV/bin (system: $(cmake_version || true))"
    pip "cmake>=$CMAKE_MIN" && PATH="$VENV/bin:$PATH"
  elif (( CHECK )); then say "would install CMake ${CMAKE_MIN}+ (pip, into $VENV/bin)"
  fi
  cmake_ok || miss "CMake ≥ $CMAKE_MIN (found: $(cmake_version || true))"
fi

# ----------------------------------------------------------------------------- summary
if (( ${#MISSING[@]} )); then
  for m in "${MISSING[@]}"; do say "MISSING: $m"; done
  exit 1
fi
info="$("$CXX_BIN" --version 2> /dev/null | head -n 1), CMake $(cmake_version), $("$PY" --version 2>&1) ($PY)"
info+=", numpy $("$PY" -c 'import numpy; print(numpy.__version__)' 2> /dev/null)"
if (( TORCH )); then
  info+=", torch ${TORCH_INFO%% *}"
  if [[ "${TORCH_INFO##* }" == 1 ]]; then info+=" (CUDA)"
  elif has_gpu; then
    info+=" (CPU)"
    if [[ "$(cut -d' ' -f2 <<< "$TORCH_INFO")" == 0 ]]; then say "warning: a GPU is visible but PyTorch is a CPU build (TORCH_INDEX_URL?); REQUIRE_GPU=1 will refuse to train"
    else say "warning: a GPU is visible but PyTorch cannot use it (driver?); REQUIRE_GPU=1 will refuse to train"; fi
  else info+=" (CPU)"; fi
fi
say "ok — $info"
exit 0
