#!/usr/bin/env bash
# The check of a clean clone: fetch the network (scripts/net.sh), build the engine and the referee, test the
# engine's rules against the referee, and check perft, the bench signature, invalid input, reproducible search,
# random games, the network evaluator, which network the engine loads (EvalFile, the embedded one), pondering,
# the referee's library, the local board and the trainer.
# Run it from anywhere; it works on the repository it is in.
#
#   bash tools/check_all.sh
#
# It needs CMake 3.20 or later, a C++20 compiler and Python 3.11 or later, and git to fetch the network of
# refs/assets/net (an existing nets/maskfish.nnue is kept when the ref cannot be fetched).
# Settings (environment variables):
#   JOBS=4            parallel build jobs
#   NICE=19           niceness of every build and test; NICE= runs them without nice
#   BUILD=build/check build directory, relative to the repository or absolute
#   CMAKE_OPTIONS=... more options of the build, e.g. CMAKE_OPTIONS=-DMASKFISH_BITBOARD=ON to check the bitboard
#                     build (the same signature: use a BUILD directory of its own)
#   DIFFTEST_GAMES=1000 DIFFTEST_SEED=1   the differential test of the engine against the referee
#   PLAY_PORT=8791    the port of the local board's smoke test (tests/play_smoke.py)
#   UNIVERSAL=1       also build the universal binary (BUILD/universal-bin, -DMASKFISH_UNIVERSAL=ON) and check every
#                     build of it this CPU runs (tests/universal_smoke.py); off by default (it builds the engine
#                     once per instruction set)
#   PYTHON=...        the Python of the trainer's smoke test (default: .venv/bin/python when it exists, as the
#                     training scripts; else python3); the other steps use python3 (standard library only)
# The bench signature is made with the test network of tests/testnet.py (a fixed file made
# from a seeded generator), so it does not depend on the published network; the bench with nets/maskfish.nnue
# is run as well and its node count printed.
# The exit status is 0 only if every step passes.
set -u
cd "$(dirname "$0")/.." || exit 2
ROOT=$(pwd)
JOBS=${JOBS:-4}
NICE=${NICE-19}
BUILD=${BUILD:-build/check}
case "$BUILD" in /*) ;; *) BUILD="$ROOT/$BUILD" ;; esac
DIFFTEST_GAMES=${DIFFTEST_GAMES:-1000}
DIFFTEST_SEED=${DIFFTEST_SEED:-1}
NET="$ROOT/nets/maskfish.nnue"
TESTNET="$BUILD/testnet.hnn"
BENCH_NODES=61977  # bench at its default depth with the test network (tests/testnet.py), on any CPU
LOG="$BUILD/logs"
if [ -z "${PYTHON:-}" ]; then
    if [ -x "$ROOT/.venv/bin/python" ]; then PYTHON="$ROOT/.venv/bin/python"; else PYTHON=python3; fi
fi
mkdir -p "$LOG"

run() { if [ -n "$NICE" ]; then nice -n "$NICE" "$@"; else "$@"; fi; }
names=() results=()
step() {  # step NAME COMMAND...: run, keep the output in $LOG/NAME.log, record pass or fail
    local name=$1; shift
    local t0=$SECONDS
    printf '%-26s ' "$name"
    if "$@" > "$LOG/$name.log" 2>&1; then
        results+=("PASS"); echo "pass ($((SECONDS - t0)) s)"
    else
        results+=("FAIL"); echo "FAIL ($((SECONDS - t0)) s), see $LOG/$name.log"; tail -n 15 "$LOG/$name.log" | sed 's/^/    /'
    fi
    names+=("$name")
}

build() {  # the engine, the differential test and the referee's shared library (BUILD/lib)
    run cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release ${CMAKE_OPTIONS:-} &&
    run cmake --build "$BUILD" -j "$JOBS" --target maskfish maskfish_difftest jieqi_rules jqnz_tool
}
difftest() { run "$BUILD/maskfish_difftest" "$DIFFTEST_GAMES" "$DIFFTEST_SEED"; }
perft() {
    local out
    out=$(printf 'position startpos\nperft 1\nperft 2\nperft 3\nquit\n' | run "$BUILD/maskfish") || return 1
    echo "$out"
    echo "$out" | grep -q '^perft 1 44 ' && echo "$out" | grep -q '^perft 2 1926 ' && echo "$out" | grep -q '^perft 3 83907 '
}
network() { run sh scripts/net.sh --require && echo "net sha256 $(sha256sum "$NET" | cut -d' ' -f1)"; }
bench() {  # the signature with the test network, then the bench with the published network
    local out
    python3 tests/testnet.py "$TESTNET" || return 1
    out=$(printf 'setoption name EvalFile value %s\nbench\nquit\n' "$TESTNET" | run "$BUILD/maskfish") || return 1
    echo "$out"
    echo "$out" | grep -qx "Nodes searched: $BENCH_NODES" || { echo "the signature is not $BENCH_NODES"; return 1; }
    out=$(printf 'setoption name EvalFile value %s\nbench\nquit\n' "$NET" | run "$BUILD/maskfish") || return 1
    echo "with nets/maskfish.nnue: $(echo "$out" | grep -E 'EvalFile|Nodes searched' | tr '\n' ' ')"
    echo "$out" | grep -q "EvalFile loaded" && echo "$out" | grep -q "Nodes searched"
}
hnn_smoke() { run python3 tests/hnn_smoke.py "$BUILD/maskfish"; }  # the network evaluator (src/hnn/)
# refused positions, moves and option values; fuzzing; the engine survives them
invalid_input() { run python3 tests/invalid_input.py "$BUILD/maskfish"; }
# random games with reveals and captures of face-down pieces, searched along the way
random_games() { run python3 tests/random_games.py "$BUILD/maskfish" --games 4; }
# the same searches give the same results after ucinewgame and in a new engine
reprosearch() { run python3 tests/reprosearch.py "$BUILD/maskfish" --evalfile "$NET"; }
# EvalFile first, then the embedded network (nets/maskfish.nnue, embedded by the build), else an error
evalfile_smoke() { run python3 tests/evalfile_smoke.py "$BUILD/maskfish" --net "$NET"; }
universal() {
    run cmake -S . -B "$BUILD/universal-bin" -DCMAKE_BUILD_TYPE=Release -DMASKFISH_UNIVERSAL=ON &&
    run cmake --build "$BUILD/universal-bin" -j "$JOBS" --target maskfish &&
    run python3 tests/universal_smoke.py "$BUILD/universal-bin/maskfish" &&
    run python3 tests/evalfile_smoke.py "$BUILD/universal-bin/maskfish" --net "$NET"
}
ponder_smoke() { run python3 tests/ponder_smoke.py "$BUILD/maskfish" "$NET"; }
referee_smoke() {  # the referee's Python side loads the library just built
    JIEQI_NATIVE_LIBRARY="$BUILD/lib/libjieqi_rules.so" run python3 -c '
import sys; sys.path.insert(0, "tools")
from jieqi.native import NativeRules
from jieqi.setup import Setup
r = NativeRules(); s = Setup.from_seed(1); print("referee library loaded, deal of seed 1 made:", type(s).__name__)'
}
play_smoke() {  # the local board (tools/play.py): two games over HTTP, no hidden identity reaches the browser
    JIEQI_NATIVE_LIBRARY="$BUILD/lib/libjieqi_rules.so" run python3 tests/play_smoke.py 2 "${PLAY_PORT:-8791}" --engine "$BUILD/maskfish"
}
train_smoke() { run "$PYTHON" tests/train_smoke.py "$BUILD/maskfish" --jqnz "$BUILD/jqnz_tool"; }  # needs PyTorch, else skipped

echo "repository $ROOT"
echo "build      $BUILD (jobs $JOBS, nice ${NICE:-off})"
echo "compiler   $(c++ --version 2>/dev/null | head -n 1); $(cmake --version | head -n 1); $(python3 --version)"
echo
step network             network
step build               build
step "difftest-$DIFFTEST_GAMES" difftest
step perft-1-2-3         perft
step bench               bench
step invalid-input       invalid_input
step reprosearch         reprosearch
step random-games        random_games
step hnn-smoke           hnn_smoke
step evalfile-smoke      evalfile_smoke
step ponder-smoke        ponder_smoke
step referee-smoke       referee_smoke
step play-smoke          play_smoke
step train-smoke         train_smoke
if [ "${UNIVERSAL:-0}" = 1 ]; then step universal universal; fi

echo
fail=0
for i in "${!names[@]}"; do
    printf '  %-4s %s\n' "${results[$i]}" "${names[$i]}"
    [ "${results[$i]}" = PASS ] || fail=1
done
grep -h '^games .* failures' "$LOG/difftest-$DIFFTEST_GAMES.log" 2>/dev/null | sed 's/^/  difftest: /'
if [ "$fail" = 0 ]; then echo "all steps passed"; else echo "some steps FAILED"; fi
exit "$fail"
