#!/usr/bin/env bash
# =============================================================================
#  Maskfish — automatic loop: generate data → train → test → try every option
#
#  Wraps scripts/pipeline.sh (one generation: gen → prepare → train → verify →
#  SPRT → promote) and repeats it until:
#     (1) Elo has reached its maximum: PATIENCE rounds in a row without an Elo
#         gain measured against the 2 outside engines (AB-JChess and jieqi), AND
#     (2) BOTH outside engines are beaten convincingly: lower 95% bound of the Elo ≥ WIN_MIN_ELO.
#
#  Option testing:
#   • Engine UCI options (read from src/engine.cpp): changed from the default →
#     SPRT with the same net. Win (H1, then confirmed against the 2 outside
#     engines) → kept and made the default. No win → the option is removed from
#     the code (default value hard-coded, dead branches removed). Every decision
#     = 1 commit on the branch autotune/options in trainer/autotune/work → 1 git
#     patch file in trainer/autotune/patches/. Every patch is built and `bench`
#     must give exactly the expected node count.
#   • Trainer flags (train.py): every variant is trained on the same data / for
#     the same number of epochs as the baseline, then SPRT variant net vs
#     baseline net. Win → made the default; loss → removed.
#
#  Recovery: every step is recorded in trainer/autotune/state.sh; running the
#  same command again resumes where it stopped (data .part files, train --resume,
#  finished matches are reused). --supervise restarts after errors;
#  --install-cron restarts after a reboot.
#
#  Telegram report after every step (TG_BOT_TOKEN / TG_CHAT_ID from pipeline.env).
#
#  Usage (from the trainer/ directory):
#     scripts/autotune.sh                 # run / resume
#     scripts/autotune.sh --supervise     # run, restart automatically after errors
#     nohup scripts/autotune.sh --supervise >> autotune/supervise.log 2>&1 &
#     scripts/autotune.sh --status        # current state
#     scripts/autotune.sh --install-cron  # restart after a reboot
#     scripts/autotune.sh --stop          # stop safely (can be resumed)
#     scripts/autotune.sh --report        # rewrite autotune/REPORT.md + send it to Telegram
#
#  Configuration: scripts/pipeline.env (Telegram, hardware, data, training) and
#  scripts/autotune.env (create a template with: scripts/autotune.sh --init-config).
#  This file is self-contained: the source rewriting tool (Python) is embedded
#  below and extracted to trainer/autotune/optpatch.py when it runs.
# =============================================================================
set -uo pipefail
if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4) )); then
  echo "bash >= 4.4 required (this is $BASH_VERSION)" >&2; exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$TRAINER_DIR/.." && pwd)"
SRC_DIR="$ROOT_DIR/src"
cd "$TRAINER_DIR" || exit 1
SELF="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

# ----------------------------------------------------------------------------- CLI
ACTION="run"; DRY_RUN=0; CONFIG_AT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --supervise)    ACTION="supervise"; shift ;;
    --status)       ACTION="status"; shift ;;
    --stop)         ACTION="stop"; shift ;;
    --report)       ACTION="report"; shift ;;
    --install-cron) ACTION="cron"; shift ;;
    --test-telegram) ACTION="tgtest"; shift ;;
    --init-config)  ACTION="initcfg"; shift ;;
    --config)       CONFIG_AT="${2:-}"; shift 2 ;;
    --dry-run)      DRY_RUN=1; shift ;;
    -h|--help)      sed -n '2,40p' "$SELF" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

# ----------------------------------------------------------------------------- config
PIPE_ENV="$SCRIPT_DIR/pipeline.env"
[[ -f "$PIPE_ENV" ]] && source "$PIPE_ENV"
# what the user configured for pipeline.sh, kept to be merged with the tuned values
USER_GEN_OPTS="${GEN_OPTS:-}"; USER_MATCH_OPTS="${MATCH_OPTS:-}"; USER_TRAIN_ARGS="${TRAIN_ARGS:-}"
[[ -z "$CONFIG_AT" && -f "$SCRIPT_DIR/autotune.env" ]] && CONFIG_AT="$SCRIPT_DIR/autotune.env"
if [[ -n "$CONFIG_AT" ]]; then
  [[ -f "$CONFIG_AT" ]] || { echo "Config file not found: $CONFIG_AT" >&2; exit 2; }
  source "$CONFIG_AT"
fi

NCPU=$(nproc 2>/dev/null || echo 2)
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$TRAINER_DIR/.venv/bin/python" ]]; then PYTHON="$TRAINER_DIR/.venv/bin/python"; else PYTHON=python3; fi
fi
: "${TG_BOT_TOKEN:=}"; : "${TG_CHAT_ID:=}"; : "${TG_NAME:=$(hostname -s 2>/dev/null || hostname)}"
: "${MAKE_ARCH:=native}"; : "${NNUE_ARCH:=default}"; : "${LAYERS:=1}"
: "${GAMES:=50000}"; : "${VAL_GAMES:=1000}"; : "${NODES:=5000}"
: "${EPOCHS:=40}"; : "${EPOCH_SIZE:=10000000}"; : "${LR:=8.75e-4}"; : "${LAMBDA:=1.0}"
: "${BATCH_SIZE:=}"; : "${CHANCE_SCALING:=}"; : "${MATCH_NODES:=20000}"; : "${TRAIN_FROM:=0}"
: "${MATCH_WORKERS:=$(( NCPU > 1 ? NCPU - 1 : 1 ))}"

# --- autotune.env (default values) ---
JIEQI_DIR="${JIEQI_DIR:-$ROOT_DIR/jieqi}"
AB_ENGINE="${AB_ENGINE:-$JIEQI_DIR/abjchess}"
AB_NET="${AB_NET:-$(ls "$JIEQI_DIR"/abjchess*.nnue 2>/dev/null | sort | tail -1)}"
AB_OPTS="${AB_OPTS:-ChasingRule=github}"
JQ_ENGINE="${JQ_ENGINE:-$JIEQI_DIR/jieqi}"
JQ_OPTS="${JQ_OPTS:-}"
: "${EXT_MOVETIME:=200}"        # ms per move against the outside engines (same limit for both sides)
: "${EXT_PAIRS:=200}"           # game pairs (2 games with colors swapped) per outside engine
: "${EXT_HASH:=64}"
: "${EXT_SEED:=20261015}"       # fixed: every net plays the same openings → fair comparison
: "${WIN_MIN_ELO:=50}"          # "convincing win" = Elo − 95% error ≥ this value, against BOTH engines
: "${EXT_EPS:=5}"               # the average Elo (2 engines) must rise by > EXT_EPS to count as an improvement
: "${PATIENCE:=3}"              # rounds in a row without improvement → Elo is at its maximum
: "${MAX_HOURS:=0}"             # 0 = no time limit

: "${OPT_AT_GEN:=1}"            # test engine options once the current net is ≥ this genN (or on a plateau)
: "${OPT_NODES:=$MATCH_NODES}"
: "${OPT_SPRT:=0 5}"
: "${OPT_MAX_PAIRS:=3000}"      # no decision after this many pairs = no Elo gain → dropped
: "${OPT_EXT_CONFIRM:=1}"       # 1 = an option that wins its SPRT must be confirmed against the 2 outside engines
: "${OPT_EXT_PAIRS:=100}"
: "${OPT_EXT_TOL:=0}"           # outside Elo (average) with the option ≥ without it − TOL
: "${OPT_EXCLUDE:=Ponder UCI_ShowWDL RawScore SharedHistory StopEarly JieqiMaterial}"
: "${OPT_INT_TESTS:=ChanceRiskAversion:10,20,-10}"     # numeric options: Name:value,value,...
: "${OPT_INT_REMOVABLE:=ChanceRiskAversion}"            # numeric options that may be removed if every value loses
: "${OPT_GROUP_TESTS:=RevealQuietOff:RevealQuietBase=0+RevealQuietSafety=0+RevealQuietHighValue=0}"
: "${OPT_ONLY:=}"               # if set: only test these options (space separated)

: "${TRAIN_ABL:=1}"             # 1 = test the trainer flags
: "${TRAIN_ABL_AT_GEN:=$OPT_AT_GEN}"
: "${ABL_EPOCHS:=$(( EPOCHS / 2 > 5 ? EPOCHS / 2 : 5 ))}"
: "${ABL_EPOCH_SIZE:=$EPOCH_SIZE}"
: "${ABL_SPRT:=0 5}"
: "${ABL_MAX_PAIRS:=3000}"
if ! declare -p TRAIN_VARIANTS >/dev/null 2>&1; then
  # id | train.py flag | values ("flag" = on/off switch) | 1 = remove from the code if it loses
  TRAIN_VARIANTS=(
    "factorize|--no-factorize|flag|1"
    "factorize-kb|--factorize-kb|flag|1"
    "l1-fact|--no-l1-fact|flag|1"
    "stack-balance|--stack-balance|8 16|1"
    "output-lr|--output-lr-scale|0.05 0.2|0"
    "lambda|--lambda|0.9 0.75|0"
  )
fi
: "${BENCH_ARGS:=16 1 13}"      # bench used to verify the patches (hash threads depth)
: "${KEEP_ATTEMPT_PT:=0}"       # 1 = keep the .pt checkpoints of failed training attempts
: "${LEVEL_CAP:=8}"             # maximum scale-up level (data ×(1+level/2), nodes ×(1+level/4))
: "${RETRY_MAX:=5}"             # retries of a failed pipeline step before stopping
: "${PIPELINE:=$SCRIPT_DIR/pipeline.sh}"

case "$NNUE_ARCH" in
  default) BIN_BASE=maskfish ;; att) BIN_BASE=maskfish-att ;;
  threats) BIN_BASE=maskfish-thr ;; att-threats) BIN_BASE=maskfish-att-thr ;;
  *) echo "Invalid NNUE_ARCH: $NNUE_ARCH" >&2; exit 2 ;;
esac
ARCH_FULL="$NNUE_ARCH"; (( LAYERS == 2 )) && { ARCH_FULL="${NNUE_ARCH}-l2"; BIN_BASE+="-l2"; }
ENGINE="${ENGINE:-$SRC_DIR/$BIN_BASE}"
if [[ "$ARCH_FULL" == default ]]; then LP=""; else LP="${ARCH_FULL}-"; fi
PREP_DIR="data/prep-${NNUE_ARCH}"

AT_DIR="$TRAINER_DIR/autotune"
WORK="$AT_DIR/work"
PATCH_DIR="$AT_DIR/patches"
RES_DIR="$AT_DIR/results"
LOGF="$AT_DIR/autotune.log"
STATE_FILE="$AT_DIR/state.sh"
DEC_FILE="$AT_DIR/decisions.tsv"
mkdir -p "$AT_DIR" "$PATCH_DIR" "$RES_DIR" "$AT_DIR/logs" nets data runs


# ----------------------------------------------------------------------------- embedded files
# autotune.env template (scripts/autotune.sh --init-config)
embedded_env() {
cat <<'__AUTOTUNE_ENV__'
# Configuration of autotune.sh (scripts/autotune.env), created by:
#   scripts/autotune.sh --init-config      (from trainer/)
# Telegram, hardware, games/nodes/epochs... come from scripts/pipeline.env (shared).

# ---------------- the 2 reference engines (objective opponents) ----------------
# Defaults: ../jieqi/abjchess, ../jieqi/abjchess*.nnue (newest), ../jieqi/jieqi
#JIEQI_DIR="$HOME/maskfish/jieqi"
#AB_ENGINE="$JIEQI_DIR/abjchess"
#AB_NET="$JIEQI_DIR/abjchess-20261010.nnue"
AB_OPTS="ChasingRule=github"      # chasing rules close to Maskfish's
#JQ_ENGINE="$JIEQI_DIR/jieqi"
JQ_OPTS=""

# ---------------- measuring against the outside engines ----------------
EXT_MOVETIME=200      # ms per move (node counts of different engines are not comparable → use time)
EXT_PAIRS=200         # game pairs / engine / measurement (each pair: same opening + hidden pieces, colors swapped)
EXT_HASH=64
WIN_MIN_ELO=50        # "convincing win": Elo − 95% error ≥ 50 against BOTH AB-JChess AND jieqi
EXT_EPS=5             # the average Elo must rise by more than this to count as an improvement
PATIENCE=3            # rounds in a row without improvement = Elo at its maximum
LEVEL_CAP=8           # on a plateau without a convincing win: scale data/nodes up to this level
MAX_HOURS=0           # 0 = run until the goal is reached

# ---------------- engine option tests (UCI) ----------------
OPT_AT_GEN=1          # start once nets/gen1.nnue exists (a very weak net gives noisy results)
OPT_NODES=20000
OPT_SPRT="0 5"
OPT_MAX_PAIRS=3000    # max pairs reached without H1 → treated as no Elo gain
OPT_EXT_CONFIRM=1     # an option winning its SPRT must not lose Elo against the 2 outside engines
OPT_EXT_PAIRS=100
OPT_EXT_TOL=0
# not tested (only affect output / multi-threading / data, or need their own data):
OPT_EXCLUDE="Ponder UCI_ShowWDL RawScore SharedHistory StopEarly JieqiMaterial"
# numeric options: Name:value,value ; win → made the default ; all lose → removed (if in OPT_INT_REMOVABLE)
OPT_INT_TESTS="ChanceRiskAversion:10,20,-10"
OPT_INT_REMOVABLE="ChanceRiskAversion"
# option groups tested together (Name:Opt=value+Opt=value) ; loss → kept as is, not removed
OPT_GROUP_TESTS="RevealQuietOff:RevealQuietBase=0+RevealQuietSafety=0+RevealQuietHighValue=0"
#OPT_ONLY="TTPenalize PruningTweaks"   # only test these options

# ---------------- trainer flag tests (train.py) ----------------
TRAIN_ABL=1
ABL_EPOCHS=20         # every variant trains this many epochs on the same data, from the same net
#ABL_EPOCH_SIZE=10000000
ABL_SPRT="0 5"
ABL_MAX_PAIRS=3000
# id | flag | values ("flag" = on/off switch) | 1 = remove from train.py if no Elo gain
TRAIN_VARIANTS=(
  "factorize|--no-factorize|flag|1"
  "factorize-kb|--factorize-kb|flag|1"
  "l1-fact|--no-l1-fact|flag|1"
  "stack-balance|--stack-balance|8 16|1"
  "output-lr|--output-lr-scale|0.05 0.2|0"
  "lambda|--lambda|0.9 0.75|0"
)

# ---------------- other ----------------
BENCH_ARGS="16 1 13"  # bench used to verify every patch (must give exactly the same node count)
RETRY_MAX=5
KEEP_ATTEMPT_PT=0
__AUTOTUNE_ENV__
}

# Source rewriting tool: list UCI options, set defaults, remove options (with dead branches), edit train.py
embedded_optpatch() {
cat <<'__OPTPATCH_PY__'
#!/usr/bin/env python3
"""Source transforms for autotune.sh: list the engine's UCI options and rewrite
the source so that an option becomes the default, gets a new default value, or
disappears from the code.

    optpatch.py list   --src SRC_DIR
    optpatch.py engine --src SRC_DIR --option NAME --action flip|set|remove [--value V] [--no-fold]
    optpatch.py trainer --file train.py --flag --no-factorize --action set|remove [--value V]

engine actions
  flip    bool option: default becomes the opposite (the option stays)
  set     default becomes --value (int / bool)
  remove  the option disappears: registration + reading removed, the member
          variable becomes `static constexpr` with the CURRENT default, then
          (unless --no-fold) dead branches are folded where it is provably safe:
          if (X) / if (!X) [+ else], X ? a : b, `X && `, ` || X` ...
          Remaining uses keep compiling against the constexpr (the compiler
          removes the dead code).

The caller (autotune.sh) builds the patched engine and checks that `bench` gives
exactly the node count expected, so a wrong transform is never committed.

trainer actions (train.py argparse)
  set     the flag's default becomes --value (store_true flags: True)
  remove  the add_argument call is removed and every args.<dest> is replaced by
          the literal default
Only the standard library is used.
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys

# ------------------------------------------------------------------ C++ scanning helpers


def skip_ws_comments(s: str, i: int) -> int:
    n = len(s)
    while i < n:
        if s[i].isspace():
            i += 1
        elif s.startswith("//", i):
            j = s.find("\n", i)
            i = n if j < 0 else j + 1
        elif s.startswith("/*", i):
            j = s.find("*/", i + 2)
            i = n if j < 0 else j + 2
        else:
            break
    return i


def match_close(s: str, i: int) -> int:
    """s[i] is an opening bracket; returns the index of the matching closing one."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack = [pairs[s[i]]]
    j = i + 1
    n = len(s)
    while j < n:
        c = s[j]
        if s.startswith("//", j):
            k = s.find("\n", j)
            j = n if k < 0 else k
            continue
        if s.startswith("/*", j):
            k = s.find("*/", j + 2)
            j = n if k < 0 else k + 2
            continue
        if c in "\"'":
            k = j + 1
            while k < n and s[k] != c:
                k += 2 if s[k] == "\\" else 1
            j = k + 1
            continue
        if c in pairs:
            stack.append(pairs[c])
        elif c in ")]}":
            if not stack or stack[-1] != c:
                raise ValueError(f"unbalanced bracket at {j}")
            stack.pop()
            if not stack:
                return j
        j += 1
    raise ValueError("no matching bracket")


def statement_end(s: str, i: int) -> int:
    """Index just past the ';' that ends the statement starting at i (depth 0)."""
    n = len(s)
    j = i
    while j < n:
        c = s[j]
        if c in "([{":
            j = match_close(s, j) + 1
            continue
        if c in "\"'":
            k = j + 1
            while k < n and s[k] != c:
                k += 2 if s[k] == "\\" else 1
            j = k + 1
            continue
        if s.startswith("//", j):
            k = s.find("\n", j)
            j = n if k < 0 else k
            continue
        if c == ";":
            return j + 1
        j += 1
    raise ValueError("no end of statement")


def line_start(s: str, i: int) -> int:
    return s.rfind("\n", 0, i) + 1


def line_end(s: str, i: int) -> int:
    j = s.find("\n", i)
    return len(s) if j < 0 else j + 1


# ------------------------------------------------------------------ options in engine.cpp

ADD_RE = re.compile(r'options\.add\(\s*(?://[^\n]*\n\s*)?"([^"]+)"\s*,\s*Option\(')


def find_option_add(src: str, name: str):
    """(start, end, option_args_start, option_args_end) of the options.add statement."""
    for m in ADD_RE.finditer(src):
        if m.group(1) != name:
            continue
        start = m.start()
        paren = src.index("(", start)
        close = match_close(src, paren)
        end = statement_end(src, close)
        oparen = m.end() - 1
        oclose = match_close(src, oparen)
        return start, end, oparen + 1, oclose
    return None


def split_top_args(s: str):
    """Split a C++ argument list at depth-0 commas."""
    out, depth, cur, i = [], 0, [], 0
    while i < len(s):
        c = s[i]
        if c in "([{":
            j = match_close(s, i)
            cur.append(s[i:j + 1])
            i = j + 1
            continue
        if c == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
        i += 1
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def list_options(src_dir: str):
    src = open(os.path.join(src_dir, "engine.cpp"), encoding="utf-8").read()
    res = []
    for m in ADD_RE.finditer(src):
        name = m.group(1)
        oparen = m.end() - 1
        args = split_top_args(src[oparen + 1:match_close(src, oparen)])
        if not args:
            kind, default, lo, hi = "button", "", "", ""
        elif args[0] in ("true", "false"):
            kind, default, lo, hi = "bool", args[0], "", ""
        elif re.fullmatch(r"-?\d+", args[0]) and len(args) >= 3 and re.fullmatch(r"-?\d+", args[1]) \
                and re.fullmatch(r"-?\d+", args[2]):
            kind, default, lo, hi = "int", args[0], args[1], args[2]
        else:
            kind, default, lo, hi = "other", args[0].replace(" ", ""), "", ""
        res.append((name, kind, default, lo, hi))
    return res


def set_default(src: str, name: str, value: str) -> str:
    f = find_option_add(src, name)
    if not f:
        raise SystemExit(f"option {name} not found in engine.cpp")
    _, _, a0, a1 = f
    args_txt = src[a0:a1]
    first = split_top_args(args_txt)[0]
    k = args_txt.index(first)
    new_args = args_txt[:k] + value + args_txt[k + len(first):]
    return src[:a0] + new_args + src[a1:]


KEEP_COMMENT = re.compile(r"Experimental|off by default|test\s+with SPRT|match\.py|ideas? (from|ported)", re.I)


def remove_option_add(src: str, name: str) -> str:
    f = find_option_add(src, name)
    if not f:
        raise SystemExit(f"option {name} not found in engine.cpp")
    start, end, _, _ = f
    ls = line_start(src, start)
    le = line_end(src, end)
    # the comment lines right above describe this option, unless they introduce a group
    cstart = ls
    while cstart > 0:
        prev = line_start(src, cstart - 1)
        line = src[prev:cstart].strip()
        if line.startswith("//") and not KEEP_COMMENT.search(line):
            cstart = prev
        else:
            break
    return src[:cstart] + src[le:]


# ------------------------------------------------------------------ remove: member -> constexpr, fold

def literal(kind: str, value: str) -> str:
    if kind == "bool":
        return "true" if value in ("true", "1") else "false"
    return value


def remove_option_engine(src_dir: str, name: str, fold: bool):
    files = {}
    for fn in sorted(os.listdir(src_dir)):
        if fn.endswith((".cpp", ".h")):
            p = os.path.join(src_dir, fn)
            files[p] = open(p, encoding="utf-8").read()
    eng = os.path.join(src_dir, "engine.cpp")
    opts = {o[0]: o for o in list_options(src_dir)}
    if name not in opts:
        raise SystemExit(f"option {name} not found")
    _, kind, default, _, _ = opts[name]
    if kind not in ("bool", "int"):
        raise SystemExit(f"option {name} has kind {kind}: cannot remove automatically")
    lit = literal(kind, default)
    files[eng] = remove_option_add(files[eng], name)

    # IDENT = bool(options["NAME"]);  -> member variable
    assign = re.compile(r'^[ \t]*(\w+)\s*=\s*(?:bool|int)\(\s*(?:[\w.]+\.)?options\["%s"\]\s*\)\s*;[ \t]*\n'
                        % re.escape(name), re.M)
    idents = []
    for p, s in list(files.items()):
        for m in list(assign.finditer(s)):
            idents.append(m.group(1))
        files[p] = assign.sub("", s)
    # every other read becomes the literal
    read = re.compile(r'(?:bool|int)\(\s*(?:[\w.]+\.)?options\["%s"\]\s*\)' % re.escape(name))
    for p, s in files.items():
        files[p] = read.sub(lit, s)
    ctype = "bool" if kind == "bool" else "int"
    for ident in idents:
        decl = re.compile(r'^([ \t]*)(?:bool|int|double)(\s+)%s(\s*)(?:=\s*[^;]+)?;' % re.escape(ident), re.M)
        n = 0
        for p, s in files.items():
            s2, k = decl.subn(lambda m: f"{m.group(1)}static constexpr {ctype}{m.group(2)}{ident}{m.group(3)}= {lit};", s)
            files[p] = s2
            n += k
        if n != 1:
            raise SystemExit(f"declaration of member {ident} found {n} times (expected 1)")
    if fold and kind == "bool":
        for ident in idents:
            for p in files:
                files[p] = fold_bool(files[p], ident, lit == "true")
            # no use left: drop the constexpr declaration
            uses = sum(len(re.findall(r"\b%s\b" % re.escape(ident), mask(s)[0])) for s in files.values())
            if uses == 1:
                d = re.compile(r'^[ \t]*static constexpr bool\s+%s\s*= (?:true|false);[ \t]*\n' % re.escape(ident), re.M)
                for p in files:
                    files[p] = d.sub("", files[p])
    changed = []
    for p, s in files.items():
        old = open(p, encoding="utf-8").read()
        if s != old:
            open(p, "w", encoding="utf-8").write(s)
            changed.append(os.path.basename(p))
    print(f"removed {name} ({kind}, hard-coded {lit}); members {idents or '-'}; changed {changed}")


ATOM = r"(?:\b\w+(?:\.|->))*%s\b"
# where a boolean operand may start: after '(' (not a cast/call check needed: the
# operand is the whole parenthesized expression's first term), '&&', '||', ',',
# '?', ':', or an assignment '=' (not ==, !=, <=, >=)
LB_BASE = r"(?<=\()|(?<=\|\|)|(?<=,)|(?<=\?)|(?<=[^:]:)|(?<=[^=!<>]=)|(?<=\breturn)"
LB = r"(?:%s|(?<=&&))" % LB_BASE
LB_NO_AND = r"(?:%s)" % LB_BASE        # 'x && LIT || E': LIT belongs to the && term


def mask(s: str):
    """Replaces comments and string / char literals by placeholders, so that
    the folding only touches code. Returns (masked, restore list)."""
    out, saved, i, n = [], [], 0, len(s)
    while i < n:
        if s.startswith("//", i):
            j = s.find("\n", i)
            j = n if j < 0 else j
        elif s.startswith("/*", i):
            j = s.find("*/", i + 2)
            j = n if j < 0 else j + 2
        elif s[i] in "\"'" and not (s[i] == "'" and i > 0 and s[i - 1].isalnum()):   # 1'000 digit separators
            q = s[i]
            j = i + 1
            while j < n and s[j] != q:
                j += 2 if s[j] == "\\" else 1
            j += 1
        else:
            out.append(s[i])
            i += 1
            continue
        saved.append(s[i:j])
        out.append(f"\x01{len(saved) - 1}\x02")
        i = j
    return "".join(out), saved


def unmask(s: str, saved) -> str:
    return re.sub(r"\x01(\d+)\x02", lambda m: saved[int(m.group(1))], s)


def operand_chain_end(s: str, i: int, stop_ops) -> int:
    """From i, skip operands joined by anything but the operators in stop_ops,
    up to a depth-0 ')', ';', ',', '?', ':' or one of stop_ops."""
    n = len(s)
    j = i
    while j < n:
        c = s[j]
        if c in "([{":
            j = match_close(s, j) + 1
            continue
        if c in ")]};,?" or (c == ":" and s[j + 1:j + 2] != ":" and s[j - 1] != ":"):
            return j
        if any(s.startswith(op, j) for op in stop_ops):
            return j
        j += 1
    return j


def fold_bool(s: str, ident: str, value: bool) -> str:
    """Safe boolean simplifications for a compile-time constant member
    (comments and strings are left untouched)."""
    s, saved = mask(s)
    s = fold_bool_masked(s, ident, value)
    s = fold_local_consts(s)
    s = tidy_arith(s)
    return unmask(s, saved)


def fold_bool_masked(s: str, ident: str, value: bool) -> str:
    atom = ATOM % re.escape(ident)
    tv, fv = ("true", "false") if value else ("false", "true")
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"!\s*(%s)(?!\s*(?:\.|->|\(|\[))" % atom, fv, s)
        s = re.sub(atom + r"(?=\s*(?:[)&|;,?]|:(?!:)))", tv, s)
        # identities, valid whatever the folded value: true && E, E && true,
        # false || E, E || false -> E ; then absorption: false && E, true || E
        s = re.sub(LB + r"(\s*)true\s*&&\s*", r"\1", s)
        s = re.sub(r"\s*&&\s*true(?=\s*(?:[)&|;,?]|:(?!:)))", "", s)
        s = re.sub(LB_NO_AND + r"(\s*)false\s*\|\|\s*", r"\1", s)
        s = re.sub(r"\s*\|\|\s*false(?=\s*(?:[)&|;,?]|:(?!:)))", "", s)
        for absorb, pat, stop in (("false", re.compile(LB + r"(\s*)false\s*&&"), ("||",)),
                                  ("true", re.compile(LB_NO_AND + r"(\s*)true\s*\|\|"), ())):
            pos = 0
            while True:
                m = pat.search(s, pos)
                if not m:
                    break
                e = operand_chain_end(s, m.end(), stop)
                s = s[:m.start()] + m.group(1) + absorb + s[e:]
                pos = m.start() + 1
        s = re.sub(r"(\b\w+\s*)?(?<![\]>)])\(\s*(true|false)\s*\)(?=\s*(?:[)&|;,?]|:(?!:)))",
                   lambda m: m.group(0) if m.group(1) else m.group(2), s)
        s = fold_ternaries(s)
        s = fold_ifs(s)
    return s


def fold_ternaries(s: str) -> str:
    """(true ? A : B) / (false ? A : B) where the literal directly follows '(' or '='."""
    out = s
    for lit in ("true", "false"):
        pos = 0
        while True:
            m = re.compile(r"(?:(?<=\()|(?<=,)|(?<=[^=!<>]=)|(?<=\breturn)|(?<=\?)|(?<=[^:]:))(\s*)%s\s*\?" % lit).search(out, pos)
            if not m:
                break
            a_start = m.end()
            # find ':' at depth 0 (nested ternaries counted), then the end of B
            j, depth_q, n = a_start, 0, len(out)
            colon = None
            while j < n:
                c = out[j]
                if c in "([{":
                    j = match_close(out, j) + 1
                    continue
                if c in ")]};,":
                    break
                if c == "?":
                    depth_q += 1
                elif c == ":" and not out.startswith("::", j) and out[j - 1] != ":":
                    if depth_q == 0:
                        colon = j
                        break
                    depth_q -= 1
                j += 1
            if colon is None:
                pos = m.end()
                continue
            j = colon + 1
            while j < n:
                c = out[j]
                if c in "([{":
                    j = match_close(out, j) + 1
                    continue
                if c in ")]};,":
                    break
                if c == "?":       # a ternary inside B without parentheses: give up
                    j = -1
                    break
                j += 1
            if j < 0:
                pos = m.end()
                continue
            a = out[a_start:colon].strip()
            b = out[colon + 1:j].strip()
            keep = a if lit == "true" else b
            out = out[:m.start()] + m.group(1) + keep + out[j:]
            pos = m.start()
    return out


def stmt_end(s: str, i: int) -> int:
    """End (exclusive) of the statement starting at i: blocks, if/else chains,
    for/while loops and simple statements."""
    i = skip_ws_comments(s, i)
    if s[i] == "{":
        return match_close(s, i) + 1
    m = re.match(r"(if(?:\s+constexpr)?|for|while|switch)\b\s*\(", s[i:])
    if m:
        paren = i + m.end() - 1
        j = stmt_end(s, match_close(s, paren) + 1)
        if m.group(1).startswith("if"):
            k = skip_ws_comments(s, j)
            if re.match(r"else\b", s[k:]):
                j = stmt_end(s, k + 4)
        return j
    return statement_end(s, i)


def fold_ifs(s: str) -> str:
    """if (true) S [else T] -> S ; if (false) S [else T] -> T or nothing.
    T may itself be an if/else chain."""
    pos = 0
    pat = re.compile(r"\bif\s*\(\s*(true|false)\s*\)")
    while True:
        m = pat.search(s, pos)
        if not m:
            return s
        body_start = skip_ws_comments(s, m.end())
        body_end = stmt_end(s, body_start)
        k = skip_ws_comments(s, body_end)
        else_body = None
        end = body_end
        if re.match(r"else\b", s[k:]):
            eb = skip_ws_comments(s, k + 4)
            end = stmt_end(s, eb)
            else_body = (eb, end)
        taken = m.group(1) == "true"
        before = s[:m.start()].rstrip()
        is_else_branch = re.search(r"\belse$", before) is not None
        if taken:
            repl = s[body_start:body_end]
        elif else_body:
            repl = s[else_body[0]:else_body[1]]
        else:
            repl = ""
        if repl == "" and is_else_branch:
            cut = before.rfind("else")
            s = s[:cut].rstrip() + s[end:]
            pos = cut
            continue
        if repl == "":
            ls = line_start(s, m.start())
            if s[ls:m.start()].strip() == "" and s[end:line_end(s, end)].strip() == "":
                s = s[:ls] + s[line_end(s, end):]
                pos = ls
            else:
                s = s[:m.start()] + ";" + s[end:]
                pos = m.start()
            continue
        s = s[:m.start()] + repl + s[end:]
        pos = m.start()


def fold_local_consts(s: str) -> str:
    """const bool X = true|false;  -> fold X in the rest of its block, drop it if unused."""
    pat = re.compile(r"^[ \t]*const\s+bool\s+(\w+)\s*=\s*(true|false)\s*;[ \t]*\n", re.M)
    pos = 0
    while True:
        m = pat.search(s, pos)
        if not m:
            return s
        # end of the enclosing block: first unmatched '}'
        j, n = m.end(), len(s)
        while j < n:
            if s[j] in "([{":
                j = match_close(s, j) + 1
                continue
            if s[j] == "}":
                break
            j += 1
        region = fold_bool_masked(s[m.end():j], m.group(1), m.group(2) == "true")
        if re.search(r"\b%s\b" % re.escape(m.group(1)), region):
            s = s[:m.end()] + region + s[j:]
            pos = m.end()
        else:
            s = s[:m.start()] + region + s[j:]
            pos = m.start()


def tidy_arith(s: str) -> str:
    """Cosmetic: (0) / (literal) parentheses, '+ 0' / '- 50 * false' terms."""
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"(\b\w+\s*)?(?<![\]>)])\(\s*(-?\d+|true|false)\s*\)",
                   lambda m: m.group(0) if m.group(1) else m.group(2), s)
        # additive zero terms: operand end before, no * / % after
        s = re.sub(r"(?<=[\w)\]])(\s*)[+-]\s*(?:\d+\s*\*\s*)?(?:0|false)\b(?!\s*[*/%.(\[])", "", s)
    return s


# ------------------------------------------------------------------ trainer (train.py argparse)

def find_add_argument(src: str, flag: str):
    for m in re.finditer(r"ap\.add_argument\(", src):
        close = match_close(src, m.end() - 1)
        call = src[m.start():close + 1]
        if re.search(r'["\']%s["\']' % re.escape(flag), call):
            return m.start(), close + 1, call
    raise SystemExit(f"flag {flag} not found")


def flag_info(call: str, flag: str):
    tree = ast.parse(call.replace("ap.add_argument", "f", 1))
    c = tree.body[0].value
    kw = {k.arg: k.value for k in c.keywords}
    dest = ast.literal_eval(kw["dest"]) if "dest" in kw else flag.lstrip("-").replace("-", "_")
    action = ast.literal_eval(kw["action"]) if "action" in kw and isinstance(kw["action"], ast.Constant) else None
    if "default" in kw:
        default = ast.literal_eval(kw["default"])
    else:
        default = False if action == "store_true" else None
    return dest, action, default


def _const_expr(node) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp):
        return _const_expr(node.operand)
    if isinstance(node, ast.BoolOp):
        return all(_const_expr(v) for v in node.values)
    if isinstance(node, ast.Compare):
        return _const_expr(node.left) and all(_const_expr(c) for c in node.comparators)
    return False


def fold_python_ifs(src: str) -> str:
    """`if <constant expression>:` blocks without elif/else: drop a false one,
    dedent a true one (text edit, comments are kept)."""
    while True:
        tree = ast.parse(src)
        target = None
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and not node.orelse and _const_expr(node.test):
                target = node
                break
        if target is None:
            return src
        lines = src.splitlines(keepends=True)
        first, last = target.lineno - 1, target.end_lineno      # if-line .. last body line
        if eval(compile(ast.Expression(target.test), "<c>", "eval"), {"__builtins__": {}}):
            ind = len(lines[first]) - len(lines[first].lstrip())
            body_ind = len(lines[first + 1]) - len(lines[first + 1].lstrip())
            shift = body_ind - ind
            body = [l[shift:] if l.strip() else l for l in lines[first + 1:last]]
            lines[first:last] = body
        else:
            del lines[first:last]
        src = "".join(lines)


def trainer_patch(path: str, flag: str, action: str, value: str | None):
    src = open(path, encoding="utf-8").read()
    start, end, call = find_add_argument(src, flag)
    dest, act, default = flag_info(call, flag)
    if action == "set":
        if act == "store_true":
            new = call.replace('action="store_true"', 'action="store_true", default=True', 1)
        else:
            v = ast.literal_eval(value)
            if re.search(r"default\s*=", call):
                new = re.sub(r"default\s*=\s*[^,)]+", f"default={v!r}", call, count=1)
            else:
                new = call[:-1] + f", default={v!r})"
        src = src[:start] + new + src[end:]
        print(f"{flag}: default {default!r} -> {True if act == 'store_true' else value}")
    elif action == "remove":
        ls, le = line_start(src, start), line_end(src, end)
        src = src[:ls] + src[le:]
        n = len(re.findall(r"\bargs\.%s\b" % re.escape(dest), src))
        src = re.sub(r"\bargs\.%s\b" % re.escape(dest), repr(default), src)
        print(f"{flag}: removed, args.{dest} -> {default!r} ({n} uses)")
        src = re.sub(r"\bnot (True|False)\b", lambda m: "False" if m.group(1) == "True" else "True", src)
        src = fold_python_ifs(src)
    ast.parse(src)
    open(path, "w", encoding="utf-8").write(src)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--src", required=True)
    p = sub.add_parser("engine")
    p.add_argument("--src", required=True)
    p.add_argument("--option", required=True)
    p.add_argument("--action", choices=("flip", "set", "remove"), required=True)
    p.add_argument("--value")
    p.add_argument("--no-fold", action="store_true")
    p = sub.add_parser("trainer")
    p.add_argument("--file", required=True)
    p.add_argument("--flag", required=True)
    p.add_argument("--action", choices=("set", "remove"), required=True)
    p.add_argument("--value")
    a = ap.parse_args()

    if a.cmd == "list":
        for o in list_options(a.src):
            print("\t".join(o))
        return
    if a.cmd == "trainer":
        trainer_patch(a.file, a.flag, a.action, a.value)
        return
    eng = os.path.join(a.src, "engine.cpp")
    if a.action == "remove":
        remove_option_engine(a.src, a.option, not a.no_fold)
        return
    opts = {o[0]: o for o in list_options(a.src)}
    if a.option not in opts:
        raise SystemExit(f"option {a.option} not found")
    _, kind, default, lo, hi = opts[a.option]
    if a.action == "flip":
        if kind != "bool":
            raise SystemExit("flip needs a bool option")
        value = "false" if default == "true" else "true"
    else:
        value = literal(kind, a.value)
        if kind == "int" and not (int(lo) <= int(value) <= int(hi)):
            raise SystemExit(f"{value} outside [{lo}, {hi}]")
    s = set_default(open(eng, encoding="utf-8").read(), a.option, value)
    open(eng, "w", encoding="utf-8").write(s)
    print(f"{a.option}: default {default} -> {value}")


if __name__ == "__main__":
    sys.exit(main())
__OPTPATCH_PY__
}

# embedded helper → trainer/autotune/optpatch.py
OPTPATCH="$AT_DIR/optpatch.py"
embedded_optpatch > "$OPTPATCH.tmp" && { cmp -s "$OPTPATCH.tmp" "$OPTPATCH" || mv -f "$OPTPATCH.tmp" "$OPTPATCH"; }
rm -f "$OPTPATCH.tmp"

# ----------------------------------------------------------------------------- utilities
html() { local s="$1"; s="${s//&/&amp;}"; s="${s//</&lt;}"; s="${s//>/&gt;}"; printf '%s' "$s"; }
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGF"; }
fmt_dur() { local s=$1; printf '%dh%02dm' $((s/3600)) $((s%3600/60)); }
TAG="<b>[$(html "$TG_NAME")] autotune</b>"

tg_send() {
  local text="$1"
  (( ${#text} > 4000 )) && text="${text:0:3990}…"
  if (( DRY_RUN )); then echo "[telegram] $text"; return 0; fi
  [[ -z "$TG_BOT_TOKEN" || -z "$TG_CHAT_ID" ]] && return 0
  local resp
  resp=$(curl -sS --max-time 20 --retry 3 --retry-delay 3 --data-urlencode "chat_id=${TG_CHAT_ID}" \
         --data-urlencode "text=${text}" --data-urlencode "parse_mode=HTML" \
         --data-urlencode "disable_web_page_preview=true" \
         "https://api.telegram.org/bot${TG_BOT_TOKEN}/sendMessage" 2>&1)
  if [[ "$resp" == *"can't parse entities"* ]]; then
    text=$(printf '%s' "$text" | sed -e 's/<[^>]*>//g' -e 's/&lt;/</g; s/&gt;/>/g; s/&amp;/\&/g')
    resp=$(curl -sS --max-time 20 --retry 3 --data-urlencode "chat_id=${TG_CHAT_ID}" \
           --data-urlencode "text=${text}" "https://api.telegram.org/bot${TG_BOT_TOKEN}/sendMessage" 2>&1)
  fi
  [[ "$resp" == *'"ok":true'* ]] || { echo "⚠ Telegram error: ${resp:0:200}" >&2; return 1; }
}
tg_doc() {   # $1 = file, $2 = caption
  (( DRY_RUN )) && { echo "[telegram file] $1"; return 0; }
  [[ -z "$TG_BOT_TOKEN" || -z "$TG_CHAT_ID" || ! -s "$1" ]] && return 0
  curl -sS --max-time 60 --retry 2 -F "chat_id=${TG_CHAT_ID}" -F "document=@$1" \
       -F "caption=${2:0:1000}" "https://api.telegram.org/bot${TG_BOT_TOKEN}/sendDocument" >/dev/null 2>&1
}
say() { log "$*"; tg_send "$TAG · $*"; }

kill_tree() {
  local p; for p in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$p"; done
  # commands started by run_logged lead their own process group: this also catches
  # pool workers that multiprocessing respawns while its workers are being killed
  [[ "$(ps -o pgid= -p "$1" 2>/dev/null | tr -d ' ')" == "$1" ]] && kill -TERM -- "-$1" 2>/dev/null
  kill -TERM "$1" 2>/dev/null
}
kill_children() { local p; for p in $(pgrep -P $$ 2>/dev/null); do kill_tree "$p"; done; }
if command -v setsid >/dev/null 2>&1; then GROUP=(setsid -w); else GROUP=(); fi

run_logged() {   # run_logged LOGFILE cmd...  (background + wait: signals are handled at once)
  local lf="$1"; shift
  echo "[$(date '+%F %T')] \$ $*" >> "$lf"
  (( DRY_RUN )) && { echo "\$ $*"; return 0; }
  # 9>&-: the autotune lock is not inherited, so a stray process cannot block a restart
  ( "${GROUP[@]}" "$@" >> "$lf" 2>&1 9>&- ) &
  wait $!
}

# ----------------------------------------------------------------------------- state
declare -A ST=()
state_load() { [[ -f "$STATE_FILE" ]] && source "$STATE_FILE"; return 0; }
state_save() {
  (( DRY_RUN )) && return 0
  local k tmp="$STATE_FILE.tmp"
  { echo "# autotune state — $(date '+%F %T')"
    for k in "${!ST[@]}"; do printf 'ST[%q]=%q\n' "$k" "${ST[$k]}"; done | sort
  } > "$tmp" && mv -f "$tmp" "$STATE_FILE"
}
st() { printf '%s' "${ST[$1]:-${2:-}}"; }
st_set() { ST[$1]="$2"; state_save; }

decision() {   # kind id test result elo action patch
  [[ -f "$DEC_FILE" ]] || printf 'time\tkind\tid\ttest\tresult\telo\taction\tpatch\n' > "$DEC_FILE"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$@" >> "$DEC_FILE"
}

best_opts() { local s; s="$(st best_opts)"; [[ -n "$CHANCE_SCALING" ]] && s+=" ChanceScaling=$CHANCE_SCALING"; echo "$s"; }
opts_merge() {  # base-opts override-opts → merged (later wins, order kept)
  declare -A m=(); local order=() o k
  for o in $1 $2; do k="${o%%=*}"; [[ -z "${m[$k]+x}" ]] && order+=("$k"); m[$k]="${o#*=}"; done
  for k in "${order[@]}"; do printf '%s=%s ' "$k" "${m[$k]}"; done
}

# ----------------------------------------------------------------------------- matches
# match result: sets M_DEC (H1|H0|NONE), M_ELO, M_ERR, M_PAIRS, M_LINE; returns 1 if incomplete
match_parse() {
  local f="$1"
  M_DEC=""; M_ELO=0; M_ERR=999; M_PAIRS=0; M_LINE=""
  [[ -s "$f" ]] || return 1
  local last; last=$(grep -E '^(H1|H0|no decision|interrupted)' "$f" | tail -1)
  case "$last" in
    H1*) M_DEC=H1 ;; H0*) M_DEC=H0 ;; "no decision"*) M_DEC=NONE ;; *) return 1 ;;
  esac
  M_LINE=$(grep -E '^pairs ' "$f" | tail -1)
  [[ "$M_LINE" =~ ^pairs\ ([0-9]+) ]] && M_PAIRS=${BASH_REMATCH[1]}
  [[ "$M_LINE" =~ Elo\ ([-+]?[0-9.]+)\ \+/-\ ([0-9.]+|inf) ]] && { M_ELO=${BASH_REMATCH[1]}; M_ERR=${BASH_REMATCH[2]}; }
  [[ "$M_ERR" == inf ]] && M_ERR=999
  return 0
}

run_match() {   # run_match NAME match.py-args...   (result reused if already complete)
  local name="$1"; shift
  local f="$RES_DIR/$name.log"
  if match_parse "$f"; then log "  (already done) $name: $M_DEC $M_LINE"; return 0; fi
  rm -f "$f"
  run_logged "$f" "$PYTHON" -u match.py "$@" || { log "match.py failed ($name)"; return 1; }
  (( DRY_RUN )) && { M_DEC=NONE; M_ELO=0; M_ERR=999; return 0; }
  match_parse "$f" || { log "match not finished ($name)"; return 1; }
}

ext_names() {
  local e=()
  [[ -x "$AB_ENGINE" ]] && e+=(ab)
  [[ -x "$JQ_ENGINE" ]] && e+=(jq)
  echo "${e[@]}"
}

# gauntlet LABEL NET "OPTS" PAIRS → G_SUMMARY, G_AVG, G_CONVINCING (1/0), per engine G_ELO[e], G_ERR[e]
declare -A G_ELO=() G_ERR=()
gauntlet() {
  local label="$1" net="$2" opts="$3" pairs="$4" e
  G_SUMMARY=""; G_AVG=0; G_CONVINCING=1; G_ELO=(); G_ERR=()
  local sum=0 n=0
  for e in $(ext_names); do
    local eng engnet eopts
    if [[ $e == ab ]]; then eng="$AB_ENGINE"; engnet="$AB_NET"; eopts="$AB_OPTS"
    else eng="$JQ_ENGINE"; engnet=""; eopts="$JQ_OPTS"; fi
    local args=(--engine-a "$ENGINE" --net-a "$net" --engine-b "$eng"
                --movetime "$EXT_MOVETIME" --pairs "$pairs" --workers "$MATCH_WORKERS"
                --hash "$EXT_HASH" --seed "$EXT_SEED")
    [[ -n "$engnet" ]] && args+=(--net-b "$engnet") || args+=(--net-b "$net")
    local oa; read -r -a oa <<< "$opts"; (( ${#oa[@]} )) && args+=(--opt-a "${oa[@]}")
    local ob; read -r -a ob <<< "$eopts"; (( ${#ob[@]} )) && args+=(--opt-b "${ob[@]}")
    run_match "ext-${label}-${e}" "${args[@]}" || return 1
    G_ELO[$e]=$M_ELO; G_ERR[$e]=$M_ERR
    local name; [[ $e == ab ]] && name="AB-JChess" || name="jieqi"
    G_SUMMARY+="
• vs ${name}: Elo <b>${M_ELO}</b> ± ${M_ERR} (${M_PAIRS} pairs)  <code>$(html "$M_LINE")</code>"
    awk -v e="$M_ELO" -v r="$M_ERR" -v w="$WIN_MIN_ELO" 'BEGIN{exit !(e - r >= w)}' || G_CONVINCING=0
    sum=$(awk -v a="$sum" -v b="$M_ELO" 'BEGIN{print a+b}'); n=$((n + 1))
  done
  (( n > 0 )) || { G_CONVINCING=0; return 0; }
  G_AVG=$(awk -v a="$sum" -v n="$n" 'BEGIN{printf "%.1f", a/n}')
  (( n < 2 )) && G_CONVINCING=0      # both engines must be beaten
  return 0
}

# ----------------------------------------------------------------------------- nets / generations
cur_gen() {   # highest N with nets/genN.nnue
  local f g max=-1
  for f in nets/${LP}gen*.nnue; do
    [[ -f "$f" ]] || continue
    g=${f##*/${LP}gen}; g=${g%.nnue}
    [[ "$g" =~ ^[0-9]+$ ]] && (( g > max )) && max=$g
  done
  echo "$max"
}
net_of() { echo "nets/${LP}gen$1.nnue"; }

# Record the strength of a promoted net against the two outside engines
evaluate_net() {   # $1 = gen
  local g="$1" net; net=$(net_of "$g")
  [[ "$(st "ext_done_g$g")" == 1 ]] && return 0
  say "🎯 Measuring net <b>gen$g</b> against the 2 outside engines (${EXT_PAIRS} pairs, ${EXT_MOVETIME} ms/move)…"
  gauntlet "g${g}-$(opts_hash "$(best_opts)")" "$net" "$(best_opts)" "$EXT_PAIRS" || return 1
  ST[ext_avg_g$g]=$G_AVG; ST[ext_conv_g$g]=$G_CONVINCING
  ST[ext_sum_g$g]="$G_SUMMARY"
  local best; best=$(st best_ext -99999)
  local improved=0
  awk -v a="$G_AVG" -v b="$best" -v e="$EXT_EPS" 'BEGIN{exit !(a > b + e)}' && improved=1
  if (( improved )); then
    ST[best_ext]=$G_AVG; ST[best_gen]=$g; ST[best_conv]=$G_CONVINCING; ST[best_sum]="$G_SUMMARY"
    ST[plateau]=0
    cp -f "$net" "nets/${LP}best.nnue" 2>/dev/null
  else
    ST[plateau]=$(( $(st plateau 0) + 1 ))
  fi
  ST[ext_done_g$g]=1; state_save
  tg_send "$TAG · ✅ <b>gen$g</b> vs outside engines: average <b>${G_AVG}</b> Elo$( ((G_CONVINCING)) && echo ' · 🏆 convincing win against both')${G_SUMMARY}
Best: gen$(st best_gen) (${ST[best_ext]} Elo) · $( ((improved)) && echo '📈 improved' || echo "no improvement ($(st plateau)/${PATIENCE})")"
}

opts_hash() { printf '%s' "$1" | cksum | cut -d' ' -f1; }

# ----------------------------------------------------------------------------- pipeline.sh calls
write_run_env() {   # pipeline config for this call, on top of pipeline.env
  local f="$AT_DIR/run.env" lvl att
  lvl=$(st level 0); att=$(st "attempt_g$(cur_gen)" 1)
  (( lvl > LEVEL_CAP )) && lvl=$LEVEL_CAP
  local games=$(( GAMES + GAMES * lvl / 2 )) nodes=$(( NODES + NODES * lvl / 4 ))
  local epochs=$EPOCHS lr=$LR init=1 lambda=$LAMBDA
  case $(( att <= 1 ? 0 : (att - 2) % 3 + 1 )) in
    1) : ;;                                                     # more data, continue from the current net
    2) init=0; epochs=$(( EPOCHS * 3 / 2 )) ;;                  # train from scratch on all the data
    3) lr=$(awk -v l="$LR" 'BEGIN{printf "%g", l/2}')            # lower LR, mix in the game results
       [[ " $(st train_args) " == *" --lambda "* ]] || lambda=0.9 ;;
  esac
  (( att >= 5 )) && nodes=$(( nodes * 3 / 2 ))
  local gen_opts; gen_opts="$USER_GEN_OPTS $(printf '%s ' $(st best_opts) | tr ' ' '\n' | grep -v '^RevealQuiet' | tr '\n' ' ')"
  {
    [[ -f "$PIPE_ENV" ]] && echo "source $(printf '%q' "$PIPE_ENV")"
    printf 'GAMES=%q\nNODES=%q\nEPOCHS=%q\nLR=%q\nLAMBDA=%q\nINIT_FROM_CURRENT=%q\n' \
      "$games" "$nodes" "$epochs" "$lr" "$lambda" "$init"
    printf 'GEN_OPTS=%q\nMATCH_OPTS=%q\nTRAIN_ARGS=%q\n' "$gen_opts" \
      "$USER_MATCH_OPTS $(st best_opts)" "$USER_TRAIN_ARGS $(st train_args)"
    local me="$epochs" k; for k in 4 8; do (( epochs - k > 0 )) && me+=" $(( epochs - k ))"; done
    printf 'MATCH_EPOCHS=%q\nAUTO_PROMOTE=1\nENGINE=%q\n' "$me" "$ENGINE"
  } > "$f"
  RUN_INFO="data ${games} games × ${nodes} nodes · train ${epochs} epochs, lr ${lr}, λ ${lambda}, $( ((init)) && echo 'from the current net' || echo 'from scratch')"
}

pipeline() {   # pipeline ARGS...  (retries; pipeline.sh itself resumes)
  local try=0 rc
  while :; do
    log "\$ pipeline.sh $* --config $AT_DIR/run.env"
    if (( DRY_RUN )); then "$PIPELINE" "$@" --config "$AT_DIR/run.env" --dry-run; return 0; fi
    "$PIPELINE" "$@" --config "$AT_DIR/run.env" >> "$AT_DIR/logs/pipeline.log" 2>&1 9>&- &
    wait $!; rc=$?
    (( rc == 0 )) && return 0
    (( rc == 130 || rc == 143 )) && return "$rc"
    try=$(( try + 1 ))
    (( try >= RETRY_MAX )) && { say "❌ pipeline.sh $* failed $try times in a row (exit $rc) — stopping. See autotune/logs/pipeline.log"; return 1; }
    log "pipeline.sh exit $rc → retrying in $(( 60 * try ))s ($try/$RETRY_MAX)"
    sleep $(( 60 * try ))
  done
}

# ----------------------------------------------------------------------------- patches (work tree)
setup_work() {
  [[ -d "$WORK/.git" || -f "$WORK/.git" ]] && return 0
  if git -C "$ROOT_DIR" rev-parse --git-dir >/dev/null 2>&1; then
    local base; base=$(git -C "$ROOT_DIR" rev-parse HEAD)
    if ! git -C "$ROOT_DIR" diff --quiet HEAD -- src trainer/*.py 2>/dev/null; then
      say "⚠ src/ or trainer/*.py has uncommitted changes: patches are made on commit $base (HEAD). The bench check may differ → commit first if needed."
    fi
    local excl; excl="$(git -C "$ROOT_DIR" rev-parse --git-path info/exclude)"
    [[ "$excl" = /* ]] || excl="$ROOT_DIR/$excl"
    grep -qx 'trainer/autotune/' "$excl" 2>/dev/null || echo 'trainer/autotune/' >> "$excl"
    git -C "$ROOT_DIR" worktree add -f -B autotune/options "$WORK" "$base" >> "$LOGF" 2>&1 || return 1
    ST[base_sha]=$base
  else
    mkdir -p "$WORK" && cp -a "$SRC_DIR" "$WORK/src" && mkdir -p "$WORK/trainer" && cp -a "$TRAINER_DIR"/*.py "$WORK/trainer/"
    ( cd "$WORK" && git init -q && git add -A && git -c user.name=autotune -c user.email=autotune@localhost commit -qm "base" )
    ST[base_sha]=$(git -C "$WORK" rev-parse HEAD)
  fi
  git -C "$WORK" config user.name >/dev/null || git -C "$WORK" config user.name "maskfish-autotune"
  git -C "$WORK" config user.email >/dev/null || git -C "$WORK" config user.email "autotune@localhost"
  state_save
}

bench_sig() {   # bench_sig ENGINE NET OPTS... → node count
  local eng="$1" net="$2" o; shift 2
  { printf 'setoption name EvalFile value %s\n' "$(realpath "$net")"
    for o in "$@"; do printf 'setoption name %s value %s\n' "${o%%=*}" "${o#*=}"; done
    printf 'bench %s\nquit\n' "$BENCH_ARGS"; } | "$eng" 2>&1 | awk '/Nodes searched/{print $NF}' | tail -1
}

build_work() {   # clean build: header changes must reach every object file
  run_logged "$AT_DIR/logs/build-work.log" make -C "$WORK/src" clean >/dev/null
  run_logged "$AT_DIR/logs/build-work.log" make -C "$WORK/src" -j"$NCPU" build ARCH="$MAKE_ARCH" \
    $( [[ "$NNUE_ARCH" == att* ]] && echo nnue=att ) $( [[ "$NNUE_ARCH" == *threats ]] && echo threats=yes ) \
    $( (( LAYERS == 2 )) && echo layers=2 )
}

# engine_patch ACTION OPTION VALUE "MESSAGE" → commits + writes a patch; returns 1 (manual) if it cannot be verified
engine_patch() {
  local action="$1" opt="$2" value="$3" msg="$4"
  (( DRY_RUN )) && { echo "[patch] $action $opt $value"; return 0; }
  setup_work || return 1
  if [[ "$(git -C "$WORK" log -1 --format=%s 2>/dev/null)" == "$msg" ]]; then   # done before a restart
    patch_files_from_head; return 0
  fi
  local seq; seq=$(( $(st patch_seq 0) + 1 ))
  local net; net=$(net_of "$(cur_gen)")
  local expect eo; read -r -a eo <<< "$(st best_opts)"
  expect=$(bench_sig "$ENGINE" "$net" "${eo[@]}")
  local fold folds=("")
  [[ "$action" == remove ]] && folds+=("--no-fold")     # 2nd try: no dead-code folding
  for fold in "${folds[@]}"; do
    git -C "$WORK" checkout -q -- . 2>/dev/null
    local cmd=("$PYTHON" "$OPTPATCH" engine --src "$WORK/src" --option "$opt" --action "$action")
    [[ -n "$value" ]] && cmd+=(--value "$value")
    [[ -n "$fold" ]] && cmd+=("$fold")
    run_logged "$AT_DIR/logs/patch.log" "${cmd[@]}" || continue
    build_work || continue
    # the patched engine gets the other accepted options (in case an earlier patch
    # could not be made), but not the one this patch turns into a default
    local po=() o2; for o2 in "${eo[@]}"; do [[ "${o2%%=*}" == "$opt" ]] || po+=("$o2"); done
    local got; got=$(bench_sig "$WORK/src/maskfish" "$net" "${po[@]}")
    echo "bench $opt $action$fold: expected $expect got $got" >> "$AT_DIR/logs/patch.log"
    if [[ -n "$got" && "$got" == "$expect" ]]; then
      git -C "$WORK" commit -qam "$msg" -m "bench ($BENCH_ARGS): $got nodes — identical to the unpatched engine with: $(st best_opts)" \
        || { log "commit failed"; return 1; }
      patch_files_from_head
      return 0
    fi
  done
  git -C "$WORK" checkout -q -- .
  LAST_PATCH=""
  return 1
}

trainer_patch() {   # trainer_patch ACTION FLAG VALUE "MESSAGE"
  local action="$1" flag="$2" value="$3" msg="$4"
  (( DRY_RUN )) && { echo "[patch trainer] $action $flag $value"; return 0; }
  setup_work || return 1
  if [[ "$(git -C "$WORK" log -1 --format=%s 2>/dev/null)" == "$msg" ]]; then patch_files_from_head; return 0; fi
  git -C "$WORK" checkout -q -- .
  local cmd=("$PYTHON" "$OPTPATCH" trainer --file "$WORK/trainer/train.py" "--flag=$flag" --action "$action")
  [[ -n "$value" ]] && cmd+=(--value "$value")
  run_logged "$AT_DIR/logs/patch.log" "${cmd[@]}" || { git -C "$WORK" checkout -q -- .; return 1; }
  if "$PYTHON" -c "import torch" 2>/dev/null; then
    ( cd "$WORK/trainer" && "$PYTHON" train.py --help >/dev/null 2>>"$AT_DIR/logs/patch.log" ) \
      || { git -C "$WORK" checkout -q -- .; return 1; }
  fi
  git -C "$WORK" commit -qam "$msg" || return 1
  patch_files_from_head
}

patch_files_from_head() {   # numbered patch of HEAD (number = commits since the base) + combined patch
  local n; n=$(git -C "$WORK" rev-list --count "$(st base_sha)..HEAD")
  LAST_PATCH=$(git -C "$WORK" format-patch -1 --start-number "$n" -o "$PATCH_DIR" HEAD | tail -1)
  git -C "$WORK" diff "$(st base_sha)" HEAD > "$PATCH_DIR/all-options.patch"
  ST[patch_seq]=$n; state_save
}

# ----------------------------------------------------------------------------- engine option ablation
build_option_catalog() {   # lines: id|kind|values (space-separated, each = Name=V+Name=V)|removable|option
  CATALOG=()
  local name kind def lo hi
  while IFS=$'\t' read -r name kind def lo hi; do
    [[ "$kind" == bool ]] || continue
    [[ " $OPT_EXCLUDE " == *" $name "* ]] && continue
    local alt=true; [[ "$def" == true ]] && alt=false
    CATALOG+=("$name|bool|$name=$alt|1|$name")
  done < <("$PYTHON" "$OPTPATCH" list --src "$SRC_DIR")
  local t n v
  for t in $OPT_INT_TESTS; do
    n=${t%%:*}; v=""
    v=$(echo "${t#*:}" | tr ',' '\n' | sed "s/^/$n=/" | tr '\n' ' ')
    local rem=0; [[ " $OPT_INT_REMOVABLE " == *" $n "* ]] && rem=1
    CATALOG+=("$n|int|$v|$rem|$n")
  done
  for t in $OPT_GROUP_TESTS; do
    CATALOG+=("${t%%:*}|group|${t#*:}|0|${t%%:*}")
  done
  if [[ -n "$OPT_ONLY" ]]; then
    local keep=() c; for c in "${CATALOG[@]}"; do [[ " $OPT_ONLY " == *" ${c%%|*} "* ]] && keep+=("$c"); done
    CATALOG=("${keep[@]}")
  fi
}

option_ablation() {
  [[ "$(st opt_done)" == 1 ]] && return 0
  build_option_catalog
  local g; g=$(cur_gen); local net; net=$(net_of "$g")
  local total=${#CATALOG[@]} i=0
  say "🔬 Testing ${total} engine options with net gen$g (SPRT [${OPT_SPRT}], ${OPT_NODES} nodes, at most ${OPT_MAX_PAIRS} pairs)"
  local c
  for c in "${CATALOG[@]}"; do
    i=$((i + 1))
    IFS='|' read -r id kind values removable optname <<< "$c"
    local stv; stv="$(st "opt_$id")"
    local accepted="" v vname lastline="" elo=""
    if [[ "$stv" == accepting:* ]]; then            # stopped while writing the patch
      accepted="${stv#accepting:}"; values=""
    elif [[ -n "$stv" ]]; then continue; fi
    for v in $values; do
      local tryopts="${v//+/ }"
      vname=$(printf '%s' "$v" | tr -c 'A-Za-z0-9=_-' '_')
      local sprt; read -r -a sprt <<< "$OPT_SPRT"
      local base; base=$(best_opts)
      local a=(--engine "$ENGINE" --net-a "$net" --nodes "$OPT_NODES" --workers "$MATCH_WORKERS"
               --pairs "$OPT_MAX_PAIRS" --sprt "${sprt[@]}")
      local oa ob; read -r -a oa <<< "$(opts_merge "$base" "$tryopts")"; read -r -a ob <<< "$base"
      (( ${#oa[@]} )) && a+=(--opt-a "${oa[@]}")
      (( ${#ob[@]} )) && a+=(--opt-b "${ob[@]}")
      run_match "opt-g${g}-${vname}-$(opts_hash "$base")" "${a[@]}" || return 1
      lastline="$M_LINE"
      local res="$M_DEC"; elo="$M_ELO±$M_ERR"
      if [[ "$res" == H1 && "$OPT_EXT_CONFIRM" == 1 && -n "$(ext_names)" ]]; then
        gauntlet "optbase-g${g}-$(opts_hash "$base")" "$net" "$base" "$OPT_EXT_PAIRS" || return 1
        local avg0=$G_AVG
        gauntlet "optnew-g${g}-$(opts_hash "${oa[*]}")" "$net" "${oa[*]}" "$OPT_EXT_PAIRS" || return 1
        if awk -v n="$G_AVG" -v o="$avg0" -v t="$OPT_EXT_TOL" 'BEGIN{exit !(n >= o - t)}'; then
          elo+=" · outside ${avg0}→${G_AVG}"
        else
          res="H1-ext-fail"; elo+=" · outside ${avg0}→${G_AVG} (worse)"
        fi
      fi
      decision engine "$id" "$tryopts" "$res" "$elo" "-" "-"
      if [[ "$res" == H1 ]]; then accepted="$tryopts"; break; fi
      log "  $id [$tryopts]: $res $elo"
    done
    if [[ -n "$accepted" ]]; then
      if [[ "$stv" != accepting:* ]]; then          # best_opts + pending mark: one atomic save
        ST[best_opts]="$(opts_merge "$(st best_opts)" "$accepted")"; ST[opt_$id]="accepting:$accepted"; state_save
      fi
      local okp=1 o
      for o in $accepted; do
        if [[ "$kind" == bool ]]; then engine_patch set "${o%%=*}" "${o#*=}" "${o%%=*}: ${o#*=} by default (autotune: +Elo, SPRT H1)" || okp=0
        else engine_patch set "${o%%=*}" "${o#*=}" "${o%%=*}: default ${o#*=} (autotune: +Elo, SPRT H1)" || okp=0; fi
      done
      st_set "opt_$id" "accepted:$accepted"
      decision engine "$id" "$accepted" "ACCEPT" "$elo" "default" "$( ((okp)) && basename "${LAST_PATCH:-}" || echo MANUAL)"
      tg_send "$TAG · ✅ [$i/$total] <b>$(html "$id")</b> → <b>KEPT, on by default</b> ($(html "$accepted"))
Elo $(html "$elo")$( ((okp)) || echo '
⚠ Could not make a verified patch (bench mismatch) → edit by hand; the option is still used in the runs.')
<code>$(html "$lastline")</code>"
      [[ $okp == 1 && -n "${LAST_PATCH:-}" ]] && tg_doc "$LAST_PATCH" "patch: $id"
    else
      local act="kept"
      if [[ "$removable" == 1 ]]; then
        if engine_patch remove "$optname" "" "$optname: removed (autotune: no Elo gain)"; then act="removed"
        else act="manual"; fi
      fi
      st_set "opt_$id" "$act"
      decision engine "$id" "$values" "REJECT" "${elo:-}" "$act" "$( [[ $act == removed ]] && basename "${LAST_PATCH:-}" || echo -)"
      local what
      case "$act" in
        removed) what="<b>REMOVED from the code</b> (patch $(html "$(basename "${LAST_PATCH:-}")"))" ;;
        manual)  what="no Elo gain; ⚠ could not remove it automatically (bench mismatch) → remove by hand" ;;
        *)       what="no Elo gain → default value kept (this option is not removable)" ;;
      esac
      tg_send "$TAG · ❎ [$i/$total] <b>$(html "$id")</b>: ${what}
Elo $(html "${elo:-?}") <code>$(html "$lastline")</code>"
      [[ $act == removed && -n "${LAST_PATCH:-}" ]] && tg_doc "$LAST_PATCH" "patch: remove $id"
    fi
  done
  st_set opt_done 1
  say "🔬 Engine option tests done. Options in use: <code>$(html "$(st best_opts)")</code>"
  [[ -s "$PATCH_DIR/all-options.patch" ]] && tg_doc "$PATCH_DIR/all-options.patch" "Combined patch (git apply) so far"
  return 0
}

# ----------------------------------------------------------------------------- trainer flag ablation
collect_prefixes() {   # $1 = train|val, $2 = gen
  local kind="$1" g="$2" k a from=$TRAIN_FROM
  [[ "$kind" == val ]] && from=$g
  for (( k = from; k <= g; k++ )); do
    for a in "$PREP_DIR/${LP}${kind}${k}.arch" "$PREP_DIR/${LP}${kind}${k}"_*.arch; do
      [[ -f "$a" ]] && echo "${a%.arch}"
    done
  done
}

abl_train() {   # abl_train DIR "extra args" → DIR/epoch_ABL_EPOCHS.nnue
  local dir="$1" extra="$2" g; g=$(cur_gen)
  [[ -f "$dir/epoch_${ABL_EPOCHS}.nnue" ]] && return 0
  local train=() val=() ex
  mapfile -t train < <(collect_prefixes train "$g"); mapfile -t val < <(collect_prefixes val "$g")
  (( ${#train[@]} )) || { log "no prepared data in $PREP_DIR"; return 1; }
  local cmd=("$PYTHON" -u train.py "${train[@]}")
  (( ${#val[@]} )) && cmd+=(--val "${val[@]}")
  cmd+=(--out "$dir" --epochs "$ABL_EPOCHS" --epoch-size "$ABL_EPOCH_SIZE" --lr "$LR" --lambda "$LAMBDA"
        --layers "$LAYERS" --checkpoint-every "$ABL_EPOCHS" --seed 42)
  [[ -n "${SCALING:-}" ]] && cmd+=(--scaling "$SCALING")
  [[ -n "$BATCH_SIZE" ]] && cmd+=(--batch-size "$BATCH_SIZE")
  [[ -n "${DEVICE:-}" ]] && cmd+=(--device "$DEVICE")
  if [[ -f "$dir/last.pt" ]]; then cmd+=(--resume "$dir/last.pt"); else cmd+=(--init-net "$(net_of "$g")"); fi
  read -r -a ex <<< "$USER_TRAIN_ARGS $(st train_args) $extra"; cmd+=("${ex[@]}")
  mkdir -p "$dir"
  run_logged "$dir/train.log" "${cmd[@]}" || return 1
  (( DRY_RUN )) && return 0
  [[ -f "$dir/epoch_${ABL_EPOCHS}.nnue" ]] || return 1
  run_logged "$dir/verify.log" "$PYTHON" -u verify.py --engine "$ENGINE" --net "$dir/epoch_${ABL_EPOCHS}.nnue" \
    || { log "verify.py: $dir is not bit-exact"; return 1; }
  rm -f "$dir"/epoch_*.pt
}

train_ablation() {
  [[ "$TRAIN_ABL" == 1 && "$(st train_abl_done)" != 1 ]] || return 0
  local g; g=$(cur_gen); local adir="runs/abl-g$g"
  say "🧪 Testing ${#TRAIN_VARIANTS[@]} trainer flag groups: every variant trains ${ABL_EPOCHS} epochs × ${ABL_EPOCH_SIZE} on the same data, then SPRT [${ABL_SPRT}] against the baseline"
  if [[ "$(st abl_base_done_g$g)" != 1 ]]; then
    abl_train "$adir/base" "" || { say "❌ training the baseline (ablation) failed — see $adir/base/train.log"; return 1; }
    st_set "abl_base_done_g$g" 1
    tg_send "$TAG · ✅ ablation: baseline trained ($adir/base)"
  fi
  local base="$adir/base/epoch_${ABL_EPOCHS}.nnue" v
  for v in "${TRAIN_VARIANTS[@]}"; do
    IFS='|' read -r id flag values removable <<< "$v"
    local stv; stv="$(st "tabl_$id")"
    local accepted="" val lastres=""
    if [[ "$stv" == accepting:* ]]; then accepted="${stv#accepting:}"; values=""
    elif [[ -n "$stv" ]]; then continue; fi
    for val in $values; do
      local vargs="$flag"; [[ "$val" != flag ]] && vargs+=" $val"
      local vd="$adir/${id}-${val}"
      abl_train "$vd" "$vargs" || { say "❌ training variant $id ($vargs) failed — see $vd/train.log"; return 1; }
      local sprt; read -r -a sprt <<< "$ABL_SPRT"
      local mo; read -r -a mo <<< "$(best_opts)"
      local a=(--engine "$ENGINE" --net-a "$vd/epoch_${ABL_EPOCHS}.nnue" --net-b "$base" --nodes "$MATCH_NODES"
               --workers "$MATCH_WORKERS" --pairs "$ABL_MAX_PAIRS" --sprt "${sprt[@]}")
      (( ${#mo[@]} )) && a+=(--opt-a "${mo[@]}" --opt-b "${mo[@]}")
      run_match "tabl-g${g}-${id}-${val}" "${a[@]}" || return 1
      lastres="$M_DEC Elo $M_ELO±$M_ERR"
      decision trainer "$id" "$vargs" "$M_DEC" "$M_ELO±$M_ERR" "-" "-"
      tg_send "$TAG · 🧪 trainer <b>$(html "$vargs")</b> vs baseline: <b>$M_DEC</b> · Elo $M_ELO ± $M_ERR"
      [[ "$M_DEC" == H1 ]] && { accepted="$vargs"; break; }
    done
    if [[ -n "$accepted" ]]; then
      if [[ "$stv" != accepting:* ]]; then
        ST[train_args]="$(st train_args) $accepted"; ST[tabl_$id]="accepting:$accepted"; state_save
      fi
      local pv=""; [[ "$accepted" == *" "* ]] && pv="${accepted#* }"
      trainer_patch set "$flag" "$pv" "train.py: $accepted by default (autotune: +Elo, SPRT H1)" || LAST_PATCH=""
      st_set "tabl_$id" "accepted:$accepted"
      decision trainer "$id" "$accepted" "ACCEPT" "$lastres" "default" "$(basename "${LAST_PATCH:-MANUAL}")"
      tg_send "$TAG · ✅ trainer <b>$(html "$accepted")</b> → KEPT, made the default"
      [[ -n "${LAST_PATCH:-}" ]] && tg_doc "$LAST_PATCH" "patch trainer: $id"
    else
      local act=kept
      if [[ "$removable" == 1 ]]; then
        if trainer_patch remove "$flag" "" "train.py: $flag removed (autotune: no Elo gain)"; then act=removed; else act=manual; fi
      fi
      st_set "tabl_$id" "$act"
      decision trainer "$id" "$flag $values" "REJECT" "$lastres" "$act" "$( [[ $act == removed ]] && basename "${LAST_PATCH:-}" || echo -)"
      tg_send "$TAG · ❎ trainer <b>$(html "$flag")</b>: no Elo gain → $( [[ $act == removed ]] && echo '<b>REMOVED from train.py</b>' || echo "$act")"
      [[ $act == removed && -n "${LAST_PATCH:-}" ]] && tg_doc "$LAST_PATCH" "trainer patch: remove $flag"
    fi
  done
  st_set train_abl_done 1
  say "🧪 Trainer flag tests done. In use: <code>$(html "$(st train_args)")</code>"
}

# ----------------------------------------------------------------------------- report / stop check
write_report() {
  local f="$AT_DIR/REPORT.md" g
  {
    echo "# Maskfish autotune — report ($(date '+%F %T'))"
    echo
    echo "- Best net (measured against the outside engines): **gen$(st best_gen ?)** → \`nets/${LP}best.nnue\`, average Elo **$(st best_ext ?)**"
    echo "- Convincing win against both engines (Elo − error ≥ ${WIN_MIN_ELO}): **$( [[ "$(st best_conv)" == 1 ]] && echo yes || echo not yet)**"
    echo "- Rounds without improvement: $(st plateau 0)/${PATIENCE} · scale-up level: $(st level 0)"
    echo "- Engine options in use: \`$(st best_opts)\`"
    echo "- Trainer flags in use: \`$(st train_args)\`"
    echo
    echo "## Elo per generation (vs AB-JChess / jieqi, average)"
    echo
    for (( g = 0; g <= $(cur_gen); g++ )); do
      [[ -n "$(st "ext_avg_g$g")" ]] && echo "- gen$g: $(st "ext_avg_g$g")$( [[ "$(st "ext_conv_g$g")" == 1 ]] && echo ' (convincing win)')"
    done
    echo
    echo "## Decision for every option"
    echo
    if [[ -f "$DEC_FILE" ]]; then
      echo '| time | kind | option | tested | result | Elo | action | patch |'
      echo '|---|---|---|---|---|---|---|---|'
      tail -n +2 "$DEC_FILE" | awk -F'\t' '{printf "| %s | %s | %s | `%s` | %s | %s | %s | %s |\n",$1,$2,$3,$4,$5,$6,$7,$8}'
    fi
    echo
    echo "## Applying the patches to the repository"
    echo
    echo '```'
    echo "cd $(realpath --relative-to="$PWD" "$ROOT_DIR" 2>/dev/null || echo ..)"
    echo "git am trainer/autotune/patches/0*.patch      # one commit each, in order"
    echo "# or: git apply trainer/autotune/patches/all-options.patch"
    echo '```'
  } > "$f"
  echo "$f"
}

check_stop() {   # returns 0 if finished
  local p; p=$(st plateau 0)
  if (( p >= PATIENCE )); then
    if [[ "$(st best_conv)" == 1 ]]; then
      local rep; rep=$(write_report)
      say "🏁 <b>FINISHED</b>: Elo is at its maximum (${PATIENCE} rounds without improvement) and net gen$(st best_gen) convincingly beats both engines.$(st best_sum)
Net: <code>nets/${LP}best.nnue</code> · Total time $(fmt_dur $(( $(date +%s) - $(st start_ts "$(date +%s)") )))"
      tg_doc "$rep" "Final report"
      [[ -s "$PATCH_DIR/all-options.patch" ]] && tg_doc "$PATCH_DIR/all-options.patch" "Final combined patch"
      st_set finished 1
      return 0
    fi
    ST[level]=$(( $(st level 0) + 1 )); ST[plateau]=0; state_save
    say "⛰ Elo flat for ${PATIENCE} rounds but both engines are NOT convincingly beaten yet → scaling up (level $(st level)): more data, more nodes$( [[ "$(st opt_done)" != 1 ]] && echo ', engine option tests')"
  fi
  if (( MAX_HOURS > 0 )) && (( $(date +%s) - $(st start_ts "$(date +%s)") > MAX_HOURS * 3600 )); then
    say "⏱ MAX_HOURS=${MAX_HOURS}h reached — stopping (run again to resume)."; write_report >/dev/null; exit 4
  fi
  return 1
}

# ----------------------------------------------------------------------------- one round
round() {
  local g; g=$(cur_gen)
  local next=$(( g + 1 )) att; att=$(st "attempt_g$g" 1)
  ST[attempt_g$g]=$att; state_save

  # 1. engine flags (once), before generating data so the data uses them
  if [[ "$(st opt_done)" != 1 ]] && { (( g >= OPT_AT_GEN )) || (( $(st level 0) > 0 )); }; then
    option_ablation || return 1
  fi

  # 2. data with the current net (an extra batch on every new attempt)
  write_run_env
  local tagargs=()
  (( att > 1 )) && tagargs=(--tag "a${att}")
  if [[ "$(st "data_done_g${g}_a$att")" != 1 ]]; then
    say "▶ gen$g · attempt $att: generating$( ((att > 1)) && echo " more") data$( ((att > 1)) && echo " (tag a$att)") — ${RUN_INFO}"
    pipeline -g "$g" --gen-only "${tagargs[@]}" || return 1
    st_set "data_done_g${g}_a$att" 1
  fi

  # 3. trainer flags (once)
  if [[ "$TRAIN_ABL" == 1 && "$(st train_abl_done)" != 1 ]] && { (( g >= TRAIN_ABL_AT_GEN )) || (( $(st level 0) > 0 )); }; then
    train_ablation || return 1
    write_run_env
  fi

  # 4. train → verify → SPRT vs current → promote (pipeline.sh)
  local rdir="runs/${LP}gen${next}"
  if [[ -d "$rdir" && "$(st "train_dir_owner_g$g")" != "$att" && ! -f "$(net_of "$next")" ]]; then
    # a finished run of an earlier (failed) attempt: archive it so this attempt trains anew
    local arch="${rdir}.try$(( att - 1 ))"; [[ -e "$arch" ]] && arch="${rdir}.try$(( att - 1 )).$(date +%s)"
    mv "$rdir" "$arch"
    (( KEEP_ATTEMPT_PT )) || rm -f "$arch"/*.pt
    log "previous training run archived: $arch"
  fi
  st_set "train_dir_owner_g$g" "$att"
  say "▶ gen$g · attempt $att: train → bit-exact check → SPRT against gen$g"
  pipeline -g "$g" --train-only || return 1

  if [[ -f "$(net_of "$next")" ]]; then
    say "🏆 <b>gen$next</b> won the SPRT against gen$g → new net"
    evaluate_net "$next" || return 1
  else
    ST[attempt_g$g]=$(( att + 1 )); ST[plateau]=$(( $(st plateau 0) + 1 )); state_save
    say "↩ gen$g · attempt $att: no epoch won the SPRT (plateau $(st plateau)/${PATIENCE}). Next: $(
      case $(( (att - 1) % 3 + 1 )) in 1) echo 'more data, continue training';; 2) echo 'more data, train from scratch, more epochs';; 3) echo 'more data, lower LR, λ 0.9';; esac)"
  fi
  write_report >/dev/null
}

# ----------------------------------------------------------------------------- preflight
preflight() {
  local err=() f
  for f in "$AB_ENGINE" "$JQ_ENGINE"; do      # copied binaries often lose the x bit
    [[ -f "$f" && ! -x "$f" ]] && chmod +x "$f" 2>/dev/null && log "chmod +x $f"
  done
  [[ -x "$PIPELINE" ]] || err+=("$PIPELINE not found")
  command -v "$PYTHON" >/dev/null || err+=("python not found: $PYTHON")
  command -v git >/dev/null || err+=("git is needed to make the patches")
  [[ -x "$AB_ENGINE" ]] || err+=("AB-JChess not found: $AB_ENGINE")
  [[ -n "$AB_NET" && -f "$AB_NET" ]] || err+=("AB-JChess network not found (AB_NET)")
  [[ -x "$JQ_ENGINE" ]] || err+=("jieqi engine not found: $JQ_ENGINE")
  (( $(cur_gen) >= 0 )) || err+=("no nets/${LP}gen0.nnue yet (run scripts/pipeline.sh -g 0 first)")
  if (( ${#err[@]} )); then
    local e; for e in "${err[@]}"; do log "ERROR: $e"; done
    tg_send "$TAG · ❌ Cannot start:
$(html "$(printf '• %s\n' "${err[@]}")")"
    return 1
  fi
  if [[ ! -x "$ENGINE" ]]; then
    log "build engine…"
    run_logged "$AT_DIR/logs/build.log" make -C "$SRC_DIR" -j"$NCPU" build ARCH="$MAKE_ARCH" || return 1
  fi
  # the two outside engines answer UCI?
  local e
  for e in "$AB_ENGINE" "$JQ_ENGINE"; do
    printf 'uci\nisready\nquit\n' | timeout 30 "$e" 2>/dev/null | grep -q readyok \
      || { say "❌ engine $e does not answer UCI (uci/isready)"; return 1; }
  done
  return 0
}

# ----------------------------------------------------------------------------- actions
case "$ACTION" in
  initcfg)
    if [[ -e "$SCRIPT_DIR/autotune.env" ]]; then echo "$SCRIPT_DIR/autotune.env already exists" >&2; exit 2; fi
    embedded_env > "$SCRIPT_DIR/autotune.env" && echo "Created $SCRIPT_DIR/autotune.env"; exit $? ;;
  tgtest)
    tg_send "$TAG · 🔔 Telegram works!" && echo "Sent."; exit $? ;;
  status)
    state_load
    echo "Current net: gen$(cur_gen) · best: gen$(st best_gen ?) ($(st best_ext ?) Elo, convincing: $(st best_conv 0))"
    echo "Plateau: $(st plateau 0)/$PATIENCE · level: $(st level 0) · engine options done: $(st opt_done 0) · trainer flags done: $(st train_abl_done 0)"
    echo "Engine options: $(st best_opts)"; echo "Trainer flags: $(st train_args)"
    [[ -f "$AT_DIR/.pid" ]] && kill -0 "$(cat "$AT_DIR/.pid")" 2>/dev/null && echo "Running (pid $(cat "$AT_DIR/.pid"))" || echo "Not running"
    [[ -f "$DEC_FILE" ]] && { echo; column -t -s $'\t' "$DEC_FILE" 2>/dev/null || cat "$DEC_FILE"; }
    exit 0 ;;
  stop)
    touch "$AT_DIR/.stop"
    if [[ -f "$AT_DIR/.pid" ]] && kill -0 "$(cat "$AT_DIR/.pid")" 2>/dev/null; then
      kill -TERM "$(cat "$AT_DIR/.pid")"; echo "Stop signal sent."
    else echo "No process is running."; fi
    exit 0 ;;
  report)
    state_load; r=$(write_report); echo "$r"; tg_doc "$r" "Autotune report"; exit 0 ;;
  cron)
    line="@reboot cd $(printf '%q' "$TRAINER_DIR") && nohup $(printf '%q' "$SELF") --supervise >> $(printf '%q' "$AT_DIR/supervise.log") 2>&1 &"
    ( crontab -l 2>/dev/null | grep -vF "$SELF --supervise"; echo "$line" ) | crontab - && echo "Added to crontab: $line"
    exit $? ;;
  supervise)
    rm -f "$AT_DIR/.stop"
    n=0
    while :; do
      "$SELF" ${CONFIG_AT:+--config "$CONFIG_AT"}; rc=$?
      [[ -f "$AT_DIR/.stop" ]] && { echo "stopped on request"; exit 0; }
      case $rc in
        0) exit 0 ;;                       # finished
        3) exit 3 ;;                       # already running
        4) exit 4 ;;                       # MAX_HOURS
        5) exit 5 ;;                       # missing engine / file: no automatic retry
      esac
      n=$(( n + 1 )); d=$(( n < 6 ? 60 * n : 600 ))
      tg_send "$TAG · 🔁 Process stopped (exit $rc) → restarting in ${d}s (restart $n)"
      sleep "$d"
    done ;;
esac

# ----------------------------------------------------------------------------- main run
exec 9>"$AT_DIR/.lock"
if command -v flock >/dev/null && ! flock -n 9; then echo "autotune is already running (lock $AT_DIR/.lock)" >&2; exit 3; fi
echo $$ > "$AT_DIR/.pid"
rm -f "$AT_DIR/.stop"
state_load
[[ "$(st finished)" == 1 ]] && { echo "Already finished (remove ST[finished] from $STATE_FILE to continue)."; write_report; exit 0; }
[[ -n "$(st start_ts)" ]] || st_set start_ts "$(date +%s)"

STOPPING=0
on_signal() {
  trap - INT TERM; STOPPING=1
  kill_children 2>/dev/null
  say "⏸ Stopped (signal $1). Run the same command again to resume where it stopped."
  exit 130
}
on_exit() { local rc=$?; (( rc != 0 && rc != 130 && rc != 3 && rc != 4 && rc != 5 && ! STOPPING )) && \
  tg_send "$TAG · ❌ Unexpected exit (exit $rc). Log: $(html "$LOGF")"; }
trap 'on_signal INT' INT; trap 'on_signal TERM' TERM; trap on_exit EXIT

preflight || exit 5
RESUMED=""; [[ -f "$STATE_FILE" && -n "$(st plateau)" ]] && RESUMED=" (resuming from the saved state)"
say "🚀 Started${RESUMED}: current net gen$(cur_gen); stops when Elo is at its maximum (${PATIENCE} rounds without gain) and AB-JChess + jieqi are convincingly beaten (Elo − error ≥ ${WIN_MIN_ELO})."

# baseline: the current net against the outside engines
evaluate_net "$(cur_gen)" || exit 1

while :; do
  check_stop && exit 0
  round || { say "❌ Loop failed — see $(html "$LOGF"). Run again (or --supervise) to resume."; exit 1; }
  (( DRY_RUN )) && { echo "(dry-run: stopping after 1 round)"; exit 0; }
done
exit 0
