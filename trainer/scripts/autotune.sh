#!/usr/bin/env bash
# =============================================================================
#  Maskfish — automatic training loop, in stages, until the Elo stops rising
#
#  Stage 1 · ref      gen0: the reference engine plays both sides and scores the
#                     positions; Maskfish learns from it → gen1.
#  Stage 2 · distill  the current Maskfish net plays, the reference engine scores every
#                     position; train → SPRT against the current net →
#                     promote. Every new net is measured against the reference engine.
#                     Ends when the net is close to the reference engine (Elo ≥
#                     NEAR_REF_ELO) or after DISTILL_PATIENCE failed attempts
#                     in a row (the reference engine has nothing more to teach).
#  Stage 3 · self     the net plays and scores its own games (deeper search,
#                     SELF_NODES); same train → SPRT → promote loop.
#  Stage 4 · options  once the net is strong enough (Elo vs the reference engine ≥
#                     OPT_START_ELO, or stage 3), every engine option is tested
#                     by SPRT; again every OPT_EVERY new nets and before
#                     stopping. Decisions:
#                       H1                       → on by default (verified patch)
#                       H0 and Elo ≤ 0           → removed from the code (verified patch)
#                       H0 and Elo > 0           → left as it is (small gain, not retested)
#                       no decision (max pairs)  → left as it is, retested in later
#                                                  cycles (OPT_RETRY times); the options
#                                                  that leaned positive are also tested
#                                                  together (small gains may add up)
#                     What is tested lives in the option plan (OPT_PLAN, default
#                     scripts/option-plan.env), not in this script: exclusions,
#                     numeric values, groups and tests on the clock (OPT_TC_TESTS:
#                     STC then LTC, never removed from the code). Without a plan
#                     no option cycle runs.
#  Stage 5 · loop     stages 3 + 4 repeat. After PATIENCE failed attempts in a
#                     row the data and search depth are scaled up (level +1,
#                     up to LEVEL_CAP). At LEVEL_CAP, PATIENCE more failures
#                     and an option cycle without any gain → finished: the
#                     Elo is at its maximum for this setup.
#
#  REF_MODE (autotune.env) chooses how the reference engine is used: teach (all the
#  stages above), measure (stage self from the start, the reference engine only
#  measures every new net) or off (stage self only, no reference engine at all).
#
#  Every generation is one call of scripts/pipeline.sh (gen → prepare → train →
#  verify → SPRT → promote). Every step is saved in trainer/autotune/state.sh:
#  running the same command again resumes where it stopped. Telegram reports
#  after every step (TG_BOT_TOKEN / TG_CHAT_ID in pipeline.env).
#
#  Usage (any directory):
#     scripts/autotune.sh --reset         # archive ALL old results (autotune, nets, data,
#                                         # runs, logs) to trainer/archive-DATE/, start from zero
#     scripts/autotune.sh                 # run / resume
#     nohup scripts/autotune.sh --supervise >> autotune/supervise.log 2>&1 &
#     scripts/autotune.sh --status        # stage, nets, Elo, option decisions
#     scripts/autotune.sh --stop          # stop safely (resume later)
#     scripts/autotune.sh --report        # write autotune/REPORT.md and send it to Telegram
#     scripts/autotune.sh --install-cron  # restart after a reboot
#     scripts/autotune.sh --init-config   # write scripts/autotune.env, option-plan.env from the examples
#     scripts/autotune.sh --dry-run       # print what the next round would run
#
#  Configuration: scripts/pipeline.env (Telegram, hardware, data, training,
#  the reference engine paths), scripts/autotune.env (stages, how options are
#  tested; see scripts/autotune.env.example) and scripts/option-plan.env (which
#  options are tested; see scripts/option-plan.env.example).
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
ACTION="run"; DRY_RUN=0; CONFIG_AT=""; RESET_YES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --supervise)     ACTION="supervise"; shift ;;
    --status)        ACTION="status"; shift ;;
    --stop)          ACTION="stop"; shift ;;
    --report)        ACTION="report"; shift ;;
    --reset)         ACTION="reset"; shift ;;
    --yes)           RESET_YES=1; shift ;;
    --install-cron)  ACTION="cron"; shift ;;
    --test-telegram) ACTION="tgtest"; shift ;;
    --init-config)   ACTION="initcfg"; shift ;;
    --config)        CONFIG_AT="${2:-}"; shift 2 ;;
    --dry-run)       DRY_RUN=1; shift ;;
    -h|--help)       sed -n '2,62p' "$SELF" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

# ----------------------------------------------------------------------------- config
PIPE_ENV="$SCRIPT_DIR/pipeline.env"
[[ -f "$PIPE_ENV" ]] && source "$PIPE_ENV"
USER_GEN_OPTS="${GEN_OPTS:-}"; USER_MATCH_OPTS="${MATCH_OPTS:-}"
[[ -z "$CONFIG_AT" && -f "$SCRIPT_DIR/autotune.env" ]] && CONFIG_AT="$SCRIPT_DIR/autotune.env"
if [[ -n "$CONFIG_AT" ]]; then
  [[ -f "$CONFIG_AT" ]] || { echo "Config file not found: $CONFIG_AT" >&2; exit 2; }
  source "$CONFIG_AT"
fi
# The option plan: which engine options the option cycles test and how (scripts/README.md 4.4).
# Its tests are temporary (an entry is deleted once decided), so none is defined in this script.
: "${OPT_PLAN:=$SCRIPT_DIR/option-plan.env}"
OPT_PLAN_OK=0
if [[ -f "$OPT_PLAN" ]]; then source "$OPT_PLAN"; OPT_PLAN_OK=1; fi

NCPU=$(nproc 2>/dev/null || echo 2)
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$TRAINER_DIR/.venv/bin/python" ]]; then PYTHON="$TRAINER_DIR/.venv/bin/python"; else PYTHON=python3; fi
fi
: "${TG_BOT_TOKEN:=}"; : "${TG_CHAT_ID:=}"; : "${TG_NAME:=$(hostname -s 2>/dev/null || hostname)}"
: "${MAKE_ARCH:=native}"; : "${NNUE_ARCH:=default}"; : "${LAYERS:=1}"
: "${GAMES:=50000}"; : "${NODES:=5000}"
: "${EPOCHS:=40}"; : "${LR:=8.75e-4}"; : "${LAMBDA:=1.0}"
: "${CHANCE_SCALING:=}"; : "${MATCH_NODES:=20000}"
[[ -n "${MATCH_WORKERS:-}" ]] || MATCH_WORKERS=$(( NCPU > 1 ? NCPU - 1 : 1 ))

# --- reference engine (same defaults as pipeline.sh) ---
: "${JIEQI_DIR:=$ROOT_DIR/jieqi}"
: "${REF_ENGINE:=$JIEQI_DIR/ref-engine}"
: "${REF_NET=$(ls "$JIEQI_DIR"/ref-engine*.nnue 2>/dev/null | sort | tail -1)}"   # empty: none
: "${REF_OPTS=}"
: "${REF_NAME:=$(basename "$REF_ENGINE")}"   # in the result names; a new name → nets measured again
# REF_MODE: how the reference engine is used, one switch for the whole loop
#   teach    stages ref → distill → self: it plays the gen0 games, scores the distill data and
#            measures every new net (NEAR_REF_ELO, OPT_START_ELO, EXT_VETO)
#   measure  no data from it: Maskfish plays and scores its own games from the start (stage
#            self); the reference engine only measures every new net (report, OPT_START_ELO)
#   off      not used at all (REF_ENGINE is not needed): stage self only, no Elo against an
#            engine outside the loop; option cycles from net OPT_START_GEN
# A stage ref / distill in the state goes to stage self as soon as REF_MODE is not teach.
: "${REF_MODE:=teach}"
case "$REF_MODE" in
  teach|measure|off) ;;
  *) echo "Invalid REF_MODE: $REF_MODE (teach | measure | off)" >&2; exit 5 ;;
esac

# --- stages (autotune.env) ---
: "${GEN0_GAMES:=$GAMES}"            # reference engine games for gen0
: "${GEN0_EPOCHS:=$EPOCHS}"
: "${DISTILL_NODES:=$NODES}"         # Maskfish nodes per move in the distillation games
: "${SELF_NODES:=$(( NODES * 2 ))}"  # Maskfish nodes per move when it scores its own games
: "${NEAR_REF_ELO:=-30}"              # distill → self once the net is this close to the reference engine
: "${DISTILL_PATIENCE:=3}"           # failed attempts in a row before scaling up (level +1) …
: "${DISTILL_LEVEL_CAP:=2}"          # … and at this level, before going to stage self
: "${PATIENCE:=3}"                   # stage self: failed attempts in a row before scaling up
: "${LEVEL_CAP:=4}"                  # scale-up levels: games ×(1+level/2), nodes ×(1+level/4)
: "${TRAIN_WINDOW:=4}"               # train on the data of the last N generations (0 = all)
: "${MAX_HOURS:=0}"                  # 0 = no time limit
: "${RETRY_MAX:=5}"                  # retries of a failed pipeline step before stopping

# --- measuring against the reference engine ---
: "${EXT_MOVETIME:=200}"             # ms per move (the same for both engines)
: "${EXT_PAIRS:=400}"                # game pairs per measurement (about ±25 Elo)
: "${EXT_HASH:=64}"
: "${EXT_SEED:=20261017}"            # fixed: every net plays the same openings
# EXT_REFEREE: match.py --referee, unset = $ENGINE; set it empty to let the reference engine judge
: "${EXT_GO:=}"                      # instead of EXT_MOVETIME: any limit of both engines' searches
                                     # as "go" arguments (an engine without movetime), e.g. "depth 12"
: "${ENGINE_TIMEOUT:=600}"           # seconds to wait for an answer of an engine before it is killed
: "${MAX_ENGINE_FAILURES:=20}"       # illegal moves / timeouts per worker before a match stops
: "${EXT_VETO:=1}"                   # stage distill: a new net clearly weaker than the current one
                                     # against the reference engine is not promoted
: "${MATCH_EARLIER:=}"               # when the last epoch loses its SPRT, also try EPOCHS - k for
                                     # each k (e.g. "4 8"); empty = the last epoch only

# --- engine option tests ---
: "${OPT_START_ELO:=$NEAR_REF_ELO}"    # start once Elo vs the reference engine ≥ this (or in stage self)
: "${OPT_START_GEN:=1}"              # … and not before net genN (e.g. 6 with REF_MODE=off from zero:
                                     # the first nets are too weak to be worth a week of option tests)
: "${OPT_EVERY:=3}"                  # a new option cycle every N new nets
: "${OPT_NODES:=$MATCH_NODES}"
: "${OPT_SPRT:=0 3}"
: "${OPT_MAX_PAIRS:=12000}"
: "${OPT_RETRY:=2}"                  # later cycles in which an undecided option is tested again
: "${OPT_GROUP:=1}"                  # test the undecided options that leaned positive together
: "${OPT_ONLY:=}"                    # only test these options (space separated)
# From the option plan (empty without one; see option-plan.env.example):
#   OPT_EXCLUDE        boolean options never tested at fixed nodes (output, threads, time
#                      management, speed, data or network format, no effect alone, or tested on
#                      the clock instead)
#   NO_DATA_OPTS       accepted options (name prefixes) never used to generate data
#   OPT_INT_TESTS      numeric options: Name:value,value     OPT_INT_REMOVABLE: those that may go
#   OPT_GROUP_TESTS    options tested together: id:Opt=V+Opt=V (a winning group is applied by hand)
#   OPT_TC_TESTS       tests on the clock, in order: every one is an SPRT at OPT_TC (STC) and,
#                      if it wins, at OPT_LTC; on by default (verified patch) only when both win;
#                      never removed from the code. A = options in use + the test's options,
#                      B = options in use. One word each:
#     id:Opt=V+Opt=V[:need=[!]Name,...][:alt=id][:sprt=E0,E1][:both=Opt=V+Opt=V][:workers=N]
#     need=Name    only when Name is in use with a non-default value; need=!Name: only when it
#                  is not (else skipped, looked at again in the next cycle)
#     alt=id       an alternative to the test id: if id was accepted, its options are set back
#                  to their defaults in A (a head-to-head match; the winner stays in use)
#     both=…       options of A and B (e.g. Threads=4), not part of the result; with Threads=T
#                  the default workers are physical cores / T
#     workers=N    workers of this test
#   A test whose options are all in use already is skipped.
: "${OPT_EXCLUDE:=}"; : "${NO_DATA_OPTS:=}"; : "${OPT_INT_TESTS:=}"; : "${OPT_INT_REMOVABLE:=}"
: "${OPT_GROUP_TESTS:=}"; : "${OPT_TC_TESTS:=}"
# the plan may write a list over several lines: one space between the words
for _v in OPT_EXCLUDE NO_DATA_OPTS OPT_INT_TESTS OPT_INT_REMOVABLE OPT_GROUP_TESTS OPT_TC_TESTS OPT_ONLY; do
  read -r -d '' -a _w <<< "${!_v}"; printf -v "$_v" '%s' "${_w[*]}"
done
: "${OPT_TC:=10+0.1}"                # STC of OPT_TC_TESTS (match.py --tc)
: "${OPT_LTC:=40+0.4}"               # LTC, after an STC win (empty: the STC decides)
: "${OPT_TC_SPRT:=0 5}"              # default bounds of OPT_TC_TESTS (sprt=… per test)
: "${OPT_TC_MAX_PAIRS:=6000}"
PHYS_CORES=$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
(( PHYS_CORES > 0 )) || PHYS_CORES=$(( $(nproc 2>/dev/null || echo 2) / 2 ))
(( PHYS_CORES > 0 )) || PHYS_CORES=1
# physical cores − 1: games on the clock need a core each
: "${OPT_TC_WORKERS:=$(( PHYS_CORES > 1 ? PHYS_CORES - 1 : 1 ))}"
: "${BENCH_ARGS:=16 1 13}"           # bench that every patch must reproduce exactly
: "${PIPELINE:=$SCRIPT_DIR/pipeline.sh}"

# ARCH_FULL, FEATURE_ARCH, BIN_BASE, MAKE_FLAGS, TRAIN_ARCH, LP, PREP_DIR (see arch.sh;
# META, BLEND, STACKS, CTX, PSQT, QUANT of pipeline.env)
source "$SCRIPT_DIR/arch.sh"
set_arch_vars || exit 2
ENGINE="${ENGINE:-$SRC_DIR/$BIN_BASE}"
# the engine option EvalScaling of pipeline.env (README 7.13) in every match
: "${EVAL_SCALING:=Complexity}"
EVAL_OPT=""; [[ "$EVAL_SCALING" != Complexity ]] && EVAL_OPT="EvalScaling=$EVAL_SCALING"

AT_DIR="$TRAINER_DIR/autotune"
WORK="$AT_DIR/work"
PATCH_DIR="$AT_DIR/patches"
RES_DIR="$AT_DIR/results"
LOGF="$AT_DIR/autotune.log"
STATE_FILE="$AT_DIR/state.sh"
DEC_FILE="$AT_DIR/decisions.tsv"
OPTPATCH="$SCRIPT_DIR/optpatch.py"

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
  [[ "$(ps -o pgid= -p "$1" 2>/dev/null | tr -d ' ')" == "$1" ]] && kill -TERM -- "-$1" 2>/dev/null
  kill -TERM "$1" 2>/dev/null
}
kill_children() { local p; for p in $(pgrep -P $$ 2>/dev/null); do kill_tree "$p"; done; }
if command -v setsid >/dev/null 2>&1; then GROUP=(setsid -w); else GROUP=(); fi

run_logged() {   # run_logged LOGFILE cmd...  (background + wait: signals are handled at once)
  local lf="$1"; shift
  echo "[$(date '+%F %T')] \$ $*" >> "$lf"
  (( DRY_RUN )) && { echo "\$ $*"; return 0; }
  ( "${GROUP[@]}" "$@" >> "$lf" 2>&1 9>&- ) &
  wait $!
}

# ----------------------------------------------------------------------------- state
declare -A ST=()
state_load() {
  [[ -f "$STATE_FILE" ]] && source "$STATE_FILE"
  return 0
}
state_save() {
  (( DRY_RUN )) && return 0
  local k tmp="$STATE_FILE.tmp"
  { echo "# autotune state — $(date '+%F %T')"
    for k in "${!ST[@]}"; do printf 'ST[%q]=%q\n' "$k" "${ST[$k]}"; done | sort
  } > "$tmp" && mv -f "$tmp" "$STATE_FILE"
}
st() { printf '%s' "${ST[$1]:-${2:-}}"; }
st_set() { ST[$1]="$2"; state_save; }

# The ext_g* / best_ext values are Elo against one reference engine (ST[ext_ref]). With another
# REF_NAME they are kept as <old name>_ext_g* … and every net is measured again (evaluate_net).
ref_switch() {
  local old; old=$(st ext_ref)
  [[ "$old" == "$REF_NAME" ]] && return 0
  [[ -z "$old" ]] && { st_set ext_ref "$REF_NAME"; return 0; }
  local k keys=("${!ST[@]}")
  for k in "${keys[@]}"; do
    case "$k" in ext_g*|exterr_g*|best_ext|best_ext_gen) ST[${old}_$k]="${ST[$k]}"; unset "ST[$k]" ;; esac
  done
  ST[ext_ref]=$REF_NAME; state_save
  say "🔄 Reference engine changed: ${old} → ${REF_NAME}. Elo values against ${old} kept as ${old}_ext_g*; nets are measured again against ${REF_NAME}"
}
gt() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a + 0 > b + 0)}'; }    # gt A B: A > B (decimals)
ge() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a + 0 >= b + 0)}'; }

decision() {   # cycle gen option tested result elo action patch
  [[ -f "$DEC_FILE" ]] || printf 'time\tcycle\tgen\toption\ttested\tresult\telo\taction\tpatch\n' > "$DEC_FILE"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$@" >> "$DEC_FILE"
}

best_opts() { st best_opts; }
opts_merge() {  # base-opts override-opts → merged (later wins, order kept)
  declare -A m=(); local order=() o k
  for o in $1 $2; do k="${o%%=*}"; [[ -z "${m[$k]+x}" ]] && order+=("$k"); m[$k]="${o#*=}"; done
  for k in "${order[@]}"; do printf '%s=%s ' "$k" "${m[$k]}"; done
}
opts_hash() { printf '%s' "$1" | cksum | cut -d' ' -f1; }
pipeline_match_opts() {   # user's + accepted (+ ChanceScaling); pipeline.sh adds EvalScaling itself
  local s; s="$(opts_merge "$USER_MATCH_OPTS" "$(best_opts)")"
  [[ -n "$CHANCE_SCALING" ]] && s="$(opts_merge "$s" "ChanceScaling=$CHANCE_SCALING")"
  echo $s
}
match_opts() {   # options of Maskfish in every match of autotune: EvalScaling + the above
  local s; s="$(opts_merge "$EVAL_OPT" "$(pipeline_match_opts)")"
  echo $s
}

cur_gen() {   # highest N with nets/genN.nnue (-1 if none)
  local f g max=-1
  for f in nets/${LP}gen*.nnue; do
    [[ -f "$f" ]] || continue
    g=${f##*/${LP}gen}; g=${g%.nnue}
    [[ "$g" =~ ^[0-9]+$ ]] && (( g > max )) && max=$g
  done
  echo "$max"
}
net_of() { echo "nets/${LP}gen$1.nnue"; }

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

run_match() {   # run_match NAME match.py-args...   (a finished result is reused)
  local name="$1"; shift
  local f="$RES_DIR/$name.log"
  if match_parse "$f"; then log "  (already done) $name: $M_DEC $M_LINE"; return 0; fi
  rm -f "$f"
  run_logged "$f" "$PYTHON" -u match.py "$@" || { log "match.py failed ($name), see $f"; return 1; }
  (( DRY_RUN )) && { M_DEC=NONE; M_ELO=0; M_ERR=999; M_PAIRS=0; M_LINE=""; return 0; }
  match_parse "$f" || { log "match not finished ($name)"; return 1; }
}

# Elo of a net against the reference engine → X_ELO, X_ERR, X_LINE
ref_gauntlet() {   # ref_gauntlet NAME NET "OPTS" PAIRS
  local name="$1" net="$2" opts="$3" pairs="$4"
  local args=(--engine-a "$ENGINE" --net-a "$net" --engine-b "$REF_ENGINE"
              --movetime "$EXT_MOVETIME" --pairs "$pairs" --workers "$MATCH_WORKERS"
              --hash "$EXT_HASH" --seed "$EXT_SEED"
              --engine-timeout "$ENGINE_TIMEOUT" --max-failures "$MAX_ENGINE_FAILURES")
  [[ -n "$EXT_GO" ]] && args+=(--go "$EXT_GO")
  [[ -n "$REF_NET" ]] && args+=(--net-b "$REF_NET")
  # Maskfish (default options, own process) gives the board state and judges the rules, so
  # the reference engine needs none of the Maskfish commands "d" / "go perft" / "rule"
  local referee="${EXT_REFEREE-$ENGINE}"
  [[ -n "$referee" ]] && args+=(--referee "$referee")
  local oa; read -r -a oa <<< "$opts"; (( ${#oa[@]} )) && args+=(--opt-a "${oa[@]}")
  local ob; read -r -a ob <<< "$REF_OPTS"; (( ${#ob[@]} )) && args+=(--opt-b "${ob[@]}")
  run_match "$name" "${args[@]}" || return 1
  X_ELO=$M_ELO; X_ERR=$M_ERR; X_LINE=$M_LINE
}

evaluate_net() {   # $1 = gen: strength of a promoted net against the reference engine
  local g="$1" net; net=$(net_of "$g")
  [[ "$REF_MODE" == off ]] && return 0
  [[ -n "$(st "ext_g$g")" ]] && return 0
  say "🎯 Measuring <b>gen$g</b> against the reference engine (${EXT_PAIRS} pairs, $( [[ -n "$EXT_GO" ]] && echo "go $(html "$EXT_GO")" || echo "${EXT_MOVETIME} ms/move"))…"
  # EXT_GO in the name (when set): a result with another search limit is not reused
  ref_gauntlet "ext-${REF_NAME}-g${g}-$(opts_hash "$(match_opts)${EXT_GO:+|$EXT_GO}")" "$net" "$(match_opts)" "$EXT_PAIRS" || return 1
  (( DRY_RUN )) && return 0
  ST[ext_g$g]=$X_ELO; ST[exterr_g$g]=$X_ERR
  if [[ -z "$(st best_ext)" ]] || gt "$X_ELO" "$(st best_ext)"; then ST[best_ext]=$X_ELO; ST[best_ext_gen]=$g; fi
  state_save
  tg_send "$TAG · 📊 <b>gen$g</b> vs the reference engine: <b>$(printf '%+.1f' "$X_ELO")</b> ± ${X_ERR} Elo
<code>$(html "$X_LINE")</code>
Stage: $(st stage) · best so far: gen$(st best_ext_gen) ($(st best_ext) Elo)"
}

# ----------------------------------------------------------------------------- pipeline.sh calls
level() { local l; l=$(st level 0); (( l > LEVEL_CAP )) && l=$LEVEL_CAP; echo "$l"; }

write_run_env() {   # $1 = source → pipeline config for this call, on top of pipeline.env
  local source="$1" f="$AT_DIR/run.env"
  local g; g=$(cur_gen); (( g < 0 )) && g=0
  local lvl; lvl=$(level)
  local att; att=$(st "attempt_g$g" 1)
  local games=$(( GAMES + GAMES * lvl / 2 )) nodes=$NODES epochs=$EPOCHS lr=$LR init=1 lambda=$LAMBDA
  case "$source" in
    ref)      games=$GEN0_GAMES; epochs=$GEN0_EPOCHS ;;
    distill) nodes=$(( DISTILL_NODES + DISTILL_NODES * lvl / 4 )) ;;
    self)    nodes=$(( SELF_NODES + SELF_NODES * lvl / 4 ))
             (( g == 0 )) && epochs=$GEN0_EPOCHS ;;   # REF_MODE measure / off: the first net, from material
  esac
  case $(( att <= 1 ? 0 : (att - 2) % 3 + 1 )) in
    1) : ;;                                                    # more data, continue from the current net
    2) init=0; epochs=$(( epochs * 3 / 2 )) ;;                 # train from scratch on all the data
    3) lr=$(awk -v l="$LR" 'BEGIN{printf "%g", l/2}')           # lower LR, mix in the game results
       lambda=$(awk -v l="$LAMBDA" 'BEGIN{printf "%g", (l > 0.9 ? 0.9 : l)}') ;;
  esac
  local from=0
  (( TRAIN_WINDOW > 0 && g - TRAIN_WINDOW + 1 > 0 )) && from=$(( g - TRAIN_WINDOW + 1 ))
  # the safe reveal bonus is a search preference: never in the data
  local o gen_opts="$USER_GEN_OPTS" nd
  for o in $(best_opts); do
    for nd in $NO_DATA_OPTS; do [[ "$o" == "$nd"* ]] && continue 2; done
    gen_opts+=" $o"
  done
  [[ -n "$CHANCE_SCALING" ]] && gen_opts+=" ChanceScaling=$CHANCE_SCALING"
  {
    [[ -f "$PIPE_ENV" ]] && echo "source $(printf '%q' "$PIPE_ENV")"
    printf 'GAMES=%q\nNODES=%q\nEPOCHS=%q\nLR=%q\nLAMBDA=%q\nINIT_FROM_CURRENT=%q\nTRAIN_FROM=%q\n' \
      "$games" "$nodes" "$epochs" "$lr" "$lambda" "$init" "$from"
    printf 'GEN_OPTS=%q\nMATCH_OPTS=%q\nCHANCE_SCALING=\n' "$gen_opts" "$(pipeline_match_opts)"
    # the reference engine settings, also those set in autotune.env only
    local v; for v in REF_ENGINE REF_NET REF_OPTS REF_NAME REF_GEN_NODES REF_LABEL_NODES REF_HASH \
                      REF_ADJUDICATE REF_SCALE_FILE REF_SCALE_GAMES REF_GEN_GO REF_LABEL_GO \
                      ENGINE_TIMEOUT MAX_ENGINE_FAILURES; do
      [[ -n "${!v+x}" ]] && printf '%s=%q\n' "$v" "${!v}"
    done
    local me="$epochs" k; for k in $MATCH_EARLIER; do (( epochs - k > 0 )) && me+=" $(( epochs - k ))"; done
    printf 'MATCH_EPOCHS=%q\nAUTO_PROMOTE=1\nENGINE=%q\nSOURCE=%q\n' "$me" "$ENGINE" "$source"
  } > "$f"
  RUN_INFO="${games} games, $(case $source in
      ref) echo "the reference engine plays and scores";;
      distill) echo "Maskfish plays ${nodes} nodes, the reference engine scores";;
      self) echo "Maskfish plays and scores, ${nodes} nodes";; esac) · train ${epochs} epochs on gen${from}..${g}, lr ${lr}, λ ${lambda}, $( ((init)) && echo 'from the current net' || echo 'from scratch')"
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

# ----------------------------------------------------------------------------- patches (git worktree)
setup_work() {
  [[ -d "$WORK/.git" || -f "$WORK/.git" ]] && return 0
  if git -C "$ROOT_DIR" rev-parse --git-dir >/dev/null 2>&1; then
    local base; base=$(git -C "$ROOT_DIR" rev-parse HEAD)
    if ! git -C "$ROOT_DIR" diff --quiet HEAD -- src 2>/dev/null; then
      say "⚠ src/ has uncommitted changes: patches are made on commit $base (HEAD). The bench check may fail → commit first."
    fi
    local excl; excl="$(git -C "$ROOT_DIR" rev-parse --git-path info/exclude)"
    [[ "$excl" = /* ]] || excl="$ROOT_DIR/$excl"
    grep -qx 'trainer/autotune/' "$excl" 2>/dev/null || echo 'trainer/autotune/' >> "$excl"
    git -C "$ROOT_DIR" worktree add -f -B autotune/options "$WORK" "$base" >> "$LOGF" 2>&1 || return 1
    ST[base_sha]=$base
  else
    mkdir -p "$WORK" && cp -a "$SRC_DIR" "$WORK/src"
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
    "${MAKE_FLAGS[@]}"
}

# engine_patch ACTION OPTION VALUE "MESSAGE" → commit + patch file; returns 1 if it cannot be verified
engine_patch() {
  local action="$1" opt="$2" value="$3" msg="$4"
  LAST_PATCH=""
  (( DRY_RUN )) && { echo "[patch] $action $opt $value"; return 0; }
  setup_work || return 1
  if [[ "$(git -C "$WORK" log -1 --format=%s 2>/dev/null)" == "$msg" ]]; then   # done before a restart
    patch_files_from_head; return 0
  fi
  local net; net=$(net_of "$(cur_gen)")
  local expect eo; read -r -a eo <<< "$(match_opts)"
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
    # the patched engine gets the other accepted options, not the one this patch makes a default
    local po=() o2; for o2 in "${eo[@]}"; do [[ "${o2%%=*}" == "$opt" ]] || po+=("$o2"); done
    local got; got=$(bench_sig "$WORK/src/maskfish" "$net" "${po[@]}")
    echo "bench $opt $action$fold: expected $expect got $got" >> "$AT_DIR/logs/patch.log"
    if [[ -n "$got" && "$got" == "$expect" ]]; then
      git -C "$WORK" commit -qam "$msg" -m "bench ($BENCH_ARGS): $got nodes — identical to the unpatched engine with: $(match_opts)" \
        || { log "commit failed"; return 1; }
      patch_files_from_head
      return 0
    fi
  done
  git -C "$WORK" checkout -q -- .
  return 1
}

patch_files_from_head() {
  local n; n=$(git -C "$WORK" rev-list --count "$(st base_sha)..HEAD")
  LAST_PATCH=$(git -C "$WORK" format-patch -1 --start-number "$n" -o "$PATCH_DIR" HEAD | tail -1)
  git -C "$WORK" diff "$(st base_sha)" HEAD > "$PATCH_DIR/all-options.patch"
}

# ----------------------------------------------------------------------------- engine options (stage 4)
build_option_catalog() {   # CATALOG lines: id|kind|values (space separated, each Name=V+Name=V)|removable|option
  CATALOG=(); OPT_DEF=()
  local name kind def lo hi t o spec
  # the options of the clock tests are not tested at fixed nodes as well
  local skip=" $OPT_EXCLUDE "
  for t in $OPT_TC_TESTS; do spec=${t#*:}; spec=${spec%%:*}; for o in ${spec//+/ }; do skip+="${o%%=*} "; done; done
  while IFS=$'\t' read -r name kind def lo hi; do
    OPT_DEF[$name]=$def
    [[ "$kind" == bool ]] || continue
    [[ "$skip" == *" $name "* ]] && continue
    local alt=true; [[ "$def" == true ]] && alt=false
    CATALOG+=("$name|bool|$name=$alt|1|$name")
  done < <("$PYTHON" "$OPTPATCH" list --src "$SRC_DIR")
  local t n v
  for t in $OPT_INT_TESTS; do
    n=${t%%:*}; v=$(echo "${t#*:}" | tr ',' '\n' | sed "s/^/$n=/" | tr '\n' ' ')
    local rem=0; [[ " $OPT_INT_REMOVABLE " == *" $n "* ]] && rem=1
    CATALOG+=("$n|int|$v|$rem|$n")
  done
  for t in $OPT_GROUP_TESTS; do CATALOG+=("${t%%:*}|group|${t#*:}|0|${t%%:*}"); done
  for t in $OPT_TC_TESTS; do CATALOG+=("${t%%:*}|tc|${t#*:}|0|${t%%:*}"); done
  plan_check
  if [[ -n "$OPT_ONLY" ]]; then
    local keep=() c; for c in "${CATALOG[@]}"; do [[ " $OPT_ONLY " == *" ${c%%|*} "* ]] && keep+=("$c"); done
    CATALOG=("${keep[@]}")
  fi
}

opt_sprt() {   # opt_sprt NAME "TRY-OPTS" → M_*  (A = options in use + TRY-OPTS, B = options in use)
  local name="$1" tryopts="$2" g net; g=$(cur_gen); net=$(net_of "$g")
  local base; base=$(match_opts)
  local sprt; read -r -a sprt <<< "$OPT_SPRT"
  local a=(--engine "$ENGINE" --net-a "$net" --nodes "$OPT_NODES" --workers "$MATCH_WORKERS"
           --pairs "$OPT_MAX_PAIRS" --sprt "${sprt[@]}")
  local oa ob; read -r -a oa <<< "$(opts_merge "$base" "$tryopts")"; read -r -a ob <<< "$base"
  (( ${#oa[@]} )) && a+=(--opt-a "${oa[@]}")
  (( ${#ob[@]} )) && a+=(--opt-b "${ob[@]}")
  run_match "opt-c$(st opt_cycles 0)-g${g}-${name}-$(opts_hash "$base")" "${a[@]}"
}

accept_opts() {   # accept_opts ID "OPTS" "ELO-TEXT" "STAT-LINE"
  local id="$1" accepted="$2" elo="$3" line="$4" cyc; cyc=$(st opt_cycles 0)
  if [[ "$(st "opt_$id")" != accepting:* ]]; then   # options in use + pending mark: one atomic save
    ST[best_opts]="$(opts_merge "$(best_opts)" "$accepted")"; ST[opt_$id]="accepting:$accepted"; state_save
  fi
  local okp=1 o patches=()
  for o in $accepted; do
    engine_patch set "${o%%=*}" "${o#*=}" "${o%%=*}: default ${o#*=} (autotune: SPRT H1, Elo ${elo})" || okp=0
    [[ -n "${LAST_PATCH:-}" ]] && patches+=("$(basename "$LAST_PATCH")")
  done
  ST[opt_$id]="default:$accepted"; ST[opt_gain_c$cyc]=1; state_save
  decision "$cyc" "$(cur_gen)" "$id" "$accepted" ACCEPT "$elo" default "$( ((okp)) && echo "${patches[*]:-}" || echo MANUAL)"
  tg_send "$TAG · ✅ <b>$(html "$id")</b> → <b>on by default</b> ($(html "$accepted"))
Elo $(html "$elo")$( ((okp)) || echo '
⚠ No verified patch (bench mismatch): edit by hand; the option is still used in the runs.')
<code>$(html "$line")</code>"
}

# Names of the option plan that are no options of the engine (any more, e.g. removed after their
# SPRT): reported once per cycle, so that the entry is deleted from the plan.
plan_check() {
  local names=() t spec m o w p unknown=""
  names+=($OPT_EXCLUDE)
  for t in $OPT_INT_TESTS; do names+=("${t%%:*}"); done
  for t in $OPT_GROUP_TESTS $OPT_TC_TESTS; do
    local IFS=':'; local parts=(${t#*:}); unset IFS
    for o in ${parts[0]//+/ }; do names+=("${o%%=*}"); done
    for m in "${parts[@]:1}"; do
      case "$m" in
        need=*) w=${m#need=}; for o in ${w//,/ }; do names+=("${o#!}"); done ;;
        both=*) w=${m#both=}; for o in ${w//+/ }; do names+=("${o%%=*}"); done ;;
      esac
    done
  done
  for w in "${names[@]}"; do
    [[ -n "$w" && -z "${OPT_DEF[$w]+x}" && " $unknown " != *" $w "* ]] && unknown+=" $w"
  done
  for p in $NO_DATA_OPTS; do   # name prefixes
    local hit=0; for w in "${!OPT_DEF[@]}"; do [[ "$w" == "$p"* ]] && { hit=1; break; }; done
    (( hit )) || unknown+=" $p*"
  done
  [[ -n "$unknown" ]] && say "⚠ Option plan $(html "$OPT_PLAN"): no such engine option (delete these entries):<code>$(html "$unknown")</code>"
  return 0
}

# ---- tests on the clock (OPT_TC_TESTS)
declare -A OPT_DEF=()   # option → default (build_option_catalog)
tc_parse() {   # tc_parse "Opt=V+Opt=V:mod=…" → TC_OPTS TC_NEED TC_ALT TC_SPRT TC_BOTH TC_WORKERS
  local spec="$1" m
  TC_OPTS="${spec%%:*}"; TC_OPTS="${TC_OPTS//+/ }"
  TC_NEED=""; TC_ALT=""; TC_SPRT="$OPT_TC_SPRT"; TC_BOTH=""; TC_WORKERS=""
  local mods=()
  if [[ "$spec" == *:* ]]; then local IFS=':'; mods=(${spec#*:}); unset IFS; fi
  for m in "${mods[@]}"; do
    case "$m" in
      need=*)    TC_NEED="${m#need=}"; TC_NEED="${TC_NEED//,/ }" ;;
      alt=*)     TC_ALT="${m#alt=}" ;;
      sprt=*)    TC_SPRT="${m#sprt=}"; TC_SPRT="${TC_SPRT//,/ }" ;;
      both=*)    TC_BOTH="${m#both=}"; TC_BOTH="${TC_BOTH//+/ }" ;;
      workers=*) TC_WORKERS="${m#workers=}" ;;
      *) log "⚠ OPT_TC_TESTS: unknown modifier '$m' (ignored)" ;;
    esac
  done
  if [[ -z "$TC_WORKERS" ]]; then   # several threads per engine: fewer games at a time
    local t=1 o; for o in $TC_BOTH; do [[ "${o%%=*}" == Threads ]] && t="${o#*=}"; done
    if (( t > 1 )); then TC_WORKERS=$(( PHYS_CORES / t > 0 ? PHYS_CORES / t : 1 )); else TC_WORKERS=$OPT_TC_WORKERS; fi
  fi
}
tc_need_ok() {   # 0 if every need= condition of the test holds (TC_NEED); else prints the first that fails
  local c; for c in $TC_NEED; do
    if [[ "$c" == '!'* ]]; then opt_in_use "${c#!}" && { printf '%s is in use' "${c#!}"; return 1; }
    else opt_in_use "$c" || { printf '%s is not in use' "$c"; return 1; }
    fi
  done
  return 0
}
opt_value() {   # value of option $1 in the options in use (empty: not set)
  local o v=""; for o in $(best_opts); do [[ "${o%%=*}" == "$1" ]] && v="${o#*=}"; done; printf '%s' "$v"
}
opt_in_use() {   # 0 if option $1 is in use with a value other than its default
  local v; v=$(opt_value "$1")
  [[ -n "$v" && "$v" != "${OPT_DEF[$1]:-}" ]]
}
opt_sprt_tc() {   # opt_sprt_tc NAME "TRY-OPTS" TC → M_*  (uses TC_SPRT TC_BOTH TC_WORKERS)
  local name="$1" tryopts="$2" tc="$3" g net; g=$(cur_gen); net=$(net_of "$g")
  local base; base=$(match_opts)
  local sprt; read -r -a sprt <<< "$TC_SPRT"
  local a=(--engine "$ENGINE" --net-a "$net" --tc "$tc" --workers "$TC_WORKERS"
           --pairs "$OPT_TC_MAX_PAIRS" --sprt "${sprt[@]}")
  local oa ob
  read -r -a oa <<< "$(opts_merge "$(opts_merge "$base" "$tryopts")" "$TC_BOTH")"
  read -r -a ob <<< "$(opts_merge "$base" "$TC_BOTH")"
  (( ${#oa[@]} )) && a+=(--opt-a "${oa[@]}")
  (( ${#ob[@]} )) && a+=(--opt-b "${ob[@]}")
  run_match "opt-c$(st opt_cycles 0)-g${g}-tc-${name}-$(opts_hash "$base|$TC_BOTH|$tc")" "${a[@]}"
}
tc_test() {   # tc_test I N CYCLE GEN ID SPEC STATE: one OPT_TC_TESTS entry (STC, then LTC)
  local i="$1" n="$2" cyc="$3" g="$4" id="$5" spec="$6" stv="$7"
  tc_parse "$spec"
  local why; if ! why=$(tc_need_ok); then log "  [$i/$n] $id: skipped, $why"; return 0; fi
  local tryopts="$TC_OPTS" o vs=""
  if [[ -n "$TC_ALT" && "$(st "opt_$TC_ALT")" == default:* ]]; then   # head-to-head with the winner
    for o in $(opt_value_list "$TC_ALT"); do vs+=" ${o%%=*}=${OPT_DEF[${o%%=*}]}"; done
    tryopts="$(opts_merge "$vs" "$TC_OPTS")"
  fi
  tryopts=$(echo $tryopts)
  local inuse=1; for o in $tryopts; do [[ "$(opt_value "${o%%=*}")" == "${o#*=}" ]] || inuse=0; done
  if (( inuse )); then log "  [$i/$n] $id: skipped, its options are in use already"; return 0; fi
  local res="" elo="" line="" tcv lvl
  for lvl in STC LTC; do
    tcv=$OPT_TC; [[ $lvl == LTC ]] && tcv=$OPT_LTC
    [[ -z "$tcv" ]] && break
    say "⏱ [$i/$n] <b>$(html "$id")</b> on the clock (${lvl} ${tcv}, SPRT [${TC_SPRT}], ${TC_WORKERS} workers): <code>$(html "$tryopts")</code>"
    opt_sprt_tc "${id}-$(tr 'A-Z' 'a-z' <<< "$lvl")" "$tryopts" "$tcv" || return 1
    res=$M_DEC; elo="$(printf '%+.1f' "$M_ELO")±$M_ERR"; line=$M_LINE
    decision "$cyc" "$g" "$id" "$tryopts @$lvl $tcv" "$res" "$elo" "-" "-"
    log "  [$i/$n] $id [$tryopts] $lvl $tcv: $res $elo"
    [[ "$res" == H1 ]] || break
  done
  (( DRY_RUN )) && return 0
  ST[opt_${id}_c]=$cyc; state_save
  if [[ "$res" == H1 ]]; then accept_opts "$id" "$tryopts" "$elo (clock)" "$line"; return 0; fi
  local act what
  if [[ "$res" == NONE ]]; then
    local k; k=$(( $(inc_count "$stv") + 1 ))
    act="inc:$k"; st_set "opt_$id" "inc:$k:$M_ELO"
    what="no decision after ${M_PAIRS} pairs → left as it is$( (( k <= OPT_RETRY )) && echo ', tested again in a later cycle')"
  else
    act=kept; st_set "opt_$id" kept
    what="no gain → left as it is (chance node options are never removed automatically)"
  fi
  decision "$cyc" "$g" "$id" "$tryopts" REJECT "$elo" "$act" "-"
  tg_send "$TAG · ❎ [$i/$n] <b>$(html "$id")</b> (clock): ${what}
Elo $(html "$elo") <code>$(html "$line")</code>"
}
opt_value_list() {   # options accepted for the test id $1 (from its state default:…)
  local s; s=$(st "opt_$1"); printf '%s' "${s#default:}"
}

inc_count() { local s="$1"; [[ "$s" == inc:* ]] || { echo 0; return; }; s=${s#inc:}; echo "${s%%:*}"; }

option_cycle() {
  local cyc
  if [[ "$(st opt_running)" != 1 ]]; then
    cyc=$(( $(st opt_cycles 0) + 1 ))
    ST[opt_cycles]=$cyc; ST[opt_running]=1; ST[opt_gain_c$cyc]=0; ST[opt_pos_c$cyc]=""; state_save
  else
    cyc=$(st opt_cycles)                         # resuming a cycle
  fi
  build_option_catalog
  local g; g=$(cur_gen)
  local todo=() c id stv
  for c in "${CATALOG[@]}"; do
    id=${c%%|*}; stv=$(st "opt_$id")
    [[ "$(st "opt_${id}_c")" == "$cyc" && "$stv" != accepting:* ]] && continue   # already decided in this cycle
    case "$stv" in
      ""|accepting:*) todo+=("$c") ;;
      inc:*) (( $(inc_count "$stv") <= OPT_RETRY )) && todo+=("$c") ;;
    esac
  done
  say "🔬 Option cycle ${cyc} with net gen$g: ${#todo[@]} option(s) to test (SPRT [${OPT_SPRT}], ${OPT_NODES} nodes, ≤ ${OPT_MAX_PAIRS} pairs; tests on the clock: ${OPT_TC}${OPT_LTC:+ then $OPT_LTC}, ${OPT_TC_WORKERS} workers)"
  local i=0 kind values removable optname
  for c in "${todo[@]}"; do
    IFS='|' read -r id kind values removable optname <<< "$c"
    i=$((i + 1))
    stv=$(st "opt_$id")
    if [[ "$stv" == accepting:* ]]; then accept_opts "$id" "${stv#accepting:}" "(resumed)" ""; continue; fi
    if [[ "$kind" == tc ]]; then tc_test "$i" "${#todo[@]}" "$cyc" "$g" "$id" "$values" "$stv" || return 1; continue; fi
    local v accepted="" best_elo="" lastline="" res="" elo="" anynone=0
    for v in $values; do
      local tryopts="${v//+/ }" vname; vname=$(printf '%s' "$v" | tr -c 'A-Za-z0-9=_-' '_')
      opt_sprt "$vname" "$tryopts" || return 1
      res=$M_DEC; elo="$(printf '%+.1f' "$M_ELO")±$M_ERR"; lastline=$M_LINE
      decision "$cyc" "$g" "$id" "$tryopts" "$res" "$elo" "-" "-"
      log "  [$i/${#todo[@]}] $id [$tryopts]: $res $elo"
      if [[ "$res" == H1 ]]; then accepted="$tryopts"; break; fi
      [[ "$res" == NONE ]] && anynone=1
      if [[ -z "$best_elo" ]] || gt "$M_ELO" "$best_elo"; then best_elo=$M_ELO; fi
    done
    (( DRY_RUN )) && continue
    ST[opt_${id}_c]=$cyc; state_save
    if [[ -n "$accepted" ]]; then accept_opts "$id" "$accepted" "$elo" "$lastline"; continue; fi
    local act what
    if (( ! anynone )) && ! gt "$best_elo" 0; then          # H0 everywhere and no positive estimate
      act=kept
      if [[ "$removable" == 1 ]]; then
        if engine_patch remove "$optname" "" "$optname: removed (autotune: SPRT H0, Elo ${elo})"; then act=removed
        else act=manual; fi
      fi
      st_set "opt_$id" "$act"
      case "$act" in
        removed) what="<b>removed from the code</b> (patch $(html "$(basename "${LAST_PATCH:-}")"))" ;;
        manual)  what="no gain; ⚠ could not be removed automatically (bench mismatch) → remove by hand" ;;
        *)       what="no gain → default kept (not removable)" ;;
      esac
    elif (( ! anynone )); then
      act=weak; st_set "opt_$id" weak
      what="H0 but Elo > 0 (a gain below ${OPT_SPRT##* } Elo, if any) → left as it is, not retested"
    else
      local k; k=$(( $(inc_count "$stv") + 1 ))
      act="inc:$k"; st_set "opt_$id" "inc:$k:$best_elo"
      what="no decision after ${M_PAIRS} pairs → left as it is$( (( k <= OPT_RETRY )) && echo ', tested again in a later cycle')"
      if [[ "$kind" == bool ]] && gt "$best_elo" 0; then ST[opt_pos_c$cyc]+=" $values"; state_save; fi
    fi
    decision "$cyc" "$g" "$id" "$values" REJECT "$elo" "$act" "$( [[ $act == removed ]] && basename "${LAST_PATCH:-}" || echo -)"
    tg_send "$TAG · ❎ [$i/${#todo[@]}] <b>$(html "$id")</b>: ${what}
Elo $(html "$elo") <code>$(html "$lastline")</code>"
  done

  # the undecided options that leaned positive, all together (small gains may add up)
  local pos; pos=$(st "opt_pos_c$cyc")
  if [[ "$OPT_GROUP" == 1 ]] && (( $(wc -w <<< "$pos") >= 2 )) && [[ "$(st "opt_grp_c$cyc")" != done ]]; then
    local tryopts="${pos//+/ }"
    say "🧩 Testing the $(wc -w <<< "$pos") undecided options that leaned positive together: <code>$(html "$tryopts")</code>"
    opt_sprt "group" "$tryopts" || return 1
    local gelo; gelo="$(printf '%+.1f' "$M_ELO")±$M_ERR"
    decision "$cyc" "$g" "group" "$tryopts" "$M_DEC" "$gelo" "-" "-"
    if [[ "$M_DEC" == H1 ]]; then
      local o gline="$M_LINE"; for o in $tryopts; do accept_opts "${o%%=*}" "$o" "$gelo (group)" "$gline"; done
    else
      tg_send "$TAG · 🧩 Group: ${M_DEC} (Elo $(html "$gelo")) → nothing changed"
    fi
    st_set "opt_grp_c$cyc" done
  fi

  ST[opt_running]=0; ST[opt_last_promos]=$(st promos 0); ST[opt_last_gen]=$g; state_save
  say "🔬 Option cycle ${cyc} done ($( [[ "$(st "opt_gain_c$cyc")" == 1 ]] && echo 'with gains' || echo 'no gain')). Options in use: <code>$(html "$(best_opts)")</code>"
  [[ -s "$PATCH_DIR/all-options.patch" ]] && tg_doc "$PATCH_DIR/all-options.patch" "Combined patch so far (git apply)"
  return 0
}

option_cycle_due() {   # 0 if an option cycle should run now
  (( OPT_PLAN_OK )) || return 1                                               # no option plan
  [[ "$(st opt_running)" == 1 ]] && return 0
  local g; g=$(cur_gen); (( g >= 1 && g >= OPT_START_GEN )) || return 1
  local strong=0
  [[ "$(st stage)" == self ]] && strong=1
  [[ -n "$(st "ext_g$g")" ]] && ge "$(st "ext_g$g")" "$OPT_START_ELO" && strong=1
  (( strong )) || return 1
  [[ -z "$(st opt_last_gen)" ]] && return 0                                   # first cycle
  (( $(st promos 0) - $(st opt_last_promos 0) >= OPT_EVERY )) && return 0      # every OPT_EVERY new nets
  return 1
}

# ----------------------------------------------------------------------------- one round
round() {
  local g; g=$(cur_gen); (( g < 0 )) && g=0
  local stage; stage=$(st stage ref)
  if [[ "$stage" == ref ]] && (( g >= 1 )); then stage=distill; st_set stage distill; fi
  # a net that was promoted but not measured or checked yet (stopped in between)
  if (( g >= 1 )); then
    evaluate_net "$g" || return 1
    check_new_net "$g" "$stage" || return 0
  fi

  if option_cycle_due; then option_cycle || return 1; fi

  local source="$stage" att; att=$(st "attempt_g$g" 1)
  ST[attempt_g$g]=$att; state_save
  write_run_env "$source"
  local tagargs=(); (( att > 1 )) && tagargs=(--tag "a${att}")
  if [[ "$(st "data_done_g${g}_a$att")" != 1 ]]; then
    say "▶ gen$g · stage <b>$stage</b> · attempt $att: generating data — ${RUN_INFO}"
    pipeline -g "$g" --source "$source" --gen-only "${tagargs[@]}" || return 1
    st_set "data_done_g${g}_a$att" 1
  fi

  local next=$(( g + 1 )) rdir="runs/${LP}gen$(( g + 1 ))"
  if [[ -d "$rdir" && "$(st "train_dir_owner_g$g")" != "$att" && ! -f "$(net_of "$next")" ]]; then
    local arch="${rdir}.try$(( att - 1 ))"; [[ -e "$arch" ]] && arch="${rdir}.try$(( att - 1 )).$(date +%s)"
    mv "$rdir" "$arch"; rm -f "$arch"/*.pt
    log "previous training run archived: $arch"
  fi
  st_set "train_dir_owner_g$g" "$att"
  say "▶ gen$g · attempt $att: train → bit-exact check → SPRT against gen$g"
  pipeline -g "$g" --source "$source" --train-only || return 1
  (( DRY_RUN )) && return 0

  if [[ -f "$(net_of "$next")" ]]; then
    ST[promos]=$(( $(st promos 0) + 1 )); state_save
    say "🏆 <b>gen$next</b> won the SPRT against gen$g (stage $stage)"
    evaluate_net "$next" || return 1
    if check_new_net "$next" "$stage" && [[ "$(st stage)" == distill ]] && ge "$(st "ext_g$next")" "$NEAR_REF_ELO"; then
      ST[stage]=self; ST[level]=0; state_save
      say "🔁 gen$next is close to the reference engine ($(st "ext_g$next") ≥ ${NEAR_REF_ELO} Elo) → stage <b>self</b>: the net plays and scores its own games (${SELF_NODES} nodes)"
    fi
  else
    count_failure "$g" "$stage" "↩ gen$g · attempt $att: no epoch won the SPRT"
  fi
  write_report >/dev/null
}

count_failure() {   # count_failure GEN STAGE "MESSAGE": one more failed attempt for net GEN
  local g="$1" stage="$2"
  ST[attempt_g$g]=$(( $(st "attempt_g$g" 1) + 1 )); ST[fails]=$(( $(st fails 0) + 1 )); state_save
  say "$3 ($(st fails) failure(s) in a row)"
  if [[ "$stage" == distill ]] && (( $(st fails) >= DISTILL_PATIENCE )); then
    # failures in stage distill are more often too little data than the reference engine's
    # scores running out: scale up first, then go to stage self (from level 0)
    if (( $(st level 0) < DISTILL_LEVEL_CAP )); then
      ST[level]=$(( $(st level 0) + 1 )); ST[fails]=0; state_save
      say "⛰ ${DISTILL_PATIENCE} failures in a row → scaling up to level $(st level) (stage distill): more games, deeper search"
    else
      ST[stage]=self; ST[fails]=0; ST[level]=0; state_save
      say "🔁 ${DISTILL_PATIENCE} failures in a row at level ${DISTILL_LEVEL_CAP} with the reference engine's scores → stage <b>self</b> (the net scores its own games)"
    fi
  elif [[ "$stage" == self ]] && (( $(st fails) >= PATIENCE && $(st level 0) < LEVEL_CAP )); then
    ST[level]=$(( $(st level 0) + 1 )); ST[fails]=0; state_save
    say "⛰ ${PATIENCE} failures in a row → scaling up to level $(st level): more games, deeper search"
  fi
}

# A net promoted in stage distill that is clearly weaker than the previous one against the
# reference engine (difference above the combined 95% errors) is taken back: the SPRT is
# self-play, the reference engine is the goal of the stage. Returns 1 if the net was taken back.
check_new_net() {   # check_new_net GEN STAGE
  local n="$1" stage="$2" p=$(( $1 - 1 ))
  [[ "$(st "checked_g$n")" == 1 ]] && return 0
  if (( EXT_VETO )) && [[ "$stage" == distill && -n "$(st "ext_g$p")" && -n "$(st "ext_g$n")" ]] && \
     awk -v a="$(st "ext_g$p")" -v ea="$(st "exterr_g$p" 0)" -v b="$(st "ext_g$n")" -v eb="$(st "exterr_g$n" 0)" \
         'BEGIN{exit !(a - b > sqrt(ea * ea + eb * eb))}'; then
    local ts; ts=$(date +%s)
    local dst="nets/archive/${LP}gen$n.vetoed-$ts.nnue" elo="$(st "ext_g$n") ± $(st "exterr_g$n")" f
    mkdir -p nets/archive && mv -f "$(net_of "$n")" "$dst" || return 0
    # a later gen$n is measured again (run_match reuses a finished result of the same name)
    for f in "$RES_DIR/ext-${REF_NAME}-g${n}-"*.log; do [[ -f "$f" ]] && mv -f "$f" "${f%.log}.vetoed-$ts.log"; done
    unset "ST[ext_g$n]" "ST[exterr_g$n]"
    ST[promos]=$(( $(st promos 0) - 1 )); state_save
    count_failure "$p" "$stage" "🚫 gen$n won the SPRT against gen$p but is clearly weaker against the reference engine ($elo vs $(st "ext_g$p") ± $(st "exterr_g$p") Elo) → not promoted, moved to $(html "$dst")"
    return 1
  fi
  ST[fails]=0; ST[checked_g$n]=1; state_save
  return 0
}

check_stop() {   # returns 0 if finished
  if [[ "$(st stage)" == self ]] && (( $(st level 0) >= LEVEL_CAP && $(st fails 0) >= PATIENCE )); then
    # last chance: an option cycle with the current net (unless one already ran on it)
    if (( OPT_PLAN_OK )) && [[ "$(st opt_last_gen)" != "$(cur_gen)" || "$(st opt_running)" == 1 ]]; then
      say "⛰ Level ${LEVEL_CAP} and ${PATIENCE} failures in a row → one more option cycle before stopping"
      option_cycle || { say "❌ Option cycle failed — see $(html "$LOGF")"; exit 1; }
      if [[ "$(st "opt_gain_c$(st opt_cycles)")" == 1 ]]; then st_set fails 0; return 1; fi
    fi
    local rep; rep=$(write_report)
    say "🏁 <b>FINISHED</b>: no new net after ${PATIENCE} attempts at the highest level, and no option gain: the Elo is at its maximum for this setup.
Final net: <b>gen$(cur_gen)</b> (<code>nets/${LP}gen$(cur_gen).nnue</code>) · vs the reference engine: $(st "ext_g$(cur_gen)" '?') Elo · best measured: gen$(st best_ext_gen) ($(st best_ext) Elo)
Options: <code>$(html "$(best_opts)")</code> · total time $(fmt_dur $(( $(date +%s) - $(st start_ts "$(date +%s)") )))"
    tg_doc "$rep" "Final report"
    [[ -s "$PATCH_DIR/all-options.patch" ]] && tg_doc "$PATCH_DIR/all-options.patch" "Final combined patch"
    st_set finished 1
    return 0
  fi
  if (( MAX_HOURS > 0 )) && (( $(date +%s) - $(st start_ts "$(date +%s)") > MAX_HOURS * 3600 )); then
    say "⏱ MAX_HOURS=${MAX_HOURS}h reached — stopping (run again to resume)."; write_report >/dev/null; exit 4
  fi
  return 1
}

write_report() {
  local f="$AT_DIR/REPORT.md" g
  {
    echo "# Maskfish autotune — report ($(date '+%F %T'))"
    echo
    echo "- Stage: **$(st stage ref)** · level $(st level 0)/${LEVEL_CAP} · failures in a row: $(st fails 0) · new nets: $(st promos 0)"
    echo "- Current net: **gen$(cur_gen)** · vs the reference engine: $(st "ext_g$(cur_gen)" '?') Elo · best measured: gen$(st best_ext_gen '?') ($(st best_ext '?') Elo)"
    echo "- Engine options in use: \`$(best_opts)\`"
    echo "- Option cycles: $(st opt_cycles 0)"
    echo
    echo "## Elo against the reference engine (${EXT_PAIRS} pairs, ${EXT_MOVETIME} ms/move)"
    echo
    for (( g = 1; g <= $(cur_gen); g++ )); do
      [[ -n "$(st "ext_g$g")" ]] && echo "- gen$g: $(st "ext_g$g") ± $(st "exterr_g$g")"
    done
    echo
    echo "## Option decisions"
    echo
    if [[ -f "$DEC_FILE" ]]; then
      echo '| time | cycle | gen | option | tested | result | Elo | action | patch |'
      echo '|---|---|---|---|---|---|---|---|---|'
      tail -n +2 "$DEC_FILE" | awk -F'\t' '{printf "| %s | %s | %s | %s | `%s` | %s | %s | %s | %s |\n",$1,$2,$3,$4,$5,$6,$7,$8,$9}'
    fi
    echo
    echo "## Applying the patches"
    echo
    echo '```'
    echo "git am trainer/autotune/patches/0*.patch      # one commit each, in order"
    echo "# or: git apply trainer/autotune/patches/all-options.patch"
    echo '```'
  } > "$f"
  echo "$f"
}

preflight() {
  local err=()
  [[ -f "$REF_ENGINE" && ! -x "$REF_ENGINE" ]] && chmod +x "$REF_ENGINE" 2>/dev/null
  [[ -x "$PIPELINE" ]] || err+=("$PIPELINE not found")
  [[ -f "$OPTPATCH" ]] || err+=("$OPTPATCH not found")
  command -v "$PYTHON" >/dev/null || err+=("python not found: $PYTHON")
  command -v git >/dev/null || err+=("git is needed to make the patches")
  if [[ "$REF_MODE" != off ]]; then
    [[ -x "$REF_ENGINE" ]] || err+=("Reference engine not found: $REF_ENGINE (put it in jieqi/, set REF_ENGINE, or REF_MODE=off)")
    [[ -z "$REF_NET" || -f "$REF_NET" ]] || err+=("Reference engine network not found: $REF_NET (set REF_NET, or empty for none)")
  fi
  if (( ${#err[@]} )); then
    local e; for e in "${err[@]}"; do log "ERROR: $e"; done
    tg_send "$TAG · ❌ Cannot start:
$(html "$(printf '• %s\n' "${err[@]}")")"
    return 1
  fi
  if [[ ! -x "$ENGINE" ]]; then
    log "build engine…"
    run_logged "$AT_DIR/logs/build.log" make -C "$SRC_DIR" -j"$NCPU" build ARCH="$MAKE_ARCH" \
      "${MAKE_FLAGS[@]}" || return 1
  fi
  [[ "$REF_MODE" == off ]] && return 0
  printf 'uci\nisready\nquit\n' | timeout 60 "$REF_ENGINE" 2>/dev/null | grep -q readyok \
    || { say "❌ The reference engine ($REF_ENGINE) does not answer uci/isready"; return 1; }
  return 0
}

# ----------------------------------------------------------------------------- actions
case "$ACTION" in
  initcfg)
    rc=0
    for f in autotune.env option-plan.env; do
      if [[ -e "$SCRIPT_DIR/$f" ]]; then echo "$SCRIPT_DIR/$f already exists (kept)" >&2
      else cp "$SCRIPT_DIR/$f.example" "$SCRIPT_DIR/$f" && echo "Created $SCRIPT_DIR/$f" || rc=1; fi
    done
    exit $rc ;;
  tgtest)
    tg_send "$TAG · 🔔 Telegram works!" && echo "Sent."; exit $? ;;
  reset)
    if [[ -f "$AT_DIR/.pid" ]] && kill -0 "$(cat "$AT_DIR/.pid")" 2>/dev/null; then
      echo "autotune is running (pid $(cat "$AT_DIR/.pid")): run --stop first" >&2; exit 3
    fi
    dst="$TRAINER_DIR/archive-$(date +%Y%m%d-%H%M%S)"
    items=(); for d in autotune nets data runs logs; do [[ -n "$(ls -A "$d" 2>/dev/null)" ]] && items+=("$d"); done
    (( ${#items[@]} )) || { echo "Nothing to reset."; exit 0; }
    echo "Moves trainer/{${items[*]}} to $dst and starts from zero."
    if (( ! RESET_YES )); then read -r -p "Continue? [y/N] " ans; [[ "$ans" == [yY]* ]] || exit 1; fi
    if [[ -e "$WORK/.git" ]] && git -C "$ROOT_DIR" rev-parse --git-dir >/dev/null 2>&1; then
      git -C "$ROOT_DIR" worktree remove --force "$WORK" 2>/dev/null
      git -C "$ROOT_DIR" worktree prune 2>/dev/null
      git -C "$ROOT_DIR" branch -D autotune/options >/dev/null 2>&1
    fi
    mkdir -p "$dst"
    for d in "${items[@]}"; do mv "$d" "$dst/"; done
    mkdir -p nets data runs
    echo "Archived to $dst (delete it when no longer needed: rm -rf $dst)"
    exit 0 ;;
esac

mkdir -p "$AT_DIR" "$PATCH_DIR" "$RES_DIR" "$AT_DIR/logs" nets data runs
case "$ACTION" in
  status)
    state_load
    echo "REF_MODE: $REF_MODE · stage: $(st stage ref) · level $(st level 0)/$LEVEL_CAP · failures in a row: $(st fails 0) · new nets: $(st promos 0)"
    echo "Current net: gen$(cur_gen) · vs the reference engine: $(st "ext_g$(cur_gen)" '?') Elo · best: gen$(st best_ext_gen '?') ($(st best_ext '?'))"
    echo "Option cycles: $(st opt_cycles 0)$( [[ "$(st opt_running)" == 1 ]] && echo ' (one running)') · options in use: $(best_opts)"
    if (( OPT_PLAN_OK )); then
      echo "Option plan: $OPT_PLAN ($(wc -w <<< "$OPT_TC_TESTS") clock tests, $(wc -w <<< "$OPT_GROUP_TESTS") groups, $(wc -w <<< "$OPT_INT_TESTS") numeric)"
    else echo "Option plan: none ($OPT_PLAN missing): no option cycle"; fi
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
      case $rc in 0|3|4|5) exit $rc ;; esac
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
[[ "$REF_MODE" == off ]] || ref_switch
[[ "$(st finished)" == 1 ]] && { echo "Already finished (remove ST[finished] from $STATE_FILE to continue)."; write_report; exit 0; }
[[ -n "$(st start_ts)" ]] || st_set start_ts "$(date +%s)"
case "$(st stage)" in ref|distill|self) ;; *) st_set stage ref ;; esac
if [[ "$REF_MODE" != teach && "$(st stage)" != self ]]; then
  old_stage=$(st stage); ST[stage]=self; ST[level]=0; ST[fails]=0; state_save
  say "🔁 REF_MODE=${REF_MODE}: no data from the reference engine → stage ${old_stage} → <b>self</b> (the net plays and scores its own games)"
fi

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
RESUMED=""; (( $(cur_gen) >= 1 )) && RESUMED=" (resuming)"
case "$REF_MODE" in
  teach)   PLAN="reference engine games (gen0) → the reference engine scores Maskfish games until ≥ ${NEAR_REF_ELO} Elo vs the reference engine → self-play → engine options from ${OPT_START_ELO} Elo" ;;
  measure) PLAN="self-play from the start (the reference engine only measures every new net) → engine options from gen${OPT_START_GEN}" ;;
  off)     PLAN="self-play only, no reference engine → engine options from gen${OPT_START_GEN}" ;;
esac
say "🚀 Started${RESUMED} (REF_MODE=${REF_MODE}): stage <b>$(st stage)</b>, current net gen$(cur_gen). Plan: ${PLAN} → loop until no gain at level ${LEVEL_CAP}."
(( OPT_PLAN_OK )) || say "⚠ No option plan ($(html "$OPT_PLAN")): no option cycle will run (see scripts/option-plan.env.example)"

while :; do
  check_stop && exit 0
  round || { say "❌ Loop failed — see $(html "$LOGF"). Run again (or --supervise) to resume."; exit 1; }
  (( DRY_RUN )) && { echo "(dry-run: stopping after 1 round)"; exit 0; }
done
