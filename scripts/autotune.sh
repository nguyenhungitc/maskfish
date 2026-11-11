#!/usr/bin/env bash
# =============================================================================
#  Maskfish — automatic training loop, in stages, until the Elo stops rising
#
#  Stage 1 · ref      gen0: the reference engine plays both sides and scores the
#                     positions; Maskfish learns from it → gen1. With a starting net
#                     (INIT_NET, or nets/maskfish.nnue) gen1 must beat it by SPRT;
#                     after DISTILL_PATIENCE failures the next stage starts from it.
#  Stage 2 · distill  the current Maskfish net plays, the reference engine scores every
#                     position; train → SPRT against the current net → promote.
#                     Every new net is measured against the reference engine. Ends
#                     when the net is close to the reference engine (Elo ≥
#                     NEAR_REF_ELO) or after DISTILL_PATIENCE failed attempts in a
#                     row at level DISTILL_LEVEL_CAP.
#  Stage 3 · self     the net plays and scores its own games (SELF_NODES); same
#                     train → SPRT → promote loop.
#  Stage 4 · options  once the net is strong enough (Elo vs the reference engine ≥
#                     OPT_START_ELO, or stage self), the engine options of the option
#                     plan (scripts/option-plan.env) are tested by SPRT; again every
#                     OPT_EVERY new nets and before stopping. Decisions:
#                       H1           → in use: the option is set (UCI) in every later
#                                      generation and match, and listed in the report
#                                      (to be made the default in src/ by hand)
#                       H0           → rejected
#                       no decision  → left as it is, tested again in later cycles
#                                      (OPT_RETRY times); the options that leaned
#                                      positive are also tested together
#                     Without a plan no option cycle runs.
#  Stage 5 · loop     stages 3 + 4 repeat. After PATIENCE failed attempts in a row the
#                     data and search depth are scaled up (level +1, up to LEVEL_CAP).
#                     At LEVEL_CAP, PATIENCE more failures and an option cycle without
#                     any gain → finished: the Elo is at its maximum for this setup.
#
#  Every generation trains from the current net (train/train.py --init: the
#  published net for gen0). A failed attempt is tried again differently: more
#  data (a new batch, --tag aK), then 1.5x the epochs, then a lower learning rate
#  and more weight on the game results.
#
#  REF_MODE (autotune.env) chooses how the reference engine is used: teach (all the
#  stages above), measure (stage self from the start, the reference engine only
#  measures every new net) or off (stage self only, no reference engine at all).
#
#  Every generation is two calls of scripts/pipeline.sh (--gen-only, then
#  --train-only). Every step is saved in autotune/state.sh: running the same
#  command again resumes where it stopped. Telegram reports after every step
#  (TG_BOT_TOKEN / TG_CHAT_ID in pipeline.env).
#
#  Usage (any directory):
#     scripts/autotune.sh --reset         # archive ALL old results (autotune, nets, data,
#                                         # runs, logs) to archive-DATE/, start from zero
#     scripts/autotune.sh                 # run / resume
#     nohup scripts/autotune.sh --supervise >> autotune/supervise.log 2>&1 &
#     scripts/autotune.sh --status        # stage, nets, Elo, option decisions
#     scripts/autotune.sh --stop          # stop safely (resume later)
#     scripts/autotune.sh --report        # write autotune/REPORT.md and send it to Telegram
#     scripts/autotune.sh --options       # only one option cycle with the current net (no training;
#                                         # nets/maskfish.nnue before gen1), then the report to Telegram
#     scripts/autotune.sh --install-cron  # restart after a reboot
#     scripts/autotune.sh --init-config   # write scripts/pipeline.env, autotune.env, option-plan.env from the examples
#     scripts/autotune.sh --dry-run       # print what the next round would run
#
#  Configuration: scripts/pipeline.env (Telegram, hardware, data, training, the
#  reference engine), scripts/autotune.env (stages, measuring, option tests; see
#  scripts/autotune.env.example) and scripts/option-plan.env (which options are
#  tested; see scripts/option-plan.env.example).
# =============================================================================
set -uo pipefail
if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4) )); then
  echo "bash >= 4.4 required (this is $BASH_VERSION)" >&2; exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 1
SELF="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
PIPELINE="$SCRIPT_DIR/pipeline.sh"

# ----------------------------------------------------------------------------- CLI
ACTION="run"; DRY_RUN=0; CONFIG_AT=""; RESET_YES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --supervise)     ACTION="supervise"; shift ;;
    --status)        ACTION="status"; shift ;;
    --stop)          ACTION="stop"; shift ;;
    --report)        ACTION="report"; shift ;;
    --options)       ACTION="options"; shift ;;
    --reset)         ACTION="reset"; shift ;;
    --yes)           RESET_YES=1; shift ;;
    --install-cron)  ACTION="cron"; shift ;;
    --test-telegram) ACTION="tgtest"; shift ;;
    --init-config)   ACTION="initcfg"; shift ;;
    --config)        CONFIG_AT="${2:-}"; shift 2 ;;
    --dry-run)       DRY_RUN=1; shift ;;
    -h|--help)       sed -n '2,66p' "$SELF" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $1 (--help)" >&2; exit 2 ;;
  esac
done

# ----------------------------------------------------------------------------- config
PIPE_ENV="$SCRIPT_DIR/pipeline.env"
[[ -f "$PIPE_ENV" ]] && source "$PIPE_ENV"
AT_ENV="${CONFIG_AT:-$SCRIPT_DIR/autotune.env}"
[[ -f "$AT_ENV" ]] && source "$AT_ENV"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then PYTHON="$ROOT/.venv/bin/python"; else PYTHON=python3; fi
fi
: "${TG_BOT_TOKEN:=}"; : "${TG_CHAT_ID:=}"; : "${TG_NAME:=$(hostname -s 2>/dev/null || hostname)}"
# reference engine (same defaults as pipeline.sh)
: "${JIEQI_DIR:=$ROOT/jieqi}"
: "${REF_ENGINE:=$JIEQI_DIR/ref-engine}"
: "${REF_NET=$(ls "$JIEQI_DIR"/ref-engine*.nnue 2>/dev/null | sort | tail -1)}"
: "${REF_OPTS=}"; : "${REF_NAME:=$(basename "$REF_ENGINE")}"; : "${REF_HASH:=16}"
: "${REF_MODE:=teach}"            # teach | measure | off
# stages
: "${GAMES:=20000}"; : "${NODES:=5000}"; : "${EPOCHS:=10}"; : "${LAMBDA:=0.7}"
[[ -n "${LR:-}" ]] || LR=5e-4   # the trainer's default
: "${GEN0_GAMES:=$GAMES}"; : "${GEN0_EPOCHS:=$(( EPOCHS * 2 ))}"
: "${DISTILL_NODES:=$NODES}"; : "${NEAR_REF_ELO:=-30}"; : "${DISTILL_PATIENCE:=3}"; : "${DISTILL_LEVEL_CAP:=2}"
: "${SELF_NODES:=$(( NODES * 2 ))}"; : "${PATIENCE:=3}"; : "${LEVEL_CAP:=4}"
: "${TRAIN_WINDOW:=4}"; : "${TRAIN_MIN_GEN:=0}"; : "${MAX_HOURS:=0}"; : "${RETRY_MAX:=5}"
# measuring against the reference engine
: "${EXT_LIMIT:=--movetime 200}"  # match.py limit of both engines (--movetime MS, --nodes N, --depth D or --tc B+I)
: "${EXT_PAIRS:=400}"; : "${EXT_HASH:=64}"; : "${EXT_SEED:=20261111}"; : "${EXT_VETO:=1}"
: "${MATCH_EARLIER:=}"            # also SPRT these epochs: "best" is always tried, then "last"
# options
: "${OPT_PLAN:=$SCRIPT_DIR/option-plan.env}"
: "${OPT_START_ELO:=-30}"; : "${OPT_START_GEN:=1}"; : "${OPT_EVERY:=3}"
: "${OPT_NODES:=20000}"; : "${OPT_SPRT:=0 3}"; : "${OPT_SPRT_MODEL:=normalized}"; : "${OPT_MAX_PAIRS:=12000}"; : "${OPT_RETRY:=2}"; : "${OPT_GROUP:=1}"
: "${OPT_TC:=10+0.1}"; : "${OPT_LTC=40+0.4}"; : "${OPT_TC_SPRT:=0 5}"; : "${OPT_TC_MAX_PAIRS:=6000}"
: "${USER_GEN_OPTS:=${GEN_OPTS:-}}"; : "${USER_MATCH_OPTS:=${MATCH_OPTS:-}}"

AT_DIR="$ROOT/autotune"; STATE_FILE="$AT_DIR/state.sh"; LOGF="$AT_DIR/autotune.log"
RES_DIR="$AT_DIR/results"; DEC_FILE="$AT_DIR/decisions.tsv"
ENGINE="$ROOT/build/pipeline/maskfish"
export JIEQI_NATIVE_LIBRARY="$ROOT/build/pipeline/lib/libjieqi_rules.so"
NCPU=$(nproc 2>/dev/null || echo 2)
: "${MATCH_WORKERS:=$(( NCPU > 1 ? NCPU - 1 : 1 ))}"

OPT_TESTS=(); OPT_TC_TESTS=(); OPT_PLAN_OK=0
if [[ -f "$OPT_PLAN" ]]; then source "$OPT_PLAN"; (( ${#OPT_TESTS[@]} + ${#OPT_TC_TESTS[@]} )) && OPT_PLAN_OK=1; fi

# ----------------------------------------------------------------------------- utilities
log() { mkdir -p "$AT_DIR"; echo "[$(date '+%F %T')] $*" | tee -a "$LOGF"; }
fmt_dur() { local s=$1; printf '%dh%02dm' $((s/3600)) $((s%3600/60)); }
tg_send() {
  (( DRY_RUN )) && { echo "TG: $1" | head -c 300; echo; return 0; }
  [[ -z "$TG_BOT_TOKEN" || -z "$TG_CHAT_ID" ]] && return 0
  curl -fsS -m 30 -o /dev/null "https://api.telegram.org/bot$TG_BOT_TOKEN/sendMessage" \
    --data-urlencode "chat_id=$TG_CHAT_ID" --data-urlencode "text=[$TG_NAME] ${1:0:4000}" || log "(Telegram failed)"
}
tg_doc() {   # tg_doc FILE CAPTION
  (( DRY_RUN )) || [[ -z "$TG_BOT_TOKEN" || -z "$TG_CHAT_ID" ]] && return 0
  curl -fsS -m 60 -o /dev/null "https://api.telegram.org/bot$TG_BOT_TOKEN/sendDocument" \
    -F "chat_id=$TG_CHAT_ID" -F "caption=[$TG_NAME] $2" -F "document=@$1" || log "(Telegram failed)"
}
say() { log "$*"; tg_send "$*"; }
kill_children() { local p; for p in $(pgrep -P $$ 2>/dev/null); do pkill -TERM -P "$p" 2>/dev/null; kill -TERM "$p" 2>/dev/null; done; }
gt() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a + 0 > b + 0)}'; }
ge() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a + 0 >= b + 0)}'; }

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
decision() {   # cycle gen option tested result elo action
  [[ -f "$DEC_FILE" ]] || printf 'time\tcycle\tgen\toption\ttested\tresult\telo\taction\n' > "$DEC_FILE"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$@" >> "$DEC_FILE"
}
ref_switch() {   # Elo values are against one reference engine: another one measures every net again
  local old; old=$(st ext_ref)
  [[ "$old" == "$REF_NAME" ]] && return 0
  [[ -z "$old" ]] && { st_set ext_ref "$REF_NAME"; return 0; }
  local k keys=("${!ST[@]}")
  for k in "${keys[@]}"; do
    case "$k" in ext_g*|exterr_g*|best_ext|best_ext_gen) ST[${old}_$k]="${ST[$k]}"; unset "ST[$k]" ;; esac
  done
  ST[ext_ref]=$REF_NAME; state_save
  say "🔄 Reference engine changed: $old → $REF_NAME; the nets are measured again"
}

# options as Name=Value lists (space separated here; pipeline.sh and match.py take them comma separated)
best_opts() { st best_opts; }
opts_merge() {   # base override → merged (later wins, order kept)
  declare -A m=(); local order=() o k
  for o in $1 $2; do k="${o%%=*}"; [[ -z "${m[$k]+x}" ]] && order+=("$k"); m[$k]="${o#*=}"; done
  for k in "${order[@]}"; do printf '%s=%s ' "$k" "${m[$k]}"; done
}
commas() { local s; s=$(echo $1); printf '%s' "${s// /,}"; }
match_opts() { opts_merge "$(echo "${USER_MATCH_OPTS//,/ }")" "$(best_opts)"; }
opts_hash() { printf '%s' "$1" | cksum | cut -d' ' -f1; }

cur_gen() {   # highest N with nets/genN.nnue (-1 if none)
  local f g max=-1
  for f in nets/gen*.nnue; do
    [[ -f "$f" ]] || continue
    g=${f##*/gen}; g=${g%.nnue}
    [[ "$g" =~ ^[0-9]+$ ]] && (( g > max )) && max=$g
  done
  echo "$max"
}
net_of() { echo "nets/gen$1.nnue"; }
opt_net() { if (( $1 >= 0 )); then net_of "$1"; else echo nets/maskfish.nnue; fi; }   # the net option tests use
opt_label() { if (( $1 >= 0 )); then echo "gen$1"; else echo "the published net"; fi; }

# ----------------------------------------------------------------------------- matches
# run_match NAME match.py-args... → M_DEC (H1|H0|NONE), M_ELO, M_ERR, M_PAIRS, M_LINE (a finished result is reused)
run_match() {
  local name="$1"; shift
  local dir="$RES_DIR/$name"
  M_DEC=NONE; M_ELO=0; M_ERR=999; M_PAIRS=0; M_LINE=""
  if [[ ! -f "$dir/summary.json" ]]; then
    rm -rf "$dir"
    log "  \$ match.py $* --out $dir"
    (( DRY_RUN )) && return 0
    "$PYTHON" tools/match.py "$@" --out "$dir" > "$dir.log" 2>&1 &
    wait $! || { log "match.py failed ($name), see $dir.log"; return 1; }
  else
    log "  (already done) $name"
  fi
  local line
  line=$("$PYTHON" - "$dir/summary.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
p = d["pent"]
res = {"H1": "H1", "H0": "H0"}.get(d.get("sprt", {}).get("result"), "NONE")
err = (p["elo_hi"] - p["elo_lo"]) / 2 if p["elo_hi"] - p["elo_lo"] < 2000 else 999
print(res, round(p["elo"], 1), round(err, 1), p["pairs"],
      f"pairs {p['pairs']} W {d['w']} D {d['d']} L {d['l']} Elo {p['elo']:+.1f} [{p['elo_lo']:+.1f}, {p['elo_hi']:+.1f}]")
EOF
) || return 1
  read -r M_DEC M_ELO M_ERR M_PAIRS M_LINE <<< "$line"
}

ref_spec() { printf 'ref=%s:%s' "$REF_ENGINE" "$(commas "${REF_NET:+EvalFile=$REF_NET} ${REF_OPTS//,/ }")"; }
evaluate_net() {   # evaluate_net GEN: Elo of a promoted net against the reference engine
  local g="$1" net; net=$(net_of "$g")
  [[ "$REF_MODE" == off ]] && return 0
  [[ -n "$(st "ext_g$g")" ]] && return 0
  say "🎯 Measuring gen$g against the reference engine ($EXT_PAIRS pairs, $EXT_LIMIT)…"
  # shellcheck disable=SC2086
  run_match "ext-$REF_NAME-g$g-$(opts_hash "$(match_opts)|$EXT_LIMIT")" \
    --engine "maskfish=$ENGINE:$(commas "EvalFile=$ROOT/$net $(match_opts)")" --engine "$(ref_spec)" \
    --games $(( 2 * EXT_PAIRS )) $EXT_LIMIT --hash "$EXT_HASH" --concurrency "$MATCH_WORKERS" --seed "$EXT_SEED" || return 1
  (( DRY_RUN )) && return 0
  ST[ext_g$g]=$M_ELO; ST[exterr_g$g]=$M_ERR
  if [[ -z "$(st best_ext)" ]] || gt "$M_ELO" "$(st best_ext)"; then ST[best_ext]=$M_ELO; ST[best_ext_gen]=$g; fi
  state_save
  say "📊 gen$g vs the reference engine: $(printf '%+.1f' "$M_ELO") ± $M_ERR Elo ($M_LINE). Stage: $(st stage) · best so far: gen$(st best_ext_gen) ($(st best_ext) Elo)"
}

# ----------------------------------------------------------------------------- pipeline.sh calls
level() { local l; l=$(st level 0); (( l > LEVEL_CAP )) && l=$LEVEL_CAP; echo "$l"; }

write_run_env() {   # write_run_env SOURCE → autotune/run.env for this call, on top of pipeline.env
  local source="$1" f="$AT_DIR/run.env"
  local g; g=$(cur_gen); (( g < 0 )) && g=0
  local lvl; lvl=$(level)
  local att; att=$(st "attempt_g$g" 1)
  local games=$(( GAMES + GAMES * lvl / 2 )) nodes=$NODES epochs=$EPOCHS lr=$LR init=1 lambda=$LAMBDA
  case "$source" in
    ref)     games=$GEN0_GAMES; epochs=$GEN0_EPOCHS ;;
    distill) nodes=$(( DISTILL_NODES + DISTILL_NODES * lvl / 4 )) ;;
    self)    nodes=$(( SELF_NODES + SELF_NODES * lvl / 4 ))
             (( g == 0 )) && epochs=$GEN0_EPOCHS ;;
  esac
  case $(( att <= 1 ? 0 : (att - 2) % 3 + 1 )) in
    1) : ;;                                                    # more data, continue from the current net
    2) epochs=$(( epochs * 3 / 2 )) ;;                         # longer, from the current net
    3) lr=$(awk -v l="$lr" 'BEGIN{printf "%g", l/2}')           # lower LR, more weight on the game results
       lambda=$(awk -v l="$lambda" 'BEGIN{printf "%g", (l > 0.6 ? 0.6 : l)}') ;;
  esac
  local from=0
  (( TRAIN_WINDOW > 0 && g - TRAIN_WINDOW + 1 > 0 )) && from=$(( g - TRAIN_WINDOW + 1 ))
  (( from < TRAIN_MIN_GEN && TRAIN_MIN_GEN <= g )) && from=$TRAIN_MIN_GEN
  local epochs_try="best last"; [[ -n "$MATCH_EARLIER" ]] && epochs_try="$MATCH_EARLIER"
  {
    [[ -f "$PIPE_ENV" ]] && echo "source $(printf '%q' "$PIPE_ENV")"
    [[ -f "$AT_ENV" ]] && echo "source $(printf '%q' "$AT_ENV")"   # its pipeline.sh values (SPRT, MATCH_NODES, ...) too
    printf 'GAMES=%q\nNODES=%q\nEPOCHS=%q\nLR=%q\nLAMBDA=%q\nINIT_FROM_CURRENT=%q\nTRAIN_FROM=%q\nTRAIN_WINDOW=0\n' \
      "$games" "$nodes" "$epochs" "$lr" "$lambda" "$init" "$from"
    printf 'GEN_OPTS=%q\nMATCH_OPTS=%q\nMATCH_EPOCHS=%q\nAUTO_PROMOTE=1\nSOURCE=%q\n' \
      "$(commas "$(opts_merge "$(echo "${USER_GEN_OPTS//,/ }")" "$(best_opts)")")" "$(commas "$(match_opts)")" "$epochs_try" "$source"
    local v; for v in REF_ENGINE REF_NET REF_OPTS REF_NAME REF_HASH; do printf '%s=%q\n' "$v" "${!v}"; done
  } > "$f"
  RUN_INFO="$games games, $(case $source in
      ref) echo "the reference engine plays and scores";;
      distill) echo "Maskfish plays $nodes nodes, the reference engine scores";;
      self) echo "Maskfish plays and scores, $nodes nodes";; esac) · train $epochs epochs on gen$from..$g, lr $lr, λ $lambda, $( ((init)) && echo 'from the current net' || echo 'from scratch')"
}

pipeline() {   # pipeline ARGS... → 0 done / promoted, 3 not promoted (retries on errors)
  local try=0 rc
  while :; do
    log "\$ pipeline.sh $* --config $AT_DIR/run.env"
    if (( DRY_RUN )); then "$PIPELINE" "$@" --config "$AT_DIR/run.env" --dry-run > /dev/null; return 0; fi
    "$PIPELINE" "$@" --config "$AT_DIR/run.env" >> "$AT_DIR/logs/pipeline.log" 2>&1 9>&- &
    wait $!; rc=$?
    (( rc == 0 || rc == 3 )) && return "$rc"
    (( rc == 130 || rc == 143 )) && return "$rc"
    try=$(( try + 1 ))
    (( try >= RETRY_MAX )) && { say "❌ pipeline.sh $* failed $try times in a row (exit $rc) — stopping. See autotune/logs/pipeline.log"; return 1; }
    log "pipeline.sh exit $rc → retrying in $(( 60 * try ))s ($try/$RETRY_MAX)"
    sleep $(( 60 * try ))
  done
}

# ----------------------------------------------------------------------------- engine options (stage 4)
opt_sprt() {   # opt_sprt NAME "TRY-OPTS" [TC] → M_*  (A = options in use + TRY, B = options in use)
  local name="$1" try="$2" tc="${3:-}" g; g=$(cur_gen)
  local net="EvalFile=$ROOT/$(opt_net "$g")" use; use=$(match_opts)
  local lim=(--nodes "$OPT_NODES") sprt=$OPT_SPRT pairs=$OPT_MAX_PAIRS
  [[ -n "$tc" ]] && { lim=(--tc "$tc"); sprt=$OPT_TC_SPRT; pairs=$OPT_TC_MAX_PAIRS; }
  # shellcheck disable=SC2086
  run_match "opt-g$g-$name-$(opts_hash "$use|$try|$tc|$sprt")" \
    --engine "try=$ENGINE:$(commas "$net $(opts_merge "$use" "$try")")" --engine "base=$ENGINE:$(commas "$net $use")" \
    --games $(( 2 * pairs )) "${lim[@]}" --hash 16 --concurrency "$MATCH_WORKERS" --seed $(( g < 0 ? 1 : g * 7919 + 1 )) \
    --sprt $sprt --sprt-model "$OPT_SPRT_MODEL"
}
accept_opts() {   # accept_opts "OPTS" ID "ELO-TEXT"
  ST[best_opts]=$(echo $(opts_merge "$(best_opts)" "$1")); ST["opt_$2"]=accepted; ST[opt_gain_c$(st opt_cycles)]=1; state_save
  say "✅ $2 → in use: $1 (Elo $3). Options in use: $(best_opts) — make them the defaults in src/search.cpp when convenient"
}
option_cycle() {
  local cyc g; g=$(cur_gen)
  if [[ "$(st opt_running)" != 1 ]]; then
    cyc=$(( $(st opt_cycles 0) + 1 ))
    ST[opt_cycles]=$cyc; ST[opt_running]=1; ST[opt_gain_c$cyc]=0; ST[opt_pos_c$cyc]=""; state_save
  else cyc=$(st opt_cycles); fi
  local todo=() e id stv known bad n
  known=" $(printf 'uci\nquit\n' | "$ENGINE" 2> /dev/null | awk '$1 == "option" && $2 == "name" {print $3}' | tr '\n' ' ') "
  for e in "${OPT_TESTS[@]}" "${OPT_TC_TESTS[@]/#/tc:}"; do
    id=${e%%|*}; id=${id#tc:}; stv=$(st "opt_$id")
    bad=""; for n in $(tr ' +' '\n\n' <<< "${e#*|}" | sed -n 's/=.*//p' | sort -u); do
      [[ "$known" == *" $n "* ]] || bad+=" $n"; done
    if [[ -n "$bad" && "$known" != "  " ]]; then log "  option plan: $id skipped, no such engine option:$bad"; continue; fi
    [[ "$(st "opt_${id}_c")" == "$cyc" ]] && continue
    case "$stv" in
      "") todo+=("$e") ;;
      inc:*) local k=${stv#inc:}; k=${k%%:*}; (( k <= OPT_RETRY )) && todo+=("$e") ;;
    esac
  done
  say "🔬 Option cycle $cyc with $(opt_label "$g"): ${#todo[@]} test(s) (SPRT [$OPT_SPRT], $OPT_NODES nodes; on the clock: $OPT_TC${OPT_LTC:+ then $OPT_LTC})"
  local i=0
  for e in "${todo[@]}"; do
    i=$(( i + 1 ))
    local clock=0; [[ "$e" == tc:* ]] && { clock=1; e=${e#tc:}; }
    id=${e%%|*}; local values=${e#*|}
    local v accepted="" best_elo="" res="" elo="" anynone=0
    for v in $values; do
      local try="${v//+/ }"
      if (( clock )); then
        opt_sprt "$id-stc" "$try" "$OPT_TC" || return 1
        if [[ "$M_DEC" == H1 && -n "$OPT_LTC" ]]; then opt_sprt "$id-ltc" "$try" "$OPT_LTC" || return 1; fi
      else
        opt_sprt "$id" "$try" || return 1
      fi
      res=$M_DEC; elo="$(printf '%+.1f' "$M_ELO")±$M_ERR"
      decision "$cyc" "$g" "$id" "$try" "$res" "$elo" "-"
      log "  [$i/${#todo[@]}] $id [$try]: $res $elo ($M_LINE)"
      [[ "$res" == H1 ]] && { accepted="$try"; break; }
      [[ "$res" == NONE ]] && anynone=1
      if [[ -z "$best_elo" ]] || gt "$M_ELO" "$best_elo"; then best_elo=$M_ELO; fi
    done
    (( DRY_RUN )) && continue
    ST[opt_${id}_c]=$cyc; state_save
    if [[ -n "$accepted" ]]; then accept_opts "$accepted" "$id" "$elo"; decision "$cyc" "$g" "$id" "$accepted" H1 "$elo" accepted; continue; fi
    if (( ! anynone )); then
      st_set "opt_$id" rejected; decision "$cyc" "$g" "$id" "$values" H0 "$elo" rejected
      tg_send "❎ [$i/${#todo[@]}] $id: H0 → rejected (Elo $elo)"
    else
      local stv k; stv=$(st "opt_$id"); k=0; [[ "$stv" == inc:* ]] && { k=${stv#inc:}; k=${k%%:*}; }
      k=$(( k + 1 )); st_set "opt_$id" "inc:$k:$best_elo"; decision "$cyc" "$g" "$id" "$values" NONE "$elo" "inc:$k"
      tg_send "❔ [$i/${#todo[@]}] $id: no decision after $M_PAIRS pairs → left as it is$( (( k <= OPT_RETRY )) && echo ', tested again later')"
      if (( ! clock )) && gt "$best_elo" 0 && [[ "$values" != *" "* ]]; then ST[opt_pos_c$cyc]+=" $values"; state_save; fi
    fi
  done
  local pos; pos=$(st "opt_pos_c$cyc")
  if [[ "$OPT_GROUP" == 1 ]] && (( $(wc -w <<< "$pos") >= 2 )) && [[ "$(st "opt_grp_c$cyc")" != done ]]; then
    local try="${pos//+/ }"
    say "🧩 Testing together the undecided options that leaned positive: $try"
    opt_sprt "group" "$try" || return 1
    decision "$cyc" "$g" group "$try" "$M_DEC" "$(printf '%+.1f' "$M_ELO")±$M_ERR" "-"
    if [[ "$M_DEC" == H1 ]]; then accept_opts "$try" "group" "$(printf '%+.1f' "$M_ELO")±$M_ERR"
    else tg_send "🧩 Group: $M_DEC → nothing changed"; fi
    st_set "opt_grp_c$cyc" done
  fi
  ST[opt_running]=0; ST[opt_last_promos]=$(st promos 0); ST[opt_last_gen]=$g; state_save
  say "🔬 Option cycle $cyc done ($( [[ "$(st "opt_gain_c$cyc")" == 1 ]] && echo 'with gains' || echo 'no gain')). Options in use: $(best_opts)"
}
option_cycle_due() {
  (( OPT_PLAN_OK )) || return 1
  [[ "$(st opt_running)" == 1 ]] && return 0
  local g; g=$(cur_gen); (( g >= 1 && g >= OPT_START_GEN )) || return 1
  local strong=0
  [[ "$(st stage)" == self ]] && strong=1
  [[ -n "$(st "ext_g$g")" ]] && ge "$(st "ext_g$g")" "$OPT_START_ELO" && strong=1
  (( strong )) || return 1
  [[ -z "$(st opt_last_gen)" ]] && return 0
  (( $(st promos 0) - $(st opt_last_promos 0) >= OPT_EVERY )) && return 0
  return 1
}

# ----------------------------------------------------------------------------- one round
round() {
  local g; g=$(cur_gen); (( g < 0 )) && g=0
  local stage; stage=$(st stage ref)
  if [[ "$stage" == ref ]] && (( $(cur_gen) >= 1 )); then stage=distill; st_set stage distill; fi
  if (( $(cur_gen) >= 0 )); then   # gen0 too: the starting net's Elo is the baseline
    evaluate_net "$g" || return 1
    check_new_net "$g" "$stage" || return 0
  fi
  if option_cycle_due; then option_cycle || return 1; fi

  local att; att=$(st "attempt_g$g" 1)
  ST[attempt_g$g]=$att; state_save
  write_run_env "$stage"
  local tagargs=(); (( att > 1 )) && tagargs=(--tag "a$att")
  if [[ "$(st "data_done_g${g}_a$att")" != 1 ]]; then
    say "▶ gen$g · stage $stage · attempt $att: generating data — $RUN_INFO"
    pipeline -g "$g" --source "$stage" --gen-only "${tagargs[@]}" || return 1
    st_set "data_done_g${g}_a$att" 1
  fi
  # a fresh start: nets/gen0.nnue was only made by the call above; its Elo is the baseline (and the veto's reference)
  if (( ! DRY_RUN )) && [[ -f "$(net_of "$g")" ]]; then
    evaluate_net "$g" || return 1
    [[ "$(st "checked_g$g")" == 1 ]] || st_set "checked_g$g" 1   # (check_new_net: nothing before gen0 to compare with)
  fi
  local next=$(( g + 1 )) rdir="runs/gen$(( g + 1 ))"
  if (( ! DRY_RUN )) && [[ -f "$rdir/.done-train" || -n "$(ls "$rdir"/best.* 2> /dev/null)" ]] && \
     [[ "$(st "train_dir_owner_g$g")" != "$att" && ! -f "$(net_of "$next")" ]]; then
    local arch="$rdir.try$(( att - 1 ))"; [[ -e "$arch" ]] && arch="$arch.$(date +%s)"
    mv "$rdir" "$arch"; rm -f "$arch"/*.pt
    local d; for d in runs/sprt-gen$next-*; do [[ -d "$d" ]] && mv "$d" "$arch/$(basename "$d")"; done
    log "previous training run archived: $arch"
  fi
  st_set "train_dir_owner_g$g" "$att"
  say "▶ gen$g · attempt $att: train → verify → SPRT against gen$g"
  local rc; pipeline -g "$g" --source "$stage" --train-only; rc=$?
  (( rc == 0 || rc == 3 )) || return 1
  (( DRY_RUN )) && return 0
  if [[ -f "$(net_of "$next")" ]]; then
    ST[promos]=$(( $(st promos 0) + 1 )); state_save
    say "🏆 gen$next won against gen$g (stage $stage)"
    evaluate_net "$next" || return 1
    if check_new_net "$next" "$stage" && [[ "$(st stage)" == distill && -n "$(st "ext_g$next")" ]] && ge "$(st "ext_g$next")" "$NEAR_REF_ELO"; then
      ST[stage]=self; ST[level]=0; state_save
      say "🔁 gen$next is close to the reference engine ($(st "ext_g$next") ≥ $NEAR_REF_ELO Elo) → stage self ($SELF_NODES nodes)"
    fi
  else
    count_failure "$g" "$stage" "↩ gen$g · attempt $att: no candidate won the SPRT"
  fi
  write_report > /dev/null
}

count_failure() {   # count_failure GEN STAGE "MESSAGE"
  local g="$1" stage="$2"
  ST[attempt_g$g]=$(( $(st "attempt_g$g" 1) + 1 )); ST[fails]=$(( $(st fails 0) + 1 )); state_save
  say "$3 ($(st fails) failure(s) in a row)"
  if [[ "$stage" == ref ]] && (( $(st fails) >= DISTILL_PATIENCE )); then
    ST[stage]=distill; ST[fails]=0; state_save   # the reference engine's own games did not improve gen0
    say "🔁 $DISTILL_PATIENCE failures in a row against gen0 → stage distill (gen0 plays, the reference engine scores)"
  elif [[ "$stage" == distill ]] && (( $(st fails) >= DISTILL_PATIENCE )); then
    if (( $(st level 0) < DISTILL_LEVEL_CAP )); then
      ST[level]=$(( $(st level 0) + 1 )); ST[fails]=0; state_save
      say "⛰ $DISTILL_PATIENCE failures in a row → level $(st level) (stage distill): more games, deeper search"
    else
      ST[stage]=self; ST[fails]=0; ST[level]=0; state_save
      say "🔁 $DISTILL_PATIENCE failures in a row at level $DISTILL_LEVEL_CAP → stage self"
    fi
  elif [[ "$stage" == self ]] && (( $(st fails) >= PATIENCE && $(st level 0) < LEVEL_CAP )); then
    ST[level]=$(( $(st level 0) + 1 )); ST[fails]=0; state_save
    say "⛰ $PATIENCE failures in a row → level $(st level): more games, deeper search"
  fi
}

# A net promoted in stage distill that is clearly weaker than the previous one against the
# reference engine (difference above the combined errors) is taken back. Returns 1 if so.
check_new_net() {   # check_new_net GEN STAGE
  local n="$1" stage="$2" p=$(( $1 - 1 ))
  [[ "$(st "checked_g$n")" == 1 ]] && return 0
  if (( EXT_VETO )) && [[ "$stage" == distill && -n "$(st "ext_g$p")" && -n "$(st "ext_g$n")" ]] && \
     awk -v a="$(st "ext_g$p")" -v ea="$(st "exterr_g$p" 0)" -v b="$(st "ext_g$n")" -v eb="$(st "exterr_g$n" 0)" \
         'BEGIN{exit !(a - b > sqrt(ea * ea + eb * eb))}'; then
    local ts dst; ts=$(date +%s); dst="nets/archive/gen$n.vetoed-$ts.nnue"
    mkdir -p nets/archive && mv -f "$(net_of "$n")" "$dst" || return 0
    local d; for d in "$RES_DIR/ext-$REF_NAME-g$n-"*; do [[ -e "$d" ]] && mv -f "$d" "$d.vetoed-$ts"; done
    local elo="$(st "ext_g$n") ± $(st "exterr_g$n")"
    unset "ST[ext_g$n]" "ST[exterr_g$n]"
    ST[promos]=$(( $(st promos 0) - 1 )); state_save
    count_failure "$p" "$stage" "🚫 gen$n won the SPRT but is clearly weaker against the reference engine ($elo vs $(st "ext_g$p") ± $(st "exterr_g$p")) → taken back ($dst)"
    return 1
  fi
  ST[fails]=0; ST[checked_g$n]=1; state_save
  return 0
}

check_stop() {   # 0 if finished
  if [[ "$(st stage)" == self ]] && (( $(st level 0) >= LEVEL_CAP && $(st fails 0) >= PATIENCE )); then
    if (( OPT_PLAN_OK )) && [[ "$(st opt_last_gen)" != "$(cur_gen)" || "$(st opt_running)" == 1 ]]; then
      say "⛰ Level $LEVEL_CAP and $PATIENCE failures in a row → one more option cycle before stopping"
      option_cycle || { say "❌ Option cycle failed — see $LOGF"; exit 1; }
      if [[ "$(st "opt_gain_c$(st opt_cycles)")" == 1 ]]; then st_set fails 0; return 1; fi
    fi
    local rep; rep=$(write_report)
    say "🏁 FINISHED: no new net after $PATIENCE attempts at the highest level, and no option gain. Final net: gen$(cur_gen) (nets/gen$(cur_gen).nnue) · vs the reference engine: $(st "ext_g$(cur_gen)" '?') Elo · options in use: $(best_opts) · total time $(fmt_dur $(( $(date +%s) - $(st start_ts "$(date +%s)") )))"
    tg_doc "$rep" "Final report"
    st_set finished 1
    return 0
  fi
  if (( MAX_HOURS > 0 )) && (( $(date +%s) - $(st start_ts "$(date +%s)") > MAX_HOURS * 3600 )); then
    say "⏱ MAX_HOURS=${MAX_HOURS}h reached — stopping (run again to resume)."; write_report > /dev/null; exit 4
  fi
  return 1
}

write_report() {
  local f="$AT_DIR/REPORT.md" g
  {
    echo "# Maskfish autotune — report ($(date '+%F %T'))"
    echo
    echo "- REF_MODE: $REF_MODE · stage: **$(st stage ref)** · level $(st level 0)/$LEVEL_CAP · failures in a row: $(st fails 0) · new nets: $(st promos 0)"
    echo "- Current net: **$(opt_label "$(cur_gen)")** ($(opt_net "$(cur_gen)")) · vs the reference engine: $(st "ext_g$(cur_gen)" '?') Elo · best measured: gen$(st best_ext_gen '?') ($(st best_ext '?') Elo)"
    echo "- Engine options in use (set as UCI options in every later generation; make them the defaults in src/ by hand): \`$(best_opts)\`"
    echo "- Option cycles: $(st opt_cycles 0)"
    echo
    echo "## Elo against the reference engine ($EXT_PAIRS pairs, $EXT_LIMIT)"
    echo
    for (( g = 0; g <= $(cur_gen); g++ )); do
      [[ -n "$(st "ext_g$g")" ]] && echo "- gen$g: $(st "ext_g$g") ± $(st "exterr_g$g")"
    done
    echo
    echo "## Option decisions"
    echo
    if [[ -f "$DEC_FILE" ]]; then
      echo '| time | cycle | gen | option | tested | result | Elo | action |'
      echo '|---|---|---|---|---|---|---|---|'
      tail -n +2 "$DEC_FILE" | awk -F'\t' '{printf "| %s | %s | %s | %s | `%s` | %s | %s | %s |\n",$1,$2,$3,$4,$5,$6,$7,$8}'
    fi
    echo
    echo "To publish the final net: \`scripts/push_net.sh nets/gen$(cur_gen).nnue\`."
  } > "$f"
  echo "$f"
}

preflight() {
  local err=()
  [[ -x "$PIPELINE" ]] || err+=("$PIPELINE not found or not executable")
  command -v "$PYTHON" > /dev/null || err+=("python not found: $PYTHON")
  if [[ "$REF_MODE" != off ]]; then
    [[ -x "$REF_ENGINE" ]] || err+=("reference engine not found: $REF_ENGINE (put it in jieqi/, set REF_ENGINE, or REF_MODE=off)")
    [[ -z "$REF_NET" || -f "$REF_NET" ]] || err+=("reference engine network not found: $REF_NET (set REF_NET, or empty for none)")
  fi
  case "$REF_MODE" in teach|measure|off) ;; *) err+=("REF_MODE must be teach, measure or off") ;; esac
  if [[ "$REF_MODE" != teach && ! -f nets/gen0.nnue && -z "${INIT_NET:-}" ]] && (( $(cur_gen) < 0 )); then
    sh scripts/net.sh > /dev/null 2>&1
    [[ -f nets/maskfish.nnue ]] || err+=("REF_MODE=$REF_MODE starts from a net: none in nets/ (scripts/net.sh), and no INIT_NET")
  fi
  if (( ${#err[@]} )); then
    local e; for e in "${err[@]}"; do log "ERROR: $e"; done
    tg_send "❌ Cannot start: $(printf '%s; ' "${err[@]}")"
    return 1
  fi
  [[ "$REF_MODE" == off ]] && return 0
  printf 'uci\nisready\nquit\n' | timeout 60 "$REF_ENGINE" 2> /dev/null | grep -q readyok \
    || { say "❌ The reference engine ($REF_ENGINE) does not answer uci/isready"; return 1; }
}

# ----------------------------------------------------------------------------- actions
case "$ACTION" in
  initcfg)
    rc=0
    for f in pipeline.env autotune.env option-plan.env; do
      if [[ -e "$SCRIPT_DIR/$f" ]]; then echo "$SCRIPT_DIR/$f already exists (kept)" >&2
      else cp "$SCRIPT_DIR/$f.example" "$SCRIPT_DIR/$f" && echo "Created $SCRIPT_DIR/$f" || rc=1; fi
    done
    exit $rc ;;
  tgtest)
    [[ -n "$TG_BOT_TOKEN" && -n "$TG_CHAT_ID" ]] || { echo "Set TG_BOT_TOKEN and TG_CHAT_ID in scripts/pipeline.env" >&2; exit 2; }
    tg_send "🔔 autotune.sh: Telegram works!" && echo "Sent."; exit $? ;;
  reset)
    if [[ -f "$AT_DIR/.pid" ]] && kill -0 "$(cat "$AT_DIR/.pid")" 2> /dev/null; then
      echo "autotune is running (pid $(cat "$AT_DIR/.pid")): run --stop first" >&2; exit 3
    fi
    dst="$ROOT/archive-$(date +%Y%m%d-%H%M%S)"
    items=(); for d in autotune data runs logs; do [[ -n "$(ls -A "$d" 2> /dev/null)" ]] && items+=("$d"); done
    gens=(); for f in nets/gen*.nnue nets/archive; do [[ -e "$f" ]] && gens+=("$f"); done
    (( ${#items[@]} + ${#gens[@]} )) || { echo "Nothing to reset."; exit 0; }
    echo "Moves ${items[*]} ${gens[*]} to $dst and starts from zero (nets/maskfish.nnue is kept)."
    if (( ! RESET_YES )); then read -r -p "Continue? [y/N] " ans; [[ "$ans" == [yY]* ]] || exit 1; fi
    mkdir -p "$dst/nets"
    for d in "${items[@]}"; do mv "$d" "$dst/"; done
    for f in "${gens[@]}"; do mv "$f" "$dst/nets/"; done
    echo "Archived to $dst (delete it when no longer needed: rm -rf $dst)"
    exit 0 ;;
esac

mkdir -p "$AT_DIR" "$RES_DIR" "$AT_DIR/logs" nets data runs
case "$ACTION" in
  status)
    state_load
    echo "REF_MODE: $REF_MODE · stage: $(st stage ref) · level $(st level 0)/$LEVEL_CAP · failures in a row: $(st fails 0) · new nets: $(st promos 0)"
    echo "Current net: gen$(cur_gen) · vs the reference engine: $(st "ext_g$(cur_gen)" '?') Elo · best: gen$(st best_ext_gen '?') ($(st best_ext '?'))"
    echo "Option cycles: $(st opt_cycles 0)$( [[ "$(st opt_running)" == 1 ]] && echo ' (one running)') · options in use: $(best_opts)"
    if (( OPT_PLAN_OK )); then echo "Option plan: $OPT_PLAN (${#OPT_TESTS[@]} tests, ${#OPT_TC_TESTS[@]} on the clock)"
    else echo "Option plan: none ($OPT_PLAN): no option cycle"; fi
    [[ -f "$AT_DIR/.pid" ]] && kill -0 "$(cat "$AT_DIR/.pid")" 2> /dev/null && echo "Running (pid $(cat "$AT_DIR/.pid"))" || echo "Not running"
    [[ -f "$DEC_FILE" ]] && { echo; column -t -s $'\t' "$DEC_FILE" 2> /dev/null || cat "$DEC_FILE"; }
    exit 0 ;;
  stop)
    touch "$AT_DIR/.stop"
    if [[ -f "$AT_DIR/.pid" ]] && kill -0 "$(cat "$AT_DIR/.pid")" 2> /dev/null; then
      kill -TERM "$(cat "$AT_DIR/.pid")"; echo "Stop signal sent."
    else echo "No process is running."; fi
    exit 0 ;;
  report)
    state_load; r=$(write_report); echo "$r"; tg_doc "$r" "Autotune report"; exit 0 ;;
  cron)
    line="@reboot cd $(printf '%q' "$ROOT") && nohup $(printf '%q' "$SELF") --supervise >> $(printf '%q' "$AT_DIR/supervise.log") 2>&1 &"
    ( crontab -l 2> /dev/null | grep -vF "$SELF --supervise"; echo "$line" ) | crontab - && echo "Added to crontab: $line"
    exit $? ;;
  supervise)
    rm -f "$AT_DIR/.stop"
    n=0
    while :; do
      "$SELF" ${CONFIG_AT:+--config "$CONFIG_AT"}; rc=$?
      [[ -f "$AT_DIR/.stop" ]] && { echo "stopped on request"; exit 0; }
      case $rc in 0|3|4|5) exit $rc ;; esac
      n=$(( n + 1 )); d=$(( n < 6 ? 60 * n : 600 ))
      tg_send "🔁 autotune stopped (exit $rc) → restarting in ${d}s (restart $n)"
      sleep "$d"
    done ;;
esac

# ----------------------------------------------------------------------------- main run
exec 9> "$AT_DIR/.lock"
if command -v flock > /dev/null && ! flock -n 9; then echo "autotune is already running (lock $AT_DIR/.lock)" >&2; exit 3; fi
echo $$ > "$AT_DIR/.pid"
rm -f "$AT_DIR/.stop"
state_load
if [[ "$ACTION" == options ]]; then   # only one option cycle with the current net, no training
  trap 'kill_children 2> /dev/null; say "⏸ Option tests stopped. Run --options again to resume."; exit 130' INT TERM
  (( OPT_PLAN_OK )) || { echo "No option plan ($OPT_PLAN): see scripts/option-plan.env.example" >&2; exit 5; }
  g=$(cur_gen)
  (( g < 0 )) && [[ ! -f nets/maskfish.nnue ]] && sh scripts/net.sh > /dev/null 2>&1
  [[ -f "$(opt_net "$g")" ]] || { say "❌ Option tests: no net ($(opt_net "$g"); scripts/net.sh)"; exit 5; }
  log "\$ pipeline.sh --build-only"
  "$PIPELINE" --build-only >> "$AT_DIR/logs/pipeline.log" 2>&1 9>&- || { say "❌ Option tests: build failed, see autotune/logs/pipeline.log"; exit 5; }
  option_cycle || { say "❌ Option cycle failed — see $LOGF. Run --options again to resume."; exit 1; }
  rep=$(write_report); tg_doc "$rep" "Option report"
  exit 0
fi
[[ "$REF_MODE" == off ]] || ref_switch
[[ "$(st finished)" == 1 ]] && { echo "Already finished (remove ST[finished] from $STATE_FILE to continue)."; write_report; exit 0; }
[[ -n "$(st start_ts)" ]] || st_set start_ts "$(date +%s)"
case "$(st stage)" in ref|distill|self) ;; *) st_set stage ref ;; esac
if [[ "$REF_MODE" != teach && "$(st stage)" != self ]]; then
  old_stage=$(st stage); ST[stage]=self; ST[level]=0; ST[fails]=0; state_save
  say "🔁 REF_MODE=$REF_MODE: no data from the reference engine → stage $old_stage → self"
fi

STOPPING=0
on_signal() {
  trap - INT TERM; STOPPING=1
  kill_children 2> /dev/null
  say "⏸ Stopped (signal $1). Run the same command again to resume where it stopped."
  exit 130
}
on_exit() { local rc=$?; (( rc != 0 && rc != 130 && rc != 3 && rc != 4 && rc != 5 && ! STOPPING )) && \
  tg_send "❌ autotune: unexpected exit (exit $rc). Log: $LOGF"; }
trap 'on_signal INT' INT; trap 'on_signal TERM' TERM; trap on_exit EXIT

preflight || exit 5
RESUMED=""; (( $(cur_gen) >= 1 )) && RESUMED=" (resuming)"
case "$REF_MODE" in
  teach)   PLAN="reference engine games (gen0) → the reference engine scores Maskfish games until ≥ $NEAR_REF_ELO Elo vs it → self-play → engine options from $OPT_START_ELO Elo" ;;
  measure) PLAN="self-play from the start (the reference engine only measures every new net) → engine options from gen$OPT_START_GEN" ;;
  off)     PLAN="self-play only, no reference engine → engine options from gen$OPT_START_GEN" ;;
esac
say "🚀 Started$RESUMED (REF_MODE=$REF_MODE): stage $(st stage), current net $( (( $(cur_gen) >= 0 )) && echo "gen$(cur_gen)" || echo none). Plan: $PLAN → loop until no gain at level $LEVEL_CAP."
(( OPT_PLAN_OK )) || say "⚠ No option plan ($OPT_PLAN): no option cycle will run (see scripts/option-plan.env.example)"

while :; do
  check_stop && exit 0
  round || { say "❌ Loop failed — see $LOGF. Run again (or --supervise) to resume."; exit 1; }
  (( DRY_RUN )) && { echo "(dry-run: stopping after 1 round)"; exit 0; }
done
