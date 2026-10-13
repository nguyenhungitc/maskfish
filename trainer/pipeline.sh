#!/usr/bin/env bash
# =============================================================================
#  Maskfish / Jieqi NNUE — pipeline huấn luyện 1 "generation" hoàn chỉnh
#
#  build engine → kiểm tra bit-exact → (net gen0) → sinh data train + val
#  → fit scaling → prepare → train → verify net xuất ra → SPRT → promote
#
#  Mỗi bước xong bắn Telegram ✅, lỗi bắn ❌ ngay kèm đuôi log rồi dừng.
#
#  Quy ước số generation (giống trainer/README.md):
#     -g N : nets/genN.nnue là net HIỆN TẠI
#            data/genN.txt, data/valN.txt do net đó sinh ra
#            train trên data 0..N  →  runs/gen(N+1)/  →  nets/gen(N+1).nnue
#
#  Ví dụ:
#     ./pipeline.sh -g 0                    # chạy trọn gen 0 → gen1
#     ./pipeline.sh -g 1 --gen-only         # chỉ sinh data (+prepare) bằng nets/gen1.nnue
#     ./pipeline.sh -g 1 --train-only       # chỉ train/verify/SPRT trên data đã có
#     ./pipeline.sh -g 1 --gen-only --tag m2 --no-prepare   # máy phụ thứ 2
#     ./pipeline.sh --test-telegram
#
#  Cấu hình: trainer/pipeline.env (copy từ pipeline.env.example) hoặc --config.
# =============================================================================
set -uo pipefail
if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4) )); then
  echo "Cần bash >= 4.4 (đang là $BASH_VERSION)" >&2; exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
SRC_DIR="$(cd "$SCRIPT_DIR/../src" && pwd)"

# ----------------------------------------------------------------------------- CLI
usage() { sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; cat <<'EOF'
Cờ:
  -g, --gen N          Số generation của net hiện tại (bắt buộc, trừ --test-telegram)
      --gen-only       Chỉ: build, init net gen0 (nếu N=0), sinh data train+val, prepare
      --train-only     Chỉ: kiểm tra, prepare phần còn thiếu, train, verify, SPRT, promote
      --no-prepare     (với --gen-only) không prepare — dùng cho máy phụ chỉ sinh text
      --tag NAME       Tên máy/lô data (file data/genN_NAME.txt), dùng khi nhiều máy sinh data
      --config FILE    File cấu hình (mặc định: pipeline.env nếu có)
      --skip-check     Bỏ bước verify.py kiểm tra engine/trainer bit-exact ban đầu
      --skip-match     Bỏ SPRT (và promote)
      --rebuild        Build lại engine dù đã có binary
      --no-resume      Không tự --resume từ runs/.../last.pt
      --force          Sinh lại data / prepare lại dù đã có (XOÁ shard cũ của tag này)
      --dry-run        Chỉ in lệnh, không chạy, không gửi Telegram thật
      --test-telegram  Gửi 1 tin nhắn thử rồi thoát
  -h, --help
EOF
}

GEN=""; MODE="full"; TAG=""; CONFIG=""; NO_PREPARE=0; SKIP_CHECK=0; SKIP_MATCH=0
REBUILD=0; NO_RESUME=0; FORCE=0; DRY_RUN=0; TEST_TG=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -g|--gen)        GEN="${2:-}"; shift 2 ;;
    --gen-only)      MODE="gen"; shift ;;
    --train-only)    MODE="train"; shift ;;
    --no-prepare)    NO_PREPARE=1; shift ;;
    --tag)           TAG="${2:-}"; shift 2 ;;
    --config)        CONFIG="${2:-}"; shift 2 ;;
    --skip-check)    SKIP_CHECK=1; shift ;;
    --skip-match)    SKIP_MATCH=1; shift ;;
    --rebuild)       REBUILD=1; shift ;;
    --no-resume)     NO_RESUME=1; shift ;;
    --force)         FORCE=1; shift ;;
    --dry-run)       DRY_RUN=1; shift ;;
    --test-telegram) TEST_TG=1; shift ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "Cờ không hợp lệ: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# ----------------------------------------------------------------------------- config
if [[ -z "$CONFIG" && -f "$SCRIPT_DIR/pipeline.env" ]]; then CONFIG="$SCRIPT_DIR/pipeline.env"; fi
if [[ -n "$CONFIG" ]]; then
  [[ -f "$CONFIG" ]] || { echo "Không thấy file cấu hình: $CONFIG" >&2; exit 2; }
  # shellcheck disable=SC1090
  source "$CONFIG"
fi

NCPU=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then PYTHON="$SCRIPT_DIR/.venv/bin/python"; else PYTHON=python3; fi
fi
: "${TG_BOT_TOKEN:=}"; : "${TG_CHAT_ID:=}"; : "${TG_EPOCH_EVERY:=5}"; : "${TG_NAME:=$(hostname -s 2>/dev/null || hostname)}"
: "${MAKE_ARCH:=native}"; : "${NNUE_ARCH:=default}"; : "${LAYERS:=1}"
: "${GAMES:=50000}"; : "${VAL_GAMES:=1000}"; : "${NODES:=5000}"; : "${DEPTH:=0}"
: "${GEN_WORKERS:=$(( NCPU > 1 ? NCPU - 1 : 1 ))}"; : "${ADJUDICATE:=3000}"; : "${GEN_OPTS:=}"
: "${CHANCE_SCALING:=}"; : "${SEED_BASE:=}"
: "${SHARD_SIZE:=2000000}"; : "${MAX_SCORE:=10000}"; : "${SCORE_SCALE:=1.0}"; : "${PREP_WORKERS:=}"
: "${EPOCHS:=40}"; : "${EPOCH_SIZE:=10000000}"; : "${BATCH_SIZE:=16384}"; : "${LR:=8.75e-4}"
: "${GAMMA:=0.992}"; : "${LAMBDA:=1.0}"; : "${SCALING:=361}"; : "${TRAIN_FROM:=0}"
: "${CHECKPOINT_EVERY:=1}"; : "${DEVICE:=}"; : "${REQUIRE_GPU:=1}"; : "${INIT_FROM_CURRENT:=1}"
: "${TRAIN_ARGS:=}"; : "${EXTRA_TRAIN:=}"
: "${MATCH_NODES:=20000}"; : "${MATCH_MOVETIME:=0}"; : "${MATCH_WORKERS:=$(( NCPU > 1 ? NCPU - 1 : 1 ))}"
: "${SPRT=0 5}"; : "${MATCH_EPOCHS:=last}"; : "${MATCH_OPTS:=}"; : "${AUTO_PROMOTE:=1}"

# ----------------------------------------------------------------------------- telegram
html_escape() { local s="$1"; s="${s//&/&amp;}"; s="${s//</&lt;}"; s="${s//>/&gt;}"; printf '%s' "$s"; }

tg_send() {   # $1 = text (đã escape HTML nếu cần)
  local text="$1"
  (( ${#text} > 4000 )) && text="${text:0:3990}…"
  if (( DRY_RUN )); then echo "[telegram] $text"; return 0; fi
  [[ -z "$TG_BOT_TOKEN" || -z "$TG_CHAT_ID" ]] && return 0
  local resp
  resp=$(curl -sS --max-time 20 --retry 3 --retry-delay 3 \
          --data-urlencode "chat_id=${TG_CHAT_ID}" \
          --data-urlencode "text=${text}" \
          --data-urlencode "parse_mode=HTML" \
          --data-urlencode "disable_web_page_preview=true" \
          "https://api.telegram.org/bot${TG_BOT_TOKEN}/sendMessage" 2>&1)
  if [[ "$resp" == *"can't parse entities"* ]]; then   # HTML lỗi → gửi lại dạng chữ thường
    text=$(printf '%s' "$text" | sed -e 's/<[^>]*>//g' -e 's/&lt;/</g; s/&gt;/>/g; s/&amp;/\&/g')
    resp=$(curl -sS --max-time 20 --retry 3 --retry-delay 3 \
            --data-urlencode "chat_id=${TG_CHAT_ID}" --data-urlencode "text=${text}" \
            "https://api.telegram.org/bot${TG_BOT_TOKEN}/sendMessage" 2>&1)
  fi
  if [[ "$resp" != *'"ok":true'* ]]; then
    echo "⚠ Telegram gửi lỗi: ${resp:0:300}" >&2
    return 1
  fi
}

if (( TEST_TG )); then
  [[ -z "$TG_BOT_TOKEN" || -z "$TG_CHAT_ID" ]] && { echo "Chưa đặt TG_BOT_TOKEN / TG_CHAT_ID" >&2; exit 2; }
  tg_send "🔔 <b>[$(html_escape "$TG_NAME")]</b> Maskfish pipeline: Telegram hoạt động!" && echo "Đã gửi." || exit 1
  exit 0
fi

[[ "$GEN" =~ ^[0-9]+$ ]] || { echo "Thiếu hoặc sai -g/--gen N" >&2; usage >&2; exit 2; }
NEXT=$(( GEN + 1 ))

# ----------------------------------------------------------------------------- tên file / kiến trúc
case "$NNUE_ARCH" in
  default)     BIN_BASE="maskfish";         MAKE_FLAGS=() ;;
  att)         BIN_BASE="maskfish-att";     MAKE_FLAGS=(nnue=att) ;;
  threats)     BIN_BASE="maskfish-thr";     MAKE_FLAGS=(threats=yes) ;;
  att-threats) BIN_BASE="maskfish-att-thr"; MAKE_FLAGS=(threats=yes nnue=att) ;;
  *) echo "NNUE_ARCH không hợp lệ: $NNUE_ARCH (default|att|threats|att-threats)" >&2; exit 2 ;;
esac
case "$LAYERS" in
  1) ARCH_FULL="$NNUE_ARCH" ;;
  2) ARCH_FULL="${NNUE_ARCH}-l2"; BIN_BASE="${BIN_BASE}-l2"; MAKE_FLAGS+=(layers=2) ;;
  *) echo "LAYERS phải là 1 hoặc 2" >&2; exit 2 ;;
esac
ENGINE="${ENGINE:-$SRC_DIR/$BIN_BASE}"
# mỗi kiến trúc là một "dòng" net riêng; default giữ đúng bố cục của README
if [[ "$ARCH_FULL" == "default" ]]; then LP=""; else LP="${ARCH_FULL}-"; fi
TAGSFX=""; [[ -n "$TAG" ]] && TAGSFX="_${TAG}"
[[ -z "$TAG" || "$TAG" =~ ^[A-Za-z0-9.]+$ ]] || { echo "--tag chỉ dùng chữ, số, dấu chấm" >&2; exit 2; }

NET_CUR="nets/${LP}gen${GEN}.nnue"
NET_NEXT="nets/${LP}gen${NEXT}.nnue"
PREP_DIR="data/prep-${NNUE_ARCH}"          # data đã prepare theo bộ feature
RUN_DIR="runs/${LP}gen${NEXT}"
LOG_DIR="logs/${LP}gen${GEN}${TAGSFX}"
GEN_TXT="data/${LP}gen${GEN}${TAGSFX}.txt"
VAL_TXT="data/${LP}val${GEN}${TAGSFX}.txt"
mkdir -p nets data runs "$LOG_DIR" "$PREP_DIR"

# seed khác nhau cho mỗi generation / tag (đặt SEED_BASE để ghi đè)
if [[ -z "$SEED_BASE" ]]; then
  th=0; [[ -n "$TAG" ]] && th=$(( $(printf '%s' "$TAG" | cksum | cut -d' ' -f1) % 9973 ))
  SEED_BASE=$(( (GEN + 1) * 100000 + th * 10 ))
fi
SEED_TRAIN=$(( SEED_BASE + 1 )); SEED_VAL=$(( SEED_BASE + 2 ))

# ----------------------------------------------------------------------------- tiện ích
NAME_TAG="<b>[$(html_escape "$TG_NAME")]</b> $(html_escape "${LP}gen${GEN}${TAGSFX}")"
T_START=$SECONDS; CUR_STEP=""; CUR_LOG=""; FAILED_SENT=0; DONE=0
declare -a SUMMARY=()

fmt_dur() { local s=$1; printf '%dh%02dm%02ds' $((s/3600)) $((s%3600/60)) $((s%60)); }
fmt_num() { printf "%'d" "$1" 2>/dev/null || printf '%d' "$1"; }
log()  { echo "[$(date '+%F %T')] $*"; }
words() { local -n _arr=$1; read -r -a _arr <<< "${2:-}"; }   # tách chuỗi cấu hình thành mảng

kill_tree() { local p; for p in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$p"; done; kill -TERM "$1" 2>/dev/null; }
kill_children() { local p; for p in $(pgrep -P $$ 2>/dev/null); do kill_tree "$p"; done; }

notify_fail() {   # $1 = lý do
  (( FAILED_SENT )) && return
  FAILED_SENT=1
  local tail_txt=""
  if [[ -n "$CUR_LOG" && -f "$CUR_LOG" ]]; then
    tail_txt=$(tail -n 25 "$CUR_LOG" | cut -c1-200)
  fi
  local msg="❌ ${NAME_TAG} · THẤT BẠI ở bước <b>$(html_escape "${CUR_STEP:-?}")</b>
$(html_escape "$1")
Tổng thời gian: $(fmt_dur $((SECONDS - T_START)))"
  [[ -n "$tail_txt" ]] && msg+="
<pre>$(html_escape "$tail_txt")</pre>"
  [[ -n "$CUR_LOG" ]] && msg+="
Log: <code>$(html_escape "$SCRIPT_DIR/$CUR_LOG")</code>"
  log "THẤT BẠI [$CUR_STEP]: $1"
  tg_send "$msg"
}

on_signal() {
  trap - INT TERM
  notify_fail "Bị dừng bởi tín hiệu $1 (Ctrl+C / kill)"
  kill_children 2>/dev/null
  exit 130
}
on_exit() {
  local rc=$?
  if (( rc != 0 && ! DONE )); then notify_fail "Script thoát bất thường (exit $rc)"; fi
}
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap on_exit EXIT

# Chạy 1 lệnh, ghi log; trả về exit code của lệnh
run() {   # run LOG cmd...
  local logf="$1"; shift
  log "\$ $*" | tee -a "$logf"
  (( DRY_RUN )) && return 0
  # chạy nền + wait: để Ctrl+C / kill được xử lý ngay (bash hoãn trap khi chờ lệnh foreground)
  ( "$@" 2>&1 | tee -a "$logf"; exit "${PIPESTATUS[0]}" ) &
  wait $!
}

# Chạy 1 bước: step "Tên" hàm  (hàm in ra stdout dòng "INFO: ..." để đưa vào tin nhắn thành công)
STEP_INFO=""
step() {
  local title="$1" fn="$2"
  CUR_STEP="$title"; CUR_LOG="$LOG_DIR/$(date +%m%d-%H%M%S)-${fn}.log"; STEP_INFO=""
  : > "$CUR_LOG"
  log "════ BẮT ĐẦU: $title  (log: $CUR_LOG)"
  local t0=$SECONDS
  "$fn"; local rc=$?
  local dur; dur=$(fmt_dur $((SECONDS - t0)))
  if (( rc != 0 )); then
    notify_fail "Exit code $rc sau $dur"
    exit 1
  fi
  log "════ XONG: $title ($dur)"
  SUMMARY+=("✅ ${title} (${dur})")
  local msg="✅ ${NAME_TAG} · <b>$(html_escape "$title")</b> xong (${dur})"
  [[ -n "$STEP_INFO" ]] && msg+="
${STEP_INFO}"
  tg_send "$msg"
  CUR_LOG=""
}
info() { STEP_INFO+="${STEP_INFO:+
}• $*"; }   # thêm 1 dòng (đã escape) vào tin nhắn thành công

fail() { echo "LỖI: $*" | tee -a "$CUR_LOG" >&2; return 1; }

count_lines() { local n=0 f; for f in "$@"; do [[ -f "$f" ]] && n=$(( n + $(wc -l < "$f") )); done; echo "$n"; }

# ----------------------------------------------------------------------------- các bước
s_preflight() {
  command -v "$PYTHON" >/dev/null || fail "Không tìm thấy Python: $PYTHON" || return 1
  command -v curl >/dev/null || echo "Cảnh báo: không có curl → không gửi được Telegram" | tee -a "$CUR_LOG"
  local mods="numpy"
  [[ "$MODE" != "gen" || "$GEN" == 0 ]] && mods+=" zstandard torch"
  run "$CUR_LOG" "$PYTHON" -c "
import importlib, sys
for m in '$mods'.split():
    importlib.import_module(m)
print('python', sys.version.split()[0], 'OK:', '$mods')
" || fail "Thiếu gói Python — chạy: $PYTHON -m pip install -r requirements.txt" || return 1
  if [[ "$MODE" != "gen" ]]; then
    local cuda
    cuda=$("$PYTHON" -c "import torch;print(torch.cuda.is_available(), torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')" 2>&1 | tail -n 1)
    echo "torch: $cuda" | tee -a "$CUR_LOG"
    info "torch: $(html_escape "$cuda")"
    if [[ "$cuda" != True* && "$REQUIRE_GPU" == 1 && "${DEVICE:-}" != cpu ]]; then
      fail "Không có CUDA. Cài torch bản CUDA, hoặc đặt REQUIRE_GPU=0 / DEVICE=cpu để train bằng CPU (rất chậm)" || return 1
    fi
  fi
  if [[ "$MODE" != train ]] && (( GEN > 0 )) && [[ ! -f "$NET_CUR" ]]; then
    fail "Chưa có net hiện tại $NET_CUR (copy net vào thư mục nets/ trước khi sinh data)" || return 1
  fi
  local avail; avail=$(df -Pk . | awk 'NR==2{printf "%.0f", $4/1048576}')
  info "Đĩa trống: ${avail} GB · CPU: ${NCPU} · mode: ${MODE}"
  (( avail < 5 )) && echo "Cảnh báo: dưới 5 GB đĩa trống" | tee -a "$CUR_LOG"
  return 0
}

s_build() {
  if [[ -x "$ENGINE" && $REBUILD == 0 ]]; then
    echo "Đã có engine: $ENGINE (dùng --rebuild để build lại)" | tee -a "$CUR_LOG"
    info "Dùng engine có sẵn: $(html_escape "$ENGINE")"
    return 0
  fi
  command -v make >/dev/null || fail "Không có make" || return 1
  local keep=""
  # make clean xoá src/maskfish — giữ lại binary default khi build kiến trúc khác
  if [[ "$BIN_BASE" != maskfish && -x "$SRC_DIR/maskfish" ]]; then
    keep="$SRC_DIR/maskfish.keep.$$"; cp -p "$SRC_DIR/maskfish" "$keep"
  fi
  run "$CUR_LOG" make -C "$SRC_DIR" clean || return 1
  run "$CUR_LOG" make -C "$SRC_DIR" -j"$NCPU" build ARCH="$MAKE_ARCH" "${MAKE_FLAGS[@]}" || return 1
  if [[ "$BIN_BASE" != maskfish ]] && (( ! DRY_RUN )); then
    mv -f "$SRC_DIR/maskfish" "$ENGINE" || return 1
    [[ -n "$keep" ]] && mv -f "$keep" "$SRC_DIR/maskfish"
  fi
  (( DRY_RUN )) || [[ -x "$ENGINE" ]] || fail "Build xong nhưng không thấy $ENGINE" || return 1
  info "Engine: $(html_escape "$ENGINE") (ARCH=$(html_escape "$MAKE_ARCH") ${MAKE_FLAGS[*]})"
}

s_check() {
  run "$CUR_LOG" "$PYTHON" -u verify.py --engine "$ENGINE" --arch "$ARCH_FULL" \
    || fail "Engine và trainer KHÔNG khớp bit-exact → không được train (xem README mục 11)" || return 1
  info "$(html_escape "$(grep -h 'bit-exact' "$CUR_LOG" | tail -1)")"
}

s_init() {
  if [[ -f "$NET_CUR" ]]; then
    echo "Đã có $NET_CUR" | tee -a "$CUR_LOG"; info "Đã có $(html_escape "$NET_CUR"), bỏ qua"; return 0
  fi
  run "$CUR_LOG" "$PYTHON" -u init_net.py "$NET_CUR" --arch "$ARCH_FULL" || return 1
  info "Tạo net material-only: $(html_escape "$NET_CUR")"
}

# gom các file .partK còn sót của lần chạy bị ngắt (đã chứa game hoàn chỉnh)
salvage_parts() {
  local out="$1" parts=("$1".part*)
  [[ -e "${parts[0]}" ]] || return 0
  local dst; dst="${out}.salvage-$(date +%Y%m%d-%H%M%S)"
  cat "${parts[@]}" > "$dst" && rm -f "${parts[@]}"
  echo "Gom $(count_lines "$dst") vị trí từ .part cũ → $dst" | tee -a "$CUR_LOG"
  info "Cứu $(fmt_num "$(count_lines "$dst")") vị trí từ lần chạy bị ngắt → $(html_escape "$dst")"
}

gen_common() {   # $1 = file out, $2 = số ván, $3 = seed
  local out="$1" games="$2" seed="$3"
  if [[ -s "$out" && $FORCE == 0 ]]; then
    echo "Đã có $out ($(count_lines "$out") dòng), bỏ qua (dùng --force để sinh lại)" | tee -a "$CUR_LOG"
    info "Đã có $(html_escape "$out") ($(fmt_num "$(count_lines "$out")") vị trí), bỏ qua"
    return 0
  fi
  [[ -f "$NET_CUR" ]] || (( DRY_RUN )) || fail "Không có net hiện tại $NET_CUR" || return 1
  salvage_parts "$out"
  local cmd=("$PYTHON" -u gen_data.py --engine "$ENGINE" --net "$NET_CUR"
             --games "$games" --nodes "$NODES" --workers "$GEN_WORKERS"
             --adjudicate "$ADJUDICATE" --seed "$seed" --out "$out")
  (( DEPTH > 0 )) && cmd+=(--depth "$DEPTH")
  local opts; words opts "$GEN_OPTS"
  [[ -n "$CHANCE_SCALING" ]] && opts+=("ChanceScaling=$CHANCE_SCALING")
  (( ${#opts[@]} )) && cmd+=(--opt "${opts[@]}")
  run "$CUR_LOG" "${cmd[@]}" || return 1
  (( DRY_RUN )) && return 0
  local n; n=$(count_lines "$out")
  (( n > 0 )) || fail "gen_data chạy xong nhưng $out rỗng" || return 1
  info "$(fmt_num "$n") vị trí từ $(fmt_num "$games") ván → $(html_escape "$out") ($(du -h "$out" | cut -f1))"
}

s_gendata() { gen_common "$GEN_TXT" "$GAMES" "$SEED_TRAIN"; }
s_genval()  { gen_common "$VAL_TXT" "$VAL_GAMES" "$SEED_VAL"; }

s_fit() {   # chỉ để tham khảo, không làm hỏng pipeline
  [[ -s "$GEN_TXT" ]] || (( DRY_RUN )) || { info "Không có $GEN_TXT để fit"; return 0; }
  if run "$CUR_LOG" "$PYTHON" -u fit_scaling.py "$GEN_TXT"; then
    local line; line=$(grep -h 'fitted scale' "$CUR_LOG" | tail -1)
    [[ -n "$line" ]] && info "$(html_escape "$line")"
    info "Đang dùng SCALING=${SCALING}${CHANCE_SCALING:+, ChanceScaling=$CHANCE_SCALING} (sửa trong pipeline.env nếu muốn)"
  else
    info "⚠ fit_scaling lỗi (bỏ qua, không ảnh hưởng train)"
  fi
  return 0
}

# prepare 1 file text → 1 prefix; $3=1 thì ép làm lại
prepare_one() {   # $1 = text chính, $2 = prefix out, $3 = force
  local txt="$1" prefix="$2" force="$3"
  if [[ -f "${prefix}.arch" && "$force" == 0 ]]; then
    echo "Đã prepare: $prefix" | tee -a "$CUR_LOG"; return 0
  fi
  if [[ "$force" == 1 ]] && (( ! DRY_RUN )); then
    rm -f "${prefix}".arch "${prefix}"-[0-9][0-9][0-9][0-9][0-9].*.npy
  fi
  local inputs=("$txt") s
  for s in "${txt}".salvage-*; do [[ -s "$s" ]] && inputs+=("$s"); done
  local cmd=("$PYTHON" -u prepare_data.py "${inputs[@]}" --out "$prefix" --arch "$NNUE_ARCH"
             --shard-size "$SHARD_SIZE" --max-score "$MAX_SCORE" --score-scale "$SCORE_SCALE")
  [[ -n "$PREP_WORKERS" ]] && cmd+=(--workers "$PREP_WORKERS")
  run "$CUR_LOG" "${cmd[@]}" || return 1
  (( DRY_RUN )) && return 0
  [[ -f "${prefix}.arch" ]] || fail "prepare xong nhưng không có ${prefix}.arch" || return 1
  local res; res=$(grep -h 'shard(s) ->' "$CUR_LOG" | tail -1)
  info "$(html_escape "$(basename "$prefix")"): $(html_escape "${res%% -> *}")"
}

# prepare mọi file text của generation k còn thiếu (gồm cả của máy khác đã copy về)
s_prepare() {
  local k kind txt base tag prefix f did=0
  for (( k = TRAIN_FROM; k <= GEN; k++ )); do
    for kind in gen val; do
      (( k < GEN )) && [[ "$kind" == val ]] && continue      # chỉ cần val của gen hiện tại
      for txt in "data/${LP}${kind}${k}.txt" "data/${LP}${kind}${k}"_*.txt; do
        [[ -s "$txt" ]] || continue
        base=$(basename "$txt" .txt); tag="${base#"${LP}${kind}${k}"}"   # "" hoặc "_TAG"
        [[ "$kind" == gen ]] && prefix="$PREP_DIR/${LP}train${k}${tag}" || prefix="$PREP_DIR/${LP}val${k}${tag}"
        f=0; (( FORCE )) && (( k == GEN )) && [[ "$tag" == "$TAGSFX" ]] && f=1
        prepare_one "$txt" "$prefix" "$f" || return 1
        did=1
      done
    done
  done
  if (( DRY_RUN && ! did )); then
    prepare_one "$GEN_TXT" "$PREP_DIR/${LP}train${GEN}${TAGSFX}" 0
    prepare_one "$VAL_TXT" "$PREP_DIR/${LP}val${GEN}${TAGSFX}" 0
    return 0
  fi
  (( did )) || fail "Không thấy file text nào để prepare (data/${LP}gen${GEN}*.txt)" || return 1
  return 0
}

collect_prefixes() {   # $1 = train|val → in ra danh sách prefix đã prepare
  local kind="$1" k a from=$TRAIN_FROM
  [[ "$kind" == val ]] && from=$GEN
  for (( k = from; k <= GEN; k++ )); do
    for a in "$PREP_DIR/${LP}${kind}${k}.arch" "$PREP_DIR/${LP}${kind}${k}"_*.arch; do
      [[ -f "$a" ]] && echo "${a%.arch}"
    done
  done
}

last_epoch() {   # epoch lớn nhất đã có .nnue trong RUN_DIR
  local f e max=0
  for f in "$RUN_DIR"/epoch_*.nnue; do
    [[ -f "$f" ]] || continue
    e=${f##*/epoch_}; e=${e%.nnue}
    [[ "$e" =~ ^[0-9]+$ ]] && (( e > max )) && max=$e
  done
  echo "$max"
}

s_train() {
  local done_ep; done_ep=$(last_epoch)
  if (( done_ep >= EPOCHS )); then
    echo "$RUN_DIR đã có epoch_${done_ep}.nnue ≥ EPOCHS=$EPOCHS, bỏ qua train" | tee -a "$CUR_LOG"
    info "Đã train xong từ trước (epoch ${done_ep}), bỏ qua"
    return 0
  fi
  local train=() val=() extra
  mapfile -t train < <(collect_prefixes train)
  mapfile -t val < <(collect_prefixes val)
  words extra "$EXTRA_TRAIN"; train+=("${extra[@]}")
  (( ${#train[@]} )) || (( DRY_RUN )) || fail "Không có data đã prepare trong $PREP_DIR (chạy --gen-only trước)" || return 1
  local cmd=("$PYTHON" -u train.py "${train[@]}")
  if (( ${#val[@]} )); then cmd+=(--val "${val[@]}"); else
    echo "Cảnh báo: không có validation set" | tee -a "$CUR_LOG"; info "⚠ Không có validation set"; fi
  cmd+=(--out "$RUN_DIR" --epochs "$EPOCHS" --epoch-size "$EPOCH_SIZE" --batch-size "$BATCH_SIZE"
        --lr "$LR" --gamma "$GAMMA" --lambda "$LAMBDA" --scaling "$SCALING"
        --layers "$LAYERS" --checkpoint-every "$CHECKPOINT_EVERY")
  [[ -n "$DEVICE" ]] && cmd+=(--device "$DEVICE")
  if [[ -f "$RUN_DIR/last.pt" && $NO_RESUME == 0 ]]; then
    cmd+=(--resume "$RUN_DIR/last.pt"); info "Tiếp tục từ $(html_escape "$RUN_DIR/last.pt") (epoch ${done_ep})"
  elif (( GEN > 0 )) && [[ "$INIT_FROM_CURRENT" == 1 ]]; then
    [[ -f "$NET_CUR" ]] || (( DRY_RUN )) || fail "Không có $NET_CUR cho --init-net" || return 1
    cmd+=(--init-net "$NET_CUR")
  fi
  local targs; words targs "$TRAIN_ARGS"; cmd+=("${targs[@]}")
  mkdir -p "$RUN_DIR"
  log "\$ ${cmd[*]}" | tee -a "$CUR_LOG"
  (( DRY_RUN )) && return 0

  # chạy nền + theo dõi log: bắn Telegram mỗi TG_EPOCH_EVERY epoch, dừng ngay nếu loss = nan
  "${cmd[@]}" >> "$CUR_LOG" 2>&1 &
  local pid=$! nanflag="$LOG_DIR/.nan.$$"
  rm -f "$nanflag"
  tail -n +1 -f --pid="$pid" "$CUR_LOG" 2>/dev/null | while IFS= read -r line; do
    printf '%s\n' "$line"
    if [[ "$line" =~ loss\ [-+]?(nan|inf) ]]; then
      : > "$nanflag"; kill_tree "$pid"
    elif [[ "$line" =~ ^epoch\ ([0-9]+):\ train\ loss ]]; then
      local ep=${BASH_REMATCH[1]}
      if (( TG_EPOCH_EVERY > 0 && ep % TG_EPOCH_EVERY == 0 && ep < EPOCHS )); then
        tg_send "📈 ${NAME_TAG} · train $(html_escape "$line") (/${EPOCHS})" >/dev/null 2>&1
      fi
    fi
  done &
  local mon=$!
  wait "$pid"; local rc=$?
  wait "$mon" 2>/dev/null
  if [[ -f "$nanflag" ]]; then rm -f "$nanflag"; fail "Loss = NaN/Inf → đã dừng train. Thử giảm LR hoặc MAX_SCORE"; return 1; fi
  (( rc == 0 )) || return "$rc"
  [[ -f "$RUN_DIR/epoch_${EPOCHS}.nnue" ]] || fail "Train xong nhưng thiếu $RUN_DIR/epoch_${EPOCHS}.nnue" || return 1
  local lastl best
  lastl=$(grep -hE '^epoch [0-9]+: train loss' "$CUR_LOG" | tail -1)
  best=$(grep -hE '^epoch [0-9]+: .*val loss' "$CUR_LOG" \
         | awk '{v=$0; sub(/.*val loss /,"",v); sub(/,.*/,"",v); e=$2; sub(":","",e)
                 if (b=="" || v+0<b+0) {b=v; be=e}} END{if(b!="") print "epoch " be " (val " b ")"}')
  info "$(html_escape "$lastl")"
  [[ -n "$best" ]] && info "Val loss tốt nhất: $(html_escape "$best")"
  info "Output: $(html_escape "$RUN_DIR")/epoch_${EPOCHS}.nnue"
}

candidates() {
  local e
  for e in $MATCH_EPOCHS; do [[ "$e" == last ]] && echo "$EPOCHS" || echo "$e"; done
}

s_verify() {
  local e net ck ok=0
  for e in $(candidates); do
    net="$RUN_DIR/epoch_${e}.nnue"; ck="$RUN_DIR/epoch_${e}.pt"
    [[ -f "$net" ]] || (( DRY_RUN )) || fail "Không có $net" || return 1
    local cmd=("$PYTHON" -u verify.py --engine "$ENGINE" --net "$net")
    [[ -f "$ck" ]] && cmd+=(--checkpoint "$ck")
    run "$CUR_LOG" "${cmd[@]}" || fail "epoch_${e}.nnue KHÔNG bit-exact với engine" || return 1
    info "epoch_${e}: $(html_escape "$(grep -h 'bit-exact\|largest' "$CUR_LOG" | tail -2 | tr '\n' ' ')")"
    ok=1
  done
  (( ok )) || (( DRY_RUN )) || fail "MATCH_EPOCHS rỗng" || return 1
}

MATCH_RESULT=""
s_match() {
  [[ -f "$NET_CUR" ]] || (( DRY_RUN )) || fail "Không có net tham chiếu $NET_CUR" || return 1
  local e net res mopts sprt
  words mopts "$MATCH_OPTS"; [[ -n "$CHANCE_SCALING" ]] && mopts+=("ChanceScaling=$CHANCE_SCALING")
  words sprt "$SPRT"
  for e in $(candidates); do
    net="$RUN_DIR/epoch_${e}.nnue"
    local cmd=("$PYTHON" -u match.py --engine "$ENGINE" --net-a "$net" --net-b "$NET_CUR"
               --workers "$MATCH_WORKERS")
    if (( MATCH_MOVETIME > 0 )); then cmd+=(--movetime "$MATCH_MOVETIME"); else cmd+=(--nodes "$MATCH_NODES"); fi
    (( ${#sprt[@]} == 2 )) && cmd+=(--sprt "${sprt[@]}")
    (( ${#mopts[@]} )) && cmd+=(--opt-a "${mopts[@]}" --opt-b "${mopts[@]}")
    local mark; mark=$(wc -l < "$CUR_LOG")
    run "$CUR_LOG" "${cmd[@]}" || return 1
    (( DRY_RUN )) && return 0
    res=$(tail -n +"$((mark + 1))" "$CUR_LOG" | grep -hE '^(H1|H0|no decision)' | tail -1)
    local stat; stat=$(tail -n +"$((mark + 1))" "$CUR_LOG" | grep -h '^pairs ' | tail -1)
    grep -q '^interrupted' <(tail -n +"$((mark + 1))" "$CUR_LOG") && { fail "match bị ngắt"; return 1; }
    info "epoch_${e} vs $(html_escape "$NET_CUR"): <b>$(html_escape "${res:-?}")</b>"
    [[ -n "$stat" ]] && info "<code>$(html_escape "$stat")</code>"
    if [[ "$res" == H1* ]]; then MATCH_RESULT="$net"; break; fi
  done
  return 0
}

s_promote() {
  if [[ -z "$MATCH_RESULT" ]]; then
    info "⚠ Không epoch nào thắng SPRT → giữ $(html_escape "$NET_CUR"). Thử thêm data / epoch khác / chỉnh LR, LAMBDA"
    return 0
  fi
  if [[ "$AUTO_PROMOTE" != 1 ]]; then
    info "AUTO_PROMOTE=0: tự copy bằng tay: cp $(html_escape "$MATCH_RESULT") $(html_escape "$NET_NEXT")"; return 0
  fi
  if [[ -e "$NET_NEXT" ]]; then
    info "⚠ $(html_escape "$NET_NEXT") đã tồn tại, KHÔNG ghi đè. Net thắng: $(html_escape "$MATCH_RESULT")"; return 0
  fi
  run "$CUR_LOG" cp "$MATCH_RESULT" "$NET_NEXT" || return 1
  info "🏆 Net mới: <b>$(html_escape "$NET_NEXT")</b> ← $(html_escape "$MATCH_RESULT")"
  info "Gen tiếp: ./pipeline.sh -g ${NEXT}"
}

# ----------------------------------------------------------------------------- khoá chống chạy trùng
LOCK="$LOG_DIR/.lock"
if command -v flock >/dev/null && (( ! DRY_RUN )); then
  exec 9>"$LOCK"
  flock -n 9 || { echo "Đang có pipeline khác chạy cho ${LP}gen${GEN}${TAGSFX} (khoá $LOCK)" >&2; DONE=1; exit 3; }
fi

# ----------------------------------------------------------------------------- chạy
case "$MODE" in
  full)  MODE_TXT="đầy đủ (gen data → train → SPRT)" ;;
  gen)   MODE_TXT="CHỈ SINH DATA" ;;
  train) MODE_TXT="CHỈ TRAIN" ;;
esac
tg_send "🚀 ${NAME_TAG} · Bắt đầu pipeline: <b>${MODE_TXT}</b>
net: $(html_escape "$NET_CUR") · arch: $(html_escape "$ARCH_FULL")$( [[ $MODE != train ]] && printf '\ndata: %s ván + %s val, %s nodes, %s workers, seed %s' "$GAMES" "$VAL_GAMES" "$NODES" "$GEN_WORKERS" "$SEED_BASE")$( [[ $MODE != gen ]] && printf '\ntrain: %s epoch × %s, lr %s, λ %s, scaling %s → %s' "$EPOCHS" "$EPOCH_SIZE" "$LR" "$LAMBDA" "$SCALING" "$(html_escape "$RUN_DIR")")"

step "Kiểm tra môi trường" s_preflight
step "Build engine" s_build

if [[ "$MODE" != gen ]] && (( ! SKIP_CHECK )); then
  step "Kiểm tra engine/trainer bit-exact" s_check
fi

if [[ "$MODE" != train ]]; then
  (( GEN == 0 )) && step "Tạo net gen0 (material)" s_init
  step "Sinh data train" s_gendata
  step "Sinh data validation" s_genval
  step "Fit scaling (tham khảo)" s_fit
  if ! { [[ "$MODE" == gen ]] && (( NO_PREPARE )); }; then
    step "Prepare data" s_prepare
  fi
else
  # chỉ train: prepare những file text còn thiếu (vd. data copy từ máy khác về)
  missing=0
  for (( k = TRAIN_FROM; k <= GEN; k++ )); do
    for txt in data/${LP}gen${k}.txt data/${LP}gen${k}_*.txt data/${LP}val${GEN}.txt data/${LP}val${GEN}_*.txt; do
      [[ -s "$txt" ]] || continue
      b=$(basename "$txt" .txt); b="${b/#${LP}gen/${LP}train}"
      [[ -f "$PREP_DIR/$b.arch" ]] || missing=1
    done
  done
  (( missing )) && step "Prepare data còn thiếu" s_prepare
fi

if [[ "$MODE" != gen ]]; then
  step "Train ${RUN_DIR}" s_train
  step "Verify net đã train" s_verify
  if (( ! SKIP_MATCH )); then
    step "SPRT match" s_match
    step "Promote" s_promote
  fi
fi

DONE=1
final="🎉 ${NAME_TAG} · <b>HOÀN TẤT</b> ${MODE_TXT} sau $(fmt_dur $((SECONDS - T_START)))"
for l in "${SUMMARY[@]}"; do final+="
$(html_escape "$l")"; done
[[ "$MODE" == gen ]] && final+="
Tiếp theo: ./pipeline.sh -g ${GEN} --train-only"
tg_send "$final"
log "HOÀN TẤT ($(fmt_dur $((SECONDS - T_START))))"
