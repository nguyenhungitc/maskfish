# Training automation scripts

Two Bash scripts drive the trainer of [`trainer/README.md`](../README.md)
without manual steps:

| Script | What it does |
| --- | --- |
| `pipeline.sh` | Runs **one generation**: build → bit-exact check → generate data → prepare → train → verify → SPRT → promote |
| `autotune.sh` | Runs **generation after generation** with `pipeline.sh`, tests every engine option and trainer flag, measures each new network against AB-JChess and jieqi, and stops when the Elo stops rising *and* both engines are beaten convincingly |
| `hw_probe.py` | Hardware probe used by `pipeline.sh` (CPU, RAM, disk, GPU, largest batch that fits in VRAM) |
| `pipeline.env.example` | Configuration template shared by both scripts |

Both scripts report every step to Telegram and can be stopped and restarted
at any time. Run them from the `trainer/` directory (they `cd` there
themselves, so any working directory works).

## Contents

1. [Requirements](#1-requirements)
2. [Configuration](#2-configuration)
3. [pipeline.sh — one generation](#3-pipelinesh--one-generation)
4. [autotune.sh — the full loop](#4-autotunesh--the-full-loop)
5. [Files and directories](#5-files-and-directories)
6. [Applying the patches](#6-applying-the-patches)
7. [Troubleshooting](#7-troubleshooting)

## 1. Requirements

- bash ≥ 4.4, `make`, a C++17 compiler, `git`, `curl` (Telegram), `flock`
  (optional, prevents two runs at once).
- Python with `numpy`, `zstandard`, `torch` (`pip install -r requirements.txt`);
  a CUDA build of torch to train on the GPU. A `trainer/.venv` is used
  automatically if it exists.
- For `autotune.sh`, the two reference engines. Default layout (next to
  `src/` and `trainer/`):

  ```
  maskfish/
    jieqi/
      abjchess                    AB-JChess binary (Linux build)
      abjchess-20261010.nnue      AB-JChess network (the newest abjchess*.nnue is used)
      jieqi                       jieqi engine binary (https://github.com/nguyenhungitc/jieqi)
    src/  trainer/ ...
  ```

  Other locations are set with `AB_ENGINE`, `AB_NET`, `JQ_ENGINE` (or
  `JIEQI_DIR`) in `autotune.env`.

## 2. Configuration

```
cp scripts/pipeline.env.example scripts/pipeline.env     # shared settings, holds the Telegram token
scripts/autotune.sh --init-config                         # optional: writes scripts/autotune.env
```

`pipeline.env` holds the Telegram bot, the build architecture, the data
generation, training and SPRT parameters. Hardware values left empty
(workers, shard size, batch size, checkpoint interval, ...) are chosen for the
machine by `hw_probe.py`. Every variable is documented in the file.

`autotune.env` only holds the settings of the loop (reference engines, stop
criteria, which options to test). Without it, the defaults listed in
[section 4.6](#46-settings) are used.

**Telegram.** Create a bot with @BotFather (`/newbot`), send it a message,
then read your chat id from
`https://api.telegram.org/bot<TOKEN>/getUpdates` (`"chat":{"id":...}`). Put
both in `pipeline.env` (`TG_BOT_TOKEN`, `TG_CHAT_ID`) and test:

```
scripts/pipeline.sh --test-telegram
scripts/autotune.sh --test-telegram
```

Both files are ignored by git (`pipeline.env` contains the token).

## 3. pipeline.sh — one generation

```
current net ──► generate data ──► prepare ──► train ──► verify ──► SPRT match
     ▲                                                                  │
     └─────────────── the new net becomes the current one if it wins ◄──┘
```

`-g N` names the **current** network `nets/genN.nnue`. The script generates
`data/genN.txt` and `data/valN.txt` with it, trains on the data of generations
`TRAIN_FROM..N` into `runs/gen(N+1)/`, and promotes the winner to
`nets/gen(N+1).nnue`.

| Step | Details |
| --- | --- |
| Environment check | Python packages, hardware probe, automatic worker / batch / checkpoint choices |
| Build engine | `make build ARCH=$MAKE_ARCH` (skipped if the binary exists, `--rebuild` forces it) |
| Bit-exact check | `verify.py`: the trainer must compute exactly what the engine computes, otherwise it stops |
| Create gen0 net | Only for `-g 0`: material-only network (`init_net.py`) |
| Generate data | `gen_data.py`, training set and validation set with different seeds |
| Fit scaling | `fit_scaling.py`, informational only |
| Prepare | `prepare_data.py` for every text file not prepared yet (including files copied from other machines) |
| Train | `train.py`, resumes from `runs/gen(N+1)/last.pt` automatically; stops at once if the loss becomes NaN |
| Verify | `verify.py` on the candidate epochs (`MATCH_EPOCHS`) |
| SPRT | `match.py` candidate vs current net; the first epoch accepting H1 wins |
| Promote | Copies the winner to `nets/gen(N+1).nnue` (never overwrites) |

Examples:

```
scripts/pipeline.sh -g 0                    # from scratch: gen0 (material) → gen1
scripts/pipeline.sh -g 3                    # nets/gen3.nnue → data gen3 → runs/gen4 → nets/gen4.nnue
scripts/pipeline.sh -g 3 --gen-only         # only generate + prepare data
scripts/pipeline.sh -g 3 --train-only       # only train / verify / SPRT / promote on existing data
scripts/pipeline.sh -g 3 --gen-only --tag m2 --no-prepare   # helper machine: text data only
scripts/pipeline.sh -g 3 --dry-run          # print the commands, run nothing
```

| Option | Meaning |
| --- | --- |
| `-g N` | Generation of the current net (required) |
| `--gen-only` / `--train-only` | Only the data part / only the training part |
| `--no-prepare` | With `--gen-only`: write text data only (helper machines) |
| `--tag NAME` | Data batch name (`data/genN_NAME.txt`), gives different seeds; for several machines |
| `--config FILE` | Another config file instead of `scripts/pipeline.env` |
| `--skip-check` / `--skip-match` | Skip the bit-exact check / the SPRT and promotion |
| `--rebuild` | Rebuild the engine |
| `--no-resume` | Do not resume training from `last.pt` |
| `--force` | Regenerate / re-prepare data that already exists |
| `--dry-run` | Print the commands only |
| `--test-telegram` | Send a test message |

**Interrupted?** Run the same command again: existing data is kept
(finished games of an interrupted generation are recovered from the `.part`
files), training resumes from `last.pt`. On any error a ❌ Telegram message
with the end of the log is sent at once; logs are in `logs/genN/`.

**Several machines.** Run `--gen-only --tag NAME --no-prepare` on every
helper machine, copy their `data/genN_NAME.txt` to the training machine, then
run `--train-only` there: the missing files are prepared first.

## 4. autotune.sh — the full loop

`autotune.sh` is a single self-contained file (its Python helper is embedded
and extracted to `autotune/optpatch.py` at run time). It never edits your
working tree: option changes are made in a separate git worktree and
delivered as patches.

### 4.1. Usage

```
scripts/autotune.sh                    # run / resume
nohup scripts/autotune.sh --supervise >> autotune/supervise.log 2>&1 &   # recommended for long runs
scripts/autotune.sh --install-cron     # also restart after a reboot (@reboot crontab entry)
scripts/autotune.sh --status           # current state and every decision so far
scripts/autotune.sh --stop             # stop safely; run again to resume
scripts/autotune.sh --report           # rewrite autotune/REPORT.md and send it to Telegram
scripts/autotune.sh --init-config      # write scripts/autotune.env (template)
scripts/autotune.sh --test-telegram
```

Before the first run, `nets/gen0.nnue` must exist (`scripts/pipeline.sh -g 0`
creates it). The loop starts from the highest `nets/genN.nnue`.

### 4.2. One round

1. **Measure** the current net against AB-JChess and jieqi (once per
   generation): `EXT_PAIRS` game pairs per engine at `EXT_MOVETIME` ms per
   move, always the same openings (`EXT_SEED`), so that generations are
   compared fairly.
2. **Engine option tests** (once, from generation `OPT_AT_GEN`, see 4.3).
3. **Generate data** with the current net: `pipeline.sh -g N --gen-only`.
   Later attempts of the same generation add a new batch (`--tag aK`).
4. **Trainer flag tests** (once, see 4.4).
5. **Train → verify → SPRT → promote**: `pipeline.sh -g N --train-only`,
   with the accepted engine options in the data generation and in the SPRT,
   and the accepted trainer flags in `TRAIN_ARGS`.
6. If a new net was promoted, it is measured against the two engines (step 1).
   If no epoch won the SPRT, the next attempt changes strategy in turn:
   more data and continue training → more data, train from scratch with 1.5×
   the epochs → more data, half the learning rate and λ 0.9 (from the 5th
   attempt also 1.5× the nodes).

### 4.3. Engine option tests

The options are read from `src/engine.cpp` (`options.add(...)`), so new
options are picked up automatically.

- **Boolean options**: the opposite of the default is tested.
- **Numeric options**: the values of `OPT_INT_TESTS` (e.g.
  `ChanceRiskAversion:10,20,-10`).
- **Option groups**: `OPT_GROUP_TESTS` (e.g. all `RevealQuiet*` set to 0).
- **Not tested** (`OPT_EXCLUDE`): `Ponder`, `UCI_ShowWDL`, `RawScore`,
  `SharedHistory` (only matters with several threads), `StopEarly` (time
  management only), `JieqiMaterial` (needs data prepared with
  `--jieqi-material`).

Each value plays an SPRT `OPT_SPRT` (default `[0, 5]`) against the current
settings, same net, `OPT_NODES` nodes per move:

| Result | Action |
| --- | --- |
| H1, and the Elo against AB-JChess + jieqi does not drop (`OPT_EXT_CONFIRM`) | **Kept**: used from now on, and a patch makes it the default in `src/engine.cpp` |
| H0 or no decision after `OPT_MAX_PAIRS` pairs | **Removed**: a patch deletes the option from the code. Numeric options are only removed if listed in `OPT_INT_REMOVABLE`; groups are never removed |

Removing an option means: the `options.add` call and its comment go, the
member variable becomes `static constexpr` with the default value, and the
dead branches are folded (`if (opt) …`, `opt ? a : b`, `opt && …`, …).
Every patch is **built and checked with `bench`**: the patched engine must
search exactly the same number of nodes as the original engine with the
accepted options set. If not (or if folding is impossible), it retries
without folding; if it still differs, nothing is committed and Telegram says
the option has to be edited by hand.

### 4.4. Trainer flag tests

Every variant of `TRAIN_VARIANTS` is trained on the same prepared data, from
the same net, for `ABL_EPOCHS` epochs, like a baseline without the flag; the
variant's net then plays an SPRT `ABL_SPRT` against the baseline's net.

| Default variants | Tested values | Removable |
| --- | --- | --- |
| `--no-factorize` | on | yes |
| `--factorize-kb` | on | yes |
| `--no-l1-fact` | on | yes |
| `--stack-balance` | 8, 16 | yes |
| `--output-lr-scale` | 0.05, 0.2 | no |
| `--lambda` | 0.9, 0.75 | no |

A winner becomes the default in `train.py` (patch) and is used for the
following generations; a removable loser is deleted from `train.py` (the
`add_argument` goes, `args.x` becomes its default value, constant `if`
blocks are folded).

### 4.5. When does it stop?

After every round:

- **Improvement** = the average Elo against the two engines rose by more
  than `EXT_EPS` compared with the best net so far. The best net is copied to
  `nets/best.nnue`.
- **Plateau** = `PATIENCE` rounds in a row without improvement (a failed SPRT
  also counts as a round without improvement).
- **Convincing win** = against **both** engines, Elo − 95 % error ≥
  `WIN_MIN_ELO`.

The run **finishes** only when the plateau is reached **and** the best net
wins convincingly against both engines. A plateau without a convincing win
raises the scale level instead (data × (1 + level/2), nodes × (1 + level/4),
up to `LEVEL_CAP`) and the loop continues. `MAX_HOURS` puts an optional time
limit on a run.

When it finishes, Telegram receives the final results, `REPORT.md` and the
combined patch.

### 4.6. Settings

Set in `scripts/autotune.env` (generation, training and SPRT settings come from
`pipeline.env`).

| Variable | Default | Meaning |
| --- | --- | --- |
| `JIEQI_DIR` | `../jieqi` | Directory of the reference engines |
| `AB_ENGINE`, `AB_NET`, `AB_OPTS` | `$JIEQI_DIR/abjchess`, newest `abjchess*.nnue`, `ChasingRule=github` | AB-JChess |
| `JQ_ENGINE`, `JQ_OPTS` | `$JIEQI_DIR/jieqi`, empty | jieqi |
| `EXT_MOVETIME` | 200 | ms per move against the reference engines |
| `EXT_PAIRS` | 200 | Game pairs per engine and measurement |
| `EXT_HASH` / `EXT_SEED` | 64 / 20261015 | Hash (MB) / fixed opening seed |
| `WIN_MIN_ELO` | 50 | Convincing win threshold (Elo − error) |
| `EXT_EPS` | 5 | Minimum average Elo gain that counts as an improvement |
| `PATIENCE` | 3 | Rounds without improvement = maximum reached |
| `LEVEL_CAP` | 8 | Maximum scale-up level |
| `MAX_HOURS` | 0 | Time limit (0 = none) |
| `OPT_AT_GEN` | 1 | First generation for the option tests (earlier on a plateau) |
| `OPT_NODES` / `OPT_SPRT` / `OPT_MAX_PAIRS` | `MATCH_NODES` / `0 5` / 3000 | Option SPRT |
| `OPT_EXT_CONFIRM` / `OPT_EXT_PAIRS` / `OPT_EXT_TOL` | 1 / 100 / 0 | Confirmation of a winning option against the reference engines |
| `OPT_EXCLUDE` | see 4.3 | Options never tested |
| `OPT_INT_TESTS` / `OPT_INT_REMOVABLE` | `ChanceRiskAversion:10,20,-10` / `ChanceRiskAversion` | Numeric options |
| `OPT_GROUP_TESTS` | `RevealQuietOff:RevealQuietBase=0+RevealQuietSafety=0+RevealQuietHighValue=0` | Option groups |
| `OPT_ONLY` | empty | Only test these options |
| `TRAIN_ABL` | 1 | Test the trainer flags |
| `ABL_EPOCHS` / `ABL_EPOCH_SIZE` / `ABL_SPRT` / `ABL_MAX_PAIRS` | `EPOCHS/2` / `EPOCH_SIZE` / `0 5` / 3000 | Trainer flag tests |
| `TRAIN_VARIANTS` | see 4.4 | `id\|flag\|values\|removable` (bash array) |
| `BENCH_ARGS` | `16 1 13` | `bench` used to verify the patches |
| `RETRY_MAX` | 5 | Retries of a failed `pipeline.sh` call |
| `KEEP_ATTEMPT_PT` | 0 | Keep the checkpoints of failed attempts |

### 4.7. Recovery

All progress is written to `autotune/state.sh` after every step; finished
matches are kept in `autotune/results/` and reused. Running the same command
again (or the restart done by `--supervise`, or by cron after a reboot)
continues exactly where it stopped: no finished match, data batch or patch is
redone, `pipeline.sh` resumes the data and the training itself. An interrupted
match is restarted from its beginning.

`--supervise` restarts after an error (after 1, 2, … 5 minutes, then every 10 minutes) but not
when a reference engine or a file is missing (exit code 5).

To redo a part, edit `autotune/state.sh` while nothing runs, e.g. delete the
`ST[opt_done]` and `ST[opt_<Name>]` lines to test the options again with a
newer net.

### 4.8. Telegram messages

| Icon | When |
| --- | --- |
| 🚀 | Start / resume |
| 🎯 ✅ | Measurement against the reference engines, Elo per engine, best so far |
| 🔬 | Engine option tests: start, every decision (✅ kept / ❎ removed, with the patch attached), end |
| 🧪 | Trainer flag tests |
| ▶ ✅ 📈 | Steps of `pipeline.sh` (its own messages, including training progress) |
| 🏆 / ↩ | New net promoted / no epoch won the SPRT |
| ⛰ | Plateau without a convincing win → scale-up |
| ⏸ ❌ 🔁 | Stopped / error / automatic restart |
| 🏁 | Finished: final results, report and combined patch |

## 5. Files and directories

All in `trainer/`:

| Path | Content |
| --- | --- |
| `nets/genN.nnue` | Promoted networks; `nets/best.nnue` = best against the reference engines |
| `data/genN*.txt`, `data/valN*.txt`, `data/prep-<arch>/` | Text and prepared data |
| `runs/gen(N+1)/` | Training output of the current attempt (`runs/gen(N+1).tryK` = earlier attempts) |
| `runs/abl-gN/` | Trainer flag test runs |
| `logs/genN/` | `pipeline.sh` logs, one file per step |
| `autotune/state.sh` | Saved state of `autotune.sh` |
| `autotune/autotune.log`, `autotune/logs/` | Log of the loop; pipeline, build and patch logs |
| `autotune/results/` | Log of every match (option SPRTs, measurements) |
| `autotune/decisions.tsv` | Every test and decision |
| `autotune/REPORT.md` | Summary: Elo per generation, decisions, patches |
| `autotune/patches/` | `000N-*.patch` (one per decision) and `all-options.patch` (all combined) |
| `autotune/work/` | Git worktree of the branch `autotune/options` where the patches are made |
| `autotune/run.env` | Configuration given to `pipeline.sh` for the current round |

## 6. Applying the patches

The patches are made on the commit that was checked out when the first patch
was created (branch `autotune/options`, worktree `trainer/autotune/work`).
Each commit message records the `bench` node count it was verified with.

```
cd maskfish
git am trainer/autotune/patches/0*.patch          # one commit per decision, in order
# or everything at once, uncommitted:
git apply trainer/autotune/patches/all-options.patch
# or merge the branch:
git merge autotune/options
```

Rebuild the engine afterwards and check with `bench` (or `verify.py`).

## 7. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `Cannot start: AB-JChess not found` / `jieqi engine not found` | Set `AB_ENGINE` / `JQ_ENGINE`, or put the binaries in `jieqi/`. They must be Linux builds (not `.exe`); a missing executable bit is fixed automatically |
| `engine ... does not answer UCI` | Wrong binary or missing library: try `printf 'uci\nisready\nquit\n' \| jieqi/abjchess` |
| `No CUDA` in the environment check | Install a CUDA build of torch, or set `DEVICE=cpu` / `REQUIRE_GPU=0` in `pipeline.env` (very slow) |
| `Engine and trainer are NOT bit-exact` | The engine and the trainer disagree: use the same commit for both, rebuild; see `trainer/README.md` section 11 |
| An option says "edit by hand" | Its patch did not reproduce the `bench` node count; see `autotune/logs/patch.log`. The option is still used in the runs |
| `uncommitted changes` warning | `src/` or `trainer/*.py` differ from `HEAD`: patches are made on `HEAD`, the `bench` check may then fail. Commit first |
| `autotune is already running` | Another run holds `autotune/.lock`; `--status` shows its pid |
| Telegram messages missing | `scripts/pipeline.sh --test-telegram`; check `TG_BOT_TOKEN` / `TG_CHAT_ID` and that `curl` is installed |
| The run never finishes | The goal is not reached yet: check `--status` (plateau, level, Elo per generation). Lower `WIN_MIN_ELO`, raise `EXT_PAIRS` (smaller error), or set `MAX_HOURS` |
