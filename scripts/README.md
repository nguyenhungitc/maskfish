# Scripts

| Script | What it does |
| --- | --- |
| `net.sh` | Fetches the default network `maskfish.nnue` from `refs/assets/net` into `nets/` ([README](../README.md#network)) |
| `push_net.sh` | Publishes a network as the default one on `refs/assets/net` |
| `package_binary.sh` | Packs a release archive: binary, network, README, AUTHORS, licence |
| `pipeline.sh` | Runs **one training generation**: build → check → generate data → train → verify → SPRT → promote |
| `autotune.sh` | Runs **generation after generation** with `pipeline.sh` in stages (reference engine games → reference engine scores → self-play), measures each new network against the reference engine, tests the engine options once the net is strong, and stops when the Elo stops rising |
| `datatool.py` | Record files: count, score scale (`fit-k`), rescale, cat, stats |
| `hw_probe.py` | Hardware probe used by `pipeline.sh` (CPUs, RAM, disk, GPU, a training batch size) |
| `pipeline.env.example` | Configuration template shared by both training scripts |
| `autotune.env.example` | Settings of the stages and option tests |
| `option-plan.env.example` | Option plan: which engine options the option cycles test |

The rest of this file is about training: `pipeline.sh` and `autotune.sh`.

The **reference engine** ("ref-engine" in file names and messages) is another
Jieqi engine, stronger than the current network, used as a teacher and a
yardstick: any UCI engine that speaks Maskfish's FEN and move notation (a
third-party engine, an older Maskfish build, ...). The scripts never depend on
which engine it is: switching to another one, or running without one
(`REF_MODE`, section 4.2), only changes the configuration (section 4.8).

Both scripts report every step to Telegram (when configured) and can be
stopped and restarted at any time. They run from the repository root (they
`cd` there themselves, so any working directory works).

## Contents

1. [Requirements](#1-requirements)
2. [Configuration](#2-configuration)
3. [pipeline.sh — one generation](#3-pipelinesh--one-generation)
4. [autotune.sh — the full loop](#4-autotunesh--the-full-loop)
5. [Files and directories](#5-files-and-directories)
6. [Troubleshooting](#6-troubleshooting)

## 1. Requirements

- Linux, bash ≥ 4.4, CMake 3.20+, a C++20 compiler, Python 3 with numpy.
- For training: PyTorch with CUDA. With `REQUIRE_GPU=0`, a machine without
  CUDA trains on the CPU (`train.py --device cpu` on the physical cores),
  slowly. A virtual environment in `.venv/` is used when it exists:
  ```sh
  python3 -m venv .venv && .venv/bin/pip install -r train/requirements.txt
  ```
- Optional: a reference engine (with its network) in `jieqi/`
  (`jieqi/ref-engine`, `jieqi/ref-engine*.nnue`), or anywhere with
  `REF_ENGINE` / `REF_NET`.
- Optional: a Telegram bot for reports (`TG_BOT_TOKEN`, `TG_CHAT_ID`).

## 2. Configuration

```sh
scripts/autotune.sh --init-config   # scripts/pipeline.env, autotune.env, option-plan.env from the examples
```

- `scripts/pipeline.env` — hardware, data, training, SPRT, the reference
  engine, Telegram ([example](pipeline.env.example)). Read by both scripts.
- `scripts/autotune.env` — stages, measuring, option tests
  ([example](autotune.env.example)).
- `scripts/option-plan.env` — which engine options are tested
  ([example](option-plan.env.example)).

The three files are local (in `.gitignore`). Every value has a default; empty
means automatic (hardware values are chosen by `hw_probe.py`).

## 3. pipeline.sh — one generation

```sh
scripts/pipeline.sh -g 0                       # gen0 → gen1
scripts/pipeline.sh -g 3 --source distill      # gen3 → gen4, the reference engine scores the positions
scripts/pipeline.sh -g 3 --gen-only --tag m2   # only data, e.g. on a second machine (data/gen3_m2.bin)
scripts/pipeline.sh -g 3 --train-only          # train on the data there is, then SPRT
scripts/pipeline.sh -g 3 --dry-run             # print the commands
scripts/pipeline.sh --test-telegram
```

Steps of `-g N` (each one is skipped when its result is already there, so a
run that stopped resumes where it stopped; `--force` redoes them):

1. **build** `build/pipeline/maskfish`, `maskfish_difftest`, the referee
   library and `jqnz_tool` (`BUILD_ARCH`, `BUILD_OPTS`). The build is incremental: it only
   recompiles what changed (`--rebuild` builds from scratch).
2. **gen0 net**: for `-g 0`, `nets/gen0.nnue` from `INIT_NET`, else
   `nets/maskfish.nnue` (scripts/net.sh); none is fine for `--source ref`.
3. **check**: perft 3, bench with the current net, the network evaluator
   (`tests/hnn_smoke.py`), and 100 games of `maskfish_difftest` against the
   referee.
4. **data** `data/genN.bin` (`GAMES`) then `data/valN.bin` (`VAL_GAMES`), by
   `--source`:
   - `self` — `nets/genN.nnue` plays (`NODES` per move, `selfplay`) and its
     scores are the labels;
   - `distill` — it plays, the reference engine scores every position
     (`tools/label.py`, `REF_LABEL_GO`);
   - `ref` — the reference engine plays both sides (`REF_GEN_NODES`) and
     scores (`tools/match.py --scores`, then `tools/games2rec.py`).

   Scores of the reference engine are in its own units: their scale K is
   fitted once (`data/REF_NAME.scale`, `datatool.py fit-k`) and they are
   rescaled by `MASKFISH_K / K`; `MASKFISH_K`, Maskfish's own scale, is fitted
   the same way on the first data Maskfish scored (`data/maskfish.scale`)
   unless it is set. `FILE.src` names the engine that scored `FILE`.
5. **train** on `data/gen{TRAIN_FROM..N}*.bin` (`TRAIN_WINDOW`, `EXTRA_TRAIN`),
   validated on the `valN*.bin` files, with `train/train.py`, starting from the
   current net (`INIT_FROM_CURRENT`): `runs/genN/best.pt` when the pipeline
   trained it, else the net file `nets/genN.nnue` itself, so the first
   generation continues from the published net with its width, king buckets
   and head. Output: `runs/gen(N+1)/best.hnn`, `last.hnn` (and their `.pt`
   states).
6. **verify**: the engine loads every candidate (`MATCH_EPOCHS`) and runs
   bench; identical candidates are tested once.
7. **SPRT** of every candidate against `nets/genN.nnue` with `tools/match.py`
   (`MATCH_NODES`, `MATCH_MOVETIME` or `MATCH_TC`; `SPRT`, `SPRT_MODEL`,
   `MATCH_MAX_PAIRS`). A candidate wins with H1, or without a decision when
   the lower bound of its Elo is above 0.
8. **promote** the winner to `nets/gen(N+1).nnue` (`AUTO_PROMOTE`).

Exit codes: 0 promoted (or the requested part is done), 3 no candidate won,
1 error, 2 bad usage.

## 4. autotune.sh — the full loop

### 4.1. Usage

```sh
scripts/autotune.sh                 # run / resume
nohup scripts/autotune.sh --supervise >> autotune/supervise.log 2>&1 &   # restarts after errors
scripts/autotune.sh --status        # stage, nets, Elo, option decisions
scripts/autotune.sh --stop          # stop safely (run again to resume)
scripts/autotune.sh --report        # autotune/REPORT.md (and to Telegram)
scripts/autotune.sh --options       # one option cycle now, with the current net, no training
scripts/autotune.sh --install-cron  # start --supervise after a reboot
scripts/autotune.sh --dry-run       # what the next round would run
scripts/autotune.sh --reset         # archive autotune/ data/ runs/ logs/ nets/gen* to archive-DATE/
scripts/autotune.sh --test-telegram
```

### 4.2. Stages

`REF_MODE` in autotune.env chooses how the reference engine is used:

| REF_MODE | Stages | The reference engine |
| --- | --- | --- |
| `teach` (default) | ref → distill → self | plays gen0's games, scores the distill data, measures every net |
| `measure` | self | only measures every new net |
| `off` | self | not used (no `REF_ENGINE` needed) |

- **ref** (gen0): the reference engine plays and scores `GEN0_GAMES` games;
  the starting net (`INIT_NET` or `nets/maskfish.nnue`) is trained further on
  them (`GEN0_EPOCHS`) and the new net must beat it by SPRT; after
  `DISTILL_PATIENCE` failures the loop goes on with distill. Without any net,
  the first net is trained from scratch and promoted without SPRT.
- **distill**: the current net plays (`DISTILL_NODES`), the reference engine
  scores. Ends when a new net is within `NEAR_REF_ELO` of the reference
  engine, or after `DISTILL_PATIENCE` failures in a row at level
  `DISTILL_LEVEL_CAP`.
- **self**: the net plays and scores its own games (`SELF_NODES`).

### 4.3. One round

1. Measure the current net against the reference engine (once a net:
   `EXT_PAIRS` pairs, `EXT_LIMIT`), unless `REF_MODE=off`.
2. Run an option cycle when one is due (4.4).
3. `pipeline.sh -g N --source STAGE --gen-only`, then `--train-only`, with
   `autotune/run.env`: pipeline.env plus the values of this round (level,
   attempt).
4. A new net: measure it. In stage distill, a net that won its SPRT but is
   clearly weaker against the reference engine than the previous one (by more
   than the combined errors) is taken back (`EXT_VETO`, to `nets/archive/`).
5. No new net: a failed attempt. The next attempt of the same generation adds
   a new batch of data (`data/genN_aK.bin`) and changes the training in turn
   (always from the current net): as before, 1.5× the epochs, a halved
   learning rate with more weight on the game results.

Levels: after `PATIENCE` failures in a row (`DISTILL_PATIENCE` in stage
distill) the level rises by one, up to `LEVEL_CAP`: level L plays
`GAMES × (1 + L/2)` games at `NODES × (1 + L/4)` nodes. Training uses the data
of the last `TRAIN_WINDOW` generations.

### 4.4. Engine option tests

Once the net is strong enough (`OPT_START_ELO` against the reference engine,
or stage self, and not before gen`OPT_START_GEN`), the entries of the option
plan are tested, then again every `OPT_EVERY` new nets and once more before
stopping. Every entry is a setting of UCI options (see
option-plan.env.example), tested by SPRT of the engine with it against the
engine with the options in use, same net:

- `OPT_TESTS` at `OPT_NODES` nodes (`OPT_SPRT`, `OPT_SPRT_MODEL`,
  `OPT_MAX_PAIRS`);
- `OPT_TC_TESTS` on the clock, `OPT_TC` then `OPT_LTC` after a win
  (`OPT_TC_SPRT`).

| Result | Action |
| --- | --- |
| H1 | in use: set in every later generation (`GEN_OPTS`) and match (`MATCH_OPTS`), and in the report |
| H0 | rejected |
| no decision | left as it is, tested again in up to `OPT_RETRY` later cycles |

The undecided entries that leaned positive are also tested together
(`OPT_GROUP`). The options in use are only UCI settings: make them the
defaults in `src/search.cpp` by hand when convenient (the report lists them).
Every decision is in `autotune/decisions.tsv`.

`scripts/autotune.sh --options` runs one option cycle at once, without
training: with the current net (`nets/genN.nnue`, or `nets/maskfish.nnue`
before gen1), the same decisions, state and Telegram messages, then the report.
It needs no reference engine. Run it again to resume a stopped cycle or to
test the undecided entries again.

### 4.5. When does it stop?

In stage self at level `LEVEL_CAP`, after `PATIENCE` failures in a row and an
option cycle without gain: "finished", with the final net and
`autotune/REPORT.md`. `MAX_HOURS` stops earlier (resumable). Publish the final
net with `scripts/push_net.sh nets/genN.nnue`.

### 4.6. Recovery

Every step is saved in `autotune/state.sh`; finished matches are reused from
`autotune/results/` and `runs/`. After a crash, a reboot or `--stop`, run the
same command again. `--supervise` restarts after errors (with growing delays)
and stops on `--stop`, on "finished" and on configuration errors. The pipeline
output is in `autotune/logs/pipeline.log` and `logs/pipeline-genN*.log`.

### 4.7. Telegram messages

With `TG_BOT_TOKEN` and `TG_CHAT_ID` in pipeline.env, every message starts
with the machine name (`TG_NAME`). `pipeline.sh` sends ✅ after every step,
❌ with the tail of the log on an error (then stops), and ❎ when no candidate
beat the current net; `autotune.sh` reports the rounds, measurements, option
decisions and the final report. `--test-telegram` sends a test message.

### 4.8. Another reference engine, or none

Set `REF_ENGINE`, `REF_NET`, `REF_OPTS` and `REF_NAME` in pipeline.env. The
Elo values are per reference engine: with another `REF_NAME` every net is
measured again, and its score scale is fitted again (`data/REF_NAME.scale`).
Use `TRAIN_MIN_GEN` to leave out data scored by the old one. Without a
reference engine, set `REF_MODE=off` in autotune.env.

## 5. Files and directories

Everything runs from the repository root, in these directories (none is
tracked by git):

| Directory | Contents |
| --- | --- |
| `build/pipeline/` | the engine, the difftest, the referee library and the network packer the scripts build |
| `nets/` | `maskfish.nnue` (scripts/net.sh), `genN.nnue` (the net of every generation), `archive/` (vetoed nets) |
| `data/` | `genN*.bin` training records, `valN*.bin` validation records (112 bytes a record), `*.scale` score scales |
| `runs/` | `genN/` trainer output, `sprt-genN-*` SPRT matches, `ref-*` reference engine games |
| `logs/` | `pipeline-genN.log`, build, difftest and match logs |
| `autotune/` | state, results, decisions, report of autotune.sh |
| `jieqi/` | the reference engine and its network (by default) |

## 6. Troubleshooting

- `CUDA is not available`: install PyTorch with CUDA (`train/requirements.txt`),
  or set `REQUIRE_GPU=0` to train on the CPU.
- `no current net nets/genN.nnue`: run `scripts/net.sh` (or set `INIT_NET`),
  or start with `--source ref`.
- `reference engine … not found`: put it in `jieqi/`, set `REF_ENGINE`, or
  `REF_MODE=off`.
- `native ABI version mismatch`: an old referee library from another build;
  `pipeline.sh` rebuilds `build/pipeline/` itself, other build directories
  need `cmake --build`.
- Engine timeouts during labelling: raise `ENGINE_TIMEOUT` or lower
  `REF_LABEL_GO`.
- Disk: every generation keeps its data; delete old `data/genN*.bin` outside
  `TRAIN_WINDOW`.
