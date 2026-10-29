# Training automation scripts

Two Bash scripts drive the trainer of [`trainer/README.md`](../README.md)
without manual steps:

| Script | What it does |
| --- | --- |
| `pipeline.sh` | Runs **one generation**: build → bit-exact check → generate data → prepare → train → verify → SPRT → promote |
| `autotune.sh` | Runs **generation after generation** with `pipeline.sh` in stages (reference engine games → reference engine scores → self-play), measures each new network against the reference engine, tests the engine options once the net is strong, and stops when the Elo stops rising |
| `arch.sh` | Network architecture of both scripts, from `pipeline.env` (`NNUE_ARCH`, `META`, ...) |
| `hw_probe.py` | Hardware probe used by `pipeline.sh` (CPU, RAM, disk, GPU, largest batch that fits in VRAM) |
| `pipeline.env.example` | Configuration template shared by both scripts |
| `autotune.env.example` | Settings of the stages and option tests |
| `optpatch.py` | Source rewriting used for the option patches |

The **reference engine** ("ref-engine" in file names and messages) is
another Jieqi engine, stronger than the current network, used as a teacher
and a yardstick: any engine that speaks Maskfish's protocol (a third-party
engine, an older Maskfish build, ...), see
[trainer/README.md, section 6](../README.md#6-using-other-engines).

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
- The reference engine, a **Linux** binary, and its network, by default in
  `jieqi/` at the root of the repository (the directory is in `.gitignore`).
  `autotune.sh` always needs it; `pipeline.sh` needs it for the data sources
  `ref` and `distill` (the defaults), not for `--source self`:

  ```
  maskfish/
    jieqi/
      ref-engine
      ref-engine.nnue
    src/  trainer/ ...
  ```

  Other locations are set with `REF_ENGINE`, `REF_NET` (or `JIEQI_DIR`) in
  `pipeline.env`.

## 2. Configuration

```
cp scripts/pipeline.env.example scripts/pipeline.env     # shared settings, holds the Telegram token
scripts/autotune.sh --init-config                        # optional: writes scripts/autotune.env
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
`data/genN.txt` and `data/valN.txt`, trains on the data of generations
`TRAIN_FROM..N` into `runs/gen(N+1)/`, and promotes the winner to
`nets/gen(N+1).nnue`. Where the data comes from is `--source` (or `SOURCE`
in `pipeline.env`):

| Source | Data | Default for |
| --- | --- | --- |
| `ref` | The reference engine plays both sides and scores the positions; Maskfish is only the referee for the board | `-g 0` |
| `distill` | The current net plays, the reference engine scores every position kept | `-g 1` and later |
| `self` | The current net plays and scores its own games (no reference engine needed) | — |

The scores of the reference engine are converted to Maskfish units (section
4.2).

| Step | Details |
| --- | --- |
| Environment check | Python packages, hardware probe, automatic worker / batch / checkpoint choices |
| Build engine | `make build ARCH=$MAKE_ARCH` (skipped if the binary exists, `--rebuild` forces it) |
| Bit-exact check | `verify.py`: the trainer must compute exactly what the engine computes, otherwise it stops |
| Create gen0 net | Only for `-g 0`: material-only network (`init_net.py`) |
| Generate data | `gen_data.py` (according to `--source`), training set and validation set with different seeds |
| Fit scaling | `fit_scaling.py`, informational only |
| Prepare | `prepare_data.py` for every text file not prepared yet (including files copied from other machines) |
| Train | `train.py`, resumes from `runs/gen(N+1)/last.pt` automatically; stops at once if the loss becomes NaN |
| Verify | `verify.py` on the candidate epochs (`MATCH_EPOCHS`) |
| SPRT | `match.py` candidate vs current net; the first epoch accepting H1 wins |
| Promote | Copies the winner to `nets/gen(N+1).nnue` (never overwrites) |

Examples:

```
scripts/pipeline.sh -g 0                    # from scratch: reference engine data → gen1, SPRT against gen0 (material)
scripts/pipeline.sh -g 0 --source self      # from scratch without a reference engine: gen0 (material) self-play → gen1
scripts/pipeline.sh -g 3                    # nets/gen3.nnue → data gen3 → runs/gen4 → nets/gen4.nnue
scripts/pipeline.sh -g 3 --gen-only         # only generate + prepare data
scripts/pipeline.sh -g 3 --train-only       # only train / verify / SPRT / promote on existing data
scripts/pipeline.sh -g 3 --gen-only --tag m2 --no-prepare   # helper machine: text data only
scripts/pipeline.sh -g 3 --dry-run          # print the commands, run nothing
```

| Option | Meaning |
| --- | --- |
| `-g N` | Generation of the current net (required) |
| `--source S` | `ref`, `distill` or `self` (default: `SOURCE` of `pipeline.env`, otherwise `ref` for `-g 0` and `distill` after) |
| `--gen-only` / `--train-only` | Only the data part / only the training part |
| `--no-prepare` | With `--gen-only`: write text data only (helper machines) |
| `--tag NAME` | Data batch name (`data/genN_NAME.txt`), gives different seeds; for several machines |
| `--config FILE` | Another config file instead of `scripts/pipeline.env` |
| `--skip-check` / `--skip-match` | Skip the bit-exact check / the SPRT and promotion |
| `--rebuild` | Rebuild the engine |
| `--no-resume` | Do not resume training from `last.pt` |
| `--force` | Regenerate / re-prepare data that already exists (deletes this tag's old shards) |
| `--dry-run` | Print the commands only |
| `--test-telegram` | Send a test message |

**Architecture** (`pipeline.env`, shared with `autotune.sh` by `scripts/arch.sh`):
`NNUE_ARCH` (feature set), `LAYERS`, `META`, `BLEND`, `CTX`, `PSQT`, `QUANT`,
`STACKS` select the network ([trainer README, section 7](../README.md#7-network-architectures-and-training-options)); each
architecture has its own engine binary (`src/maskfish-meta-ctx`, ...), its own
line of nets (`nets/default-meta-ctx-genN.nnue`, ...) and, for `META` /
`BLEND` / `STACKS`, its own prepared data (`data/prep-default-meta`, ...). With `INIT_NET` set, generation 0
of a new line is that net converted ([section 7.9](../README.md#79-converting-a-network-to-another-architecture)) instead of a
material-only net: with `META=1` / `CTX=1` it evaluates exactly as the original,
so the new line starts at the strength of the current one:

```
# pipeline.env: META=1 CTX=1 INIT_NET="nets/gen5.nnue"
scripts/pipeline.sh -g 0       # nets/default-meta-ctx-gen0.nnue = gen5 converted → data → gen1
```

`PSQT=0` (and `LAYERS=2` from a `LAYERS=1` net) cannot be converted: train a
first net with `train.py --init-net` (trainer README, section 7.9) and copy it to
`nets/<arch>-gen0.nnue`.

**Interrupted?** Run the same command again: existing data is kept
(finished games of an interrupted generation are recovered from the `.part`
files), training resumes from `last.pt`. On any error a ❌ Telegram message
with the end of the log is sent at once; logs are in `logs/genN/`.

**Several machines.** Run `--gen-only --tag NAME --no-prepare` on every
helper machine, copy their `data/genN_NAME.txt` to the training machine, then
run `--train-only` there: the missing files are prepared first.

## 4. autotune.sh — the full loop

`autotune.sh` trains Maskfish in stages, using the reference engine first as
a teacher and then only as a yardstick, and tests the engine options once the net is
strong enough. It never edits your working tree: option changes are made in a
separate git worktree and delivered as patches.

### 4.1. Usage

```
scripts/autotune.sh --reset            # archive ALL old results (autotune, nets, data, runs, logs)
                                       # to trainer/archive-DATE/ and start from zero (--yes: no question)
scripts/autotune.sh                    # run / resume
nohup scripts/autotune.sh --supervise >> autotune/supervise.log 2>&1 &   # recommended for long runs
scripts/autotune.sh --install-cron     # also restart after a reboot (@reboot crontab entry)
scripts/autotune.sh --status           # stage, current net, Elo vs ref-engine, every decision
scripts/autotune.sh --stop             # stop safely; run again to resume
scripts/autotune.sh --report           # rewrite autotune/REPORT.md and send it to Telegram
scripts/autotune.sh --init-config      # write scripts/autotune.env from autotune.env.example
scripts/autotune.sh --dry-run          # print what the next round would run
scripts/autotune.sh --test-telegram
scripts/autotune.sh --config FILE      # another settings file instead of scripts/autotune.env
```

Nothing has to exist before the first run except the reference engine
(`jieqi/ref-engine` and `jieqi/ref-engine*.nnue`, see section 1).

### 4.2. Stages

| Stage | Data (`pipeline.sh --source`) | Leaves the stage when |
| --- | --- | --- |
| **1 · ref** | ref-engine plays both sides (`REF_GEN_NODES`) and scores the positions; Maskfish is only the referee for the board. `GEN0_GAMES` games, then train from scratch → SPRT against the material-only `gen0` | a net wins (→ gen1) |
| **2 · distill** | The current Maskfish net plays (`DISTILL_NODES`), ref-engine scores every position kept (`REF_LABEL_NODES`) | a new net measures ≥ `NEAR_REF_ELO` against ref-engine, or `DISTILL_PATIENCE` failed attempts in a row |
| **3 · self** | The net plays and scores its own games (`SELF_NODES`) | — (runs until the end) |
| **4 · options** | Option cycles (4.4), interleaved with stages 2–3 | — |

Every new net is measured against ref-engine (`EXT_PAIRS` pairs at
`EXT_MOVETIME` ms per move, same openings every time).

The reference engine prints its scores on a win-probability scale of its own. Its scale
is fitted once on its own games (gen0, saved in `data/ref.scale`) and every
file it scored is prepared with `--score-scale SCALING / ref.scale`, so all
the data is in Maskfish units. Each text file says who scored it in
`FILE.src`; the scale used for each prepared prefix is in `PREFIX.scale`.

### 4.3. One round

1. If a net was promoted but not measured yet: measure it against ref-engine.
2. If an option cycle is due (4.4): run it.
3. **Generate data** for the current stage: `pipeline.sh -g N --source S --gen-only`.
   Later attempts of the same generation add a new batch (`--tag aK`).
4. **Train → verify → SPRT → promote**: `pipeline.sh -g N --train-only`, on
   the data of the last `TRAIN_WINDOW` generations, with the options in use.
   A candidate is promoted on SPRT H1, or, when the SPRT has no decision
   after `MATCH_MAX_PAIRS` pairs, if its Elo − 95 % error is above 0.
5. A new net is measured against ref-engine. Failed attempts change strategy in
   turn: more data and continue training → more data, from scratch with 1.5×
   the epochs → more data, half the learning rate and λ 0.9.
6. In stage self, `PATIENCE` failures in a row raise the level (games ×
   (1 + level/2), nodes × (1 + level/4), up to `LEVEL_CAP`).

### 4.4. Engine option tests

The first cycle starts once a net reaches `OPT_START_ELO` against ref-engine
(or in stage self); then a new cycle runs every `OPT_EVERY` new nets, and once
more before stopping. The options are read from `src/engine.cpp`
(`options.add(...)`), so new options are picked up automatically: booleans are
tested with the opposite of their default, numeric options with the values
of `OPT_INT_TESTS`, groups from `OPT_GROUP_TESTS`. `OPT_EXCLUDE` are never
tested: output only, several threads only (`SharedHistory`,
`SimpleThreadVote`), time management only (`StopEarly`, `SimpleMtg`: no
effect with a node limit, test them with `match.py --tc`), speed only
(`ThreatCache`, `ThreatsIncremental`: same search, compare the speed with
`bench`; a fixed-node SPRT would see 0 Elo and remove them), data format
(`JieqiMaterial`), and `SeekMate`, which does not aim at playing strength.
Accepted options are used for the data too, except `NO_DATA_OPTS`
(`RevealQuiet*` and `RevealQsearchEvasions`, and `EvalScaling`: the data
is made with the scaling the training uses, `EVAL_SCALING` of pipeline.env;
the group `EvalCorrelation` tests `EvalScaling=Correlation`, which the
trainer does not model).

Each value plays an SPRT `OPT_SPRT` (default `[0, 3]`) with the current net,
at `OPT_NODES` nodes per move, against the options in use, for at most
`OPT_MAX_PAIRS` pairs:

| Result | Action |
| --- | --- |
| H1 | **On by default**: used from now on, and a verified patch changes the default in `src/engine.cpp` |
| H0 with Elo ≤ 0 | **Removed**: a verified patch deletes the option (numeric options only if in `OPT_INT_REMOVABLE`) |
| H0 with Elo > 0 | Left as it is (a gain smaller than the SPRT bound, if any); not retested |
| No decision | Left as it is and **tested again** in up to `OPT_RETRY` later cycles (with a stronger net). The undecided options with a positive estimate are also tested **together** in one SPRT (`OPT_GROUP`); if the group wins, all of them become defaults |

Why these rules: Jieqi games are very noisy (about 2 % draws, a pair score
standard deviation of ~0.35), so an option worth +3 Elo needs ~10 000 pairs to
be proven. Treating "not proven yet" as "useless" removed good options in
the previous version; now only an option that is shown to be no better
(H0) is removed. There is no confirmation against the outside engines
either: 100 pairs (±50 Elo) cannot see a few Elo and only added noise.

Removing an option means: the `options.add` call and its comment go, the
member variable becomes `static constexpr` with the default value, and the
dead branches are folded. Every patch is **built and checked with `bench`**:
the patched engine must search exactly as many nodes as the original engine
with the options in use. If not, nothing is committed and Telegram says the
option has to be edited by hand (it is still used in the runs).

### 4.5. When does it stop?

In stage self, at level `LEVEL_CAP`, after `PATIENCE` failed attempts in a
row, one more option cycle runs with the current net. If it brings no gain,
the run **finishes**: no new net can be found with the largest data / search
budget and no option helps, so the Elo is at its maximum for this setup.
Telegram then receives the final results, `REPORT.md` and the combined patch.
`MAX_HOURS` puts an optional time limit on a run.

To go further afterwards: raise `LEVEL_CAP`, `GAMES` or `SELF_NODES`, try
another architecture (`NNUE_ARCH`, `LAYERS`), delete `ST[finished]` from
`autotune/state.sh`, and run again.

### 4.6. Settings

In `scripts/autotune.env` (see `autotune.env.example`); data, training and SPRT
settings come from `pipeline.env`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `REF_ENGINE`, `REF_NET`, `REF_OPTS` | `../jieqi/ref-engine`, newest `ref-engine*.nnue`, `ChasingRule=github` | Reference engine, its network and its UCI options (also in `pipeline.env`; set `REF_OPTS` for your engine, empty for none) |
| `REF_GEN_NODES` / `REF_LABEL_NODES` / `REF_ADJUDICATE` (pipeline.env) | 10000 / 20000 / 1500 | ref-engine nodes when it plays / when it scores; adjudication threshold in ref-engine units |
| `GEN0_GAMES` / `GEN0_EPOCHS` | `GAMES` / `EPOCHS` | Stage ref |
| `DISTILL_NODES` | `NODES` | Maskfish nodes per move in stage distill |
| `NEAR_REF_ELO` / `DISTILL_PATIENCE` | -30 / 3 | End of stage distill |
| `SELF_NODES` | 2 × `NODES` | Maskfish nodes per move in stage self |
| `PATIENCE` / `LEVEL_CAP` | 3 / 4 | Failures before a scale-up / highest level |
| `TRAIN_WINDOW` | 4 | Train on the data of the last N generations (0 = all) |
| `EXT_MOVETIME` / `EXT_PAIRS` / `EXT_SEED` | 200 / 400 / 20261017 | Measurement against ref-engine |
| `OPT_START_ELO` / `OPT_EVERY` | `NEAR_REF_ELO` / 3 | When option cycles run |
| `OPT_NODES` / `OPT_SPRT` / `OPT_MAX_PAIRS` | `MATCH_NODES` / `0 3` / 12000 | Option SPRT |
| `OPT_RETRY` / `OPT_GROUP` | 2 / 1 | Undecided options: retests, group test |
| `OPT_EXCLUDE`, `OPT_INT_TESTS`, `OPT_INT_REMOVABLE`, `OPT_GROUP_TESTS`, `OPT_ONLY` | see the example | What is tested |
| `NO_DATA_OPTS` | `RevealQuiet RevealQsearchEvasions EvalScaling` | Accepted options (name prefixes) never used to generate data |
| `BENCH_ARGS` | `16 1 13` | `bench` used to verify the patches |
| `MAX_HOURS` / `RETRY_MAX` | 0 / 5 | Time limit / retries of a failed `pipeline.sh` call |

### 4.7. Recovery

All progress is written to `autotune/state.sh` after every step; finished
matches are kept in `autotune/results/` and reused; data generation resumes
from the finished games (`.partK` files); training resumes from `last.pt`.
Running the same command again (or the restart done by `--supervise`, or by
cron after a reboot) continues where it stopped. An interrupted match is
restarted from its beginning.

`--supervise` restarts after an error (after 1, 2, … 5 minutes, then every 10
minutes) but not when ref-engine or a file is missing (exit code 5).

### 4.8. Telegram messages

| Icon | When |
| --- | --- |
| 🚀 | Start / resume |
| ▶ ✅ 📈 | Steps of `pipeline.sh` (its own messages, including training progress) |
| 🏆 / ↩ | New net promoted / no epoch won the SPRT |
| 🎯 📊 | Measurement against ref-engine |
| 🔁 | Next stage |
| 🔬 ✅ ❎ 🧩 | Option cycle: start, every decision (with the patch), group test, end |
| ⛰ | Scale-up / last option cycle |
| ⏸ ❌ | Stopped / error / automatic restart |
| 🏁 | Finished: final results, report and combined patch |

## 5. Files and directories

All in `trainer/`:

| Path | Content |
| --- | --- |
| `nets/genN.nnue` | Promoted networks (`gen0` = material only); the highest N is the current net |
| `data/ref.scale` | Score scale of ref-engine (fitted on its gen0 games) |
| `data/genN*.txt`, `data/valN*.txt`, `data/prep-<arch>/` | Text and prepared data; `FILE.src` = who played / scored it |
| `runs/gen(N+1)/` | Training output of the current attempt (`runs/gen(N+1).tryK` = earlier attempts) |
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
| `Cannot start: Reference engine not found` | Set `REF_ENGINE` / `REF_NET`, or put the binary and its network in `jieqi/`. They must be Linux builds (not `.exe`); a missing executable bit is fixed automatically |
| `engine ... does not answer UCI` | Wrong binary or missing library: try `printf 'uci\nisready\nquit\n' \| jieqi/ref-engine` |
| `No CUDA` in the environment check | Install a CUDA build of torch, or set `DEVICE=cpu` / `REQUIRE_GPU=0` in `pipeline.env` (very slow) |
| `Engine and trainer are NOT bit-exact` | The engine and the trainer disagree: use the same commit for both, rebuild; see [trainer/README.md, section 11](../README.md#11-when-the-engines-network-architecture-changes) |
| An option says "edit by hand" | Its patch did not reproduce the `bench` node count; see `autotune/logs/patch.log`. The option is still used in the runs |
| `uncommitted changes` warning | `src/` or `trainer/*.py` differ from `HEAD`: patches are made on `HEAD`, the `bench` check may then fail. Commit first |
| `autotune is already running` | Another run holds `autotune/.lock`; `--status` shows its pid |
| Telegram messages missing | `scripts/pipeline.sh --test-telegram`; check `TG_BOT_TOKEN` / `TG_CHAT_ID` and that `curl` is installed |
| `no UCI option X` | An option in `REF_OPTS`, `GEN_OPTS`, `MATCH_OPTS` or `--opt` does not exist in that engine (misspelt, or a binary older than the option: rebuild). Options are no longer ignored silently |
| Stuck in stage ref | gen1 must beat the material-only gen0: every failed attempt adds `GEN0_GAMES` more ref-engine games; check the SPRT lines in `logs/gen0/` |
| The run never finishes | It ends at level `LEVEL_CAP` after `PATIENCE` failures and an option cycle without gain: check `--status`; lower `LEVEL_CAP` / `PATIENCE` or set `MAX_HOURS` |