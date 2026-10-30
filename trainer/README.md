# Jieqi NNUE trainer

A PyTorch toolkit to train NNUE networks for the Jieqi engine
[Maskfish](https://github.com/nguyenhungitc/maskfish), from scratch to a usable network or onward from an existing one, and to test
changes with SPRT matches.

The networks are trained for exactly the engine's architecture (`src/nnue` of
Maskfish):
HalfKAv2_hm features for Jieqi (7560 inputs per side, including dark pieces
and the pool), a 2048 feature transformer with 16 PSQT buckets, and 16 layer
stacks 2048 → 16 → 32 → 1. The engine reads an exported network **bit-exactly**
as the trainer computes it (`verify.py` checks this).

## Contents

1. [Components](#1-components)
2. [Requirements](#2-requirements)
3. [Setup](#3-setup)
4. [Training workflow in detail](#4-training-workflow-in-detail)
5. [Suggested schedule for the generations](#5-suggested-schedule-for-the-generations)
6. [Using other engines](#6-using-other-engines)
7. [Network architectures and training options](#7-network-architectures-and-training-options)
8. [Data formats](#8-data-formats)
9. [Disk space and time](#9-disk-space-and-time)
10. [Troubleshooting](#10-troubleshooting)
11. [When the engine's network architecture changes](#11-when-the-engines-network-architecture-changes)

The scripts of [`scripts/`](scripts/README.md) (`pipeline.sh`, `autotune.sh`)
run the whole workflow below automatically.

## 1. Components

| File | Purpose |
| --- | --- |
| `init_net.py` | Creates a material-only starting network |
| `gen_data.py` | Self-play data generation with real hidden identities, multi-process |
| `prepare_data.py` | Converts text data into shuffled binary shards (streamed, fixed RAM) |
| `train.py` | Training loop (one GPU, or several with `torchrun`); writes a checkpoint and a `.nnue` file every epoch |
| `watch_nets.py` | Measures every network of a run against a reference while it trains, ranks them, keeps the best |
| `textio.py` | Reads / writes text data plain or compressed (`.zst`, `.gz`) |
| `quant.py` | The two quantization schemes of the engine (`make quant=1/2`) as scales |
| `ftperm.py` | Reorders the feature transformer neurons of a network so that the engine evaluates it faster (same evaluations, [section 7.10](#710-faster-networks-ftpermpy)) |
| `verify.py` | Compares the engine's evaluation with the trainer's; must be bit-exact |
| `match.py` | Matches between two networks / option sets / builds, with SPRT |
| `fit_scaling.py` | Fits the score → win probability constant from self-play data |
| `ranger21.py` | Ranger21 optimizer (`train.py --optimizer ranger21`), the optimizer of nnue-pytorch |
| `chance_tables.py` | Score ↔ mass tables for the engine option `ChanceModel Tables` (fitted to self-play data, logistic, or copied from a ref-engine network) |
| `rescore.py` | Has another engine (a ref-engine, section 6) re-score positions |
| `serialize.py` | Writes / reads `.nnue` files, converts `.nnue` into a checkpoint |
| `jieqi_features.py` | Input features, a copy of Maskfish's `src/nnue/features/half_ka_v2_hm` |
| `model.py` | PyTorch model and weight clipping |
| `eval_scale.py` | The scaling of the engine's `evaluate()`, applied to the network output in the loss |
| `jieqi_uci.py` | Helpers to drive the engine over UCI |
| `scripts/` | Automation of the generations: `pipeline.sh`, `autotune.sh` ([README](scripts/README.md)) |

The engine has two things dedicated to the trainer: the UCI option `RawScore`
(scores in internal units) and the `evalraw` command (raw network output).

## 2. Requirements

**Hardware**

- **Data generation:** as many CPU cores as possible, one per worker. Several
  machines can be used at the same time.
- **Training:** an NVIDIA GPU with CUDA. The CPU works too but is only good
  for tests. The default architecture needs about 300 MB of GPU memory for
  the parameters and the optimizer (plus the batches), the `att` architecture
  about 1 GB, the `threats` architecture about 2 GB.
- **RAM:** about 2 GB for `prepare_data.py` with the default shard size;
  `train.py` memory-maps the data, so it does not need to fit in RAM.
- **Disk:** an SSD is recommended for the prepared data. See
  [section 9](#9-disk-space-and-time) for size estimates.

**Software**

- A C++17 compiler (g++ or clang) and `make` to build the engine.
- Python 3.9 or later.
- The packages of `requirements.txt`: `torch`, `numpy`, `zstandard`.
- Optional: `tensorboard` (`pip install tensorboard`) for `train.py
  --tensorboard` / `watch_nets.py --tensorboard`.

## 3. Setup

### 3.1. Build the engine

The trainer is the `trainer` directory of the Maskfish repository. The
commands of this README are run from that directory, so that the engine is
`../src/maskfish`:

```
git clone https://github.com/nguyenhungitc/maskfish.git
cd maskfish/src
make -j build ARCH=x86-64-avx2
cd ../trainer
```

- Choose `ARCH` for your CPU (`make help` lists them, e.g. `x86-64-bmi2`,
  `x86-64-avx512`, `apple-silicon`). When generating data on several
  different machines, build on each machine.
- The build embeds the published network (`embedded net: maskfish.nnue`;
  `src/maskfish.nnue` is a copy of it). The trainer does not depend on it:
  every tool gives the engine its network explicitly (`--net`, `EvalFile`),
  and `verify.py` exports its own test network.
- For the experimental architectures, add their `make` options
  ([section 7.1](#71-overview)).

### 3.2. Install Python

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy zstandard
pip install torch                # see the note below
```

(or `pip install -r requirements.txt`, which holds the same packages).

To train on a GPU, install a CUDA build of `torch` following
https://pytorch.org (choose your OS, `pip`, and your CUDA version). Check:

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

It must print `True` to train on the GPU. Machines that only generate data do
not need a GPU, only `numpy` and `zstandard` (and `torch` to run `verify.py`).

### 3.3. Check that the trainer matches the engine

```
python verify.py --engine ../src/maskfish
```

The result must be `N/N positions bit-exact with the engine`, followed by
`evaluate() scaling: N/N positions bit-exact` (the scaling that the loss
applies, see step 4). **If it does not match, do not train**: the engine and
the trainer understand the network differently (usually because the engine's
architecture changed, see
[section 11](#11-when-the-engines-network-architecture-changes)).

### 3.4. Directory layout (suggestion)

```
trainer/
  nets/      networks selected at each generation: gen0.nnue, gen1.nnue, ...
  data/      text data (genN.txt, valN.txt) and prepared data
  runs/      training output: runs/genN/epoch_K.pt, epoch_K.nnue, last.pt
```

These directories are in the `.gitignore` of the repository.

## 4. Training workflow in detail

Training is a loop, each round being a "generation":

```
current net ──► generate data ──► prepare ──► train ──► verify ──► SPRT match
     ▲                                                                  │
     └─────────────── the new net becomes the current one if it wins ◄──┘
```

Generation 0 starts from a material-only network. If a stronger Jieqi engine
(a "ref-engine") is available, generation 0 can be replaced by data generated
by it, for a much stronger first network ([section 6](#6-using-other-engines)).
To continue from the published network instead, copy it as the current one
(`cp ../src/maskfish.nnue nets/gen0.nnue`, after a build) and start at step 2.
The steps below describe one generation; repeat them with increasing
generation numbers.

### Step 1. Starting network (once)

```
python init_net.py nets/gen0.nnue
```

The PSQT weights of this network hold the piece values (a dark piece counts
as the average value of the 15 pieces of a side), and the layer stacks output
0. The engine plays with a pure material evaluation plus search, which is
enough to generate meaningful first data. The command also writes
`nets/gen0.nnue.pt` (the matching checkpoint). `--arch` makes the starting
network of another architecture (section 7), except the `-nopsqt` ones,
which need the PSQT buckets for the material.

### Step 2. Generate data

```
python gen_data.py --engine ../src/maskfish --net nets/gen0.nnue \
    --games 50000 --nodes 5000 --workers 15 --seed 100 --out data/gen0.txt
```

Also generate a small **validation set**, with another seed, to monitor
training:

```
python gen_data.py --engine ../src/maskfish --net nets/gen0.nnue \
    --games 1000 --nodes 5000 --workers 15 --seed 999 --out data/val0.txt
```

Main parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--net` | | Network used to play (the current network) |
| `--games` | 100 | Number of games. Each game gives about 40–120 positions |
| `--nodes` | 5000 | Search nodes per move. More gives more accurate scores, proportionally slower |
| `--depth` | 0 | Use `go depth N` instead of a node count |
| `--hash` | 16 | Hash (MB) of every engine |
| `--workers` | cores − 1 | Number of parallel processes |
| `--seed` | system time | **Set it explicitly**, and differently for every run / machine |
| `--random-plies` / `--random-plies-min` | 8 / 2 | Random moves at the start of a game (not written), for opening variety |
| `--max-plies` | 400 | Games longer than this (in plies) are drawn |
| `--adjudicate` | 3000 | A game is won when \|score\| ≥ this threshold for 8 plies in a row |
| `--opt` | | Extra UCI options for the engine, e.g. `--opt ChanceScaling=400` |
| `--full-info` | off | Tell both sides the identity of a captured dark piece (by default only the capturer knows it, as in a real game) |
| `--teacher`, `--teacher-net`, `--teacher-opt` | the player | Another engine that scores the positions written (distillation: the current network plays, a stronger engine scores, see [section 6](#6-using-other-engines)) |
| `--teacher-nodes` / `--teacher-depth` / `--teacher-hash` | 20000 / 0 / 16 | Search limit and hash of the teacher |
| `--referee`, `--referee-net` | the player | Engine that gives the FEN, the checks and the legal moves (needed when the player is another engine, e.g. `--referee ../src/maskfish`; Maskfish needs no network for it) |

Good to know:

- Every line reads `fen | score | result`. The score is in internal units,
  for the side to move; the result is 1 / 0 / −1 (win / draw / loss) for the
  side to move. Positions in check, positions whose best move is a capture
  and positions scored as a mate are skipped.
- The FEN is the side to move's view: the identity of a dark piece it lost
  is unknown to it (the piece stays in its pool), as when the engine plays.
- The safe reveal bonus (`RevealQuiet*` options) is disabled during data
  generation: it is a preference of the search between moves, and would
  otherwise be part of the scores the network learns. `--opt` can set it.
- A game ends on the third occurrence of a position: perpetual check loses,
  other repetitions are drawn.
- The identities of the dark pieces are really drawn at random for every
  game. Each side only knows what it would know in a real game: reveals are
  public, the identity of a captured dark piece is only told to the capturer
  (`--full-info` tells both).
- Games are drawn after 80 plies without a capture (the engine's 40-move
  rule).
- Every worker writes its own `.partN` file; they are merged when all are
  done. **Interrupted:** run the same command again: it only plays the games
  still missing (the `.partN` files hold the finished games), then merges
  the parts.
- **Several machines:** run `gen_data.py` on each machine with different
  `--seed` and `--out`, then copy the files back. They can be prepared
  together (several files given to `prepare_data.py`) or separately into
  different prefixes; `train.py` accepts several prefixes.
- Keep the text files: they are needed to prepare the data again when the
  architecture changes (e.g. `att`, section 7) and to run `fit_scaling.py`.
- **Compressed text:** `--out data/gen0.txt.zst` writes a zstd file, about 6
  times smaller (15 bytes per position instead of 92 in a test). The
  `.partN` files stay plain until they are merged. `prepare_data.py`,
  `fit_scaling.py`, `chance_tables.py` and `rescore.py` (input) read `.zst`
  and `.gz` files directly, so existing files can also be compressed with
  `zstd --rm data/gen0.txt` and used as `data/gen0.txt.zst`.

### Step 3. Prepare the data

```
python prepare_data.py data/gen0.txt --out data/train0
python prepare_data.py data/val0.txt --out data/val0
```

The result is the shards `data/train0-00000.features.npy`, `...meta.npy`, ...
and the file `data/train0.arch` holding the architecture.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--shard-size` | 2 000 000 | Positions per shard. RAM needed is about 2 × 260 bytes × this value; lower it if RAM is short |
| `--max-score` | 10000 | Drop positions with a larger \|score\| (internal units, after `--score-scale`) |
| `--score-scale` | 1.0 | Multiply the scores by this factor, to convert another engine's scores to internal units |
| `--workers` | cores | Number of FEN parsing processes |
| `--arch` | `default` | Feature set of the network (`att`, `threats`, `att-threats`, with `-meta` / `-blend` / `-s32`; see [section 7](#7-network-architectures-and-training-options)). The other options (`-l2`, `-ctx`, `-nopsqt`, `-q2`) do not change the data |
| `--jieqi-material` | off | The data was generated with the engine option `JieqiMaterial=true` (store the material `evaluate()` then uses) |
| `--max-dups` | all | Keep at most N copies of each position ([section 7.15](#715-duplicate-positions---max-dups)) |
| `--seed` | 1 | Seed of the shuffling |

Every shard is shuffled before it is written; during training every batch
comes from a random shard, so the data is mixed across shards. Every position
mirrored left-right is also added when its features differ from the
original's (only when both kings stand on the middle file, and never with the
`att` feature sets); the script reports how many. Lines that do not parse are
skipped and counted.

### Step 4. Train

```
python train.py data/train0 --val data/val0 --out runs/gen1 \
    --epochs 40 --epoch-size 10000000 --batch-size 16384
```

From the second generation on, train on **all the data** of the generations
and start from the current network:

```
python train.py data/train0 data/train1 --val data/val1 --out runs/gen2 \
    --init-net nets/gen1.nnue --epochs 40 --epoch-size 10000000
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `train` (positional) | | One or more prefixes created by `prepare_data.py` |
| `--val` | | One or more validation prefixes (at most 1 million positions are used) |
| `--out` | | Output directory |
| `--epochs` | 100 | **Total** number of epochs (with `--resume`, including the epochs already run) |
| `--epoch-size` | 20 000 000 | Positions per epoch (need not equal the data size) |
| `--batch-size` | 16384 | Lower it if GPU memory is short |
| `--lr` | 8.75e-4 | Initial learning rate |
| `--gamma` | 0.992 | Learning rate decay after every epoch |
| `--lambda` | 1.0 | 1.0: learn from the search score only; lower: also mix in the game result |
| `--scaling` | 361 | Score → probability constant |
| `--lambda-end` | | Lambda of the last epoch: lambda goes linearly from `--lambda` to it (e.g. 1.0 → 0.8) |
| `--score-scaling` | `--scaling` | Score → probability constant of the data scores only (e.g. 410 with `--scaling 361`, for data of another engine) |
| `--loss-power` | 2.5 | Exponent of the loss (2: mixing the targets is the same as mixing the losses) |
| `--scaling-table` | | Scaling as a function of the total material, from `fit_scaling.py --by-material`, instead of the constant `--scaling` ([section 7.16](#716-score--probability-by-material---scaling-table)) |
| `--layers 2`, `--ctx`, `--no-psqt`, `--quant2` | | Architecture options of the model (`-l2`, `-ctx`, `-nopsqt`, `-q2`, [section 7](#7-network-architectures-and-training-options)); the feature set comes from the prepared data |
| `--init-net` | | Start from a `.nnue` file (new optimizer) |
| `--init-ft-only` | | With `--init-net`, take only its feature transformer and train the layer stacks from scratch (to compare layer stack architectures fairly from the same feature transformer) |
| `--resume` | | Continue an interrupted run from a checkpoint (optimizer and epoch count included) |
| `--device` | `cuda` if available | `cpu` to force the CPU |
| `--seed` | 42 | Seed of the initialization and of the batches |
| `--threads` | 0 | Number of CPU threads for PyTorch (0 = default) |
| `--no-factorize` | | Disable the virtual features (on by default, see below) |
| `--factorize-kb` | | Add virtual features per king bucket (off by default, see below) |
| `--no-l1-fact` | | Disable the fc0 part shared by the layer stacks (on by default) |
| `--output-lr-scale` | 0.1 | Learning rate of the output layer, relative to `--lr` |
| `--optimizer` | `adam` | `ranger21`: the optimizer of nnue-pytorch (AdamW with lookahead, `ranger21.py`); the learning rate still follows `--lr` and `--gamma`. A run resumed with another optimizer starts a new one at the current learning rate |
| `--stack-balance` | 0 (off) | Balance the loss across layer stacks; the value is the cap (e.g. 8) |
| `--checkpoint-every` | 1 | Write `epoch_K.pt` only every K epochs (and at the last one); `last.pt` and the `.nnue` files are always written |
| `--eval-scaling` | `complexity` | The engine's `EvalScaling` option the network will play with: `complexity`, `alignment` or `correlation` ([section 7.13](#713-evalscaling-alignment)) |
| `--no-eval-scale` | | Compare the raw network output with the data scores (old behaviour, see below) |
| `--qat-start` | 0 (off) | Quantization-aware training from this epoch on ([section 7.11](#711-quantization-aware-training)) |
| `--val-qat` | | Also print the validation loss of the quantized network before `--qat-start` |
| `--prefetch` | 4 | Batches read ahead in a background thread while the GPU trains (0: none) |
| `--tensorboard` | | Also write TensorBoard logs to `<out>/tb` |

The loss is that of nnue-pytorch: the network's score and the data score are
both converted to a win probability with `sigmoid(score / scaling)`, the
target is `lambda × probability from the score + (1 − lambda) × game result`,
and the loss is the mean of `|prediction − target|^2.5`.

**The network's score is taken after the engine's `evaluate()` scaling.** The
data scores are search scores, made of `evaluate()` values, and `evaluate()`
is not the raw network output: it multiplies it by `(18560 + material) /
33331` (at most about 0.85 with the material of Jieqi), damps it by the
"complexity" `|psqt − positional|` and by the 40-move counter. A network
trained to output the search scores directly would be scaled down again by
the engine, and every generation would learn a smaller scale than the
previous one (measured with the material network: search scores are 0.5–0.85
times its raw output). The trainer therefore applies the same function
(`eval_scale.py`) to the network output before the loss, using the material
and 40-move counter that `prepare_data.py` stores with every position; the
exported network is unchanged, and `verify.py` checks that the function is
the engine's, bit for bit. Data prepared before this change has no material:
prepare it again from the text files (`pipeline.sh` does it automatically),
or train with `--no-eval-scale`.

**Training-only techniques** (ideas from nnue-pytorch). They do
not change the network format: at export, the virtual parts are added into
the real weights, so the engine needs no change and `verify.py` must still be
bit-exact.

- **Virtual features (factorization):** every feature "piece X on square Y in
  king bucket B" also activates a weight row shared by all king buckets for
  "piece X on square Y". What the network learns for one king position is
  reused for the others, so it learns faster while data is scarce (more so
  with the `att` architecture and its 24 buckets). In a small test on 2,700
  positions, the best validation loss went from 0.00331 down to 0.00323. The
  virtual row per king bucket (`--factorize-kb`) is added once for every
  active feature, so it learns very fast; in the same test it made the result
  worse (0.00343), so it is off by default.
- **Shared fc0 (`l1_fact`):** part of the fc0 layer is shared by the 16 layer
  stacks, so that the stacks with few samples (endgames) start from what the
  stacks with many samples have learned. Weight clipping applies to the sum
  after merging.
- **Separate learning rate for the output layer:** 1/10 of the base rate by
  default.
- **Layer stack balancing (`--stack-balance CAP`):** every sample is weighted
  by `min(CAP, total samples / samples of its stack)`, normalized so that the
  average weight of every stack is 1. With `CAP=8` and 16
  stacks, only the stacks holding more than 1/8 of the data are weighted
  down; to boost rare stacks more, use a larger CAP (16–32). When enabled, the
  command prints the sample count and weight of every stack. The validation
  loss is always unweighted (the real distribution). Worth trying with a lot
  of data and too few endgames; decide with `match.py`, not with the loss.

The last two are nearly neutral at small scale; their effect only shows with
a lot of data. The checkpoint records the factorization settings: `--resume`
automatically uses the settings of that run, while `--init-net` from a
`.nnue` file starts with zero virtual parts. Old checkpoints (from before
these techniques) can still be loaded and trained further; their optimizer
state is then created anew.

**Monitoring:** the average loss is printed every 100 steps; at the end of
every epoch the train loss, the validation loss, the learning rate, the
time and the training speed (positions per second). The validation loss
should go down and then level off. Only compare validation losses of runs
that use **the same validation set**; a lower loss does not mean a stronger
network, only a match tells (step 6).

The same numbers are appended to `runs/gen1/metrics.jsonl` (one JSON object
per line: `"type": "step"` every 100 steps, `"type": "epoch"` per epoch; it
continues across `--resume`). With `--tensorboard` they also go to
`runs/gen1/tb`:

```
tensorboard --logdir runs        # then open http://localhost:6006
```

**Data loading:** batches are read from the memory-mapped shards in a
background thread (`--prefetch`, 4 batches ahead) while the GPU computes the
previous step, in page-locked memory, and are sent to the GPU in their
stored types (16-bit feature indices, 4 times fewer bytes than before),
then converted there. The batches and their order do not depend on
`--prefetch` (same seed, same batches). On a 2-core test machine, building
a batch of 16384 positions took 3.0 ms instead of 4.8 ms, and it now
overlaps the GPU step instead of adding to it.

**Several GPUs** (one machine): start the same command with `torchrun`:

```
torchrun --standalone --nproc_per_node 4 train.py data/train0 --val data/val0 \
    --out runs/gen1 --epochs 40 --epoch-size 10000000 --batch-size 16384
```

- `--batch-size` stays the **total** batch: each of the 4 processes draws its
  own random 4096 positions, and DistributedDataParallel averages the
  gradients, so a step is the same as a step on one GPU with 16384 positions:
  same hyperparameters, same number of steps per epoch, about 4 times
  faster when the GPUs are the bottleneck. `--batch-size` must be a multiple
  of the number of processes.
- Only process 0 prints, validates and writes the files; `--resume`,
  `--init-net` and every other option work as with one GPU.
- With threats networks (about 400 MB of gradients to average per step) the
  speed-up depends on the link between the GPUs (NVLink ≫ PCIe).
- `scripts/pipeline.sh`: set `TRAIN_GPUS=4` in `pipeline.env`.
- On the CPU (`--device cpu`, tests only) the processes use the gloo backend.

**Output** in `runs/gen1/`: for every epoch `epoch_K.pt` (checkpoint, about
190 MB, about 750 MB for `att`, about 1.3 GB for `threats`) and `epoch_K.nnue`
(ready for the engine), plus `last.pt`. Delete old checkpoints if space is
short, but keep `last.pt` and the `.nnue` files.

**Interrupted:** run the same command again, adding `--resume runs/gen1/last.pt`.

### Step 5. Check the exported network

```
python verify.py --engine ../src/maskfish \
    --net runs/gen1/epoch_40.nnue --checkpoint runs/gen1/epoch_40.pt
```

It must be bit-exact (`N/N positions bit-exact`). The line `largest
float/quantized difference` gives the quantization error between the PyTorch
model and the integer network; a few dozen internal units is normal.

### Step 6. Select the network with an SPRT match

Play the last few epochs against the current network:

```
python match.py --engine ../src/maskfish \
    --net-a runs/gen1/epoch_40.nnue --net-b nets/gen0.nnue \
    --nodes 20000 --workers 15 --sprt 0 5
```

- A is the version being tested, B the reference; results are given for A.
- Games are played in pairs: same hidden identities, same random opening,
  colors swapped. Each engine only knows what its side knows (the identity of
  a captured dark piece is only given to the capturer; `--full-info` gives it
  to both).
- `--sprt 0 5`: stops when H1 (A is stronger by about 5 Elo or more) or H0 (A
  is not stronger) is accepted. For small changes use `--sprt 0 3`. When the
  new network is much stronger (early generations), the SPRT ends very fast.
- Printed periodically: number of pairs, wins/draws/losses of A, pair
  statistics (pentanomial), Elo ± 95% confidence interval, and LLR.
- Use the same kind of limit as the engine will play with (`--nodes`,
  `--depth`, `--movetime`, or `--tc 10+0.1` for real clocks, which also tests
  the time management). Every engine uses 1 thread; `--workers` pairs of
  games run in parallel.
- `match.py` also compares option sets (`--opt-a Name=value`, `--opt-b`) and
  builds (`--engine-a`, `--engine-b`; `--engine` is the default of both). B
  uses A's engine and network unless told otherwise. `--pairs` (10000) caps
  the match, `--min-pairs` (20) delays the first SPRT decision, `--alpha` /
  `--beta` (0.05) are the error rates.

**Measuring every epoch while training** (as nnue-pytorch's `run_games.py`):
`watch_nets.py` plays a fixed number of pairs of every new `epoch_K.nnue`
against the reference, with the same openings for every network, prints a
ranking, and keeps the best one as `runs/gen1/best.nnue`. Run it next to the
training, on the CPU cores the trainer does not use:

```
python watch_nets.py runs/gen1 --engine ../src/maskfish --ref nets/gen0.nnue \
    --pairs 200 --nodes 5000 --workers 8 --every 2 --follow --until-epoch 40
```

Results are kept in `runs/gen1/watch.json` (restartable; a larger `--pairs`
tops up the networks already measured), and `--tensorboard` adds the Elo of
each epoch to the run's TensorBoard logs. A few hundred pairs give about
±15–30 Elo: use the ranking to choose which epochs go to the SPRT above,
not as the final decision.

If the new network wins (H1), copy it as the current network:

```
cp runs/gen1/epoch_40.nnue nets/gen1.nnue
```

If it loses (H0), try another epoch, add data, or tune the training
parameters (`--lr`, `--lambda`, number of epochs) and train again.

### Step 7. Calibrate the score → probability constant

Once a new network exists, generate data with it (step 2 of the next
generation), then:

```
python fit_scaling.py data/gen1.txt
```

The script prints the best fitting constant and a table comparing the
observed win rates with the predicted ones. Use that value for:

- the engine when generating data and in matches: `--opt ChanceScaling=<s>`
  (for `gen_data.py`) / `--opt-a ChanceScaling=<s> --opt-b ChanceScaling=<s>`
  (for `match.py`), or `setoption name ChanceScaling value <s>` in a GUI;
- the trainer: `train.py --scaling <s>`.

Fit it again after every generation. Check with `match.py` that the new value
does not weaken the engine before making it the default. `--by-material`
also fits it by ranges of material, for `train.py --scaling-table`
([section 7.16](#716-score--probability-by-material---scaling-table)).

The same data can also give lookup tables for the engine option
`ChanceModel Tables` (averaging of chance nodes in "mass" space, not tested
for Maskfish yet):

```
python chance_tables.py fit data/gen1.txt --out tables/gen1
python match.py ... --opt-a ChanceModel=Tables ChanceTables=$PWD/tables/gen1
```

`chance_tables.py logistic --scale <s>` writes tables equivalent to the default
averaging (a reference), and `chance_tables.py extract <ref-engine network>`
copies the tables held by a ref-engine network that has them (in the score
units of that engine: they fit Maskfish's only roughly).

### Step 8. Next generation

1. Generate data with the current network (`nets/genN.nnue`) and a new seed,
   plus a new validation set (`valN.txt`).
2. Prepare them into `data/trainN` and `data/valN`.
3. Train on all the data: `train.py data/train0 ... data/trainN --val data/valN --init-net nets/genN.nnue --out runs/genN+1`.
4. Verify, play against `nets/genN.nnue`, keep the winner.

After several generations, the data of the first generations (lower quality)
can be left out of the training command.

## 5. Suggested schedule for the generations

The table below is only a **starting point**, not validated at real scale;
adjust it according to the match results.

| Generation | Network generating the data | New data | `--nodes` | Training | `--lambda` |
| --- | --- | --- | --- | --- | --- |
| 1 | gen0 (material) | 5–10 million positions | 5000 | 20–30 epochs × 10 million | 1.0 |
| 2 | gen1 | 10–20 million | 5000 | 30–40 epochs × 10 million, `--init-net` | 1.0 |
| 3 | gen2 | 20–40 million | 5000–10000 | 40–60 epochs × 20 million | 1.0 |
| 4+ | genN | 50 million or more | 10000 | 60–100 epochs × 20–50 million | 0.75–1.0 |

General principles:

- The first generations improve fast; increase the data and the node count
  gradually as the improvement slows down.
- Only lower `--lambda` (mixing in game results) once the network is decent:
  the results of games played by a weak network are very noisy.
- After a few generations, try training from scratch (without `--init-net`)
  on all the good data and compare with the continued network; keep whichever
  wins.

## 6. Using other engines

Another complete Jieqi engine, called **ref-engine** in this README and in
the scripts, can be used in two roles:

- **teacher**: an engine stronger than the current network generates the
  data or scores the positions (distillation), much faster than self-play
  from a material-only network;
- **yardstick / reference opponent**: matches against it measure progress.

Any UCI engine can be a ref-engine (a third-party engine, an older Maskfish
build or branch, ...) if it understands this engine's protocol: the FEN with
the pool, the `d` command, `go perft`, the notation of moves with the revealed
identity, and captures of dark pieces without identity (see the
[engine's README](../README.md#uci-protocol)). Check it with `go perft` and a
few `position ... moves ...` commands before using it.

- Do not use as a teacher an engine whose evaluation is biased (e.g. one that
  favors one color in the opening position): the network would learn the bias.
  Such an engine can still be a reference opponent.
- An engine without the `RawScore` option prints its own score scale: data
  scored by it must be **rescaled** before training
  ([section 6.2](#62-rescale-the-scores-and-train)).
- `scripts/pipeline.sh` and `autotune.sh` expect a Linux binary.

The commands below use the layout of the automation scripts: the binary and
its network in `../jieqi/`, named `ref-engine` and `ref-engine.nnue` (any
names work on the command line). `--opt Name=value ...` passes the UCI
options the ref-engine needs (e.g. its repetition rules closest to
Maskfish's).

### 6.1. Option A: the ref-engine plays itself to generate data

Instead of steps 1–2 of the first generation (material network):

```
python gen_data.py --engine ../jieqi/ref-engine \
    --net ../jieqi/ref-engine.nnue --opt Name=value \
    --referee ../src/maskfish \
    --games 50000 --nodes 5000 --adjudicate 1000 --workers 15 \
    --seed 100 --out data/ref0.txt
```

- `--referee ../src/maskfish`: Maskfish gives the board state (FEN, checks,
  legal moves), so that the FEN written is in Maskfish's format; it needs no
  network for that.
- `--adjudicate` is in the ref-engine's score scale. 1000 is only a starting
  value: once data exists, look at the table of `fit_scaling.py` (section 6.2)
  and choose the score at which the observed win rate is close to 100%.
- A ref-engine with a much larger network is slower than this engine (in one
  test, roughly 7 positions per second per worker, including network
  loading).

### 6.2. Rescale the scores and train

```
python fit_scaling.py data/ref0.txt
# suppose it prints "fitted scale: 250.0" -> factor = 361 / 250 = 1.444
python prepare_data.py data/ref0.txt --out data/train_ref0 --score-scale 1.444
python prepare_data.py data/ref_val.txt --out data/val_ref --score-scale 1.444
python train.py data/train_ref0 --val data/val_ref --out runs/ref1 \
    --epochs 40 --epoch-size 10000000
```

`361` is the trainer's default constant (`--scaling`) and close to the one
this engine uses; if a constant has been fitted for this engine, use it
instead of 361. The rescaling factor must be fitted again for every setup of
the ref-engine (engine, network, node count).

Rescaled data can be combined with this engine's self-play data: prepare each
part separately (each with its own `--score-scale`) and give several prefixes
to `train.py`.

### 6.3. Option B: re-score self-play data with the ref-engine

Once this engine has a network, its self-play data is varied and cheap to
generate, while a stronger ref-engine scores more accurately. Combine both,
either while generating (`gen_data.py --teacher ../jieqi/ref-engine
--teacher-net ../jieqi/ref-engine.nnue --teacher-opt Name=value`), or
afterwards with `rescore.py`:

```
python rescore.py data/gen3.txt --out data/gen3_ref.txt \
    --engine ../jieqi/ref-engine --net ../jieqi/ref-engine.nnue \
    --opt Name=value --nodes 20000 --workers 15
python fit_scaling.py data/gen3_ref.txt
python prepare_data.py data/gen3_ref.txt --out data/train3_ref --score-scale <361/s>
```

- Every position keeps its original game result and gets the ref-engine's
  search score for the side to move.
- Positions where the ref-engine reports a mate, or whose best move is a
  capture, are dropped (`--keep-captures` keeps the latter). `--depth` and
  `--hash` work as in `gen_data.py`.
- After an interruption, run the same command again, adding `--resume`.

## 7. Network architectures and training options

The default architecture (section 1) is the one the engine is built with by
`make build`. The engine can also be built with experimental architectures;
**none of them has been shown to be stronger yet**: compare them with
`match.py` before using one.

### 7.1. Overview

An architecture is a **feature set** (`default`, `att`, `threats`,
`att-threats`) followed by **options**, as name suffixes in this order:
`-meta`, `-blend`, `-s32`, `-l2`, `-ctx`, `-nopsqt`, `-q2` (e.g.
`att-meta-l2-ctx`; `-blend` and `-s32` cannot be combined). The engine, the
network and the prepared data must agree:

| Part of the name | Engine (`make`) | Trainer | Section |
| --- | --- | --- | --- |
| `att` | `nnue=att` | `prepare_data.py --arch att...` | 7.2 |
| `threats` | `threats=yes` | `prepare_data.py --arch threats...` | 7.3 |
| `-meta` | `meta=yes` | `prepare_data.py --arch ...-meta` | 7.8 |
| `-blend` | `lsblend=yes` | `prepare_data.py --arch ...-blend` | 7.7 |
| `-s32` | `stacks=32` | `prepare_data.py --arch ...-s32` | 7.14 |
| `-l2` | `layers=2` | `train.py --layers 2` | 7.4 |
| `-ctx` | `ctx=yes` | `train.py --ctx` | 7.6 |
| `-nopsqt` | `psqt=no` | `train.py --no-psqt` | 7.5 |
| `-q2` | `quant=2` | `train.py --quant2` | 7.12 |

The feature set, `-meta`, `-blend` and `-s32` change the prepared data
(prepare the text data again into a new prefix); the last four only change
the model: `train.py` takes them as options and the feature set from the
data. For example:

```
cd ../src && make clean && make -j build ARCH=x86-64-avx2 nnue=att meta=yes ctx=yes && cd ../trainer
python verify.py --engine ../src/maskfish --arch att-meta-ctx
python prepare_data.py data/gen3.txt --out data/train3-att-meta --arch att-meta
python train.py data/train3-att-meta --ctx --out runs/att-meta-ctx ...
```

Run `make clean` whenever the architecture options change. A network of
another architecture is refused by the engine (`network file ... was not
loaded`). `init_net.py --arch` and `verify.py --arch` take the full name.
`scripts/pipeline.sh` selects the architecture with `NNUE_ARCH`, `META`,
`BLEND`, `STACKS`, `LAYERS`, `CTX`, `PSQT` and `QUANT` (one engine binary,
line of networks and prepared data per architecture, see
[scripts/README.md](scripts/README.md#3-pipelinesh--one-generation)).

### 7.2. Attack buckets (`att`)

The feature transformer is also bucketed by the visible attacking material
of the perspective's side (4 attack buckets: has a rook? has a knight or a
cannon?), so 6 king buckets × 4 = 24 buckets (30240 inputs per side instead
of 7560). When both kings stand on the middle file, the board is mirrored
according to its left/right balance ("mid mirror"); the prepared data then
holds no mirrored copies (section 4, step 3). About 1 GB of GPU memory and
750 MB per checkpoint; virtual features (step 4) matter more with its 24
buckets.

Variants of the mid mirror, to be tested with an SPRT: `prepare_data.py
--mid-pawn-tiebreak` (engine option `MidMirrorPawnTieBreak=true`) also tells
apart pairs of pawns with the same sum of squares, and `--mid-no-king-flag`
(`MidMirrorNoKingFlag=true`) does not test the king flag of the balance
encodings, which unbalanced positions can corrupt. The prepared data, the
network and the engine must use the same settings, which the network file
does not record; `verify.py` takes the same options.

### 7.3. Threat features (`threats`, `att-threats`)

A second feature transformer reads 92,711 threat features (a piece attacking
another, adapted to dark pieces: a dark piece attacks like the piece of its
square), summed with the first one; the feature transformer has 1024 outputs
instead of 2048. About 2 GB of GPU memory, 1.3 GB per checkpoint, `.nnue`
files up to about 60 MB; the prepared data takes about 90 more bytes per
position (section 8) and `prepare_data.py` is slower (section 9). The order
of the features must stay that of the engine (section 11). The engine option
`ThreatsIncremental` updates them incrementally (same evaluation, a speed
option).

### 7.4. Layer stacks 2 (`-l2`, `make layers=2`)

Each layer stack becomes: fc0 L1 → 32, both squared and plain clipped ReLU
→ fc1 64 → 32, both activations again → fc2 reads all 128 activations → 1,
plus a direct term fc0[30] − fc0[31] (see `model.py`). A network with the
default layer stacks cannot be converted: `train.py --layers 2 --init-net
<network>` starts from its feature transformer only and trains the layer
stacks from scratch.

### 7.5. Networks without PSQT (`-nopsqt`, `make psqt=no`)

The feature transformer has no PSQT buckets: the output is that of the layer
stack alone. `init_net.py` cannot make one (the material lives in the PSQT
buckets): train it with `train.py --no-psqt --init-net <network with PSQT>`,
which takes the feature transformer of that network. With `EvalScaling
Complexity` the material balance (dark pieces at the average value of their
pool, field `sed` of the prepared data) stands for the missing PSQT part;
`EvalScaling Alignment` (section 7.13) does not need it and suits these
networks.

### 7.6. Inventory context (`-ctx`, `make ctx=yes`)

16 inputs describing the uncertainty of the position, from the side to move's
point of view: the pool counts of each piece kind of both sides, the number of
dark pieces of each side on the board, and the number of dark pieces of each
side captured without being identified, each scaled to 0..127. A small layer
(16 → 16, clipped ReLU) turns them into 16 more inputs of fc0 in every layer
stack. The inputs are computed from the features, so the data does not
change. A network without it can be converted (section 7.9): the new
weights are zero, the evaluation is the same.

### 7.7. Blended layer stacks (`-blend`, `make lsblend=yes`)

The layer stack is chosen from the piece count plus an "uncertainty phase"
(the number of dark pieces on the board and of rooks, cannons and knights
left in the pools), and the output is interpolated between that stack and the
next one, with a weight in 1/256 (stored in the prepared data as `blend`).
The data must be prepared again. A network without it can be converted
(section 7.9), but the evaluation changes since the stacks are selected
differently. Not with `-s32`.

### 7.8. Meta features (`-meta`, `make meta=yes`)

88 more rows per feature transformer bucket, 9 active features per
perspective: the number of rooks, cannons and knights in each side's pool, the
size of each side's pool, the density of dark pieces on the board, the
number of dark pieces of each side captured without being identified, and
the number of visible enemy pieces attacked by each side's visible pieces.
The prepared features become `(N, 2, 80)`: prepare the data again. A
network without them can be converted (section 7.9): zero rows, the same
evaluation.

### 7.9. Converting a network to another architecture

```
python serialize.py --convert default-meta-ctx nets/gen5.nnue nets/default-meta-ctx-gen0.nnue
```

`--convert ARCH` adds options to a network: `-meta` and `-ctx` (zero weights:
the same evaluation), `-s32` (every layer stack copied into the two that split
it: the same evaluation), `-blend` (the layer stacks are kept, the evaluation
changes) and `-q2` either way (requantized: a few weights may be clipped, the
evaluation changes slightly). It cannot change the feature set, add or
remove `-l2` or `-nopsqt`, or remove an option.

`train.py --init-net` does the same conversion automatically when the network
is of another architecture; when the conversion is impossible (`-l2`,
`-nopsqt`) but the feature set is the same, it takes only the feature
transformer (`--init-ft-only` forces that). `serialize.py --to-checkpoint in.nnue out.pt` turns a network into a
checkpoint. In `scripts/pipeline.sh`, `INIT_NET` starts a new architecture
from a converted network.

### 7.10. Faster networks (`ftperm.py`)

```
python ftperm.py nets/gen3.nnue nets/gen3.nnue --data data/gen3.txt.zst
python ftperm.py nets/gen3.nnue --data data/gen3.txt.zst --stats-only
```

Reorders the neurons of the feature transformer (and the matching fc0
inputs) so that neurons that are often zero together fall into the same
4-byte chunk, which the engine skips in fc0: the engine evaluates faster and
every evaluation is exactly the same (the script checks it on positions of
the data and refuses to write otherwise; `bench` gives the same node count).
Run it on the network that will be used, e.g. after the SPRT and before
publishing; a permuted network can be trained further like any other.
Options: `--positions` (20000) and `--holdout` (5000) positions, `--scan`,
`--sweeps`, `--check`, `--seed`.

### 7.11. Quantization-aware training

`train.py --qat-start K`: from epoch K on, the forward pass computes what the
engine computes, with the weights rounded as the export rounds them and the
integer arithmetic of the engine (straight-through gradients), so the loss is
that of the exported network. `--val-qat` prints, before epoch K, the
validation loss of the quantized network next to the float one. `verify.py`
reports the float/quantized difference (step 5).

### 7.12. Alternative quantization (`-q2`, `make quant=2`)

fc0 and the output layer at twice the resolution, the feature transformer at
twice the resolution but half the range (weights and biases at scale 255
instead of 127, not doubled by the engine); the scales are listed in
`quant.py`. Train with `train.py --quant2` (`--init-net` from a network of
the other quantization requantizes it), or convert a network (section 7.9).

### 7.13. `EvalScaling Alignment`

The engine's `evaluate()` scales the network output (step 4). The default,
`EvalScaling Complexity`, damps it by `|psqt − positional|` and scales it by
the major material. `Alignment` does not use the PSQT / positional split (it
suits `-nopsqt` networks): it scales the output by how much it agrees with
the material balance and by the total material (formulas in
`eval_scale.py`). A network is trained for one of them: use the same value
everywhere, `train.py --eval-scaling alignment`, `verify.py --eval-scaling
alignment`, `setoption name EvalScaling value Alignment` in the engine (and
`--opt EvalScaling=Alignment` when generating data and in matches;
`EVAL_SCALING` in `pipeline.env`). It needs the fields `se` and `allmat` of
the prepared data (prepare old data again).

The third value, `Correlation`, measures the same agreement with the
material balance of `simple_eval_dark()` (a dark piece at the average value
of its pool) and scales by the major material, as `Complexity` (so
`JieqiMaterial` applies). Train with `train.py --eval-scaling correlation`
(`verify.py --eval-scaling correlation`, `EvalScaling Correlation` in the
engine, `EVAL_SCALING="Correlation"` in pipeline.env). It needs the field
`sed` of the prepared data.

### 7.14. 32 layer stacks (`-s32`, `make stacks=32`)

Each piece count bucket is split in two by whether more than half of the
pieces other than the kings are dark; the layer stacks 2b and 2b + 1 share the
PSQT bucket b. The layer stack of every position changes: prepare the data
again. A network of 16 layer stacks can be converted (section 7.9) with the
same evaluation. Not with `-blend`.

### 7.15. Duplicate positions (`--max-dups`)

`prepare_data.py --max-dups N` keeps at most N copies of each position (the
same FEN but for the move number), the first ones of the input, so that
positions repeated in many games (openings) weigh less in the training. The
inputs are read twice (a first pass hashes the positions). `PREP_MAX_DUPS` in
`pipeline.env`.

### 7.16. Score → probability by material (`--scaling-table`)

```
python fit_scaling.py data/gen1.txt --by-material 8 --table-out nets/scaling.json
python train.py ... --scaling-table nets/scaling.json
```

`--by-material 8` also fits one scale per range of total material (8 ranges
with as many positions each), a win rate model that depends on the material;
`train.py --scaling-table` then interpolates the scaling between these points
instead of using the constant `--scaling` (`--score-scaling` is scaled by the
same factor). It needs the field `allmat` of the prepared data. The engine's
`ChanceScaling` stays one constant: use the global scale for it.

## 8. Data formats

**Text** (written by `gen_data.py`, read by `prepare_data.py` and
`fit_scaling.py`; plain, or compressed when the name ends with `.zst` or
`.gz`, see `textio.py`):

```
xxxxkxxxx/9/1x5x1/x1x1x3x/1P4n2/9/X1X1X1X1X/6B2/9/XXXXKXXXX b R2A2C2P4N2B1r2a2c2p5n1b2 5 3 | -272 | 1
```

The engine's FEN (with the pool), the score for the side to move in internal
units, and the result for the side to move (1 / 0 / −1). Data from other
sources can be added if it uses this format and internal units (`RawScore`).

**Prepared** (per shard, about 264 bytes per position):

- `PREFIX-NNNNN.features.npy`: `int16 (N, 2, 64)` (`(N, 2, 80)` for the `-meta`
  architectures), feature indices of the side to move and of the other side,
  padded with the padding index;
- `PREFIX-NNNNN.meta.npy`: records `bucket` (layer stack), `result`, `score`,
  `material` and `rule40` (inputs of the `evaluate()` scaling, see step 4),
  `se` and `allmat` (material balance for the side to move and total
  material, for `EvalScaling Alignment`, section 7.13), `sed` (material
  balance with dark pieces at the average value of their pool, for `-nopsqt`
  networks, section 7.5), and `blend` (interpolation weight of the layer
  stacks) for the `-blend` architectures;
- `PREFIX.arch`: architecture name;
- `PREFIX.material`: `major` or `jieqi` (`--jieqi-material`);
- `PREFIX.fields`: names of the fields of the `.meta.npy` records.

With the `threats` / `att-threats` architectures every shard also has (about
90 bytes per position):

- `PREFIX-NNNNN.threats.npy`: `uint32 (T,)`, threat feature indices of all
  positions concatenated, each position holding the side to move's part and
  then the other side's;
- `PREFIX-NNNNN.thrcount.npy`: `uint8 (N, 2)`, number of threat features of
  the side to move and of the other side.

## 9. Disk space and time

Numbers measured on the test machine, only as estimates; measure again on
your own machine.

| Item | Estimate |
| --- | --- |
| Data generation speed | about 40–50 positions/s per worker with `--nodes 5000` (roughly inversely proportional to the node count) |
| 16-core machine, `--nodes 5000` | about 2–2.5 million positions/hour |
| Text file | about 100 bytes/position (10 million ≈ 1 GB); `.zst`: about 15 bytes/position |
| Prepared data | about 264 bytes/position (10 million ≈ 2.6 GB) |
| `prepare_data.py` | about 15,000 lines/s per CPU core (`threats`: about 6,500): 50 million positions take a few minutes on 16 cores, much less than generating them |
| Checkpoint | about 190 MB (`att`: 750 MB, `threats`: 1.3 GB) per epoch |
| `.nnue` file | about 8 MB (`att`: larger, `threats`: up to about 60 MB), zstd compressed |

Training speed depends on the GPU; run a small epoch first
(`--epoch-size 1000000`) to measure it before planning.

## 10. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `verify.py` reports `MISMATCH` | The engine and the trainer do not use the same architecture. Use the engine and the trainer of the same Maskfish commit and rebuild the engine; for the other architectures remember `make clean`, the matching `make` options and `--arch` ([section 7.1](#71-overview)) |
| The engine reports `network file ... was not loaded` | Wrong architecture (e.g. an `att` / `threats` / `-l2` network on a regular engine or the other way round: every `make` option of [section 7.1](#71-overview) must match the network's architecture), or the file is not a zstd compressed `.nnue` exported by the trainer |
| `no threat data (prepare the data again ...)` | The data was prepared by an old version of `prepare_data.py`; run it again with `--arch threats` |
| The engine reports `network file ... was not loaded` for a network trained before the engine used 8-bit weights | The file has the old version (16-bit HalfKA weights): convert it with `serialize.py --upgrade`, see [section 11.1](#111-file-format-change-8-bit-weights) |
| `the train data has no material / rule40` | The data was prepared before the loss used the `evaluate()` scaling: run `prepare_data.py` again on the text files (or train with `--no-eval-scale`) |
| `the train data has no se / allmat` | `--eval-scaling alignment` needs data prepared by the current `prepare_data.py`: prepare it again from the text files |
| `the train data has no sed` | `--eval-scaling correlation` (and `--no-psqt` with `complexity`) need data prepared by the current `prepare_data.py`: prepare it again from the text files |
| `SCALE MISMATCH` with a network trained with `--eval-scaling alignment` / `correlation` | Run `verify.py` (and the engine) with the same `--eval-scaling` / `EvalScaling` value |
| `SCALE MISMATCH` in `verify.py` | `eval_scale.py` no longer matches `src/evaluate.cpp` (constants or material): update it |
| `incompatible network for arch ...` | A network of the wrong architecture was given to `--init-net` / `serialize.py`; the message gives the file's real architecture |
| `prepare_data.py` runs out of RAM | Lower `--shard-size` (e.g. 500000) |
| `train.py` runs out of GPU memory | Lower `--batch-size` (e.g. 8192); the `att` architecture needs more |
| The loss becomes `nan` or jumps up | Lower `--lr` (e.g. 5e-4), check the data for abnormal scores (lower `--max-score`); try `--no-factorize` to see whether factorization is the cause |
| `warning: optimizer state not restored` with `--resume` | The checkpoint was made by an old trainer version (a single parameter group): the model is loaded, only the optimizer state is created anew |
| Slow training, GPU idle | The data is on a slow disk: move it to an SSD; increase `--batch-size` or `--prefetch`; compare the `pos/s` of the epoch lines |
| `torchrun`: `--batch-size N is not a multiple of the K processes` | Choose a batch size divisible by `--nproc_per_node` |
| `torchrun` hangs at the start, or NCCL errors | Check that every process sees its GPU (`CUDA_VISIBLE_DEVICES`), try `NCCL_P2P_DISABLE=1`; one GPU (plain `python train.py`) is always a fallback |
| `watch_nets.py` measures nothing | The networks are younger than 2 seconds, not multiples of `--every`, or already measured with the same `--ref` and limit (see `watch.json`) |
| `gen_data.py` was interrupted | Run the same command again: it plays only the missing games, then merges the `.partN` files |
| The ref-engine prints `No such option: RawScore` | Normal: other engines do not have that option; their scores must be rescaled with `--score-scale` ([section 6.2](#62-rescale-the-scores-and-train)) |
| Two data generation runs are identical | `--seed` was not set to different values |
| The SPRT never ends | The two sides are nearly equal; stop it and consider it no improvement, or use a narrower range such as `--sprt 0 3` with many pairs |

## 11. When the engine's network architecture changes

`jieqi_features.py` (the feature sets, `ARCHS`), `model.py` and the constants
at the top of `serialize.py` (hash, sizes, quantization factors) describe the
engine; `eval_scale.py` describes `Eval::evaluate()` (`src/evaluate.cpp`) and
must follow any change of its constants or of `major_material()`. After any
change in Maskfish's `src/nnue`, update these files and run `verify.py` for
every architecture until it is bit-exact. Prepared data must be created again;
text data can still be used.

For the architectures with threats, the "Threat features" part of
`jieqi_features.py` must list the features in exactly the order of
Maskfish's `src/nnue/features/jieqi_threats.cpp` (the engine checks the number of
features at startup and stops if it is not 92,711).

### 11.1. File format change: 8-bit weights

The feature transformer weights (HalfKA features) are now 8-bit integers like
the threat weights (they were 16-bit), so the file version changed from
`0x7AF32F20` to `0x7AF32F21`; the new engine does not read old files. To
convert an old network:

```
python serialize.py --upgrade old.nnue new.nnue
```

The command reports the number of weights outside ±127 (i.e. ±1.0 before
quantization) that were clipped to the bounds. If it is 0, the new engine
evaluates the new file **exactly** like the old engine evaluated the old file
(bench gives the same node count); otherwise the network changes slightly and
should be checked with a match before use. `.pt` checkpoints need no
conversion: `train.py` clips the weights into the new range
(`model.clip_weights()`, including the virtual features of the factorization)
at the first step, and `serialize.py` exports them as usual.

### 11.2. Optional: blocks without LEB128

The biases and the PSQT weights of the feature transformer are written as
LEB128 blocks (with the `COMPRESSED_LEB128` marker). Like Pikafish ("Remove
leb128"), they can also be written as plain little-endian integers: the whole
file is compressed with zstd anyway, so the size hardly changes and loading is
a little simpler. The engine reads both forms (a block without the marker is
read as plain integers), so this is not a new file version, but **older engines
only read LEB128 blocks**: keep the default for networks that older builds must
load.

```
python serialize.py net.nnue net-raw.nnue --no-leb128     # rewrite a network
python serialize.py runs/x/epoch_40.pt net.nnue --no-leb128  # export a checkpoint
```

The evaluation is the same (bench gives the same node count).