# Jieqi NNUE trainer

A PyTorch toolkit to train NNUE networks for the Jieqi engine
[Maskfish](https://github.com/nguyenhungitc/maskfish), from scratch (no network exists yet) to a usable network, and to test changes
with SPRT matches.

The networks are trained for exactly the engine's architecture (`src/nnue` of
Maskfish):
HalfKAv2_hm features for Jieqi (7560 inputs per side, including dark pieces
and the pool), a 2048 feature transformer with 16 PSQT buckets, and 16 layer
stacks 2048 → 16 → 32 → 1. The engine reads an exported network **bit-exactly**
as the trainer computes it (`verify.py` checks this).

Part of the design follows the trainers of AB-JChess and
[nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch) (Stockfish); the
code is written specifically for this engine's feature set and file format.

## Contents

1. [Components](#1-components)
2. [Requirements](#2-requirements)
3. [Setup](#3-setup)
4. [Training workflow in detail](#4-training-workflow-in-detail)
5. [Suggested schedule for the generations](#5-suggested-schedule-for-the-generations)
6. [Using other engines: AB-JChess and the old branch](#6-using-other-engines-ab-jchess-and-the-old-branch)
7. [Experiments once a base network exists](#7-experiments-once-a-base-network-exists)
8. [Data formats](#8-data-formats)
9. [Disk space and time](#9-disk-space-and-time)
10. [Troubleshooting](#10-troubleshooting)
11. [When the engine's network architecture changes](#11-when-the-engines-network-architecture-changes)

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
| `ftperm.py` | Reorders the feature transformer neurons of a network so that the engine evaluates it faster (same evaluations) |
| `verify.py` | Compares the engine's evaluation with the trainer's; must be bit-exact |
| `match.py` | Matches between two networks / option sets / builds, with SPRT |
| `fit_scaling.py` | Fits the score → win probability constant from self-play data |
| `ranger21.py` | Ranger21 optimizer (`train.py --optimizer ranger21`), from AB-JChess |
| `chance_tables.py` | Score ↔ mass tables for the engine option `ChanceModel Tables` (fitted to self-play data, logistic, or copied from an AB-JChess network) |
| `rescore.py` | Has another engine (e.g. AB-JChess) re-score positions |
| `serialize.py` | Writes / reads `.nnue` files, converts `.nnue` into a checkpoint |
| `jieqi_features.py` | Input features, a copy of Maskfish's `src/nnue/features/half_ka_v2_hm` |
| `model.py` | PyTorch model and weight clipping |
| `eval_scale.py` | The scaling of the engine's `evaluate()`, applied to the network output in the loss |
| `jieqi_uci.py` | Helpers to drive the engine over UCI |

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
- The build fetches the network of the hidden ref `refs/assets/net` to
  `src/maskfish.nnue` (`make net`) and embeds it in the binary. The scripts
  do not use it: every script passes the Jieqi network with the `EvalFile`
  option. A trained network is published with `scripts/push_net.sh` (from
  `net/maskfish.nnue`, see the README).

### 3.2. Install Python

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy zstandard
pip install torch                # see the note below
```

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

No Jieqi network exists yet, so training is a loop, each round being a
"generation":

```
current net ──► generate data ──► prepare ──► train ──► verify ──► SPRT match
     ▲                                                                  │
     └─────────────── the new net becomes the current one if it wins ◄──┘
```

Generation 0 starts from a material-only network. If AB-JChess is available,
generation 0 can be replaced by data generated by AB, for a much stronger
first network ([section 6](#6-using-other-engines-ab-jchess-and-the-old-branch)).
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
`nets/gen0.nnue.pt` (the matching checkpoint).

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
| `--workers` | cores − 1 | Number of parallel processes |
| `--seed` | system time | **Set it explicitly**, and differently for every run / machine |
| `--random-plies` / `--random-plies-min` | 8 / 2 | Random moves at the start of a game (not written), for opening variety |
| `--max-plies` | 400 | Games longer than this (in plies) are drawn |
| `--adjudicate` | 3000 | A game is won when \|score\| ≥ this threshold for 8 plies in a row |
| `--opt` | | Extra UCI options for the engine, e.g. `--opt ChanceScaling=400` |
| `--full-info` | off | Tell both sides the identity of a captured dark piece (by default only the capturer knows it, as in a real game) |

Good to know:

- Every line reads `fen | score | result`. The score is in internal units,
  for the side to move; the result is 1 / 0 / −1 (win / draw / loss) for the
  side to move. Positions in check and positions whose best move is a capture
  are skipped.
- The FEN is the side to move's view: the identity of a dark piece it lost
  is unknown to it (the piece stays in its pool), as when the engine plays.
- The safe reveal bonus (`RevealQuiet*` options) is disabled during data
  generation: it is a preference of the search between moves, and would
  otherwise be part of the scores the network learns. `--opt` can set it.
- A game ends on the third occurrence of a position: perpetual check loses,
  other repetitions are drawn.
- The identities of the dark pieces are really drawn at random for every
  game. Since one engine plays both sides, both sides know the identity of
  every captured dark piece.
- Games are drawn after 80 plies without a capture (the engine's 40-move
  rule).
- Every worker writes its own `.partN` file; they are merged when all are
  done. After an interruption the `.partN` files remain and can be used (each
  one only holds finished games).
- **Several machines:** run `gen_data.py` on each machine with different
  `--seed` and `--out`, then copy the files back. They can be prepared
  together (several files given to `prepare_data.py`) or separately into
  different prefixes; `train.py` accepts several prefixes.
- Keep the text files: they are needed to prepare the data again when the
  architecture changes (e.g. `att`) and to run `fit_scaling.py`.
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
| `--score-scale` | 1.0 | Multiply the scores by this factor, to convert another engine's scores to internal units ([section 6](#6-using-other-engines-ab-jchess-and-the-old-branch)) |
| `--workers` | cores | Number of FEN parsing processes |
| `--arch` | `default` | `att` for the experimental architecture ([section 7.2](#72-the-att-architecture)) |
| `--jieqi-material` | off | The data was generated with the engine option `JieqiMaterial=true` (store the material `evaluate()` then uses) |

Every shard is shuffled before it is written; during training every batch
comes from a random shard, so the data is mixed across shards.

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
| `--scaling` | 361 | Score → probability constant ([step 7](#step-7-calibrate-the-score--probability-constant)) |
| `--lambda-end` | | Lambda of the last epoch: lambda goes linearly from `--lambda` to it (AB-JChess: 1.0 → 0.8) |
| `--score-scaling` | `--scaling` | Score → probability constant of the data scores only (AB-JChess: 410, with `--scaling 361`) |
| `--loss-power` | 2.5 | Exponent of the loss (AB-JChess: 2) |
| `--layers`, `--ctx`, `--no-psqt` | | Architecture options of the model (sections 7.4 to 7.6) |
| `--init-net` | | Start from a `.nnue` file (new optimizer) |
| `--resume` | | Continue an interrupted run from a checkpoint (optimizer and epoch count included) |
| `--device` | `cuda` if available | `cpu` to force the CPU |
| `--threads` | 0 | Number of CPU threads for PyTorch (0 = default) |
| `--no-factorize` | | Disable the virtual features (on by default, see below) |
| `--factorize-kb` | | Add virtual features per king bucket (used by AB-JChess; off by default) |
| `--no-l1-fact` | | Disable the fc0 part shared by the layer stacks (on by default) |
| `--output-lr-scale` | 0.1 | Learning rate of the output layer, relative to `--lr` |
| `--optimizer` | `adam` | `ranger21`: the optimizer of AB-JChess and nnue-pytorch (AdamW with lookahead, `ranger21.py`); the learning rate still follows `--lr` and `--gamma`. A run resumed with another optimizer starts a new one at the current learning rate |
| `--stack-balance` | 0 (off) | Balance the loss across layer stacks; the value is the cap (e.g. 8) |
| `--checkpoint-every` | 1 | Write `epoch_K.pt` only every K epochs (and at the last one); `last.pt` and the `.nnue` files are always written |
| `--eval-scaling` | `complexity` | The engine's `EvalScaling` option the network will play with: `complexity` or `alignment` ([section 7.13](#713-evaluation-scaling-evalscaling)) |
| `--no-eval-scale` | | Compare the raw network output with the data scores (old behaviour, see below) |
| `--qat-start` | 0 (off) | Quantization-aware training from this epoch on ([section 7.10](#710-quantization-aware-training-qat-start)) |
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

**Training-only techniques** (ideas from nnue-pytorch and AB-JChess). They do
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
  average weight of every stack is 1. With `CAP=8` (the value AB uses) and 16
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
  `--depth` or `--movetime`). Every engine uses 1 thread; `--workers` pairs
  of games run in parallel.

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
does not weaken the engine before making it the default.

The same data can also give lookup tables for the engine option
`ChanceModel Tables` (AB-JChess's averaging of chance nodes in "mass" space,
not tested for Maskfish yet):

```
python chance_tables.py fit data/gen1.txt --out tables/gen1
python match.py ... --opt-a ChanceModel=Tables ChanceTables=$PWD/tables/gen1
```

`chance_tables.py logistic --scale <s>` writes tables equivalent to the default
averaging (a reference), and `chance_tables.py extract <AB-JChess .nnue>`
copies the tables of an AB-JChess network (in the score units of its
networks: they fit Maskfish's only roughly).

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

## 6. Using other engines: AB-JChess and the old branch

Two other complete Jieqi engines work with these tools (checked: the `d`
command, `go perft`, the notation of moves with the revealed identity, and
captures of dark pieces without identity are all compatible):

- **AB-JChess** (https://github.com/lxsgx23/AB-JChess): has a trained NNUE
  network, published under `Releases` (the file `0.2b_bmi2.zip` contains
  `AB-JChess.exe` for Windows and the network `abjchess-20260911.nnue`). Much
  stronger than the material-only network, so use it as a **"teacher"** to
  generate data (distillation) and as a **yardstick**.
- **The `old` branch** of the original Jieqi repository: hand-written evaluation, strength
  not measured, and a **color-biased** evaluation (about +375 cp for Red in the
  opening position at 5000 nodes). Do not use it as a teacher (the network
  would learn the bias); use it as a **reference opponent**.

These two engines have no `RawScore` option: they print cp normalized to each
engine's own scale. Data scored by them must therefore be **rescaled** before
training (step 6.3).

### 6.1. Build AB-JChess

```
git clone https://github.com/lxsgx23/AB-JChess
# download 0.2b_bmi2.zip from the Releases page, copy abjchess-20260911.nnue into AB-JChess/src
make -C AB-JChess/src -j build ARCH=x86-64-avx2
```

On Windows, `AB-JChess.exe` from the zip file can be used directly. When
calling AB, add `ChasingRule=github` so that AB uses chasing rules close to
this engine's (AB's default is SkyRule).

### 6.2. Option A: AB plays itself to generate data

Instead of steps 1–2 of the first generation (material network):

```
python gen_data.py --engine AB-JChess/src/AB-JChess \
    --net AB-JChess/src/abjchess-20260911.nnue --opt ChasingRule=github \
    --games 50000 --nodes 5000 --adjudicate 1000 --workers 15 \
    --seed 100 --out data/ab0.txt
```

- `--adjudicate` is in AB's cp scale. 1000 is only a starting value: once
  data exists, look at the table of `fit_scaling.py` (step 6.3) and choose the
  score at which the observed win rate is close to 100%.
- AB is slower than this engine (much larger network): roughly 7 positions
  per second per worker in a short run (including network loading).

### 6.3. Rescale the scores and train

```
python fit_scaling.py data/ab0.txt
# suppose it prints "fitted scale: 250.0" -> factor = 361 / 250 = 1.444
python prepare_data.py data/ab0.txt --out data/train_ab0 --score-scale 1.444
python prepare_data.py data/ab_val.txt --out data/val_ab --score-scale 1.444
python train.py data/train_ab0 --val data/val_ab --out runs/ab1 \
    --epochs 40 --epoch-size 10000000
```

`361` is the trainer's default constant (`--scaling`) and close to the one
this engine uses; if a constant has been fitted for this engine, use it
instead of 361. The rescaling factor must be fitted again for every AB setup
(network, node count).

Rescaled data can be combined with this engine's self-play data: prepare each
part separately (each with its own `--score-scale`) and give several prefixes
to `train.py`.

### 6.4. Option B: re-score self-play data with AB

Once this engine has a network, its self-play data is varied and cheap to
generate, while AB scores more accurately. Combine both with `rescore.py`:

```
python rescore.py data/gen3.txt --out data/gen3_ab.txt \
    --engine AB-JChess/src/AB-JChess --net AB-JChess/src/abjchess-20260911.nnue \
    --opt ChasingRule=github --nodes 20000 --workers 15
python fit_scaling.py data/gen3_ab.txt
python prepare_data.py data/gen3_ab.txt --out data/train3_ab --score-scale <361/s>
```

- Every position keeps its original game result and gets AB's search score
  for the side to move.
- Positions where AB reports a mate, or whose best move is a capture, are
  dropped.
- After an interruption, run the same command again, adding `--resume`.

### 6.5. Measuring strength against AB and the old branch

```
# against AB (use --movetime: the node counts of the two engines are not comparable)
python match.py --engine-a ../src/maskfish --net-a nets/genN.nnue \
    --engine-b AB-JChess/src/AB-JChess --net-b AB-JChess/src/abjchess-20260911.nnue \
    --opt-b ChasingRule=github --movetime 1000 --workers 15

# against the old branch (no --net-b needed)
python match.py --engine-a ../src/maskfish --net-a nets/genN.nnue \
    --engine-b ../../jieqi-old/src/PikaJieQi --movetime 1000 --workers 15
```

`--sprt` is not needed to estimate the Elo: leave it out and set `--pairs`.
Since the two sides use different score scales, the adjudication of
`match.py` hardly ever triggers; games end by mate, 80 plies without a
capture, or `--max-plies` (slower but still correct).

Suggested path: train the first few generations on AB's data (option A),
then self-play with this engine and re-score with AB (option B), and switch
to pure self-play once this engine's network gets close to AB in the
measuring matches.

### 6.6. About usage rights

The source code of AB-JChess is GPL-3; its release notes say it is free and
must not be resold. A network trained on AB's evaluations can be considered a
derived work: credit AB-JChess when publishing such a network and consider
asking its authors.

## 7. Experiments once a base network exists

### 7.1. Search options

The engine has experimental options (see the Maskfish README); test them one at a
time with `match.py`, with the same network:

```
python match.py --engine ../src/maskfish --net-a nets/genN.nnue \
    --opt-a TTPenalize=true --nodes 20000 --workers 15 --sprt 0 3
```

Options worth testing: `ProbCutDarkCapture`, `TTPenalize`,
`ChanceBranchDepth`, `JieqiMaterial` (on/off, off by default);
`ChanceRiskAversion` (e.g. 10, 20, −10); `HiddenCaptureModel=false` and
`RevealQuietBase=0 RevealQuietSafety=0 RevealQuietHighValue=0` (to re-check
the two features that are on by default). An option that wins its SPRT can
be made the default in Maskfish's `src/engine.cpp`.

### 7.2. The `att` architecture

An experimental architecture with attack buckets and mid mirror (see the
Maskfish README). `att` networks are not compatible with the regular build.

**Build a separate engine** (`make clean` is required when switching between
the two architectures, since the object files are not rebuilt
automatically):

```
cd ../src
make clean && make -j build ARCH=x86-64-avx2 nnue=att && mv maskfish maskfish-att
make clean && make -j build ARCH=x86-64-avx2
cd ../trainer
python verify.py --engine ../src/maskfish-att --arch att
```

**Train** on the same text data, prepared again with `--arch att`:

```
python init_net.py nets/att0.nnue --arch att          # or skip it, see below
python prepare_data.py data/gen0.txt data/gen1.txt --out data/train_att --arch att
python prepare_data.py data/val1.txt --out data/val_att --arch att
python train.py data/train_att --val data/val_att --out runs/att1 \
    --epochs 40 --epoch-size 10000000
python verify.py --engine ../src/maskfish-att --net runs/att1/epoch_40.nnue
```

`train.py` takes the architecture from the data; `--init-net`, `--resume`,
`verify.py` and `serialize.py` detect the architecture of a file and report
an error on a mismatch. `init_net.py` is only needed to generate data with the
`att` engine from scratch: the text data of the regular engine can be shared.

**Compare** with a regular network trained on **the same data** (to be fair):

```
python match.py --engine-a ../src/maskfish-att --net-a runs/att1/epoch_40.nnue \
    --engine-b ../src/maskfish --net-b nets/genN.nnue \
    --nodes 20000 --workers 15 --sprt 0 5
```

An `att` network can be slower (the accumulator is refreshed more often), so
comparing with `--movetime` is fairer than with `--nodes`.

### 7.3. The `threats` architecture (threat features)

The "full threats" idea of Stockfish / Pikafish, adapted to Jieqi: besides
the HalfKA features, the network has one feature for every (attacking piece,
from square, to square, attacked piece) present on the board, including
defended pieces of its own side:

- pieces are split into 16 classes: every kind of revealed piece of each
  side, plus "dark piece" of each side; a dark piece attacks like the kind of
  piece of its starting square (a dark advisor only inside the palace);
- threats of the kings, of advisors to their own king, and checks by rooks,
  cannons and knights are not used (as in Pikafish);
- 92,711 features in total, usually 10–40 active at a time per side;
- the feature transformer has 1024 outputs (instead of 2048) to offset the
  cost.

`threats` networks (and `att-threats` = `att` + threats) are **not
compatible** with the other builds. Build separately (remember `make clean`):

```
cd ../src
make clean && make -j build ARCH=x86-64-avx2 threats=yes && mv maskfish maskfish-thr
make clean && make -j build ARCH=x86-64-avx2 threats=yes nnue=att && mv maskfish maskfish-att-thr
make clean && make -j build ARCH=x86-64-avx2
cd ../trainer
python verify.py --engine ../src/maskfish-thr --arch threats
python verify.py --engine ../src/maskfish-att-thr --arch att-threats
```

**Train** as for `att`: the same text data, prepared again with
`--arch threats` (data prepared for other architectures cannot be used):

```
python init_net.py nets/thr0.nnue --arch threats      # only to generate data from scratch
python prepare_data.py data/gen0.txt data/gen1.txt --out data/train_thr --arch threats
python prepare_data.py data/val1.txt --out data/val_thr --arch threats
python train.py data/train_thr --val data/val_thr --out runs/thr1 \
    --epochs 40 --epoch-size 10000000 --checkpoint-every 10
python verify.py --engine ../src/maskfish-thr --net runs/thr1/epoch_40.nnue \
    --checkpoint runs/thr1/epoch_40.pt
```

The threat weights start at 0 (a new network computes exactly the same as
without threats) and are kept within the int8 range during training.
`--init-net` only accepts a network of the same architecture; a `default`
network (2048 outputs) cannot be converted to `threats`.

**Costs**, to keep in mind when comparing:

| | `default` | `threats` |
| --- | --- | --- |
| Engine speed (bench, 1 thread, AVX2) | 100 % | about 65–70 % |
| Engine memory for the network | about 15 MB | about 105 MB (int8 threat weights ≈ 96 MB) |
| `.nnue` file (zstd compressed) | about 8 MB | up to about 60 MB once trained |
| Prepared data | about 260 bytes/position | about 350 bytes/position |
| `prepare_data.py` | about 15,000 lines/s per core | about 6,000 lines/s per core |
| Checkpoint | about 190 MB | about 1.3 GB |

(Measured before the threat updates were paired, Pikafish's "Pair int8 weight
rows in accumulator updates": with SSSE3 or better, the threat accumulator is
now updated two weight rows at a time, about +15 % nps for `threats=yes` on
AVX2, measured with a random network, which narrows the gap.)

Since the engine is about a third slower, compare with a regular network
trained on the same data using `--movetime` (or at least check again with
`--movetime` after winning with `--nodes`):

```
python match.py --engine-a ../src/maskfish-thr --net-a runs/thr1/epoch_40.nnue \
    --engine-b ../src/maskfish --net-b nets/genN.nnue \
    --movetime 100 --workers 15 --sprt 0 5
```

### 7.4. New layer stacks (`layers=2`)

After Pikafish's "Update nnue-architecture" (July 2026). Compared with the
current layer stacks (`fc0`: L1 → 15 + 1, `fc1`: 30 → 32, `fc2`: 32 → 1):

- `fc0` has 32 outputs, all 32 going through both activations (squared and
  clipped);
- `fc1` (64 → 32) also gets a squared activation;
- `fc2` takes all 128 activations (of `fc0` and `fc1`), not only those of
  `fc1`;
- the term added directly to the result is the difference of the last two
  outputs of `fc0` (`fc0[30] − fc0[31]`) instead of a separate 16th output.

The quantization stays that of the current layer stacks by default; the
quantization of current Pikafish is an option of its own (`make quant=2`,
[section 7.12](#712-quantization-of-current-pikafish-make-quant2)).

The new layer stacks work with every feature set: the architecture names get
the suffix `-l2` (`default-l2`, `att-l2`, `threats-l2`, `att-threats-l2`). The
input features do not change, so **prepared data can be shared**.

```
cd ../src
make clean && make -j build ARCH=x86-64-avx2 layers=2 && mv maskfish maskfish-l2
make clean && make -j build ARCH=x86-64-avx2 layers=2 threats=yes && mv maskfish maskfish-thr-l2
make clean && make -j build ARCH=x86-64-avx2
cd ../trainer
python verify.py --engine ../src/maskfish-l2 --arch default-l2
python train.py data/train --val data/val --out runs/l2 --layers 2 \
    --epochs 40 --epoch-size 10000000 --init-net nets/genN.nnue
python verify.py --engine ../src/maskfish-l2 --net runs/l2/epoch_40.nnue \
    --checkpoint runs/l2/epoch_40.pt
```

`--init-net` accepts an `-l2` network with the same feature set, or a network
with the old layer stacks and the same feature set: then only the feature
transformer is used (the largest part, the slowest to train), and the layer
stacks are trained from scratch. `--resume` needs the same architecture.

A quick measurement on the development machine (small data, 166,000
self-play positions, CPU, 15 epochs, so only indicative): validation loss
0.00188 versus 0.00200 for the current layer stacks; the engine is about 12 %
(`default`) and 8 % (`threats`) slower because `fc0` is twice as large. (Since
then the two activations of each layer are computed together with AVX2 /
AVX-512, Pikafish's "paired activations": about +3 % nps for `layers=2` on
AVX2, measured with a random network, within the noise of the measure;
Pikafish measured +2.5 %.)
Compare with `--movetime`:

```
python match.py --engine-a ../src/maskfish-l2 --net-a runs/l2/epoch_40.nnue \
    --engine-b ../src/maskfish --net-b nets/genN.nnue \
    --movetime 100 --workers 15 --sprt 0 5
```

### 7.5. Networks without PSQT (`psqt=no`)

The engine built with `make psqt=no` reads networks without the PSQT part of
the feature transformer (as Pikafish's SFNNv17 and AB-JChess's networks): the
evaluation is the layer stack output alone, the material goes through the
layer stacks too. The architecture names get the suffix `-nopsqt` (after
`-l2`: `default-nopsqt`, `att-l2-nopsqt`, ...). The input features do not
change, so prepared data can be shared.

A network without PSQT cannot be a material-only starting network
(`init_net.py` refuses): start from the feature transformer of a network with
PSQT and the same feature set.

```
cd ../src
make clean && make -j build ARCH=x86-64-avx2 psqt=no && mv maskfish maskfish-nopsqt
make clean && make -j build ARCH=x86-64-avx2
cd ../trainer
python verify.py --engine ../src/maskfish-nopsqt --arch default-nopsqt
python train.py data/train --val data/val --out runs/nopsqt --no-psqt \
    --epochs 40 --epoch-size 10000000 --init-net nets/genN.nnue
python match.py --engine-a ../src/maskfish-nopsqt --net-a runs/nopsqt/epoch_40.nnue \
    --engine-b ../src/maskfish --net-b nets/genN.nnue --movetime 100 --sprt 0 5
```

### 7.6. Inventory context (`ctx=yes`)

From AB-JChess: the pool counts are already input features, but only through
the feature transformer, whose output does not change much when one dark
piece is revealed or captured. The context gives the layer stacks the
uncertainty of the position directly: 16 values (see
`jieqi_features.context_inputs()`), from the side to move, go through a small
layer (`ctx_tower`, 16 → 16, clipped ReLU, shared by the layer stacks) whose
outputs are 16 more inputs of `fc0` in every layer stack. The trainer
recovers the 16 values from the features of the side to move, so **prepared
data can be shared**; `verify.py` checks that they are the ones the engine
computes.

The architecture names get the suffix `-ctx` (after `-l2`, before
`-nopsqt`: `default-ctx`, `att-l2-ctx`, `threats-ctx-nopsqt`, ...).

```
cd ../src
make clean && make -j build ARCH=x86-64-avx2 ctx=yes && mv maskfish maskfish-ctx
make clean && make -j build ARCH=x86-64-avx2
cd ../trainer
python verify.py --engine ../src/maskfish-ctx --arch default-ctx
python train.py data/train --val data/val --out runs/ctx --ctx \
    --epochs 40 --epoch-size 10000000 --init-net nets/genN.nnue
python match.py --engine-a ../src/maskfish-ctx --net-a runs/ctx/epoch_40.nnue \
    --engine-b ../src/maskfish --net-b nets/genN.nnue --movetime 100 --sprt 0 5
```

`--init-net` with a network without context takes its feature transformer;
the layer stacks and the context layer are trained from scratch.

### 7.7. Layer stack selection of AB-JChess (`lsblend=yes`)

The layer stack of a position is chosen by its number of pieces. AB-JChess
raises that bucket by an "uncertainty phase": +1 to +4 for the number of dark
pieces on the board (1, 5, 9, 17 or more), +1 to +3 for the number of rooks,
cannons and knights left in the pools (2, 5, 9 or more), at most 15. And it
interpolates between the chosen layer stack and the next one, with a weight
given by the position of the three counts within their intervals (the
evaluation then changes smoothly with them instead of jumping at the
boundaries of the buckets). See `jieqi_features.layer_stack_selection()`;
about half of the positions use two layer stacks, which costs the engine
about 10 % of its speed.

The architecture names get the suffix `-blend`, first (`default-blend`,
`att-blend-l2-ctx`, ...). **The prepared data changes** (layer stack and
interpolation weight of each position): prepare it with `--arch
default-blend` (or `att-blend`, ...); the other options (`--layers 2`,
`--ctx`, `--no-psqt`) are given to `train.py`.

```
cd ../src
make clean && make -j build ARCH=x86-64-avx2 lsblend=yes && mv maskfish maskfish-blend
make clean && make -j build ARCH=x86-64-avx2
cd ../trainer
python verify.py --engine ../src/maskfish-blend --arch default-blend
python prepare_data.py data/gen*.txt --out data/train-blend --arch default-blend
python train.py data/train-blend --out runs/blend --epochs 40 --epoch-size 10000000 \
    --init-net nets/genN.nnue
```

`--init-net` with a network without `-blend` (same feature set) takes its
feature transformer. `--stack-balance` weights the layer stacks of the new
selection.

### 7.8. Meta features of AB-JChess (`meta=yes`)

88 more input features per feature transformer bucket, of which 9 are active
for each side, one per group (see `jieqi_features.meta_offsets()`): from the
side of the perspective, the numbers of rooks, cannons and knights in its
pool and in the opponent's (up to 7), the numbers of pieces in both pools (up
to 7), the number of dark pieces on the board (8 buckets), the numbers of
dark pieces of each side captured without being identified (0..15), and the
numbers of revealed pieces of each side attacked by revealed enemy pieces
(up to 7). The pool counts are already features (repeated once per piece),
but only linearly: the one-hot groups let the network react to them
non-linearly at no cost; the attacked pieces are new information (a small
summary of what the `threats` architecture gives in full).

The engine computes them at each evaluation and keeps them in the HalfKA
accumulator (about 30% slower). The architecture names get the suffix
`-meta`, first (`default-meta`, `att-meta-blend`, ...). **The input features
change**: prepare the data with `--arch default-meta` (the feature arrays have
80 columns instead of 64), and `--init-net` needs a network with them.

```
cd ../src
make clean && make -j build ARCH=x86-64-avx2 meta=yes && mv maskfish maskfish-meta
make clean && make -j build ARCH=x86-64-avx2
cd ../trainer
python verify.py --engine ../src/maskfish-meta --arch default-meta
python prepare_data.py data/gen*.txt --out data/train-meta --arch default-meta
python train.py data/train-meta --out runs/meta --epochs 40 --epoch-size 10000000
```

### 7.9. Starting a new architecture from an existing network

A new architecture does not have to start from a material-only network:

- `-meta`, `-ctx` (and both) only add inputs: `serialize.py --convert` turns a
  network without them into one with zero weights for them, which evaluates
  every position exactly as the original (checked with the released network).
  It is a generation 0 as strong as the current one:

  ```
  python serialize.py --convert default-meta-ctx nets/gen5.nnue nets/default-meta-ctx-gen0.nnue
  ```

- `-blend` can also be added by `--convert`, keeping the layer stacks; the
  evaluation changes (the layer stacks are selected differently), so it is
  only a starting point for training.
- `-l2` and `-nopsqt` change the layer stacks or the output: they cannot be
  converted. `train.py --init-net` takes the feature transformer of a network
  with the same feature set (also without `-meta`: zero rows are added), and
  the layer stacks are trained. Train them on the data of the current line
  (prepared for the new feature set) and compare with `match.py
  --engine-a ... --engine-b ...` before using them as a generation 0.

`train.py --init-net` converts automatically when it can (the whole network is
then used), otherwise it takes the feature transformer only; it says which.

### 7.10. Quantization-aware training (`--qat-start`)

The engine plays with the integer network that `serialize.py` exports, not
with the float model: weights rounded to int8 / int16, the accumulator
products shifted, the activations saturated, the divisions truncated. The
difference is what `verify.py --checkpoint` reports as "largest
float/quantized difference" (a few dozen internal units for a trained
network, a few hundred for a random one), and the loss printed by the trainer
is the loss of the float model, not of the network that plays.

With `--qat-start E`, from epoch E on the forward pass is the engine's
integer computation (`model.forward_parts_qat()`): the weights rounded
exactly as at export (the virtual features and the shared fc0 part folded
in), the integer arithmetic of the feature transformer, the layer stacks and
`evaluate()`. The roundings let the gradient through unchanged
(straight-through estimator), so the float weights keep learning, while the
loss is the loss of the exported network. Nothing changes in the file format
or in the engine; every architecture is supported (`-l2`, `threats`,
`-ctx`, `-blend`, `-meta`, `-nopsqt`).

```
# float training, then the last 5 of 40 epochs quantization-aware
python train.py data/train --val data/val --out runs/gen2 --epochs 40 \
    --epoch-size 10000000 --init-net nets/gen1.nnue --qat-start 36 --val-qat

# or a short QAT fine-tuning of an existing run
python train.py data/train --val data/val --out runs/gen2 --epochs 45 \
    --resume runs/gen2/last.pt --qat-start 41 --lr 2e-4
```

- `--val-qat` prints, before `--qat-start`, the validation loss of the
  quantized network as well (`quantized net loss`), so that the effect of QAT
  can be seen; during QAT, `val loss` is already the quantized network's
  (the epoch line says `qat`). Both go to `metrics.jsonl` /
  TensorBoard (`quantized_val_loss`).
- `verify.py --checkpoint` checks that the QAT forward pass of the
  checkpoint gives the engine's `evalraw` and `evaluate()` values (line
  `quantization-aware forward: N/N positions equal to the engine`).
- Cost: QAT builds a rounded copy of the feature transformer weights at
  every step (more GPU memory: about 3 × 62 MB for `default`, 3 × 385 MB for
  `threats`; `scripts/pipeline.sh` probes the batch size with QAT when
  `QAT_START` is set), but folds the virtual features into it, so it does one
  embedding lookup instead of two: on the CPU test machine a QAT step was
  even faster (5.4k instead of 3.7k positions/s). Compare the `pos/s` of the
  epoch lines on your GPU.
- Measured so far only at a tiny scale (92,000 positions, 7 epochs, CPU): the
  quantized network's validation loss was within 0.5 % of the float model's
  in every epoch, and QAT for the last 3 epochs gave the same quantized loss
  as float training (0.006771 both), QAT from the first epoch 0.006597, all
  within the noise of such a run. Whether QAT helps at real scale is open:
  try it on the last few epochs of a generation and decide with `match.py`
  (SPRT), not with the loss.

### 7.11. Engine speed: neuron order (`ftperm.py`)

The first layer of the layer stacks (fc0) reads the 0..127 outputs of the
feature transformer in chunks of 4 and skips the chunks that are entirely
zero (`AffineTransformSparseInput`). The order of the neurons is arbitrary,
so `ftperm.py` (after nnue-pytorch's script of the same name) puts neurons
that are often zero at the same time in the same chunk:

```
python ftperm.py nets/gen3.nnue nets/gen3.nnue --data data/gen3.txt.zst
python ftperm.py nets/gen3.nnue --data data/gen3.txt.zst --stats-only   # report only
```

- The network computes exactly the same values: the permutation moves the
  two halves of the feature transformer (biases, HalfKA and threat weights)
  and the matching fc0 input columns together. The script compares
  `serialize.quantized_eval()` before and after on held-out positions and
  writes nothing if one differs; `bench` gives the same node count.
- Every architecture is supported (checked with `default`, `default-l2`,
  `threats-blend-ctx`, `att-threats-meta-blend-l2-ctx-nopsqt`: same
  evaluations, `verify.py` bit-exact, same `bench` node count).
- The gain depends on how sparse the network is. On a small test network
  (7 epochs on 92,000 positions, 38 % zero outputs) the all-zero chunks went
  from 4.1 % to 15.7 % of the held-out positions, and the instructions spent
  in `Network::evaluate()` during `bench` from 654 to 594 million (−9 %),
  −1.7 % for the whole bench (measured with callgrind: wall-clock nodes per
  second on the test machine vary more than that). Well-trained networks are
  sparser, so the gain should be larger: measure with `bench` on your machine.
- Run it on the network that will be used (`scripts/pipeline.sh` does it
  after promoting the SPRT winner, `FTPERM=1`). The permuted `.nnue` no
  longer matches the training checkpoint column for column, so compare it
  with `verify.py` without `--checkpoint`; training can continue from it
  (`--init-net`) or from the checkpoint as before.
- It takes about a minute and a half (test machine) with the default 20,000 + 5,000 positions
  (`--positions`, `--holdout`); positions are drawn from the first
  `--scan` lines of the data.

### 7.12. Quantization of current Pikafish (`make quant=2`)

Pikafish's networks use another quantization than Maskfish's default (since
before its "Update nnue-architecture"): the feature transformer at scale 255
instead of 127 doubled (twice the resolution, half the range: weights within
±0.498 instead of ±1), fc0 at scale 128 with activations shifted by 7 (twice
the resolution), and the last layer in the units of fc0 (scale 128 instead of
75.6: twice the resolution, weights within ±0.99 instead of ±1.68), the direct
term added before the final conversion. `make quant=2` builds the engine
with it, for both layer stacks (`layers=1` and `2`) and every other option;
the networks are the `*-q2` architectures (`default-q2`, `default-l2-q2`,
`threats-l2-ctx-q2`, ...), with another hash: each build rejects the
networks of the other. `quant.py` lists the scales of both schemes; the
default (`quant=1`) is unchanged, bit for bit (same `bench` node count as
before).

Checked: `verify.py` bit-exact, and the QAT forward pass equal to the engine,
with `default-q2`, `default-l2-q2`, `threats-blend-ctx-q2` and
`att-threats-meta-blend-l2-ctx-nopsqt-q2`.

```
cd ../src
make clean && make -j build ARCH=x86-64-avx2 quant=2 && mv maskfish maskfish-q2
make clean && make -j build ARCH=x86-64-avx2
cd ../trainer
python verify.py --engine ../src/maskfish-q2 --arch default-q2
```

**Starting from a network of the default quantization** (same feature set):

```
# the same network requantized (rounded to the new grid, the feature
# transformer weights beyond +-0.498 clipped; the script says how many)
python serialize.py --convert default-q2 nets/gen5.nnue nets/gen5-q2.nnue

# or trained further in the new quantization
python train.py data/train --val data/val --out runs/q2 --quant2 \
    --init-net nets/gen5.nnue --epochs 40 --epoch-size 10000000 [--qat-start 36]
```

`--init-net` with the other quantization dequantizes the network with its own
scales and clips it into the new ranges (it prints how many feature
transformer weights were clipped). The prepared data is shared (`-q2` does not
change the features). In `scripts/pipeline.sh`, `QUANT=2` (with `INIT_NET` to
start from an existing network) is a line of networks of its own
(`nets/default-q2-genN.nnue`, ...).

**Testing it with an SPRT**: compare networks trained the same way, one in
each quantization, each with its own build:

```
python train.py data/train --val data/val --out runs/q1 --init-net nets/gen5.nnue \
    --epochs 40 --epoch-size 10000000 --seed 1
python train.py data/train --val data/val --out runs/q2 --init-net nets/gen5.nnue \
    --epochs 40 --epoch-size 10000000 --seed 1 --quant2
python match.py --engine-a ../src/maskfish-q2 --net-a runs/q2/epoch_40.nnue \
    --engine-b ../src/maskfish --net-b runs/q1/epoch_40.nnue \
    --movetime 100 --workers 15 --sprt 0 5
```

`--movetime` makes the speed of each build count (on the test machine the
two had the same speed within the noise of `bench`); `--nodes` measures the
networks alone. The requantized network of the first command above only
tests the rounding (it should be within a few Elo of the original: on a small
test network its raw evaluations differed by 0.9 internal units on average,
8 at most); the real question is whether training in the new quantization
gives a stronger network, which only the trained pair answers. Combine with
QAT (section 7.10) in a second test rather than in the same one, to know
which change helped.

### 7.13. Evaluation scaling (`EvalScaling`)

`evaluate()` does not return the network output as is: it scales it (see
step 4 and `eval_scale.py`), and the trainer applies the same scaling in the
loss, so the two must agree. The engine has two scalings, chosen by the UCI
option `EvalScaling`:

- `Complexity` (default, unchanged): the output is damped by the
  "complexity" |psqt − positional| and scaled by the major material
  (`JieqiMaterial` applies to it). With a network **without PSQT**
  (`make psqt=no`) the complexity becomes |output| itself, so every large
  evaluation is damped (by about a quarter at 3000): it stays consistent
  (the trainer does the same) but it is not what that term was meant for.
- `Alignment`: Pikafish's "Replace nnueComplexity with simple_eval
  equivalent" (September 2026), made to allow networks without PSQT: the
  output is scaled up or down by its agreement with the material balance
  (`simple_eval`), then by the total material, then by the 40-move counter as
  before. It needs the material balance and the total material of every
  position, which `prepare_data.py` now stores (fields `se`, `allmat`):
  data prepared before must be prepared again from the text files.

Both are checked bit for bit by `verify.py --eval-scaling ...` (also with
`psqt=no` and `quant=2`). To use `Alignment`:

```
python prepare_data.py data/gen5.txt --out data/train5        # again, for se / allmat
python train.py data/train5 --val data/val5 --out runs/align --init-net nets/gen5.nnue \
    --eval-scaling alignment --epochs 40 --epoch-size 10000000
python verify.py --engine ../src/maskfish --net runs/align/epoch_40.nnue --eval-scaling alignment
python match.py --engine ../src/maskfish --net-a runs/align/epoch_40.nnue --net-b nets/gen5.nnue \
    --opt-a EvalScaling=Alignment --movetime 100 --workers 15 --sprt 0 5
```

The net trained with `--eval-scaling alignment` must play with
`EvalScaling Alignment` (A), the reference with the default (B). The two
scalings give evaluations of different sizes (`Alignment` scales up with
the material, `Complexity` down), so do not switch the option of an existing
network without training it again. The data may come from either scaling:
the loss compares win probabilities, and `fit_scaling.py` fits the constant
of the data used. `scripts/pipeline.sh`: `EVAL_SCALING=Alignment` sets the
option for the data generation, the checks and the SPRT, and
`--eval-scaling alignment` for the training. It suits `PSQT=0` best.

### 7.14. 32 layer stacks (`make stacks=32`)

Pikafish went from 16 to 32 layer stacks ("32 LayerStack buckets", October
2026), splitting its buckets (rooks and knights + cannons of each side) more
finely. Maskfish selects its 16 layer stacks by the number of pieces; with
`stacks=32` each of these buckets is split in two by the share of dark
pieces: layer stack `2b + 1` when more than half of the pieces other than the
kings are still dark, `2b` otherwise (`HalfKAv2_hm::make_layer_stack_bucket()`,
`jieqi_features.layer_stack_bucket()`). Two positions with the same number of
pieces, one still closed (many unknown identities) and one almost revealed,
then have their own layer stacks. This split is a choice of Maskfish (the one
of AB-JChess's `lsblend` uses the dark pieces too), not Pikafish's: test it
with an SPRT. The layer stacks `2b` and `2b + 1` share the PSQT bucket `b`, so
the feature transformer is the same as with 16 layer stacks; the network file
has 32 layer stacks (about 0.5 MB more) and the engine still evaluates one
layer stack per position, so its speed should hardly change. Not with
`lsblend=yes`.

The architecture names get the suffix `-s32`, after `-meta` / `-blend`
(`default-s32`, `att-meta-s32-l2`, ...). **The prepared data changes** (layer
stack of each position): prepare it with `--arch default-s32`. A network with
16 layer stacks converts into `-s32` with the same evaluation (each layer
stack copied into the two that split it), so a new line can start from the
current network:

```
cd ../src
make clean && make -j build ARCH=x86-64-avx2 stacks=32 && mv maskfish maskfish-s32
make clean && make -j build ARCH=x86-64-avx2
cd ../trainer
python verify.py --engine ../src/maskfish-s32 --arch default-s32
python prepare_data.py data/gen5.txt --out data/train5-s32 --arch default-s32
python train.py data/train5-s32 --out runs/s32 --epochs 40 --epoch-size 10000000 \
    --init-net nets/gen5.nnue --stack-balance 8
python match.py --engine-a ../src/maskfish-s32 --net-a runs/s32/epoch_40.nnue \
    --engine-b ../src/maskfish --net-b nets/gen5.nnue --movetime 100 --workers 15 --sprt 0 5
```

(`python serialize.py --convert default-s32 nets/gen5.nnue nets/gen5-s32.nnue`
writes the converted network itself.) The layer stacks of the endgames and of
the openings with many revealed pieces are seen less often: `--stack-balance`
weights them up. `scripts/pipeline.sh`: `STACKS=32` (with `INIT_NET` of 16
layer stacks: the same evaluation at generation 0).

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
  material, for `EvalScaling Alignment`, section 7.13), and `blend`
  (interpolation weight of the layer stacks) for the `-blend` architectures;
- `PREFIX.arch`: architecture name;
- `PREFIX.material`: `major` or `jieqi` (`--jieqi-material`).

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
| `verify.py` reports `MISMATCH` | The engine and the trainer do not use the same architecture. Use the engine and the trainer of the same Maskfish commit and rebuild the engine; for `att` / `threats` remember `make clean` and the matching `--arch` |
| The engine reports `network file ... was not loaded` | Wrong architecture (an `att` / `threats` / `-l2` network on a regular engine or the other way round; `threats=yes`, `nnue=att`, `layers=2` and `quant=2` must match the network's architecture), or the file is not a zstd compressed `.nnue` exported by the trainer |
| `no threat data (prepare the data again ...)` | The data was prepared by an old version of `prepare_data.py`; run it again with `--arch threats` |
| The engine reports `network file ... was not loaded` for a network trained before the engine used 8-bit weights | The file has the old version (16-bit HalfKA weights): convert it with `serialize.py --upgrade`, see [section 11.1](#111-file-format-change-8-bit-weights) |
| `the train data has no material / rule40` | The data was prepared before the loss used the `evaluate()` scaling: run `prepare_data.py` again on the text files (or train with `--no-eval-scale`) |
| `the train data has no se / allmat` | `--eval-scaling alignment` needs data prepared by the current `prepare_data.py`: prepare it again from the text files |
| `SCALE MISMATCH` with a network trained with `--eval-scaling alignment` | Run `verify.py` (and the engine) with `--eval-scaling alignment` / `EvalScaling Alignment` |
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
| `gen_data.py` was interrupted | Use the remaining `.partN` files as normal data |
| AB-JChess / the old branch print `No such option: RawScore` | Normal: these engines do not have that option; their scores must be rescaled with `--score-scale` ([section 6.3](#63-rescale-the-scores-and-train)) |
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
