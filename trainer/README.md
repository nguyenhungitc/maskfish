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
| `train.py` | Training loop; writes a checkpoint and a `.nnue` file every epoch |
| `verify.py` | Compares the engine's evaluation with the trainer's; must be bit-exact |
| `match.py` | Matches between two networks / option sets / builds, with SPRT |
| `fit_scaling.py` | Fits the score → win probability constant from self-play data |
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
- The build downloads the latest released network to `src/maskfish.nnue`
  if that file does not exist (`make net`). The scripts do not use it:
  every script passes the Jieqi network with the `EvalFile` option.

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
  architecture changes (e.g. `att`) and to run `fit_scaling.py`. They can be
  compressed for storage, but `prepare_data.py` only reads plain text files,
  so decompress them before use.

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
| `--init-net` | | Start from a `.nnue` file (new optimizer) |
| `--resume` | | Continue an interrupted run from a checkpoint (optimizer and epoch count included) |
| `--device` | `cuda` if available | `cpu` to force the CPU |
| `--threads` | 0 | Number of CPU threads for PyTorch (0 = default) |
| `--no-factorize` | | Disable the virtual features (on by default, see below) |
| `--factorize-kb` | | Add virtual features per king bucket (used by AB-JChess; off by default) |
| `--no-l1-fact` | | Disable the fc0 part shared by the layer stacks (on by default) |
| `--output-lr-scale` | 0.1 | Learning rate of the output layer, relative to `--lr` |
| `--stack-balance` | 0 (off) | Balance the loss across layer stacks; the value is the cap (e.g. 8) |
| `--checkpoint-every` | 1 | Write `epoch_K.pt` only every K epochs (and at the last one); `last.pt` and the `.nnue` files are always written |
| `--no-eval-scale` | | Compare the raw network output with the data scores (old behaviour, see below) |

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
every epoch the train loss, the validation loss, the learning rate and the
time. The validation loss should go down and then level off. Only compare
validation losses of runs that use **the same validation set**; a lower loss
does not mean a stronger network, only a match tells (step 6).

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

The quantization stays that of the current layer stacks (upstream changed the
quantization scales at the same time, for quantization-aware training — QAT
— which this trainer does not do).

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
(`default`) and 8 % (`threats`) slower because `fc0` is twice as large.
Compare with `--movetime`:

```
python match.py --engine-a ../src/maskfish-l2 --net-a runs/l2/epoch_40.nnue \
    --engine-b ../src/maskfish --net-b nets/genN.nnue \
    --movetime 100 --workers 15 --sprt 0 5
```

### 7.5. 32 layer stacks (`-s32`) and no PSQT (`-np`)

Two more suffixes, after Pikafish's SFNNv17 ("32 LayerStack buckets",
"Remove psqt accumulator"), adapted to Jieqi. Suffixes come in the order
`-s32`, `-l2`, `-np` (e.g. `threats-s32-l2-np`).

- `-s32` (engine: `make stacks=32`): 32 layer stacks and 32 PSQT buckets.
  Pikafish picks its 32 stacks by visible material composition (rooks,
  knights and cannons of each side); in Jieqi most of that is hidden in the
  opening, so the stack is the piece count bucket (as for 16 stacks) times
  2, plus 1 when more pieces are still dark than the median for that piece
  count (`DARK_STACK_THRESHOLDS`, measured on self-play games). The stack is
  stored in the prepared data: **prepare the data again** with `--arch
  default-s32` (or `threats-s32`, ...).
- `-np` (engine: `make psqt=no`, `train.py --no-psqt`): no PSQT part. The
  complexity term of `evaluate()` then uses `simple_eval()` (material of the
  side to move, a dark piece counting as the average of its pool) in place
  of the PSQT output, as Pikafish does; `prepare_data.py` stores
  `simple_eval()` of every position (field `se`), so **data prepared before
  this change cannot train an `-np` network**.

```
cd ../src
make clean && make -j build ARCH=x86-64-avx2 stacks=32 && mv maskfish maskfish-s32
make clean && make -j build ARCH=x86-64-avx2 psqt=no && mv maskfish maskfish-np
make clean && make -j build ARCH=x86-64-avx2
cd ../trainer
python verify.py --engine ../src/maskfish-s32 --arch default-s32
python verify.py --engine ../src/maskfish-np --arch default-np
python prepare_data.py data/gen1.txt --out data/train_s32 --arch default-s32
python train.py data/train_s32 --val data/val_s32 --out runs/s32 --epochs 40
python train.py data/train --val data/val --out runs/np --no-psqt --epochs 40
```

Both change the speed of the engine little (`-np` saves the PSQT
accumulator). Compare by matches against the 16-stack network trained on the
same data, as in [section 7.4](#74-new-layer-stacks-layers2).

## 8. Data formats

**Text** (written by `gen_data.py`, read by `prepare_data.py` and
`fit_scaling.py`):

```
xxxxkxxxx/9/1x5x1/x1x1x3x/1P4n2/9/X1X1X1X1X/6B2/9/XXXXKXXXX b R2A2C2P4N2B1r2a2c2p5n1b2 5 3 | -272 | 1
```

The engine's FEN (with the pool), the score for the side to move in internal
units, and the result for the side to move (1 / 0 / −1). Data from other
sources can be added if it uses this format and internal units (`RawScore`).

**Prepared** (per shard, about 260 bytes per position):

- `PREFIX-NNNNN.features.npy`: `int16 (N, 2, 64)`, feature indices of the side
  to move and of the other side, padded with the padding index;
- `PREFIX-NNNNN.meta.npy`: records `bucket` (layer stack), `result`, `score`,
  `material`, `rule40` and `se` (inputs of the `evaluate()` scaling, see step
  4; `se` is `simple_eval()`, used by `-np` networks);
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
| Text file | about 100 bytes/position (10 million ≈ 1 GB) |
| Prepared data | about 260 bytes/position (10 million ≈ 2.6 GB) |
| `prepare_data.py` | about 15,000 lines/s per CPU core |
| Checkpoint | about 190 MB (`att`: 750 MB, `threats`: 1.3 GB) per epoch |
| `.nnue` file | about 8 MB (`att`: larger, `threats`: up to about 60 MB), zstd compressed |

Training speed depends on the GPU; run a small epoch first
(`--epoch-size 1000000`) to measure it before planning.

## 10. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `verify.py` reports `MISMATCH` | The engine and the trainer do not use the same architecture. Use the engine and the trainer of the same Maskfish commit and rebuild the engine; for `att` / `threats` remember `make clean` and the matching `--arch` |
| The engine reports `network file ... was not loaded` | Wrong architecture (an `att` / `threats` / `-l2` network on a regular engine or the other way round; `threats=yes`, `nnue=att` and `layers=2` must match the network's architecture), or the file is not a zstd compressed `.nnue` exported by the trainer |
| `no threat data (prepare the data again ...)` | The data was prepared by an old version of `prepare_data.py`; run it again with `--arch threats` |
| The engine reports `network file ... was not loaded` for a network trained before the engine used 8-bit weights | The file has the old version (16-bit HalfKA weights): convert it with `serialize.py --upgrade`, see [section 11.1](#111-file-format-change-8-bit-weights) |
| `the train data has no material / rule40` | The data was prepared before the loss used the `evaluate()` scaling: run `prepare_data.py` again on the text files (or train with `--no-eval-scale`) |
| `SCALE MISMATCH` in `verify.py` | `eval_scale.py` no longer matches `src/evaluate.cpp` (constants or material): update it |
| `incompatible network for arch ...` | A network of the wrong architecture was given to `--init-net` / `serialize.py`; the message gives the file's real architecture |
| `prepare_data.py` runs out of RAM | Lower `--shard-size` (e.g. 500000) |
| `train.py` runs out of GPU memory | Lower `--batch-size` (e.g. 8192); the `att` architecture needs more |
| The loss becomes `nan` or jumps up | Lower `--lr` (e.g. 5e-4), check the data for abnormal scores (lower `--max-score`); try `--no-factorize` to see whether factorization is the cause |
| `warning: optimizer state not restored` with `--resume` | The checkpoint was made by an old trainer version (a single parameter group): the model is loaded, only the optimizer state is created anew |
| Slow training, GPU idle | The data is on a slow disk: move it to an SSD; increase `--batch-size` |
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
