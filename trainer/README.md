# Jieqi NNUE trainer

PyTorch trainer for the engine's network (`src/nnue`): HalfKAv2_hm Jieqi
features (7560 inputs per side, dark pieces and pool included), a 2048-wide
feature transformer with 16 PSQT buckets, and 16 layer stacks
(2048 → 16 → 32 → 1). The exported `.nnue` is read by the engine
**bit-exactly** as the trainer computes it (checked by `verify.py`).

Parts of the design follow AB-JChess's trainer and nnue-pytorch; the code is
written for this engine's feature set and file format.

## Files

| File | Purpose |
| --- | --- |
| `jieqi_features.py` | FEN parsing and feature indices, mirror of `half_ka_v2_hm` |
| `model.py` | PyTorch model and weight clipping |
| `serialize.py` | export to / import from `.nnue` (zstd), integer reference evaluation |
| `init_net.py` | material-only starting network |
| `gen_data.py` | self-play data generation with the engine (real hidden identities) |
| `prepare_data.py` | text data → shuffled binary shards (streaming, bounded memory) |
| `train.py` | training loop, writes a checkpoint and a `.nnue` per epoch |
| `verify.py` | checks engine vs. trainer evaluations |
| `fit_scaling.py` | fits the score → win probability scale from self-play data |
| `match.py` | games between two networks / option sets, with SPRT |
| `jieqi_uci.py` | UCI helpers |

## Setup

```
pip install -r requirements.txt     # install the CUDA build of torch for GPU training
cd ../src && make -j build ARCH=x86-64-avx2 && cd ../trainer
python verify.py --engine ../src/pikafish          # must print "N/N positions bit-exact"
```

The engine options used by the tools: `RawScore` (scores in internal units)
and the `evalraw` command (raw network output).

## Training loop

No Jieqi network exists yet, so start from material and iterate:

```
python init_net.py nets/gen0.nnue

# 1. generate data with the current network
python gen_data.py --engine ../src/pikafish --net nets/gen0.nnue \
    --games 20000 --nodes 5000 --workers 15 --out data/gen0.txt

# 2. convert (validation data generated separately, with another seed)
python prepare_data.py data/gen0.txt --out data/train0
python prepare_data.py data/val.txt  --out data/val

# 3. train
python train.py data/train0 --val data/val --out runs/gen1 \
    --epochs 100 --epoch-size 20000000 --batch-size 16384

# 4. check the exported net and test it against the previous one
python verify.py --engine ../src/pikafish --net runs/gen1/epoch_100.nnue \
    --checkpoint runs/gen1/epoch_100.pt
python match.py --engine ../src/pikafish --net-a runs/gen1/epoch_100.nnue \
    --net-b nets/gen0.nnue --nodes 20000 --workers 15 --sprt 0 5
```

Then generate with the new network and train again on all the data
(`train.py data/train0 data/train1 ...`, with `--init-net` to start from the
last network, or `--resume` to continue a run). Always select networks by
playing games (SPRT), not by the loss.

### Data format

`prepare_data.py` reads the text files as a stream and writes shards of
`--shard-size` positions (default 2 million, about 520 MB each), shuffled
inside each shard: memory use stays at about 1 GB whatever the size of the
data. `train.py` accepts several prefixes and draws each batch from a random
shard, in proportion to the shard sizes.

### Testing with match.py

`match.py` plays pairs of games with the same hidden identities and random
opening, colors swapped, and runs an SPRT on the pentanomial results
(`--sprt elo0 elo1`, e.g. `0 5` for a new network, `0 3` for a search
option). A is the tested version, B the reference:

```
# a new network against the current one
python match.py --engine ../src/pikafish --net-a new.nnue --net-b cur.nnue --sprt 0 5
# a search option, same network
python match.py --engine ../src/pikafish --net-a cur.nnue --opt-a TTPenalize=true --sprt 0 3
# two engine builds (e.g. default vs nnue=att, each with its network)
python match.py --engine-a ../src_att/pikafish --net-a att.nnue \
    --engine-b ../src/pikafish --net-b cur.nnue --sprt 0 5
```

Each engine is told only what its side knows: reveals are public, the
identity of a captured dark piece goes to the capturer only (`--full-info`
to tell both). Use the same time control as the one the engine will play
with (`--nodes`, `--depth` or `--movetime`); `--workers` games run in
parallel, one thread per engine.

### Rough sizes

A first useful network needs tens of millions of positions, a strong one
hundreds of millions or more. The prepared data takes about 260 bytes per
position on disk. `gen_data.py` drives the engine over UCI from Python: it is simple
and correct but not fast; run many workers, on many machines if possible.

### Calibrating the score scale

```
python fit_scaling.py data/gen1.txt
```

prints the scale that best predicts the game results from the scores, with a
calibration table. Use it for the engine (`setoption name ChanceScaling value
<s>`, averaging of chance nodes) and for training (`train.py --scaling <s>`).
Re-fit it for each new network.

### The "att" architecture

Build the engine with `make nnue=att` (attack buckets + mid mirror, see the
main README) and pass `--arch att` to `init_net.py` and `prepare_data.py`.
`train.py` takes the architecture from the prepared data, `serialize.py` and
`verify.py` detect it from the network file. Data must be prepared again for
each architecture (the text files from `gen_data.py` are shared). The model
has about 4 times more feature transformer weights: count about 1 GB of
memory for parameters and optimizer state.

```
python init_net.py nets/att0.nnue --arch att
python prepare_data.py data/gen0.txt --out data/train0_att --arch att   # also for val
python train.py data/train0_att --out runs/att1 ...
python verify.py --engine ../src/pikafish --net runs/att1/epoch_100.nnue
```

### Main options

* `--lambda`: 1.0 trains on search scores only, lower values mix in game
  results (e.g. 0.75 once the data comes from a decent network).
* `--scaling 361`: score → win probability scale, the same constant the
  engine uses to average chance nodes.
* `--lr 8.75e-4`, `--gamma 0.992` (decay per epoch), `--batch-size 16384`.

## If the engine's network changes

`jieqi_features.py` (feature sets, `ARCHS`), `model.py` and the constants at
the top of `serialize.py` (hashes, sizes, scales) describe the engine. After
any change to `src/nnue`, update them and run `verify.py`, for every
architecture, until it is bit-exact.
