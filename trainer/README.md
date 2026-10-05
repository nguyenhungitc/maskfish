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
| `prepare_data.py` | text data → shuffled binary arrays |
| `train.py` | training loop, writes a checkpoint and a `.nnue` per epoch |
| `verify.py` | checks engine vs. trainer evaluations |
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

# 2. convert (keep a few percent for validation)
python prepare_data.py data/gen0.txt --out data/train0
python prepare_data.py data/val.txt  --out data/val

# 3. train
python train.py data/train0 --val data/val --out runs/gen1 \
    --epochs 100 --epoch-size 20000000 --batch-size 16384

# 4. check the exported net and test it against the previous one
python verify.py --engine ../src/pikafish --net runs/gen1/epoch_100.nnue \
    --checkpoint runs/gen1/epoch_100.pt
```

Then generate with the new network, add the data to the previous one, and
train again (`--init-net` to start from the last network, `--resume` to
continue a run). Always select networks by playing games (SPRT), not by the
loss.

### Rough sizes

A first useful network needs tens of millions of positions, a strong one
hundreds of millions or more. `prepare_data.py` stores about 260 bytes per
position. `gen_data.py` drives the engine over UCI from Python: it is simple
and correct but not fast; run many workers, on many machines if possible.

### Main options

* `--lambda`: 1.0 trains on search scores only, lower values mix in game
  results (e.g. 0.75 once the data comes from a decent network).
* `--scaling 361`: score → win probability scale, the same constant the
  engine uses to average chance nodes.
* `--lr 8.75e-4`, `--gamma 0.992` (decay per epoch), `--batch-size 16384`.

## If the engine's network changes

`jieqi_features.py`, `model.py` and the constants at the top of
`serialize.py` (hashes, sizes, scales) describe the current engine. After any
change to `src/nnue`, update them and run `verify.py` until it is bit-exact.
