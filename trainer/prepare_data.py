"""Converts text training data (fen | score | result) into binary shards that
train.py reads directly (memory-mapped, no parsing during training).

    python prepare_data.py data/gen1.txt data/gen2.txt --out data/train [--arch att]
    -> data/train-00000.features.npy  int16 (N, 2, 64): stm / nstm feature indices
       data/train-00000.meta.npy      (N,) records: bucket, result, score
       data/train-00001.* ...         one pair of files per shard
       data/train.arch                architecture of the features

The input is read as a stream and every shard (--shard-size positions) is
shuffled before it is written, so the memory needed is about
2 x 260 bytes x shard size whatever the amount of data. train.py draws its
batches from random shards, which mixes the data across shards.
Lines that do not parse are skipped.
"""

from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import os

import numpy as np

import jieqi_features as F

META_DTYPE = np.dtype([("bucket", np.int8), ("result", np.int8), ("score", np.int16)])
CHUNK_LINES = 16384


def _init_worker(arch: str, max_score: int):
    global _MAX_SCORE
    F.set_arch(arch)
    _MAX_SCORE = max_score


def parse_chunk(lines: list[str]):
    """Parses lines into compact arrays (done in the worker processes)."""
    feats = np.full((len(lines), 2, F.MAX_ACTIVE), F.PAD_INDEX, dtype=np.int16)
    meta = np.zeros(len(lines), dtype=META_DTYPE)
    n = 0
    for line in lines:
        try:
            fen, score, result = (x.strip() for x in line.split("|"))
            score, result = int(float(score)), int(float(result))
            if abs(score) > _MAX_SCORE or result not in (-1, 0, 1):
                continue
            s, o, b, _ = F.fen_to_features(fen)
            if len(s) > F.MAX_ACTIVE or len(o) > F.MAX_ACTIVE:
                continue
        except Exception:
            continue
        feats[n, 0, :len(s)] = s
        feats[n, 1, :len(o)] = o
        meta[n] = (b, result, max(-32000, min(32000, score)))
        n += 1
    return feats[:n], meta[:n], len(lines)


def read_chunks(paths: list[str]):
    for path in paths:
        with open(path) as f:
            while True:
                lines = [l for l in itertools.islice(f, CHUNK_LINES) if l.strip()]
                if not lines:
                    break
                yield lines


def write_shard(out: str, index: int, feats: list, metas: list, rng) -> int:
    f = np.concatenate(feats)
    m = np.concatenate(metas)
    order = rng.permutation(len(m))
    np.save(f"{out}-{index:05d}.features.npy", f[order])
    np.save(f"{out}-{index:05d}.meta.npy", m[order])
    return len(m)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-score", type=int, default=10000,
                    help="drop positions with a larger |score| (internal units)")
    ap.add_argument("--shard-size", type=int, default=2_000_000, help="positions per shard")
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--arch", default="default", choices=sorted(F.ARCHS))
    args = ap.parse_args()
    F.set_arch(args.arch)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    rng = np.random.default_rng(args.seed)
    shard, buf_f, buf_m, buffered = 0, [], [], 0
    total_in = total_out = 0
    with mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(args.arch, args.max_score)) as pool:
        for feats, meta, n_in in pool.imap(parse_chunk, read_chunks(args.inputs)):
            total_in += n_in
            buf_f.append(feats)
            buf_m.append(meta)
            buffered += len(meta)
            if buffered >= args.shard_size:
                total_out += write_shard(args.out, shard, buf_f, buf_m, rng)
                print(f"shard {shard}: {buffered} positions ({total_in} lines read)", flush=True)
                shard, buf_f, buf_m, buffered = shard + 1, [], [], 0
        if buffered:
            total_out += write_shard(args.out, shard, buf_f, buf_m, rng)
            shard += 1

    with open(args.out + ".arch", "w") as f:
        f.write(args.arch + "\n")
    print(f"{total_out} positions ({total_in - total_out} skipped) in {shard} shard(s) -> {args.out}-*.npy")


if __name__ == "__main__":
    main()
