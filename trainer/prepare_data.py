"""Converts text training data (fen | score | result) into binary arrays that
train.py reads directly (memory-mapped, no parsing during training).

    python prepare_data.py data/gen1.txt data/gen2.txt --out data/train [--arch att]
    -> data/train.features.npy  int16 (N, 2, 64): stm / nstm feature indices
       data/train.meta.npy      (N,) records: bucket, score, result
       data/train.arch          architecture of the features (see jieqi_features)

Positions are shuffled. Lines that do not parse are skipped.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp

import numpy as np

import jieqi_features as F

META_DTYPE = np.dtype([("bucket", np.int8), ("result", np.int8), ("score", np.int16)])


def _init_worker(arch: str):
    F.set_arch(arch)


def parse_line(line: str):
    try:
        fen, score, result = (x.strip() for x in line.split("|"))
        s, n, b, _ = F.fen_to_features(fen)
        if len(s) > F.MAX_ACTIVE or len(n) > F.MAX_ACTIVE:
            return None
        return (s, n, b, int(float(score)), int(float(result)))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-score", type=int, default=10000,
                    help="drop positions with a larger |score| (internal units)")
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--arch", default="default", choices=sorted(F.ARCHS))
    args = ap.parse_args()
    F.set_arch(args.arch)

    lines = []
    for path in args.inputs:
        with open(path) as f:
            lines.extend(l for l in f if l.strip())

    with mp.Pool(args.workers, initializer=_init_worker, initargs=(args.arch,)) as pool:
        parsed = [p for p in pool.imap(parse_line, lines, chunksize=4096)
                  if p is not None and abs(p[3]) <= args.max_score]

    n = len(parsed)
    order = np.random.default_rng(args.seed).permutation(n)
    feats = np.lib.format.open_memmap(args.out + ".features.npy", mode="w+",
                                      dtype=np.int16, shape=(n, 2, F.MAX_ACTIVE))
    meta = np.lib.format.open_memmap(args.out + ".meta.npy", mode="w+",
                                     dtype=META_DTYPE, shape=(n,))
    feats[:] = F.PAD_INDEX
    for dst, src in enumerate(order):
        s, nn, b, score, result = parsed[src]
        feats[dst, 0, :len(s)] = s
        feats[dst, 1, :len(nn)] = nn
        meta[dst] = (b, result, max(-32000, min(32000, score)))
    feats.flush()
    meta.flush()
    with open(args.out + ".arch", "w") as f:
        f.write(args.arch + "\n")
    print(f"{n} positions ({len(lines) - n} skipped) -> {args.out}.*.npy")


if __name__ == "__main__":
    main()
