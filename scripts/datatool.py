#!/usr/bin/env python3
"""TrainRecord files (112 bytes a record, src/selfplay.cpp) for scripts/pipeline.sh and scripts/autotune.sh.

  datatool.py count FILE...                 number of records of each file, and the total
  datatool.py fit-k FILE...                 the score scale K of the files: P(win) ~ sigmoid(score / K)
  datatool.py rescale IN OUT FACTOR         OUT = IN with every score multiplied by FACTOR (mate / known-win scores kept)
  datatool.py cat OUT IN...                 OUT = the records of IN... one after the other
  datatool.py stats FILE...                 records, games ended, result shares, score spread (one line)

Scores of another engine are in its own units: pipeline.sh fits K on its scores (fit-k) and on Maskfish's,
and rescales them by K_maskfish / K_ref, so that the data of both can be trained on together.
Only numpy is needed.
"""
import sys
from pathlib import Path

import numpy as np

REC = np.dtype([("board", "u1", 90), ("stm", "u1"), ("urn", "i1", (2, 6)), ("score", "<i2"), ("result", "i1"),
                ("flags", "u1"), ("ply", "<u2"), ("move", "<u2"), ("nocap", "u1")])
assert REC.itemsize == 112
MATE = 20000  # |score| at or above: a mate or known-win score (VALUE_KNOWN_WIN), not rescaled; train/records.py keep_mask drops them


def load(path):
    size = Path(path).stat().st_size
    if size % REC.itemsize:
        sys.exit(f"{path}: {size} bytes, not a whole number of {REC.itemsize}-byte records")
    return np.fromfile(path, dtype=REC)


def fit_k(score, result):
    """As train/records.py fit_k: K in 100..1490 minimising the log loss of sigmoid(score / K) on |score| < 3000
    (here on every record; the trainer fits it again on the records it keeps)."""
    m = np.abs(score) < 3000
    s = score[m].astype(np.float64)
    y = (result[m].astype(np.float64) + 1) / 2
    if len(s) < 100:
        sys.exit(f"fit-k: only {len(s)} records with |score| < 3000")
    best = None
    for k in range(100, 1500, 10):
        p = np.clip(1 / (1 + np.exp(-s / k)), 1e-6, 1 - 1e-6)
        ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
        if best is None or ll < best[1]:
            best = (k, ll)
    return best[0]


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "count":
        total = 0
        for p in args:
            n = Path(p).stat().st_size // REC.itemsize
            total += n
            print(f"{n} {p}")
        print(f"{total} total")
    elif cmd == "fit-k":
        r = np.concatenate([load(p) for p in args])
        print(fit_k(r["score"], r["result"]))
    elif cmd == "rescale":
        src, dst, factor = args[0], args[1], float(args[2])
        r = load(src)
        s = r["score"].astype(np.float64)
        mate = np.abs(s) >= MATE
        s = np.where(mate, s, np.clip(np.round(s * factor), -(MATE - 1), MATE - 1))
        r["score"] = s.astype(np.int16)
        r.tofile(dst)
        print(f"{len(r)} records, scores x {factor:.4f}")
    elif cmd == "cat":
        dst, srcs = args[0], args[1:]
        with open(dst, "wb") as out:
            for p in srcs:
                out.write(Path(p).read_bytes())
        print(f"{Path(dst).stat().st_size // REC.itemsize} records")
    elif cmd == "stats":
        r = np.concatenate([load(p) for p in args])
        if not len(r):
            print("0 records")
            return
        res = r["result"]
        print(f"{len(r)} records, results +{(res > 0).mean():.3f} ={(res == 0).mean():.3f} -{(res < 0).mean():.3f}, "
              f"|score| median {int(np.median(np.abs(r['score'].astype(int))))}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
