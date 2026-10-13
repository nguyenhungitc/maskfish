"""Fits the score -> win probability scale from self-play data.

    python fit_scaling.py data/gen1.txt [more files] [--max-score 3000]

Finds s maximizing the likelihood of the game results under
    P(result for the side to move) = sigmoid(score / s)
(result 1 / 0 / -1 counted as 1 / 0.5 / 0, i.e. expected score), and prints
a calibration table. Use the value for:
  * the engine:  setoption name ChanceScaling value <s>
    (averaging of reveal / hidden capture outcomes)
  * the trainer: train.py --scaling <s>
The data must come from the network you want to calibrate, generated with
RawScore (gen_data.py does that).
"""

from __future__ import annotations

import argparse
import math

import numpy as np


def load(paths, max_score):
    scores, results = [], []
    for path in paths:
        with open(path) as f:
            for line in f:
                try:
                    _, s, r = (x.strip() for x in line.split("|"))
                    s, r = float(s), float(r)
                except ValueError:
                    continue
                if abs(s) <= max_score:
                    scores.append(s)
                    results.append((r + 1.0) / 2.0)
    return np.array(scores), np.array(results)


def nll(scale, s, r):
    p = 1.0 / (1.0 + np.exp(-s / scale))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(r * np.log(p) + (1 - r) * np.log(1 - p))


def fit(s, r, lo=50.0, hi=5000.0):
    # golden-section search on log(scale)
    a, b = math.log(lo), math.log(hi)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(math.exp(c), s, r), nll(math.exp(d), s, r)
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(math.exp(c), s, r)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(math.exp(d), s, r)
    return math.exp((a + b) / 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--max-score", type=float, default=3000.0,
                    help="ignore positions with a larger |score| (internal units)")
    args = ap.parse_args()

    s, r = load(args.inputs, args.max_score)
    if len(s) < 1000:
        print(f"warning: only {len(s)} positions, the fit will be noisy")
    scale = fit(s, r)
    print(f"{len(s)} positions, fitted scale: {scale:.1f} "
          f"(log-loss {nll(scale, s, r):.5f}, engine default 360.8: {nll(360.83524, s, r):.5f})")
    print()
    print(f"{'score range':>16} {'positions':>10} {'observed':>9} {'fitted':>7}")
    edges = np.linspace(-args.max_score, args.max_score, 13)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & (s < hi)
        if m.sum():
            fitted = np.mean(1.0 / (1.0 + np.exp(-s[m] / scale)))
            print(f"{lo:7.0f}..{hi:<7.0f} {int(m.sum()):10d} {r[m].mean():9.3f} {fitted:7.3f}")
    print()
    print(f"setoption name ChanceScaling value {round(scale)}")
    print(f"train.py --scaling {scale:.0f}")


if __name__ == "__main__":
    main()
