"""Fits the score -> win probability scale from self-play data.

    python fit_scaling.py data/gen1.txt [more files] [--max-score 3000]

The files can be compressed (.zst, .gz, see textio.py).

Finds s maximizing the likelihood of the game results under
    P(result for the side to move) = sigmoid(score / s)
(result 1 / 0 / -1 counted as 1 / 0.5 / 0, i.e. expected score), and prints
a calibration table. Use the value for:
  * the engine:  setoption name ChanceScaling value <s>
    (averaging of reveal / hidden capture outcomes)
  * the trainer: train.py --scaling <s>
The data must come from the network you want to calibrate, generated with
RawScore (gen_data.py does that).

    python fit_scaling.py data/gen1.txt --by-material 8 --table-out nets/scaling.json

also fits one scale per range of total material (sum of the piece values of
both sides but the kings, eval_scale.total_material(): the field allmat of the
prepared data), with as many positions in each of the 8 ranges, as a win
rate model that depends on the material. The table (the median
material of each range and its scale) is for train.py --scaling-table, which
interpolates between its points. The engine's ChanceScaling stays one
constant: use the global scale for it.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from textio import open_text


def load(paths, max_score, material: bool = False):
    """(scores, results as 0 / 0.5 / 1, total materials or None)."""
    scores, results, mats = [], [], []
    if material:
        import eval_scale as E
        import jieqi_features as F
    for path in paths:
        with open_text(path) as f:
            for line in f:
                try:
                    fen, s, r = (x.strip() for x in line.split("|"))
                    s, r = float(s), float(r)
                    if abs(s) <= max_score and material:
                        m = E.total_material(F.parse_fen(fen))
                except (ValueError, IndexError):
                    continue
                if abs(s) <= max_score:
                    scores.append(s)
                    results.append((r + 1.0) / 2.0)
                    if material:
                        mats.append(m)
    return np.array(scores), np.array(results), (np.array(mats) if material else None)


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
    ap.add_argument("--by-material", type=int, default=0, metavar="N",
                    help="also fit a scale for each of N ranges of total material")
    ap.add_argument("--table-out", metavar="FILE",
                    help="with --by-material: write the table for train.py --scaling-table")
    args = ap.parse_args()
    if args.table_out and not args.by_material:
        ap.error("--table-out needs --by-material")

    s, r, mat = load(args.inputs, args.max_score, material=args.by_material > 0)
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

    if args.by_material:
        fit_by_material(s, r, mat, args.by_material, scale, args.table_out)


def fit_by_material(s, r, mat, n, global_scale, table_out):
    """One scale per range of total material (equal numbers of positions)."""
    import json
    order = np.argsort(mat, kind="stable")
    points, scales, counts = [], [], []
    print()
    print(f"{'total material':>16} {'positions':>10} {'scale':>7} {'log-loss':>9} {'global':>9}")
    for part in np.array_split(order, n):
        if len(part) < 1000:
            print(f"warning: only {len(part)} positions in a range, the fit will be noisy")
        ps, pr, pm = s[part], r[part], mat[part]
        sc = fit(ps, pr)
        points.append(float(np.median(pm)))
        scales.append(sc)
        counts.append(len(part))
        print(f"{pm.min():7.0f}..{pm.max():<7.0f} {len(part):10d} {sc:7.1f} {nll(sc, ps, pr):9.5f} "
              f"{nll(global_scale, ps, pr):9.5f}")
    # equal medians (much repeated material): keep the first point of each
    keep = [0] + [k for k in range(1, n) if points[k] > points[k - 1]]
    table = {"material": [points[k] for k in keep], "scale": [round(scales[k], 2) for k in keep],
             "positions": [counts[k] for k in keep], "global": round(global_scale, 2)}
    if table_out:
        with open(table_out, "w") as f:
            json.dump(table, f, indent=1)
        print(f"\nwritten {table_out}: train.py --scaling-table {table_out}")


if __name__ == "__main__":
    main()
