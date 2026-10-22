"""Score <-> mass tables for the engine option ChanceModel Tables.

The engine averages the outcomes of a chance node (a reveal, or a capture of a
dark piece whose identity the capturer learns) in "mass" space when
ChanceModel is Tables: from the point of view of the side
whose move caused the chance event, every outcome score is converted to a mass
with the table score -> mass, the masses are averaged (weighted by the number
of pieces of each kind), and the average is converted back with the table
mass -> score. The tables are stored as:

    probability_score_to_mass.i32le   4001 little-endian int32, scores -2000..2000
    probability_mass_to_score.i32le   1901 little-endian int32, masses -950..950

Both tables must be non-decreasing (the engine checks it). A mass is an
expected game result in thousandths: 1000 * (2 * E - 1), E = win + draw / 2
for the side to move, clamped to -950..950.

    # fit the tables to self-play data of the network to be used
    # (gen_data.py text format: "fen | score | result", scores in internal units)
    python chance_tables.py fit data/gen1.txt [more files] --out tables/gen1

    # tables equivalent to the default win-rate averaging (a reference)
    python chance_tables.py logistic --scale 360.8 --out tables/logistic

    # copy the tables embedded in a reference engine's network package
    python chance_tables.py extract reference.nnue --out tables/extracted [--monotone]

Then:

    setoption name ChanceTables value <the --out directory>
    setoption name ChanceModel value Tables

Extracted tables are in the score units of the other engine's networks: they
fit Maskfish's networks only roughly. Fitting tables to Maskfish's own data is preferable.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import struct
import sys

from textio import open_text

SCORE_MIN, SCORE_MAX = -2000, 2000
MASS_MIN, MASS_MAX = -950, 950
SCORE_FILE = "probability_score_to_mass.i32le"
MASS_FILE = "probability_mass_to_score.i32le"
N_SCORES = SCORE_MAX - SCORE_MIN + 1
N_MASSES = MASS_MAX - MASS_MIN + 1
VALUE_MATE_IN_MAX_PLY = 32000 - 246


def mass_of_expectation(e: float) -> int:
    """Expected result E (0..1) -> mass (thousandths of 2E - 1), clamped."""
    return max(MASS_MIN, min(MASS_MAX, round(1000.0 * (2.0 * e - 1.0))))


def monotone(values: list[int]) -> list[int]:
    """Running maximum: the smallest non-decreasing sequence above values."""
    out, best = [], None
    for v in values:
        best = v if best is None else max(best, v)
        out.append(best)
    return out


def invert(score_to_mass: list[int]) -> list[int]:
    """Table mass -> score from a non-decreasing table score -> mass: the middle
    of the scores that have the mass, or for a mass that no score has, the
    first score with a larger mass (SCORE_MAX if there is none)."""
    first, last = {}, {}
    for i, x in enumerate(score_to_mass):
        first.setdefault(x, i)
        last[x] = i
    out = []
    for m in range(MASS_MIN, MASS_MAX + 1):
        if m in first:
            s = SCORE_MIN + (first[m] + last[m]) // 2
        else:
            # first index with a larger mass (the table is non-decreasing)
            above = bisect.bisect_right(score_to_mass, m)
            s = SCORE_MAX if above == len(score_to_mass) else SCORE_MIN + above
        out.append(s)
    return monotone(out)


def write_tables(out_dir: str, score_to_mass: list[int], mass_to_score: list[int], info: dict):
    assert len(score_to_mass) == N_SCORES and len(mass_to_score) == N_MASSES
    check(score_to_mass, mass_to_score)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, SCORE_FILE), "wb") as f:
        f.write(struct.pack(f"<{N_SCORES}i", *score_to_mass))
    with open(os.path.join(out_dir, MASS_FILE), "wb") as f:
        f.write(struct.pack(f"<{N_MASSES}i", *mass_to_score))
    with open(os.path.join(out_dir, "chance-tables.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
        f.write("\n")
    print(f"tables written to {out_dir}")
    print(f"setoption name ChanceTables value {os.path.abspath(out_dir)}")
    print("setoption name ChanceModel value Tables")


def check(score_to_mass: list[int], mass_to_score: list[int]):
    """The checks of the engine (Search::load_chance_tables())."""
    errors = []
    if any(b < a for a, b in zip(score_to_mass, score_to_mass[1:])):
        errors.append("score -> mass is not non-decreasing")
    if any(b < a for a, b in zip(mass_to_score, mass_to_score[1:])):
        errors.append("mass -> score is not non-decreasing")
    if any(m < MASS_MIN or m > MASS_MAX for m in score_to_mass):
        errors.append("a mass is outside -950..950")
    if any(abs(s) >= VALUE_MATE_IN_MAX_PLY for s in mass_to_score):
        errors.append("a score is in the mate range")
    if errors:
        raise SystemExit("invalid tables: " + "; ".join(errors) + " (try --monotone)")


def cmd_logistic(args):
    s2m = [mass_of_expectation(1.0 / (1.0 + math.exp(-s / args.scale)))
           for s in range(SCORE_MIN, SCORE_MAX + 1)]
    write_tables(args.out, s2m, invert(s2m), {"source": "logistic", "scale": args.scale})


def load_data(paths, max_score):
    """(score, result) pairs of gen_data.py text files, result in -1 / 0 / 1."""
    pairs = []
    for path in paths:
        with open_text(path) as f:
            for line in f:
                try:
                    _, s, r = (x.strip() for x in line.split("|"))
                    s, r = int(round(float(s))), int(float(r))
                except ValueError:
                    continue
                if abs(s) <= max_score and r in (-1, 0, 1):
                    pairs.append((s, r))
    return pairs


def pava(values: list[float], weights: list[float]) -> list[float]:
    """Weighted isotonic regression (pool adjacent violators), non-decreasing."""
    blocks = []   # [value, weight, length]
    for v, w in zip(values, weights):
        blocks.append([v, w, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, n2 = blocks.pop()
            v1, w1, n1 = blocks.pop()
            w = w1 + w2
            blocks.append([(v1 * w1 + v2 * w2) / w if w else (v1 + v2) / 2, w, n1 + n2])
    out = []
    for v, _, n in blocks:
        out.extend([v] * n)
    return out


def cmd_fit(args):
    pairs = load_data(args.inputs, args.max_score)
    if len(pairs) < 10000:
        print(f"warning: only {len(pairs)} positions, the tables will be noisy", file=sys.stderr)
    # Results per score bin, with a logistic prior of `prior` pseudo-positions
    # per score so that sparse bins stay sensible
    width = args.bin
    nbins = (SCORE_MAX - SCORE_MIN) // width + 1
    sums = [0.0] * nbins
    counts = [0.0] * nbins
    for s, r in pairs:
        b = (max(SCORE_MIN, min(SCORE_MAX, s)) - SCORE_MIN) // width
        sums[b] += (r + 1) / 2.0
        counts[b] += 1
    expect, weights = [], []
    for b in range(nbins):
        centre = SCORE_MIN + b * width + (width - 1) / 2
        prior_e = 1.0 / (1.0 + math.exp(-centre / args.prior_scale))
        w = counts[b] + args.prior * width
        expect.append((sums[b] + prior_e * args.prior * width) / w)
        weights.append(w)
    fitted = pava(expect, weights)
    # Linear interpolation between bin centres, then masses
    centres = [SCORE_MIN + b * width + (width - 1) / 2 for b in range(nbins)]
    s2m = []
    for s in range(SCORE_MIN, SCORE_MAX + 1):
        if s <= centres[0]:
            e = fitted[0]
        elif s >= centres[-1]:
            e = fitted[-1]
        else:
            b = int((s - centres[0]) // width)
            t = (s - centres[b]) / width
            e = fitted[b] * (1 - t) + fitted[b + 1] * t
        s2m.append(mass_of_expectation(e))
    s2m = monotone(s2m)

    print(f"{len(pairs)} positions")
    print(f"{'score':>7} {'positions':>10} {'observed':>9} {'mass':>6}")
    for b in range(0, nbins, max(1, nbins // 20)):
        lo = SCORE_MIN + b * width
        obs = sums[b] / counts[b] if counts[b] else float("nan")
        print(f"{lo:7d} {int(counts[b]):10d} {obs:9.3f} {s2m[lo - SCORE_MIN]:6d}")
    write_tables(args.out, s2m, invert(s2m),
                 {"source": "fit", "inputs": [os.path.abspath(p) for p in args.inputs],
                  "positions": len(pairs), "bin": width, "prior": args.prior,
                  "prior_scale": args.prior_scale})


# Headers (16-byte magic, version) of the network packages that hold the tables
PACKAGE_FORMATS = {(b"ABJCHESSV11" + b"\0" * 5, 110), (b"ABJCHESSV82" + b"\0" * 5, 82)}


def cmd_extract(args):
    """Copies the tables of a network package."""
    with open(args.source, "rb") as f:
        data = f.read()
    version, length = struct.unpack_from("<II", data, 16)
    if (data[:16], version) not in PACKAGE_FORMATS:
        raise SystemExit("not a supported network package (unknown header)")
    text = data[24:24 + length].decode("utf-8")
    metadata, end = json.JSONDecoder().raw_decode(text)
    if text[end:].strip() not in ("", "\\n"):
        raise SystemExit("unexpected trailing metadata")
    wanted = {SCORE_FILE: N_SCORES * 4, MASS_FILE: N_MASSES * 4}
    tables = {}
    for entry in metadata["chunks"]:
        name = entry["name"]
        start = 24 + length + entry["data_offset"]
        payload = data[start:start + entry["size"]]
        if "sha256" in entry and hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise SystemExit(f"chunk SHA-256 mismatch: {name}")
        if name in wanted:
            if name in tables or len(payload) != wanted[name]:
                raise SystemExit(f"invalid probability table: {name}")
            tables[name] = list(struct.unpack(f"<{wanted[name] // 4}i", payload))
    if tables.keys() != wanted.keys():
        raise SystemExit("the package has no probability tables")
    s2m, m2s = tables[SCORE_FILE], tables[MASS_FILE]
    if args.monotone:
        s2m, m2s = monotone(s2m), monotone(m2s)
    write_tables(args.out, s2m, m2s,
                 {"source": "network package", "package": os.path.abspath(args.source),
                  "package_sha256": hashlib.sha256(data).hexdigest(), "monotone": args.monotone})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fit", help="fit the tables to self-play data")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--max-score", type=int, default=10000,
                   help="ignore positions with a larger |score| (others are clamped to +-2000)")
    p.add_argument("--bin", type=int, default=20, help="width of the score bins")
    p.add_argument("--prior", type=float, default=1.0,
                   help="logistic prior, in pseudo-positions per score")
    p.add_argument("--prior-scale", type=float, default=360.83524,
                   help="scale of the logistic prior (fit_scaling.py)")
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("logistic", help="tables of the sigmoid of ChanceModel Winrate")
    p.add_argument("--scale", type=float, default=360.83524)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_logistic)

    p = sub.add_parser("extract", help="copy the tables of a network package")
    p.add_argument("source")
    p.add_argument("--out", required=True)
    p.add_argument("--monotone", action="store_true",
                   help="make the tables non-decreasing (running maximum)")
    p.set_defaults(func=cmd_extract)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
