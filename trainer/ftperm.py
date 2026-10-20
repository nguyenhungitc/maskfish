"""Reorders the neurons of the feature transformer of a network so that the
engine evaluates it faster, without changing a single evaluation (the idea
of nnue-pytorch's ftperm.py).

    python ftperm.py nets/gen3.nnue nets/gen3.nnue --data data/gen3.txt.zst
    python ftperm.py nets/gen3.nnue --data data/gen3.txt.zst --stats-only

Why it helps: the first layer of every layer stack (fc0) is computed by
AffineTransformSparseInput (src/nnue/layers/affine_transform_sparse_input.h),
which reads the 0..127 outputs of the feature transformer in chunks of 4
bytes and skips every chunk that is entirely zero. The order of the neurons
does not matter to the network, so neurons that are often zero together are
put in the same chunk: more chunks are skipped, fc0 does less work.

What is permuted: neuron j of the transformed output is the product of
accumulator entries j and j + L1/2, so a permutation of 0 .. L1/2 - 1 is
applied to both halves of the feature transformer (biases, HalfKA weights,
threat weights; the PSQT columns do not move) and to the matching fc0 input
columns of both perspectives in every layer stack. The integer network
computes exactly the same values: the script checks it on positions of the
data (serialize.quantized_eval() before and after) and refuses to write
otherwise; `bench` gives the same node count with the new file.

How: activations of --positions positions of the data (both perspectives),
then groups of 4 neurons are built greedily (the neuron most often zero,
then the neurons most often zero at the same time as the group), and
improved by swapping neurons between groups while that increases the number
of all-zero chunks. The statistics are reported on positions not used to
choose the permutation (--holdout).

Run it on the network that will be used (e.g. after the SPRT, before
publishing): a permuted network can be trained further (train.py
--init-net) like any other; after more training, run the script again.
"""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np

import jieqi_features as F
import serialize as S
from textio import open_text

CHUNK = 4      # AffineTransformSparseInput::ChunkSize (SSSE3 / AVX2 / AVX-512 / NEON dot)


# ---------------------------------------------------------------- activations

def read_fens(paths: list[str], count: int, scan: int, seed: int) -> list[str]:
    """'count' FENs drawn at random among the first 'scan' lines of the files
    (text data, plain or compressed)."""
    fens = []
    for path in paths:
        with open_text(path) as f:
            for line in f:
                fen = line.split("|", 1)[0].strip()
                if fen:
                    fens.append(fen)
                if len(fens) >= scan:
                    break
        if len(fens) >= scan:
            break
    if not fens:
        raise SystemExit("no position found in the data")
    random.Random(seed).shuffle(fens)
    return fens[:count]


def transformed(q: dict, fens: list[str]) -> np.ndarray:
    """Transformed feature transformer outputs (0..127, as the engine's
    FeatureTransformer::transform()) of both perspectives of each position:
    uint8 (2 * len(fens), L1 / 2)."""
    L1, h = F.L1, F.L1 // 2
    w = q["ft_weight"].astype(np.int32)
    tw = q["thr_weight"].astype(np.int32) if F.THREATS else None
    bias = q["ft_bias"].astype(np.int32)
    out = np.empty((2 * len(fens), h), dtype=np.uint8)
    for n, fen in enumerate(fens):
        stm_f, nstm_f, _, _ = F.fen_to_features(fen)
        thr = F.fen_to_threats(fen) if F.THREATS else ([], [])
        for p, (feats, tf) in enumerate(zip((stm_f, nstm_f), thr)):
            acc = bias + w[feats].sum(axis=0) if feats else bias.copy()
            if tw is not None and tf:
                acc = acc + tw[tf].sum(axis=0)
            acc = np.clip(2 * acc[:L1], 0, 254)
            out[2 * n + p] = (acc[:h] * acc[h:]) // 512
    return out


# ---------------------------------------------------------------- permutation search

def pack_zero(act: np.ndarray) -> np.ndarray:
    """Bit matrix (neurons, words): bit s of row j is 1 when neuron j is zero
    in sample s."""
    z = act == 0                                   # (samples, neurons)
    return np.packbits(z.T, axis=1, bitorder="little").view(np.uint8)


def popcount(a: np.ndarray) -> np.ndarray:
    return np.bitwise_count(a).sum(axis=-1, dtype=np.int64)


def zero_chunks(act: np.ndarray, perm: np.ndarray) -> float:
    """Fraction of the 4-neuron chunks that are entirely zero with this order."""
    a = act[:, perm]
    blocks = a.reshape(a.shape[0], -1, CHUNK)
    return float((blocks == 0).all(axis=2).mean())


def greedy_groups(Z: np.ndarray) -> list[list[int]]:
    """Groups of CHUNK neurons: seed = the neuron most often zero, then the
    neuron most often zero together with the group, until full."""
    n = Z.shape[0]
    free = np.ones(n, dtype=bool)
    freq = popcount(Z)
    groups = []
    while free.any():
        cand = np.flatnonzero(free)
        seed = int(cand[np.argmax(freq[cand])])
        free[seed] = False
        group, cur = [seed], Z[seed].copy()
        while len(group) < CHUNK and free.any():
            cand = np.flatnonzero(free)
            j = int(cand[np.argmax(popcount(Z[cand] & cur))])
            free[j] = False
            group.append(j)
            cur &= Z[j]
        groups.append(group)
    return groups


def improve(Z: np.ndarray, groups: list[list[int]], max_sweeps: int, verbose: bool) -> list[list[int]]:
    """Swaps of two neurons of different groups while one increases the
    number of samples where a whole group is zero. For neuron i of group a
    and j of group b, with O[x] = AND of the zero bits of the other members
    of x's group:
        gain = |O[i] & Z[j]| + |O[j] & Z[i]| - score(a) - score(b)."""
    n = Z.shape[0]
    group_of = np.empty(n, dtype=np.int64)
    for g, members in enumerate(groups):
        group_of[members] = g
    full = np.full(Z.shape[1], 0xFF, dtype=np.uint8)

    def others(g: int) -> None:
        for x in groups[g]:
            acc = full.copy()
            for y in groups[g]:
                if y != x:
                    acc &= Z[y]
            O[x] = acc

    O = np.empty_like(Z)
    for g in range(len(groups)):
        others(g)
    score = np.array([popcount(O[m[0]] & Z[m[0]]) for m in groups], dtype=np.int64)

    for sweep in range(max_sweeps):
        swaps, gained = 0, 0
        for i in range(n):
            a = int(group_of[i])
            gains = popcount(O[i] & Z) + popcount(O & Z[i]) - score[a] - score[group_of]
            gains[group_of == a] = -1
            j = int(np.argmax(gains))
            if gains[j] <= 0:
                continue
            b = int(group_of[j])
            ga, gb = groups[a], groups[b]
            ga[ga.index(i)] = j
            gb[gb.index(j)] = i
            group_of[i], group_of[j] = b, a
            others(a)
            others(b)
            score[a] = popcount(O[ga[0]] & Z[ga[0]])
            score[b] = popcount(O[gb[0]] & Z[gb[0]])
            swaps += 1
            gained += int(gains[j])
        if verbose:
            print(f"  sweep {sweep + 1}: {swaps} swaps, +{gained} zero chunks", flush=True)
        if swaps == 0:
            break
    return groups


def find_permutation(act: np.ndarray, max_sweeps: int, verbose: bool = True) -> np.ndarray:
    Z = pack_zero(act)
    t = time.time()
    groups = greedy_groups(Z)
    if verbose:
        print(f"greedy grouping: {time.time() - t:.1f}s", flush=True)
    groups = improve(Z, groups, max_sweeps, verbose)
    perm = np.array([j for g in groups for j in g], dtype=np.int64)
    assert sorted(perm.tolist()) == list(range(act.shape[1]))
    return perm


# ---------------------------------------------------------------- applying it

def permute(q: dict, perm: np.ndarray) -> dict:
    """The same network with the transformed neurons in the order 'perm'
    (new neuron k = old neuron perm[k], in both halves and both perspectives)."""
    h = F.L1 // 2
    cols = np.concatenate([perm, h + perm])
    p = dict(q)
    p["ft_bias"] = q["ft_bias"][cols]
    p["ft_weight"] = q["ft_weight"][:, cols]
    if F.THREATS:
        p["thr_weight"] = q["thr_weight"][:, cols]
    # fc0 input: [stm transformed (h), nstm transformed (h)], same mapping
    p["fc0_weight"] = q["fc0_weight"][..., cols]
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("net")
    ap.add_argument("out", nargs="?", help="output network (may be the input)")
    ap.add_argument("--data", nargs="+", required=True,
                    help="text data (fen | score | result), plain or .zst / .gz")
    ap.add_argument("--positions", type=int, default=20000,
                    help="positions used to choose the order")
    ap.add_argument("--holdout", type=int, default=5000,
                    help="other positions, to report the result and check the evaluations")
    ap.add_argument("--scan", type=int, default=1_000_000,
                    help="the positions are drawn among the first SCAN lines of the data")
    ap.add_argument("--sweeps", type=int, default=20, help="maximum improvement sweeps")
    ap.add_argument("--check", type=int, default=300,
                    help="held-out positions whose evaluation is compared before / after")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--stats-only", action="store_true", help="only report the zero chunks")
    args = ap.parse_args()
    if not args.stats_only and not args.out:
        raise SystemExit("give the output network (or --stats-only)")

    F.set_arch(S.detect_arch(args.net))
    q = S.read_nnue(args.net)
    if q["version"] != S.VERSION:
        raise SystemExit(f"{args.net} has the old file format: run serialize.py --upgrade first")
    print(f"arch {F.ARCH}, {F.L1 // 2} transformed neurons per perspective, "
          f"{F.L1 // CHUNK} chunks of {CHUNK} in the fc0 input", flush=True)

    fens = read_fens(args.data, args.positions + args.holdout, args.scan, args.seed)
    held, train = fens[:args.holdout], fens[args.holdout:]
    t = time.time()
    act_train = transformed(q, train)
    act_held = transformed(q, held)
    print(f"activations of {len(train)} + {len(held)} positions: {time.time() - t:.0f}s; "
          f"zero outputs: {(act_held == 0).mean():.1%}", flush=True)

    ident = np.arange(F.L1 // 2)
    before = zero_chunks(act_held, ident)
    if args.stats_only:
        print(f"all-zero chunks (held-out positions): {before:.1%}")
        return

    perm = find_permutation(act_train, args.sweeps)
    after_train = zero_chunks(act_train, perm)
    after = zero_chunks(act_held, perm)
    print(f"all-zero chunks: {before:.1%} -> {after:.1%} on held-out positions "
          f"({after_train:.1%} on the positions used); fc0 reads about "
          f"{(1 - after) / max(1e-9, 1 - before):.0%} of the chunks it read before", flush=True)

    p = permute(q, perm)
    bad = 0
    for fen in held[:args.check]:
        if S.quantized_eval(q, fen) != S.quantized_eval(p, fen):
            bad += 1
    if bad:
        raise SystemExit(f"BUG: {bad} of {min(args.check, len(held))} evaluations changed; "
                         "nothing written")
    print(f"evaluations unchanged on {min(args.check, len(held))} positions", flush=True)

    tmp = args.out + ".tmp"
    S.write_nnue(p, tmp, q["description"])
    os.replace(tmp, args.out)
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
