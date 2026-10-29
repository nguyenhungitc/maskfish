"""Converts text training data (fen | score | result) into binary shards that
train.py reads directly (memory-mapped, no parsing during training).

    python prepare_data.py data/gen1.txt data/gen2.txt --out data/train [--arch att]
    -> data/train-00000.features.npy  int16 (N, 2, 64): stm / nstm feature indices
       data/train-00000.meta.npy      (N,) records: bucket, result, score, material, rule40,
                                      se, allmat, sed
       data/train-00001.* ...         one pair of files per shard
       data/train.arch                architecture of the features
       data/train.material            "major" or "jieqi" (see --jieqi-material)
       data/train.fields              names of the meta fields

material and rule40 are the inputs of the scaling that the engine's evaluate()
applies to the network output (see eval_scale.py); train.py applies the same
scaling before comparing the network with the data scores. se (material
balance for the side to move) and allmat (total material) are those of the
EvalScaling Alignment option (train.py --eval-scaling alignment). sed
(simple_eval_dark(): material balance with dark pieces at the average value of
their pool) stands for the PSQT part of networks without one (train.py
--no-psqt) in EvalScaling Complexity.

The threat architectures (--arch threats / att-threats) also write, per shard:
       data/train-00000.threats.npy   uint32 (T,): threat feature indices of all
                                      positions, stm then nstm, position by position
       data/train-00000.thrcount.npy  uint8 (N, 2): number of stm / nstm threats

The input is read as a stream and every shard (--shard-size positions) is
shuffled before it is written, so the memory needed is about
2 x 260 bytes x shard size (2 x 500 bytes with threats) whatever the amount of
data. train.py draws its batches from random shards, which mixes the data
across shards.
Lines that do not parse are skipped, counted and reported (the first ones
are printed); any other error stops the script.

Data scored by another engine (see rescore.py) uses that engine's
score scale: convert it with --score-scale 361/s, where s is the scale that
fit_scaling.py finds on that data.

Every position mirrored left-right is also added (files a <-> i, same score
and result: the game is symmetric) when its features differ from the
original's. With HalfKAv2_hm they differ only when the two kings stand on the
middle file (the features of a king off it are mirrored to the same side), and
never with the "att" feature sets (mid mirror): the script reports how many
positions were added.

--max-dups N keeps at most N copies of each position (the same FEN but for
the move number), the first ones of the input: positions repeated in many
games (openings) then weigh less in the training. The inputs are read twice
(a first pass hashes the positions).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import multiprocessing as mp
import os
import sys

import numpy as np

import eval_scale as E
import jieqi_features as F
from textio import open_text

META_DTYPE = np.dtype([("bucket", np.int8), ("result", np.int8), ("score", np.int16),
                       ("material", np.int16), ("rule40", np.uint8), ("se", np.int16),
                       ("allmat", np.int16), ("sed", np.int16)])
# "-blend" architectures: also the interpolation weight of the layer stacks
META_DTYPE_BLEND = np.dtype(META_DTYPE.descr + [("blend", np.uint8)])
CHUNK_LINES = 16384


def _init_worker(arch: str, max_score: int, score_scale: float, jieqi_material: bool,
                 mid_mirror: dict | None = None):
    global _MAX_SCORE, _SCORE_SCALE, _JIEQI_MATERIAL
    F.set_arch(arch)
    F.set_mid_mirror_options(**(mid_mirror or {}))
    _MAX_SCORE = max_score
    _SCORE_SCALE = score_scale
    _JIEQI_MATERIAL = jieqi_material


# Malformed lines (wrong field count, bad numbers, bad FEN: ValueError; a
# board with more than 32 pieces: IndexError in layer_stack_bucket()) are
# skipped and counted. Any other exception is a bug and is not hidden.
MALFORMED_ERRORS = (ValueError, IndexError)
MAX_EXAMPLES = 5   # malformed lines printed, at most


def position_key(line: str) -> bytes:
    """Identity of the position of a data line for --max-dups: the FEN without
    the move number (a malformed line is its own key)."""
    fen = line.split("|", 1)[0].split()
    return " ".join(fen[:4] if len(fen) >= 4 else fen).encode()


def hash_chunk(lines: list[str]) -> np.ndarray:
    """64-bit hashes of the positions of lines (done in the worker processes)."""
    return np.fromiter((int.from_bytes(hashlib.blake2b(position_key(l), digest_size=8).digest(),
                                       "little") for l in lines), dtype=np.uint64, count=len(lines))


def keep_mask(hashes: np.ndarray, max_dups: int) -> np.ndarray:
    """True for the first max_dups occurrences of every hash, in order."""
    order = np.argsort(hashes, kind="stable")
    h = hashes[order]
    start = np.r_[True, h[1:] != h[:-1]]
    first = np.maximum.accumulate(np.where(start, np.arange(len(h)), 0))
    keep = np.empty(len(h), dtype=bool)
    keep[order] = (np.arange(len(h)) - first) < max_dups
    return keep


def parse_chunk(lines: list[str]):
    """Parses lines into compact arrays (done in the worker processes).
    Returns (features, meta, threats, lines read, positions filtered out,
    malformed lines, a few malformed lines with their error, mirrored
    positions added)."""
    rows = 2 * len(lines)
    feats = np.full((rows, 2, F.MAX_ACTIVE), F.PAD_INDEX, dtype=np.int16)
    meta = np.zeros(rows, dtype=META_DTYPE_BLEND if F.BLEND else META_DTYPE)
    counts = np.zeros((rows, 2), dtype=np.uint8)
    threats = []
    n = filtered = malformed = mirrored = 0
    examples = []
    for line in lines:
        try:
            fen, score, result = (x.strip() for x in line.split("|"))
            score, result = int(round(float(score) * _SCORE_SCALE)), int(float(result))
            if abs(score) > _MAX_SCORE or result not in (-1, 0, 1):
                filtered += 1
                continue
            pos = F.parse_fen(fen)
            stm = pos.side_to_move
            s, o = F.active_features(pos, stm), F.active_features(pos, 1 - stm)
            b, blend = F.layer_stack_selection(pos) if F.BLEND else (F.layer_stack_bucket(pos), 0)
            if len(s) > F.MAX_ACTIVE or len(o) > F.MAX_ACTIVE:
                filtered += 1
                continue
            material, rule40 = E.position_scale_inputs(pos, _JIEQI_MATERIAL)
            se, allmat = E.alignment_inputs(pos)
            sed = max(-32000, min(32000, E.simple_eval_dark(pos)))
            if F.THREATS:
                phys = F.physical_threats(pos)
                ts, to = F.active_threats(pos, stm, phys), F.active_threats(pos, 1 - stm, phys)
                # (the counts are stored as uint8, the engine reads at most MAX_THREATS)
                if len(ts) > F.MAX_THREATS or len(to) > F.MAX_THREATS:
                    filtered += 1
                    continue
        except MALFORMED_ERRORS as e:
            malformed += 1
            if len(examples) < MAX_EXAMPLES:
                examples.append(f"{line.strip()[:120]!r}: {type(e).__name__}: {e}")
            continue
        record = (b, result, max(-32000, min(32000, score)), material, min(rule40, 255),
                  se, allmat, sed) + ((blend,) if F.BLEND else ())
        samples = [(s, o, (ts, to) if F.THREATS else None)]
        # the mirrored position (same layer stack, blend, material, se /
        # allmat / sed, context: they only count pieces), if it brings new
        # features
        mp = F.mirror_position(pos)
        ms, mo = F.active_features(mp, stm), F.active_features(mp, 1 - stm)
        mt = None
        if F.THREATS:
            mphys = F.physical_threats(mp)
            mt = (F.active_threats(mp, stm, mphys), F.active_threats(mp, 1 - stm, mphys))
        if sorted(ms) != sorted(s) or sorted(mo) != sorted(o) or \
                (F.THREATS and (sorted(mt[0]) != sorted(ts) or sorted(mt[1]) != sorted(to))):
            samples.append((ms, mo, mt))
            mirrored += 1
        for fs, fo, ft in samples:
            feats[n, 0, :len(fs)] = fs
            feats[n, 1, :len(fo)] = fo
            meta[n] = record
            if F.THREATS:
                counts[n] = (len(ft[0]), len(ft[1]))
                threats.append(ft[0])
                threats.append(ft[1])
            n += 1
    thr = None
    if F.THREATS:
        thr = (np.fromiter((i for t in threats for i in t), dtype=np.uint32), counts[:n])
    return feats[:n], meta[:n], thr, len(lines), filtered, malformed, examples, mirrored


def read_chunks(paths: list[str]):
    for path in paths:
        with open_text(path) as f:
            while True:
                lines = [l for l in itertools.islice(f, CHUNK_LINES) if l.strip()]
                if not lines:
                    break
                yield lines


def shuffle_threats(flat: np.ndarray, counts: np.ndarray, order: np.ndarray):
    """Reorders ragged per-position threat lists (see the module docstring)."""
    per_pos = counts.astype(np.int64).sum(axis=1)
    starts = np.concatenate(([0], np.cumsum(per_pos)))[:-1]
    new_len = per_pos[order]
    # index into flat for every element of the shuffled array
    new_starts = np.concatenate(([0], np.cumsum(new_len)))[:-1]
    src = np.repeat(starts[order] - new_starts, new_len) + np.arange(new_len.sum())
    return flat[src], counts[order]


def write_shard(out: str, index: int, feats: list, metas: list, threats: list, rng) -> int:
    f = np.concatenate(feats)
    m = np.concatenate(metas)
    order = rng.permutation(len(m))
    np.save(f"{out}-{index:05d}.features.npy", f[order])
    np.save(f"{out}-{index:05d}.meta.npy", m[order])
    if F.THREATS:
        flat = np.concatenate([t[0] for t in threats])
        counts = np.concatenate([t[1] for t in threats])
        flat, counts = shuffle_threats(flat, counts, order)
        np.save(f"{out}-{index:05d}.threats.npy", flat)
        np.save(f"{out}-{index:05d}.thrcount.npy", counts)
    return len(m)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-score", type=int, default=10000,
                    help="drop positions with a larger |score| (internal units, after --score-scale)")
    ap.add_argument("--score-scale", type=float, default=1.0,
                    help="multiply the scores by this factor, to convert the scores of another "
                         "engine to this engine's internal units (361 / scale fitted on that data)")
    ap.add_argument("--shard-size", type=int, default=2_000_000, help="positions per shard")
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--arch", default="default", choices=sorted(F.ARCHS))
    ap.add_argument("--jieqi-material", action="store_true",
                    help="the data was generated with the engine option JieqiMaterial=true: "
                         "store the material evaluate() then uses (default: major_material)")
    F.add_mid_mirror_args(ap)
    ap.add_argument("--max-dups", type=int, default=0, metavar="N",
                    help="keep at most N copies of each position (default: all)")
    args = ap.parse_args()
    if args.max_dups < 0:
        ap.error("--max-dups must be positive")
    F.set_arch(args.arch)
    F.apply_mid_mirror_args(args)
    mid_mirror = {"pawn_tiebreak": args.mid_pawn_tiebreak, "no_king_flag": args.mid_no_king_flag}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    rng = np.random.default_rng(args.seed)
    shard, buf_f, buf_m, buf_t, buffered = 0, [], [], [], 0
    total_in = total_out = total_filtered = total_malformed = total_mirrored = 0
    total_dups = 0
    examples = []
    with mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(args.arch, args.max_score, args.score_scale,
                           args.jieqi_material, mid_mirror)) as pool:
        chunks = read_chunks(args.inputs)
        if args.max_dups:
            # first pass: the positions to keep, chunk by chunk
            hashes = list(pool.imap(hash_chunk, read_chunks(args.inputs)))
            keep = keep_mask(np.concatenate(hashes) if hashes else np.zeros(0, np.uint64),
                             args.max_dups)
            total_dups = int(len(keep) - keep.sum())
            print(f"--max-dups {args.max_dups}: {total_dups} of {len(keep)} lines are dropped "
                  f"as repeated positions", flush=True)
            bounds = np.cumsum([0] + [len(h) for h in hashes])

            def kept_chunks():
                for k, lines in enumerate(read_chunks(args.inputs)):
                    m = keep[bounds[k]:bounds[k + 1]]
                    if len(m) != len(lines):
                        raise SystemExit("the inputs changed between the two passes")
                    yield [l for l, keep_it in zip(lines, m) if keep_it]
            chunks = (c for c in kept_chunks() if c)
        for feats, meta, thr, n_in, n_filtered, n_malformed, ex, n_mirrored in pool.imap(
                parse_chunk, chunks):
            total_in += n_in
            total_filtered += n_filtered
            total_malformed += n_malformed
            total_mirrored += n_mirrored
            examples.extend(ex[:MAX_EXAMPLES - len(examples)])
            buf_f.append(feats)
            buf_m.append(meta)
            buf_t.append(thr)
            buffered += len(meta)
            if buffered >= args.shard_size:
                total_out += write_shard(args.out, shard, buf_f, buf_m, buf_t, rng)
                print(f"shard {shard}: {buffered} positions ({total_in} lines read)", flush=True)
                shard, buf_f, buf_m, buf_t, buffered = shard + 1, [], [], [], 0
        if buffered:
            total_out += write_shard(args.out, shard, buf_f, buf_m, buf_t, rng)
            shard += 1

    with open(args.out + ".arch", "w") as f:
        f.write(F.feature_arch(args.arch) + "\n")   # the layer stacks do not matter
    with open(args.out + ".material", "w") as f:
        f.write(("jieqi" if args.jieqi_material else "major") + "\n")
    with open(args.out + ".fields", "w") as f:      # (scripts/pipeline.sh checks it)
        f.write(" ".join(META_DTYPE.names) + "\n")
    print(f"{total_out} positions in {shard} shard(s) -> {args.out}-*.npy "
          f"({total_filtered} filtered out by score, result or feature count, "
          f"{total_malformed} malformed lines skipped"
          + (f", {total_dups} repeated positions dropped" if args.max_dups else "")
          + f", {total_mirrored} mirrored positions added)")
    if total_malformed:
        print(f"warning: {total_malformed} of {total_in} lines are malformed, e.g.:", file=sys.stderr)
        for e in examples:
            print(f"  {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
