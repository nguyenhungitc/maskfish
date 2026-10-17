"""Plays Jieqi games between two engines / networks / option sets, with an
SPRT, to decide whether a change is an improvement.

    python match.py --engine ../src/maskfish \
        --net-a runs/gen2/epoch_100.nnue --net-b nets/gen1.nnue \
        --nodes 20000 --workers 15 --sprt 0 5

    python match.py --engine ../src/maskfish --net-a nets/gen1.nnue \
        --opt-a TTPenalize=true --nodes 20000 --sprt 0 3

A is the version being tested, B the reference (by default B uses the same
engine and network as A). Results are reported for A.

Jieqi luck is reduced by playing games in pairs: both games of a pair use
the same hidden identities and the same random opening, with colors swapped.
The SPRT uses the pentanomial statistics of the pairs (logistic Elo,
normal approximation as in fishtest).

Information model: each engine only gets what its side knows. Reveals are
public; when a dark piece is captured, the capturer is told its identity but
its owner is not (use --full-info to tell both).

Games end by mate / no legal move, by adjudication (mate score, or both
sides agreeing that |score| >= --adjudicate for 4 moves each), on the third
occurrence of a position (perpetual check loses, otherwise a draw), or as a
draw after 80 plies without capture (the 40-move rule) or --max-plies.
"""

from __future__ import annotations

import argparse
import copy
import math
import multiprocessing as mp
import os
import random
import time

import jieqi_features as F
import jieqi_uci as U


# ---------------------------------------------------------------- games

def make_opening(engine: U.Engine, rng: random.Random, plies: int):
    """Random legal opening moves (raw engine moves) and the identities."""
    ids = U.HiddenIdentities(rng)
    probe = copy.deepcopy(ids)
    moves = []
    for _ in range(plies):
        engine.set_position(moves)
        legal = engine.legal_moves()
        if not legal:
            break
        raw = rng.choice(legal)
        moves.append(probe.annotate(raw))      # full info, only to list legal moves
    raw_moves = [m[:4] for m in moves]
    return ids, raw_moves


def play_game(white: U.Engine, black: U.Engine, ids: U.HiddenIdentities, opening: list[str],
              args) -> int:
    """Returns +1 white wins, 0 draw, -1 black wins."""
    ids = copy.deepcopy(ids)
    views = ([], [])                 # move lists sent to the white / black engine
    engines = (white, black)
    for e in engines:
        e.send("ucinewgame")
        e.isready()

    last_scores = {F.WHITE: [], F.BLACK: []}   # from white's point of view
    no_capture = 0
    repetitions = U.RepetitionTracker()
    for ply in range(args.max_plies):
        stm = ply % 2
        eng = engines[stm]
        eng.set_position(views[stm])
        fen, in_check = eng.board_info()
        result = repetitions.push(fen, in_check)
        if result is not None:
            return result
        if ply < len(opening):
            move = opening[ply]
            score, mate = None, False
        else:
            move, score, mate = eng.search(args.go)
            if move in ("(none)", "0000"):
                return -1 if stm == F.WHITE else 1
        capture = U.is_capture(fen, move)
        w, b = U.annotate_views(ids, move, args.full_info)
        views[0].append(w)
        views[1].append(b)

        if score is not None:
            sign = 1 if stm == F.WHITE else -1
            if mate:
                return sign if score > 0 else -sign
            last_scores[stm] = (last_scores[stm] + [sign * score])[-4:]
            sw, sb = last_scores[F.WHITE], last_scores[F.BLACK]
            if len(sw) == 4 and len(sb) == 4:
                if min(sw + sb) >= args.adjudicate:
                    return 1
                if max(sw + sb) <= -args.adjudicate:
                    return -1
        no_capture = 0 if capture else no_capture + 1
        if no_capture >= U.NO_CAPTURE_DRAW_PLIES:
            return 0
    return 0


def make_engine(path, net, opts, hash_mb):
    o = {"Threads": 1, "Hash": hash_mb, "RawScore": "true"}
    if net:
        o["EvalFile"] = os.path.abspath(net)
    o.update(opts)
    return U.Engine(path, o)


def worker_init(args):
    global _ENGINES, _ARGS
    _ARGS = args
    _ENGINES = (make_engine(args.engine_a, args.net_a, args.opt_a, args.hash),
                make_engine(args.engine_b, args.net_b, args.opt_b, args.hash))


def play_pair(seed: int) -> tuple[int, int]:
    """Two games from the same opening and identities, colors swapped.
    Returns the results for A of both games (+1 / 0 / -1)."""
    a, b = _ENGINES
    rng = random.Random(seed)
    ids, opening = make_opening(a, rng, rng.randint(_ARGS.random_plies_min, _ARGS.random_plies))
    r1 = play_game(a, b, ids, opening, _ARGS)        # A white
    r2 = -play_game(b, a, ids, opening, _ARGS)       # A black
    return r1, r2


# ---------------------------------------------------------------- statistics

def pentanomial(pairs):
    """Counts of pair scores 0, 0.5, 1, 1.5, 2 (in game points for A)."""
    penta = [0] * 5
    for r1, r2 in pairs:
        penta[(r1 + 1) + (r2 + 1)] += 1      # 0..4 half-points
    return penta


def stats(penta):
    n = sum(penta)
    if n == 0:
        return None
    scores = [0.0, 0.25, 0.5, 0.75, 1.0]
    mean = sum(s * c for s, c in zip(scores, penta)) / n
    var = sum(c * (s - mean) ** 2 for s, c in zip(scores, penta)) / n
    return n, mean, var


def elo_from_score(s):
    s = min(max(s, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / s - 1.0)


def elo_estimate(penta):
    st = stats(penta)
    if not st:
        return 0.0, 0.0
    n, mean, var = st
    elo = elo_from_score(mean)
    if var <= 0 or n < 2:
        return elo, float("inf")
    se = math.sqrt(var / n)
    lo, hi = elo_from_score(mean - 1.96 * se), elo_from_score(mean + 1.96 * se)
    return elo, (hi - lo) / 2


def llr(penta, elo0, elo1):
    """Log-likelihood ratio of H1 (elo1) vs H0 (elo0), logistic Elo,
    normal approximation on the pentanomial pair scores. Half a pseudo-pair
    is added to every category so that a few lopsided pairs (variance 0)
    cannot give a huge LLR; it is negligible once there are many pairs."""
    st = stats([c + 0.5 for c in penta])
    if not st:
        return 0.0
    n, mean, var = st
    if var <= 0:
        var = 1e-3
    s0 = 1.0 / (1.0 + 10 ** (-elo0 / 400.0))
    s1 = 1.0 / (1.0 + 10 ** (-elo1 / 400.0))
    return n * (s1 - s0) * (2 * mean - s0 - s1) / (2 * var)


def parse_opts(items):
    opts = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"bad option {item!r}, expected Name=Value")
        k, v = item.split("=", 1)
        opts[k] = v
    return opts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", help="engine for A and B (default for --engine-a/--engine-b)")
    ap.add_argument("--engine-a")
    ap.add_argument("--engine-b")
    ap.add_argument("--net-a")
    ap.add_argument("--net-b", help="default: same as --net-a")
    ap.add_argument("--opt-a", nargs="*", default=[], help="UCI options of A, Name=Value")
    ap.add_argument("--opt-b", nargs="*", default=[], help="UCI options of B, Name=Value")
    ap.add_argument("--nodes", type=int, default=20000)
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--movetime", type=int, default=0, help="ms per move")
    ap.add_argument("--pairs", type=int, default=10000, help="maximum number of game pairs")
    ap.add_argument("--sprt", nargs=2, type=float, metavar=("ELO0", "ELO1"),
                    help="stop when H0 (elo0) or H1 (elo1) is accepted")
    ap.add_argument("--min-pairs", type=int, default=20, help="no SPRT decision before this")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    ap.add_argument("--hash", type=int, default=16)
    ap.add_argument("--random-plies", type=int, default=6)
    ap.add_argument("--random-plies-min", type=int, default=2)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--adjudicate", type=int, default=3000, help="internal units")
    ap.add_argument("--full-info", action="store_true")
    ap.add_argument("--seed", type=int, default=int(time.time()))
    args = ap.parse_args()

    args.engine_a = args.engine_a or args.engine
    args.engine_b = args.engine_b or args.engine or args.engine_a
    if not args.engine_a:
        raise SystemExit("--engine or --engine-a is required")
    args.net_b = args.net_b or args.net_a
    args.opt_a, args.opt_b = parse_opts(args.opt_a), parse_opts(args.opt_b)
    args.go = (f"depth {args.depth}" if args.depth else
               f"movetime {args.movetime}" if args.movetime else f"nodes {args.nodes}")
    lower = math.log(args.beta / (1 - args.alpha))
    upper = math.log((1 - args.beta) / args.alpha)

    print(f"A: {args.engine_a} net={args.net_a} {args.opt_a}")
    print(f"B: {args.engine_b} net={args.net_b} {args.opt_b}")
    print(f"go {args.go}, {args.workers} workers"
          + (f", SPRT [{args.sprt[0]}, {args.sprt[1]}] bounds [{lower:.2f}, {upper:.2f}]"
             if args.sprt else ""), flush=True)

    pairs, decision, line = [], None, "no game finished"
    pool = mp.Pool(args.workers, initializer=worker_init, initargs=(args,))
    try:
        seeds = (args.seed * 1000003 + i for i in range(args.pairs))
        for r in pool.imap_unordered(play_pair, seeds):
            pairs.append(r)
            penta = pentanomial(pairs)
            games = [x for p in pairs for x in p]
            w, d, l = games.count(1), games.count(0), games.count(-1)
            elo, err = elo_estimate(penta)
            line = (f"pairs {len(pairs)}  games {len(games)}  A: +{w} ={d} -{l}  "
                    f"penta {penta}  Elo {elo:+.1f} +/- {err:.1f}")
            if args.sprt:
                v = llr(penta, *args.sprt)
                line += f"  LLR {v:.2f}"
                if len(pairs) < args.min_pairs:
                    pass
                elif v >= upper:
                    decision = "H1 accepted: A is better (by about elo1 or more)"
                elif v <= lower:
                    decision = "H0 accepted: A is not better than elo0"
            if len(pairs) % 10 == 0 or decision:
                print(line, flush=True)
            if decision:
                break
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        pool.terminate()
    print(line)
    print(decision or "no decision (maximum reached)")


if __name__ == "__main__":
    main()
