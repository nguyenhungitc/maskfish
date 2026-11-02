"""Measures how well the search values the reveals (the chance nodes of
flip_search()), for one or more option sets of the engine.

A reveal is valued by the expectation of its outcomes, one per identity the
piece can turn out to be, weighted by the counts of the pool. How the search
gets that expectation within alpha-beta (Star1 bounds, ChanceMix, ChanceCV,
ChanceMaxLayers, ...) can bias it. For every position of a set, the move that
reveals a piece is valued two ways, in win-rate space w(v) = 1 / (1 + e^(-v/s)):

  root:      the score of the search of the position restricted to the move,
             go depth D searchmoves <move> (or go nodes N with --root-nodes,
             to compare the option sets at the same cost), with the option set
             under test;
  reference: sum_i p_i w(v_i), where v_i is the value of outcome i, the
             position after the reveal as identity i searched alone with an
             open window, go depth D-1, with the reference option set.

error = w(root) - reference. A positive bias means that the search overvalues
the reveals: the side that moves a dark piece reveals too readily. The report
gives, per option set, the bias with its 95% interval, the RMS error, the mean
|error|, the nodes of the root searches and, against the first option set,
the paired difference of the biases. The reference values and the root values
do not depend on each other, so the reference is cached (--ref-cache) and
reused by later runs with other option sets.

The positions are reveals that the engine chose in its own games (with real
hidden identities, each side seeing what it may know), from ply --min-ply on,
whose pool holds at least two kinds of piece, and that do not capture a dark
piece (its identity would be a second chance event). They are written to
--positions (one "fen | move" per line) and read from it by later runs.

    python chance_bias.py --engine ../src/maskfish --positions reveals.txt \\
        --count 72 --depth 10 --ref-cache reveals.ref.jsonl \\
        --config star1 --config "mix:ChanceMix=6,ChanceQsearchRep=true" \\
        --config "cap:ChanceMaxLayers=1,ChanceCV=3" --csv bias.csv

Every search runs on one thread after ucinewgame (an empty TT and empty
histories), so a run is reproducible on the same build and network.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import time

import jieqi_features as F
import jieqi_uci as U

DEFAULT_SCALING = 360.83524  # the engine's ChanceScaling default (search.cpp)


# ------------------------------------------------------------------ positions

def parse_fen(fen: str) -> tuple[dict, str, dict]:
    """(square name -> FEN letter, side to move 'w'/'b', pool letter -> count)"""
    board, side, pool = fen.split()[:3]
    squares = {}
    for i, row in enumerate(board.split("/")):
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
            else:
                squares["abcdefghi"[f] + str(9 - i)] = ch
                f += 1
    counts = {}
    if pool != "-":
        for i in range(0, len(pool) - 1, 2):
            counts[pool[i]] = int(pool[i + 1])
    return squares, side, counts


def mover_pool(fen: str) -> dict:
    """Identities the piece moved by the side to move can turn out to be."""
    _, side, counts = parse_fen(fen)
    return {k: n for k, n in counts.items() if n > 0 and k.isupper() == (side == "w")}


def is_measurable_reveal(fen: str, move: str) -> bool:
    squares, _, _ = parse_fen(fen)
    return (squares.get(move[:2]) in ("X", "x")
            and squares.get(move[2:4]) not in ("X", "x")
            and len(mover_pool(fen)) >= 2)


def generate(engine: U.Engine, rng: random.Random, count: int, nodes: int, min_ply: int,
             max_ply: int, random_plies: int, per_game: int) -> list[tuple[str, str]]:
    """Reveals chosen by the engine in its own games (see the module doc)."""
    positions, seen = [], set()
    games = 0
    while len(positions) < count:
        games += 1
        ids = U.HiddenIdentities(rng)
        views = ([], [])  # the moves as each side sees them (F.WHITE, F.BLACK)
        engine.send("ucinewgame")
        found = 0
        for ply in range(max_ply):
            stm = F.WHITE if ply % 2 == 0 else F.BLACK
            engine.set_position(views[stm])
            fen, _ = engine.board_info()
            if engine.rule() not in (None, "none"):
                break
            legal = engine.legal_moves()
            if not legal:
                break
            if ply < random_plies:
                move = rng.choice(legal)
            else:
                move, _, _ = engine.search(f"nodes {nodes}")
                if move not in legal:
                    break
            if (ply >= min_ply and found < per_game and (fen, move) not in seen
                    and is_measurable_reveal(fen, move)):
                positions.append((fen, move))
                seen.add((fen, move))
                found += 1
                print(f"  position {len(positions)}/{count} (game {games}, ply {ply}): {move}",
                      flush=True)
                if len(positions) >= count:
                    break
            white_view, black_view = U.annotate_views(ids, move)
            views[F.WHITE].append(white_view)
            views[F.BLACK].append(black_view)
    return positions


def read_positions(path: str) -> list[tuple[str, str]]:
    positions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                fen, move = (s.strip() for s in line.split("|"))
                positions.append((fen, move))
    return positions


def write_positions(path: str, positions: list[tuple[str, str]]):
    with open(path, "w") as f:
        for fen, move in positions:
            f.write(f"{fen} | {move}\n")


# ------------------------------------------------------------------ searches

def winrate(v: float, scaling: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30000.0, min(30000.0, v)) / scaling))


def search(engine: U.Engine, go: str) -> tuple[float | None, int]:
    """(win rate of the side to move, nodes) of a search after ucinewgame.
    The score is the one of the last info line of the search that is not a
    bound; a mate counts as a win rate of 1 or 0."""
    engine.send("go " + go)
    score, bound, nodes = None, None, 0
    for line in engine.wait_for("bestmove"):
        t = line.split()
        if not line.startswith("info") or "score" not in t:
            continue
        i = t.index("score")
        if "nodes" in t:
            nodes = int(t[t.index("nodes") + 1])
        s = ("cp", int(t[i + 2])) if t[i + 1] == "cp" else ("mate", int(t[i + 2]))
        if "lowerbound" in t or "upperbound" in t:
            bound = s
        else:
            score = s
    return (score if score is not None else bound), nodes


def to_winrate(score, scaling: float) -> float | None:
    if score is None:
        return None
    kind, v = score
    if kind == "mate":
        return 1.0 if v > 0 else 0.0
    return winrate(v, scaling)


def reference(engine: U.Engine, fen: str, move: str, depth: int, scaling: float) -> dict:
    """Expectation of the outcomes of the reveal, from the point of view of the
    side that reveals, each outcome searched alone with an open window."""
    pool = mover_pool(fen)
    total = sum(pool.values())
    values, expected, nodes = {}, 0.0, 0
    for letter, n in sorted(pool.items()):
        engine.send("ucinewgame")
        engine.set_position([move + letter], fen=fen)
        score, k = search(engine, f"depth {depth}")
        w = to_winrate(score, scaling)
        if w is None:
            raise RuntimeError(f"no score after {move}{letter} in {fen}")
        values[letter] = {"count": n, "winrate": 1.0 - w}  # the opponent searched it
        expected += n / total * (1.0 - w)
        nodes += k
    return {"fen": fen, "move": move, "depth": depth, "scaling": scaling,
            "reference": expected, "outcomes": values, "nodes": nodes}


def root_value(engine: U.Engine, fen: str, move: str, limit: str,
               scaling: float) -> tuple[float, int]:
    """limit: "depth D" or "nodes N"."""
    engine.send("ucinewgame")
    engine.set_position([], fen=fen)
    score, nodes = search(engine, f"{limit} searchmoves {move}")
    w = to_winrate(score, scaling)
    if w is None:
        raise RuntimeError(f"no score for {move} in {fen}")
    return w, nodes


# ------------------------------------------------------------------ options

def parse_options(text: str) -> dict:
    """"Name=value,Name=value" -> dict"""
    opts = {}
    for item in filter(None, (s.strip() for s in text.split(","))):
        if "=" not in item:
            raise SystemExit(f"option '{item}': expected Name=value")
        k, v = item.split("=", 1)
        opts[k.strip()] = v.strip()
    return opts


def parse_config(text: str) -> tuple[str, dict]:
    """"name" or "name:Name=value,Name=value" -> (name, options)"""
    name, _, rest = text.partition(":")
    return name.strip(), parse_options(rest)


def ref_key(fen: str, move: str, depth: int, scaling: float, ref_opts: dict) -> str:
    return json.dumps([fen, move, depth, scaling, sorted(ref_opts.items())])


# ------------------------------------------------------------------ report

def interval(xs: list[float]) -> tuple[float, float]:
    """(mean, half-width of its 95% interval)"""
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return m, float("nan")
    return m, 1.96 * statistics.stdev(xs) / math.sqrt(len(xs))


def report(names: list[str], errors: dict, nodes: dict, ref_nodes: list[int]):
    base = names[0]
    print()
    print(f"{len(errors[base])} reveals, errors in expected score (win rate of the side that "
          f"reveals)")
    print(f"{'option set':<16} {'bias':>8} {'95%':>8} {'RMS':>7} {'mean|e|':>8} "
          f"{'nodes':>10} {'x nodes':>8}   {'bias - ' + base:>14}")
    base_nodes = statistics.fmean(nodes[base])
    for name in names:
        e = errors[name]
        bias, half = interval(e)
        rms = math.sqrt(statistics.fmean(x * x for x in e))
        mae = statistics.fmean(abs(x) for x in e)
        n = statistics.fmean(nodes[name])
        line = (f"{name:<16} {bias:+8.4f} {half:8.4f} {rms:7.4f} {mae:8.4f} "
                f"{n:10.0f} {n / base_nodes:8.2f}")
        if name != base:
            d, dh = interval([a - b for a, b in zip(e, errors[base])])
            line += f"   {d:+8.4f} ±{dh:.4f}"
        print(line)
    print(f"{'reference':<16} {'':>8} {'':>8} {'':>7} {'':>8} {statistics.fmean(ref_nodes):10.0f}"
          f"   (nodes of all the outcomes of a reveal)")


# ------------------------------------------------------------------ main

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="../src/maskfish", help="engine binary")
    ap.add_argument("--net", help="network (EvalFile) of every engine process")
    ap.add_argument("--opt", default="", help="options of every engine process, "
                    "\"Name=value,Name=value\"")
    ap.add_argument("--hash", type=int, default=64, help="hash (MB) of every search")
    ap.add_argument("--config", action="append", default=[],
                    help="option set under test, \"name\" or \"name:Name=value,...\" "
                         "(repeatable; the first one is the base of the paired differences; "
                         "default: the engine defaults, named default)")
    ap.add_argument("--ref-opt", default="", help="options of the reference searches "
                    "(default: the engine defaults)")
    ap.add_argument("--depth", type=int, default=10,
                    help="depth of the root searches; the outcomes are searched at depth - 1")
    ap.add_argument("--root-nodes", type=int, default=0,
                    help="root searches limited to this many nodes instead of --depth, to "
                         "compare the option sets at the same cost (the outcomes of the "
                         "reference are still searched at --depth - 1)")
    ap.add_argument("--scaling", type=float, default=DEFAULT_SCALING,
                    help="s of the win rate 1 / (1 + e^(-v/s)) (default: the engine's "
                         "ChanceScaling default)")
    ap.add_argument("--positions", required=True,
                    help="positions file (\"fen | move\" per line): read if it exists, "
                         "otherwise generated and written")
    ap.add_argument("--ref-cache", help="cache of the reference values (JSON lines)")
    ap.add_argument("--csv", help="per-position results")
    gen = ap.add_argument_group("generation of the positions")
    gen.add_argument("--count", type=int, default=72, help="number of reveals")
    gen.add_argument("--gen-nodes", type=int, default=20000, help="nodes per move of the games")
    gen.add_argument("--min-ply", type=int, default=16, help="first ply of a reveal taken")
    gen.add_argument("--max-ply", type=int, default=200, help="plies at most per game")
    gen.add_argument("--random-plies", type=int, default=4,
                     help="random moves at the start of every game")
    gen.add_argument("--per-game", type=int, default=3, help="reveals taken per game at most")
    gen.add_argument("--seed", type=int, default=1)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    common = {"Threads": "1", "Hash": str(args.hash), **parse_options(args.opt)}
    if args.net:
        common["EvalFile"] = os.path.abspath(args.net)
    configs = [parse_config(c) for c in args.config] or [("default", {})]
    names = [n for n, _ in configs]
    if len(set(names)) != len(names):
        raise SystemExit("the option sets need distinct names")
    ref_opts = parse_options(args.ref_opt)

    ref_engine = U.Engine(args.engine, options={**common, **ref_opts})

    if os.path.exists(args.positions):
        positions = read_positions(args.positions)
        print(f"{len(positions)} positions read from {args.positions}")
    else:
        print(f"generating {args.count} positions")
        positions = generate(ref_engine, random.Random(args.seed), args.count, args.gen_nodes,
                             args.min_ply, args.max_ply, args.random_plies, args.per_game)
        write_positions(args.positions, positions)
        print(f"{len(positions)} positions written to {args.positions}")
    bad = [(f, m) for f, m in positions if not is_measurable_reveal(f, m)]
    if bad:
        raise SystemExit(f"not a measurable reveal (a dark piece moved to a square that is not "
                         f"dark, with two kinds of piece in the pool): {bad[0]}")

    cache = {}
    if args.ref_cache and os.path.exists(args.ref_cache):
        with open(args.ref_cache) as f:
            for line in f:
                r = json.loads(line)
                cache[ref_key(r["fen"], r["move"], r["depth"], r["scaling"],
                              dict(r.get("ref_opts", {})))] = r

    engines = {name: U.Engine(args.engine, options={**common, **opts}) for name, opts in configs}
    errors = {name: [] for name in names}
    nodes = {name: [] for name in names}
    ref_nodes, rows = [], []
    limit = f"nodes {args.root_nodes}" if args.root_nodes else f"depth {args.depth}"
    start = time.time()
    try:
        for idx, (fen, move) in enumerate(positions, 1):
            key = ref_key(fen, move, args.depth - 1, args.scaling, ref_opts)
            ref = cache.get(key)
            if ref is None:
                ref = reference(ref_engine, fen, move, args.depth - 1, args.scaling)
                ref["ref_opts"] = sorted(ref_opts.items())
                cache[key] = ref
                if args.ref_cache:
                    with open(args.ref_cache, "a") as f:
                        f.write(json.dumps(ref) + "\n")
            ref_nodes.append(ref["nodes"])

            parts = []
            for name in names:
                w, k = root_value(engines[name], fen, move, limit, args.scaling)
                errors[name].append(w - ref["reference"])
                nodes[name].append(k)
                parts.append(f"{name} {w - ref['reference']:+.4f}")
                rows.append({"index": idx, "fen": fen, "move": move, "option_set": name,
                             "reference": f"{ref['reference']:.6f}", "root": f"{w:.6f}",
                             "error": f"{w - ref['reference']:+.6f}", "nodes": k})
            print(f"{idx}/{len(positions)} {move} reference {ref['reference']:.4f}: "
                  + ", ".join(parts) + f"  ({time.time() - start:.0f} s)", flush=True)
    finally:
        for e in engines.values():
            e.quit()
        ref_engine.quit()

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    report(names, errors, nodes, ref_nodes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
