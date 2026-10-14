"""Self-play training data generator.

Plays Jieqi games with the engine, drawing the real identities of the dark
pieces at random for each game, and writes one line per position:

    <fen> | <score> | <result>

score: search score for the side to move, in internal units (RawScore).
result: 1 / 0 / -1 = win / draw / loss for the side to move.

Positions in check and positions whose best move is a capture are skipped
(their static evaluation is not meaningful), like in other NNUE trainers.

    python gen_data.py --engine ../src/maskfish --net current.nnue \
        --games 1000 --nodes 5000 --workers 8 --out data/gen1.txt \
        [--opt ChanceScaling=400 ...] [--seed 1]

Notes
* Each side only knows what it would know in a real game: the engine searches
  every position from the side to move's view, where the identity of a
  captured dark piece is known to its capturer only (its owner keeps it in the
  pool). The FEN written is that view. --full-info tells both sides.
* The safe reveal bonus of the search (RevealQuiet* options) is a preference
  between moves, not part of the value of a position: it is disabled while
  generating data, so that it is not learned by the network. Pass e.g.
  --opt RevealQuietBase=-12 RevealQuietSafety=24 RevealQuietHighValue=-12 to
  keep the engine's defaults.
* The first plies (--random-plies) are random legal moves for diversity; they
  are not written.
* Games are adjudicated: |score| >= --adjudicate for 8 plies in a row, a
  mate score, no legal move (loss), the third occurrence of a position
  (perpetual check loses, otherwise a draw), 80 plies without capture (the
  40-move rule) or --max-plies (draw).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import time

import jieqi_uci as U


def play_game(engine: U.Engine, rng: random.Random, args) -> list[tuple[str, int, int]]:
    ids = U.HiddenIdentities(rng)
    views: tuple[list[str], list[str]] = ([], [])   # moves as known by white / black
    repetitions = U.RepetitionTracker()
    records: list[tuple[str, int, int]] = []   # (fen, score, side to move)
    random_plies = rng.randint(args.random_plies_min, args.random_plies)
    result_white = 0       # +1 white wins, -1 black wins, 0 draw
    streak = 0
    no_capture = 0

    for ply in range(args.max_plies):
        engine.set_position(views[ply % 2])        # the view of the side to move
        fen, in_check = engine.board_info()
        stm = 1 if fen.split()[1] == "w" else -1

        repeated = repetitions.push(fen, in_check)
        if repeated is not None:
            result_white = repeated
            break

        if ply < random_plies:
            legal = engine.legal_moves()
            if not legal:
                result_white = -stm
                break
            move = rng.choice(legal)
        else:
            move, score, mate = engine.search(args.go)
            if move in ("(none)", "0000"):
                result_white = -stm
                break
            if score is not None:
                if mate:
                    result_white = stm if score > 0 else -stm
                    break
                if not in_check and not U.is_capture(fen, move):
                    records.append((fen, score, stm))
                streak = streak + 1 if abs(score) >= args.adjudicate else 0
                if streak >= 8:
                    result_white = stm if score > 0 else -stm
                    break

        no_capture = 0 if U.is_capture(fen, move) else no_capture + 1
        white_view, black_view = U.annotate_views(ids, move, args.full_info)
        views[0].append(white_view)
        views[1].append(black_view)
        if no_capture >= U.NO_CAPTURE_DRAW_PLIES:
            break

    return [(fen, score, result_white * stm) for fen, score, stm in records]


def worker(idx: int, args, games: int, out_path: str):
    rng = random.Random(args.seed * 1000003 + idx)
    # No safe reveal bonus in the scores written (see the notes above)
    opts = {"Threads": 1, "Hash": args.hash, "RawScore": "true",
            "RevealQuietBase": 0, "RevealQuietPhase": 0, "RevealQuietSafety": 0,
            "RevealQuietHighValue": 0, "RevealQuietDiversity": 0}
    if args.net:
        opts["EvalFile"] = os.path.abspath(args.net)
    opts.update(args.opt)
    engine = U.Engine(args.engine, opts)
    n = 0
    with open(out_path, "w") as f:
        for g in range(games):
            engine.send("ucinewgame")
            engine.isready()
            for fen, score, result in play_game(engine, rng, args):
                f.write(f"{fen} | {score} | {result}\n")
                n += 1
            f.flush()
            if idx == 0 and (g + 1) % 10 == 0:
                print(f"worker 0: {g + 1}/{games} games, {n} positions", flush=True)
    engine.quit()
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--net", help="EvalFile for the engine")
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--nodes", type=int, default=5000)
    ap.add_argument("--depth", type=int, default=0, help="use go depth N instead of nodes")
    ap.add_argument("--hash", type=int, default=16)
    ap.add_argument("--random-plies", type=int, default=8)
    ap.add_argument("--random-plies-min", type=int, default=2)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--adjudicate", type=int, default=3000)
    ap.add_argument("--full-info", action="store_true",
                    help="tell both sides the identity of a captured dark piece")
    ap.add_argument("--opt", nargs="*", default=[],
                    help="extra UCI options, Name=Value (e.g. ChanceScaling=400)")
    ap.add_argument("--seed", type=int, default=int(time.time()))
    args = ap.parse_args()
    args.opt = dict(o.split("=", 1) for o in args.opt)
    args.go = f"depth {args.depth}" if args.depth else f"nodes {args.nodes}"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    per = [args.games // args.workers + (i < args.games % args.workers) for i in range(args.workers)]
    parts = [f"{args.out}.part{i}" for i in range(args.workers)]
    with mp.Pool(args.workers) as pool:
        counts = pool.starmap(worker, [(i, args, per[i], parts[i]) for i in range(args.workers)])
    with open(args.out, "w") as out:
        for p in parts:
            with open(p) as f:
                out.write(f.read())
            os.unlink(p)
    print(f"{sum(counts)} positions written to {args.out}")


if __name__ == "__main__":
    main()
