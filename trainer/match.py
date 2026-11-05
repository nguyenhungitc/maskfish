"""Plays Jieqi games between two engines / networks / option sets, with an
SPRT, to decide whether a change is an improvement.

    python match.py --engine ../src/maskfish \
        --net-a runs/gen2/epoch_100.nnue --net-b nets/gen1.nnue \
        --nodes 20000 --workers 15 --sprt 0 5

    python match.py --engine ../src/maskfish --net-a nets/gen1.nnue \
        --opt-a LMRTweaks=true --nodes 20000 --sprt 0 3

    python match.py --engine ../src/maskfish --net-a nets/gen1.nnue \
        --engine-b old/maskfish --tc 10+0.1 --workers 1 --sprt 0 5

A is the version being tested, B the reference (by default B uses the same
engine and network as A). Results are reported for A.

Jieqi luck is reduced by playing games in pairs: both games of a pair use
the same hidden identities and the same random opening, with colors swapped.
The SPRT uses the pentanomial statistics of the pairs (logistic Elo,
normal approximation as in fishtest).

Information model: each engine only gets what its side knows. Reveals are
public; when a dark piece is captured, the capturer is told its identity but
its owner is not (use --full-info to tell both).

Games end by mate / no legal move, by adjudication (the side to move
reports that it is mated, or both sides agree that |score| >= --adjudicate for
4 moves each; a mate claimed by the winning side alone is played out), by the
rules of the engine (third occurrence of a position: a draw, or a loss for
perpetual check or chase; 40-move rule: a draw after 80 plies without capture,
or a loss for a side without legal move), as judged by B with the engine
command "rule", or as a draw after --max-plies.

--referee ENGINE (Maskfish) gives the board state (FEN, check, legal moves)
and judges the rules instead, in its own process, with the view of the side
to move. It is needed for an opponent without the Maskfish commands "d",
"go perft" and "rule", e.g. a reference engine:

    python match.py --engine-a ../src/maskfish --net-a nets/gen3.nnue \
        --engine-b ../jieqi/ref-engine --net-b ../jieqi/ref-engine.nnue \
        --referee ../src/maskfish --movetime 200 --pairs 1000

With --referee, every move must be among the referee's legal moves: an
illegal move (or one in another notation) loses the game. A search without
bestmove after --engine-timeout seconds loses the game too; a referee (or the
engine giving the board state) without answer within that time voids the
pair, which is played again. In both cases the engine is killed and every
engine of the worker is started again. Each is printed as a warning; more
than --max-failures in one worker stops the match with an error.
--go gives any limit of the search instead of --nodes / --depth / --movetime
(e.g. "depth 12"), for an engine that does not have the usual ones.
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
              args, referee: U.Engine | None = None, board: U.Engine | None = None) -> int:
    """Returns +1 white wins, 0 draw, -1 black wins. referee: the engine that
    judges the rules (see U.RuleJudge), by default the engine to move.
    board: an engine outside the game (--referee) that gives the board state
    and judges the rules, with the view of the side to move."""
    global _RESTART
    ids = copy.deepcopy(ids)
    views = ([], [])                 # move lists sent to the white / black engine
    engines = (white, black)
    for e in engines:
        e.send("ucinewgame")
        e.isready()

    last_scores = {F.WHITE: [], F.BLACK: []}   # from white's point of view
    # Clocks in ms (--tc), white and black
    clock = [args.tc_base, args.tc_base] if args.tc else None
    no_capture = 0
    judge = U.RuleJudge()
    for ply in range(args.max_plies):
        stm = ply % 2
        eng = engines[stm]
        eng.set_position(views[stm])
        bd = board or eng
        if bd is not eng:
            bd.set_position(views[stm])
        fen, in_check = bd.board_info()
        verdict = (bd.rule() if board is not None
                   else U.referee_verdict(referee, engines, views, stm))
        ended = judge.result(fen, in_check, no_capture, verdict,
                             lambda: bool(bd.legal_moves()))
        if ended is not None:
            return ended[0]
        if ply < len(opening):
            move = opening[ply]
            score, mate = None, False
        else:
            if clock:
                go = (f"wtime {max(1, int(clock[0]))} btime {max(1, int(clock[1]))} "
                      f"winc {args.tc_inc} binc {args.tc_inc}")
                start = time.monotonic()
                move, score, mate = eng.search(go)
                clock[stm] -= (time.monotonic() - start) * 1000
                if clock[stm] < 0:                  # lost on time
                    return -1 if stm == F.WHITE else 1
                clock[stm] += args.tc_inc
            else:
                try:
                    move, score, mate = eng.search(args.go)
                except U.EngineTimeout as e:     # loses, as on time; new engines for the next game
                    note_failure(str(e))
                    _RESTART = True
                    return -1 if stm == F.WHITE else 1
            if move in ("(none)", "0000"):
                return -1 if stm == F.WHITE else 1
            if board is not None and move not in bd.legal_moves():
                note_failure(f"{eng.path} played {move!r} in {fen}: the game is lost")
                return -1 if stm == F.WHITE else 1
        capture = U.is_capture(fen, move)
        w, b = U.annotate_views(ids, move, args.full_info)
        views[0].append(w)
        views[1].append(b)

        if score is not None:
            sign = 1 if stm == F.WHITE else -1
            # Only the side that is mated ends the game with its score (it
            # resigns): a mate claimed by the winning side is played out, or
            # adjudicated below if the other side agrees, since a wrong mate
            # score (e.g. a bug of the version tested) would win the game.
            if mate and score < 0:
                return -sign
            last_scores[stm] = (last_scores[stm] + [sign * score])[-4:]
            sw, sb = last_scores[F.WHITE], last_scores[F.BLACK]
            if len(sw) == 4 and len(sb) == 4:
                if min(sw + sb) >= args.adjudicate:
                    return 1
                if max(sw + sb) <= -args.adjudicate:
                    return -1
        # The 40-move rule is judged before the next move (a side without a
        # legal move then loses, as in the engine)
        no_capture = 0 if capture else no_capture + 1
    return 0


def make_engine(path, net, opts, hash_mb, timeout=None):
    o = {"Threads": 1, "Hash": hash_mb}
    if net:
        o["EvalFile"] = os.path.abspath(net)
    o.update(opts)
    # RawScore only exists in this engine (adjudication compares internal units)
    return U.Engine(path, o, optional={"RawScore": "true"}, timeout=timeout)


_FAILURES = 0
_RESTART = False


def note_failure(what: str):
    """An illegal move or a timeout in this worker (see the header)."""
    global _FAILURES
    _FAILURES += 1
    if _FAILURES <= 10 or _FAILURES % 100 == 0:
        print(f"warning: pid {os.getpid()}: {what} ({_FAILURES} engine failure(s))", flush=True)
    if _FAILURES > _ARGS.max_failures:
        raise RuntimeError(f"more than {_ARGS.max_failures} engine failures in one worker "
                           f"(illegal moves or timeouts), last: {what}")


def worker_init(args):
    global _ENGINES, _BOARD, _ARGS
    _ARGS = args
    tmo = args.engine_timeout
    _ENGINES = (make_engine(args.engine_a, args.net_a, args.opt_a, args.hash, tmo),
                make_engine(args.engine_b, args.net_b, args.opt_b, args.hash, tmo))
    _BOARD = U.Engine(args.referee, {"Threads": 1, "Hash": 1}, timeout=tmo) if args.referee else None


def play_pair(seed: int) -> tuple[int, int]:
    """Two games from the same opening and identities, colors swapped.
    Returns the results for A of both games (+1 / 0 / -1)."""
    global _RESTART
    while True:
        try:
            a, b = _ENGINES
            rng = random.Random(seed)
            ids, opening = make_opening(_BOARD or a, rng,
                                        rng.randint(_ARGS.random_plies_min, _ARGS.random_plies))
            r1 = play_game(a, b, ids, opening, _ARGS, referee=b, board=_BOARD)        # A white
            a, b = restart_if_needed()
            r2 = -play_game(b, a, ids, opening, _ARGS, referee=b, board=_BOARD)       # A black
            restart_if_needed()
            return r1, r2
        except U.EngineTimeout as e:    # outside a search (board state, rules): the pair is void
            note_failure(f"{e}: the pair is played again")
            _RESTART = True
            restart_if_needed()


def restart_if_needed():
    """New engine processes after a timeout (one of them was killed)."""
    global _RESTART
    if _RESTART:
        _RESTART = False
        for e in _ENGINES:
            e.quit()
        if _BOARD is not None:
            _BOARD.quit()
        worker_init(_ARGS)
    return _ENGINES


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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", help="engine for A and B (default for --engine-a/--engine-b)")
    ap.add_argument("--engine-a")
    ap.add_argument("--engine-b")
    ap.add_argument("--net-a")
    ap.add_argument("--net-b", help="default: same as --net-a when B is the same engine, "
                                     "otherwise B's own default network")
    ap.add_argument("--opt-a", nargs="*", default=[], help="UCI options of A, Name=Value")
    ap.add_argument("--opt-b", nargs="*", default=[], help="UCI options of B, Name=Value")
    ap.add_argument("--nodes", type=int, default=20000)
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--movetime", type=int, default=0, help="ms per move")
    ap.add_argument("--go", dest="go_args", help="limit of every search, any 'go' arguments "
                    "(e.g. 'depth 12'); overrides --nodes / --depth / --movetime")
    ap.add_argument("--engine-timeout", type=float, default=600,
                    help="seconds to wait for an answer of an engine before it is killed: in a "
                         "search it loses, otherwise the pair is played again (0: no limit)")
    ap.add_argument("--max-failures", type=int, default=20,
                    help="illegal moves / timeouts allowed per worker before stopping with an error")
    ap.add_argument("--tc", help="time control 'base+inc' in seconds (e.g. 10+0.1), clocks "
                    "kept by match.py: tests the time management; a flag fall loses")
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
    ap.add_argument("--referee", help="Maskfish binary giving the board state and judging the rules "
                    "in its own process (needed against an engine without d / go perft / rule)")
    ap.add_argument("--seed", type=int, default=int(time.time()))
    return ap


def finalize_args(args):
    """Fills the defaults that depend on other arguments (engines, network of
    B, options, the go command). Returns args."""
    args.engine_a = args.engine_a or args.engine
    args.engine_b = args.engine_b or args.engine or args.engine_a
    if not args.engine_a:
        raise SystemExit("--engine or --engine-a is required")
    # another engine keeps its own network unless --net-b is given
    if not args.net_b and os.path.realpath(args.engine_b) == os.path.realpath(args.engine_a):
        args.net_b = args.net_a
    if not isinstance(args.opt_a, dict):
        args.opt_a = parse_opts(args.opt_a)
    if not isinstance(args.opt_b, dict):
        args.opt_b = parse_opts(args.opt_b)
    args.go = args.go_args or (f"depth {args.depth}" if args.depth else
                               f"movetime {args.movetime}" if args.movetime else f"nodes {args.nodes}")
    if args.tc:
        base, _, inc = args.tc.partition("+")
        args.tc_base, args.tc_inc = int(float(base) * 1000), int(float(inc or 0) * 1000)
        args.go = f"tc {args.tc}"
    return args


def play_pairs(args, count: int, seed: int, start: int = 0):
    """Plays 'count' game pairs (args from finalize_args()) on args.workers
    processes and yields the results (r1, r2) for A as they finish. Pair i
    (start <= i < start + count) uses the opening and identities of seed
    seed * 1000003 + i. Closing the generator (break) stops the games at once."""
    pool = mp.Pool(args.workers, initializer=worker_init, initargs=(args,))
    try:
        seeds = (seed * 1000003 + i for i in range(start, start + count))
        yield from pool.imap_unordered(play_pair, seeds)
    finally:
        pool.terminate()
        pool.join()


def main():
    args = finalize_args(build_parser().parse_args())
    lower = math.log(args.beta / (1 - args.alpha))
    upper = math.log((1 - args.beta) / args.alpha)

    print(f"A: {args.engine_a} net={args.net_a} {args.opt_a}")
    print(f"B: {args.engine_b} net={args.net_b} {args.opt_b}")
    print(f"go {args.go}, {args.workers} workers"
          + (f", SPRT [{args.sprt[0]}, {args.sprt[1]}] bounds [{lower:.2f}, {upper:.2f}]"
             if args.sprt else ""), flush=True)

    # check both engines and their options once here (an error in the pool
    # initializer would make multiprocessing respawn the workers forever)
    try:
        for path, net, opts in ((args.engine_a, args.net_a, args.opt_a),
                                (args.engine_b, args.net_b, args.opt_b)):
            make_engine(path, net, opts, 1).quit()
    except U.UnknownOption as e:
        raise SystemExit(f"error: {e}")

    pairs, decision, line = [], None, "no game finished"
    games_iter = play_pairs(args, args.pairs, args.seed)
    try:
        for r in games_iter:
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
        games_iter.close()
    print(line)
    print(decision or "no decision (maximum reached)")


if __name__ == "__main__":
    main()
