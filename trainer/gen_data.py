"""Training data generator: plays Jieqi games and writes one line per position:

    <fen> | <score> | <result>

score: search score for the side to move, in the internal units of the engine
that scored it (Maskfish with RawScore; other engines print their own units).
result: 1 / 0 / -1 = win / draw / loss for the side to move.

Three roles, any of them may be the same engine process:

  player   (--engine)   chooses the moves of both sides
  referee  (--referee)  board state: FEN, side in check, legal moves. Default:
                        the player. When the player is another engine
                        (a reference engine), Maskfish is the referee, so the FEN written
                        is in Maskfish's format.
  teacher  (--teacher)  scores the positions written. Default: the player's own
                        search score. With a teacher, every position kept is
                        also searched by the teacher (same moves, same
                        information) and the teacher's score is written.

Typical uses (see scripts/pipeline.sh):

  # gen0: reference engine self-play, Maskfish only as the referee
  python gen_data.py --engine ../jieqi/ref-engine --net ../jieqi/ref-engine.nnue \\
      --opt Name=value --referee ../src/maskfish --nodes 10000 ...
  # distillation: Maskfish self-play, positions scored by the reference engine
  python gen_data.py --engine ../src/maskfish --net nets/gen1.nnue --nodes 5000 \\
      --teacher ../jieqi/ref-engine --teacher-net ... --teacher-nodes 20000 ...
  # self: Maskfish self-play, own scores
  python gen_data.py --engine ../src/maskfish --net nets/gen5.nnue --nodes 10000 ...

Positions in check and positions whose best move (the scorer's) is a capture
are skipped (their static evaluation is not meaningful), like in other NNUE
trainers; so are positions the scorer reports as mate.

--out data/gen1.txt.zst writes a zstd compressed file (5 to 8 times smaller,
read as is by prepare_data.py, fit_scaling.py and rescore.py; see textio.py).

Notes
* Each side only knows what it would know in a real game: the engines search
  every position from the side to move's view, where the identity of a
  captured dark piece is known to its capturer only (its owner keeps it in the
  pool). The FEN written is that view. --full-info tells both sides.
* The safe reveal bonus (RevealQuiet* options) is a preference between moves,
  not part of the value of a position: it is set to 0 in every engine that has
  these options, so that it is not learned by the network (--opt and
  --teacher-opt override it). RawScore is set in every engine that has it.
* The first plies (--random-plies) are random legal moves for diversity; they
  are not written.
* Games are adjudicated: |player score| >= --adjudicate for 8 plies in a row,
  a mate score, no legal move (loss), the third occurrence of a position
  (perpetual check loses, otherwise a draw), 80 plies without capture (the
  40-move rule) or --max-plies (draw).
* Resumable: every worker appends finished games to OUT.partK and counts them
  in OUT.partK.games; running the same command again only plays the games
  still missing, then joins the parts into OUT.
* Another engine is checked: a move of a player other than the referee that
  is not among the referee's legal moves discards the game; a teacher's best
  move that is not legal leaves the position unscored; an engine (player,
  teacher or referee) that gives no answer within --engine-timeout seconds is
  killed, the game is discarded and new engine processes are started. Each is
  printed as a warning; more than --max-failures in one worker stops gen_data
  with an error (an engine that does not speak this protocol, ignores the
  limit of --go, or hangs).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import shutil
import time

import jieqi_uci as U
from textio import open_text

# set in every engine that has them (see the notes above)
SCORE_OPTS = {"RawScore": "true", "RevealQuietBase": 0, "RevealQuietPhase": 0,
              "RevealQuietSafety": 0, "RevealQuietHighValue": 0, "RevealQuietDiversity": 0}


def same_engine(a: str | None, b: str | None) -> bool:
    return bool(a and b) and os.path.realpath(a) == os.path.realpath(b)


class Failures:
    """Illegal moves and timeouts of the engines of one worker (see Notes)."""

    def __init__(self, worker: int, limit: int):
        self.worker, self.limit, self.count = worker, limit, 0

    def note(self, what: str):
        self.count += 1
        if self.count <= 10 or self.count % 100 == 0:
            print(f"warning: worker {self.worker}: {what} ({self.count} engine failure(s))", flush=True)
        if self.count > self.limit:
            raise RuntimeError(f"worker {self.worker}: more than {self.limit} engine failures "
                               f"(illegal moves or timeouts), last: {what}")


class Engines:
    """The player / referee / teacher processes of one worker."""

    def __init__(self, args, failures: Failures | None = None):
        self.failures = failures or Failures(0, args.max_failures)
        tmo = args.engine_timeout
        popts = {"Threads": 1, "Hash": args.hash}
        if args.net:
            popts["EvalFile"] = os.path.abspath(args.net)
        popts.update(args.opt)
        self.player = U.Engine(args.engine, popts, optional=SCORE_OPTS, timeout=tmo)

        self.referee = self.player
        if args.referee and not same_engine(args.referee, args.engine):
            ropts = {"Threads": 1, "Hash": 1}
            if args.referee_net:
                ropts["EvalFile"] = os.path.abspath(args.referee_net)
            self.referee = U.Engine(args.referee, ropts, timeout=tmo)

        self.teacher = None
        if args.teacher:
            topts = {"Threads": 1, "Hash": args.teacher_hash}
            if args.teacher_net:
                topts["EvalFile"] = os.path.abspath(args.teacher_net)
            topts.update(args.teacher_opt)
            self.teacher = U.Engine(args.teacher, topts, optional=SCORE_OPTS, timeout=tmo)

    def all(self):
        out = []
        for e in (self.player, self.referee, self.teacher):
            if e is not None and all(e is not o for o in out):
                out.append(e)
        return out

    def new_game(self):
        for e in self.all():
            e.send("ucinewgame")
            e.isready()

    def quit(self):
        for e in self.all():
            e.quit()


def play_game(eng: Engines, rng: random.Random, args) -> list[tuple[str, int, int]]:
    ids = U.HiddenIdentities(rng)
    views: tuple[list[str], list[str]] = ([], [])   # moves as known by white / black
    repetitions = U.RepetitionTracker()
    records: list[tuple[str, int, int]] = []   # (fen, score, side to move)
    random_plies = rng.randint(args.random_plies_min, args.random_plies)
    result_white = 0       # +1 white wins, -1 black wins, 0 draw
    streak = 0
    no_capture = 0

    for ply in range(args.max_plies):
        view = views[ply % 2]                       # the view of the side to move
        eng.referee.set_position(view)
        fen, in_check = eng.referee.board_info()
        stm = -1 if fen.split()[1] == "b" else 1   # as the engine reads a FEN

        repeated = repetitions.push(fen, in_check)
        if repeated is not None:
            result_white = repeated
            break

        if ply < random_plies:
            legal = eng.referee.legal_moves()
            if not legal:
                result_white = -stm
                break
            move = rng.choice(legal)
        else:
            if eng.player is not eng.referee:
                eng.player.set_position(view)
            move, score, mate = eng.player.search(args.go)
            if move in ("(none)", "0000"):
                result_white = -stm
                break
            # another engine as the player: its move must be legal for the referee
            if eng.player is not eng.referee and move not in eng.referee.legal_moves():
                raise U.IllegalMove(f"{eng.player.path} played {move!r} in {fen}")
            if score is not None:
                if mate:
                    result_white = stm if score > 0 else -stm
                    break
                if not in_check:
                    if eng.teacher is None:
                        if not U.is_capture(fen, move):
                            records.append((fen, score, stm))
                    else:
                        eng.teacher.set_position(view)
                        tmove, tscore, tmate = eng.teacher.search(args.teacher_go)
                        if (tmove not in ("(none)", "0000")
                                and tmove not in eng.referee.legal_moves()):
                            eng.failures.note(f"teacher {eng.teacher.path} answered {tmove!r} "
                                              f"in {fen}: position not scored")
                        elif (tscore is not None and not tmate and tmove not in ("(none)", "0000")
                                and not U.is_capture(fen, tmove)):
                            records.append((fen, tscore, stm))
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


def read_count(path: str) -> int:
    try:
        with open(path) as f:
            return int(f.read().strip() or 0)
    except (OSError, ValueError):
        return 0


def worker(idx: int, args, games: int, out_path: str) -> int:
    done_path = out_path + ".games"
    done = read_count(done_path)
    if done >= games:
        return 0
    # a different random stream for the games still to play after a restart
    rng = random.Random(args.seed * 1000003 + idx * 7919 + done * 104729)
    failures = Failures(idx, args.max_failures)
    eng = Engines(args, failures)
    n = 0
    g = done
    with open(out_path, "a") as f:
        while g < games:
            try:
                eng.new_game()
                records = play_game(eng, rng, args)
            except (U.IllegalMove, U.EngineTimeout) as e:
                failures.note(f"game discarded: {e}")
                if isinstance(e, U.EngineTimeout) or not all(x.alive() for x in eng.all()):
                    eng.quit()
                    eng = Engines(args, failures)
                continue
            g += 1
            lines = [f"{fen} | {score} | {result}\n" for fen, score, result in records]
            f.writelines(lines)
            f.flush()
            n += len(lines)
            with open(done_path + ".tmp", "w") as d:
                d.write(str(g))
            os.replace(done_path + ".tmp", done_path)
            if idx == 0 and g % 10 == 0:
                print(f"worker 0: {g}/{games} games, {n} new positions", flush=True)
    eng.quit()
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, help="player: plays the moves of both sides")
    ap.add_argument("--net", help="EvalFile for the player")
    ap.add_argument("--opt", nargs="*", default=[], help="UCI options of the player, Name=Value")
    ap.add_argument("--referee", help="engine giving the FEN / checks / legal moves (default: the player)")
    ap.add_argument("--referee-net", help="EvalFile for the referee (Maskfish does not need one)")
    ap.add_argument("--teacher", help="engine scoring the positions written (default: the player)")
    ap.add_argument("--teacher-net")
    ap.add_argument("--teacher-opt", nargs="*", default=[], help="UCI options of the teacher")
    ap.add_argument("--teacher-nodes", type=int, default=20000)
    ap.add_argument("--teacher-depth", type=int, default=0)
    ap.add_argument("--teacher-go", help="limit of the teacher's search, any 'go' arguments "
                    "(e.g. 'movetime 50'); overrides --teacher-nodes / --teacher-depth")
    ap.add_argument("--teacher-hash", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--nodes", type=int, default=5000)
    ap.add_argument("--depth", type=int, default=0, help="use go depth N instead of nodes")
    ap.add_argument("--go", help="limit of the player's search, any 'go' arguments "
                    "(e.g. 'movetime 50'); overrides --nodes / --depth")
    ap.add_argument("--engine-timeout", type=float, default=600,
                    help="seconds to wait for an answer of an engine (a search, or a command of "
                         "the referee) before it is killed (0: no limit)")
    ap.add_argument("--max-failures", type=int, default=20,
                    help="illegal moves / timeouts allowed per worker before stopping with an error")
    ap.add_argument("--hash", type=int, default=16)
    ap.add_argument("--random-plies", type=int, default=8)
    ap.add_argument("--random-plies-min", type=int, default=2)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--adjudicate", type=int, default=3000)
    ap.add_argument("--full-info", action="store_true",
                    help="tell both sides the identity of a captured dark piece")
    ap.add_argument("--seed", type=int, default=int(time.time()))
    args = ap.parse_args()
    args.opt = dict(o.split("=", 1) for o in args.opt)
    args.teacher_opt = dict(o.split("=", 1) for o in args.teacher_opt)
    args.go = args.go or (f"depth {args.depth}" if args.depth else f"nodes {args.nodes}")
    args.teacher_go = args.teacher_go or (f"depth {args.teacher_depth}" if args.teacher_depth
                                          else f"nodes {args.teacher_nodes}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    try:   # check the engines and options once (see match.py)
        Engines(args).quit()
    except U.UnknownOption as e:
        raise SystemExit(f"error: {e}")
    # resuming: keep the split of the games between workers of the first run
    split_path = args.out + ".workers"
    if read_count(split_path) > 0:
        args.workers = read_count(split_path)
    else:
        with open(split_path, "w") as f:
            f.write(str(args.workers))
    per = [args.games // args.workers + (i < args.games % args.workers) for i in range(args.workers)]
    parts = [f"{args.out}.part{i}" for i in range(args.workers)]
    with mp.Pool(args.workers) as pool:
        counts = pool.starmap(worker, [(i, args, per[i], parts[i]) for i in range(args.workers)])
    # Streamed (the parts are not loaded in memory); --out *.zst / *.gz is
    # compressed (textio.py), the .partN files stay plain until merged. The
    # temporary file keeps the extension, which decides the compression.
    total = 0
    tmp = os.path.join(os.path.dirname(args.out) or ".", ".tmp-" + os.path.basename(args.out))
    with open_text(tmp, "w") as out:
        for p in parts:
            if os.path.exists(p):
                with open(p) as f:
                    for line in f:
                        out.write(line)
                        total += 1
    os.replace(tmp, args.out)
    for p in parts:
        for f in (p, p + ".games"):
            if os.path.exists(f):
                os.unlink(f)
    os.unlink(split_path)
    print(f"{total} positions written to {args.out} ({sum(counts)} new in this run)")


if __name__ == "__main__":
    main()
