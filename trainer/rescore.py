"""Re-scores positions with another engine ("teacher").

    python rescore.py data/gen3.txt --out data/gen3_ref.txt \
        --engine ../jieqi/ref-engine --net ../jieqi/ref-engine.nnue \
        --opt Name=value --nodes 20000 --workers 15

Input and output are the text format of gen_data.py (fen | score | result);
the input can be compressed (.zst, .gz, see textio.py), the output is plain.
Each position keeps its game result and gets the teacher's search score for
the side to move, in the TEACHER's units: convert them when preparing the
data (prepare_data.py --score-scale 361/s, s fitted with fit_scaling.py on
the output). Positions where the teacher reports a mate or where its best
move is a capture are dropped (like gen_data.py does).

This combines the variety of this engine's self-play games with the
accuracy of a stronger engine's evaluations. Progress is saved: run the same
command with --resume to continue after an interruption.

The teacher is checked like in gen_data.py: its best move must be among the
legal moves of --referee (Maskfish; default ../src/maskfish if it exists,
"" for no check), or the position is dropped; a teacher or referee without
answer within --engine-timeout seconds is killed and started again, and the
position is dropped. Each is printed as a warning; more than --max-failures
in one worker stops rescore.py with an error. --go gives any limit of the
teacher's search instead of --nodes / --depth (e.g. "movetime 50").
"""

from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import os

import jieqi_uci as U
from textio import is_compressed, open_text

CHUNK = 256
DEFAULT_REFEREE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "maskfish")


def start_engines():
    global _ENGINE, _REFEREE
    a = _ARGS
    opts = {"Threads": 1, "Hash": a.hash}
    if a.net:
        opts["EvalFile"] = os.path.abspath(a.net)
    opts.update(a.opt)
    _ENGINE = U.Engine(a.engine, opts, timeout=a.engine_timeout)
    _REFEREE = (U.Engine(a.referee, {"Threads": 1, "Hash": 1}, timeout=a.engine_timeout)
                if a.referee else None)


def worker_init(args):
    global _ARGS, _GO, _KEEP_CAPTURES, _FAILURES
    _ARGS, _FAILURES = args, 0
    _GO = args.go or (f"depth {args.depth}" if args.depth else f"nodes {args.nodes}")
    _KEEP_CAPTURES = args.keep_captures
    start_engines()


def note_failure(what: str):
    """An illegal move or a timeout in this worker (see the header)."""
    global _FAILURES
    _FAILURES += 1
    if _FAILURES <= 10 or _FAILURES % 100 == 0:
        print(f"warning: pid {os.getpid()}: {what} ({_FAILURES} engine failure(s))", flush=True)
    if _FAILURES > _ARGS.max_failures:
        raise RuntimeError(f"more than {_ARGS.max_failures} engine failures in one worker "
                           f"(illegal moves or timeouts), last: {what}")


def teacher_move(fen: str):
    """(move, score, mate) of the teacher, or None if the position is dropped."""
    try:
        _ENGINE.send(f"position fen {fen}")
        move, score, mate = _ENGINE.search(_GO)
        if _REFEREE is not None and move not in ("(none)", "0000"):
            _REFEREE.send(f"position fen {fen}")
            if move not in _REFEREE.legal_moves():
                note_failure(f"{_ENGINE.path} answered {move!r} in {fen}: position dropped")
                return None
        return move, score, mate
    except U.EngineTimeout as e:
        note_failure(f"{e}: position dropped")
        for x in (_ENGINE, _REFEREE):
            if x is not None:
                x.quit()
        start_engines()
        return None


def rescore_chunk(lines: list[str]) -> list[str]:
    out = []
    for line in lines:
        try:
            fen, _, result = (x.strip() for x in line.split("|"))
        except ValueError:
            continue
        answer = teacher_move(fen)
        if answer is None:
            continue
        move, score, mate = answer
        if score is None or mate or move in ("(none)", "0000"):
            continue
        if not _KEEP_CAPTURES and U.is_capture(fen, move):
            continue
        out.append(f"{fen} | {score} | {result}\n")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", required=True, help="teacher engine")
    ap.add_argument("--net", help="EvalFile for the teacher")
    ap.add_argument("--opt", nargs="*", default=[], help="UCI options, Name=Value")
    ap.add_argument("--nodes", type=int, default=20000)
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--go", help="limit of the teacher's search, any 'go' arguments "
                    "(e.g. 'movetime 50'); overrides --nodes / --depth")
    ap.add_argument("--referee", default=DEFAULT_REFEREE if os.path.exists(DEFAULT_REFEREE) else "",
                    help="Maskfish binary checking the teacher's moves (default: ../src/maskfish "
                         "if it exists; empty: no check)")
    ap.add_argument("--engine-timeout", type=float, default=600,
                    help="seconds to wait for an answer of the teacher or the referee before it "
                         "is killed and started again (0: no limit)")
    ap.add_argument("--max-failures", type=int, default=20,
                    help="illegal moves / timeouts allowed per worker before stopping with an error")
    ap.add_argument("--hash", type=int, default=16)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--keep-captures", action="store_true",
                    help="keep positions whose best move is a capture")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run")
    args = ap.parse_args()
    args.opt = dict(o.split("=", 1) for o in args.opt)
    try:   # check the engines and options once (an error in the pool initializer respawns forever)
        worker_init(args)
        for x in (_ENGINE, _REFEREE):
            if x is not None:
                x.quit()
    except U.UnknownOption as e:
        raise SystemExit(f"error: {e}")

    progress_path = args.out + ".progress"
    if is_compressed(args.out):
        raise SystemExit(f"{args.out}: write a plain .txt (it is appended to by --resume), "
                         "compress it afterwards")
    skip = 0
    if args.resume and os.path.exists(progress_path):
        with open(progress_path) as f:
            skip = int(f.read().strip() or 0)
    elif os.path.exists(args.out) and not args.resume:
        raise SystemExit(f"{args.out} exists: use --resume to continue it, or remove it")

    def chunks(f):
        while True:
            lines = list(itertools.islice(f, CHUNK))
            if not lines:
                return
            yield lines

    done, written = skip, 0
    with open_text(args.input) as fin, open(args.out, "a" if args.resume else "w") as fout, \
            mp.Pool(args.workers, initializer=worker_init, initargs=(args,)) as pool:
        for _ in itertools.islice(fin, skip):
            pass
        sizes = []

        def counted(f):
            for c in chunks(f):
                sizes.append(len(c))
                yield c

        for i, result in enumerate(pool.imap(rescore_chunk, counted(fin))):
            fout.writelines(result)
            fout.flush()
            done += sizes[i]
            written += len(result)
            with open(progress_path, "w") as fp:
                fp.write(str(done))
            if (i + 1) % 20 == 0:
                print(f"{done} input lines, {written} written", flush=True)
    print(f"done: {done} input lines, {written} positions written to {args.out}")


if __name__ == "__main__":
    main()
