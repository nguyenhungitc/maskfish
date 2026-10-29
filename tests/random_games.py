#!/usr/bin/env python3
"""Plays random Jieqi games through UCI and searches along the way, to
exercise the search (chance nodes of reveals and of captures of dark pieces)
in debug / sanitizer builds.

Every move is a random legal move ('go perft 1'); a dark piece is revealed as
a random piece of its owner's pool, and the identity of a captured dark piece
is sent or not at random. Short searches are run at random plies and at the
end of every game, and the rules are asked with 'rule' at every ply. The test
fails if the engine terminates, prints an error of a sanitizer or an
assertion, or reports a position or a move as invalid.

Run from the src directory, after building:
    python3 ../tests/random_games.py [--engine ./maskfish] [--games 10] [--seed 1]
"""

import argparse
import random
import re
import subprocess
import sys

ERRORS = re.compile(r"runtime error|AddressSanitizer|LeakSanitizer|Assertion|Invalid")


class EngineFailure(Exception):
    pass


class Engine:
    def __init__(self, path):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.history = []
        self.send("uci")
        self.wait_for("uciok")

    def send(self, cmd):
        self.history = (self.history + [cmd])[-20:]
        try:
            self.p.stdin.write(cmd + "\n")
            self.p.stdin.flush()
        except OSError as e:
            raise EngineFailure(f"cannot write to the engine ({e})")

    def wait_for(self, prefix):
        lines = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise EngineFailure("the engine terminated:\n" + "".join(lines[-40:]))
            lines.append(line)
            if ERRORS.search(line):
                lines += [self.p.stdout.readline() for _ in range(40)]
                raise EngineFailure("".join(lines[-60:]))
            if line.startswith(prefix):
                return lines

    def quit(self):
        try:
            self.send("quit")
            self.p.wait(timeout=60)
        except (EngineFailure, subprocess.TimeoutExpired):
            self.p.kill()


def parse_fen(fen):
    """(squares -> letters, side to move, pool counts)"""
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
    counts = {pool[i]: int(pool[i + 1]) for i in range(0, len(pool) - 1, 2)}
    return squares, side, counts


def draw_from_pool(rng, counts, white):
    """A random piece of the pool of one side, weighted by the counts."""
    items = [(k, n) for k, n in counts.items() if n > 0 and k.isupper() == white]
    total = sum(n for _, n in items)
    if total == 0:
        return None
    r = rng.randrange(total)
    for k, n in items:
        if r < n:
            return k
        r -= n
    return None


def play(engine, rng, game):
    moves = []
    engine.send("ucinewgame")
    for _ in range(rng.randint(10, 160)):
        engine.send("position startpos moves " + " ".join(moves))
        engine.send("d")
        fen = next(l for l in engine.wait_for("Checkers") if l.startswith("Fen:"))[5:].strip()
        squares, side, counts = parse_fen(fen)

        engine.send("rule")
        engine.send("isready")
        engine.wait_for("readyok")

        engine.send("go perft 1")
        legal = [l.split(":")[0] for l in engine.wait_for("Nodes searched")
                 if re.match(r"^[a-i][0-9][a-i][0-9]:", l)]
        if not legal:
            break

        if rng.random() < 0.08:
            engine.send(rng.choice(["go depth 5", "go nodes 30000", "go depth 3"]))
            engine.wait_for("bestmove")

        move = rng.choice(legal)
        suffix = ""
        if squares.get(move[:2]) in ("X", "x"):          # reveal of the moving piece
            piece = draw_from_pool(rng, counts, side == "w")
            if piece is None:
                raise EngineFailure(f"a dark piece but an empty pool: {fen}")
            suffix += piece
            counts[piece] -= 1
        if squares.get(move[2:4]) in ("X", "x") and rng.random() < 0.6:
            captured = draw_from_pool(rng, counts, side != "w")
            if captured is not None:                    # identity of the captured piece
                suffix += captured
        moves.append(move + suffix)

    engine.send("position startpos moves " + " ".join(moves))
    engine.send("go nodes 50000")
    engine.wait_for("bestmove")
    print(f"game {game}: {len(moves)} plies OK", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="./maskfish")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    engine = Engine(args.engine)
    try:
        for game in range(args.games):
            play(engine, rng, game)
    except EngineFailure as e:
        print(f"random games testing failed (seed {args.seed}):\n{e}")
        print("last commands:\n" + "\n".join(engine.history))
        engine.p.kill()
        return 1
    engine.quit()
    print("random games testing OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
