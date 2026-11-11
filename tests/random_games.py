#!/usr/bin/env python3
"""Plays random Jieqi games through UCI with searches along the way, to exercise the search (chance nodes of
reveals and of captures of face-down pieces) on sanitizer builds:

  python3 tests/random_games.py BIN [--games 8] [--seed 1] [--evalfile NET] [--nodes 3000]

Every move is a random legal move (`moves`); a face-down mover is revealed as a random identity of its side's urn
(read from the FEN of `d`), and the identity of a captured face-down piece is sent or not, at random (the
capturer saw it or not). Short searches (`go nodes N`, on one thread or two) run at random plies and at the end
of every game; their bestmove must be legal. The test fails if the engine ends, prints a sanitizer report or a
JQ_VERIFY_LEGAL mismatch (build with -DJQ_VERIFY_LEGAL to have them), or refuses a position or a move. Python
standard library only.
"""
import argparse
import random
import re
import subprocess
import sys

ERRORS = re.compile(r"runtime error|AddressSanitizer|LeakSanitizer|mismatch|illegal move|bad fen|bad move|"
                    r"terminate called|Assertion")
FILES = "abcdefghi"


class EngineFailure(Exception):
    pass


class Engine:
    def __init__(self, binary, evalfile):
        self.p = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1)
        self.history = []
        if evalfile:
            self.send(f"setoption name EvalFile value {evalfile}")
        self.send("isready")
        self.until("readyok")

    def send(self, cmd):
        self.history = (self.history + [cmd])[-8:]
        try:
            self.p.stdin.write(cmd + "\n")
            self.p.stdin.flush()
        except OSError as e:
            raise EngineFailure(f"cannot write to the engine ({e})")

    def until(self, prefix):
        lines = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise EngineFailure("the engine ended:\n" + "".join(lines[-40:]))
            lines.append(line)
            if ERRORS.search(line) and not line.startswith("info string EvalFile"):
                lines += [self.p.stdout.readline() for _ in range(30)]
                raise EngineFailure("".join(lines[-50:]))
            if line.startswith(prefix):
                return lines

    def quit(self):
        try:
            self.send("quit")
            out, _ = self.p.communicate(timeout=60)
        except (OSError, EngineFailure):
            out = ""
        if ERRORS.search(out or "") or self.p.returncode:
            raise EngineFailure(f"at exit (code {self.p.returncode}):\n{out}")


def parse_fen(fen):
    board_text, _side, urns = fen.split()[:3]
    board = {}
    for i, row in enumerate(board_text.split("/")):
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
            else:
                board[FILES[f] + str(9 - i)] = ch
                f += 1
    urn = {"w": {}, "b": {}}
    for letter, n in re.findall(r"([A-Za-z])(\d+)", "" if urns == "-" else urns):
        urn["w" if letter.isupper() else "b"][letter.upper()] = int(n)
    return board, urn


def draw(rng, pool):
    bag = [k for k, n in pool.items() for _ in range(n)]
    return rng.choice(bag) if bag else None


def search(e, rng, nodes, legal):
    threads = rng.choice([1, 1, 2])
    e.send(f"setoption name Threads value {threads}")
    e.send(f"go nodes {nodes}")
    best = e.until("bestmove")[-1].split()[1]
    if best != "(none)" and best not in legal:
        raise EngineFailure(f"bestmove {best} is not a legal move ({' '.join(legal)})")
    return threads


def play(e, rng, nodes, max_plies):
    moves, searches = [], 0
    e.send("ucinewgame")
    for ply in range(max_plies):
        e.send("position startpos moves " + " ".join(moves) if moves else "position startpos")
        e.send("d")
        fen = next(l[4:].strip() for l in e.until("fen ") if l.startswith("fen "))
        e.send("moves")
        legal = e.until("moves")[-1].split()[1:]
        if rng.random() < 0.08:
            search(e, rng, nodes, legal)
            searches += 1
        if not legal:
            break
        board, urn = parse_fen(fen)
        stm = fen.split()[1].replace("r", "w")
        m = rng.choice(legal)
        token = m
        if board.get(m[:2]) in ("X", "x"):  # a face-down mover: revealed as an identity of its urn
            piece = draw(rng, urn[stm])
            if piece is None:
                raise EngineFailure(f"face-down mover {m} with an empty urn: {fen}")
            token += piece if stm == "w" else piece.lower()
        victim = board.get(m[2:4])
        if victim in ("X", "x") and rng.random() < 0.5:  # the capturer saw what it took
            them = "b" if stm == "w" else "w"
            piece = draw(rng, urn[them])
            if piece:
                token += piece if them == "w" else piece.lower()
        moves.append(token)
    search(e, rng, nodes, legal)
    return len(moves), searches + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--evalfile", help="the network (default: the engine's own)")
    ap.add_argument("--nodes", type=int, default=3000, help="nodes of each search")
    ap.add_argument("--plies", type=int, default=160, help="the longest game")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    e = Engine(args.binary, args.evalfile)
    try:
        for g in range(args.games):
            plies, searches = play(e, rng, args.nodes, args.plies)
            print(f"game {g + 1}: {plies} plies, {searches} searches")
        e.quit()
    except EngineFailure as f:
        print("random_games: FAIL\n" + str(f) + "\nlast commands:\n  " + "\n  ".join(e.history))
        sys.exit(1)
    print(f"random_games: pass, {args.games} games (seed {args.seed})")


if __name__ == "__main__":
    main()
