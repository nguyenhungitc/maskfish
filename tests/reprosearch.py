#!/usr/bin/env python3
"""The search must be reproducible: after `ucinewgame`, the same searches give the same results.

  python3 tests/reprosearch.py BIN [--evalfile NET] [--steps N]

On one thread, a game of a few `go nodes N` searches (the start position, a position after reveals, one after
captures of face-down pieces) is played twice in one engine, separated by `ucinewgame`, then once more in a new
engine. Every `info` line (depth, seldepth, score, nodes, PV; not the time, nps or hashfull) and every `bestmove`
must be the same in the three runs: whatever the search keeps between searches (hash table, histories, ...) must
be cleared by `ucinewgame` and must not depend on anything else. The node limits vary (N = 100 * 1.5^i, i = 1..
--steps, default 16), to stop the searches at many different points. Waits for
each `bestmove` before the next command (a `position` during a search would stop it). Python standard library
only.
"""
import argparse
import re
import subprocess
import sys

POSITIONS = [
    "position startpos",
    "position startpos moves a3a4P a6a5p",
    "position startpos moves a3a4P i6i5r h0g2P h9g7a b0c2N c6c5a c3c4P e6e5n",
    "position startpos moves h2e2C h7e7c e2e6p e7e3P",
]
VOLATILE = re.compile(r" (?:nps|time|hashfull) \d+")


class Engine:
    def __init__(self, binary, evalfile):
        self.p = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        if evalfile:
            self.send(f"setoption name EvalFile value {evalfile}")
        self.send("setoption name Threads value 1")
        self.send("isready")
        self.until("readyok")

    def send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def until(self, prefix):
        lines = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"the engine ended before '{prefix}': {lines[-5:]}")
            line = line.rstrip("\n")
            lines.append(line)
            if line.startswith(prefix):
                return lines

    def game(self, nodes):
        out = []
        for pos in POSITIONS:
            self.send(pos)
            self.send(f"go nodes {nodes}")
            out += [VOLATILE.sub("", l) for l in self.until("bestmove")
                    if l.startswith("bestmove") or (l.startswith("info ") and " pv " in l)]
        return out

    def close(self):
        self.send("quit")
        self.p.communicate(timeout=60)
        return self.p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("--evalfile", help="the network (default: the engine's own)")
    ap.add_argument("--steps", type=int, default=16)
    args = ap.parse_args()
    failures = []
    for i in range(1, args.steps + 1):
        nodes = 100 * 3 ** i // 2 ** i
        e = Engine(args.binary, args.evalfile)
        e.send("ucinewgame")
        first = e.game(nodes)
        e.send("ucinewgame")
        second = e.game(nodes)
        code = e.close()
        f = Engine(args.binary, args.evalfile)
        f.send("ucinewgame")
        fresh = f.game(nodes)
        code = code or f.close()
        bests = [l.split()[1] for l in first if l.startswith("bestmove")]
        ok = first == second == fresh and code == 0 and len(bests) == len(POSITIONS)
        print(f"{'ok  ' if ok else 'FAIL'} go nodes {nodes}: {len(first)} lines, bestmoves {' '.join(bests)}")
        if not ok:
            for name, other in (("after ucinewgame", second), ("in a new engine", fresh)):
                diff = next(((a, b) for a, b in zip(first, other) if a != b), None)
                if diff or len(first) != len(other):
                    print(f"     {name}: first difference\n       {diff[0] if diff else len(first)}\n"
                          f"       {diff[1] if diff else len(other)}")
            failures.append(str(nodes))
    if failures:
        sys.exit("reprosearch: FAIL with go nodes " + ", ".join(failures))
    print("reprosearch: pass")


if __name__ == "__main__":
    main()
