#!/usr/bin/env python3
"""Invalid input must be refused, never crash the engine, and never leave it in a broken state:

  python3 tests/invalid_input.py BIN [--fuzz N] [--seed S]

- Fixed cases: positions the engine must refuse (as the referee does, plus what would overflow its tables: more
  pieces of a kind than at the start, more than one king, a king out of its palace, a bad counter, ...) with
  `info string bad fen: ...`, keeping the previous position; moves it must refuse with `info string bad move ...`,
  keeping the position before that move; option values that are not numbers, out-of-range values (clamped) and
  odd `go` and command arguments. After each case the engine must still answer `go depth 1` and `isready`, exit
  normally and print no sanitizer report.
- Fuzzing (--fuzz N, default 2000; seeded): mutated FENs, random move tokens and option values sent to one engine,
  each accepted position checked with perft 1; at the end perft 3 from the start must still be 83,907.

Best run on a build with -DMASKFISH_SANITIZE=ON (the CI does): too many pieces of a kind used to overflow the move
lists, and a bad counter or option value to end the engine with an uncaught exception. No network is needed.
Python standard library only.
"""
import argparse
import random
import re
import subprocess
import sys

START = "xxxxkxxxx/9/1x5x1/x1x1x1x1x/9/9/X1X1X1X1X/1X5X1/9/XXXXKXXXX w R2A2C2P5N2B2r2a2c2p5n2b2 0 1"
AFTER_A3A4 = "xxxxkxxxx/9/1x5x1/x1x1x1x1x/9/P8/2X1X1X1X/1X5X1/9/XXXXKXXXX b R2A2C2P4N2B2r2a2c2p5n2b2 1 1"
ENDGAME = "3ak4/4a4/9/9/9/9/9/9/9/3RK4 w - 0 1"
# After b2b9Pr a9b9r: both black rooks are accounted for, one captured face down, one revealed
AFTER_CAPTURES = "1rxxkxxxx/9/1x5x1/x1x1x1x1x/9/9/X1X1X1X1X/7X1/9/XXXXKXXXX w R2A2C2P4N2B2a2c2p5n2b2 0 1"
CRASH = ("runtime error", "AddressSanitizer", "LeakSanitizer", "terminate called", "Assertion", "Segmentation")

# (name, commands, expected text, FEN the engine must hold afterwards)
CASES = [
    # Positions: refused, the previous position is kept
    ("too many rooks on the board", ["position fen 4k4/9/9/9/9/9/9/9/R3R3R/4K4 w - 0 1"],
     "bad fen: more pieces of a kind than at the start", START),
    ("rooks on the board and in the urn", ["position fen 4k4/9/9/9/9/9/9/9/R7R/4K4 w R1 0 1"],
     "bad fen: more pieces of a kind than at the start", START),
    ("fifteen rooks (more moves than the move lists hold)",
     ["position fen 3k5/RR1R1R1RR/R7R/9/9/9/R7R/9/R2R1R2R/4K4 w - 0 1"],
     "bad fen: more pieces of a kind than at the start", START),
    ("seventeen pieces", [f"position fen {START.replace('9/XXXXKXXXX', 'P8/XXXXKXXXX')}"],
     "bad fen: too many pieces", START),
    ("two kings", ["position fen 3kk4/9/9/9/9/9/9/9/9/4K4 w - 0 1"], "bad fen: more than one king", START),
    ("king outside its palace", ["position fen 4k4/9/9/9/9/9/9/9/9/K8 w - 0 1"],
     "bad fen: king outside its palace", START),
    ("black king outside its palace", ["position fen 9/9/9/4k4/9/9/9/9/9/4K4 w - 0 1"],
     "bad fen: king outside its palace", START),
    ("missing king", ["position fen 9/9/9/9/9/9/9/9/9/4K4 w - 0 1"], "bad fen: missing king", START),
    ("rank with too many squares", ["position fen 4k4/99999999999999999999999999999R/9/9/9/9/9/9/9/4K4 w - 0 1"],
     "bad fen:", START),
    ("rank with too few squares", ["position fen 4k4/8/9/9/9/9/9/9/9/4K4 w - 0 1"], "bad fen:", START),
    ("eleven ranks", ["position fen 4k4/9/9/9/9/9/9/9/9/9/4K4 w - 0 1"], "bad fen:", START),
    ("nine ranks", ["position fen 4k4/9/9/9/9/9/9/9/4K4 w - 0 1"], "bad fen:", START),
    ("bad board character", ["position fen 4k4/9/9/9/4Q4/9/9/9/9/4K4 w - 0 1"], "bad fen: bad board character", START),
    ("face-down piece off its start point", ["position fen 4k4/9/9/9/9/4X4/9/9/9/4K4 w P1 0 1"],
     "bad fen: face-down piece off its start point", START),
    ("face-down pieces without an urn", [f"position fen {START.split()[0]} w - 0 1"],
     "bad fen: pool smaller than face-down count", START),
    ("urn count too large", [f"position fen {START.split()[0]} w R3A2C2P5N2B2r2a2c2p5n2b2 0 1"],
     "bad fen: pool count too large", START),
    ("huge urn count", [f"position fen {START.split()[0]} w R99999999999999999999 0 1"], "bad fen:", START),
    ("bad urn letter", [f"position fen {START.split()[0]} w K1R2A2C2P5N2B2r2a2c2p5n2b2 0 1"],
     "bad fen: bad pool letter", START),
    ("bad side to move", [f"position fen {START.split()[0]} x R2A2C2P5N2B2r2a2c2p5n2b2 0 1"],
     "bad fen: bad side to move", START),
    ("counter not a number", [f"position fen {START.split()[0]} w R2A2C2P5N2B2r2a2c2p5n2b2 abc 1"],
     "bad fen: bad no-capture counter", START),
    ("negative counter", [f"position fen {START.split()[0]} w R2A2C2P5N2B2r2a2c2p5n2b2 -5 1"],
     "bad fen: bad no-capture counter", START),
    ("board only", ["position fen 4k4/9/9/9/9/9/9/9/9/4K4"], "bad fen: fen needs board and side", START),
    ("empty fen", ["position fen"], "bad fen:", START),
    ("bad fen after a good one", [f"position fen {ENDGAME}", "position fen 4k4 w"], "bad fen:", ENDGAME),
    # Accepted, with the counter clamped to the 80-ply limit (beyond it, the game is drawn all the same)
    ("huge counter", [f"position fen {ENDGAME.replace(' 0 1', ' 99999999999999999999 1')}"], None,
     ENDGAME.replace(" 0 1", " 80 1")),
    # Moves: refused, the position before the bad move is kept
    ("face-down mover without its identity", ["position startpos moves a3a4P a6a5"],
     "bad move a6a5: face-down mover without reveal", AFTER_A3A4),
    ("illegal move", ["position startpos moves a3a4P e9e7"], "bad move e9e7: illegal move", AFTER_A3A4),
    ("move of the side not to move", ["position startpos moves a6a5p"], "bad move a6a5p:", START),
    ("identity not in the urn", ["position startpos moves a3a4K"], "bad move a3a4K:", START),
    ("bad suffix letter", ["position startpos moves a3a4Z"], "bad move a3a4Z: bad suffix letter", START),
    ("square off the board", ["position startpos moves a3a10P"], "bad move a3a10P:", START),
    ("garbage move", ["position startpos moves zzzz"], "bad move zzzz:", START),
    ("short move", ["position startpos moves a3"], "bad move a3:", START),
    ("moves after a bad move", ["position startpos moves a3a4P zzzz a6a5p"], "bad move zzzz:", AFTER_A3A4),
    ("captured identity not in the urn", ["position startpos moves b2b9Pr a9b9r h2h9Cr"],
     "bad move h2h9Cr: captured identity not in pool", AFTER_CAPTURES),
    # After a refused position command the engine does not search the position it holds
    ("go after a bad move", ["position startpos moves a3a4P zzzz", "go depth 1"],
     "no search: the last position command was refused", AFTER_A3A4),
    ("go after a bad fen", ["position fen 4k4 w", "go depth 1"],
     "no search: the last position command was refused", START),
    # searchmoves: words that are not legal moves are left out; with none left, no search
    ("searchmoves with no legal move", ["go depth 1 searchmoves zzzz a9a8"],
     "no search: no legal move in searchmoves", START),
    ("searchmoves with a word that is not a move", ["go searchmoves h2e2 xyz depth 1"],
     "searchmoves: xyz is not a legal move", START),
    # Options: a value that is not a number is refused, one out of range is clamped
    ("Hash not a number", ["setoption name Hash value abc"], "bad value abc for option Hash", START),
    ("Hash negative", ["setoption name Hash value -5"], None, START),
    ("MultiPV too large for a number", ["setoption name MultiPV value 99999999999999999999999"], None, START),
    ("Threads out of range", ["setoption name Threads value 100000", "setoption name Threads value 0"], None, START),
    ("MultiPV out of range", ["setoption name MultiPV value 100000"], None, START),
    ("MultiPV empty", ["setoption name MultiPV value"], "bad value  for option MultiPV", START),
    ("search parameter not a number", ["setoption name ChanceK value 1e3"], "bad value 1e3 for option ChanceK",
     START),
    ("search parameter out of range", ["setoption name ChanceK value -99999999999999999999"], None, START),
    ("unknown option", ["setoption name NoSuchOption value 3"], "unknown option NoSuchOption", START),
    # Option names in any case (UCI): the option is found, and named as the engine names it
    ("option names in any case", ["setoption name HASH value 8", "setoption name multipv value 2",
                                  "setoption name chancek value abc"], "bad value abc for option ChanceK", START),
    ("setoption without a name", ["setoption", "setoption name", "setoption value 3"], None, START),
    # Other commands with odd arguments
    ("go with odd limits", ["go depth -5", "go depth abc", "go nodes -1 depth 1", "go movetime -1",
                            "go wtime -100 btime -100 winc -5", "go searchmoves zzzz a9a8 depth 1",
                            "go movestogo 0 wtime 1000 btime 1000", "stop"], None, START),
    ("unknown command", ["foo", "position", "position nonsense", "ponderhit", "stop"], None, START),
    ("diagnostic commands", ["see", "see zz", "perft -1", "perft abc", "perftc -1", "moves extra"], None, START),
]


def run(binary, commands, timeout=120):
    p = subprocess.run([binary], input="".join(c + "\n" for c in commands), capture_output=True, text=True,
                       timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def fixed_cases(binary):
    failures = []
    for name, commands, expected, fen in CASES:
        code, out, err = run(binary, ["position startpos"] + commands + ["d", "go depth 1", "isready", "quit"])
        lines = out.splitlines()
        held = next((l[4:] for l in lines if l.startswith("fen ")), None)
        crash = [w for w in CRASH if w in out + err]
        problems = []
        if code != 0 or crash:
            problems.append(f"exit code {code} {crash}")
        if expected and not any(expected in l for l in lines):
            problems.append(f"no '{expected}' in the output")
        if held != fen:
            problems.append(f"holds {held}, not {fen}")
        if not any(l.startswith("bestmove ") for l in lines) or "readyok" not in lines:
            problems.append("no bestmove or readyok after it")
        print(f"{'ok  ' if not problems else 'FAIL'} {name}" + (f": {'; '.join(problems)}" if problems else ""))
        if problems:
            failures.append(name)
            print("     " + "\n     ".join((out + err).strip().splitlines()[-12:]))
    return failures


def mutate(rng, text):
    alphabet = "KRNBACPXkrnbacpx0123456789/ -wb"
    s = list(text)
    for _ in range(rng.randint(1, 4)):
        op = rng.randrange(4)
        i = rng.randrange(len(s) + 1)
        if op == 0 and s:
            s[min(i, len(s) - 1)] = rng.choice(alphabet)
        elif op == 1:
            s.insert(i, rng.choice(alphabet))
        elif op == 2 and s:
            del s[min(i, len(s) - 1)]
        else:
            j = rng.randrange(len(s) + 1)
            s[i:i] = s[min(i, j):max(i, j)][:20]
    return "".join(s)


def fuzz(binary, n, seed):
    rng = random.Random(seed)
    bases = [START, AFTER_A3A4, ENDGAME, "4k4/9/9/9/9/9/9/9/4A4/3RK4 w - 0 1",
             "r1xxkxxxx/9/1x5c1/x1x1x1x1x/9/9/X1X1X1X1X/1C5X1/9/XXXXKXXXX w R2A2C1P5N2B2r1a2c1p5n2b2 2 2",
             "2bakab2/9/1c2r2c1/p1p1p1p1p/9/9/P1P1P1P1P/1C2R2C1/9/2BAKAB2 w R1N2r1n2 0 1"]
    letters = "abcdefghi"
    commands = []
    for k in range(n):
        r = rng.random()
        if r < 0.6:
            commands += [f"position fen {mutate(rng, rng.choice(bases))}", "perft 1"]
        elif r < 0.85:
            toks = []
            for _ in range(rng.randint(1, 6)):
                t = (rng.choice(letters) + str(rng.randrange(10)) + rng.choice(letters) + str(rng.randrange(10)) +
                     "".join(rng.choice("KRNBACPXkrnbacpx") for _ in range(rng.randrange(3))))
                toks.append(mutate(rng, t) if rng.random() < 0.2 else t)
            commands += [f"position startpos moves a3a4P {' '.join(toks)}", "perft 1"]
        else:
            name = rng.choice(["MultiPV", "ChanceK", "HNNScale", "Rule40Div"])  # not Hash, Threads: memory, time
            value = rng.choice(["-1", "0", "1", "2", "abc", "", "99999999999999999999", "-99999999999999999999",
                                "3.5", "+4", str(rng.randrange(-10**6, 10**6))])
            commands.append(f"setoption name {name} value {value}")
        if k % 50 == 49:
            commands += ["go depth 2", "isready"]
    commands += ["position startpos", "perft 3", "isready", "quit"]
    code, out, err = run(binary, commands, timeout=1200)
    crash = [w for w in CRASH if w in out + err]
    accepted = sum(1 for l in out.splitlines() if l.startswith("perft 1 "))
    refused = sum(1 for l in out.splitlines() if l.startswith(("info string bad fen", "info string bad move")))
    ok = code == 0 and not crash and re.search(r"^perft 3 83907 ", out, re.M) is not None
    print(f"{'ok  ' if ok else 'FAIL'} fuzzing, {n} inputs (seed {seed}): {refused} refused, "
          f"perft 1 on {accepted} positions, perft 3 afterwards {'83907' if ok else 'wrong or missing'}")
    if not ok:
        print("     " + "\n     ".join((out + err).strip().splitlines()[-20:]))
        return ["fuzzing"]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("--fuzz", type=int, default=2000, help="number of random inputs (0: none)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    failures = fixed_cases(args.binary)
    if args.fuzz:
        failures += fuzz(args.binary, args.fuzz, args.seed)
    if failures:
        sys.exit("invalid_input: FAIL: " + "; ".join(failures))
    print(f"invalid_input: pass, {len(CASES)} cases")


if __name__ == "__main__":
    main()
