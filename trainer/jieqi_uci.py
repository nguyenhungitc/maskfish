"""Small helpers to drive the engine over UCI and play Jieqi games with real
hidden identities (used by gen_data.py and verify.py)."""

from __future__ import annotations

import random
import subprocess

import jieqi_features as F

START_FEN = ("xxxxkxxxx/9/1x5x1/x1x1x1x1x/9/9/X1X1X1X1X/1X5X1/9/XXXXKXXXX w "
             "R2A2C2P5N2B2r2a2c2p5n2b2 0 1")


def square_from_uci(s: str) -> int:
    return F.make_square(ord(s[0]) - ord('a'), int(s[1]))


class Engine:
    def __init__(self, path: str, options: dict | None = None):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.send("uci")
        self.wait_for("uciok")
        for k, v in (options or {}).items():
            self.send(f"setoption name {k} value {v}")
        self.isready()

    def send(self, cmd: str):
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def readline(self) -> str:
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("engine terminated")
        return line.rstrip("\n")

    def wait_for(self, prefix: str) -> list[str]:
        lines = []
        while True:
            line = self.readline()
            lines.append(line)
            if line.startswith(prefix):
                return lines

    def isready(self):
        self.send("isready")
        self.wait_for("readyok")

    def set_position(self, moves: list[str], fen: str = START_FEN):
        self.send(f"position fen {fen}" + (" moves " + " ".join(moves) if moves else ""))

    def board_info(self) -> tuple[str, bool]:
        """FEN of the current position and whether the side to move is in check."""
        self.send("d")
        fen, in_check = None, False
        for line in self.wait_for("Checkers:"):
            if line.startswith("Fen: "):
                fen = line[5:].strip()
            if line.startswith("Checkers:"):
                in_check = bool(line[len("Checkers:"):].strip())
        return fen, in_check

    def legal_moves(self) -> list[str]:
        self.send("go perft 1")
        moves = []
        for line in self.wait_for("Nodes searched"):
            if ":" in line and len(line.split(":")[0]) == 4:
                moves.append(line.split(":")[0])
        return moves

    def search(self, go: str) -> tuple[str, int | None, bool]:
        """Returns (bestmove, score, is_mate) for the side to move."""
        self.send("go " + go)
        score, mate = None, False
        for line in self.wait_for("bestmove"):
            if line.startswith("info") and " score " in line and " pv" in line:
                t = line.split()
                i = t.index("score")
                if t[i + 1] == "cp":
                    score, mate = int(t[i + 2]), False
                elif t[i + 1] == "mate":
                    m = int(t[i + 2])
                    score, mate = (32000 if m > 0 else -32000), True
        best = line.split()[1]
        return best, score, mate

    def evalraw(self) -> tuple[int, int, int]:
        self.send("evalraw")
        line = self.wait_for("evalraw")[-1].split()
        return int(line[2]), int(line[4]), int(line[6])

    def quit(self):
        try:
            self.send("quit")
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


class HiddenIdentities:
    """True identities of the dark pieces of a game, drawn at random, and the
    letters to append to the engine's moves."""

    def __init__(self, rng: random.Random):
        self.hidden = {}
        for color, letters in ((F.WHITE, "RRAACCPPPPPNNBB"), (F.BLACK, "rraaccpppppnnbb")):
            squares = [s for s in range(F.SQUARE_NB)
                       if F.DARK_PIECES[s] not in ".kK"
                       and F.DARK_PIECES[s].isupper() == (color == F.WHITE)]
            ids = list(letters)
            rng.shuffle(ids)
            self.hidden.update(zip(squares, ids))

    def annotate(self, move: str, reveal_captured: bool = True) -> str:
        """Engine move -> move with revealed letters. Updates the identities."""
        frm, to = square_from_uci(move[0:2]), square_from_uci(move[2:4])
        suffix = ""
        if frm in self.hidden:
            suffix += self.hidden.pop(frm)
        if to in self.hidden:
            captured = self.hidden.pop(to)
            if reveal_captured:
                suffix += captured
        return move[:4] + suffix


def is_capture(fen: str, move: str) -> bool:
    pos = F.parse_fen(fen)
    return bool(pos.board[square_from_uci(move[2:4])])
