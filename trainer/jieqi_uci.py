"""Small helpers to drive the engine over UCI and play Jieqi games with real
hidden identities (used by gen_data.py and verify.py)."""

from __future__ import annotations

import random
import subprocess

import jieqi_features as F

START_FEN = ("xxxxkxxxx/9/1x5x1/x1x1x1x1x/9/9/X1X1X1X1X/1X5X1/9/XXXXKXXXX w "
             "R2A2C2P5N2B2r2a2c2p5n2b2 0 1")

# A game is drawn after 40 moves (80 plies) without a capture, as in the engine
# (Position::rule_judge()); reveals do not reset the counter.
NO_CAPTURE_DRAW_PLIES = 80

# Verdicts of the engine command "rule", for the side to move
RULE_VERDICTS = ("none", "draw", "win", "loss", "invalid")


def parse_rule(lines: list[str]) -> str | None:
    """Verdict in the output of "rule" (followed by "isready"), or None for an
    engine without the command (older binaries print "Unknown command")."""
    for line in lines:
        t = line.split()
        if t[:3] == ["info", "string", "rule"] and len(t) > 3 and t[3] in RULE_VERDICTS:
            return t[3]
    return None


def square_from_uci(s: str) -> int:
    return F.make_square(ord(s[0]) - ord('a'), int(s[1]))


class UnknownOption(RuntimeError):
    pass


class Engine:
    """options: set and required to exist (a misspelt option, or a binary built
    before the option was added, raises UnknownOption instead of being silently
    ignored). optional: set only if the engine has them (e.g. RawScore, which
    other engines do not have)."""

    def __init__(self, path: str, options: dict | None = None, optional: dict | None = None):
        self.path = path
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.send("uci")
        self.option_names = set()
        for line in self.wait_for("uciok"):
            if line.startswith("option name "):
                rest = line[len("option name "):]
                self.option_names.add(rest.split(" type ")[0].strip())
        unknown = [k for k in (options or {}) if k not in self.option_names]
        if unknown:
            self.quit()
            raise UnknownOption(f"{path}: no UCI option {', '.join(unknown)} "
                                f"(misspelt, or the binary is older than the option: rebuild it)")
        for k, v in (optional or {}).items():
            if k in self.option_names:
                self.send(f"setoption name {k} value {v}")
        for k, v in (options or {}).items():
            self.send(f"setoption name {k} value {v}")
        self.isready()

    def has_option(self, name: str) -> bool:
        return name in self.option_names

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

    def rule(self) -> str | None:
        """The verdict of the engine's rules in the current position (see
        RuleJudge), or None if the engine does not have the command."""
        if getattr(self, "rule_supported", True) is False:
            return None
        self.send("rule")
        self.send("isready")
        verdict = parse_rule(self.wait_for("readyok"))
        self.rule_supported = verdict is not None
        return verdict

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
        """(psqt, positional, layer stack) of the raw network output."""
        return self.evalraw_full()[:3]

    def evalraw_full(self) -> tuple[int, int, int, int | None]:
        """(psqt, positional, layer stack, final): final is the static evaluation
        after the scaling of evaluate() (None in check, or for an engine whose
        evalraw does not print it)."""
        self.send("evalraw")
        t = self.wait_for("evalraw")[-1].split()
        t = t[:t.index("fen")] if "fen" in t else t
        fields = dict(zip(t[1::2], t[2::2]))        # name value name value ...
        final = fields.get("final")
        return (int(fields["psqt"]), int(fields["positional"]), int(fields["bucket"]),
                int(final) if final not in (None, "none") else None)

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


def annotate_views(ids: HiddenIdentities, move: str, full_info: bool = False) -> tuple[str, str]:
    """Engine move -> (move as sent to the white side, move as sent to the black
    side). Updates the identities. The reveal of the moving piece is public;
    the identity of a captured dark piece is only given to the capturer (to
    both with full_info), as in a real game: its owner never learns it."""
    frm, to = square_from_uci(move[0:2]), square_from_uci(move[2:4])
    reveal = ids.hidden.pop(frm) if frm in ids.hidden else ""
    captured = ids.hidden.pop(to) if to in ids.hidden else ""
    base = move[:4] + reveal
    if not captured:
        return base, base
    with_id = base + captured
    if full_info:
        return with_id, with_id
    capturer = F.WHITE if captured.islower() else F.BLACK
    return (with_id, base) if capturer == F.WHITE else (base, with_id)


class RuleJudge:
    """Ends games with the rules of the engine. Before each move, the referee
    asks an engine (the command "rule", Position::rule_judge() of the search)
    what the rules decide in the position: the third occurrence of a position
    (a draw, or a loss for perpetual check or perpetual chase) and the 40-move
    rule (a draw, or a loss for a side without legal move). With engines that
    do not have the command, the same rules are applied here, except that a
    perpetual chase is a draw.

    result() returns None (the game goes on) or the result for white
    (+1 / 0 / -1) with the reason."""

    def __init__(self):
        self.tracker = RepetitionTracker()

    def result(self, fen: str, in_check: bool, no_capture: int, verdict: str | None,
               has_legal_moves) -> tuple[int, str] | None:
        """fen / in_check: the position before the move; verdict: what "rule"
        printed for it (None: not available); has_legal_moves(): whether the
        side to move has a legal move (only called when needed)."""
        white_to_move = fen.split()[1] != "b"
        stm_sign = 1 if white_to_move else -1
        rule40 = no_capture >= NO_CAPTURE_DRAW_PLIES
        if verdict is not None:
            if verdict in ("none", "invalid"):
                return None
            if verdict == "draw":
                return 0, ("40-move rule" if rule40 else "repetition")
            reason = "40-move rule" if rule40 else "perpetual check or chase"
            return (stm_sign if verdict == "win" else -stm_sign), reason
        r = self.tracker.push(fen, in_check)
        if r is not None:
            return r, ("repetition" if r == 0 else "perpetual check")
        if rule40:
            return (0, "40-move rule") if has_legal_moves() else (-stm_sign, "40-move rule")
        return None


def referee_verdict(referee, engines, views, stm: int) -> str | None:
    """Verdict of "rule" for the position before the move of side stm (0 white,
    1 black). The referee (an engine of the game, normally the reference one,
    so that a patch under test does not judge its own games) is asked with
    its own view of the game; if it does not have the command, the engine to
    move, whose position is already set; None if neither has it."""
    mover = engines[stm]
    for e in ([referee] if referee is not None else []) + [mover]:
        if getattr(e, "rule_supported", True) is False:
            continue
        if e is not mover:
            e.set_position(views[engines.index(e)])
        verdict = e.rule()
        if verdict is not None:
            return verdict
    return None


class RepetitionTracker:
    """Ends a game when a position occurs for the third time. Perpetual check
    loses, as in the engine (Position::rule_judge()): if one side gave check
    with every move of the last cycle and the other did not, the checking side
    loses; otherwise the game is drawn. (Perpetual chase is not detected here:
    such games are drawn.)"""

    def __init__(self):
        self.keys: list[str] = []
        self.checks: list[bool] = []

    def push(self, fen: str, in_check: bool) -> int | None:
        """Position reached (FEN from either side's view, whether the side to
        move is in check). Returns None, or the result for white (+1 / 0 / -1)
        when the position occurs for the third time."""
        # Board and side to move: the same in both views (pools may differ)
        key = " ".join(fen.split()[:2])
        self.keys.append(key)
        self.checks.append(in_check)
        seen = [i for i, k in enumerate(self.keys) if k == key]
        if len(seen) < 3:
            return None
        start, end = seen[-2], len(self.keys) - 1
        always_checks = {F.WHITE: True, F.BLACK: True}
        for i in range(start, end):     # the move played in position i ...
            mover = F.WHITE if self.keys[i].endswith(" w") else F.BLACK
            if not self.checks[i + 1]:  # ... gave check if position i + 1 is in check
                always_checks[mover] = False
        if always_checks[F.WHITE] and not always_checks[F.BLACK]:
            return -1
        if always_checks[F.BLACK] and not always_checks[F.WHITE]:
            return 1
        return 0


def is_capture(fen: str, move: str) -> bool:
    pos = F.parse_fen(fen)
    return bool(pos.board[square_from_uci(move[2:4])])
