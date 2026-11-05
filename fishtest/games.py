"""Referee: plays Jieqi game pairs between the new and the base engine.

The rules, the information model and the pairing are those of
trainer/match.py (whose helpers are reused):

* the two games of a pair use the same hidden identities and the same random
  opening, with colors swapped, which removes most of the luck of the reveals;
* each engine only gets what its side knows: reveals are public, the identity
  of a captured dark piece is only told to the capturer (unless full_info);
* games end by mate / no legal move, adjudication (the side to move reports
  that it is mated, or both sides agree on a large score; a mate claimed by
  the winning side alone is played out), the rules of the engine as judged by
  the referee with the command "rule" (third occurrence of a position: a
  draw, or a loss for perpetual check or chase; 40-move rule: a draw after 80
  plies without capture, or a loss for a side without legal move), or the
  maximum number of plies.

The referee is a separate process of one of the two binaries (the base one by
default, or the new one for a branch that changes the rules), with the view of
that side: it gives the board state and the legal moves, makes the random
openings, judges the rules, and checks every move. An illegal move loses the
game for the side that played it, so that a move generation bug of the engine
tested cannot make its opponent refuse to search. A referee without answer
voids the pair, which is played again with a new referee process.

In addition, the referee runs real clocks for time controls such as "10+0.1"
(a loss on time when an engine exceeds its time by more than the margin), and
counts crashes and timeouts as losses for the engine at fault, restarting it.

The functions worker_init() and play_pair() run in the processes of a
multiprocessing pool.
"""

from __future__ import annotations

import collections
import copy
import os
import queue
import random
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

TRAINER = Path(__file__).resolve().parent.parent / "trainer"
if str(TRAINER) not in sys.path:
    sys.path.insert(0, str(TRAINER))

import jieqi_features as F  # noqa: E402
import jieqi_uci as U       # noqa: E402

START_FEN = U.START_FEN


# ---------------------------------------------------------------- engine

class EngineError(Exception):
    """The engine terminated or could not be started."""


class EngineTimeout(EngineError):
    """The engine did not answer in time."""


class RefereeError(Exception):
    """The referee process failed: the pair is void and played again."""


REFEREE_RETRIES = 3       # new referee processes for one pair before giving up
REFEREE_OPTIONS = {"Threads": 1, "Hash": 1}


class Engine:
    """A UCI engine with timeouts on every read (portable: a reader thread
    feeds a queue)."""

    def __init__(self, path: str, options: dict, name: str = "engine"):
        self.path, self.options, self.name = path, dict(options), name
        self.p = None
        self.start()

    def start(self):
        self.p = subprocess.Popen([self.path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1,
                                  cwd=os.path.dirname(self.path) or None)
        self.q: queue.Queue = queue.Queue()
        self.tail: collections.deque = collections.deque(maxlen=40)   # last lines read
        threading.Thread(target=self._reader, args=(self.p, self.q), daemon=True).start()
        self.send("uci")
        self.wait_for("uciok", 30)
        for k, v in self.options.items():
            self.send(f"setoption name {k} value {v}")
        # Options the engine does not have (it prints "No such option: X")
        self.unknown_options = [line.split(":", 1)[1].strip() for line in self.isready(120)
                                if line.startswith("No such option:")]

    @staticmethod
    def _reader(p, q):
        try:
            for line in p.stdout:
                q.put(line.rstrip("\r\n"))
        except (OSError, ValueError):
            pass
        q.put(None)

    def send(self, cmd: str):
        try:
            self.p.stdin.write(cmd + "\n")
            self.p.stdin.flush()
        except (OSError, ValueError) as e:
            raise EngineError(f"{self.name}: cannot write to the engine ({e})")

    def readline(self, timeout: float) -> str:
        try:
            line = self.q.get(timeout=max(timeout, 0.001))
        except queue.Empty:
            raise EngineTimeout(f"{self.name}: no answer within {timeout:.1f} s")
        if line is None:
            self.q.put(None)
            raise EngineError(f"{self.name}: the engine terminated (exit code {self.p.poll()})")
        self.tail.append(line)
        return line

    def wait_for(self, prefix: str, timeout: float) -> list[str]:
        deadline = time.monotonic() + timeout
        lines = []
        while True:
            line = self.readline(deadline - time.monotonic())
            lines.append(line)
            if line.startswith(prefix):
                return lines

    def isready(self, timeout: float = 30) -> list[str]:
        self.send("isready")
        return self.wait_for("readyok", timeout)

    def set_options(self, options: dict):
        """Sets UCI options (kept for a restart after a crash)."""
        for k, v in options.items():
            self.send(f"setoption name {k} value {v}")
            self.options[k] = v
        self.isready(60)

    def set_position(self, moves: list[str]):
        self.send(f"position fen {START_FEN}" + (" moves " + " ".join(moves) if moves else ""))

    def board_info(self) -> tuple[str, bool]:
        self.send("d")
        fen, in_check = None, False
        for line in self.wait_for("Checkers:", 30):
            if line.startswith("Fen: "):
                fen = line[5:].strip()
        in_check = bool(line[len("Checkers:"):].strip())
        if fen is None:
            raise EngineError(f"{self.name}: no FEN in the output of 'd'")
        return fen, in_check

    def rule(self) -> str | None:
        """Verdict of the engine's rules in the current position, or None if the
        engine does not have the command "rule" (see U.RuleJudge)."""
        if getattr(self, "rule_supported", True) is False:
            return None
        self.send("rule")
        self.send("isready")
        verdict = U.parse_rule(self.wait_for("readyok", 30))
        self.rule_supported = verdict is not None
        return verdict

    def legal_moves(self) -> list[str]:
        self.send("go perft 1")
        moves = []
        for line in self.wait_for("Nodes searched", 30):
            head = line.split(":")[0]
            if ":" in line and len(head) == 4:
                moves.append(head)
        return moves

    def search(self, go: str, timeout: float) -> tuple[str, int | None, bool]:
        """(bestmove, score, is_mate); the score is for the side to move."""
        self.send("go " + go)
        deadline = time.monotonic() + timeout
        score, mate = None, False
        while True:
            line = self.readline(deadline - time.monotonic())
            if line.startswith("bestmove"):
                t = line.split()
                return (t[1] if len(t) > 1 else "(none)"), score, mate
            if line.startswith("info") and " score " in line and " pv" in line:
                t = line.split()
                i = t.index("score")
                try:
                    if t[i + 1] == "cp":
                        score, mate = int(t[i + 2]), False
                    elif t[i + 1] == "mate":
                        m = int(t[i + 2])
                        score, mate = (32000 if m > 0 else -32000), True
                except (IndexError, ValueError):
                    pass

    def kill(self):
        if self.p and self.p.poll() is None:
            try:
                self.p.kill()
                self.p.wait(timeout=5)
            except Exception:
                pass
        if self.p:
            try:
                self.p.stdin.close()
            except Exception:
                pass

    def restart(self):
        self.kill()
        self.start()

    def quit(self):
        try:
            if self.p.poll() is None:
                self.send("quit")
                self.p.wait(timeout=5)
        except Exception:
            pass
        self.kill()


# ---------------------------------------------------------------- time control

def parse_tc(tc: str) -> dict:
    """'10+0.1' (seconds + increment), '60' , 'nodes=20000', 'depth=8',
    'movetime=100' (ms per move)."""
    tc = tc.strip()
    m = re.fullmatch(r"(nodes|depth|movetime)=(\d+)", tc)
    if m:
        return {"type": "go", "go": f"{m.group(1)} {m.group(2)}"}
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(?:\+(\d+(?:\.\d+)?))?", tc)
    if m:
        base = float(m.group(1)) * 1000
        inc = float(m.group(2) or 0) * 1000
        if base <= 0:
            raise ValueError("the time must be positive")
        return {"type": "clock", "base": int(base), "inc": int(inc)}
    raise ValueError(f"bad time control {tc!r}: use 10+0.1, nodes=N, depth=N or movetime=MS")


# ---------------------------------------------------------------- games

RESULT_STR = {1: "1-0", 0: "1/2-1/2", -1: "0-1"}


def make_opening(engine: Engine, rng: random.Random, plies: int):
    """Random legal opening moves (raw moves, listed by the referee engine)
    and the hidden identities."""
    ids = U.HiddenIdentities(rng)
    probe = copy.deepcopy(ids)
    moves = []
    for _ in range(plies):
        engine.set_position(moves)
        legal = engine.legal_moves()
        if not legal:
            break
        moves.append(probe.annotate(rng.choice(legal)))
    return ids, [m[:4] for m in moves]


def play_game(white: Engine, black: Engine, ids, opening: list[str], cfg: dict,
              referee: Engine, judge_view: int) -> dict:
    """Plays one game. Returns {'result': +1/0/-1 for white, 'reason',
    'moves' (full information), 'plies', 'fault': color of a crash / time
    loss / illegal move or None, 'fault_kind'}. referee: the process that
    gives the board state and the legal moves and judges the rules (see
    U.RuleJudge), with the view of the color judge_view (0 white, 1 black).
    Raises RefereeError when the referee fails."""
    ids = copy.deepcopy(ids)
    engines = (white, black)
    views = ([], [])
    record = []
    tc = cfg["tc"]
    clocks = [tc.get("base", 0), tc.get("base", 0)]
    margin = cfg["timemargin"]
    out = {"fault": None, "fault_kind": ""}

    def finish(result, reason):
        out.update(result=result, reason=reason, moves=" ".join(record), plies=len(record))
        return out

    def ask_referee(fn):
        try:
            return fn()
        except EngineError as e:
            raise RefereeError(str(e)) from e

    stm = 0
    try:
        for e in engines:
            stm = engines.index(e)
            e.send("ucinewgame")
            e.isready(60)
        last_scores = {0: [], 1: []}
        no_capture = 0
        judge = U.RuleJudge()
        for ply in range(cfg["max_plies"]):
            stm = ply % 2
            eng = engines[stm]
            eng.set_position(views[stm])
            ask_referee(lambda: referee.set_position(views[judge_view]))
            fen, in_check = ask_referee(referee.board_info)
            verdict = ask_referee(referee.rule)
            if verdict is None:
                # A referee built before the command "rule": the engine to
                # move judges, as before the referee process
                verdict = eng.rule()
            legal = []

            def legal_moves():
                if not legal:
                    legal.append(ask_referee(referee.legal_moves))
                return legal[0]
            ended = judge.result(fen, in_check, no_capture, verdict, lambda: bool(legal_moves()))
            if ended is not None:
                return finish(*ended)
            score, mate = None, False
            if ply < len(opening):
                move = opening[ply]
            elif tc["type"] == "clock":
                go = (f"wtime {clocks[0]} btime {clocks[1]} "
                      f"winc {tc['inc']} binc {tc['inc']}")
                t0 = time.monotonic()
                try:
                    move, score, mate = eng.search(go, (clocks[stm] + margin) / 1000)
                except EngineTimeout:
                    out.update(fault=stm, fault_kind="time")
                    eng.restart()
                    return finish(-1 if stm == 0 else 1, "time loss")
                used = int((time.monotonic() - t0) * 1000)
                if used > clocks[stm] + margin:
                    out.update(fault=stm, fault_kind="time")
                    return finish(-1 if stm == 0 else 1, "time loss")
                clocks[stm] = max(clocks[stm] - used, 0) + tc["inc"]
            else:
                try:
                    move, score, mate = eng.search(tc["go"], cfg["move_timeout"])
                except EngineTimeout:
                    out.update(fault=stm, fault_kind="time")
                    eng.restart()
                    return finish(-1 if stm == 0 else 1, "timeout")
            if move in ("(none)", "0000"):
                return finish(-1 if stm == 0 else 1, "no legal move")
            if ply >= len(opening) and move not in legal_moves():
                out.update(fault=stm, fault_kind="illegal")
                record.append(move + "?")
                return finish(-1 if stm == 0 else 1, f"illegal move {move}")
            capture = U.is_capture(fen, move)
            w, b = U.annotate_views(ids, move, cfg["full_info"])
            views[0].append(w)
            views[1].append(b)
            record.append(max(w, b, key=len))

            if score is not None:
                sign = 1 if stm == 0 else -1
                # Only the side that is mated ends the game with its score (it
                # resigns): a mate claimed by the winning side is played out,
                # or adjudicated below if the other side agrees, since a wrong
                # mate score (e.g. a bug of the patch tested) would win the game.
                if mate and score < 0:
                    return finish(-sign, "mate")
                last_scores[stm] = (last_scores[stm] + [sign * score])[-4:]
                sw, sb = last_scores[0], last_scores[1]
                if len(sw) == 4 and len(sb) == 4:
                    if min(sw + sb) >= cfg["adjudicate"]:
                        return finish(1, "adjudication")
                    if max(sw + sb) <= -cfg["adjudicate"]:
                        return finish(-1, "adjudication")
            # The 40-move rule is judged before the next move (a side without
            # a legal move then loses, as in the engine)
            no_capture = 0 if capture else no_capture + 1
        return finish(0, "max plies")
    except EngineError as e:
        # The engine of the side being asked crashed (or did not answer)
        out.update(fault=stm, fault_kind="crash", error=str(e))
        try:
            engines[stm].restart()
        except Exception:
            pass
        return finish(-1 if stm == 0 else 1, "crash")


# ---------------------------------------------------------------- pool workers

_W: dict = {}


def _engine_options(side: dict, threads: int, hash_mb: int) -> dict:
    o = {"Threads": threads, "Hash": hash_mb, "RawScore": "true"}
    if side.get("net"):
        o["EvalFile"] = side["net"]
    o.update(side.get("options", {}))
    return o


def _referee_side(cfg: dict) -> str:
    """'base' (default) or 'new': the binary whose process is the referee."""
    return "new" if cfg.get("referee") == "new" else "base"


def _start_referee(cfg: dict) -> Engine:
    # No network: the referee only uses "d", "go perft 1" and "rule"
    return Engine(cfg[_referee_side(cfg)]["binary"], REFEREE_OPTIONS, "referee")


def worker_init(cfg: dict):
    _W["cfg"] = cfg
    _W["error"] = None
    try:
        _W["new"] = Engine(cfg["new"]["binary"], _engine_options(cfg["new"], cfg["threads"], cfg["hash"]), "new")
        _W["base"] = Engine(cfg["base"]["binary"], _engine_options(cfg["base"], cfg["threads"], cfg["hash"]), "base")
        _W["referee"] = _start_referee(cfg)
    except Exception as e:      # reported by play_pair (a failing initializer would loop)
        _W["error"] = f"cannot start the engines: {e}"

    def on_term(*_):
        for k in ("new", "base", "referee"):
            if k in _W:
                _W[k].kill()
        os._exit(0)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, on_term)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, signal.SIG_IGN)   # the runner stops the pool


def play_pair(index: int, seed: int, opts_new: dict | None = None,
              opts_base: dict | None = None) -> dict:
    """Two games from the same opening and identities, colors swapped.
    'scores' are the results of the new engine (+1 / 0 / -1). opts_new /
    opts_base are UCI options set before the pair (SPSA). A pair whose
    referee fails is played again with a new referee process."""
    if _W.get("error"):
        raise RuntimeError(_W["error"])
    cfg, new, base = _W["cfg"], _W["new"], _W["base"]
    for eng, opts in ((new, opts_new), (base, opts_base)):
        if opts:
            try:
                eng.set_options(opts)
            except EngineError:
                eng.restart()           # restart() applies the options again
    # The referee has the view of its side: the base engine by default, so
    # that the patch tested does not judge its own games
    judge_new = _referee_side(cfg) == "new"
    for attempt in range(REFEREE_RETRIES + 1):
        referee = _W["referee"]
        try:
            rng = random.Random(seed)
            plies = rng.randint(cfg["plies_min"], max(cfg["plies_min"], cfg["plies_max"]))
            ids, opening = make_opening(referee, rng, plies)
            g1 = play_game(new, base, ids, opening, cfg, referee,       # new has white
                           judge_view=0 if judge_new else 1)
            g2 = play_game(base, new, ids, opening, cfg, referee,       # new has black
                           judge_view=1 if judge_new else 0)
            break
        except (RefereeError, EngineError) as e:
            referee.kill()
            if attempt == REFEREE_RETRIES:
                raise RuntimeError(f"the referee failed {attempt + 1} times in a row "
                                   f"(pair {index}), the last time: {e}") from None
            _W["referee"] = _start_referee(cfg)
    g1["white"], g1["black"] = "new", "base"
    g2["white"], g2["black"] = "base", "new"
    scores = (g1["result"], -g2["result"])
    crashes = sum(1 for g in (g1, g2) if g["fault_kind"] == "crash")
    time_losses = sum(1 for g in (g1, g2) if g["fault_kind"] == "time")
    illegal_moves = sum(1 for g in (g1, g2) if g["fault_kind"] == "illegal")
    games = []
    for k, g in enumerate((g1, g2)):
        games.append(f"[pair {index} game {k + 1} seed {seed}] white={g['white']} black={g['black']} "
                     f"result={RESULT_STR[g['result']]} reason={g['reason']} plies={g['plies']}"
                     + (f" error={g['error']!r}" if g.get("error") else "")
                     + f"\n{g['moves']}\n")
    return {"index": index, "scores": scores, "crashes": crashes,
            "time_losses": time_losses, "illegal_moves": illegal_moves, "games": games}


def spin_options(binary: str) -> dict:
    """{name: (default, min, max)} of the spin options of an engine."""
    out = {}
    e = Engine(binary, {}, "options")
    try:
        e.send("uci")
        for line in e.wait_for("uciok", 30):
            m = re.match(r"option name (\S+) type spin default (-?\d+) min (-?\d+) max (-?\d+)", line)
            if m:
                out[m.group(1)] = tuple(int(m.group(i)) for i in (2, 3, 4))
    finally:
        e.quit()
    return out


def check_engine(binary: str, side: dict, threads: int, hash_mb: int) -> str:
    """Starts the engine and searches one position: returns '' when it works,
    or the error with the last lines of output."""
    e = None
    try:
        e = Engine(binary, _engine_options(side, threads, hash_mb), "check")
        # An option of the test that this binary does not have (renamed or
        # removed in its branch) would be ignored silently
        unknown = {o.lower() for o in e.unknown_options}
        missing = [k for k in side.get("options", {}) if k.lower() in unknown]
        if missing:
            return ("no such UCI option in this engine: " + ", ".join(missing)
                    + " (renamed or removed in its branch?)")
        e.send("position startpos")
        e.send("go depth 3")
        lines = e.wait_for("bestmove", 60)
        if lines[-1].split()[1:2] in (["(none)"], []):
            return "the engine found no move in the start position:\n" + "\n".join(e.tail)
        return ""
    except (EngineError, OSError) as ex:
        tail = []
        if e is not None:
            try:
                while True:
                    e.readline(0.2)
            except EngineError:
                pass
            tail = [l for l in e.tail if not l.startswith("option name")]
        return f"{ex}\n" + "\n".join(tail[-20:])
    finally:
        if e is not None:
            e.quit()


def check_referee(binary: str) -> str:
    """Starts the referee and asks it the board state and the legal moves of
    the start position: returns '' when it works, or the error."""
    e = None
    try:
        e = Engine(binary, REFEREE_OPTIONS, "referee")
        e.set_position([])
        e.board_info()
        if not e.legal_moves():
            return "no legal move in the start position (go perft 1)"
        return ""
    except (EngineError, OSError) as ex:
        return str(ex)
    finally:
        if e is not None:
            e.quit()
