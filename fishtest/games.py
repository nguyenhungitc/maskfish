"""Referee: plays Jieqi game pairs between the new and the base engine.

The games are played by tools/match.py (play_game), with the rules referee of the repository
(tools/native, through tools/jieqi):

* the two games of a pair use the same hidden deal (Setup.from_seed) and the same random opening, with
  colours swapped, which removes most of the luck of the reveals;
* each engine only gets what its side knows: reveals are public, the identity of a captured face-down
  piece is only told to the capturer (unless full_info);
* the referee ends a game by mate or no legal move, repetition (a draw, or a loss for perpetual check or
  chase), the 80-ply no-capture rule, or the maximum number of plies; an illegal move loses;
* with adjudication on, the side that reports a mate against itself resigns, and a game ends when both
  sides' last four scores agree on a large margin;
* real clocks for time controls such as "10+0.1" (a loss on time when an engine exceeds its time by more
  than the margin); a crash or a timeout loses for the engine at fault.

The referee is the same for both engines and is not part of either build, so a patch that changes the
engine's rules cannot judge its own games.

The functions worker_init() and play_pair() run in the processes of a multiprocessing pool.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_package(name: str, directory: Path):
    """A package loaded from its directory, without putting its parent on sys.path."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, directory / "__init__.py",
                                                  submodule_search_locations=[str(directory)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_match():
    """tools/match.py and tools/jieqi, loaded from their files: tools/ is not put on sys.path, where its
    spsa.py would hide fishtest's."""
    _load_package("jieqi", ROOT / "tools" / "jieqi")
    if "maskfish_match" in sys.modules:
        return sys.modules["maskfish_match"]
    spec = importlib.util.spec_from_file_location("maskfish_match", ROOT / "tools" / "match.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["maskfish_match"] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load_match()
from jieqi.setup import Setup  # noqa: E402

MAX_PLIES = 400
RESULT_STR = {1: "1-0", 0: "1/2-1/2", -1: "0-1"}


# ---------------------------------------------------------------- time control

def parse_tc(tc: str) -> dict:
    """'10+0.1' (seconds + increment), '60', 'nodes=20000', 'depth=8', 'movetime=100' (ms per move)."""
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


# ---------------------------------------------------------------- engines

def _engine_options(side: dict) -> dict:
    o = {}
    if side.get("net"):
        o["EvalFile"] = side["net"]
    o.update(side.get("options", {}))
    return o


def _spec(name: str, side: dict, extra: dict | None = None) -> dict:
    """An engine of tools/match.py: {name, argv, cwd, options, harness, pad}."""
    opts = _engine_options(side)
    opts.update(extra or {})
    return {"name": name, "argv": [side["binary"]], "cwd": str(Path(side["binary"]).parent),
            "options": opts, "harness": {}, "pad": True}


def _fault(reason: str) -> str:
    if reason.startswith("engine failure"):
        return "crash"
    if reason == "TIME_FORFEIT":
        return "time"
    if reason.startswith("illegal move"):
        return "illegal"
    return ""


def play_game(red: dict, black: dict, seed: int, cfg: dict) -> dict:
    """One game; red and black are engine specs. Returns {'result': +1/0/-1 for red, 'reason',
    'moves', 'plies', 'fault_kind'}."""
    tc = cfg["tc"]
    if tc["type"] == "clock":
        go_args, timeout, clock = "", 0, (tc["base"], tc["inc"])
    else:
        go_args, clock = tc["go"], None
        timeout = cfg["move_timeout"]
    old_max = M.MAX_PLIES
    M.MAX_PLIES = cfg.get("max_plies", MAX_PLIES)
    try:
        score, reason, moves = M.play_game(
            _W["rules"], Setup.from_seed(seed), red, black, go_args, cfg["hash"], cfg["threads"], timeout,
            record_scores=False, tc=clock, random_plies=(cfg["plies_min"], cfg["plies_max"]), random_seed=seed,
            time_margin_ms=cfg["timemargin"], adjudicate=cfg["adjudicate"], full_info=cfg["full_info"])
    finally:
        M.MAX_PLIES = old_max
    tokens = []
    for mv in moves:
        t = mv["m"] + (mv["r"] or "")
        if mv.get("c"):
            t += "x" + mv["c"]
        tokens.append(t)
    result = {1.0: 1, 0.5: 0, 0.0: -1}[score]
    return {"result": result, "reason": reason, "moves": " ".join(tokens), "plies": len(moves),
            "fault_kind": _fault(reason)}


# ---------------------------------------------------------------- pool workers

_W: dict = {}


def worker_init(cfg: dict):
    _W["cfg"] = cfg
    _W["error"] = None
    try:
        _W["rules"] = M.NativeRules()
    except Exception as e:      # reported by play_pair (a failing initializer would loop)
        _W["error"] = f"cannot load the referee library: {e}"
    import signal
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, signal.SIG_IGN)   # the runner stops the pool


def play_pair(index: int, seed: int, opts_new: dict | None = None, opts_base: dict | None = None) -> dict:
    """Two games on the same deal and opening, colours swapped. 'scores' are the results of the new engine
    (+1 / 0 / -1). opts_new / opts_base are UCI options set for the pair (SPSA)."""
    if _W.get("error"):
        raise RuntimeError(_W["error"])
    cfg = _W["cfg"]
    new = _spec("new", cfg["new"], opts_new)
    base = _spec("base", cfg["base"], opts_base)
    g1 = play_game(new, base, seed, cfg)      # new has red
    g2 = play_game(base, new, seed, cfg)      # new has black
    g1["red"], g1["black"] = "new", "base"
    g2["red"], g2["black"] = "base", "new"
    scores = (g1["result"], -g2["result"])
    games = []
    for k, g in enumerate((g1, g2)):
        games.append(f"[pair {index} game {k + 1} seed {seed}] red={g['red']} black={g['black']} "
                     f"result={RESULT_STR[g['result']]} reason={g['reason']} plies={g['plies']}\n{g['moves']}\n")
    count = lambda kind: sum(1 for g in (g1, g2) if g["fault_kind"] == kind)
    return {"index": index, "scores": scores, "crashes": count("crash"), "time_losses": count("time"),
            "illegal_moves": count("illegal"), "games": games}


# ---------------------------------------------------------------- checks

def spin_options(binary: str) -> dict:
    """{name: (default, min, max)} of the spin options of an engine."""
    out = {}
    e = M.Engine("options", [binary], str(Path(binary).parent))
    try:
        e.send("uci")
        while True:
            line = e.lines.get(timeout=30)
            if line is None or line.startswith("uciok"):
                break
            m = re.match(r"option name (\S+) type spin default (-?\d+) min (-?\d+) max (-?\d+)", line)
            if m:
                out[m.group(1)] = tuple(int(m.group(i)) for i in (2, 3, 4))
    finally:
        e.close()
    return out


def check_engine(binary: str, side: dict, threads: int, hash_mb: int) -> str:
    """Starts the engine with its options and searches the start position: '' when it works, or the error.
    Maskfish answers an option it does not have with 'info string unknown option NAME', and a net it cannot
    load with 'info string EvalFile load failed'."""
    import queue
    import time
    e = None
    lines = []
    try:
        opts = {"Hash": hash_mb, "Threads": threads, **_engine_options(side)}
        e = M.Engine("check", [binary], str(Path(binary).parent))
        for k, v in opts.items():
            e.send(f"setoption name {k} value {v}")
        e.send("isready")
        deadline = time.monotonic() + 120
        while True:
            line = e.lines.get(timeout=max(0.01, deadline - time.monotonic()))
            if line is None:
                return "the engine terminated:\n" + "\n".join(lines[-20:])
            lines.append(line)
            if line.startswith("readyok"):
                break
        unknown = [l.split("unknown option", 1)[1].strip() for l in lines if "unknown option" in l]
        if unknown:
            return ("no such UCI option in this engine: " + ", ".join(unknown)
                    + " (renamed or removed in its branch?)")
        failed = [l for l in lines if "load failed" in l]
        if failed:
            return failed[0]
        e.send("position startpos")
        e.send("go depth 3")
        line = e.wait_for("bestmove", 60)
        if line.split()[1:2] in (["(none)"], []):
            return "the engine found no move in the start position"
        return ""
    except (M.EngineError, OSError, queue.Empty) as ex:
        return f"{ex}\n" + "\n".join(lines[-20:])
    finally:
        if e is not None:
            e.close()


def check_referee() -> str:
    """Loads the referee library and makes the deal of a seed: '' when it works, or the error."""
    try:
        M.NativeRules()
        Setup.from_seed(1)
        return ""
    except Exception as ex:
        return f"{ex} (JIEQI_NATIVE_LIBRARY={os.environ.get('JIEQI_NATIVE_LIBRARY')})"
