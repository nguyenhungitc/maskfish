"""SPSA tuning, as on fishtest.

All the parameters are perturbed at the same time: for iteration k, every
parameter gets a random sign flip_i = +-1, the engine "plus" plays with
theta_i + c_k flip_i and the engine "minus" with theta_i - c_k flip_i (one
game pair, colors swapped), and each parameter moves toward the side that won:

    theta_i += R_k * c_k * result * flip_i        result = wins - losses of "plus"

with the gains of fishtest, for N = number of iterations (game pairs):

    A   = A_ratio * N
    c   = c_end * N^gamma              c_k = c / k^gamma          (c_N = c_end)
    a   = r_end * c_end^2 * (A + N)^alpha
    R_k = a / (A + k)^alpha / c_k^2                               (R_N = r_end)

Parameters are given one per line, in the format printed by the engine for
TUNE() (src/tune.cpp), which is the format of fishtest:

    name,start,min,max,c_end,r_end

A line may also be just the name of a UCI option of the engine (e.g.
RevealQuietBase): its default, min and max are then read from the engine, with
c_end = (max - min) / 20 and r_end = 0.0020 (the defaults of TUNE()).
"""

from __future__ import annotations

import math
import random

DEFAULT_A_RATIO = 0.1
DEFAULT_ALPHA = 0.602
DEFAULT_GAMMA = 0.101
DEFAULT_R_END = 0.002
MAX_HISTORY = 500


class SpsaError(ValueError):
    pass


def parse_params(text: str) -> list[dict]:
    """Checks the syntax. Returns [{'name', 'start', 'min', 'max', 'c_end',
    'r_end'}] with None for the values to be read from the engine."""
    params, seen = [], set()
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#")[0].strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        name = parts[0]
        if not name or " " in name or "=" in name:
            raise SpsaError(f"line {n}: bad parameter name {name!r}")
        if name in seen:
            raise SpsaError(f"line {n}: {name} appears twice")
        seen.add(name)
        if len(parts) not in (1, 4, 5, 6):
            raise SpsaError(f"line {n}: expected name,start,min,max,c_end,r_end (or just a name)")
        try:
            vals = [float(p) for p in parts[1:]]
        except ValueError:
            raise SpsaError(f"line {n}: the values must be numbers")
        p = {"name": name, "start": None, "min": None, "max": None, "c_end": None, "r_end": None}
        for key, v in zip(("start", "min", "max", "c_end", "r_end"), vals):
            p[key] = v
        if p["min"] is not None:
            if not p["min"] < p["max"]:
                raise SpsaError(f"line {n}: min must be below max")
            if not p["min"] <= p["start"] <= p["max"]:
                raise SpsaError(f"line {n}: start must be between min and max")
        if p["c_end"] is not None and p["c_end"] <= 0:
            raise SpsaError(f"line {n}: c_end must be positive")
        if p["r_end"] is not None and p["r_end"] <= 0:
            raise SpsaError(f"line {n}: r_end must be positive")
        params.append(p)
    if not params:
        raise SpsaError("no parameter to tune")
    return params


def resolve(params: list[dict], engine_options: dict) -> list[dict]:
    """Fills the missing values from the engine's spin options
    {name: (default, min, max)} and checks that every parameter is one."""
    out = []
    for p in params:
        if p["name"] not in engine_options:
            raise SpsaError(f"{p['name']} is not a numeric (spin) UCI option of the engine")
        d, lo, hi = engine_options[p["name"]]
        q = dict(p)
        if q["start"] is None:
            q.update(start=float(d), min=float(lo), max=float(hi))
        if q["c_end"] is None:
            q["c_end"] = max((q["max"] - q["min"]) / 20, 1e-9)
        if q["r_end"] is None:
            q["r_end"] = DEFAULT_R_END
        q["min"], q["max"] = max(q["min"], lo), min(q["max"], hi)
        q["start"] = min(max(q["start"], q["min"]), q["max"])
        out.append(q)
    return out


def init_state(params: list[dict], pairs: int, a_ratio: float, alpha: float, gamma: float) -> dict:
    n = max(1, pairs)
    big_a = a_ratio * n
    for p in params:
        p["theta"] = p["start"]
        p["c"] = p["c_end"] * n ** gamma
        p["a"] = p["r_end"] * p["c_end"] ** 2 * (big_a + n) ** alpha
    return {"A": big_a, "alpha": alpha, "gamma": gamma, "pairs": n, "params": params,
            "history": [[0, [p["theta"] for p in params]]]}


def gains(state: dict, k: int):
    """[(c_k, R_k)] of every parameter at iteration k (1-based)."""
    out = []
    for p in state["params"]:
        c = p["c"] / k ** state["gamma"]
        r = p["a"] / (state["A"] + k) ** state["alpha"] / c ** 2
        out.append((c, r))
    return out


def iround(x: float) -> int:
    return int(math.floor(x + 0.5))


def clamp(p, v):
    return min(max(v, p["min"]), p["max"])


def perturb(state: dict, k: int, rng: random.Random) -> dict:
    """The values of both engines for iteration k. UCI spin options take
    integers, so the values sent are rounded (theta keeps its fraction)."""
    g = gains(state, k)
    flips = [rng.choice((-1, 1)) for _ in state["params"]]
    plus, minus = {}, {}
    for p, f, (c, _) in zip(state["params"], flips, g):
        plus[p["name"]] = iround(clamp(p, p["theta"] + c * f))
        minus[p["name"]] = iround(clamp(p, p["theta"] - c * f))
    return {"k": k, "flips": flips, "gains": g, "plus": plus, "minus": minus}


def update(state: dict, step: dict, result: int, done_pairs: int):
    """result: wins - losses of the "plus" engine in the pair (-2..2)."""
    for p, f, (c, r) in zip(state["params"], step["flips"], step["gains"]):
        p["theta"] = clamp(p, p["theta"] + r * c * result * f)
    hist = state["history"]
    hist.append([done_pairs, [round(p["theta"], 4) for p in state["params"]]])
    if len(hist) > MAX_HISTORY:
        state["history"] = hist[:1] + hist[1:-1][::2] + hist[-1:]


def current_values(state: dict) -> dict:
    return {p["name"]: iround(p["theta"]) for p in state["params"]}


def params_text(params: list[dict]) -> str:
    def f(x):
        return f"{x:g}"
    return "\n".join(",".join([p["name"], f(p["start"]), f(p["min"]), f(p["max"]),
                               f(p["c_end"]), f(p["r_end"])]) for p in params)
