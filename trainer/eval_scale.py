"""The scaling that Eval::evaluate() (maskfish/src/evaluate.cpp) applies to the
network output, in Python.

The search scores written by gen_data.py are made of evaluate() values, not of
raw network outputs:

    nnue        = psqt + positional                      (network output)
    complexity  = |psqt - positional|
    nnue       -= nnue * complexity / 11418
    v           = nnue * (18560 + material) / 33331      (+ an optimism term, 0 here)
    v          -= v * rule40 / 217

With the material of Jieqi (at most 9860), the factor is always below 1:
evaluate() is about 0.5 to 0.85 times the network output. Training the raw
network output to match the search scores would make every generation learn
a smaller scale than the previous one (the engine applies the factor again
on top of what was learned). train.py therefore applies this same function to
the network output before comparing it with the data score, so that the
engine's evaluate() of the trained network matches the scores it learnt from.

`material` is Position::major_material() (rook, cannon and knight values of
both sides; a dark piece counts as the kind of piece of its square), or
Position::jieqi_major_material() when the data was generated with the
JieqiMaterial option (prepare_data.py --jieqi-material). Optimism (a search
term) is left out, as in the engine's `evalraw` command.
"""

from __future__ import annotations

import jieqi_features as F

COMPLEXITY_DIV = 11418
MATERIAL_BASE = 18560
MATERIAL_DIV = 33331
RULE40_DIV = 217
RULE40_MAX = 80                         # evaluate() caps the counter (the game is drawn there)
VALUE_MATE_IN_MAX_PLY = 32000 - 246     # VALUE_MATE - MAX_PLY (maskfish/src/types.h)

_MAJOR = (F.ROOK, F.CANNON, F.KNIGHT)


def major_material(pos: F.JieqiPosition) -> int:
    """Position::major_material(): value of the rooks, cannons and knights of both
    sides, a dark piece counting as the kind of piece that starts on its square."""
    total = 0
    for s in range(F.SQUARE_NB):
        pc = pos.board[s]
        if pc and F.type_of(pc) in _MAJOR:
            total += F.PIECE_VALUE[F.type_of(pc)]
    return total


def jieqi_major_material(pos: F.JieqiPosition) -> int:
    """Position::jieqi_major_material(): like major_material(), with each dark piece
    counted at the expected major value of its side's pool."""
    total = major_material(pos)
    for c in (F.WHITE, F.BLACK):
        dark = [s for s in range(F.SQUARE_NB) if pos.dark[s] and F.color_of(pos.board[s]) == c]
        if not dark:
            continue
        for s in dark:
            if F.type_of(pos.board[s]) in _MAJOR:
                total -= F.PIECE_VALUE[F.type_of(pos.board[s])]
        pool_count = pool_major = 0
        for pt in (F.ROOK, F.ADVISOR, F.CANNON, F.PAWN, F.KNIGHT, F.BISHOP):
            n = pos.rest[F.make_piece(c, pt)]
            pool_count += n
            if pt in _MAJOR:
                pool_major += n * F.PIECE_VALUE[pt]
        if pool_count:
            total += len(dark) * pool_major // pool_count   # both >= 0: C++ division
    return total


def position_scale_inputs(pos: F.JieqiPosition, jieqi_material: bool = False) -> tuple[int, int]:
    """(material, rule40) of a position, as evaluate() uses them."""
    m = jieqi_major_material(pos) if jieqi_material else major_material(pos)
    return m, min(RULE40_MAX, max(0, pos.rule40))


def _tdiv(a: int, b: int) -> int:
    """C++ integer division (truncates toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def _i32(x: int) -> int:
    """Conversion to a 32-bit int as the engine does it (wraps around)."""
    return (x + (1 << 31)) % (1 << 32) - (1 << 31)


def evaluate_int(psqt: int, positional: int, material: int, rule40: int) -> int:
    """Eval::evaluate() with optimism 0, bit-exact (integer arithmetic, with the
    engine's 32-bit ints: they only wrap around for network outputs far beyond
    the mate range, which a trained network never reaches)."""
    nnue = _i32(psqt + positional)
    complexity = abs(_i32(psqt - positional))
    nnue = _i32(nnue - _tdiv(nnue * complexity, COMPLEXITY_DIV))
    v = _i32(_tdiv(nnue * (MATERIAL_BASE + material), MATERIAL_DIV))
    v = _i32(v - _tdiv(v * min(rule40, RULE40_MAX), RULE40_DIV))   # 64-bit product, as the engine
    return max(-VALUE_MATE_IN_MAX_PLY + 1, min(VALUE_MATE_IN_MAX_PLY - 1, v))


def evaluate_float(psqt, positional, material, rule40):
    """Differentiable version for training (torch tensors or numpy arrays, internal
    units). Differs from evaluate_int() only by the integer truncations."""
    nnue = (psqt + positional) * (1.0 - abs(psqt - positional) / COMPLEXITY_DIV)
    v = nnue * ((MATERIAL_BASE + material) / MATERIAL_DIV)
    return v * (1.0 - rule40 / RULE40_DIV)   # rule40 is capped by position_scale_inputs()
