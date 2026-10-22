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

That is the engine's default, EvalScaling Complexity. With the UCI option
EvalScaling Alignment (Pikafish's "Replace nnueComplexity with simple_eval
equivalent", which does not use the psqt / positional split and so suits the
networks without PSQT), train.py --eval-scaling alignment:

    se          = simple_eval (material balance for the side to move)
    se_norm     = se * 1024 / (|se| + 1024)
    nnue_norm   = nnue * 1024 / (|nnue| + 1024)
    alignment   = se_norm * nnue_norm / 512
    v           = (nnue + nnue * alignment / 65536) * (80030 + total material) / 80030
    v          -= v * rule40 / 217

(C++ integer divisions), where the total material counts every piece but
the kings, a dark piece as the kind of piece of its square (JieqiMaterial
does not apply). prepare_data.py stores se and the total material of every
position for it ("se", "allmat").

Networks without PSQT ("-nopsqt", make psqt=no) with EvalScaling Complexity:
psqt is 0, and the material stands for it in the complexity term, as in
Pikafish ("Replace nnueComplexity with simple_eval equivalent"):

    complexity  = |2 * sed - nnue|

where sed is simple_eval_dark(): the material balance for the side to move, a
dark piece counting at the average value of its side's pool
(Position::dark_value()). prepare_data.py stores it too ("sed").

(EvalScaling Correlation of the engine is not modelled.)
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
_MINOR = (F.PAWN, F.ADVISOR, F.BISHOP)
SCALINGS = ("complexity", "alignment")


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


def simple_eval(pos: F.JieqiPosition) -> int:
    """simple_eval() of evaluate.cpp: material balance (rooks, cannons,
    knights, pawns, advisors, bishops) for the side to move, a dark piece
    counting as the kind of piece of its square."""
    total = 0
    for s in range(F.SQUARE_NB):
        pc = pos.board[s]
        if pc and F.type_of(pc) in _MAJOR + _MINOR:
            v = F.PIECE_VALUE[F.type_of(pc)]
            total += v if F.color_of(pc) == pos.side_to_move else -v
    return total


def dark_value(pos: F.JieqiPosition, c: int) -> int:
    """Position::dark_value(): average value of the pool of a side."""
    count = value = 0
    for pt in (F.ROOK, F.ADVISOR, F.CANNON, F.PAWN, F.KNIGHT, F.BISHOP):
        n = pos.rest[F.make_piece(c, pt)]
        count += n
        value += n * F.PIECE_VALUE[pt]
    return value // count if count else 0


def simple_eval_dark(pos: F.JieqiPosition) -> int:
    """simple_eval_dark() of evaluate.cpp: material of the side to move minus
    that of the other side (kings excluded), a dark piece counting at
    dark_value(). It stands for the PSQT part of networks without one in
    EvalScaling Complexity."""
    material = [0, 0]
    for c in (F.WHITE, F.BLACK):
        dv = dark_value(pos, c)
        for s in range(F.SQUARE_NB):
            pc = pos.board[s]
            if pc and F.color_of(pc) == c and F.type_of(pc) != F.KING:
                material[c] += dv if pos.dark[s] else F.PIECE_VALUE[F.type_of(pc)]
    us = pos.side_to_move
    return material[us] - material[1 - us]


def total_material(pos: F.JieqiPosition) -> int:
    """total_material() of evaluate.cpp: every piece of both sides but the kings."""
    return sum(F.PIECE_VALUE[F.type_of(pc)] for pc in pos.board
               if pc and F.type_of(pc) in _MAJOR + _MINOR)


def alignment_inputs(pos: F.JieqiPosition) -> tuple[int, int]:
    """(se, total material) of a position, for EvalScaling Alignment."""
    return simple_eval(pos), total_material(pos)


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


ALIGN_NORM = 1024
ALIGN_DIV = 512
ALIGN_EVAL_DIV = 65536
ALIGN_MATERIAL_BASE = 80030


def evaluate_int(psqt: int, positional: int, material: int, rule40: int,
                 scaling: str = "complexity", se: int = 0, allmat: int = 0,
                 sed: int | None = None) -> int:
    """Eval::evaluate() with optimism 0, bit-exact (integer arithmetic, with the
    engine's 32-bit ints: they only wrap around for network outputs far beyond
    the mate range, which a trained network never reaches). scaling
    "alignment" (EvalScaling Alignment) uses se and allmat instead of material.
    sed (simple_eval_dark()) is given for networks without PSQT: it stands for
    the PSQT part in the complexity term."""
    nnue = _i32(psqt + positional)
    if scaling == "alignment":
        se_norm = _tdiv(se * ALIGN_NORM, abs(se) + ALIGN_NORM)
        nnue_norm = _i32(_tdiv(nnue * ALIGN_NORM, abs(nnue) + ALIGN_NORM))
        align = _tdiv(se_norm * nnue_norm, ALIGN_DIV)
        base = _i32(nnue + _tdiv(nnue * align, ALIGN_EVAL_DIV))
        v = _i32(_tdiv(base * (ALIGN_MATERIAL_BASE + allmat), ALIGN_MATERIAL_BASE))
        v = _i32(v - _tdiv(v * min(rule40, RULE40_MAX), RULE40_DIV))
        return max(-VALUE_MATE_IN_MAX_PLY + 1, min(VALUE_MATE_IN_MAX_PLY - 1, v))
    complexity = abs(_i32(psqt - positional)) if sed is None else abs(_i32(2 * sed - nnue))
    nnue = _i32(nnue - _tdiv(nnue * complexity, COMPLEXITY_DIV))
    v = _i32(_tdiv(nnue * (MATERIAL_BASE + material), MATERIAL_DIV))
    v = _i32(v - _tdiv(v * min(rule40, RULE40_MAX), RULE40_DIV))   # 64-bit product, as the engine
    return max(-VALUE_MATE_IN_MAX_PLY + 1, min(VALUE_MATE_IN_MAX_PLY - 1, v))


def evaluate_float(psqt, positional, material, rule40, scaling: str = "complexity", se=None,
                   allmat=None, sed=None):
    """Differentiable version for training (torch tensors or numpy arrays, internal
    units). Differs from evaluate_int() only by the integer truncations."""
    if scaling == "alignment":
        nnue = psqt + positional
        se_norm = se * ALIGN_NORM / (abs(se) + ALIGN_NORM)
        nnue_norm = nnue * ALIGN_NORM / (abs(nnue) + ALIGN_NORM)
        align = se_norm * nnue_norm / ALIGN_DIV
        v = (nnue + nnue * align / ALIGN_EVAL_DIV) * ((ALIGN_MATERIAL_BASE + allmat)
                                                     / ALIGN_MATERIAL_BASE)
        return v * (1.0 - rule40 / RULE40_DIV)
    nnue = psqt + positional
    complexity = abs(psqt - positional) if sed is None else abs(2.0 * sed - nnue)
    nnue = nnue * (1.0 - complexity / COMPLEXITY_DIV)
    v = nnue * ((MATERIAL_BASE + material) / MATERIAL_DIV)
    return v * (1.0 - rule40 / RULE40_DIV)   # rule40 is capped by position_scale_inputs()


def evaluate_qat(psqt, positional, material, rule40, scaling: str = "complexity", se=None,
                 allmat=None, sed=None):
    """evaluate_int() on torch tensors holding integer values (the output of
    the quantization-aware forward pass, model.forward_parts_qat()), with the
    same truncations, differentiable through them (straight-through), the
    products in float64: equal to evaluate_int() (except for the 32-bit
    wrap-around, far beyond real scores)."""
    import torch
    from model import q_tdiv, wide
    dtype = psqt.dtype
    psqt, positional, material, rule40 = (wide(t) for t in (psqt, positional, material, rule40))
    nnue = psqt + positional
    if scaling == "alignment":
        se, allmat = wide(se), wide(allmat)
        se_norm = torch.trunc(se * ALIGN_NORM / (torch.abs(se) + ALIGN_NORM))
        nnue_norm = q_tdiv(nnue * ALIGN_NORM, torch.abs(nnue) + ALIGN_NORM)
        align = q_tdiv(se_norm * nnue_norm, ALIGN_DIV)
        base = nnue + q_tdiv(nnue * align, ALIGN_EVAL_DIV)
        v = q_tdiv(base * (ALIGN_MATERIAL_BASE + allmat), ALIGN_MATERIAL_BASE)
        v = v - q_tdiv(v * torch.clamp(rule40, max=RULE40_MAX), RULE40_DIV)
        return torch.clamp(v, -VALUE_MATE_IN_MAX_PLY + 1, VALUE_MATE_IN_MAX_PLY - 1).to(dtype)
    complexity = torch.abs(psqt - positional) if sed is None else torch.abs(2 * wide(sed) - nnue)
    nnue = nnue - q_tdiv(nnue * complexity, COMPLEXITY_DIV)
    v = q_tdiv(nnue * (MATERIAL_BASE + material), MATERIAL_DIV)
    v = v - q_tdiv(v * torch.clamp(rule40, max=RULE40_MAX), RULE40_DIV)
    return torch.clamp(v, -VALUE_MATE_IN_MAX_PLY + 1, VALUE_MATE_IN_MAX_PLY - 1).to(dtype)
