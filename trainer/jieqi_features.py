"""Jieqi NNUE input features, bit-exact with the engine.

This mirrors maskfish/src/nnue/features/half_ka_v2_hm.{h,cpp} and the refresh code in
maskfish/src/nnue/nnue_accumulator.cpp:

* 6 king buckets (with horizontal mirroring) x PS_NB = 7560 inputs per side.
* Visible pieces and kings: one feature per (square, piece kind).
* Dark pieces: one "dark" feature per square (the color is implied by the
  square, dark pieces only stand on their own starting squares).
* Pool of unrevealed pieces: one feature per piece kind and per owner
  (own / opponent), activated once per piece in the pool, i.e. its weight is
  multiplied by the count.

Four architectures exist (select with set_arch() before building a model):

* "default": the engine built normally.
* "att": the engine built with `make nnue=att`. The feature transformer is
  also bucketed by the visible attacking material of the perspective's side
  (4 attack buckets: has a rook? has a knight or a cannon?) and, when both
  kings are on the middle file, the board is mirrored according to its
  left/right balance ("mid mirror"). Both ideas come from Pikafish.
* "threats" / "att-threats": the engine built with `make threats=yes`
  (optionally with `nnue=att`). Threat features are added to the inputs above
  and the feature transformer has 1024 outputs instead of 2048 (see the
  "Threat features" section below). Idea and design from Pikafish/Stockfish
  ("full threats"), adapted to dark pieces.

Each of them also exists with options of the engine build, as name suffixes
in this order (e.g. "threats-l2-nopsqt"):

* "-l2": the layer stacks of `make layers=2` (see model.py);
* "-nopsqt": no PSQT part in the feature transformer (`make psqt=no`).

These two only change the model: the input features, and so the prepared
data, are those of the name without them (feature_arch()).

If you change the feature set in the engine, change it here too and re-run
verify.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

FILE_NB = 9
RANK_NB = 10
SQUARE_NB = 90

PS_NB = 14 * SQUARE_NB              # 1260
NUM_KING_BUCKETS = 6
MAX_ACTIVE = 64                     # engine: MaxActiveDimensions = 62

MAX_REST_COUNT = 5                  # engine: pool counts are clamped to this (types.h)

NUM_PSQT_BUCKETS = 16
NUM_LAYER_STACKS = 16

class Arch(NamedTuple):
    attack_buckets: int
    feature_hash: int
    mid_mirror: bool
    threats: bool
    layers: int = 1
    psqt: bool = True


BASE_ARCHS = {
    # name: (attack buckets, feature hash, mid mirror, threat features)
    "default": (1, 0x0D17B100, False, False),
    "att": (4, 0x7A17B104, True, False),
    "threats": (1, 0x0D17B100, False, True),
    "att-threats": (4, 0x7A17B104, True, True),
}

# Options of the engine build, as name suffixes in this order: (suffix, field
# of Arch, value with the suffix, whether the prepared data depends on it)
SUFFIXES = (
    ("-l2", "layers", 2, False),
    ("-nopsqt", "psqt", False, False),
)
LAYERS2_SUFFIX = "-l2"


def _all_archs() -> dict:
    archs = {}
    for base, values in BASE_ARCHS.items():
        for mask in range(1 << len(SUFFIXES)):
            name, fields = base, {}
            for k, (suffix, field, value, _) in enumerate(SUFFIXES):
                if mask >> k & 1:
                    name += suffix
                    fields[field] = value
            archs[name] = Arch(*values, **fields)
    return archs


ARCHS = _all_archs()


def arch_name(base: str, **fields) -> str:
    """Name of the architecture with these input features and options."""
    name = base
    for suffix, field, value, _ in SUFFIXES:
        if fields.get(field, Arch._field_defaults[field]) == value:
            name += suffix
    return name


def split_arch(name: str) -> tuple[str, dict]:
    """(base name, options) of an architecture."""
    if name not in ARCHS:
        raise ValueError(f"unknown architecture {name!r}")
    a = ARCHS[name]
    base = name
    for suffix, _, _, _ in reversed(SUFFIXES):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
    return base, {field: getattr(a, field) for _, field, _, _ in SUFFIXES}


ARCH = "default"
ATTACK_BUCKETS, FEATURE_HASH, MID_MIRROR, THREATS, LAYERS, PSQT = ARCHS[ARCH]
NUM_FEATURES = NUM_KING_BUCKETS * ATTACK_BUCKETS * PS_NB   # 7560 (default) / 30240 (att)
PAD_INDEX = NUM_FEATURES            # padding entry used by the trainer
L1 = 2048                           # feature transformer outputs (1024 with threats)


def set_arch(name: str):
    """Selects the network architecture (call before creating a model)."""
    global ARCH, ATTACK_BUCKETS, FEATURE_HASH, MID_MIRROR, THREATS, LAYERS, PSQT, NUM_FEATURES
    global PAD_INDEX, L1
    if name not in ARCHS:
        raise ValueError(f"unknown architecture {name!r}, expected one of {list(ARCHS)}")
    ARCH = name
    ATTACK_BUCKETS, FEATURE_HASH, MID_MIRROR, THREATS, LAYERS, PSQT = ARCHS[name]
    NUM_FEATURES = NUM_KING_BUCKETS * ATTACK_BUCKETS * PS_NB
    PAD_INDEX = NUM_FEATURES
    L1 = 1024 if THREATS else 2048


def feature_arch(name: str) -> str:
    """Architecture of the input features (what prepared data is made for):
    the name without the suffixes that only change the model."""
    base, fields = split_arch(name)
    return arch_name(base, **{f: v for (_, f, _, data) in SUFFIXES if data
                              for v in [fields[f]]})


def model_arch(name: str, layers: int = 1, psqt: bool = True) -> str:
    """Architecture with the input features of `name` and these model options."""
    base, fields = split_arch(feature_arch(name))
    fields.update(layers=layers, psqt=psqt)
    return arch_name(base, **fields)


def with_layers(name: str, layers: int) -> str:
    """Architecture `name` with these layer stacks."""
    base, fields = split_arch(name)
    fields["layers"] = layers
    return arch_name(base, **fields)

# Engine piece codes (maskfish/src/types.h): 1..7 white, 9..15 black, 16 = dark
ROOK, ADVISOR, CANNON, PAWN, KNIGHT, BISHOP, KING = range(1, 8)
WHITE, BLACK = 0, 1
DARK_PIECE = 16

PIECE_TO_CHAR = " RACPNBK racpnbkXx"

# Starting squares of the dark pieces, indexed by square (rank 0 first)
DARK_PIECES = ("RNBAKABNR"
               "........."
               ".C.....C."
               "P.P.P.P.P"
               "........."
               "........."
               "p.p.p.p.p"
               ".c.....c."
               "........."
               "rnbakabnr")

PS_W_ROOK, PS_B_ROOK = 0 * SQUARE_NB, 1 * SQUARE_NB
PS_W_CANNON, PS_B_CANNON = 2 * SQUARE_NB, 3 * SQUARE_NB
PS_W_KNIGHT, PS_B_KNIGHT = 4 * SQUARE_NB, 5 * SQUARE_NB
PS_W_PAWN, PS_B_PAWN = 6 * SQUARE_NB, 7 * SQUARE_NB
PS_W_ADVISOR, PS_B_ADVISOR = 8 * SQUARE_NB, 9 * SQUARE_NB
PS_W_BISHOP, PS_B_BISHOP = 10 * SQUARE_NB, 11 * SQUARE_NB
PS_WB_KING = 12 * SQUARE_NB
PS_DARK = 13 * SQUARE_NB
PS_NONE = 0

# PieceSquareIndex[perspective][piece]
_PSI_W = [PS_NONE, PS_W_ROOK, PS_W_ADVISOR, PS_W_CANNON, PS_W_PAWN, PS_W_KNIGHT, PS_W_BISHOP,
          PS_WB_KING, PS_NONE, PS_B_ROOK, PS_B_ADVISOR, PS_B_CANNON, PS_B_PAWN, PS_B_KNIGHT,
          PS_B_BISHOP, PS_WB_KING, PS_DARK]
_PSI_B = [PS_NONE, PS_B_ROOK, PS_B_ADVISOR, PS_B_CANNON, PS_B_PAWN, PS_B_KNIGHT, PS_B_BISHOP,
          PS_WB_KING, PS_NONE, PS_W_ROOK, PS_W_ADVISOR, PS_W_CANNON, PS_W_PAWN, PS_W_KNIGHT,
          PS_W_BISHOP, PS_WB_KING, PS_DARK]
PIECE_SQUARE_INDEX = (_PSI_W, _PSI_B)

# DarkRestIndex[perspective][piece]: own pool at PS_DARK+9.., opponent at PS_DARK+72..
_DRI_W = [PS_NONE] + [PS_DARK + 9 + i for i in range(6)] + [PS_NONE] \
       + [PS_NONE] + [PS_DARK + 72 + i for i in range(6)] + [PS_NONE]
_DRI_B = [PS_NONE] + [PS_DARK + 72 + i for i in range(6)] + [PS_NONE] \
       + [PS_NONE] + [PS_DARK + 9 + i for i in range(6)] + [PS_NONE]
DARK_REST_INDEX = (_DRI_W, _DRI_B)

LAYER_STACK_BUCKETS = [-1, -1, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
                       6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 13, 14, 15]


def make_square(f: int, r: int) -> int:
    return r * FILE_NB + f


def file_of(s: int) -> int:
    return s % FILE_NB


def rank_of(s: int) -> int:
    return s // FILE_NB


def flip_file(s: int) -> int:
    return make_square(FILE_NB - 1 - file_of(s), rank_of(s))


def flip_rank(s: int) -> int:
    return make_square(file_of(s), RANK_NB - 1 - rank_of(s))


def _m(s):
    return (1 << 3) | s


_KB = [
    0, 0, 0, 0, 1, _m(0), 0, 0, 0,
    0, 0, 0, 2, 3, _m(2), 0, 0, 0,
    0, 0, 0, 4, 5, _m(4), 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 4, 5, _m(4), 0, 0, 0,
    0, 0, 0, 2, 3, _m(2), 0, 0, 0,
    0, 0, 0, 0, 1, _m(0), 0, 0, 0,
]


def king_bucket(ksq: int, oksq: int) -> tuple[int, bool]:
    kb = _KB[ksq]
    bucket = kb & 0x7
    mirror = bool(kb >> 3) or (bool(bucket & 1) and bool(_KB[oksq] >> 3))
    return bucket, mirror


# ---- "att" architecture: mid mirror (Pikafish) and attack buckets

MASK64 = (1 << 64) - 1
BALANCE_ENCODING = 0xA4A92A74E989D3A7
_MID_SHIFTS = ((0, 0), (44, 0), (60, 36), (47, 7), (53, 21), (50, 14), (57, 29), (0, 0))


def _mid_mirror_encoding(pc: int, s: int) -> int:
    c, pt = pc >> 3, pc & 7
    f, r = s % FILE_NB, s // FILE_NB
    if f == 4:
        return 0
    if pt == KING:
        return 1 << 63
    r_ = r if c == WHITE else 9 - r
    f_ = f if f < 4 else 8 - f
    s1, s2 = _MID_SHIFTS[pt]
    e = (1 << s1) | (((3 - f_) * 10 + r_) << s2)
    return e if f < 4 else (-e) & MASK64


MID_MIRROR_ENCODING = [[_mid_mirror_encoding(pc, s) if 0 < (pc & 7) <= KING else 0
                        for s in range(SQUARE_NB)] for pc in range(16)]


def mid_encodings(pos) -> list[int]:
    """Sum of the encodings of the pieces of each side (dark pieces count as
    the kind of their square), like Position::midEncoding in the engine."""
    enc = [BALANCE_ENCODING, BALANCE_ENCODING]
    for s in range(SQUARE_NB):
        pc = pos.board[s]
        if pc:
            enc[pc >> 3] = (enc[pc >> 3] + MID_MIRROR_ENCODING[pc][s]) & MASK64
    return enc


def requires_mid_mirror(enc: list[int], c: int) -> bool:
    e, o = enc[c], enc[1 - c]
    return bool((1 << 63) & e & o) and (e < BALANCE_ENCODING
                                       or (e == BALANCE_ENCODING and o < BALANCE_ENCODING))


def attack_bucket(pos, c: int) -> int:
    """Visible attacking material of side c: 2 * has rook + has knight/cannon."""
    rook = minor = False
    for s in range(SQUARE_NB):
        pc = pos.board[s]
        if pc and (pc >> 3) == c and not pos.dark[s]:
            pt = pc & 7
            rook |= pt == ROOK
            minor |= pt in (KNIGHT, CANNON)
    return 2 * int(rook) + int(minor)


def feature_bucket(pos, perspective: int) -> tuple[int, bool]:
    """Feature transformer bucket and mirroring (engine: make_feature_bucket)."""
    ksq = pos.king_square(perspective)
    oksq = pos.king_square(1 - perspective)
    bucket, mirror = king_bucket(ksq, oksq)
    if not MID_MIRROR:
        return bucket, mirror
    if (bucket & 1) and (_KB[oksq] & 1) and requires_mid_mirror(mid_encodings(pos), perspective):
        mirror = True
    return bucket * ATTACK_BUCKETS + attack_bucket(pos, perspective), mirror


# IndexMap[mirror][rotate][square]
INDEX_MAP = [[[0] * SQUARE_NB for _ in range(2)] for _ in range(2)]
for _mi in range(2):
    for _ro in range(2):
        for _s in range(SQUARE_NB):
            _ss = flip_file(_s) if _mi else _s
            _ss = flip_rank(_ss) if _ro else _ss
            INDEX_MAP[_mi][_ro][_s] = _ss


def color_of(pc: int) -> int:
    return pc >> 3


def type_of(pc: int) -> int:
    return pc & 7


def make_piece(c: int, pt: int) -> int:
    return (c << 3) + pt


@dataclass
class JieqiPosition:
    """Board as seen by the features: board[sq] is a piece code, dark[sq] tells
    whether that piece is still hidden (then board[sq] is the kind of piece
    that starts on that square), rest[piece] is the pool count."""
    board: list = field(default_factory=lambda: [0] * SQUARE_NB)
    dark: list = field(default_factory=lambda: [False] * SQUARE_NB)
    rest: list = field(default_factory=lambda: [0] * 16)
    side_to_move: int = WHITE
    rule40: int = 0
    fullmove: int = 1

    def king_square(self, c: int) -> int:
        k = make_piece(c, KING)
        for s in range(SQUARE_NB):
            if self.board[s] == k:
                return s
        raise ValueError("missing king")

    def piece_count(self) -> int:
        return sum(1 for s in range(SQUARE_NB) if self.board[s])


def parse_fen(fen: str) -> JieqiPosition:
    parts = fen.split()
    pos = JieqiPosition()
    rows = parts[0].split('/')
    if len(rows) != RANK_NB:
        raise ValueError(f"bad FEN board: {fen}")
    for i, row in enumerate(rows):
        r = RANK_NB - 1 - i
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
                continue
            if f >= FILE_NB:
                raise ValueError(f"bad FEN row: {fen}")
            s = make_square(f, r)
            if ch in "Xx":
                start = DARK_PIECES[s]
                if start == '.' or start.lower() == 'k' or start.isupper() != (ch == 'X'):
                    raise ValueError(f"dark piece on a wrong square: {fen}")
                pos.board[s] = PIECE_TO_CHAR.index(start)
                pos.dark[s] = True
            else:
                idx = PIECE_TO_CHAR.find(ch)
                if idx <= 0 or idx >= 16 or type_of(idx) == 0:
                    raise ValueError(f"bad FEN piece {ch!r}: {fen}")
                pos.board[s] = idx
            f += 1
    # As Position::set() in the engine: only "b" is black; anything else (also
    # a missing field, or "r" for red as in some xiangqi FENs) is white
    pos.side_to_move = BLACK if len(parts) > 1 and parts[1] == 'b' else WHITE
    if len(parts) > 2 and parts[2] != '-':
        tok = parts[2]
        i = 0
        while i + 1 < len(tok):
            idx = PIECE_TO_CHAR.find(tok[i])
            if 0 < idx < 16 and type_of(idx) not in (0, KING) and tok[i + 1].isdigit():
                pos.rest[idx] = min(int(tok[i + 1]), MAX_REST_COUNT)
            i += 2
    if len(parts) > 3:
        pos.rule40 = int(parts[3])
    if len(parts) > 4:
        pos.fullmove = int(parts[4])
    return pos


def active_features(pos: JieqiPosition, perspective: int) -> list[int]:
    """Active feature indices for one perspective (pool features repeated)."""
    bucket, mirror = feature_bucket(pos, perspective)
    imap = INDEX_MAP[int(mirror)][int(perspective == BLACK)]
    psi = PIECE_SQUARE_INDEX[perspective]
    dri = DARK_REST_INDEX[perspective]
    base = PS_NB * bucket

    out = []
    for s in range(SQUARE_NB):
        pc = pos.board[s]
        if not pc:
            continue
        p = DARK_PIECE if pos.dark[s] else pc
        out.append(imap[s] + psi[p] + base)
    for c in (WHITE, BLACK):
        for pt in (ROOK, ADVISOR, CANNON, PAWN, KNIGHT, BISHOP):
            pc = make_piece(c, pt)
            n = pos.rest[pc]
            if n:
                out.extend([base + dri[pc]] * n)
    return out


def layer_stack_bucket(pos: JieqiPosition) -> int:
    return LAYER_STACK_BUCKETS[pos.piece_count()]


def fen_to_features(fen: str):
    """Returns (stm_features, nstm_features, bucket, side_to_move)."""
    pos = parse_fen(fen)
    stm = pos.side_to_move
    return (active_features(pos, stm), active_features(pos, 1 - stm),
            layer_stack_bucket(pos), stm)



# ---------------------------------------------------------------- threat features
#
# Mirrors maskfish/src/nnue/features/jieqi_threats.{h,cpp} (engine built with
# `make threats=yes`). One feature per (attacker, from, to, attacked) where
# the piece on `to` is attacked (or defended) by the piece on `from`:
#
# * pieces are grouped in 16 classes: class = piece code for a revealed piece
#   (1..7 white, 9..15 black) and color * 8 for a dark piece (0 white, 8 black);
# * a dark piece attacks like the kind of piece that starts on its square; a
#   dark advisor only inside the palace (it must stay there when it moves);
# * attacks are those of the engine (maskfish/src/bitboard.cpp): rook slides, cannon
#   captures over exactly one screen, knight and bishop can be blocked, revealed
#   advisors and bishops go anywhere (Jieqi rules), king steps in the palace,
#   pawns forward and also sideways once across the river;
# * squares are seen from the perspective (rotated for black, mirrored like the
#   HalfKA features) and the colors of both classes are swapped for black;
# * a threat between two pieces of the same kind that always attack each other
#   (not knights; pawns facing each other or side by side) is counted once:
#   only the direction with from > to (after the square mapping) is a feature.
#
# The feature index is BASE[a][from][to] + rank of the attacked class among the
# classes allowed on (a, from, to), enumerated in the same order as the engine:
# attacker class, from square, to square (ascending), attacked class.

NUM_CLASSES = 16
MAX_THREATS = 144                    # engine: JieqiThreats::MaxActiveDimensions
THREAT_HASH = 0x5B1E7A3D             # engine: JieqiThreats::HashValue

_ORTH = ((0, 1), (0, -1), (1, 0), (-1, 0))
_DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_KNIGHT = ((1, 2), (-1, 2), (1, -2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1))
_BISHOP = ((2, 2), (2, -2), (-2, 2), (-2, -2))
PALACE = frozenset(make_square(f, r) for f in (3, 4, 5) for r in (0, 1, 2, 7, 8, 9))


def _on_board(f: int, r: int) -> bool:
    return 0 <= f < FILE_NB and 0 <= r < RANK_NB


def _dark_kind(s: int) -> int:
    """Piece type that starts on square s (0 if none or the king)."""
    ch = DARK_PIECES[s]
    return 0 if ch in ".kK" else type_of(PIECE_TO_CHAR.index(ch))


def piece_attacks(pt: int, c: int, s: int, occupied, dark: bool = False) -> list:
    """Squares attacked by a piece of type pt and color c on s (engine rules).
    `occupied` supports `in`; for sliders the first blocker is included."""
    f, r = file_of(s), rank_of(s)
    out = []
    if pt == ROOK or pt == CANNON:
        for df, dr in _ORTH:
            x, y, hurdle = f + df, r + dr, False
            while _on_board(x, y):
                t = make_square(x, y)
                if pt == ROOK or hurdle:
                    out.append(t)
                if t in occupied:
                    if pt == CANNON and not hurdle:
                        hurdle = True
                    else:
                        break
                x, y = x + df, y + dr
    elif pt == KNIGHT:
        for df, dr in _KNIGHT:
            if _on_board(f + df, r + dr):
                leg = make_square(f + df // 2, r) if abs(df) == 2 else make_square(f, r + dr // 2)
                if leg not in occupied:
                    out.append(make_square(f + df, r + dr))
    elif pt == BISHOP:
        for df, dr in _BISHOP:
            if _on_board(f + df, r + dr) and make_square(f + df // 2, r + dr // 2) not in occupied:
                out.append(make_square(f + df, r + dr))
    elif pt == ADVISOR:
        for df, dr in _DIAG:
            if _on_board(f + df, r + dr):
                t = make_square(f + df, r + dr)
                if not dark or t in PALACE:
                    out.append(t)
    elif pt == KING:
        for df, dr in _ORTH:
            if _on_board(f + df, r + dr) and make_square(f + df, r + dr) in PALACE:
                out.append(make_square(f + df, r + dr))
    elif pt == PAWN:
        fwd = 1 if c == WHITE else -1
        if _on_board(f, r + fwd):
            out.append(make_square(f, r + fwd))
        if (r > 4) if c == WHITE else (r < 5):
            for df in (1, -1):
                if _on_board(f + df, r):
                    out.append(make_square(f + df, r))
    return out


def threat_class(pc: int, dark: bool) -> int:
    return (pc >> 3) << 3 if dark else pc


def _class_movement(cls: int, s: int) -> tuple[int, bool]:
    """(piece type it moves like, dark) for a class on square s."""
    if cls & 7:
        return cls & 7, False
    return _dark_kind(s), True


def _class_squares(cls: int) -> frozenset:
    c, pt = cls >> 3, cls & 7
    if pt == 0:
        return frozenset(s for s in range(SQUARE_NB) if _dark_kind(s)
                         and DARK_PIECES[s].isupper() == (c == WHITE))
    if pt == KING:
        return frozenset(s for s in PALACE if (rank_of(s) < 5) == (c == WHITE))
    return frozenset(range(SQUARE_NB))


def _empty_targets(cls: int, s: int) -> list:
    """Every square the class could attack from s on some board (a superset:
    sliders on an empty board, cannons with a screen next to them)."""
    pt, dark = _class_movement(cls, s)
    occupied = ()
    if pt == CANNON:
        f, r = file_of(s), rank_of(s)
        occupied = {make_square(f + df, r + dr) for df, dr in _ORTH if _on_board(f + df, r + dr)}
    return piece_attacks(pt, cls >> 3, s, occupied, dark)


def _semi_excluded(a: int, b: int, frm: int, to: int) -> bool:
    """Threats that always come in pairs (both pieces attack each other):
    only the one with frm > to is kept."""
    pa, pb = a & 7, b & 7
    if pa == 0 or pa != pb or pa == KNIGHT:
        return False
    if pa == PAWN:
        enemy = (a >> 3) != (b >> 3)
        if not ((enemy and file_of(frm) == file_of(to))
                or (not enemy and rank_of(frm) == rank_of(to))):
            return False
    return not frm > to


def _king_excluded(a: int, b: int, frm: int) -> bool:
    """Threats involving the kings that are not features: the king as an
    attacker, an advisor next to its own king, and the checks of rooks,
    cannons and knights. A dark piece counts as the kind of its square."""
    pa = (a & 7) or _dark_kind(frm)
    if pa == KING:
        return True
    if (b & 7) != KING:
        return False
    if (a >> 3) == (b >> 3):
        return pa == ADVISOR
    return pa in (ROOK, CANNON, KNIGHT)


def _build_threat_tables():
    squares = [_class_squares(cl) for cl in range(NUM_CLASSES)]
    base = [0] * (NUM_CLASSES * SQUARE_NB * SQUARE_NB)
    mask = [0] * (NUM_CLASSES * SQUARE_NB * SQUARE_NB)
    n = 0
    for a in range(NUM_CLASSES):
        for frm in range(SQUARE_NB):
            if frm not in squares[a]:
                continue
            for to in sorted(set(_empty_targets(a, frm))):
                m = 0
                for b in range(NUM_CLASSES):
                    if (to in squares[b] and not _semi_excluded(a, b, frm, to)
                            and not _king_excluded(a, b, frm)):
                        m |= 1 << b
                k = (a * SQUARE_NB + frm) * SQUARE_NB + to
                base[k], mask[k] = n, m
                n += bin(m).count("1")
    return base, mask, n


THREAT_BASE, THREAT_MASK, THREAT_DIMS = _build_threat_tables()   # 92711 features


def make_threat_index(a: int, frm: int, to: int, b: int) -> int:
    """Index of a threat whose squares are already mapped to the perspective and
    whose classes are already color-swapped for black; -1 if not a feature."""
    k = (a * SQUARE_NB + frm) * SQUARE_NB + to
    m = THREAT_MASK[k]
    if not (m >> b) & 1:
        return -1
    return THREAT_BASE[k] + bin(m & ((1 << b) - 1)).count("1")


def physical_threats(pos: JieqiPosition) -> list:
    """All (attacker class, from, to, attacked class) of a position."""
    occupied = {s for s in range(SQUARE_NB) if pos.board[s]}
    out = []
    for frm in sorted(occupied):
        pc, dark = pos.board[frm], pos.dark[frm]
        a = threat_class(pc, dark)
        for to in piece_attacks(pc & 7, pc >> 3, frm, occupied, dark):
            if to in occupied:
                out.append((a, frm, to, threat_class(pos.board[to], pos.dark[to])))
    return out


def active_threats(pos: JieqiPosition, perspective: int, physical=None) -> list:
    """Active threat feature indices for one perspective, in ascending order."""
    if physical is None:
        physical = physical_threats(pos)
    _, mirror = feature_bucket(pos, perspective)
    imap = INDEX_MAP[int(mirror)][int(perspective == BLACK)]
    flip = 8 if perspective == BLACK else 0
    out = []
    for a, frm, to, b in physical:
        i = make_threat_index(a ^ flip, imap[frm], imap[to], b ^ flip)
        if i >= 0:
            out.append(i)
    out.sort()
    return out


def fen_to_threats(fen: str):
    """Returns (stm_threats, nstm_threats) for the threat architectures."""
    pos = parse_fen(fen)
    phys = physical_threats(pos)
    stm = pos.side_to_move
    return active_threats(pos, stm, phys), active_threats(pos, 1 - stm, phys)

# Piece values (maskfish/src/types.h), in engine internal units, for the PSQT init
PIECE_VALUE = {ROOK: 1080, ADVISOR: 243, CANNON: 769, PAWN: 149, KNIGHT: 616, BISHOP: 218,
               KING: 0}
INITIAL_POOL = {ROOK: 2, ADVISOR: 2, CANNON: 2, PAWN: 5, KNIGHT: 2, BISHOP: 2}
AVERAGE_DARK_VALUE = sum(PIECE_VALUE[pt] * n for pt, n in INITIAL_POOL.items()) / 15.0


def feature_material_values() -> list[float]:
    """Material value (internal units, from the perspective's side) of each
    feature, used to initialize the PSQT weights: own pieces positive,
    opponent's negative, dark pieces at the average value of a hidden piece."""
    values = [0.0] * NUM_FEATURES
    for bucket in range(NUM_KING_BUCKETS * ATTACK_BUCKETS):
        base = PS_NB * bucket
        for pt, ps_own, ps_opp in ((ROOK, PS_W_ROOK, PS_B_ROOK),
                                   (CANNON, PS_W_CANNON, PS_B_CANNON),
                                   (KNIGHT, PS_W_KNIGHT, PS_B_KNIGHT),
                                   (PAWN, PS_W_PAWN, PS_B_PAWN),
                                   (ADVISOR, PS_W_ADVISOR, PS_B_ADVISOR),
                                   (BISHOP, PS_W_BISHOP, PS_B_BISHOP)):
            for s in range(SQUARE_NB):
                values[base + ps_own + s] = PIECE_VALUE[pt]
                values[base + ps_opp + s] = -PIECE_VALUE[pt]
        # Dark squares: the perspective's own dark pieces start on ranks 0-3
        # once the board is seen from that side (index map rotates for black).
        for s in range(SQUARE_NB):
            if DARK_PIECES[s] == '.' or DARK_PIECES[s].lower() == 'k':
                continue
            values[base + PS_DARK + s] = (AVERAGE_DARK_VALUE if DARK_PIECES[s].isupper()
                                          else -AVERAGE_DARK_VALUE)
        # Pool features stay at 0: the dark pieces on the board already count.
    return values
