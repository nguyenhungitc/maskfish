"""Jieqi NNUE input features, bit-exact with the engine.

This mirrors src/nnue/features/half_ka_v2_hm.{h,cpp} and the refresh code in
src/nnue/nnue_accumulator.cpp:

* 6 king buckets (with horizontal mirroring) x PS_NB = 7560 inputs per side.
* Visible pieces and kings: one feature per (square, piece kind).
* Dark pieces: one "dark" feature per square (the color is implied by the
  square, dark pieces only stand on their own starting squares).
* Pool of unrevealed pieces: one feature per piece kind and per owner
  (own / opponent), activated once per piece in the pool, i.e. its weight is
  multiplied by the count.

If you change the feature set in the engine, change it here too and re-run
verify.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

FILE_NB = 9
RANK_NB = 10
SQUARE_NB = 90

PS_NB = 14 * SQUARE_NB              # 1260
NUM_KING_BUCKETS = 6
NUM_FEATURES = NUM_KING_BUCKETS * PS_NB   # 7560
MAX_ACTIVE = 64                     # engine: MaxActiveDimensions = 62
PAD_INDEX = NUM_FEATURES            # padding entry used by the trainer

NUM_PSQT_BUCKETS = 16
NUM_LAYER_STACKS = 16

FEATURE_HASH = 0x0D17B100

# Engine piece codes (src/types.h): 1..7 white, 9..15 black, 16 = dark
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
    pos.side_to_move = WHITE if len(parts) < 2 or parts[1] == 'w' else BLACK
    if len(parts) > 2 and parts[2] != '-':
        tok = parts[2]
        i = 0
        while i + 1 < len(tok):
            idx = PIECE_TO_CHAR.find(tok[i])
            if 0 < idx < 16 and type_of(idx) not in (0, KING) and tok[i + 1].isdigit():
                pos.rest[idx] = min(int(tok[i + 1]), 15)
            i += 2
    if len(parts) > 3:
        pos.rule40 = int(parts[3])
    if len(parts) > 4:
        pos.fullmove = int(parts[4])
    return pos


def active_features(pos: JieqiPosition, perspective: int) -> list[int]:
    """Active feature indices for one perspective (pool features repeated)."""
    ksq = pos.king_square(perspective)
    oksq = pos.king_square(1 - perspective)
    bucket, mirror = king_bucket(ksq, oksq)
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


# Piece values (src/types.h), in engine internal units, for the PSQT init
PIECE_VALUE = {ROOK: 1080, ADVISOR: 243, CANNON: 769, PAWN: 149, KNIGHT: 616, BISHOP: 218,
               KING: 0}
INITIAL_POOL = {ROOK: 2, ADVISOR: 2, CANNON: 2, PAWN: 5, KNIGHT: 2, BISHOP: 2}
AVERAGE_DARK_VALUE = sum(PIECE_VALUE[pt] * n for pt, n in INITIAL_POOL.items()) / 15.0


def feature_material_values() -> list[float]:
    """Material value (internal units, from the perspective's side) of each
    feature, used to initialize the PSQT weights: own pieces positive,
    opponent's negative, dark pieces at the average value of a hidden piece."""
    values = [0.0] * NUM_FEATURES
    for bucket in range(NUM_KING_BUCKETS):
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
