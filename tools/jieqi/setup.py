"""Portable deterministic deals. Everything in this module is referee-only truth."""
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
import hashlib


class Color(StrEnum):
    RED = "red"
    BLACK = "black"

    @property
    def opponent(self):
        return Color.BLACK if self is Color.RED else Color.RED


class Kind(StrEnum):
    ROOK = "R"
    HORSE = "N"
    ELEPHANT = "B"
    ADVISOR = "A"
    CANNON = "C"
    SOLDIER = "P"
    KING = "K"


POOL_KINDS = tuple(Kind(x) for x in "RNBACP")
INVENTORY = (2, 2, 2, 2, 2, 5)
GENERATOR = "sha256-fy32-v1"
EXPLICIT = "explicit-v1"
RED_STARTS = (0, 1, 2, 3, 5, 6, 7, 8, 19, 25, 27, 29, 31, 33, 35)
START_ROLES = tuple(Kind(x) for x in "RNBAABNRCCPPPPP")


def square(text: str) -> int:
    if type(text) is not str or len(text) != 2 or text[0] not in "abcdefghi" or text[1] not in "0123456789":
        raise ValueError("square must be a0 through i9")
    return int(text[1]) * 9 + ord(text[0]) - ord("a")


def coordinate(index: int) -> str:
    if type(index) is not int or not 0 <= index < 90:
        raise ValueError("square index must be an integer in [0, 90)")
    return chr(ord("a") + index % 9) + str(index // 9)


def start_squares(color: Color) -> tuple[int, ...]:
    if not isinstance(color, Color):
        raise ValueError("expected Color")
    return RED_STARTS if color is Color.RED else tuple((9 - s // 9) * 9 + s % 9 for s in RED_STARTS)


@dataclass(frozen=True, slots=True)
class Piece:
    color: Color
    kind: Kind
    hidden: bool

    def __post_init__(self):
        if not isinstance(self.color, Color) or not isinstance(self.kind, Kind) or type(self.hidden) is not bool:
            raise ValueError("invalid piece fields")
        if self.kind is Kind.KING and self.hidden:
            raise ValueError("kings cannot be hidden")

    def code(self, *, reveal: bool = False) -> str:
        letter = self.kind.value if reveal or not self.hidden else "X"
        return letter if self.color is Color.RED else letter.lower()


class _WordStream:
    """Versioned SHA-256 stream, independent of Python's random implementation."""
    def __init__(self, seed: int, color: Color):
        self.prefix = b"jieqi/setup/v1\0" + seed.to_bytes(8, "little") + color.value.encode("ascii") + b"\0"
        self.block = 0
        self.words = []

    def word(self) -> int:
        if not self.words:
            digest = hashlib.sha256(self.prefix + self.block.to_bytes(8, "little")).digest()
            self.block += 1
            self.words = [int.from_bytes(digest[i:i+4], "little") for i in range(28, -1, -4)]
        return self.words.pop()

    def below(self, n: int) -> int:
        # Rejection removes modulo bias, assuming uniformly distributed hash words.
        limit = (2**32 // n) * n
        while True:
            value = self.word()
            if value < limit:
                return value % n


def _seeded_board(seed: int) -> tuple[Piece | None, ...]:
    """Generate truth without constructing Setup, allowing constructor validation."""
    board = [None] * 90
    for color, king_square in ((Color.RED, 4), (Color.BLACK, 85)):
        pieces = [kind for kind, count in zip(POOL_KINDS, INVENTORY) for _ in range(count)]
        stream = _WordStream(seed, color)
        for i in range(len(pieces) - 1, 0, -1):
            j = stream.below(i + 1)
            pieces[i], pieces[j] = pieces[j], pieces[i]
        for index, kind in zip(start_squares(color), pieces):
            board[index] = Piece(color, kind, True)
        board[king_square] = Piece(color, Kind.KING, False)
    return tuple(board)


@dataclass(frozen=True, slots=True)
class Setup:
    board: tuple[Piece | None, ...]
    generator: str = EXPLICIT
    seed: int | None = None

    def __post_init__(self):
        if type(self.board) is not tuple or len(self.board) != 90:
            raise ValueError("setup requires an immutable 90-square board")
        if any(piece is not None and not isinstance(piece, Piece) for piece in self.board):
            raise ValueError("invalid board entry")
        if self.generator not in (GENERATOR, EXPLICIT):
            raise ValueError("unknown setup generator")
        if self.generator == GENERATOR:
            if type(self.seed) is not int or not 0 <= self.seed < 2**64:
                raise ValueError("seed must be a uint64")
        elif self.seed is not None:
            raise ValueError("explicit setups cannot assert a generator seed")
        for color, king_square in ((Color.RED, 4), (Color.BLACK, 85)):
            expected = set(start_squares(color)) | {king_square}
            actual = {s for s, p in enumerate(self.board) if p is not None and p.color is color}
            if actual != expected:
                raise ValueError("setup pieces must occupy their standard starting points")
            if self.board[king_square] != Piece(color, Kind.KING, False):
                raise ValueError("king must start face up on its normal point")
            pieces = [self.board[s] for s in start_squares(color)]
            if not all(p.hidden for p in pieces):
                raise ValueError("all nonking setup pieces must be hidden")
            if Counter(p.kind for p in pieces) != Counter(dict(zip(POOL_KINDS, INVENTORY))):
                raise ValueError("setup inventory does not match Jieqi")
        if self.generator == GENERATOR:
            if self.board != _seeded_board(self.seed):
                raise ValueError("seed provenance disagrees with explicit deal")

    @classmethod
    def from_seed(cls, seed: int):
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("seed must be a uint64")
        return cls(_seeded_board(seed), GENERATOR, seed)

