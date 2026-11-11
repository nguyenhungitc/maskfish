"""ctypes interface to the C++ rules referee (tools/native, the library BUILD/lib/libjieqi_rules.so).

What the harness uses: legal moves and check of a public board (NativeRules), and a game played move by move whose
status the referee adjudicates (NativeGame: legality, repetition, perpetual check and chase, 80-ply rule).
There is no Python fallback and no automatic compiler/package installation.
"""
import ctypes as ct
from dataclasses import dataclass
from enum import IntEnum
import os
from pathlib import Path

from .setup import Color, Kind, coordinate, square


class NativeUnavailable(FileNotFoundError):
    pass


class NativeError(ValueError):
    pass


class GameStatus(IntEnum):
    ONGOING = 0
    DRAW = 1
    RED_LOSES = 2
    BLACK_LOSES = 3




def _uint32(value, name):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError(name + " must be a uint32")
    return value






class NativeRules:
    def __init__(self, library_path=None):
        if library_path is None:
            library_path = os.environ.get("JIEQI_NATIVE_LIBRARY") or None  # empty: unset
        if library_path is None:
            root = Path(__file__).resolve().parents[2]  # the repository (tools/jieqi/native.py)
            candidates = [root / "build/lib/libjieqi_rules.so", root / "build/lib/libjieqi_rules.dylib",
                          root / "build/bin/jieqi_rules.dll", root / "build/bin/Debug/jieqi_rules.dll",
                          root / "build/bin/Release/jieqi_rules.dll"]
            library_path = next((p for p in candidates if p.is_file()), None)
        if library_path is None or not Path(library_path).is_file():
            raise NativeUnavailable("Build the repository with CMake (the library lands in build/lib), or set JIEQI_NATIVE_LIBRARY")
        self.library_path = Path(library_path).resolve()
        self._lib = ct.CDLL(str(self.library_path))
        self._lib.jq_abi_version.argtypes = []
        self._lib.jq_abi_version.restype = ct.c_int
        if self._lib.jq_abi_version() != 5:
            raise NativeError("native ABI version mismatch")
        base = [ct.c_char_p, ct.c_size_t, ct.c_int]
        error = [ct.POINTER(ct.c_char), ct.c_size_t]
        self._lib.jq_game_destroy.argtypes = [ct.c_void_p]
        self._lib.jq_game_destroy.restype = None
        signatures = {
            "jq_game_create": base + [ct.c_uint32, ct.c_uint32, ct.POINTER(ct.c_void_p)] + error,
            "jq_game_advance": [ct.c_void_p, ct.c_int, ct.c_int, ct.c_char] + error,
            "jq_game_snapshot": [ct.c_void_p, ct.POINTER(ct.c_char), ct.c_size_t, ct.POINTER(ct.c_int),
                ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_int)] + error,
            "jq_legal_moves": base + [ct.POINTER(ct.c_uint16), ct.c_size_t, ct.POINTER(ct.c_size_t)] + error,
            "jq_in_check": base + [ct.POINTER(ct.c_int)] + error,
        }
        for name, args in signatures.items():
            function = getattr(self._lib, name)
            function.argtypes = args
            function.restype = ct.c_int

    @staticmethod
    def _board_args(board, side):
        if type(board) is not str or len(board) != 90 or not isinstance(side, Color):
            raise ValueError("expected a 90-cell public board and Color")
        return board.encode("ascii"), 90, 0 if side is Color.RED else 1

    def _call(self, name, *args):
        error = ct.create_string_buffer(512)
        status = getattr(self._lib, name)(*args, error, len(error))
        if status != 0:
            raise NativeError(f"{name} failed ({status}): " + error.value.decode("utf-8", errors="replace"))

    def legal_moves(self, board, side):
        output = (ct.c_uint16 * 8100)()
        count = ct.c_size_t()
        self._call("jq_legal_moves", *self._board_args(board, side), output, len(output), ct.byref(count))
        if count.value > len(output):
            raise NativeError("native move count exceeded output capacity")
        return tuple(coordinate(code // 90) + coordinate(code % 90) for code in output[:count.value])

    def in_check(self, board, side):
        result = ct.c_int()
        self._call("jq_in_check", *self._board_args(board, side), ct.byref(result))
        return bool(result.value)

    @staticmethod
    def _history_args(moves, reveals):
        if type(moves) not in (tuple, list):
            raise ValueError("moves must be a list or tuple")
        if reveals is None:
            reveals = [None] * len(moves)
        if type(reveals) not in (tuple, list) or len(reveals) != len(moves):
            raise ValueError("one reveal entry per move required")
        codes, letters = [], bytearray()
        for move, reveal in zip(moves, reveals):
            if type(move) is not str or len(move) != 4:
                raise ValueError("expected four-coordinate move")
            if reveal is not None and (not isinstance(reveal, Kind) or reveal is Kind.KING):
                raise ValueError("expected nonking Kind or None")
            codes.append(square(move[:2]) * 90 + square(move[2:]))
            letters.append(0 if reveal is None else ord(reveal.value))
        return (ct.c_uint16 * len(codes))(*codes), bytes(letters)


@dataclass(frozen=True, slots=True)
class GameSnapshot:
    board: str
    turn: Color
    no_capture_plies: int
    plies: int
    status: GameStatus


class NativeGame:
    """Owned public native history. Single owner/thread; use with or close()."""
    def __init__(self, rules, board, side, no_capture_plies=0, no_capture_limit=80):
        self._rules, self._handle = rules, ct.c_void_p()
        rules._call("jq_game_create", *rules._board_args(board, side),
                    _uint32(no_capture_plies, "no_capture_plies"), _uint32(no_capture_limit, "no_capture_limit"),
                    ct.byref(self._handle))
        if not self._handle: raise NativeError("native returned a null game")

    def _open(self):
        if not self._handle: raise NativeError("native game is closed")
        return self._handle

    def close(self):
        handle = getattr(self, "_handle", None)
        if handle:
            self._handle = ct.c_void_p()
            self._rules._lib.jq_game_destroy(handle)

    def __del__(self):
        self.close()

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *_):
        self.close()

    def snapshot(self):
        board=ct.create_string_buffer(90)
        side, idle, plies, result=ct.c_int(), ct.c_uint32(), ct.c_size_t(), ct.c_int()
        self._rules._call("jq_game_snapshot", self._open(), board, len(board), ct.byref(side), ct.byref(idle),
                          ct.byref(plies), ct.byref(result))
        if side.value not in (0, 1): raise NativeError("invalid native turn")
        return GameSnapshot(board.raw.decode("ascii"), Color.RED if side.value==0 else Color.BLACK,
                            idle.value, plies.value, GameStatus(result.value))

    def advance(self, move, revealed_mover=None):
        codes, letters=self._rules._history_args([move], [revealed_mover])
        self._rules._call("jq_game_advance", self._open(), codes[0]//90, codes[0]%90, bytes(letters))

