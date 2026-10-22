"""Text data files, plain or compressed.

gen_data.py, rescore.py and the files they write ("fen | score | result",
about 100 bytes per position) compress very well: zstd shrinks them about
5 to 8 times. Every script that reads text data opens it with open_text(),
which picks the format from the file name:

    data/gen0.txt        plain text
    data/gen0.txt.zst    zstd (zstandard package, already a requirement)
    data/gen0.txt.gz     gzip

so a file can be compressed (`zstd --rm data/gen0.txt`) and used as is.
gen_data.py writes a .zst file when its --out ends with .zst.
"""

from __future__ import annotations

import gzip
import io

ZSTD_LEVEL = 10          # ratio close to the maximum, still >100 MB/s per core


def is_compressed(path: str) -> bool:
    return path.endswith((".zst", ".gz"))


def open_text(path: str, mode: str = "r", level: int = ZSTD_LEVEL):
    """Opens a text data file for reading ("r") or writing ("w"), plain,
    zstd (.zst) or gzip (.gz) according to its name."""
    if mode not in ("r", "w"):
        raise ValueError(f"mode {mode!r}: only 'r' and 'w' are supported")
    if path.endswith(".zst"):
        import zstandard
        if mode == "r":
            raw = zstandard.ZstdDecompressor().stream_reader(open(path, "rb"), closefd=True)
        else:
            raw = zstandard.ZstdCompressor(level=level, threads=-1).stream_writer(
                open(path, "wb"), closefd=True)
        return io.TextIOWrapper(raw, encoding="utf-8")
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")
