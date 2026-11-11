"""Reads and writes the HNN net file (JQHNNUE1, src/hnn/hnnue.h) as float weights, with numpy only.

The trainer (train.py) exports its weights by rounding them to the file's integer scales; read() divides by the same
scales, so read() then write() gives back the same bytes, and a net trained from what read() returns starts exactly
from the net of the file (train.py --init NET.hnn or NET.nnue). The versions the engine reads (src/hnn/hnnue.cpp):
1 (standard head L1 16, L2 32) and 3 (other heads: L1 16|32, L2 32|64, squared L2).

  d = read("best.hnn")   # dict: version, W, scheme, L1, L2, sq2, ft_bias, ft, l1w, l1b, l2w, l2b, ow, ob
  write("copy.hnn", d)   # the same file
"""
import struct

import numpy as np

MAGIC = b"JQHNNUE1"
NPSQT = 16
NST = 16
KB_OF_SCHEME = [1, 3, 2, 6]


def nfeat(scheme):
    return KB_OF_SCHEME[scheme] * 16 * 90 + 30


def l2_inputs(l1):
    return (2 * (l1 - 1) + 31) // 32 * 32


def read(path):
    b = open(path, "rb").read()
    if b[:8] != MAGIC:
        raise ValueError(f"{path}: not a raw HNN net (JQHNNUE1); decompress a .nnue first (tools/hnn/jqnz_tool d)")
    version, W, nf, scheme, npsqt, nst, l1, l2 = struct.unpack_from("<8I", b, 8)
    if version not in (1, 3) or npsqt != NPSQT or nst != NST or scheme > 3 or nf != nfeat(scheme):
        raise ValueError(f"{path}: unsupported HNN header (version {version}, W {W}, scheme {scheme})")
    o, sq2 = 40, False
    if version == 3:
        sq2 = bool(struct.unpack_from("<4I", b, o)[0]); o += 16
    if (l1, l2) not in ((16, 32), (32, 32), (16, 64), (32, 64)) or (version != 3 and (l1, l2) != (16, 32)):
        raise ValueError(f"{path}: unsupported head L1 {l1} L2 {l2}")

    def take(dtype, n):
        nonlocal o
        a = np.frombuffer(b, dtype, n, o)
        o += a.nbytes
        return a

    d = dict(version=version, W=W, scheme=scheme, L1=l1, L2=l2, sq2=sq2)
    d["ft_bias"] = take("<i2", W).astype(np.float64) / 127
    ft = np.empty((nf, W + NPSQT))
    ft[:, :W] = take("<i2", nf * W).reshape(nf, W) / 127
    ft[:, W:] = take("<i4", nf * NPSQT).reshape(nf, NPSQT) / 16
    d["ft"] = ft
    l2in, n1, no = l2_inputs(l1), 2 * (l1 - 1), l2 * (2 if sq2 else 1)
    l1b, l1w, l2b, l2w, ob, ow = [], [], [], [], [], []
    for _ in range(NST):
        l1b.append(take("<i4", l1) / (64 * 127))
        l1w.append(take(np.int8, W * l1).reshape(W // 4, l1, 4).transpose(1, 0, 2).reshape(l1, W) / 64)
        l2b.append(take("<i4", l2) / (64 * 127))
        l2w.append(take(np.int8, l2 * l2in).reshape(l2in // 4, l2, 4).transpose(1, 0, 2).reshape(l2, l2in)[:, :n1] / 64)
        ob.append(take("<i4", 1)[0] / 16)
        ow.append(take(np.int8, no) * (127 / 16))
    d.update(l1b=np.stack(l1b), l1w=np.stack(l1w), l2b=np.stack(l2b), l2w=np.stack(l2w), ob=np.array(ob), ow=np.stack(ow))
    if o != len(b):
        raise ValueError(f"{path}: bad file size")
    return d


def write(path, d):
    """The export of train.py (HNN.export) for the weights of d; version 1 for the standard head, else 3."""
    W, L1, L2, sq2 = d["W"], d["L1"], d["L2"], d["sq2"]
    std = (L1, L2, sq2) == (16, 32, False)
    l2in, n1 = l2_inputs(L1), 2 * (L1 - 1)

    def q(t, s, lo, hi, dt):
        return np.clip(np.round(np.asarray(t, np.float32) * np.float32(s)), lo, hi).astype(dt)

    with open(path, "wb") as f:
        f.write(MAGIC)
        nf = len(d["ft"])
        if std:
            f.write(struct.pack("<8I", 1, W, nf, d["scheme"], NPSQT, NST, 16, 32))
        else:
            f.write(struct.pack("<8I", 3, W, nf, d["scheme"], NPSQT, NST, L1, L2))
            f.write(struct.pack("<4I", int(sq2), 0, 0, 0))
        f.write(q(d["ft_bias"], 127, -32767, 32767, "<i2").tobytes())
        f.write(q(d["ft"][:, :W], 127, -32767, 32767, "<i2").tobytes())
        f.write(q(d["ft"][:, W:], 16, -2**31, 2**31 - 1, "<i4").tobytes())
        for s in range(NST):
            l2 = np.zeros((L2, l2in))
            l2[:, :n1] = d["l2w"][s]
            f.write(q(d["l1b"][s], 64 * 127, -2**31, 2**31 - 1, "<i4").tobytes())
            f.write(q(d["l1w"][s], 64, -127, 127, np.int8).reshape(L1, W // 4, 4).transpose(1, 0, 2).tobytes())
            f.write(q(d["l2b"][s], 64 * 127, -2**31, 2**31 - 1, "<i4").tobytes())
            f.write(q(l2, 64, -127, 127, np.int8).reshape(L2, l2in // 4, 4).transpose(1, 0, 2).tobytes())
            f.write(q(d["ob"][s:s + 1], 16, -2**31, 2**31 - 1, "<i4").tobytes())
            f.write(q(d["ow"][s], 16 / 127, -127, 127, np.int8).tobytes())
