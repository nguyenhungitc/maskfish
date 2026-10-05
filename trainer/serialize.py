"""Export a trained model to the engine's .nnue format, and read one back.

File layout (src/nnue/network.cpp, nnue_feature_transformer.h, layers/):

    u32 version, u32 network hash, u32 len, description
    u32 feature transformer hash
      LEB128 int16 biases[2048]           (scale 127, the engine doubles them)
      LEB128 int16 weights[7560][2048]
      LEB128 int32 psqt[7560][16]         (scale 600 * 16)
    16 x layer stack:
      u32 architecture hash
      fc0: int32 bias[16]  (scale 64*127), int8 weight[16][2048] (scale 64)
      fc1: int32 bias[32]  (scale 64*127), int8 weight[32][32]   (30 used)
      fc2: int32 bias[1]   (scale 600*16), int8 weight[1][32]    (scale 600*16/127)

The whole file is then compressed with zstd, which the engine requires.

Usage:
    python serialize.py checkpoint.pt out.nnue [--description "..."]
    python serialize.py --to-checkpoint in.nnue out.pt     (start from a net)
"""

from __future__ import annotations

import argparse
import io
import struct

import numpy as np

import jieqi_features as F

VERSION = 0x7AF32F20
FT_HASH = 0x0D17A100        # FEATURE_HASH ^ (2048 * 2)
ARCH_HASH = 0x63336B4A
NETWORK_HASH = FT_HASH ^ ARCH_HASH
LEB128_MAGIC = b"COMPRESSED_LEB128"

L1, L2, L3 = 2048, 15, 32
PSQT = F.NUM_PSQT_BUCKETS
STACKS = F.NUM_LAYER_STACKS
FC1_PADDED_IN = 32


# ---------------------------------------------------------------- LEB128

def leb128_encode(values: np.ndarray) -> bytes:
    """Signed LEB128, like write_leb_128() in src/nnue/nnue_common.h (vectorized)."""
    v = values.astype(np.int64).ravel()
    n = np.ones(v.shape, dtype=np.int64)
    for k in range(1, 10):                       # bytes needed: v fits in 7k signed bits
        lim = 1 << (7 * k - 1)
        n += ~((v >= -lim) & (v < lim))
    offsets = np.cumsum(n) - n
    out = np.zeros(int(n.sum()), dtype=np.uint8)
    for j in range(int(n.max())):
        mask = j < n
        byte = (v >> (7 * j)) & 0x7F
        byte = byte | np.where(j < n - 1, 0x80, 0)
        out[offsets[mask] + j] = byte[mask].astype(np.uint8)
    return out.tobytes()


def leb128_decode(data: bytes, count: int) -> np.ndarray:
    b = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    ends = np.flatnonzero((b & 0x80) == 0)
    if len(ends) < count:
        raise ValueError("truncated LEB128 block")
    ends = ends[:count]
    starts = np.concatenate([[0], ends[:-1] + 1])
    lengths = ends - starts + 1
    result = np.zeros(count, dtype=np.int64)
    for j in range(int(lengths.max())):
        mask = j < lengths
        result[mask] |= (b[starts[mask] + j] & 0x7F) << (7 * j)
    sign = (b[ends] & 0x40) != 0
    result[sign] -= np.left_shift(1, 7 * lengths[sign])
    return result


def _write_leb(f, values: np.ndarray):
    payload = leb128_encode(values.ravel())
    f.write(LEB128_MAGIC)
    f.write(struct.pack("<I", len(payload)))
    f.write(payload)


def _read_leb(f, count: int) -> np.ndarray:
    if f.read(len(LEB128_MAGIC)) != LEB128_MAGIC:
        raise ValueError("missing LEB128 block")
    (n,) = struct.unpack("<I", f.read(4))
    return leb128_decode(f.read(n), count)


# ---------------------------------------------------------------- quantization

def quantize(model) -> dict:
    """Integer parameters exactly as they are written to the file."""
    import torch
    from model import (FT_SCALE, HIDDEN_WEIGHT_SCALE, QUANTIZED_ONE, OUTPUT_WEIGHT_SCALE,
                       NNUE2SCORE, OUTPUT_SCALE, MAX_HIDDEN_WEIGHT, MAX_OUTPUT_WEIGHT)

    with torch.no_grad():
        w = model.ft.weight[:F.NUM_FEATURES].detach().cpu().double().numpy()
        b = model.ft_bias.detach().cpu().double().numpy()

        q = {}
        q["ft_bias"] = np.round(b[:L1] * FT_SCALE).astype(np.int64)
        q["ft_weight"] = np.round(w[:, :L1] * FT_SCALE).astype(np.int64)
        q["psqt"] = np.round(w[:, L1:] * NNUE2SCORE * OUTPUT_SCALE).astype(np.int64)
        for name in ("ft_bias", "ft_weight"):
            if np.abs(q[name]).max() > 16383:   # doubled by the engine into int16
                raise ValueError(f"{name} out of range: {np.abs(q[name]).max()}")

        st = model.stacks
        l1w = st.l1.weight.detach().cpu().double().numpy().reshape(STACKS, L2 + 1, L1)
        l1b = st.l1.bias.detach().cpu().double().numpy().reshape(STACKS, L2 + 1)
        l2w = st.l2.weight.detach().cpu().double().numpy().reshape(STACKS, L3, 2 * L2)
        l2b = st.l2.bias.detach().cpu().double().numpy().reshape(STACKS, L3)
        ow = st.output.weight.detach().cpu().double().numpy().reshape(STACKS, L3)
        ob = st.output.bias.detach().cpu().double().numpy().reshape(STACKS)

    hb = HIDDEN_WEIGHT_SCALE * QUANTIZED_ONE
    q["fc0_bias"] = np.round(l1b * hb).astype(np.int64)
    q["fc0_weight"] = np.round(np.clip(l1w, -MAX_HIDDEN_WEIGHT, MAX_HIDDEN_WEIGHT)
                               * HIDDEN_WEIGHT_SCALE).astype(np.int64)
    q["fc1_bias"] = np.round(l2b * hb).astype(np.int64)
    fc1 = np.zeros((STACKS, L3, FC1_PADDED_IN), dtype=np.int64)
    fc1[:, :, :2 * L2] = np.round(np.clip(l2w, -MAX_HIDDEN_WEIGHT, MAX_HIDDEN_WEIGHT)
                                  * HIDDEN_WEIGHT_SCALE)
    q["fc1_weight"] = fc1
    q["fc2_bias"] = np.round(ob * NNUE2SCORE * OUTPUT_SCALE).astype(np.int64)
    q["fc2_weight"] = np.round(np.clip(ow, -MAX_OUTPUT_WEIGHT, MAX_OUTPUT_WEIGHT)
                               * OUTPUT_WEIGHT_SCALE).astype(np.int64)
    for k in ("fc0_weight", "fc1_weight", "fc2_weight"):
        assert np.abs(q[k]).max() <= 127, k
    return q


def write_nnue(q: dict, path: str, description: str = "Jieqi NNUE trained with trainer/"):
    buf = io.BytesIO()
    desc = description.encode("utf-8")
    buf.write(struct.pack("<III", VERSION, NETWORK_HASH, len(desc)))
    buf.write(desc)

    buf.write(struct.pack("<I", FT_HASH))
    _write_leb(buf, q["ft_bias"].astype(np.int16))
    _write_leb(buf, q["ft_weight"].astype(np.int16))
    _write_leb(buf, q["psqt"].astype(np.int32))

    for k in range(STACKS):
        buf.write(struct.pack("<I", ARCH_HASH))
        buf.write(q["fc0_bias"][k].astype("<i4").tobytes())
        buf.write(q["fc0_weight"][k].astype(np.int8).tobytes())
        buf.write(q["fc1_bias"][k].astype("<i4").tobytes())
        buf.write(q["fc1_weight"][k].astype(np.int8).tobytes())
        buf.write(q["fc2_bias"][k:k + 1].astype("<i4").tobytes())
        buf.write(q["fc2_weight"][k].astype(np.int8).tobytes())

    import zstandard
    with open(path, "wb") as f:
        f.write(zstandard.ZstdCompressor(level=19, threads=-1).compress(buf.getvalue()))


def read_nnue(path: str) -> dict:
    import zstandard
    raw = open(path, "rb").read()
    try:
        raw = zstandard.ZstdDecompressor().decompressobj().decompress(raw)
    except zstandard.ZstdError:
        pass  # not compressed
    f = io.BytesIO(raw)
    version, nhash, dlen = struct.unpack("<III", f.read(12))
    if version != VERSION or nhash != NETWORK_HASH:
        raise ValueError(f"incompatible network (version {version:#x}, hash {nhash:#x})")
    description = f.read(dlen).decode("utf-8", "replace")
    (fth,) = struct.unpack("<I", f.read(4))
    if fth != FT_HASH:
        raise ValueError("feature transformer hash mismatch")
    q = {"description": description}
    q["ft_bias"] = _read_leb(f, L1)
    q["ft_weight"] = _read_leb(f, F.NUM_FEATURES * L1).reshape(F.NUM_FEATURES, L1)
    q["psqt"] = _read_leb(f, F.NUM_FEATURES * PSQT).reshape(F.NUM_FEATURES, PSQT)

    def arr(dtype, n):
        return np.frombuffer(f.read(np.dtype(dtype).itemsize * n), dtype=dtype).astype(np.int64)

    keys = ("fc0_bias", "fc0_weight", "fc1_bias", "fc1_weight", "fc2_bias", "fc2_weight")
    parts = {k: [] for k in keys}
    for _ in range(STACKS):
        (ah,) = struct.unpack("<I", f.read(4))
        if ah != ARCH_HASH:
            raise ValueError("architecture hash mismatch")
        parts["fc0_bias"].append(arr("<i4", L2 + 1))
        parts["fc0_weight"].append(arr(np.int8, (L2 + 1) * L1).reshape(L2 + 1, L1))
        parts["fc1_bias"].append(arr("<i4", L3))
        parts["fc1_weight"].append(arr(np.int8, L3 * FC1_PADDED_IN).reshape(L3, FC1_PADDED_IN))
        parts["fc2_bias"].append(arr("<i4", 1)[0])
        parts["fc2_weight"].append(arr(np.int8, L3))
    for k in keys:
        q[k] = np.array(parts[k])
    return q


def load_into_model(q: dict, model):
    """Dequantize a network into a model (to continue training from it)."""
    import torch
    from model import (FT_SCALE, HIDDEN_WEIGHT_SCALE, QUANTIZED_ONE, OUTPUT_WEIGHT_SCALE,
                       NNUE2SCORE, OUTPUT_SCALE)
    hb = HIDDEN_WEIGHT_SCALE * QUANTIZED_ONE
    with torch.no_grad():
        model.ft.weight[:F.NUM_FEATURES, :L1] = torch.tensor(q["ft_weight"] / FT_SCALE)
        model.ft.weight[:F.NUM_FEATURES, L1:] = torch.tensor(q["psqt"] / (NNUE2SCORE * OUTPUT_SCALE))
        model.ft.weight[F.PAD_INDEX].zero_()
        model.ft_bias[:L1] = torch.tensor(q["ft_bias"] / FT_SCALE)
        model.ft_bias[L1:] = 0
        st = model.stacks
        st.l1.weight[:] = torch.tensor(q["fc0_weight"].reshape(-1, L1) / HIDDEN_WEIGHT_SCALE)
        st.l1.bias[:] = torch.tensor(q["fc0_bias"].reshape(-1) / hb)
        st.l2.weight[:] = torch.tensor(q["fc1_weight"][:, :, :2 * L2].reshape(-1, 2 * L2)
                                       / HIDDEN_WEIGHT_SCALE)
        st.l2.bias[:] = torch.tensor(q["fc1_bias"].reshape(-1) / hb)
        st.output.weight[:] = torch.tensor(q["fc2_weight"] / OUTPUT_WEIGHT_SCALE)
        st.output.bias[:] = torch.tensor(q["fc2_bias"] / (NNUE2SCORE * OUTPUT_SCALE))


# ---------------------------------------------------------------- integer reference

def _tdiv(a: int, b: int) -> int:
    """C++ integer division (truncates toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def quantized_eval(q: dict, fen: str) -> tuple[int, int, int]:
    """Bit-exact integer evaluation, like Network::evaluate() in the engine.
    Returns (psqt, positional, bucket) in internal units."""
    stm_f, nstm_f, bucket, _ = F.fen_to_features(fen)
    transformed = []
    psqt_acc = []
    for feats in (stm_f, nstm_f):
        acc = 2 * q["ft_bias"].copy()               # the engine doubles the FT on load
        ps = np.zeros(PSQT, dtype=np.int64)
        for i in feats:
            acc += 2 * q["ft_weight"][i]
            ps += q["psqt"][i]
        a = np.clip(acc[:L1 // 2], 0, 254)
        b = np.clip(acc[L1 // 2:], 0, 254)
        transformed.append((a * b) // 512)
        psqt_acc.append(ps[bucket])
    x = np.concatenate(transformed)                  # uint8 in the engine

    fc0 = q["fc0_bias"][bucket] + q["fc0_weight"][bucket] @ x
    sqr = np.minimum(127, (fc0[:L2] * fc0[:L2]) >> 19)
    crelu = np.clip(fc0[:L2] >> 6, 0, 127)
    h = np.zeros(FC1_PADDED_IN, dtype=np.int64)
    h[:L2] = sqr
    h[L2:2 * L2] = crelu
    fc1 = q["fc1_bias"][bucket] + q["fc1_weight"][bucket] @ h
    h2 = np.clip(fc1 >> 6, 0, 127)
    fc2 = int(q["fc2_bias"][bucket] + q["fc2_weight"][bucket] @ h2)
    fwd = _tdiv(int(fc0[L2]) * (600 * 16), 127 * 64)
    positional = fc2 + fwd
    psqt = _tdiv(int(psqt_acc[0] - psqt_acc[1]), 2)
    return _tdiv(psqt, 16), _tdiv(positional, 16), bucket


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("target")
    ap.add_argument("--description", default="Jieqi NNUE trained with trainer/")
    ap.add_argument("--to-checkpoint", action="store_true",
                    help="convert a .nnue file into a checkpoint to continue training")
    args = ap.parse_args()

    import torch
    from model import NNUE

    if args.to_checkpoint:
        model = NNUE(init_psqt=False)
        load_into_model(read_nnue(args.source), model)
        torch.save({"model": model.state_dict()}, args.target)
    else:
        ckpt = torch.load(args.source, map_location="cpu")
        model = NNUE(init_psqt=False)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        write_nnue(quantize(model), args.target, args.description)
    print("written", args.target)


if __name__ == "__main__":
    main()
