"""Export a trained model to the engine's .nnue format, and read one back.

File layout (src/nnue/network.cpp, nnue_feature_transformer.h, layers/):

    u32 version, u32 network hash, u32 len, description
    u32 feature transformer hash
      LEB128 int16 biases[L1]             (scale 127, |bias| <= 16383)
      raw    int8  weights[7560][L1]      (scale 127)
      LEB128 int32 psqt[7560][16]         (scale 600 * 16)
      threat architectures only:
      raw    int8  threat weights[92711][L1] (scale 127)
      LEB128 int32 threat psqt[92711][16]    (scale 600 * 16)
    The engine doubles the feature transformer outputs (biases + weights),
    so 1.0 is 254 before the clipped ReLU.
    16 x layer stack:
      u32 architecture hash
      fc0: int32 bias[16]  (scale 64*127), int8 weight[16][L1] (scale 64)
      fc1: int32 bias[32]  (scale 64*127), int8 weight[32][32]   (30 used)
      fc2: int32 bias[1]   (scale 600*16), int8 weight[1][32]    (scale 600*16/127)
    or, layer stacks 2 (architectures "*-l2", make layers=2):
      fc0: int32 bias[32], int8 weight[32][L1]
      fc1: int32 bias[32], int8 weight[32][64]
      fc2: int32 bias[1],  int8 weight[1][128]   (same scales)

L1 = 2048, or 1024 for the threat architectures. (7560 is 30240 for "att".)

The whole file is then compressed with zstd, which the engine requires.

Files of version 0x7AF32F20 (before the engine had 8-bit HalfKA weights)
have LEB128 int16 weights[7560][L1] instead; --upgrade converts them (the
evaluation is unchanged if no weight is beyond +-127, which is reported).

Usage:
    python serialize.py checkpoint.pt out.nnue [--description "..."]
    python serialize.py --to-checkpoint in.nnue out.pt     (start from a net)
    python serialize.py --upgrade old.nnue new.nnue        (old file version)
"""

from __future__ import annotations

import argparse
import io
import struct

import numpy as np

import jieqi_features as F

VERSION = 0x7AF32F21
OLD_VERSION = 0x7AF32F20      # 16-bit HalfKA weights (LEB128)
MAX_FT_BIAS = 16383           # keeps the engine accumulators from wrapping around
LEB128_MAGIC = b"COMPRESSED_LEB128"

L2, L3 = 15, 32
L2_V2 = 32


def stack_dims(layers: int | None = None) -> dict:
    """Sizes of the layer stacks: fc0 outputs, activated fc0 outputs, fc1
    inputs (used, and in the file), fc2 inputs."""
    if (F.LAYERS if layers is None else layers) == 2:
        return {"fc0_out": L2_V2, "act0": L2_V2, "fc1_in": 2 * L2_V2, "fc1_file": 2 * L2_V2,
                "fc2_in": 2 * L2_V2 + 2 * L3}
    return {"fc0_out": L2 + 1, "act0": L2, "fc1_in": 2 * L2, "fc1_file": 32, "fc2_in": L3}


def arch_hash(l1: int | None = None, layers: int | None = None) -> int:
    """Layer stacks hash, computed like NetworkArchitecture::get_hash_value()
    (0x63336B4A for L1 = 2048, 0x63336A4A for 1024)."""
    m = 0xFFFFFFFF
    l1 = F.L1 if l1 is None else l1

    def affine(prev, outputs):
        h = (0xCC03DAE4 + outputs) & m
        return h ^ (prev >> 1) ^ ((prev << 31) & m)

    def crelu(prev):
        return (0x538D24C7 + prev) & m

    h = 0xEC42E90D ^ (l1 * 2)
    h = crelu(affine(h, stack_dims(layers)["fc0_out"]))
    h = crelu(affine(h, L3))
    return affine(h, 1)


def ft_hash() -> int:
    """Feature transformer hash: feature set hash(es) ^ (L1 * 2)."""
    h = F.FEATURE_HASH ^ (F.THREAT_HASH if F.THREATS else 0)
    return h ^ (F.L1 * 2)


def network_hash() -> int:
    return ft_hash() ^ arch_hash()


ARCH_HASH = arch_hash(2048)   # kept for compatibility (default architecture)
PSQT = F.NUM_PSQT_BUCKETS
STACKS = F.NUM_LAYER_STACKS
FC1_PADDED_IN = 32


# ---------------------------------------------------------------- LEB128

LEB128_CHUNK = 1 << 22


def _leb128_encode_chunk(v: np.ndarray) -> bytes:
    v = v.astype(np.int64)
    n = np.ones(v.shape, dtype=np.int64)
    for k in range(1, 10):                       # bytes needed: v fits in 7k signed bits
        lim = 1 << (7 * k - 1)
        n += ~((v >= -lim) & (v < lim))
    offsets = np.cumsum(n) - n
    out = np.zeros(int(n.sum()), dtype=np.uint8)
    for j in range(int(n.max())):
        mask = j < n
        byte = ((v >> (7 * j)) & 0x7F) | np.where(j < n - 1, 0x80, 0)
        out[offsets[mask] + j] = byte[mask].astype(np.uint8)
    return out.tobytes()


def leb128_encode(values: np.ndarray) -> bytes:
    """Signed LEB128, like write_leb_128() in src/nnue/nnue_common.h
    (vectorized, in chunks to bound memory)."""
    v = values.ravel()
    return b"".join(_leb128_encode_chunk(v[i:i + LEB128_CHUNK])
                    for i in range(0, len(v), LEB128_CHUNK))


def leb128_decode(data: bytes, count: int, dtype=np.int64) -> np.ndarray:
    b = np.frombuffer(data, dtype=np.uint8)
    out = np.empty(count, dtype=dtype)
    pos, done = 0, 0
    while done < count:
        k = min(LEB128_CHUNK, count - done)
        window = b[pos:pos + 10 * k].astype(np.int64)
        ends = np.flatnonzero((window & 0x80) == 0)
        if len(ends) < k:
            raise ValueError("truncated LEB128 block")
        ends = ends[:k]
        starts = np.concatenate([[0], ends[:-1] + 1])
        lengths = ends - starts + 1
        result = np.zeros(k, dtype=np.int64)
        for j in range(int(lengths.max())):
            mask = j < lengths
            result[mask] |= (window[starts[mask] + j] & 0x7F) << (7 * j)
        sign = (window[ends] & 0x40) != 0
        result[sign] -= np.left_shift(1, 7 * lengths[sign])
        out[done:done + k] = result
        done += k
        pos += int(ends[-1]) + 1
    return out


def _write_leb(f, values: np.ndarray):
    payload = leb128_encode(values.ravel())
    f.write(LEB128_MAGIC)
    f.write(struct.pack("<I", len(payload)))
    f.write(payload)


def _read_leb(f, count: int, dtype=np.int64) -> np.ndarray:
    if f.read(len(LEB128_MAGIC)) != LEB128_MAGIC:
        raise ValueError("missing LEB128 block")
    (n,) = struct.unpack("<I", f.read(4))
    return leb128_decode(f.read(n), count, dtype)


# ---------------------------------------------------------------- quantization

def quantize(model) -> dict:
    """Integer parameters exactly as they are written to the file. A
    factorized model is coalesced first (virtual features and the shared fc0
    part are folded into the real weights)."""
    import torch
    from model import (FT_SCALE, HIDDEN_WEIGHT_SCALE, QUANTIZED_ONE, OUTPUT_WEIGHT_SCALE,
                       NNUE2SCORE, OUTPUT_SCALE, MAX_HIDDEN_WEIGHT, MAX_OUTPUT_WEIGHT)

    L1 = F.L1
    model = model.coalesced()
    with torch.no_grad():
        # Feature transformer: done with torch in float32 to keep memory low
        # (the "att" architecture has 30240 x 2048 weights)
        w = model.ft.weight.detach()[:F.NUM_FEATURES]
        b = model.ft_bias.detach().cpu().double().numpy()

        q = {}
        q["ft_bias"] = np.round(b[:L1] * FT_SCALE).astype(np.int64)
        ftw = torch.round(w[:, :L1].float() * FT_SCALE)
        if ftw.abs().max().item() > 127:
            raise ValueError("feature transformer weights out of the int8 range "
                             "(model.clip_weights() keeps them inside)")
        if np.abs(q["ft_bias"]).max() > MAX_FT_BIAS:
            raise ValueError("feature transformer biases out of range")
        q["ft_weight"] = ftw.to(torch.int8).cpu().numpy()
        del ftw
        q["psqt"] = np.round(w[:, L1:].cpu().double().numpy() * NNUE2SCORE * OUTPUT_SCALE).astype(np.int64)
        if F.THREATS:
            tw = model.ft_thr.weight.detach()
            tq = torch.round(tw[:, :L1].float() * FT_SCALE)
            if tq.abs().max().item() > 127:
                raise ValueError("threat weights out of the int8 range")
            q["thr_weight"] = tq.to(torch.int8).cpu().numpy()
            del tq
            q["thr_psqt"] = np.round(tw[:, L1:].cpu().double().numpy()
                                     * NNUE2SCORE * OUTPUT_SCALE).astype(np.int64)

        d = stack_dims()
        st = model.stacks
        l1w = st.l1.weight.detach().cpu().double().numpy().reshape(STACKS, d["fc0_out"], L1)
        l1b = st.l1.bias.detach().cpu().double().numpy().reshape(STACKS, d["fc0_out"])
        l2w = st.l2.weight.detach().cpu().double().numpy().reshape(STACKS, L3, d["fc1_in"])
        l2b = st.l2.bias.detach().cpu().double().numpy().reshape(STACKS, L3)
        ow = st.output.weight.detach().cpu().double().numpy().reshape(STACKS, d["fc2_in"])
        ob = st.output.bias.detach().cpu().double().numpy().reshape(STACKS)

    hb = HIDDEN_WEIGHT_SCALE * QUANTIZED_ONE
    q["fc0_bias"] = np.round(l1b * hb).astype(np.int64)
    q["fc0_weight"] = np.round(np.clip(l1w, -MAX_HIDDEN_WEIGHT, MAX_HIDDEN_WEIGHT)
                               * HIDDEN_WEIGHT_SCALE).astype(np.int64)
    q["fc1_bias"] = np.round(l2b * hb).astype(np.int64)
    fc1 = np.zeros((STACKS, L3, d["fc1_file"]), dtype=np.int64)
    fc1[:, :, :d["fc1_in"]] = np.round(np.clip(l2w, -MAX_HIDDEN_WEIGHT, MAX_HIDDEN_WEIGHT)
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
    buf.write(struct.pack("<III", VERSION, network_hash(), len(desc)))
    buf.write(desc)

    buf.write(struct.pack("<I", ft_hash()))
    if np.abs(q["ft_weight"]).max() > 127 or np.abs(q["ft_bias"]).max() > MAX_FT_BIAS:
        raise ValueError("feature transformer parameters out of range")
    _write_leb(buf, q["ft_bias"].astype(np.int16))
    buf.write(q["ft_weight"].astype(np.int8).tobytes())
    _write_leb(buf, q["psqt"].astype(np.int32))
    if F.THREATS:
        buf.write(q["thr_weight"].astype(np.int8).tobytes())
        _write_leb(buf, q["thr_psqt"].astype(np.int32))

    ahash = arch_hash()
    for k in range(STACKS):
        buf.write(struct.pack("<I", ahash))
        buf.write(q["fc0_bias"][k].astype("<i4").tobytes())
        buf.write(q["fc0_weight"][k].astype(np.int8).tobytes())
        buf.write(q["fc1_bias"][k].astype("<i4").tobytes())
        buf.write(q["fc1_weight"][k].astype(np.int8).tobytes())
        buf.write(q["fc2_bias"][k:k + 1].astype("<i4").tobytes())
        buf.write(q["fc2_weight"][k].astype(np.int8).tobytes())

    import zstandard
    with open(path, "wb") as f:
        f.write(zstandard.ZstdCompressor(level=12, threads=-1).compress(buf.getvalue()))


def read_nnue(path: str) -> dict:
    import zstandard
    raw = open(path, "rb").read()
    try:
        raw = zstandard.ZstdDecompressor().decompressobj().decompress(raw)
    except zstandard.ZstdError:
        pass  # not compressed
    f = io.BytesIO(raw)
    version, nhash, dlen = struct.unpack("<III", f.read(12))
    if version not in (VERSION, OLD_VERSION) or nhash != network_hash():
        known = {S_hash: name for name, S_hash in _arch_network_hashes().items()}
        hint = f" (looks like arch '{known[nhash]}')" if nhash in known else ""
        raise ValueError(f"incompatible network for arch '{F.ARCH}' "
                         f"(version {version:#x}, hash {nhash:#x}){hint}")
    description = f.read(dlen).decode("utf-8", "replace")
    (fth,) = struct.unpack("<I", f.read(4))
    if fth != ft_hash():
        raise ValueError("feature transformer hash mismatch")
    L1 = F.L1
    q = {"description": description, "version": version}
    q["ft_bias"] = _read_leb(f, L1)
    if version == OLD_VERSION:
        q["ft_weight"] = _read_leb(f, F.NUM_FEATURES * L1, np.int16).reshape(F.NUM_FEATURES, L1)
    else:
        n = F.NUM_FEATURES * L1
        q["ft_weight"] = np.frombuffer(f.read(n), dtype=np.int8).reshape(F.NUM_FEATURES, L1).copy()
    q["psqt"] = _read_leb(f, F.NUM_FEATURES * PSQT).reshape(F.NUM_FEATURES, PSQT)
    if F.THREATS:
        n = F.THREAT_DIMS * L1
        q["thr_weight"] = np.frombuffer(f.read(n), dtype=np.int8).reshape(F.THREAT_DIMS, L1).copy()
        q["thr_psqt"] = _read_leb(f, F.THREAT_DIMS * PSQT).reshape(F.THREAT_DIMS, PSQT)

    def arr(dtype, n):
        return np.frombuffer(f.read(np.dtype(dtype).itemsize * n), dtype=dtype).astype(np.int64)

    keys = ("fc0_bias", "fc0_weight", "fc1_bias", "fc1_weight", "fc2_bias", "fc2_weight")
    parts = {k: [] for k in keys}
    d = stack_dims()
    for _ in range(STACKS):
        (ah,) = struct.unpack("<I", f.read(4))
        if ah != arch_hash():
            raise ValueError("architecture hash mismatch")
        parts["fc0_bias"].append(arr("<i4", d["fc0_out"]))
        parts["fc0_weight"].append(arr(np.int8, d["fc0_out"] * L1).reshape(d["fc0_out"], L1))
        parts["fc1_bias"].append(arr("<i4", L3))
        parts["fc1_weight"].append(arr(np.int8, L3 * d["fc1_file"]).reshape(L3, d["fc1_file"]))
        parts["fc2_bias"].append(arr("<i4", 1)[0])
        parts["fc2_weight"].append(arr(np.int8, d["fc2_in"]))
    for k in keys:
        q[k] = np.array(parts[k])
    return q


def _arch_network_hashes() -> dict:
    out = {}
    for name, (_, h, _, threats, layers) in F.ARCHS.items():
        l1 = 1024 if threats else 2048
        out[name] = (h ^ (F.THREAT_HASH if threats else 0) ^ (l1 * 2)) ^ arch_hash(l1, layers)
    return out


def detect_arch(path: str) -> str:
    """Architecture of a .nnue file, from its hash."""
    import zstandard
    raw = open(path, "rb").read()
    try:
        raw = zstandard.ZstdDecompressor().decompressobj().decompress(raw)
    except zstandard.ZstdError:
        pass
    _, nhash, _ = struct.unpack("<III", raw[:12])
    for name, h in _arch_network_hashes().items():
        if h == nhash:
            return name
    raise ValueError(f"unknown network hash {nhash:#x}")


def upgrade(q: dict) -> tuple[int, int]:
    """Converts a network read from an old file (16-bit HalfKA weights) to
    the current format, in place: weights beyond +-127 and biases beyond
    +-MAX_FT_BIAS are clamped. Returns the numbers of clamped weights and
    biases (0, 0: the engine evaluates the new file exactly like the old one)."""
    w, b = q["ft_weight"].astype(np.int64), q["ft_bias"].astype(np.int64)
    clamped = (int(np.count_nonzero(np.abs(w) > 127)),
               int(np.count_nonzero(np.abs(b) > MAX_FT_BIAS)))
    q["ft_weight"] = np.clip(w, -127, 127).astype(np.int8)
    q["ft_bias"] = np.clip(b, -MAX_FT_BIAS, MAX_FT_BIAS)
    q["version"] = VERSION
    return clamped


def load_into_model(q: dict, model, ft_only: bool = False):
    """Dequantize a network into a model (to continue training from it). The
    training-only factorized parts, if any, are set to zero. ft_only: only the
    feature transformer (from a network with the same input features but the
    other layer stacks); the layer stacks of the model are kept."""
    import torch
    from model import (FT_SCALE, HIDDEN_WEIGHT_SCALE, QUANTIZED_ONE, OUTPUT_WEIGHT_SCALE,
                       NNUE2SCORE, OUTPUT_SCALE)
    hb = HIDDEN_WEIGHT_SCALE * QUANTIZED_ONE
    L1 = F.L1
    with torch.no_grad():
        model.ft.weight[:F.NUM_FEATURES, :L1] = torch.from_numpy(q["ft_weight"]).float() / FT_SCALE
        model.ft.weight[:F.NUM_FEATURES, L1:] = torch.tensor(q["psqt"] / (NNUE2SCORE * OUTPUT_SCALE))
        model.ft.weight[F.PAD_INDEX].zero_()
        model.ft_bias[:L1] = torch.tensor(q["ft_bias"] / FT_SCALE)
        model.ft_bias[L1:] = 0
        if F.THREATS:
            model.ft_thr.weight[:, :L1] = torch.from_numpy(q["thr_weight"].astype(np.float32)) / FT_SCALE
            model.ft_thr.weight[:, L1:] = torch.tensor(q["thr_psqt"] / (NNUE2SCORE * OUTPUT_SCALE))
        for name in ("ft_ps", "ft_kb"):
            if getattr(model, name, None) is not None:
                getattr(model, name).weight.zero_()
        if ft_only:
            return
        st = model.stacks
        st.l1.weight[:] = torch.tensor(q["fc0_weight"].reshape(-1, L1) / HIDDEN_WEIGHT_SCALE)
        st.l1.bias[:] = torch.tensor(q["fc0_bias"].reshape(-1) / hb)
        n = stack_dims()["fc1_in"]
        st.l2.weight[:] = torch.tensor(q["fc1_weight"][:, :, :n].reshape(-1, n)
                                       / HIDDEN_WEIGHT_SCALE)
        st.l2.bias[:] = torch.tensor(q["fc1_bias"].reshape(-1) / hb)
        st.output.weight[:] = torch.tensor(q["fc2_weight"] / OUTPUT_WEIGHT_SCALE)
        st.output.bias[:] = torch.tensor(q["fc2_bias"] / (NNUE2SCORE * OUTPUT_SCALE))
        if getattr(st, "l1_fact", None) is not None:
            st.l1_fact.weight.zero_()


# ---------------------------------------------------------------- integer reference

def _tdiv(a: int, b: int) -> int:
    """C++ integer division (truncates toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def quantized_eval(q: dict, fen: str) -> tuple[int, int, int]:
    """Bit-exact integer evaluation, like Network::evaluate() in the engine.
    Returns (psqt, positional, bucket) in internal units."""
    L1 = F.L1
    stm_f, nstm_f, bucket, _ = F.fen_to_features(fen)
    thr = F.fen_to_threats(fen) if F.THREATS else ([], [])
    transformed = []
    psqt_acc = []
    for feats, thr_feats in zip((stm_f, nstm_f), thr):
        acc = q["ft_bias"].astype(np.int64)         # half units, like the engine
        ps = np.zeros(PSQT, dtype=np.int64)
        for i in feats:
            acc += q["ft_weight"][i].astype(np.int64)
            ps += q["psqt"][i]
        for i in thr_feats:
            acc += q["thr_weight"][i].astype(np.int64)
            ps += q["thr_psqt"][i]
        acc *= 2                                     # doubled in transform()
        a = np.clip(acc[:L1 // 2], 0, 254)
        b = np.clip(acc[L1 // 2:], 0, 254)
        transformed.append((a * b) // 512)
        psqt_acc.append(ps[bucket])
    x = np.concatenate(transformed)                  # uint8 in the engine

    def sqr_crelu(v):   # SqrClippedReLU and ClippedReLU, concatenated
        return np.concatenate([np.minimum(127, (v * v) >> 19), np.clip(v >> 6, 0, 127)])

    d = stack_dims()
    fc0 = q["fc0_bias"][bucket] + q["fc0_weight"][bucket] @ x
    h = np.zeros(d["fc1_file"], dtype=np.int64)
    h[:d["fc1_in"]] = sqr_crelu(fc0[:d["act0"]])
    fc1 = q["fc1_bias"][bucket] + q["fc1_weight"][bucket] @ h
    if F.LAYERS == 2:
        h2 = np.concatenate([h, sqr_crelu(fc1)])
        direct = int(fc0[L2_V2 - 2]) - int(fc0[L2_V2 - 1])
    else:
        h2 = np.clip(fc1 >> 6, 0, 127)
        direct = int(fc0[L2])
    fc2 = int(q["fc2_bias"][bucket] + q["fc2_weight"][bucket] @ h2)
    fwd = _tdiv(direct * (600 * 16), 127 * 64)
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
    ap.add_argument("--upgrade", action="store_true",
                    help="convert a .nnue file of the old version (16-bit HalfKA weights)")
    args = ap.parse_args()

    if args.upgrade:
        F.set_arch(detect_arch(args.source))
        q = read_nnue(args.source)
        if q["version"] == VERSION:
            raise SystemExit(f"{args.source} already has the current version")
        nw, nb = upgrade(q)
        write_nnue(q, args.target, q["description"])
        total = q["ft_weight"].size
        print(f"written {args.target} (arch {F.ARCH}): {nw} of {total} weights "
              f"({100 * nw / total:.4f}%) and {nb} biases clamped"
              + ("" if nw or nb else ", same evaluation as the old file"))
        return

    import torch
    from model import NNUE

    if args.to_checkpoint:
        F.set_arch(detect_arch(args.source))
        model = NNUE(init_psqt=False)
        load_into_model(read_nnue(args.source), model)
        torch.save({"model": model.state_dict(), "arch": F.ARCH}, args.target)
    else:
        ckpt = torch.load(args.source, map_location="cpu")
        F.set_arch(ckpt.get("arch", "default"))
        model = NNUE.from_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        write_nnue(quantize(model), args.target, args.description)
    print(f"written {args.target} (arch {F.ARCH})")


if __name__ == "__main__":
    main()
