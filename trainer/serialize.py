"""Export a trained model to the engine's .nnue format, and read one back.

File layout (maskfish/src/nnue/network.cpp, nnue_feature_transformer.h, layers/):

    u32 version, u32 network hash, u32 len, description
    u32 feature transformer hash
      LEB128 int16 biases[L1]             (scale 127, |bias| <= 16383)
      raw    int8  weights[7560][L1]      (scale 127)
      LEB128 int32 psqt[7560][16]         (scale 600 * 16, not with "*-nopsqt")
      threat architectures only:
      raw    int8  threat weights[92711][L1] (scale 127)
      LEB128 int32 threat psqt[92711][16]    (scale 600 * 16, not with "*-nopsqt")
    The engine doubles the feature transformer outputs (biases + weights),
    so 1.0 is 254 before the clipped ReLU.
    context architectures only ("*-ctx", make ctx=yes):
    u32 context tower hash (0x0C7A16C7)
      int32 bias[16] (scale 64*127), int8 weight[16][16] (scale 64)
    16 x layer stack (32 for "*-s32", make stacks=32):
      u32 architecture hash
      fc0: int32 bias[16]  (scale 64*127), int8 weight[16][L1] (scale 64)
           "*-ctx" only: int8 context weight[16][16]  (scale 64)
      fc1: int32 bias[32]  (scale 64*127), int8 weight[32][32]   (30 used)
      fc2: int32 bias[1]   (scale 600*16), int8 weight[1][32]    (scale 600*16/127)
    or, layer stacks 2 (architectures "*-l2", make layers=2):
      fc0: int32 bias[32], int8 weight[32][L1] ("*-ctx": int8 context weight[32][16])
      fc1: int32 bias[32], int8 weight[32][64]
      fc2: int32 bias[1],  int8 weight[1][128]   (same scales)

L1 = 2048, or 1024 for the threat architectures. (7560 is 30240 for "att".)

The "*-q2" architectures (make quant=2) have the same layout with the scales
of quant.py: feature transformer weights and biases at scale 255 (not doubled
by the engine), fc0 weights at 128 (biases 128*128), fc1 at 64 (biases
128*64), fc2 weights at 128 (bias 128*128), fc0 context weights at 128*128/127;
the layer stacks hash is xored with QUANT2_ARCH_HASH.

The whole file is then compressed with zstd, which the engine requires.

Files of version 0x7AF32F20 (before the engine had 8-bit HalfKA weights)
have LEB128 int16 weights[7560][L1] instead; --upgrade converts them (the
evaluation is unchanged if no weight is beyond +-127, which is reported).

Usage:
    python serialize.py checkpoint.pt out.nnue [--description "..."]
    python serialize.py --convert default-meta-ctx in.nnue out.nnue (add architecture options)
    python serialize.py --convert default-q2 in.nnue out.nnue  (requantize, see quant.py)
    python serialize.py --to-checkpoint in.nnue out.pt     (start from a net)
    python serialize.py --upgrade old.nnue new.nnue        (old file version)
"""

from __future__ import annotations

import argparse
import io
import struct

import numpy as np

import jieqi_features as F
import quant as Q

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


CTX_ARCH_HASH = 0x0C7A0000    # engine: ContextArchHash, xored into the layer stacks hash
BLEND_ARCH_HASH = 0x0B1E0000  # engine: BlendArchHash (make lsblend=yes)
CTX_TOWER_HASH = 0x0C7A16C7   # engine: ContextTower::get_hash_value()
QUANT2_ARCH_HASH = 0x0A2C0000  # engine: Quant2ArchHash (make quant=2)
STACKS32_ARCH_HASH = 0x5A320000  # engine: Stacks32ArchHash (make stacks=32)


def arch_hash(l1: int | None = None, layers: int | None = None, ctx: bool | None = None,
              blend: bool | None = None, quant2: bool | None = None,
              stacks: int | None = None) -> int:
    """Layer stacks hash, computed like NetworkArchitecture::get_hash_value()
    (0x63336B4A for L1 = 2048, 0x63336A4A for 1024; xored with CTX_ARCH_HASH
    with the context inputs)."""
    m = 0xFFFFFFFF
    l1 = F.L1 if l1 is None else l1
    ctx = F.CTX if ctx is None else ctx
    blend = F.BLEND if blend is None else blend
    quant2 = F.QUANT2 if quant2 is None else quant2
    stacks = F.NUM_LAYER_STACKS if stacks is None else stacks

    def affine(prev, outputs):
        h = (0xCC03DAE4 + outputs) & m
        return h ^ (prev >> 1) ^ ((prev << 31) & m)

    def crelu(prev):
        return (0x538D24C7 + prev) & m

    h = 0xEC42E90D ^ (l1 * 2)
    h = crelu(affine(h, stack_dims(layers)["fc0_out"]))
    h = crelu(affine(h, L3))
    return affine(h, 1) ^ (CTX_ARCH_HASH if ctx else 0) ^ (BLEND_ARCH_HASH if blend else 0) \
        ^ (QUANT2_ARCH_HASH if quant2 else 0) ^ (STACKS32_ARCH_HASH if stacks == 32 else 0)


NO_PSQT_HASH = 0x5E0D0000     # engine: FeatureTransformer::get_hash_value(), make psqt=no


def ft_hash(arch: F.Arch | None = None) -> int:
    """Feature transformer hash: feature set hash(es) ^ (L1 * 2), and
    NO_PSQT_HASH without PSQT."""
    a = arch or F.ARCHS[F.ARCH]
    l1 = 1024 if a.threats else 2048
    h = a.feature_hash ^ (F.META_HASH if a.meta else 0) ^ (F.THREAT_HASH if a.threats else 0)
    return h ^ (l1 * 2) ^ (0 if a.psqt else NO_PSQT_HASH)


def network_hash() -> int:
    return ft_hash() ^ arch_hash()


ARCH_HASH = arch_hash(2048)   # kept for compatibility (default architecture)
PSQT = F.NUM_PSQT_BUCKETS
# (the number of layer stacks is F.NUM_LAYER_STACKS: 16, or 32 with "-s32")
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
    """Signed LEB128, like write_leb_128() in maskfish/src/nnue/nnue_common.h
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
    sc = Q.scales()
    FT_SCALE = sc.ft

    L1 = F.L1
    model = model.coalesced()
    with torch.no_grad():
        # Feature transformer: done with torch in float32 to keep memory low
        # (the "att" architecture has 30240 x 2048 weights)
        w = model.ft.weight.detach()[:F.NUM_FEATURES]
        b = model.ft_bias.detach().cpu().double().numpy()

        q = {"quant2": F.QUANT2}
        q["ft_bias"] = np.round(b[:L1] * FT_SCALE).astype(np.int64)
        ftw = torch.round(w[:, :L1].float() * FT_SCALE)
        if ftw.abs().max().item() > 127:
            raise ValueError("feature transformer weights out of the int8 range "
                             "(model.clip_weights() keeps them inside)")
        if np.abs(q["ft_bias"]).max() > MAX_FT_BIAS:
            raise ValueError("feature transformer biases out of range")
        q["ft_weight"] = ftw.to(torch.int8).cpu().numpy()
        del ftw
        if F.PSQT:
            q["psqt"] = np.round(w[:, L1:].cpu().double().numpy()
                                 * Q.OUTPUT_ONE).astype(np.int64)
        else:
            q["psqt"] = np.zeros((F.NUM_FEATURES, PSQT), dtype=np.int64)
        if F.THREATS:
            tw = model.ft_thr.weight.detach()
            tq = torch.round(tw[:, :L1].float() * FT_SCALE)
            if tq.abs().max().item() > 127:
                raise ValueError("threat weights out of the int8 range")
            q["thr_weight"] = tq.to(torch.int8).cpu().numpy()
            del tq
            q["thr_psqt"] = (np.round(tw[:, L1:].cpu().double().numpy()
                                      * Q.OUTPUT_ONE).astype(np.int64) if F.PSQT
                             else np.zeros((F.THREAT_DIMS, PSQT), dtype=np.int64))

        d = stack_dims()
        st = model.stacks
        STACKS = F.NUM_LAYER_STACKS
        l1w = st.l1.weight.detach().cpu().double().numpy().reshape(STACKS, d["fc0_out"], L1)
        l1b = st.l1.bias.detach().cpu().double().numpy().reshape(STACKS, d["fc0_out"])
        l2w = st.l2.weight.detach().cpu().double().numpy().reshape(STACKS, L3, d["fc1_in"])
        l2b = st.l2.bias.detach().cpu().double().numpy().reshape(STACKS, L3)
        ow = st.output.weight.detach().cpu().double().numpy().reshape(STACKS, d["fc2_in"])
        ob = st.output.bias.detach().cpu().double().numpy().reshape(STACKS)
        if F.CTX:
            cw = model.ctx_tower.weight.detach().cpu().double().numpy()
            cb = model.ctx_tower.bias.detach().cpu().double().numpy()
            l1c = st.l1_ctx.weight.detach().cpu().double().numpy().reshape(
                STACKS, d["fc0_out"], F.CONTEXT_INPUTS)

    def qw(w, scale):       # weights clipped to the int8 range, rounded
        m = sc.max_weight(scale)
        return np.round(np.clip(w, -m, m) * scale).astype(np.int64)

    q["fc0_bias"] = np.round(l1b * sc.one0).astype(np.int64)
    q["fc0_weight"] = qw(l1w, sc.w0)
    q["fc1_bias"] = np.round(l2b * sc.one1).astype(np.int64)
    fc1 = np.zeros((F.NUM_LAYER_STACKS, L3, d["fc1_file"]), dtype=np.int64)
    fc1[:, :, :d["fc1_in"]] = qw(l2w, sc.w1)
    q["fc1_weight"] = fc1
    q["fc2_bias"] = np.round(ob * sc.out2).astype(np.int64)
    q["fc2_weight"] = qw(ow, sc.w2)
    if F.CTX:
        q["ctx_bias"] = np.round(cb * Q.CTX_ONE).astype(np.int64)
        q["ctx_weight"] = qw(cw, Q.CTX_W)
        q["fc0_ctx"] = qw(l1c, sc.w0_ctx)
    for k in ("fc0_weight", "fc1_weight", "fc2_weight") + (("ctx_weight", "fc0_ctx") if F.CTX else ()):
        assert np.abs(q[k]).max() <= 127, k
    return q


def write_nnue(q: dict, path: str, description: str = "Jieqi NNUE trained with the Maskfish trainer"):
    buf = io.BytesIO()
    desc = description.encode("utf-8")
    buf.write(struct.pack("<III", VERSION, network_hash(), len(desc)))
    buf.write(desc)

    buf.write(struct.pack("<I", ft_hash()))
    if np.abs(q["ft_weight"]).max() > 127 or np.abs(q["ft_bias"]).max() > MAX_FT_BIAS:
        raise ValueError("feature transformer parameters out of range")
    _write_leb(buf, q["ft_bias"].astype(np.int16))
    buf.write(q["ft_weight"].astype(np.int8).tobytes())
    if F.PSQT:
        _write_leb(buf, q["psqt"].astype(np.int32))
    if F.THREATS:
        buf.write(q["thr_weight"].astype(np.int8).tobytes())
        if F.PSQT:
            _write_leb(buf, q["thr_psqt"].astype(np.int32))

    if F.CTX:
        buf.write(struct.pack("<I", CTX_TOWER_HASH))
        buf.write(q["ctx_bias"].astype("<i4").tobytes())
        buf.write(q["ctx_weight"].astype(np.int8).tobytes())

    ahash = arch_hash()
    for k in range(F.NUM_LAYER_STACKS):
        buf.write(struct.pack("<I", ahash))
        buf.write(q["fc0_bias"][k].astype("<i4").tobytes())
        buf.write(q["fc0_weight"][k].astype(np.int8).tobytes())
        if F.CTX:
            buf.write(q["fc0_ctx"][k].astype(np.int8).tobytes())
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
    q = {"description": description, "version": version, "psqt_present": F.PSQT,
         "quant2": F.QUANT2}
    q["ft_bias"] = _read_leb(f, L1)
    if version == OLD_VERSION:
        q["ft_weight"] = _read_leb(f, F.NUM_FEATURES * L1, np.int16).reshape(F.NUM_FEATURES, L1)
    else:
        n = F.NUM_FEATURES * L1
        q["ft_weight"] = np.frombuffer(f.read(n), dtype=np.int8).reshape(F.NUM_FEATURES, L1).copy()
    q["psqt"] = (_read_leb(f, F.NUM_FEATURES * PSQT).reshape(F.NUM_FEATURES, PSQT) if F.PSQT
                 else np.zeros((F.NUM_FEATURES, PSQT), dtype=np.int64))
    if F.THREATS:
        n = F.THREAT_DIMS * L1
        q["thr_weight"] = np.frombuffer(f.read(n), dtype=np.int8).reshape(F.THREAT_DIMS, L1).copy()
        q["thr_psqt"] = (_read_leb(f, F.THREAT_DIMS * PSQT).reshape(F.THREAT_DIMS, PSQT) if F.PSQT
                         else np.zeros((F.THREAT_DIMS, PSQT), dtype=np.int64))

    def arr(dtype, n):
        return np.frombuffer(f.read(np.dtype(dtype).itemsize * n), dtype=dtype).astype(np.int64)

    n_ctx = F.CONTEXT_INPUTS
    if F.CTX:
        (ch,) = struct.unpack("<I", f.read(4))
        if ch != CTX_TOWER_HASH:
            raise ValueError("context tower hash mismatch")
        q["ctx_bias"] = arr("<i4", n_ctx)
        q["ctx_weight"] = arr(np.int8, n_ctx * n_ctx).reshape(n_ctx, n_ctx)

    keys = ("fc0_bias", "fc0_weight", "fc1_bias", "fc1_weight", "fc2_bias", "fc2_weight") \
        + (("fc0_ctx",) if F.CTX else ())
    parts = {k: [] for k in keys}
    d = stack_dims()
    for _ in range(F.NUM_LAYER_STACKS):
        (ah,) = struct.unpack("<I", f.read(4))
        if ah != arch_hash():
            raise ValueError("architecture hash mismatch")
        parts["fc0_bias"].append(arr("<i4", d["fc0_out"]))
        parts["fc0_weight"].append(arr(np.int8, d["fc0_out"] * L1).reshape(d["fc0_out"], L1))
        if F.CTX:
            parts["fc0_ctx"].append(arr(np.int8, d["fc0_out"] * n_ctx).reshape(d["fc0_out"], n_ctx))
        parts["fc1_bias"].append(arr("<i4", L3))
        parts["fc1_weight"].append(arr(np.int8, L3 * d["fc1_file"]).reshape(L3, d["fc1_file"]))
        parts["fc2_bias"].append(arr("<i4", 1)[0])
        parts["fc2_weight"].append(arr(np.int8, d["fc2_in"]))
    for k in keys:
        q[k] = np.array(parts[k])
    return q


def _arch_network_hashes() -> dict:
    out = {}
    for name, a in F.ARCHS.items():
        out[name] = ft_hash(a) ^ arch_hash(1024 if a.threats else 2048, a.layers, a.ctx, a.blend,
                                           a.quant2, a.stacks)
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


def convertible(src: str, dst: str, requantize: bool = False) -> str | None:
    """None if a network of architecture src can be converted into dst by
    convert(), otherwise the reason. The conversions add options: "-meta" and
    "-ctx" (zero weights: the same evaluation), "-blend" (the layer stacks are
    kept, the evaluation changes since they are selected differently), "-s32"
    (each layer stack copied into the two that split it: the same evaluation).
    requantize: the quantizations ("-q2") may differ too: the network keeps
    its integers and its "quant2" key, load_into_model() dequantizes it with
    its own scales (train.py --init-net); not for writing a file."""
    sb, sf = F.split_arch(src)
    db, df = F.split_arch(dst)
    if sb != db:
        return f"the feature sets differ ({sb} / {db})"
    if sf["quant2"] != df["quant2"] and not requantize:
        return "the quantizations differ (not convertible: train with --init-net)"
    for field in ("layers", "psqt"):
        if sf[field] != df[field]:
            return f"the networks differ by {field} (not convertible: train with --init-net)"
    for field in ("meta", "ctx", "blend"):
        if sf[field] and not df[field]:
            return f"{field} cannot be removed"
    if sf["stacks"] > df["stacks"]:
        return "layer stacks cannot be removed"
    return None


def convert(q: dict, src: str, dst: str, requantize: bool = False) -> dict:
    """Network q of architecture src as a network of architecture dst (see
    convertible()). F's architecture is left at dst."""
    reason = convertible(src, dst, requantize)
    if reason:
        raise ValueError(f"cannot convert {src} into {dst}: {reason}")
    sf, df = F.split_arch(src)[1], F.split_arch(dst)[1]
    q = dict(q)
    F.set_arch(src)
    src_ps, buckets = F.PS_NB, F.NUM_FEATURES // F.PS_NB
    F.set_arch(dst)
    if df["meta"] and not sf["meta"]:
        # zero rows for the meta features at the end of every bucket
        def expand(a):
            a = np.asarray(a).reshape(buckets, src_ps, -1)
            z = np.zeros((buckets, F.PS_NB - src_ps, a.shape[2]), dtype=a.dtype)
            return np.concatenate([a, z], axis=1).reshape(buckets * F.PS_NB, -1)
        q["ft_weight"] = expand(q["ft_weight"])
        q["psqt"] = expand(q["psqt"])
    if df["stacks"] != sf["stacks"]:
        # "-s32": the layer stacks 2b and 2b + 1 split the bucket b of the
        # piece count and share its PSQT bucket: both are copies of the stack b
        k = df["stacks"] // sf["stacks"]
        for key in ("fc0_bias", "fc0_weight", "fc1_bias", "fc1_weight", "fc2_bias", "fc2_weight",
                    "fc0_ctx"):
            if key in q:
                q[key] = np.repeat(np.asarray(q[key]), k, axis=0)
    if df["ctx"] and not sf["ctx"]:
        n = F.CONTEXT_INPUTS
        q["ctx_bias"] = np.zeros(n, dtype=np.int64)
        q["ctx_weight"] = np.zeros((n, n), dtype=np.int64)
        q["fc0_ctx"] = np.zeros((F.NUM_LAYER_STACKS, stack_dims()["fc0_out"], n), dtype=np.int64)
    return q


def load_into_model(q: dict, model, ft_only: bool = False):
    """Dequantize a network into a model (to continue training from it). The
    training-only factorized parts, if any, are set to zero. ft_only: only the
    feature transformer (from a network with the same input features but the
    other layer stacks); the layer stacks of the model are kept."""
    import torch
    # the scales of the network q (its quantization may differ from the
    # model's: the model clips the weights into its own ranges when it trains)
    sc = Q.scales(q.get("quant2", False))
    FT_SCALE, PSQT_SCALE = sc.ft, Q.OUTPUT_ONE
    L1 = F.L1
    with torch.no_grad():
        model.ft.weight[:F.NUM_FEATURES, :L1] = torch.from_numpy(q["ft_weight"]).float() / FT_SCALE
        # (a network without PSQT leaves the model's initial PSQT weights)
        if model.psqt_buckets and q.get("psqt_present", True):
            model.ft.weight[:F.NUM_FEATURES, L1:] = torch.tensor(q["psqt"] / PSQT_SCALE)
        model.ft.weight[F.PAD_INDEX].zero_()
        model.ft_bias[:L1] = torch.tensor(q["ft_bias"] / FT_SCALE)
        model.ft_bias[L1:] = 0
        if F.THREATS:
            model.ft_thr.weight[:, :L1] = torch.from_numpy(q["thr_weight"].astype(np.float32)) / FT_SCALE
            if model.psqt_buckets and q.get("psqt_present", True):
                model.ft_thr.weight[:, L1:] = torch.tensor(q["thr_psqt"] / PSQT_SCALE)
        for name in ("ft_ps", "ft_kb"):
            if getattr(model, name, None) is not None:
                getattr(model, name).weight.zero_()
        if ft_only:
            return
        if model.ctx_tower is not None and "ctx_weight" in q:
            model.ctx_tower.weight[:] = torch.tensor(q["ctx_weight"] / Q.CTX_W)
            model.ctx_tower.bias[:] = torch.tensor(q["ctx_bias"] / Q.CTX_ONE)
            model.stacks.l1_ctx.weight[:] = torch.tensor(
                q["fc0_ctx"].reshape(-1, F.CONTEXT_INPUTS) / sc.w0_ctx)
        st = model.stacks
        st.l1.weight[:] = torch.tensor(q["fc0_weight"].reshape(-1, L1) / sc.w0)
        st.l1.bias[:] = torch.tensor(q["fc0_bias"].reshape(-1) / sc.one0)
        n = stack_dims()["fc1_in"]
        st.l2.weight[:] = torch.tensor(q["fc1_weight"][:, :, :n].reshape(-1, n) / sc.w1)
        st.l2.bias[:] = torch.tensor(q["fc1_bias"].reshape(-1) / sc.one1)
        st.output.weight[:] = torch.tensor(q["fc2_weight"] / sc.w2)
        st.output.bias[:] = torch.tensor(q["fc2_bias"] / sc.out2)
        if getattr(st, "l1_fact", None) is not None:
            st.l1_fact.weight.zero_()


# ---------------------------------------------------------------- integer reference

def _tdiv(a: int, b: int) -> int:
    """C++ integer division (truncates toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def _stack_eval(q: dict, bucket: int, x: np.ndarray, ctx_out) -> int:
    """Output of a layer stack (OutputScale units), like NetworkArchitecture::propagate()."""
    sc = Q.scales()

    def sqr_crelu(v, shift):   # SqrClippedReLU and ClippedReLU, concatenated
        return np.concatenate([np.minimum(127, (v * v) >> (2 * shift + 7)),
                               np.clip(v >> shift, 0, 127)])

    d = stack_dims()
    fc0 = q["fc0_bias"][bucket] + q["fc0_weight"][bucket] @ x
    if ctx_out is not None:
        fc0 = fc0 + q["fc0_ctx"][bucket] @ ctx_out
    h = np.zeros(d["fc1_file"], dtype=np.int64)
    h[:d["fc1_in"]] = sqr_crelu(fc0[:d["act0"]], sc.shift0)
    fc1 = q["fc1_bias"][bucket] + q["fc1_weight"][bucket] @ h
    if F.LAYERS == 2:
        h2 = np.concatenate([h, sqr_crelu(fc1, sc.shift1)])
        direct = int(fc0[L2_V2 - 2]) - int(fc0[L2_V2 - 1])
    else:
        h2 = np.clip(fc1 >> sc.shift1, 0, 127)
        direct = int(fc0[L2])
    fc2 = int(q["fc2_bias"][bucket] + q["fc2_weight"][bucket] @ h2)
    one0 = int(sc.one0)
    if sc.quant2:          # fc2 in fc0 units, the sum converted
        return _tdiv((fc2 + direct) * (600 * 16), one0)
    return fc2 + _tdiv(direct * (600 * 16), one0)


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
        sc = Q.scales()
        acc *= sc.ft_mult                            # doubled in transform() (quant=1)
        a = np.clip(acc[:L1 // 2], 0, sc.ft_clip)
        b = np.clip(acc[L1 // 2:], 0, sc.ft_clip)
        transformed.append((a * b) // 512)
        psqt_acc.append(ps[F.psqt_bucket(bucket)])
    x = np.concatenate(transformed)                  # uint8 in the engine

    ctx_out = None
    if F.CTX:
        ci = np.array(F.context_inputs(F.parse_fen(fen)), dtype=np.int64)
        ctx_out = np.clip((q["ctx_bias"] + q["ctx_weight"] @ ci) >> 6, 0, 127)
    positional = _stack_eval(q, bucket, x, ctx_out)
    if F.BLEND:
        blend = F.layer_stack_selection(F.parse_fen(fen))[1]
        if blend:
            nxt = _stack_eval(q, bucket + 1, x, ctx_out)
            positional = _tdiv(positional * (256 - blend) + nxt * blend, 256)
    psqt = _tdiv(int(psqt_acc[0] - psqt_acc[1]), 2)
    return _tdiv(psqt, 16), _tdiv(positional, 16), bucket


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("target")
    ap.add_argument("--description", default="Jieqi NNUE trained with the Maskfish trainer")
    ap.add_argument("--to-checkpoint", action="store_true",
                    help="convert a .nnue file into a checkpoint to continue training")
    ap.add_argument("--upgrade", action="store_true",
                    help="convert a .nnue file of the old version (16-bit HalfKA weights)")
    ap.add_argument("--convert", metavar="ARCH",
                    help="convert a .nnue file into architecture ARCH, adding -meta / -ctx / "
                         "-s32 (same evaluation) or -blend (layer stacks kept)")
    args = ap.parse_args()

    if args.convert:
        src = detect_arch(args.source)
        F.set_arch(src)
        q = read_nnue(args.source)
        try:
            q = convert(q, src, args.convert, requantize=True)
        except ValueError as e:
            raise SystemExit(str(e))
        note = ""
        if q["quant2"] != F.QUANT2:
            # another quantization ("-q2"): dequantized into a model of the
            # target architecture, clipped into its ranges, quantized again
            from model import NNUE
            model = NNUE(init_psqt=False)
            load_into_model(q, model)
            sc = Q.scales()
            w = model.ft.weight[:F.NUM_FEATURES, :F.L1]
            out = int((w.abs() > sc.max_ft_weight + 1e-6).sum())
            model.clip_weights()
            q = {**quantize(model), "description": q["description"]}
            note = (f", requantized: {out} feature transformer weights clipped to "
                    f"+-{sc.max_ft_weight:.3f}, the evaluation changes slightly")
        write_nnue(q, args.target, q["description"])
        same = not (F.split_arch(args.convert)[1]["blend"] and not F.split_arch(src)[1]["blend"])
        print(f"written {args.target} (arch {src} -> {F.ARCH}"
              + (", same evaluation" if same and not note else
                 note if note else ", layer stacks selected differently") + ")")
        return

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
