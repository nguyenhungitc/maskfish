#!/usr/bin/env python3
"""Trainer of the engine's network (HNN, src/hnn/hnnue.h) from TrainRecords (112-byte self-play records, records.py).

The net: king-bucketed piece-square features + 30 urn thresholds (no face-down variant doubling), the head
(pairwise FT product, 16 layer stacks by piece count x face-down pieces, W -> 16 -> [sqr, crelu] -> 32 -> 1 + skip,
16-bucket PSQT, post-processing). Exported as a raw JQHNNUE1 file (best.hnn, last.hnn) that EvalFile takes as it is;
tools/hnn/jqnz_tool z packs it into the smaller compressed .nnue.

Target in win-probability space: wt * sigmoid(teacher / K) + ws * sigmoid(score / K) + wr * (result + 1) / 2, K fitted
on the data (--k to fix it). The teacher is optional (--wt 0): an HNN net with the standard head (.hnn, or a compressed
.nnue, decoded with tools/hnn/jqnz_tool), evaluated exactly as the engine does (hnn_ref.py).

The next net of a published one: --init NET (the .nnue or .hnn file) starts from its weights exactly, with its own
width, king-bucket scheme and head (hnn_file.py); scripts/pipeline.sh does this for the first generation.

  python3 train/train.py data/gen0*.bin --val-file data/val0.bin --init nets/maskfish.nnue --wt 0 --ws 0.7 --wr 0.3 \
      --epochs 10 --out runs/gen1

Input pipeline options (non-stream mode):
  --threads N   torch CPU threads (default 4)
  --gpu-data    keep the uint8 records on the GPU (permutation still drawn on the CPU generator -> same order);
                frees the ~19 GB host copy after upload
  else          background thread gathers the next batch into pinned memory (non_blocking upload)
  --tv-cache F  teacher values cached to disk (F.npy + F.json key: data files/sizes/mtimes, teacher, filters)
  loss summed on the GPU in float64 (identical to summing loss.item()), no per-step host sync.

A new net from scratch (another width, king-bucket scheme or head), with a teacher net or without (--wt 0):

  python3 train/train.py DATA.bin ... --val-file VAL.bin --teacher nets/maskfish.nnue --width 512 --kb-scheme 3 \
      [--factor] [--l1 32 --l2 64 --sq2] --epochs 10 --out runs/hnn512s3

--kb-scheme: 0 no king buckets, 1 three by king file, 2 two (king on the centre file or not) + mirror,
3 six (rank x centre file) + mirror. --l1/--l2/--sq2: a head other than the standard L1 16, L2 32 (version-3 file).
"""
import argparse
import json
import math
import os
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import records as T  # noqa: E402
# --flip (left-right board flip augmentation) and --ema (EMA of the weights, exported as last): both off by default.
FLIP_IDX = [r * 9 + 8 - f for r in range(10) for f in range(9)]  # square s = rank*9 + file -> mirrored file


def flip_records(b, mask):
    """raw (B,112) uint8 records -> copy with the 90 board bytes mirrored left-right where mask is True. stm, urn, score,
    result, flags, ply, noCapture are mirror-invariant; the move field (109-110) is not used by the trainer."""
    fi = torch.as_tensor(FLIP_IDX, device=b.device)
    out = b.clone()
    out[:, :90] = torch.where(mask[:, None], b[:, :90].index_select(1, fi), b[:, :90])
    return out
import hnn_ref as HR  # noqa: E402  (the engine's integer inference, for an HNN teacher)
import torch.nn.functional as F  # noqa: E402
import hnn_file as HF  # noqa: E402  (net files -> float weights, for --init NET)

NPSQT, NST = 16, 16
PF = HR.PIECE_FEATS
SQR_C = 8128.0 ** 2 / 2 ** 19 / 127.0  # 0.99219: the engine's squared clipped ReLU in float units
X_C = 127.0 / 128.0                     # the pairwise product's >> 7


def bag(table, idx):
    """sum of table rows over idx (B,K) with -1 padding."""
    valid = idx >= 0
    return F.embedding_bag(idx.clamp(min=0), table, per_sample_weights=valid.to(table.dtype), mode="sum")


def ste_round(t, scale):
    """round(t * scale) / scale in the forward pass, identity gradient (quantisation-aware training)."""
    return t + (torch.round(t * scale) / scale - t).detach()


def ste_floor(t, scale):
    """floor(t * scale) / scale in the forward pass, identity gradient (models the engine's integer truncations)."""
    return t + (torch.floor(t * scale) / scale - t).detach()


class HFeaturizer:
    """raw (B,112) uint8 -> feats (B,2,66) long (perspective 0 = side to move, -1 padded), bucket (B,), post (B,)."""

    def __init__(self, dev, scheme):
        self.dev, self.scheme = dev, scheme
        kb, mir = HR.king_tables(scheme)
        self.kbt, self.mirt = torch.tensor(kb, device=dev), torch.tensor(mir, device=dev)
        self.base = HR.KB_OF_SCHEME[scheme] * PF
        uo, us, uj, uf = [], [], [], []
        for c in range(2):  # 0 = own (perspective) colour, 1 = opponent
            for sl in range(6):
                for j in range(1, HR.URN_INIT[sl] + 1):
                    uo.append(c); us.append(sl); uj.append(j); uf.append(c * 15 + HR.URN_OFF[sl] + j - 1)
        self.uo, self.us = torch.tensor(uo, device=dev), torch.tensor(us, device=dev)
        self.uj, self.uf = torch.tensor(uj, device=dev), torch.tensor(uf, device=dev)
        self.PV = torch.tensor(HR.PIECE_VAL, device=dev)

    def __call__(self, raw):
        B = len(raw)
        raw = raw.to(self.dev)
        board = raw[:, :90].long()
        typ, col = board & 15, board >> 4
        stm = raw[:, 90].long()
        urn = raw[:, 91:103].to(torch.int8).long().view(B, 2, 6)
        nocap = raw[:, 111].long()
        occ = typ > 0
        dark = ((typ == 8) & occ).sum(1)
        pieces = occ.sum(1)
        ks = torch.stack([((typ == 1) & (col == c)).long().argmax(1) for c in range(2)], 1)
        order = torch.argsort((~occ).to(torch.int8), dim=1, stable=True)[:, :36]
        t_o, c_o = typ.gather(1, order), col.gather(1, order)
        feats = []
        for k in range(2):
            p = stm if k == 0 else 1 - stm
            own = ks.gather(1, p[:, None])[:, 0]
            ko = torch.where(p == 0, own, (9 - own // 9) * 9 + own % 9)
            ki = (ko // 9) * 3 + ko % 9 - 3
            kb, mir = self.kbt[ki], self.mirt[ki]
            o = torch.where((p == 0)[:, None], order, (9 - order // 9) * 9 + order % 9)
            o = torch.where((mir == 1)[:, None], (o // 9) * 9 + 8 - o % 9, o)
            f = kb[:, None] * PF + (torch.where(c_o == p[:, None], 0, 8) + t_o - 1) * 90 + o
            f = torch.where(t_o > 0, f, -1)
            owner = torch.where(self.uo[None, :] == 0, p[:, None], 1 - p[:, None])  # (B,30) absolute owner colour
            cnt = urn.view(B, 12).gather(1, owner * 6 + self.us[None, :])
            u = torch.where(cnt >= self.uj[None, :], self.base + self.uf[None, :], -1)
            feats.append(torch.cat([f, u], 1))
        up = torch.stack([torch.stack([((typ == sl + 2) & (col == c)).sum(1) for sl in range(5)], 1) for c in range(2)], 1)
        mat = (urn * self.PV).sum((1, 2)) + (up * self.PV[:5]).sum((1, 2))
        ms = 935 + torch.div(mat * 93, 5116, rounding_mode="floor")
        post = ms.float() / 1024.0 * (218 - nocap.clamp(max=120)).float() / 176.0
        bucket = torch.clamp(torch.div(pieces - 1, 4, rounding_mode="floor"), max=7) + torch.where(dark > 0, 8, 0)
        return torch.stack(feats, 1), bucket, post


class HNN(nn.Module):
    def __init__(self, width, scheme, factor=False, l1=16, l2=32, sq2=False):
        super().__init__()
        assert l1 in (16, 32) and l2 in (32, 64)
        self.L1, self.L2, self.sq2 = l1, l2, bool(sq2)
        self.L2IN = (2 * (l1 - 1) + 31) // 32 * 32
        assert width % 128 == 0 and 128 <= width <= 1024
        self.W, self.H, self.scheme = width, width // 2, scheme
        self.nfeat = HR.nfeat(scheme)
        self.factor = factor and HR.KB_OF_SCHEME[scheme] > 1
        self.qat = False
        rows = self.nfeat + (PF if self.factor else 0)
        self.ft = nn.Parameter(torch.empty(rows, width + NPSQT))
        vmap = torch.full((self.nfeat,), -1, dtype=torch.long)
        nkb = HR.KB_OF_SCHEME[scheme]
        vmap[:nkb * PF] = torch.arange(nkb * PF) % PF
        self.register_buffer("vmap", vmap, persistent=False)
        self.ft_bias = nn.Parameter(torch.full((width,), 0.5))
        self.l1w = nn.Parameter(torch.empty(NST, l1, width))
        self.l1b = nn.Parameter(torch.zeros(NST, l1))
        self.l2w = nn.Parameter(torch.empty(NST, l2, 2 * (l1 - 1)))
        self.l2b = nn.Parameter(torch.zeros(NST, l2))
        self.ow = nn.Parameter(torch.empty(NST, l2 * (2 if sq2 else 1)))
        self.ob = nn.Parameter(torch.zeros(NST))
        with torch.no_grad():
            self.ft.uniform_(-0.05, 0.05)
            self.ft[:, width:] = 0
            val = {2: 943, 3: 438, 4: 168, 5: 190, 6: 491, 7: 101}
            for kb in range(nkb):
                for t, v in val.items():
                    for opp in (0, 1):
                        o = kb * PF + (opp * 8 + t - 1) * 90
                        self.ft[o:o + 90, width:] = -v if opp else v
            u1 = 1 / math.sqrt(width)
            self.l1w.uniform_(-u1, u1)
            self.l2w.uniform_(-1 / math.sqrt(2 * (l1 - 1)), 1 / math.sqrt(2 * (l1 - 1)))
            self.ow.uniform_(-50, 50)

    @staticmethod
    def from_state(sd):
        meta = sd["meta"].tolist() + [16, 32, 0][len(sd["meta"]) - 3:]
        m = HNN(int(meta[0]), int(meta[1]), bool(meta[2]), int(meta[3]), int(meta[4]), bool(meta[5]))
        m.load_state_dict({k: v for k, v in sd.items() if k != "meta"})
        return m

    @staticmethod
    def from_net(path):
        """The weights of a net file: raw .hnn, or a compressed .nnue (decoded with jqnz_tool). Exported again, they
        give the same file (hnn_file.py), so training continues exactly from that net."""
        magic = open(path, "rb").read(8)
        d = HF.read(jqnz_decode(path) if magic in JQNZ_MAGICS else path)
        m = HNN(d["W"], d["scheme"], False, d["L1"], d["L2"], d["sq2"])
        f32 = lambda a: torch.from_numpy(np.asarray(a, np.float32))
        with torch.no_grad():
            m.ft.copy_(f32(d["ft"]))
            m.ft_bias.copy_(f32(d["ft_bias"]))
            for k in ("l1w", "l1b", "l2w", "l2b", "ow", "ob"):
                getattr(m, k).copy_(f32(d[k]))
        return m

    def state(self):
        sd = {k: v.detach().cpu() for k, v in self.state_dict().items()}
        sd["meta"] = torch.tensor([self.W, self.scheme, int(self.factor)] + ([] if self.std_head() else [self.L1, self.L2, int(self.sq2)]))
        return sd

    def std_head(self):
        return (self.L1, self.L2, self.sq2) == (16, 32, False)

    def folded_ft(self):
        if not self.factor:
            return self.ft
        return self.ft[:self.nfeat] + self.ft[self.nfeat:][self.vmap.clamp(min=0)] * (self.vmap >= 0)[:, None]

    def forward(self, feats, bucket, post):
        B = len(feats)
        q = self.qat
        ft = self.folded_ft()
        if q:
            ft = torch.cat([ste_round(ft[:, :self.W], 127.0), ste_round(ft[:, self.W:], 16.0)], 1)
        a = bag(ft, feats.view(B * 2, -1)).view(B, 2, -1)
        bias = ste_round(self.ft_bias, 127.0) if q else self.ft_bias
        acc, ps = a[..., :self.W] + bias, a[..., self.W:]
        l1w, l1b, l2w, l2b, ow, ob = self.l1w, self.l1b, self.l2w, self.l2b, self.ow, self.ob
        if q:
            l1w, l2w = ste_round(l1w, 64.0), ste_round(l2w, 64.0)
            l1b, l2b = ste_round(l1b, 8128.0), ste_round(l2b, 8128.0)
            ow, ob = ste_round(ow, 16 / 127), ste_round(ob, 16.0)
        u, v = acc[..., :self.H].clamp(0, 1), acc[..., self.H:].clamp(0, 1)
        x = (u * v * X_C).reshape(B, self.W)
        if q:
            x = ste_floor(x, 127.0)
        f = torch.bmm(l1w[bucket], x.unsqueeze(2)).squeeze(2) + l1b[bucket]
        g = f[:, :self.L1 - 1]
        s = torch.cat([torch.clamp(SQR_C * g.clamp(-4.03, 4.03) ** 2, max=1.0), g.clamp(0, 1)], 1)
        if q:
            s = ste_floor(s, 127.0)
        h = (torch.bmm(l2w[bucket], s.unsqueeze(2)).squeeze(2) + l2b[bucket]).clamp(0, 1)
        if q:
            h = ste_floor(h, 127.0)
        if self.sq2:   # engine: sq = (h_int * h_int) >> 7  ->  float h^2 * 127/128
            h2 = h * h * X_C
            if q:
                h2 = ste_floor(h2, 127.0)
            h = torch.cat([h, h2], 1)
        ar = torch.arange(B, device=x.device)
        nn_ = (ow[bucket] * h).sum(1) + ob[bucket] + 600.0 * f[:, self.L1 - 1] + 0.5 * (ps[ar, 0, bucket] - ps[ar, 1, bucket])
        return nn_ * post

    @torch.no_grad()
    def clamp_(self):
        self.ft[:, :self.W].clamp_(-7.0, 7.0)
        self.l1w.clamp_(-127 / 64, 127 / 64)
        self.l2w.clamp_(-127 / 64, 127 / 64)
        self.ow.clamp_(-127 * 127 / 16, 127 * 127 / 16)

    @torch.no_grad()
    def export(self, path):
        """The JQHNNUE1 file of the weights (version 1 for the standard head, else 3), as src/hnn/hnnue.cpp reads it."""
        W, L1, L2, L2IN = self.W, self.L1, self.L2, self.L2IN
        w = self.folded_ft().detach()
        q = lambda t, s, lo, hi, dt: torch.clamp(torch.round(t * s), lo, hi).cpu().numpy().astype(dt)
        with open(path, "wb") as f:
            f.write(HR.MAGIC)
            if self.std_head():
                f.write(struct.pack("<8I", 1, W, self.nfeat, self.scheme, NPSQT, NST, 16, 32))
            else:
                f.write(struct.pack("<8I", 3, W, self.nfeat, self.scheme, NPSQT, NST, L1, L2))
                f.write(struct.pack("<4I", int(self.sq2), 0, 0, 0))
            f.write(q(self.ft_bias, 127, -32767, 32767, "<i2").tobytes())
            f.write(q(w[:, :W], 127, -32767, 32767, "<i2").tobytes())
            f.write(q(w[:, W:], 16, -2**31, 2**31 - 1, "<i4").tobytes())
            l2 = torch.zeros(NST, L2, L2IN, device=w.device)
            l2[:, :, :2 * (L1 - 1)] = self.l2w
            for s in range(NST):
                f.write(q(self.l1b[s], 64 * 127, -2**31, 2**31 - 1, "<i4").tobytes())
                l1 = q(self.l1w[s], 64, -127, 127, np.int8)
                f.write(l1.reshape(L1, W // 4, 4).transpose(1, 0, 2).tobytes())
                f.write(q(self.l2b[s], 64 * 127, -2**31, 2**31 - 1, "<i4").tobytes())
                f.write(q(l2[s], 64, -127, 127, np.int8).reshape(L2, L2IN // 4, 4).transpose(1, 0, 2).tobytes())
                f.write(q(self.ob[s:s + 1], 16, -2**31, 2**31 - 1, "<i4").tobytes())
                f.write(q(self.ow[s], 16 / 127, -127, 127, np.int8).tobytes())


def split_path(p):
    """PATH or PATH@N (only the first N records, e.g. of a file that is still growing)."""
    if "@" in os.path.basename(p):
        path, n = p.rsplit("@", 1)
        n = int(n)
        if not os.path.exists(path) and path.endswith(".part") and os.path.exists(path[:-5]):
            path = path[:-5]
        assert n * 112 <= os.path.getsize(path), f"{p}: file has fewer records"
        return path, n
    return p, os.path.getsize(p) // 112


def block_list(paths, block):
    out = []
    for p in paths:
        path, n = split_path(p)
        out += [(path, s, min(n, s + block)) for s in range(0, n, block)]
    return out


def read_block(b, keep_captures=False):
    p, s, e = b
    if not os.path.exists(p) and p.endswith(".part") and os.path.exists(p[:-5]):
        p = p[:-5]  # a growing file renamed to its final name since the run started (same leading records)
    a = np.fromfile(p, dtype=np.uint8, count=(e - s) * 112, offset=s * 112).reshape(-1, 112)
    return a[T.keep_mask(a, keep_captures)]


def load_records_gpu(paths, max_records, keep_captures, dev, min_ply=0):
    """T.load_records, but the output array is a preallocated GPU tensor filled chunk by chunk (same filter, same order)."""
    chunk = 4_000_000
    counts, total = [], 0
    for p in paths:
        m = np.memmap(p, dtype=np.uint8, mode="r")
        m = m[: len(m) // 112 * 112].reshape(-1, 112)
        total += len(m)
        counts.append(sum(int(T.keep_mask(np.asarray(m[i:i + chunk]), keep_captures, min_ply).sum())
                          for i in range(0, len(m), chunk)))
    out = torch.empty((sum(counts), 112), dtype=torch.uint8, device=dev)
    pos = 0
    for p in paths:
        m = np.memmap(p, dtype=np.uint8, mode="r")
        m = m[: len(m) // 112 * 112].reshape(-1, 112)
        for i in range(0, len(m), chunk):
            a = np.asarray(m[i:i + chunk])
            k = np.ascontiguousarray(a[T.keep_mask(a, keep_captures, min_ply)])
            out[pos:pos + len(k)] = torch.from_numpy(k).to(dev)
            pos += len(k)
        del m
    assert pos == len(out)
    if max_records and len(out) > max_records:
        out = out[-max_records:]
    return out, total



# ---------------------------------------------------------------- teacher loading (an HNN net)
HNN_MAGIC, JQNZ_MAGICS = b"JQHNNUE1", (b"JQHNNZ01", b"JQHNNL01")


def jqnz_decode(path):
    """Decompress a JQHNNZ01 / JQHNNL01 net to a temporary .hnn with tools/hnn/jqnz_tool.
    The tool is taken from $HNN_JQNZ_TOOL, from PATH, or compiled from tools/hnn/jqnz_tool.cpp ($CXX, default g++)."""
    import shutil, subprocess, tempfile
    here = Path(__file__).resolve().parent
    tmp = Path(tempfile.mkdtemp(prefix="hnn_teacher_"))
    tool = os.environ.get("HNN_JQNZ_TOOL") or shutil.which("jqnz_tool")
    if not tool:
        src = here.parent / "tools" / "hnn" / "jqnz_tool.cpp"
        if not src.exists():
            raise SystemExit(f"{path}: compressed net; set HNN_JQNZ_TOOL to a built jqnz_tool or pass the decoded .hnn")
        tool = str(tmp / "jqnz_tool")
        subprocess.run([os.environ.get("CXX", "g++"), "-std=c++20", "-O2", f"-I{here.parent / 'src' / 'hnn'}", str(src),
                        "-o", tool], check=True)
    out = tmp / "teacher.hnn"
    subprocess.run([tool, "d", str(path), str(out)], check=True, stdout=subprocess.DEVNULL)
    return out


class HnnTeacher:
    """HNN net as the teacher: exact integer engine inference (hnn_ref, CPU) of the raw records."""

    def __init__(self, path, dev):
        self.net, self.dev = HR.Net(str(path)), dev

    @torch.no_grad()
    def __call__(self, b):
        rec = b.cpu().numpy() if torch.is_tensor(b) else b
        return torch.from_numpy(HR.evaluate(self.net, rec)[2].astype(np.float32)).to(self.dev)


def load_teacher(args, dev):
    if not args.teacher:
        assert args.wt == 0 and args.wt_end in (None, 0), \
            "--teacher is required unless --wt 0 (and no --wt-end); e.g. --teacher nets/maskfish.nnue"
        print("no teacher (--wt 0)", flush=True)
        return None
    magic = open(args.teacher, "rb").read(8)
    if magic not in JQNZ_MAGICS + (HNN_MAGIC,):
        raise SystemExit(f"{args.teacher}: not an HNN net (JQHNNUE1, JQHNNL01, JQHNNZ01)")
    path = jqnz_decode(args.teacher) if magic in JQNZ_MAGICS else args.teacher
    t = HnnTeacher(path, dev)
    print(f"teacher: HNN net {args.teacher}", flush=True)
    return t

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+")
    ap.add_argument("--val-file", required=True)
    ap.add_argument("--val-max", type=int, default=500_000)
    ap.add_argument("--teacher", default="", help="teacher net: an HNN net with the standard head (.hnn, or a compressed .nnue); may be omitted when --wt 0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--kb-scheme", type=int, default=2, choices=[0, 1, 2, 3])
    ap.add_argument("--factor", action="store_true", help="shared piece-square rows over king buckets (folded at export)")
    ap.add_argument("--init", default="", help="start from these weights: a .pt state of this trainer, or a net file "
                    "(.hnn, or a compressed .nnue such as the released net); --width, --kb-scheme and the head are then its own")
    ap.add_argument("--l1", type=int, default=16, help="head L1 outputs (16|32), only without --init")
    ap.add_argument("--l2", type=int, default=32, help="head L2 outputs (32|64), only without --init")
    ap.add_argument("--sq2", action="store_true", help="squared clipped ReLU after L2 (only without --init)")
    ap.add_argument("--wt", type=float, default=0.6)
    ap.add_argument("--ws", type=float, default=0.3)
    ap.add_argument("--wr", type=float, default=0.1)
    ap.add_argument("--k", type=float, default=0)
    ap.add_argument("--wt-end", type=float, default=None, help="lambda ramp: target weights move linearly from "
                    "(--wt,--ws,--wr) at step 0 to (--wt-end,--ws-end,--wr-end) at the last step (all three needed)")
    ap.add_argument("--ws-end", type=float, default=None)
    ap.add_argument("--wr-end", type=float, default=None)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.0, help="AdamW decoupled weight decay on the weight matrices (not biases)")
    ap.add_argument("--ll-k", type=float, default=430.0, help="also report the val result-logloss at this sigmoid scale")
    ap.add_argument("--qat-from", type=float, default=0.75)
    ap.add_argument("--keep-captures", action="store_true")
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--data-on-gpu", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--gpu-data", action="store_true")
    ap.add_argument("--gpu-load", action="store_true",
                    help="with --gpu-data: fill the records straight into a GPU tensor chunk by chunk (host RAM peak ~0.5 GB "
                         "instead of the full kept size); same records in the same order as load_records")
    ap.add_argument("--tv-cache", default="")
    ap.add_argument("--ckpt-every", type=int, default=0, help="also save ck<step>.pt/.hnn every N steps")
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop after this many steps (schedule unchanged)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stream", type=int, default=0,
                    help="stream from disk: about this many raw records per in-memory group (0 = load everything); "
                         "teacher values are computed per block in epoch 1 and cached in host RAM (4 bytes/record)")
    ap.add_argument("--block", type=int, default=1_000_000, help="with --stream: records per shuffled block")
    ap.add_argument("--eval-only", action="store_true", help="evaluate --init on the val set only (needs --k); no training data loaded")
    ap.add_argument("--flip", type=float, default=0.0, help="probability of mirroring a TRAINING record left-right (0 = off; val never flipped)")
    ap.add_argument("--flip-seed", type=int, default=777, help="seed of the separate flip generator")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights (0 = off); last.pt/.hnn = EMA weights, last_raw.* = raw")
    ap.add_argument("--ema-from", type=float, default=-1.0, help="start EMA at this fraction of the steps (-1 = --qat-from)")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.stream:
        assert not args.data_on_gpu and not args.max_records and not args.eval_only
        blocks = block_list(args.data, args.block)
        with ThreadPoolExecutor(8) as ex:
            bkept = list(ex.map(lambda b: len(read_block(b, args.keep_captures)), blocks))
        n_all = sum(e - s for _, s, e in blocks)
        rng = np.random.default_rng(args.seed)
        per_group = max(1, args.stream // args.block)
        nep = math.ceil(args.epochs)
        plan = []  # per epoch: list of groups (lists of block indices)
        for _ in range(nep):
            order = rng.permutation(len(blocks)).tolist()
            plan.append([order[i:i + per_group] for i in range(0, len(order), per_group)])
        group_steps = [[sum(bkept[j] for j in g) // args.batch for g in ep] for ep in plan]
        samp = rng.choice(len(blocks), min(3, len(blocks)), replace=False)
        raw = np.concatenate([read_block(blocks[j], args.keep_captures) for j in samp])  # K fit only
        for p in args.data:
            path, n = split_path(p)
            kp = sum(k for b, k in zip(blocks, bkept) if b[0] == path)
            print(f"file {p}: {n} records, kept {kp}", flush=True)
        print(f"streaming {n_all} records, kept {sum(bkept)}, {len(blocks)} blocks, {per_group} per group", flush=True)
    elif args.eval_only:
        assert args.init and args.k, "--eval-only needs --init and --k"
        raw, n_all = np.empty((0, 112), np.uint8), 0
    else:
        if args.gpu_load and args.gpu_data:
            raw, n_all = load_records_gpu(args.data, args.max_records, args.keep_captures, dev)
        else:
            raw, n_all = T.load_records(args.data, args.max_records, args.keep_captures)
    vraw, _ = T.load_records([args.val_file], 0, args.keep_captures)
    vraw = vraw[: args.val_max]
    print(f"train {len(raw)} of {n_all} records, val {len(vraw)} ({time.time()-t0:.0f}s)", flush=True)
    host = raw if torch.is_tensor(raw) else torch.from_numpy(raw)
    val = torch.from_numpy(np.ascontiguousarray(vraw)).to(dev)
    hfeat = HFeaturizer(dev, args.kb_scheme)
    teacher = load_teacher(args, dev)

    def teach(data):
        outv = torch.zeros(len(data), dtype=torch.float32, device=dev)
        if teacher is None:  # --wt 0 without --teacher: the teacher term has weight 0
            return outv
        for i in range(0, len(data), 32768):
            b = data[i:i + 32768].to(dev)
            outv[i:i + len(b)] = teacher(b)
        return outv
    tv_train = None
    if len(host) and not args.stream:
        key = None
        if args.tv_cache:
            key = json.dumps(dict(data=[(os.path.abspath(p), os.path.getsize(p), int(os.path.getmtime(p))) for p in args.data],
                                  teacher=(os.path.abspath(args.teacher), os.path.getsize(args.teacher)) if args.teacher else None, n=len(host),
                                  keep_captures=args.keep_captures, max_records=args.max_records), sort_keys=True)
            cf, cj = args.tv_cache + ".npy", args.tv_cache + ".json"
            if os.path.exists(cf) and os.path.exists(cj) and open(cj).read() == key:
                tv_train = torch.from_numpy(np.load(cf)).to(dev)
                print(f"teacher values loaded from cache {cf}", flush=True)
        if tv_train is None:
            tv_train = teach(host)
            if key is not None:
                tmp = f"{args.tv_cache}.tmp{os.getpid()}"
                np.save(tmp + ".npy", tv_train.cpu().numpy())
                open(tmp + ".json", "w").write(key)
                os.replace(tmp + ".npy", args.tv_cache + ".npy"); os.replace(tmp + ".json", args.tv_cache + ".json")
                print(f"teacher values cached to {args.tv_cache}.npy", flush=True)
    tv_val = teach(val)
    print(f"teacher values computed ({time.time()-t0:.0f}s)", flush=True)
    if args.k:
        K = args.k
    else:
        _, _, _, s_all, r_all, _ = T.fields(host[: min(len(host), 2_000_000)].to(dev))
        K = T.fit_k(s_all, r_all)
    print(f"K = {K}", flush=True)
    train = host.to(dev) if args.data_on_gpu else host
    n_train = len(raw)
    gpu_data = args.gpu_data and not args.stream
    if gpu_data:
        train = host.to(dev)
        del raw
        host = train  # host copy freed (K already fitted above)
        print(f"records uploaded to the GPU ({time.time()-t0:.0f}s)", flush=True)
    net = HNN(args.width, args.kb_scheme, args.factor, args.l1, args.l2, args.sq2).to(dev)
    if args.init:
        if open(args.init, "rb").read(8) in JQNZ_MAGICS + (HNN_MAGIC,):
            net = HNN.from_net(args.init).to(dev)
            print(f"init from the net {args.init} (W {net.W}, scheme {net.scheme}, head L1 {net.L1} L2 {net.L2})", flush=True)
        else:
            net = HNN.from_state(torch.load(args.init, map_location="cpu")).to(dev)
    print(f"head L1 {net.L1} L2 {net.L2} sq2 {net.sq2} (L2 input {net.L2IN})", flush=True)
    if args.wd > 0:
        decay = [net.ft, net.l1w, net.l2w, net.ow]
        rest = [p for p in net.parameters() if all(p is not q for q in decay)]
        opt = torch.optim.AdamW([dict(params=decay, weight_decay=args.wd), dict(params=rest, weight_decay=0.0)], lr=args.lr)
    else:
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    if args.stream:
        full = sum(sum(g) for g in group_steps)
        total = int(round(full * args.epochs / nep))
        steps_per_epoch = full // nep
    else:
        steps_per_epoch = max(1, len(train) // args.batch)
        total = int(args.epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in opt.param_groups], total_steps=total, pct_start=0.05,
                                                anneal_strategy="cos", div_factor=10, final_div_factor=100)

    W = [args.wt, args.ws, args.wr]  # same floats as before when no ramp -> unchanged maths
    ramp = None
    if args.wt_end is not None or args.ws_end is not None or args.wr_end is not None:
        assert None not in (args.wt_end, args.ws_end, args.wr_end), "--wt-end/--ws-end/--wr-end go together"
        ramp = ((args.wt, args.ws, args.wr), (args.wt_end, args.ws_end, args.wr_end))
        print(f"lambda ramp {ramp[0]} -> {ramp[1]} over {total} steps", flush=True)

    def loss_of(b, tv):
        _, _, _, score, result, _ = T.fields(b)
        hf, bu, po = hfeat(b)
        y = net(hf, bu, po)
        target = (W[0] * torch.sigmoid(tv / K) + W[1] * torch.sigmoid(score.float() / K)
                  + W[2] * (result.float() + 1) / 2)
        return ((torch.sigmoid(y / K) - target) ** 2).mean(), y

    def logloss(y, b, k):
        r = (T.fields(b)[4].double() + 1) / 2
        p = torch.sigmoid(y.double() / k).clamp(1e-6, 1 - 1e-6)
        return (-(r * p.log() + (1 - r) * (1 - p).log())).sum().item()

    def evaluate():
        net.eval()
        tot, mae, n, ll, ll2 = 0.0, 0.0, 0, 0.0, 0.0
        with torch.no_grad():
            for i in range(0, len(val), 65536):
                b = val[i:i + 65536]
                l, y = loss_of(b, tv_val[i:i + 65536])
                tot += l.item() * len(b)
                mae += (y.clamp(-1500, 1500) - tv_val[i:i + 65536].clamp(-1500, 1500)).abs().sum().item()
                ll += logloss(y, b, K)
                ll2 += logloss(y, b, args.ll_k)
                n += len(b)
        net.train()
        return tot / n, mae / n, ll / n, ll2 / n

    with torch.no_grad():
        tl = sum(logloss(tv_val[i:i + 65536], val[i:i + 65536], K) for i in range(0, len(val), 65536)) / len(val)
        tl2 = sum(logloss(tv_val[i:i + 65536], val[i:i + 65536], args.ll_k) for i in range(0, len(val), 65536)) / len(val)
    print(f"teacher val result-logloss {tl:.4f} (K {K}) {tl2:.4f} (K {args.ll_k})", flush=True)
    v0 = evaluate()
    if args.eval_only:
        print(json.dumps(dict(eval=args.init, K=K, val=v0[0], teacher_mae=v0[1], val_result_logloss=v0[2],
                              val_result_logloss_llk=v0[3])), flush=True)
        return
    print(f"init val loss {v0[0]:.6f} teacher-MAE {v0[1]:.1f} result-logloss {v0[2]:.4f} (K) {v0[3]:.4f} (ll-k)", flush=True)
    log, best, step = [], None, 0
    if args.flip > 0:
        assert args.wt == 0 and (args.wt_end in (None, 0)), "--flip: teacher values are cached for the unflipped records; use --wt 0"
        fgen = torch.Generator(device=dev); fgen.manual_seed(args.flip_seed)
        print(f"flip augmentation p={args.flip} (seed {args.flip_seed})", flush=True)
    ema_sd, ema_from = None, (args.qat_from if args.ema_from < 0 else args.ema_from)
    if args.ema > 0:
        print(f"EMA decay {args.ema} from {ema_from:.3f} of the steps", flush=True)
    def train_on(data, tv, nsteps):
        nonlocal step, ema_sd
        on_gpu = data.device.type != "cpu"
        # same generator call as before: on_gpu here only when the ORIGINAL would also have had the data on
        # the device (--data-on-gpu / --stream); --gpu-data keeps drawing the permutation on the CPU
        perm = torch.randperm(len(data), device=data.device if (args.data_on_gpu or args.stream) else "cpu")
        run_t, rc = torch.zeros((), dtype=torch.float64, device=dev), 0
        nst = min(nsteps, max(0, total - step))
        if args.max_steps:
            nst = min(nst, max(0, args.max_steps - step))
        if not on_gpu:
            import queue, threading
            q = queue.Queue(maxsize=3)
            def producer():
                for i in range(nst):
                    idx = perm[i * args.batch:(i + 1) * args.batch]
                    q.put((idx, data[idx].pin_memory() if dev.type == "cuda" else data[idx]))
            th = threading.Thread(target=producer, daemon=True); th.start()
        else:
            permd = perm.to(dev)
        for i in range(nst):
            if not net.qat and step >= args.qat_from * total:
                net.qat = True
                print(f"step {step}: quantisation-aware from here", flush=True)
            if ramp is not None:
                f = step / max(1, total - 1)
                W[:] = [a + (e - a) * f for a, e in zip(*ramp)]
            if on_gpu:
                idx = permd[i * args.batch:(i + 1) * args.batch]
                b = data[idx]
            else:
                idx, bp = q.get()
                b = bp.to(dev, non_blocking=True)
            if args.flip > 0:
                b = flip_records(b, torch.rand(len(b), device=dev, generator=fgen) < args.flip)
            loss, _ = loss_of(b, tv[idx.to(tv.device)].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            net.clamp_()
            step += 1
            if args.ema > 0 and step >= ema_from * total:
                with torch.no_grad():
                    if ema_sd is None:
                        ema_sd = {k: v.detach().clone() for k, v in net.state_dict().items() if v.is_floating_point()}
                        print(f"step {step}: EMA started", flush=True)
                    else:
                        sd = net.state_dict()
                        for k, v in ema_sd.items():
                            v.lerp_(sd[k], 1.0 - args.ema)
            run_t += loss.detach().double()
            if args.ckpt_every and step % args.ckpt_every == 0:
                torch.save(net.state(), out / f"ck{step}.pt.tmp"); os.replace(out / f"ck{step}.pt.tmp", out / f"ck{step}.pt")
                net.export(out / f"ck{step}.hnn.tmp"); os.replace(out / f"ck{step}.hnn.tmp", out / f"ck{step}.hnn")
                print(f"step {step}: checkpoint ck{step}.pt/.hnn written", flush=True)
                cv = evaluate()
                print(json.dumps(dict(ckval=step, val=cv[0], teacher_mae=cv[1], val_result_logloss=cv[2], val_result_logloss_llk=cv[3],
                                      lr=sched.get_last_lr()[0], sec=round(time.time() - t0))), flush=True)
            if step % 1000 == 0:
                print(f"step {step} {time.time()-t0:.0f}s", flush=True)
            rc += 1
        return run_t.item(), rc

    tv_cache = {}  # block index -> teacher values (host float32)
    if args.stream:
        pool = ThreadPoolExecutor(1)
        load = lambda g: [read_block(blocks[j], args.keep_captures) for j in g]
    for ep in range(math.ceil(args.epochs)):
        if args.stream:
            run, rc = 0.0, 0
            groups = plan[ep]
            fut = pool.submit(load, groups[0])
            for gi, g in enumerate(groups):
                parts = fut.result()
                if gi + 1 < len(groups) and step < total:
                    fut = pool.submit(load, groups[gi + 1])
                if step >= total:
                    break
                data = torch.from_numpy(np.concatenate(parts)).to(dev)
                del parts
                miss = [j for j in g if j not in tv_cache]
                if miss:
                    tvg = teach(data).cpu().numpy()
                    o = 0
                    for j in g:
                        tv_cache[j] = tvg[o:o + bkept[j]].copy()
                        o += bkept[j]
                    assert o == len(data)
                tv = torch.from_numpy(np.concatenate([tv_cache[j] for j in g])).to(dev)
                r, c = train_on(data, tv, len(data) // args.batch)
                run, rc = run + r, rc + c
                del data, tv
        else:
            run, rc = train_on(train, tv_train, steps_per_epoch)
        raw_rec = {}
        if ema_sd is not None:
            rv = evaluate()
            raw_rec = dict(raw_val=rv[0], raw_teacher_mae=rv[1], raw_val_result_logloss=rv[2])
            torch.save(net.state(), out / "last_raw.pt"); net.export(out / "last_raw.hnn")
            print(json.dumps(dict(raw_epoch=ep + 1, step=step, val=rv[0], teacher_mae=rv[1], val_result_logloss=rv[2])), flush=True)
            with torch.no_grad():
                sd = net.state_dict()
                for k, v in ema_sd.items():
                    sd[k].copy_(v)
            net.clamp_()
            print(f"EMA weights swapped in for the epoch-{ep + 1} export", flush=True)
        vl, vm, vll, vll2 = evaluate()
        rec = dict(epoch=ep + 1, step=step, **(dict(w=[round(x, 4) for x in W]) if ramp is not None else {}), train=run / max(rc, 1), val=vl, teacher_mae=vm, val_result_logloss=vll,
                   val_result_logloss_llk=vll2, lr=sched.get_last_lr()[0],
                   sec=round(time.time() - t0), **raw_rec)
        log.append(rec)
        print(json.dumps(rec), flush=True)
        torch.save(net.state(), out / "last.pt")
        net.export(out / "last.hnn")
        if best is None or vl < best:
            best = vl
            torch.save(net.state(), out / "best.pt")
            net.export(out / "best.hnn")
        if args.max_steps and step >= args.max_steps:
            break
    (out / "train.json").write_text(json.dumps(dict(args=vars(args), K=K, records=sum(bkept) if args.stream else n_train, log=log), indent=2) + "\n")


if __name__ == "__main__":
    main()
