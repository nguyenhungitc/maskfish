"""PyTorch model of the engine's NNUE (src/nnue/nnue_architecture.h).

    feature transformer: 7560 -> L1 (+16 PSQT buckets), per perspective
    pairwise product:     2 x (L1/2 x L1/2) -> L1
    layer stack (one of 16, chosen by piece count):
        fc0: L1 -> 16   (output 15 is added directly to the result)
        [sqr_crelu(15), crelu(15)] -> fc1: 30 -> 32 -> crelu -> fc2: 32 -> 1

L1 = 2048, or 1024 for the threat architectures, which also have a second
feature transformer for the 92711 threat features (ft_thr, same outputs and
PSQT buckets, summed with the first one). Threat features are passed as a
flat index tensor with offsets (torch EmbeddingBag format), two bags per
position: side to move, then the other side.

The float model works in "network units": 1.0 == 600 internal units. The
quantization that serialize.py applies is the one the engine expects (see
the comments there); clip_weights() keeps every weight in its int8 range.

Training-only factorization (ideas from nnue-pytorch, also used by AB-JChess);
none of it exists in the engine, coalesced() folds it into the real weights:

* virtual input features: every active feature (king bucket b, piece-square
  p) also activates a row shared by all king buckets for p (ft_ps). What is
  learnt about a piece on a square in one king bucket then helps all the
  others, which matters while data is scarce, and more with the "att"
  architecture (24 buckets). Optionally (kb_factor, used by AB-JChess) also
  a row per king bucket (ft_kb); it is added once per active feature, so it
  learns very fast, and it made results worse in a small test: off by default.
* l1_fact: a fc0 component shared by the 16 layer stacks, so that rarely
  selected stacks (endgames) start from what the frequent ones learnt.

Both start at zero, so a freshly built factorized model computes exactly the
same function as a plain one.
"""

from __future__ import annotations

import torch
from torch import nn

import jieqi_features as F

L1 = 2048            # feature transformer outputs per perspective (F.L1 for the current arch)
L2 = 15
L3 = 32
NNUE2SCORE = 600.0   # network unit -> internal score units

# Quantization constants (must match src/nnue)
FT_SCALE = 127.0                         # file FT weight scale (engine doubles it)
HIDDEN_WEIGHT_SCALE = 64.0               # 1 << WeightScaleBits
QUANTIZED_ONE = 127.0
OUTPUT_SCALE = 16.0                      # OutputScale
OUTPUT_WEIGHT_SCALE = NNUE2SCORE * OUTPUT_SCALE / QUANTIZED_ONE
MAX_HIDDEN_WEIGHT = QUANTIZED_ONE / HIDDEN_WEIGHT_SCALE
MAX_OUTPUT_WEIGHT = QUANTIZED_ONE / OUTPUT_WEIGHT_SCALE


class LayerStacks(nn.Module):
    """All 16 layer stacks, evaluated for the bucket of each sample."""

    def __init__(self, count: int = F.NUM_LAYER_STACKS, l1_fact: bool = False):
        super().__init__()
        self.count = count
        self.in_dim = F.L1
        self.l1 = nn.Linear(self.in_dim, (L2 + 1) * count)
        self.l2 = nn.Linear(L2 * 2, L3 * count)
        self.output = nn.Linear(L3, count)
        with torch.no_grad():
            self.output.weight.zero_()
            self.output.bias.zero_()
        self.l1_fact = None
        if l1_fact:
            self.add_l1_fact()

    def add_l1_fact(self):
        """Shared fc0 component (outputs 0..14 only: output 15 is the direct
        term added to the result, kept per stack like in AB-JChess), zeroed."""
        self.l1_fact = nn.Linear(self.in_dim, L2, bias=False)
        with torch.no_grad():
            self.l1_fact.weight.zero_()

    def forward(self, x: torch.Tensor, buckets: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        idx = torch.arange(n, device=x.device) * self.count + buckets.long()

        l1 = self.l1(x).reshape(n * self.count, L2 + 1)[idx]
        l1x, l1out = l1.split(L2, dim=1)
        if self.l1_fact is not None:
            l1x = l1x + self.l1_fact(x)
        # SqrClippedReLU and ClippedReLU of the engine
        l1x = torch.clamp(torch.cat([l1x.pow(2.0) * (127.0 / 128.0), l1x], dim=1), 0.0, 1.0)

        l2 = self.l2(l1x).reshape(n * self.count, L3)[idx]
        l2x = torch.clamp(l2, 0.0, 1.0)

        w = self.output.weight[buckets.long()]          # (n, L3)
        b = self.output.bias[buckets.long()]            # (n,)
        out = (l2x * w).sum(dim=1) + b
        return (out + l1out.squeeze(1)).unsqueeze(1)

    def clip_weights(self):
        with torch.no_grad():
            if self.l1_fact is None:
                self.l1.weight.clamp_(-MAX_HIDDEN_WEIGHT, MAX_HIDDEN_WEIGHT)
            else:
                # Clip the coalesced weights (stack + shared part), as exported
                w = self.l1.weight.view(self.count, L2 + 1, self.in_dim)
                f = self.l1_fact.weight.unsqueeze(0)
                w[:, :L2] = torch.clamp(w[:, :L2] + f, -MAX_HIDDEN_WEIGHT, MAX_HIDDEN_WEIGHT) - f
                w[:, L2].clamp_(-MAX_HIDDEN_WEIGHT, MAX_HIDDEN_WEIGHT)
            self.l2.weight.clamp_(-MAX_HIDDEN_WEIGHT, MAX_HIDDEN_WEIGHT)
            self.output.weight.clamp_(-MAX_OUTPUT_WEIGHT, MAX_OUTPUT_WEIGHT)


def threat_inputs(bags: list) -> tuple:
    """(indices, offsets) for NNUE.forward from a list of threat index lists
    (stm, nstm, stm, nstm, ... as returned by F.fen_to_threats)."""
    off, n = [], 0
    for bag in bags:
        off.append(n)
        n += len(bag)
    idx = [i for bag in bags for i in bag]
    return torch.tensor(idx, dtype=torch.long), torch.tensor(off, dtype=torch.long)


class NNUE(nn.Module):
    def __init__(self, init_psqt: bool = True, factorized: bool = False, l1_fact: bool = False,
                 kb_factor: bool = False):
        super().__init__()
        L1 = self.l1_size = F.L1
        width = L1 + F.NUM_PSQT_BUCKETS
        self.ft_buckets = F.NUM_FEATURES // F.PS_NB      # king (x attack) buckets
        # Feature transformer: one row per feature (+1 padding row, never exported)
        self.ft = nn.EmbeddingBag(F.NUM_FEATURES + 1, width, mode="sum", padding_idx=F.PAD_INDEX)
        self.ft_bias = nn.Parameter(torch.zeros(width))
        self.ft_ps = self.ft_kb = None
        self.stacks = LayerStacks()

        with torch.no_grad():
            self.ft.weight.uniform_(-0.05, 0.05)
            self.ft.weight[F.PAD_INDEX].zero_()
            self.ft_bias[:L1].uniform_(0.0, 0.5)
            self.ft_bias[L1:].zero_()
            self.ft.weight[:, L1:].zero_()
            if init_psqt:
                values = torch.tensor(F.feature_material_values()) / NNUE2SCORE
                self.ft.weight[:F.NUM_FEATURES, L1:] = values.unsqueeze(1)

        # Threat features (threat architectures): start at zero, so that a new
        # model computes the same function as without them
        self.ft_thr = None
        if F.THREATS:
            self.ft_thr = nn.EmbeddingBag(F.THREAT_DIMS, width, mode="sum")
            with torch.no_grad():
                self.ft_thr.weight.zero_()

        # Training-only parts are created last (and zeroed), so that the real
        # weights drawn from a given seed do not depend on them
        if l1_fact:
            self.stacks.add_l1_fact()
        if factorized:
            self.ft_ps = nn.EmbeddingBag(F.PS_NB + 1, width, mode="sum", padding_idx=F.PS_NB)
            with torch.no_grad():
                self.ft_ps.weight.zero_()
        if factorized and kb_factor:
            self.ft_kb = nn.EmbeddingBag(self.ft_buckets + 1, width, mode="sum",
                                         padding_idx=self.ft_buckets)
            with torch.no_grad():
                self.ft_kb.weight.zero_()

    @property
    def factorized(self) -> bool:
        return self.ft_ps is not None

    @property
    def has_l1_fact(self) -> bool:
        return self.stacks.l1_fact is not None

    @classmethod
    def from_state_dict(cls, state: dict) -> "NNUE":
        """Builds a model with the factorization present in 'state' and loads it."""
        model = cls(init_psqt=False, factorized="ft_ps.weight" in state,
                    l1_fact="stacks.l1_fact.weight" in state, kb_factor="ft_kb.weight" in state)
        model.load_state_dict(state)
        return model

    def transform(self, idx: torch.Tensor) -> torch.Tensor:
        """Feature transformer output (before bias) for (n, MAX_ACTIVE) indices."""
        idx = idx.long()
        out = self.ft(idx)
        if self.ft_ps is not None:
            real = idx != F.PAD_INDEX
            out = out + self.ft_ps(torch.where(real, idx % F.PS_NB, F.PS_NB))
            if self.ft_kb is not None:
                out = out + self.ft_kb(torch.where(real, idx // F.PS_NB, self.ft_buckets))
        return out

    def coalesced(self) -> "NNUE":
        """Plain model (no factorization) computing the same function, on the
        CPU: this is what is exported to the engine."""
        if not self.factorized and not self.has_l1_fact:
            return self
        plain = NNUE(init_psqt=False)
        with torch.no_grad():
            w = self.ft.weight.detach().cpu().clone()
            if self.factorized:
                n = F.NUM_FEATURES
                rows = torch.arange(n)
                w[:n] += self.ft_ps.weight.detach().cpu()[rows % F.PS_NB]
                if self.ft_kb is not None:
                    w[:n] += self.ft_kb.weight.detach().cpu()[rows // F.PS_NB]
                w[F.PAD_INDEX].zero_()
            plain.ft.weight.copy_(w)
            del w
            plain.ft_bias.copy_(self.ft_bias.detach().cpu())
            if self.ft_thr is not None:
                plain.ft_thr.weight.copy_(self.ft_thr.weight.detach().cpu())
            src, dst = self.stacks, plain.stacks
            l1w = src.l1.weight.detach().cpu().clone()
            if src.l1_fact is not None:
                v = l1w.view(src.count, L2 + 1, src.in_dim)
                v[:, :L2] += src.l1_fact.weight.detach().cpu().unsqueeze(0)
            dst.l1.weight.copy_(l1w)
            dst.l1.bias.copy_(src.l1.bias.detach().cpu())
            for name in ("l2", "output"):
                getattr(dst, name).weight.copy_(getattr(src, name).weight.detach().cpu())
                getattr(dst, name).bias.copy_(getattr(src, name).bias.detach().cpu())
        return plain

    def forward(self, stm_idx: torch.Tensor, nstm_idx: torch.Tensor,
                buckets: torch.Tensor, threats=None) -> torch.Tensor:
        """stm_idx / nstm_idx: (n, MAX_ACTIVE) feature indices padded with
        PAD_INDEX. threats (threat architectures): (indices, offsets) of the 2n
        bags of threat features (stm, nstm, stm, nstm, ...). Returns the
        evaluation for the side to move, network units."""
        L1 = self.l1_size
        stm = self.transform(stm_idx) + self.ft_bias
        nstm = self.transform(nstm_idx) + self.ft_bias
        if self.ft_thr is not None:
            if threats is None:
                raise ValueError("this architecture needs the threat features")
            idx, off = threats
            t = self.ft_thr(idx.long(), off.long()).view(stm.shape[0], 2, -1)
            stm = stm + t[:, 0]
            nstm = nstm + t[:, 1]

        b = buckets.long().unsqueeze(1)
        psqt = (stm[:, L1:].gather(1, b) - nstm[:, L1:].gather(1, b)) * 0.5

        s = torch.clamp(stm[:, :L1], 0.0, 1.0)
        o = torch.clamp(nstm[:, :L1], 0.0, 1.0)
        h = L1 // 2
        x = torch.cat([s[:, :h] * s[:, h:], o[:, :h] * o[:, h:]], dim=1) * (127.0 / 128.0)

        return self.stacks(x, buckets) + psqt

    def clip_weights(self):
        self.stacks.clip_weights()
        with torch.no_grad():
            # Keep the FT weights inside the int16 range of the file
            lim = 32767.0 / FT_SCALE / 2.0
            L1 = self.l1_size
            self.ft.weight[:, :L1].clamp_(-lim, lim)
            if self.ft_thr is not None:   # int8 in the file
                self.ft_thr.weight[:, :L1].clamp_(-127.0 / FT_SCALE, 127.0 / FT_SCALE)
            self.ft.weight[F.PAD_INDEX].zero_()
            if self.ft_ps is not None:
                self.ft_ps.weight[F.PS_NB].zero_()
            if self.ft_kb is not None:
                self.ft_kb.weight[self.ft_buckets].zero_()
