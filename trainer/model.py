"""PyTorch model of the engine's NNUE (src/nnue/nnue_architecture.h).

    feature transformer: 7560 -> 2048 (+16 PSQT buckets), per perspective
    pairwise product:     2 x (1024 x 1024) -> 2048
    layer stack (one of 16, chosen by piece count):
        fc0: 2048 -> 16   (output 15 is added directly to the result)
        [sqr_crelu(15), crelu(15)] -> fc1: 30 -> 32 -> crelu -> fc2: 32 -> 1

The float model works in "network units": 1.0 == 600 internal units. The
quantization that serialize.py applies is the one the engine expects (see
the comments there); clip_weights() keeps every weight in its int8 range.
"""

from __future__ import annotations

import torch
from torch import nn

import jieqi_features as F

L1 = 2048            # feature transformer outputs per perspective
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

    def __init__(self, count: int = F.NUM_LAYER_STACKS):
        super().__init__()
        self.count = count
        self.l1 = nn.Linear(L1, (L2 + 1) * count)
        self.l2 = nn.Linear(L2 * 2, L3 * count)
        self.output = nn.Linear(L3, count)
        with torch.no_grad():
            self.output.weight.zero_()
            self.output.bias.zero_()

    def forward(self, x: torch.Tensor, buckets: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        idx = torch.arange(n, device=x.device) * self.count + buckets.long()

        l1 = self.l1(x).reshape(n * self.count, L2 + 1)[idx]
        l1x, l1out = l1.split(L2, dim=1)
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
            self.l1.weight.clamp_(-MAX_HIDDEN_WEIGHT, MAX_HIDDEN_WEIGHT)
            self.l2.weight.clamp_(-MAX_HIDDEN_WEIGHT, MAX_HIDDEN_WEIGHT)
            self.output.weight.clamp_(-MAX_OUTPUT_WEIGHT, MAX_OUTPUT_WEIGHT)


class NNUE(nn.Module):
    def __init__(self, init_psqt: bool = True):
        super().__init__()
        # Feature transformer: one row per feature (+1 padding row, never exported)
        self.ft = nn.EmbeddingBag(F.NUM_FEATURES + 1, L1 + F.NUM_PSQT_BUCKETS, mode="sum",
                                  padding_idx=F.PAD_INDEX)
        self.ft_bias = nn.Parameter(torch.zeros(L1 + F.NUM_PSQT_BUCKETS))
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

    def forward(self, stm_idx: torch.Tensor, nstm_idx: torch.Tensor,
                buckets: torch.Tensor) -> torch.Tensor:
        """stm_idx / nstm_idx: (n, MAX_ACTIVE) feature indices padded with
        PAD_INDEX. Returns the evaluation for the side to move, network units."""
        stm = self.ft(stm_idx.long()) + self.ft_bias
        nstm = self.ft(nstm_idx.long()) + self.ft_bias

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
            self.ft.weight[:, :L1].clamp_(-lim, lim)
            self.ft.weight[F.PAD_INDEX].zero_()
