"""PyTorch model of the engine's NNUE (maskfish/src/nnue/nnue_architecture.h).

    feature transformer: 7560 -> L1 (+16 PSQT buckets), per perspective
    pairwise product:     2 x (L1/2 x L1/2) -> L1
    layer stack (one of 16, chosen by piece count):
        fc0: L1 -> 16   (output 15 is added directly to the result)
        [sqr_crelu(15), crelu(15)] -> fc1: 30 -> 32 -> crelu -> fc2: 32 -> 1
    or, for the "*-l2" architectures (make layers=2), see LayerStacks:
        fc0: L1 -> 32 -> [sqr_crelu, crelu] -> fc1: 64 -> 32 -> [sqr_crelu, crelu]
        fc2: all 128 activations -> 1, plus fc0[30] - fc0[31]

The "*-ctx" architectures (make ctx=yes, AB-JChess's inventory context) also
have a small layer, ctx_tower: 16 counts describing the uncertainty of the
position (see jieqi_features.context_inputs(), recovered here from the
features of the side to move) -> 16 -> clipped ReLU, whose outputs are 16
more inputs of fc0 in every layer stack (LayerStacks.l1_ctx).

L1 = 2048, or 1024 for the threat architectures, which also have a second
feature transformer for the 92711 threat features (ft_thr, same outputs and
PSQT buckets, summed with the first one). The "*-nopsqt" architectures (make
psqt=no) have no PSQT buckets: the output is the layer stack alone. Threat features are passed as a
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

Quantization-aware training (train.py --qat-start, see forward_parts_qat()):
the forward pass computes what the engine computes, with the weights rounded
as serialize.quantize() rounds them and the integer arithmetic of the engine
(shifts, divisions, saturations), in floats holding integer values. The
roundings let the gradient through unchanged (straight-through estimator),
so the float weights keep learning while the loss is that of the exported
network. In float64 it is exactly serialize.quantized_eval() (verify.py
--checkpoint checks it).
"""

from __future__ import annotations

import torch
from torch import nn

import jieqi_features as F
import quant as Q

L1 = 2048            # feature transformer outputs per perspective (F.L1 for the current arch)
L2 = 15              # layer stacks 1: activated fc0 outputs (+1 direct term)
L2_V2 = 32           # layer stacks 2 (make layers=2): fc0 outputs
L3 = 32
NNUE2SCORE = 600.0   # network unit -> internal score units

# Quantization constants of the default scheme (make quant=1), kept for other
# scripts; the model uses quant.scales() of its architecture ("-q2": make quant=2)
FT_SCALE = 127.0                         # file FT scale (the engine doubles the outputs)
HIDDEN_WEIGHT_SCALE = 64.0               # 1 << WeightScaleBits
QUANTIZED_ONE = 127.0
OUTPUT_SCALE = 16.0                      # OutputScale
OUTPUT_WEIGHT_SCALE = NNUE2SCORE * OUTPUT_SCALE / QUANTIZED_ONE
MAX_HIDDEN_WEIGHT = QUANTIZED_ONE / HIDDEN_WEIGHT_SCALE
MAX_OUTPUT_WEIGHT = QUANTIZED_ONE / OUTPUT_WEIGHT_SCALE
# Feature transformer: int8 weights (HalfKA and threats) and int16 biases at
# scale FT_SCALE. The bias bound keeps the engine's 16-bit accumulators from
# wrapping around: |bias| + 62 x 127 <= 32767 (maskfish/src/nnue/nnue_feature_transformer.h)
MAX_FT_WEIGHT = 127.0 / FT_SCALE
MAX_FT_BIAS = 16383.0 / FT_SCALE

# Quantized units (see serialize.quantize() and the engine's propagate()):
# fc0 / fc1 outputs and biases use 1.0 == HIDDEN_ONE, the PSQT and the fc2
# output 1.0 == OUTPUT_ONE (network unit, then / OUTPUT_SCALE = internal units)
HIDDEN_ONE = HIDDEN_WEIGHT_SCALE * QUANTIZED_ONE        # 8128
OUTPUT_ONE = NNUE2SCORE * OUTPUT_SCALE                  # 9600


def _ste(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Value of y, gradient of x (straight-through estimator)."""
    return x + (y - x).detach()


def q_round(x: torch.Tensor) -> torch.Tensor:
    return _ste(x, torch.round(x))      # half to even, like np.round in serialize.py


def q_round_ft(w: torch.Tensor, scale: float) -> torch.Tensor:
    """round(w * scale) computed in float32 like serialize.quantize() does for
    the feature transformer weights (whatever the dtype of w)."""
    return _ste(w * scale, torch.round(w.float() * scale).to(w.dtype))


def q_floor(x: torch.Tensor) -> torch.Tensor:
    """Arithmetic shift right of the engine (x >> k == floor(x / 2^k))."""
    return _ste(x, torch.floor(x))


def q_tdiv(x: torch.Tensor, d: float) -> torch.Tensor:
    """C++ integer division by d (truncates toward zero)."""
    return _ste(x / d, torch.trunc(x / d))


def wide(x: torch.Tensor) -> torch.Tensor:
    """float64 for the few per-position products that exceed the 24 bits
    float32 holds exactly (not on MPS, which has no float64)."""
    return x if x.device.type == "mps" else x.double()


def _sqr_crelu_q(v: torch.Tensor, shift: int) -> torch.Tensor:
    """SqrClippedReLU and ClippedReLU of the engine on integer fc outputs:
    min(127, v * v >> (2 * shift + 7)) and clamp(v >> shift, 0, 127)."""
    sq = torch.clamp(q_floor(v * v / float(1 << (2 * shift + 7))), max=127.0)
    return torch.cat([sq, torch.clamp(q_floor(v / float(1 << shift)), 0.0, 127.0)], dim=1)


class LayerStacks(nn.Module):
    """All 16 layer stacks, evaluated for the bucket of each sample.

    Layer stacks 1 (default):
        fc0: L1 -> 16 (output 15 is added directly to the result)
        [sqr_crelu(15), crelu(15)] -> fc1: 30 -> 32 -> crelu -> fc2: 32 -> 1
    Layer stacks 2 (make layers=2, architectures "*-l2"):
        fc0: L1 -> 32, a0 = [sqr_crelu(32), crelu(32)]
        fc1: 64 -> 32, a1 = [sqr_crelu(32), crelu(32)]
        fc2: [a0, a1] (128) -> 1, plus the direct term fc0[30] - fc0[31]
    """

    def __init__(self, count: int = F.NUM_LAYER_STACKS, l1_fact: bool = False):
        super().__init__()
        self.count = count
        self.s = Q.scales()          # quantization of the architecture
        self.ctx = F.CTX
        self.in_dim = F.L1
        self.layers = F.LAYERS
        if self.layers == 2:
            self.fc0_out, self.act0 = L2_V2, L2_V2         # 32 outputs, all activated
            self.out_in = 2 * L2_V2 + 2 * L3
        else:
            self.fc0_out, self.act0 = L2 + 1, L2           # 15 activated + direct term
            self.out_in = L3
        self.l1 = nn.Linear(self.in_dim, self.fc0_out * count)
        # Context inputs of fc0 ("*-ctx"): every output, also the direct term
        self.l1_ctx = None
        if self.ctx:
            self.l1_ctx = nn.Linear(F.CONTEXT_INPUTS, self.fc0_out * count, bias=False)
        self.l2 = nn.Linear(self.act0 * 2, L3 * count)
        self.output = nn.Linear(self.out_in, count)
        with torch.no_grad():
            self.output.weight.zero_()
            self.output.bias.zero_()
        self.l1_fact = None
        if l1_fact:
            self.add_l1_fact()

    def add_l1_fact(self):
        """Shared fc0 component, zeroed. Layer stacks 1: outputs 0..14 only
        (output 15 is the direct term added to the result, kept per stack like
        in AB-JChess); layer stacks 2: all the outputs."""
        self.l1_fact = nn.Linear(self.in_dim, self.act0, bias=False)
        with torch.no_grad():
            self.l1_fact.weight.zero_()

    def _activate(self, v: torch.Tensor) -> torch.Tensor:
        """SqrClippedReLU and ClippedReLU of the engine, concatenated (float:
        quant=1 clamps to 1.0 with v^2 * 127/128, quant=2 to 127/128 with v^2)."""
        s = self.s
        return torch.clamp(torch.cat([v.pow(2.0) * s.sqr_gain, v], dim=1), 0.0, s.act_max)

    def forward(self, x: torch.Tensor, buckets: torch.Tensor, ctx=None, blend=None) -> torch.Tensor:
        """Output of the layer stack of each sample. blend ("-blend"
        architectures): interpolation weights (n,) in 0..254, the output is then
        that of stack b + (that of stack b + 1 - that of stack b) * blend / 256."""
        n = x.shape[0]
        l1_all = self.l1(x)
        if self.l1_ctx is not None:
            l1_all = l1_all + self.l1_ctx(ctx)
        l1_all = l1_all.reshape(n * self.count, self.fc0_out)
        fact = self.l1_fact(x) if self.l1_fact is not None else None
        b = buckets.long()
        out = self._forward_stack(l1_all, fact, b)
        if blend is not None:
            nxt = self._forward_stack(l1_all, fact, torch.clamp(b + 1, max=self.count - 1))
            out = out + (nxt - out) * (blend.float() / 256.0).unsqueeze(1)
        return out

    def _forward_stack(self, l1_all: torch.Tensor, fact, buckets: torch.Tensor) -> torch.Tensor:
        n = buckets.shape[0]
        idx = torch.arange(n, device=l1_all.device) * self.count + buckets
        l1 = l1_all[idx]
        if self.layers == 2:
            if fact is not None:
                l1 = l1 + fact
            a0 = self._activate(l1)
            l2 = self.l2(a0).reshape(n * self.count, L3)[idx]
            h = torch.cat([a0, self._activate(l2)], dim=1)
            direct = l1[:, L2_V2 - 2] - l1[:, L2_V2 - 1]
        else:
            l1x, l1out = l1.split(L2, dim=1)
            if fact is not None:
                l1x = l1x + fact
            a0 = self._activate(l1x)
            l2 = self.l2(a0).reshape(n * self.count, L3)[idx]
            h = torch.clamp(l2, 0.0, self.s.act_max)
            direct = l1out.squeeze(1)

        w = self.output.weight[buckets]          # (n, out_in)
        b = self.output.bias[buckets]            # (n,)
        out = (h * w).sum(dim=1) + b
        return (out + direct).unsqueeze(1)

    def quantized_weights(self) -> dict:
        """Integer-valued weights (with straight-through gradients), exactly
        those serialize.quantize() writes: the shared fc0 part folded in,
        weights clipped to the int8 range and rounded."""
        w0 = self.l1.weight
        if self.l1_fact is not None:
            v = w0.view(self.count, self.fc0_out, self.in_dim)
            a = self.act0
            w0 = torch.cat([v[:, :a] + self.l1_fact.weight.unsqueeze(0), v[:, a:]], dim=1)
            w0 = w0.reshape(self.count * self.fc0_out, self.in_dim)
        s = self.s

        def qw(w, scale):
            m = s.max_weight(scale)
            return q_round(torch.clamp(w, -m, m) * scale)

        q = {"w0": qw(w0, s.w0), "b0": q_round(self.l1.bias * s.one0),
             "w1": qw(self.l2.weight, s.w1), "b1": q_round(self.l2.bias * s.one1),
             "w2": qw(self.output.weight, s.w2), "b2": q_round(self.output.bias * s.out2)}
        if self.l1_ctx is not None:
            q["w0_ctx"] = qw(self.l1_ctx.weight, s.w0_ctx)
        return q

    def forward_qat(self, x: torch.Tensor, buckets: torch.Tensor, ctx=None,
                    blend=None) -> torch.Tensor:
        """Integer output of the layer stack of each sample (OutputScale
        units, before the final / 16), as NetworkArchitecture::propagate():
        x the integer feature transformer output (0..127), ctx the integer
        context tower output."""
        q = self.quantized_weights()
        n = x.shape[0]
        l1_all = torch.nn.functional.linear(x, q["w0"], q["b0"])
        if ctx is not None:
            l1_all = l1_all + torch.nn.functional.linear(ctx, q["w0_ctx"])
        l1_all = l1_all.reshape(n * self.count, self.fc0_out)
        b = buckets.long()
        out = self._forward_stack_qat(q, l1_all, b)
        if blend is not None:
            nxt = self._forward_stack_qat(q, l1_all, torch.clamp(b + 1, max=self.count - 1))
            w = wide(blend.to(out.dtype).unsqueeze(1))
            out = q_tdiv(wide(out) * (256.0 - w) + wide(nxt) * w, 256.0).to(out.dtype)
        return out

    def _forward_stack_qat(self, q: dict, l1_all: torch.Tensor, buckets: torch.Tensor):
        s = self.s
        n = buckets.shape[0]
        idx = torch.arange(n, device=l1_all.device) * self.count + buckets
        l1 = l1_all[idx]
        if self.layers == 2:
            a0 = _sqr_crelu_q(l1, s.shift0)
            l2 = torch.nn.functional.linear(a0, q["w1"], q["b1"]).reshape(n * self.count, L3)[idx]
            h = torch.cat([a0, _sqr_crelu_q(l2, s.shift1)], dim=1)
            direct = l1[:, L2_V2 - 2] - l1[:, L2_V2 - 1]
        else:
            a0 = _sqr_crelu_q(l1[:, :L2], s.shift0)
            l2 = torch.nn.functional.linear(a0, q["w1"], q["b1"]).reshape(n * self.count, L3)[idx]
            h = torch.clamp(q_floor(l2 / float(1 << s.shift1)), 0.0, 127.0)
            direct = l1[:, L2]
        out = (h * q["w2"][buckets]).sum(dim=1) + q["b2"][buckets]
        if s.quant2:
            # fc2 is in fc0 units: the sum is converted to OUTPUT_ONE (C++ division)
            return q_tdiv(wide(out + direct) * OUTPUT_ONE, s.one0).to(out.dtype).unsqueeze(1)
        # the direct term: 1.0 is one0, scaled to OUTPUT_ONE (C++ division)
        return (out + q_tdiv(wide(direct) * OUTPUT_ONE, s.one0).to(out.dtype)).unsqueeze(1)

    def clip_weights(self):
        s = self.s
        m0, m1, m2 = s.max_weight(s.w0), s.max_weight(s.w1), s.max_weight(s.w2)
        with torch.no_grad():
            if self.l1_fact is None:
                self.l1.weight.clamp_(-m0, m0)
            else:
                # Clip the coalesced weights (stack + shared part), as exported
                w = self.l1.weight.view(self.count, self.fc0_out, self.in_dim)
                f = self.l1_fact.weight.unsqueeze(0)
                a = self.act0
                w[:, :a] = torch.clamp(w[:, :a] + f, -m0, m0) - f
                w[:, a:].clamp_(-m0, m0)
            self.l2.weight.clamp_(-m1, m1)
            self.output.weight.clamp_(-m2, m2)
            if self.l1_ctx is not None:
                mc = s.max_weight(s.w0_ctx)
                self.l1_ctx.weight.clamp_(-mc, mc)


def context_from_features(stm_idx: torch.Tensor) -> torch.Tensor:
    """The 16 quantized context inputs (0..127, as floats) of each position,
    recovered from the (n, MAX_ACTIVE) features of the side to move: exactly
    jieqi_features.context_inputs() of the position."""
    idx = stm_idx.long()
    real = idx != F.PAD_INDEX
    local = torch.where(real, idx % F.PS_NB, torch.full_like(idx, -1))

    def count(mask):
        return mask.sum(dim=1)

    own = [count(local == f) for f in F.CONTEXT_OWN_POOL]
    opp = [count(local == f) for f in F.CONTEXT_OPP_POOL]
    sq = local - F.PS_DARK
    dark_sq = (sq >= 0) & (sq < F.SQUARE_NB)
    rank = torch.div(sq, F.FILE_NB, rounding_mode="floor")
    pool_sq = (sq >= 9) & (sq < 15) | (sq >= 72) & (sq < 78)
    dark_us = count(dark_sq & ~pool_sq & (rank < 5))
    dark_them = count(dark_sq & ~pool_sq & (rank >= 5))
    pool_us, pool_them = sum(own), sum(opp)

    def q(n, d):   # round(127 * n / d), half up, at most 127
        return torch.clamp(torch.div(2 * 127 * n + d, 2 * d, rounding_mode="floor"), max=127)

    cols = [q(n, d) for n, d in zip(own, F.CONTEXT_POOL_MAX)]
    cols += [q(n, d) for n, d in zip(opp, F.CONTEXT_POOL_MAX)]
    cols += [q(dark_us, F.CONTEXT_DARK_MAX), q(dark_them, F.CONTEXT_DARK_MAX),
             q(torch.clamp(pool_us - dark_us, min=0), F.CONTEXT_UNKNOWN_MAX),
             q(torch.clamp(pool_them - dark_them, min=0), F.CONTEXT_UNKNOWN_MAX)]
    return torch.stack(cols, dim=1).float()


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
        self.s = Q.scales()          # quantization of the architecture
        self.psqt_buckets = F.NUM_PSQT_BUCKETS if F.PSQT else 0
        width = L1 + self.psqt_buckets
        self.ft_buckets = F.NUM_FEATURES // F.PS_NB      # king (x attack) buckets
        # Feature transformer: one row per feature (+1 padding row, never exported)
        self.ft = nn.EmbeddingBag(F.NUM_FEATURES + 1, width, mode="sum", padding_idx=F.PAD_INDEX)
        self.ft_bias = nn.Parameter(torch.zeros(width))
        self.ft_ps = self.ft_kb = None
        self.stacks = LayerStacks()
        # Inventory context ("*-ctx"): 16 -> 16, clipped ReLU
        self.ctx_tower = nn.Linear(F.CONTEXT_INPUTS, F.CONTEXT_INPUTS) if F.CTX else None

        with torch.no_grad():
            self.ft.weight.uniform_(-0.05, 0.05)
            self.ft.weight[F.PAD_INDEX].zero_()
            self.ft_bias[:L1].uniform_(0.0, 0.5)
            self.ft_bias[L1:].zero_()
            self.ft.weight[:, L1:].zero_()
            if init_psqt and self.psqt_buckets:
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
            if self.ctx_tower is not None:
                plain.ctx_tower.load_state_dict(self.ctx_tower.state_dict())
                plain.stacks.l1_ctx.weight.copy_(self.stacks.l1_ctx.weight.detach().cpu())
            src, dst = self.stacks, plain.stacks
            l1w = src.l1.weight.detach().cpu().clone()
            if src.l1_fact is not None:
                v = l1w.view(src.count, src.fc0_out, src.in_dim)
                v[:, :src.act0] += src.l1_fact.weight.detach().cpu().unsqueeze(0)
            dst.l1.weight.copy_(l1w)
            dst.l1.bias.copy_(src.l1.bias.detach().cpu())
            for name in ("l2", "output"):
                getattr(dst, name).weight.copy_(getattr(src, name).weight.detach().cpu())
                getattr(dst, name).bias.copy_(getattr(src, name).bias.detach().cpu())
        return plain

    def forward(self, stm_idx: torch.Tensor, nstm_idx: torch.Tensor,
                buckets: torch.Tensor, threats=None, blend=None) -> torch.Tensor:
        """stm_idx / nstm_idx: (n, MAX_ACTIVE) feature indices padded with
        PAD_INDEX. threats (threat architectures): (indices, offsets) of the 2n
        bags of threat features (stm, nstm, stm, nstm, ...). Returns the
        evaluation for the side to move, network units (psqt + positional)."""
        psqt, positional = self.forward_parts(stm_idx, nstm_idx, buckets, threats, blend)
        return psqt + positional

    def forward_parts(self, stm_idx: torch.Tensor, nstm_idx: torch.Tensor,
                      buckets: torch.Tensor, threats=None,
                      blend=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Like forward(), but returns the two parts of the network output
        separately, (psqt, positional), each (n, 1) in network units: the
        engine's evaluate() scaling needs both (see eval_scale.py)."""
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
        if self.psqt_buckets:
            psqt = (stm[:, L1:].gather(1, b) - nstm[:, L1:].gather(1, b)) * 0.5
        else:
            psqt = stm.new_zeros((stm.shape[0], 1))

        s = torch.clamp(stm[:, :L1], 0.0, 1.0)
        o = torch.clamp(nstm[:, :L1], 0.0, 1.0)
        h = L1 // 2
        x = torch.cat([s[:, :h] * s[:, h:], o[:, :h] * o[:, h:]], dim=1) * self.s.x_gain

        ctx = None
        if self.ctx_tower is not None:
            ctx = torch.clamp(self.ctx_tower(context_from_features(stm_idx) / 127.0), 0.0, 1.0)
        if F.BLEND and blend is None:
            raise ValueError("this architecture needs the layer stack interpolation weights")
        return psqt, self.stacks(x, buckets, ctx, blend if F.BLEND else None)

    def forward_parts_qat(self, stm_idx: torch.Tensor, nstm_idx: torch.Tensor,
                          buckets: torch.Tensor, threats=None,
                          blend=None) -> tuple[torch.Tensor, torch.Tensor]:
        """(psqt, positional), each (n, 1), in INTERNAL units with integer
        values: the output of the exported network as the engine computes it
        (Network::evaluate()), differentiable with straight-through roundings.
        Costs about one extra copy of the feature transformer weights."""
        L1 = self.l1_size
        n = F.NUM_FEATURES
        w = self.ft.weight[:n]
        if self.ft_ps is not None:             # the virtual features, folded in
            rows = torch.arange(n, device=w.device)
            w = w + self.ft_ps.weight[rows % F.PS_NB]
            if self.ft_kb is not None:
                w = w + self.ft_kb.weight[rows // F.PS_NB]
        pad = w.new_zeros((1, w.shape[1]))
        sc = self.s
        wq = torch.cat([q_round_ft(w[:, :L1], sc.ft), q_round(w[:, L1:] * OUTPUT_ONE)], dim=1)
        wq = torch.cat([wq, pad])              # PAD_INDEX row
        bias = torch.cat([q_round(self.ft_bias[:L1] * sc.ft),
                          self.ft_bias.new_zeros(self.psqt_buckets)])
        stm = torch.nn.functional.embedding_bag(stm_idx.long(), wq, mode="sum") + bias
        nstm = torch.nn.functional.embedding_bag(nstm_idx.long(), wq, mode="sum") + bias
        if self.ft_thr is not None:
            if threats is None:
                raise ValueError("this architecture needs the threat features")
            tw = self.ft_thr.weight
            tq = torch.cat([q_round_ft(tw[:, :L1], sc.ft), q_round(tw[:, L1:] * OUTPUT_ONE)],
                           dim=1)
            idx, off = threats
            t = torch.nn.functional.embedding_bag(idx.long(), tq, off.long(), mode="sum")
            t = t.view(stm.shape[0], 2, -1)
            stm = stm + t[:, 0]
            nstm = nstm + t[:, 1]

        b = buckets.long().unsqueeze(1)
        if self.psqt_buckets:
            psqt = q_tdiv(stm[:, L1:].gather(1, b) - nstm[:, L1:].gather(1, b), 2.0)
        else:
            psqt = stm.new_zeros((stm.shape[0], 1))

        # FeatureTransformer::transform(): accumulators doubled (quant=1),
        # clamped to 0..254 (0..255), the two halves multiplied, >> 9
        h = L1 // 2
        parts = []
        for acc in (stm, nstm):
            a = torch.clamp(sc.ft_mult * acc[:, :h], 0.0, float(sc.ft_clip))
            c = torch.clamp(sc.ft_mult * acc[:, h:L1], 0.0, float(sc.ft_clip))
            parts.append(q_floor(a * c / 512.0))
        x = torch.cat(parts, dim=1)

        ctx = None
        if self.ctx_tower is not None:
            m = sc.max_weight(Q.CTX_W)
            cw = q_round(torch.clamp(self.ctx_tower.weight, -m, m) * Q.CTX_W)
            cb = q_round(self.ctx_tower.bias * Q.CTX_ONE)
            ci = context_from_features(stm_idx).to(x.dtype)
            ctx = torch.clamp(q_floor(torch.nn.functional.linear(ci, cw, cb) / 64.0), 0.0, 127.0)
        if F.BLEND and blend is None:
            raise ValueError("this architecture needs the layer stack interpolation weights")
        positional = self.stacks.forward_qat(x, buckets, ctx, blend if F.BLEND else None)
        return q_tdiv(psqt, OUTPUT_SCALE), q_tdiv(positional, OUTPUT_SCALE)

    def clip_weights(self):
        self.stacks.clip_weights()
        with torch.no_grad():
            if self.ctx_tower is not None:
                m = self.s.max_weight(Q.CTX_W)
                self.ctx_tower.weight.clamp_(-m, m)
            # Keep the FT weights inside the int8 range of the file: the
            # coalesced weights (with the virtual features), as exported
            L1 = self.l1_size
            lim = self.s.max_ft_weight
            if self.ft_ps is None:
                self.ft.weight[:, :L1].clamp_(-lim, lim)
            else:
                n, w = F.NUM_FEATURES, self.ft.weight[:F.NUM_FEATURES]
                rows = torch.arange(n, device=w.device)
                chunk = 4096                   # bounds the temporary memory
                for i in range(0, n, chunk):
                    r = rows[i:i + chunk]
                    f = self.ft_ps.weight[r % F.PS_NB, :L1]
                    if self.ft_kb is not None:
                        f = f + self.ft_kb.weight[r // F.PS_NB, :L1]
                    w[i:i + chunk, :L1] = torch.clamp(w[i:i + chunk, :L1] + f, -lim, lim) - f
            self.ft_bias[:L1].clamp_(-self.s.max_ft_bias, self.s.max_ft_bias)
            if self.ft_thr is not None:
                self.ft_thr.weight[:, :L1].clamp_(-lim, lim)
            self.ft.weight[F.PAD_INDEX].zero_()
            if self.ft_ps is not None:
                self.ft_ps.weight[F.PS_NB].zero_()
            if self.ft_kb is not None:
                self.ft_kb.weight[self.ft_buckets].zero_()
