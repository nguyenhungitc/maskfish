"""Quantization of the exported network: how the float model maps to the
integers of the engine (maskfish/src/nnue/nnue_common.h), for the two
schemes the engine can be built with.

    quant=1 (default)                         quant=2 ("-q2" architectures, make quant=2)
    FT weights / biases  x 127, the engine    x 255, not doubled
      doubles the accumulators, clip 0..254     clip 0..255
    FT output  floor(a * b / 512), 0..127     same formula
      float x = clamp(a) clamp(b) * 127/128     float x = clamp(a) clamp(b) * 255^2 / 512 / 128
      1.0 = 127                                 1.0 = 128 (HiddenOneVal)
    fc0  weights x 64, 1.0 = 127 * 64          weights x 128, 1.0 = 128 * 64 * 2
         activations >> 6, sqr >> 19            activations >> 7, sqr >> 21
    fc1  weights x 64, 1.0 = 127 * 64          weights x 64, 1.0 = 128 * 64
    activations: 1.0 = 127, clip 0..1 float    1.0 = 128, clip 0..127/128 float
         sqr: 127/128 v^2                       sqr: v^2
    fc2  weights x 600 * 16 / 127, output in   weights x 128, output in fc0 units
         OutputScale units (1.0 = 9600); the    (1.0 = 16384); the direct term is
         direct term converted and added        added, then the sum converted:
         out = fc2 + direct * 9600 / 8128       out = (fc2 + direct) * 9600 / 16384
    context tower (ctx): unchanged (outputs 127 = 1.0); its fc0 weights are at
    scale 64 (quant=1) or 16384 / 127 (quant=2), so that they add fc0 units.

In both, the PSQT weights are at scale 600 * 16 and the float network unit is
600 internal units. quant=2 is the alternative scheme (fc0 and the final
layer at twice the resolution, the
feature transformer at twice the resolution but half the range).
"""

from __future__ import annotations

from typing import NamedTuple

import jieqi_features as F

NNUE2SCORE = 600.0      # network unit -> internal score units
OUTPUT_SCALE = 16.0     # OutputScale
OUTPUT_ONE = NNUE2SCORE * OUTPUT_SCALE    # 9600: PSQT and (quant=1) fc2 outputs
MAX_FT_BIAS_INT = 16383  # keeps the engine's 16-bit accumulators from wrapping around


class Scales(NamedTuple):
    quant2: bool
    ft: float           # feature transformer weights and biases
    ft_mult: int        # the engine multiplies the accumulators by this ...
    ft_clip: int        # ... then clips them to 0..ft_clip
    act_one: float      # quantized 1.0 of every activation (FT output, fc0 / fc1)
    act_max: float      # largest activation, float (127 / act_one)
    x_gain: float       # float FT output = clamp(a) * clamp(b) * x_gain
    sqr_gain: float     # float squared activation = sqr_gain * v^2
    w0: float           # fc0 weights
    shift0: int         # fc0 activations >> shift0 (squared: >> 2 * shift0 + 7)
    w1: float           # fc1 weights
    shift1: int
    w2: float           # fc2 weights
    out2: float         # quantized 1.0 of the fc2 output (and its bias)
    w0_ctx: float       # fc0 weights of the context inputs

    @property
    def one0(self) -> float:        # quantized 1.0 of the fc0 outputs (and biases)
        return self.act_one * self.w0

    @property
    def one1(self) -> float:
        return self.act_one * self.w1

    @property
    def max_ft_weight(self) -> float:
        return 127.0 / self.ft

    @property
    def max_ft_bias(self) -> float:
        return MAX_FT_BIAS_INT / self.ft

    def max_weight(self, scale: float) -> float:
        """Largest float weight that fits in int8 at this scale."""
        return 127.0 / scale


CTX_W = 64.0            # context tower weights (both schemes)
CTX_ONE = 64.0 * 127.0  # context tower biases / pre-activations

QUANT1 = Scales(quant2=False, ft=127.0, ft_mult=2, ft_clip=254, act_one=127.0, act_max=1.0,
                x_gain=127.0 / 128.0, sqr_gain=127.0 / 128.0, w0=64.0, shift0=6, w1=64.0,
                shift1=6, w2=OUTPUT_ONE / 127.0, out2=OUTPUT_ONE, w0_ctx=64.0)
QUANT2 = Scales(quant2=True, ft=255.0, ft_mult=1, ft_clip=255, act_one=128.0,
                act_max=127.0 / 128.0, x_gain=255.0 * 255.0 / 512.0 / 128.0, sqr_gain=1.0,
                w0=128.0, shift0=7, w1=64.0, shift1=6, w2=128.0, out2=128.0 * 128.0,
                w0_ctx=128.0 * 128.0 / 127.0)


def scales(quant2: bool | None = None) -> Scales:
    """Scales of the current architecture (F.ARCH), or of the given scheme."""
    return QUANT2 if (F.QUANT2 if quant2 is None else quant2) else QUANT1
