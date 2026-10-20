/*
  Maskfish, a UCI jieqi engine derived from Pikafish and Stockfish
  Copyright (C) 2026 The Maskfish developers (see AUTHORS file)
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)

  Maskfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Maskfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// Definition of layer ClippedReLU of NNUE evaluation function

#ifndef NNUE_LAYERS_SQR_CLIPPED_RELU_H_INCLUDED
#define NNUE_LAYERS_SQR_CLIPPED_RELU_H_INCLUDED

#include <algorithm>
#include <cstdint>
#include <iosfwd>

#include "../nnue_common.h"

namespace Maskfish::Eval::NNUE::Layers {

// Squared clipped ReLU: min(127, input^2 >> (2 * WeightScaleBitsLocal + 7)).
// make quant=2 uses WeightScaleBits + 1 after fc_0 (as Pikafish).
template<IndexType InDims, int WeightScaleBitsLocal = WeightScaleBits>
class SqrClippedReLU {
   public:
    // Input/output type
    using InputType  = std::int32_t;
    using OutputType = std::uint8_t;

    // Number of input/output dimensions
    static constexpr IndexType InputDimensions  = InDims;
    static constexpr IndexType OutputDimensions = InputDimensions;
    static constexpr IndexType PaddedOutputDimensions =
      ceil_to_multiple<IndexType>(OutputDimensions, 32);

    using OutputBuffer = OutputType[PaddedOutputDimensions];

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t get_hash_value(std::uint32_t prevHash) {
        std::uint32_t hashValue = 0x538D24C7u;
        hashValue += prevHash;
        return hashValue;
    }

    // Read network parameters
    bool read_parameters(std::istream&) { return true; }

    // Write network parameters
    bool write_parameters(std::ostream&) const { return true; }

    std::size_t get_content_hash() const {
        std::size_t h = 0;
        hash_combine(h, get_hash_value(0));
        return h;
    }

    // Forward propagation
    void propagate(const InputType* input, OutputType* output) const {
        static_assert(WeightScaleBitsLocal >= 5 && WeightScaleBitsLocal <= 8,
                      "SqrClippedReLU only supports WeightScaleBitsLocal between 5 and 8");
        // After squaring we need to shift by WeightScaleBitsLocal * 2 + 7.
        // MulHi strips the lower 16 bits (a shift by 16), so we shift out the rest.
        [[maybe_unused]] constexpr int SimdShiftAmount = WeightScaleBitsLocal * 2 + 7 - 16;

#if defined(USE_SSE2)
        constexpr IndexType NumChunks = InputDimensions / 16;

        const auto in  = reinterpret_cast<const __m128i*>(input);
        const auto out = reinterpret_cast<__m128i*>(output);
        for (IndexType i = 0; i < NumChunks; ++i)
        {
            __m128i words0 =
              _mm_packs_epi32(_mm_load_si128(&in[i * 4 + 0]), _mm_load_si128(&in[i * 4 + 1]));
            __m128i words1 =
              _mm_packs_epi32(_mm_load_si128(&in[i * 4 + 2]), _mm_load_si128(&in[i * 4 + 3]));

            words0 = _mm_srli_epi16(_mm_mulhi_epi16(words0, words0), SimdShiftAmount);
            words1 = _mm_srli_epi16(_mm_mulhi_epi16(words1, words1), SimdShiftAmount);

            _mm_store_si128(&out[i], _mm_packs_epi16(words0, words1));
        }
        constexpr IndexType Start = NumChunks * 16;

#elif defined(USE_NEON)
        constexpr IndexType NumChunks = InputDimensions / 16;
        const auto          in        = reinterpret_cast<const int32x4_t*>(input);
        const auto          out       = reinterpret_cast<int8x16_t*>(output);
        for (IndexType i = 0; i < NumChunks; ++i)
        {
            const int16x8_t words0 =
              vcombine_s16(vqmovn_s32(in[i * 4 + 0]), vqmovn_s32(in[i * 4 + 1]));
            const int16x8_t words1 =
              vcombine_s16(vqmovn_s32(in[i * 4 + 2]), vqmovn_s32(in[i * 4 + 3]));

            // vqdmulhq computes (2 * a * b) >> 16, so shift out one more bit to
            // get the same total shift as the SSE2 code above
            const int16x8_t r0 = vshrq_n_s16(vqdmulhq_s16(words0, words0), SimdShiftAmount + 1);
            const int16x8_t r1 = vshrq_n_s16(vqdmulhq_s16(words1, words1), SimdShiftAmount + 1);

            out[i] = vcombine_s8(vqmovn_s16(r0), vqmovn_s16(r1));
        }
        constexpr IndexType Start = NumChunks * 16;

#else
        constexpr IndexType Start = 0;
#endif

        for (IndexType i = Start; i < InputDimensions; ++i)
        {
            output[i] = static_cast<OutputType>(
              // Really should be /127 but we need to make it fast so we right-shift
              // by an extra 7 bits instead. Needs to be accounted for in the trainer.
              std::min(127ll,
                       ((long long) (input[i]) * input[i]) >> (2 * WeightScaleBitsLocal + 7)));
        }
    }
};

}  // namespace Maskfish::Eval::NNUE::Layers

#endif  // NNUE_LAYERS_SQR_CLIPPED_RELU_H_INCLUDED
