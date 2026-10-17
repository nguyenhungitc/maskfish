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

// Clipped ReLU
template<IndexType InDims>
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

#if defined(USE_PAIR_ACTIVATIONS)
    // Produce the squared and linear clipped activations together, sharing the
    // input loads and the saturating narrowing of the inputs to 16 bits (the
    // linear ones are those of ClippedReLU). No shuffle is needed to fix up the
    // pack lane order because the next layer's weights were already reverse
    // pre-permuted to match by AffineTransform::get_weight_index_scrambled().
    void propagate_pair(const InputType* input, OutputType* squared, OutputType* clipped) const {
        static_assert(WeightScaleBits == 6);
        static_assert(InputDimensions % 32 == 0, "propagate_pair() needs a tail path");

        constexpr IndexType NumChunks = InputDimensions / 32;
        // (x * x) >> (2 * WeightScaleBits + 7): mulhi shifts by 16
        constexpr int SqrShift = 2 * WeightScaleBits + 7 - 16;

    #if defined(USE_PAIR_ACTIVATIONS_512)
        const auto in      = reinterpret_cast<const __m512i*>(input);
        auto       sqrOut  = reinterpret_cast<__m256i*>(squared);
        auto       clipOut = reinterpret_cast<__m256i*>(clipped);

        const __m512i zero = _mm512_setzero_si512();

        for (IndexType i = 0; i < NumChunks; ++i)
        {
            const __m512i words    = _mm512_packs_epi32(_mm512_load_si512(&in[i * 2 + 0]),
                                                        _mm512_load_si512(&in[i * 2 + 1]));
            const __m512i sqrWords = _mm512_srli_epi16(_mm512_mulhi_epi16(words, words), SqrShift);
            _mm256_store_si256(&sqrOut[i], _mm512_cvtsepi16_epi8(sqrWords));

            const __m512i clipWords =
              _mm512_srli_epi16(_mm512_max_epi16(words, zero), WeightScaleBits);
            _mm256_store_si256(&clipOut[i], _mm512_cvtsepi16_epi8(clipWords));
        }

    #else
        const auto in      = reinterpret_cast<const __m256i*>(input);
        auto       sqrOut  = reinterpret_cast<__m256i*>(squared);
        auto       clipOut = reinterpret_cast<__m256i*>(clipped);

        const __m256i zero = _mm256_setzero_si256();

        for (IndexType i = 0; i < NumChunks; ++i)
        {
            const __m256i words0 = _mm256_packs_epi32(_mm256_load_si256(&in[i * 4 + 0]),
                                                      _mm256_load_si256(&in[i * 4 + 1]));
            const __m256i words1 = _mm256_packs_epi32(_mm256_load_si256(&in[i * 4 + 2]),
                                                      _mm256_load_si256(&in[i * 4 + 3]));

            const __m256i sqr0 = _mm256_srli_epi16(_mm256_mulhi_epi16(words0, words0), SqrShift);
            const __m256i sqr1 = _mm256_srli_epi16(_mm256_mulhi_epi16(words1, words1), SqrShift);
            _mm256_store_si256(&sqrOut[i], _mm256_packs_epi16(sqr0, sqr1));

            const __m256i clip0 =
              _mm256_srli_epi16(_mm256_max_epi16(words0, zero), WeightScaleBits);
            const __m256i clip1 =
              _mm256_srli_epi16(_mm256_max_epi16(words1, zero), WeightScaleBits);
            _mm256_store_si256(&clipOut[i], _mm256_packs_epi16(clip0, clip1));
        }
    #endif
    }

#else
    // Forward propagation
    void propagate(const InputType* input, OutputType* output) const {

    #if defined(USE_SSE2)
        constexpr IndexType NumChunks = InputDimensions / 16;

        static_assert(WeightScaleBits == 6);
        const auto in  = reinterpret_cast<const __m128i*>(input);
        const auto out = reinterpret_cast<__m128i*>(output);
        for (IndexType i = 0; i < NumChunks; ++i)
        {
            __m128i words0 =
              _mm_packs_epi32(_mm_load_si128(&in[i * 4 + 0]), _mm_load_si128(&in[i * 4 + 1]));
            __m128i words1 =
              _mm_packs_epi32(_mm_load_si128(&in[i * 4 + 2]), _mm_load_si128(&in[i * 4 + 3]));

            // We shift by WeightScaleBits * 2 = 12 and divide by 128
            // which is an additional shift-right of 7, meaning 19 in total.
            // MulHi strips the lower 16 bits so we need to shift out 3 more to match.
            words0 = _mm_srli_epi16(_mm_mulhi_epi16(words0, words0), 3);
            words1 = _mm_srli_epi16(_mm_mulhi_epi16(words1, words1), 3);

            _mm_store_si128(&out[i], _mm_packs_epi16(words0, words1));
        }
        constexpr IndexType Start = NumChunks * 16;

    #elif defined(USE_NEON)
        static_assert(WeightScaleBits == 6);
        constexpr IndexType NumChunks = InputDimensions / 16;
        const auto          in        = reinterpret_cast<const int32x4_t*>(input);
        const auto          out       = reinterpret_cast<int8x16_t*>(output);
        for (IndexType i = 0; i < NumChunks; ++i)
        {
            const int16x8_t words0 =
              vcombine_s16(vqmovn_s32(in[i * 4 + 0]), vqmovn_s32(in[i * 4 + 1]));
            const int16x8_t words1 =
              vcombine_s16(vqmovn_s32(in[i * 4 + 2]), vqmovn_s32(in[i * 4 + 3]));

            // vqdmulhq computes (2 * a * b) >> 16, so shift out 4 more bits to
            // get the same total shift of 19 as the SSE2 code above
            const int16x8_t r0 = vshrq_n_s16(vqdmulhq_s16(words0, words0), 4);
            const int16x8_t r1 = vshrq_n_s16(vqdmulhq_s16(words1, words1), 4);

            out[i] = vcombine_s8(vqmovn_s16(r0), vqmovn_s16(r1));
        }
        constexpr IndexType Start = NumChunks * 16;

    #elif defined(USE_LSX)
        // 128 bits also with LASX, so that the outputs stay in order
        static_assert(WeightScaleBits == 6);
        constexpr IndexType NumChunks = InputDimensions / 16;
        const auto          in        = reinterpret_cast<const __m128i*>(input);
        const auto          out       = reinterpret_cast<__m128i*>(output);
        for (IndexType i = 0; i < NumChunks; ++i)
        {
            // Signed saturation to 16 bits, high half of the square (a shift of
            // 16), shift of 3 more and signed saturation to 8 bits, as with SSE2
            const __m128i words0 = __lsx_vssrani_h_w(in[i * 4 + 1], in[i * 4 + 0], 0);
            const __m128i words1 = __lsx_vssrani_h_w(in[i * 4 + 3], in[i * 4 + 2], 0);
            const __m128i sqr0   = __lsx_vmuh_h(words0, words0);
            const __m128i sqr1   = __lsx_vmuh_h(words1, words1);
            out[i]               = __lsx_vssrlni_b_h(sqr1, sqr0, 3);
        }
        constexpr IndexType Start = NumChunks * 16;

    #elif defined(USE_RVV)
        for (std::size_t j = 0; j < InputDimensions;)
        {
            std::size_t vl = __riscv_vsetvl_e32m4(InputDimensions - j);
            vint32m4_t  in = __riscv_vle32_v_i32m4(&input[j], vl);

            // Saturation to 16 bits, high half of the square (a shift of 16),
            // then a shift of 3 more with saturation to 8 bits, as with SSE2
            vint16m2_t words    = __riscv_vnclip_wx_i16m2(in, 0, __RISCV_VXRM_RDN, vl);
            vint16m2_t sqr      = __riscv_vmulh_vv_i16m2(words, words, vl);
            vint8m1_t  narrowed = __riscv_vnclip_wx_i8m1(sqr, 3, __RISCV_VXRM_RDN, vl);

            __riscv_vse8_v_u8m1(&output[j], __riscv_vreinterpret_v_i8m1_u8m1(narrowed), vl);
            j += vl;
        }
        constexpr IndexType Start = InputDimensions;

    #else
        constexpr IndexType Start = 0;
    #endif

        for (IndexType i = Start; i < InputDimensions; ++i)
        {
            output[i] = static_cast<OutputType>(
              // Really should be /127 but we need to make it fast so we right-shift
              // by an extra 7 bits instead. Needs to be accounted for in the trainer.
              std::min(127ll, ((long long) (input[i]) * input[i]) >> (2 * WeightScaleBits + 7)));
        }
    }
#endif
};

}  // namespace Maskfish::Eval::NNUE::Layers

#endif  // NNUE_LAYERS_SQR_CLIPPED_RELU_H_INCLUDED
