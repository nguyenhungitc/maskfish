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

#ifndef NNUE_LAYERS_CLIPPED_RELU_H_INCLUDED
#define NNUE_LAYERS_CLIPPED_RELU_H_INCLUDED

#include <algorithm>
#include <cstdint>
#include <iosfwd>

#include "../nnue_common.h"

namespace Maskfish::Eval::NNUE::Layers {

// Clipped ReLU
template<IndexType InDims>
class ClippedReLU {
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

#if !defined(USE_PAIR_ACTIVATIONS)
    // Forward propagation (with pair activations, see SqrClippedReLU::propagate_pair())
    void propagate(const InputType* input, OutputType* output) const {

    #if defined(USE_SSE2)
        constexpr IndexType NumChunks = InputDimensions / 16;

        #ifndef USE_SSE41
        const __m128i k0x80s = _mm_set1_epi8(-128);
        #endif

        const auto in  = reinterpret_cast<const __m128i*>(input);
        const auto out = reinterpret_cast<__m128i*>(output);
        for (IndexType i = 0; i < NumChunks; ++i)
        {
        #if defined(USE_SSE41)
            const __m128i words0 = _mm_srli_epi16(
              _mm_packus_epi32(_mm_load_si128(&in[i * 4 + 0]), _mm_load_si128(&in[i * 4 + 1])),
              WeightScaleBits);
            const __m128i words1 = _mm_srli_epi16(
              _mm_packus_epi32(_mm_load_si128(&in[i * 4 + 2]), _mm_load_si128(&in[i * 4 + 3])),
              WeightScaleBits);
            _mm_store_si128(&out[i], _mm_packs_epi16(words0, words1));
        #else
            const __m128i words0 = _mm_srai_epi16(
              _mm_packs_epi32(_mm_load_si128(&in[i * 4 + 0]), _mm_load_si128(&in[i * 4 + 1])),
              WeightScaleBits);
            const __m128i words1 = _mm_srai_epi16(
              _mm_packs_epi32(_mm_load_si128(&in[i * 4 + 2]), _mm_load_si128(&in[i * 4 + 3])),
              WeightScaleBits);
            const __m128i packedbytes = _mm_packs_epi16(words0, words1);
            _mm_store_si128(&out[i], _mm_subs_epi8(_mm_adds_epi8(packedbytes, k0x80s), k0x80s));
        #endif
        }
        constexpr IndexType Start = NumChunks * 16;

    #elif defined(USE_NEON)
        constexpr IndexType    NumChunks = InputDimensions / (SimdWidth / 2);
        const SIMD::vec_i8x8_t Zero      = {0};
        const auto             in        = reinterpret_cast<const SIMD::vec_i32x4_t*>(input);
        const auto             out       = reinterpret_cast<SIMD::vec_i8x8_t*>(output);
        for (IndexType i = 0; i < NumChunks; ++i)
        {
            int16x8_t  shifted;
            const auto pack = reinterpret_cast<int16x4_t*>(&shifted);
            pack[0]         = vqshrn_n_s32(in[i * 2 + 0], WeightScaleBits);
            pack[1]         = vqshrn_n_s32(in[i * 2 + 1], WeightScaleBits);
            out[i]          = vmax_s8(vqmovn_s16(shifted), Zero);
        }
        constexpr IndexType Start = NumChunks * (SimdWidth / 2);

    #elif defined(USE_LSX)
        // 128 bits also with LASX, so that the outputs stay in order (the
        // 256-bit packs work within 128-bit lanes)
        constexpr IndexType NumChunks = InputDimensions / 16;
        const auto          in        = reinterpret_cast<const __m128i*>(input);
        const auto          out       = reinterpret_cast<__m128i*>(output);
        for (IndexType i = 0; i < NumChunks; ++i)
        {
            // Unsigned saturation to 16 bits, then shift and signed saturation
            // to 8 bits: clamp(input >> WeightScaleBits, 0, 127)
            const __m128i packed0 = __lsx_vssrani_hu_w(in[i * 4 + 1], in[i * 4 + 0], 0);
            const __m128i packed1 = __lsx_vssrani_hu_w(in[i * 4 + 3], in[i * 4 + 2], 0);
            out[i]                = __lsx_vssrlni_b_h(packed1, packed0, WeightScaleBits);
        }
        constexpr IndexType Start = NumChunks * 16;

    #elif defined(USE_RVV)
        for (std::size_t j = 0; j < InputDimensions;)
        {
            std::size_t vl = __riscv_vsetvl_e32m4(InputDimensions - j);

            vint32m4_t in = __riscv_vle32_v_i32m4(&input[j], vl);
            in            = __riscv_vmax_vx_i32m4(in, 0, vl);

            // Saturating narrowing shifts (rounding down): clamp(input >> 6, 0, 127)
            vint16m2_t words = __riscv_vnclip_wx_i16m2(in, WeightScaleBits, __RISCV_VXRM_RDN, vl);
            vint8m1_t  narrowed = __riscv_vnclip_wx_i8m1(words, 0, __RISCV_VXRM_RDN, vl);

            __riscv_vse8_v_u8m1(&output[j], __riscv_vreinterpret_v_i8m1_u8m1(narrowed), vl);
            j += vl;
        }
        constexpr IndexType Start = InputDimensions;

    #else
        constexpr IndexType Start = 0;
    #endif

        for (IndexType i = Start; i < InputDimensions; ++i)
        {
            output[i] = static_cast<OutputType>(std::clamp(input[i] >> WeightScaleBits, 0, 127));
        }
    }
#endif
};

}  // namespace Maskfish::Eval::NNUE::Layers

#endif  // NNUE_LAYERS_CLIPPED_RELU_H_INCLUDED
