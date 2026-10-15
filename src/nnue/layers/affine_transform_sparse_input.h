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

// Definition of layer AffineTransformSparseInput of NNUE evaluation function

#ifndef NNUE_LAYERS_AFFINE_TRANSFORM_SPARSE_INPUT_H_INCLUDED
#define NNUE_LAYERS_AFFINE_TRANSFORM_SPARSE_INPUT_H_INCLUDED

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>

#include "../../bitboard.h"
#include "../../memory.h"
#include "../simd.h"
#include "../nnue_common.h"
#include "../nnz_helper.h"

/*
  This file contains the definition for a fully connected layer (aka affine transform) with block sparse input.
*/

namespace Maskfish::Eval::NNUE::Layers {

// Sparse input implementation
template<IndexType InDims, IndexType OutDims>
class AffineTransformSparseInput {
   public:
    // Input/output type
    using InputType  = std::uint8_t;
    using OutputType = std::int32_t;

    // Number of input/output dimensions
    static constexpr IndexType InputDimensions  = InDims;
    static constexpr IndexType OutputDimensions = OutDims;

    static_assert(OutputDimensions % 16 == 0,
                  "Only implemented for OutputDimensions divisible by 16.");

    static constexpr IndexType PaddedInputDimensions =
      ceil_to_multiple<IndexType>(InputDimensions, MaxSimdWidth);
    static constexpr IndexType PaddedOutputDimensions =
      ceil_to_multiple<IndexType>(OutputDimensions, MaxSimdWidth);

#if (USE_SSSE3 | (USE_NEON >= 8) | USE_LSX | USE_LASX)
    static constexpr IndexType ChunkSize = 4;
#else
    static constexpr IndexType ChunkSize = 1;
#endif

    using OutputBuffer = OutputType[PaddedOutputDimensions];

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t get_hash_value(std::uint32_t prevHash) {
        std::uint32_t hashValue = 0xCC03DAE4u;
        hashValue += OutputDimensions;
        hashValue ^= prevHash >> 1;
        hashValue ^= prevHash << 31;
        return hashValue;
    }

    static constexpr IndexType get_weight_index_scrambled(IndexType i) {
        return (i / ChunkSize) % (PaddedInputDimensions / ChunkSize) * OutputDimensions * ChunkSize
             + i / PaddedInputDimensions * ChunkSize + i % ChunkSize;
    }

    static constexpr IndexType get_weight_index(IndexType i) {
#if (USE_SSSE3 | (USE_NEON >= 8) | USE_LSX | USE_LASX | USE_RVV)
        return get_weight_index_scrambled(i);
#else
        return i;
#endif
    }

    // Read network parameters
    bool read_parameters(std::istream& stream) {
        read_little_endian<BiasType>(stream, biases, OutputDimensions);
        for (IndexType i = 0; i < OutputDimensions * PaddedInputDimensions; ++i)
            weights[get_weight_index(i)] = read_little_endian<WeightType>(stream);

        return !stream.fail();
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) const {
        write_little_endian<BiasType>(stream, biases, OutputDimensions);

        for (IndexType i = 0; i < OutputDimensions * PaddedInputDimensions; ++i)
            write_little_endian<WeightType>(stream, weights[get_weight_index(i)]);

        return !stream.fail();
    }

    std::size_t get_content_hash() const {
        std::size_t h = 0;
        hash_combine(h, get_raw_data_hash(biases));
        hash_combine(h, get_raw_data_hash(weights));
        hash_combine(h, get_hash_value(0));
        return h;
    }

    // Forward propagation
    // nnzInfo holds the indices of the non-zero 32-bit chunks of the input,
    // recorded by the feature transformer
    void propagate(const InputType*                        input,
                   OutputType*                             output,
                   [[maybe_unused]] const NNZInfo<InDims>& nnzInfo) const {

#if (USE_SSSE3 | (USE_NEON >= 8) | USE_LSX | USE_LASX)
    #if defined(USE_AVX512)
        using invec_t  = __m512i;
        using outvec_t = __m512i;
        #define vec_add_32 _mm512_add_epi32
        #define vec_set_32 _mm512_set1_epi32
        #define vec_add_dpbusd_32 SIMD::m512_add_dpbusd_epi32
    #elif defined(USE_AVX2)
        using invec_t  = __m256i;
        using outvec_t = __m256i;
        #define vec_add_32 _mm256_add_epi32
        #define vec_set_32 _mm256_set1_epi32
        #define vec_add_dpbusd_32 SIMD::m256_add_dpbusd_epi32
    #elif defined(USE_SSSE3)
        using invec_t  = __m128i;
        using outvec_t = __m128i;
        #define vec_set_32 _mm_set1_epi32
        #define vec_add_dpbusd_32 SIMD::m128_add_dpbusd_epi32
    #elif defined(USE_NEON_DOTPROD)
        using invec_t  = int8x16_t;
        using outvec_t = int32x4_t;
        #define vec_set_32(a) vreinterpretq_s8_u32(vdupq_n_u32(a))
        #define vec_add_dpbusd_32 SIMD::dotprod_m128_add_dpbusd_epi32
    #elif defined(USE_NEON)
        using invec_t  = int8x16_t;
        using outvec_t = int32x4_t;
        #define vec_set_32(a) vreinterpretq_s8_u32(vdupq_n_u32(a))
        #define vec_add_dpbusd_32 SIMD::neon_m128_add_dpbusd_epi32
    #elif defined(USE_LASX)
        using invec_t  = __m256i;
        using outvec_t = __m256i;
        #define vec_set_32 __lasx_xvreplgr2vr_w
        #define vec_add_dpbusd_32 SIMD::lasx_m256_add_dpbusd_epi32
    #elif defined(USE_LSX)
        using invec_t  = __m128i;
        using outvec_t = __m128i;
        #define vec_set_32 __lsx_vreplgr2vr_w
        #define vec_add_dpbusd_32 SIMD::lsx_m128_add_dpbusd_epi32
    #endif
        constexpr IndexType OutputSimdWidth = sizeof(outvec_t) / sizeof(OutputType);
        constexpr IndexType NumAccums = OutputDimensions / OutputSimdWidth;
        // If we're using high-latency dot product instructions, split the accumulators
        // into separate dependency chains and merge at the end
        constexpr IndexType NumRegs =
    #if defined(USE_AVXVNNI)
          2 * NumAccums;
    #elif defined(USE_VNNI) || defined(USE_NEON_DOTPROD)
          3 * NumAccums;
    #else
          NumAccums;
    #endif
        const outvec_t* biasvec = reinterpret_cast<const outvec_t*>(biases);
        outvec_t        acc[NumRegs];
        for (IndexType k = 0; k < NumAccums; ++k)
            acc[k] = biasvec[k];

        const auto* start = nnzInfo.nnz;
        const auto* end   = nnzInfo.nnz + nnzInfo.count;

        // convince GCC to not do weird pointer arithmetic in the following loop
        const std::int8_t* weights_cp = weights;
    #if defined(USE_AVXVNNI)
        for (IndexType k = NumAccums; k < NumRegs; ++k)
            acc[k] = vec_zero();

        while (end - start >= 2)
        {
            const std::ptrdiff_t i0 = *start++;
            const std::ptrdiff_t i1 = *start++;
            const invec_t        in0 =
              vec_set_32(load_as<std::int32_t>(input + i0 * sizeof(std::int32_t)));
            const invec_t in1 =
              vec_set_32(load_as<std::int32_t>(input + i1 * sizeof(std::int32_t)));
            const auto col0 =
              reinterpret_cast<const invec_t*>(&weights_cp[i0 * OutputDimensions * ChunkSize]);
            const auto col1 =
              reinterpret_cast<const invec_t*>(&weights_cp[i1 * OutputDimensions * ChunkSize]);
            for (IndexType k = 0; k < NumAccums; ++k)
            {
                vec_add_dpbusd_32(acc[k], in0, col0[k]);
                vec_add_dpbusd_32(acc[k + NumAccums], in1, col1[k]);
            }
        }
        for (IndexType k = 0; k < NumAccums; ++k)
            acc[k] = vec_add_32(acc[k], acc[k + NumAccums]);
    #elif defined(USE_VNNI) || defined(USE_NEON_DOTPROD)
        #if defined(USE_VNNI)
        for (IndexType k = NumAccums; k < NumRegs; ++k)
            acc[k] = vec_zero();
        #else
        for (IndexType k = NumAccums; k < NumRegs; ++k)
            acc[k] = vdupq_n_s32(0);
        #endif

        while (start < end - 2)
        {
            const std::ptrdiff_t i0 = *start++;
            const std::ptrdiff_t i1 = *start++;
            const std::ptrdiff_t i2 = *start++;
            const invec_t        in0 =
              vec_set_32(load_as<std::int32_t>(input + i0 * sizeof(std::int32_t)));
            const invec_t in1 =
              vec_set_32(load_as<std::int32_t>(input + i1 * sizeof(std::int32_t)));
            const invec_t in2 =
              vec_set_32(load_as<std::int32_t>(input + i2 * sizeof(std::int32_t)));
            const auto col0 =
              reinterpret_cast<const invec_t*>(&weights_cp[i0 * OutputDimensions * ChunkSize]);
            const auto col1 =
              reinterpret_cast<const invec_t*>(&weights_cp[i1 * OutputDimensions * ChunkSize]);
            const auto col2 =
              reinterpret_cast<const invec_t*>(&weights_cp[i2 * OutputDimensions * ChunkSize]);
            for (IndexType k = 0; k < NumAccums; ++k)
            {
                vec_add_dpbusd_32(acc[k], in0, col0[k]);
                vec_add_dpbusd_32(acc[k + NumAccums], in1, col1[k]);
                vec_add_dpbusd_32(acc[k + 2 * NumAccums], in2, col2[k]);
            }
        }
        #if defined(USE_VNNI)
        for (IndexType k = 0; k < NumAccums; ++k)
            acc[k] = vec_add_32(vec_add_32(acc[k], acc[k + NumAccums]), acc[k + 2 * NumAccums]);
        #else
        for (IndexType k = 0; k < NumAccums; ++k)
            acc[k] = vaddq_s32(vaddq_s32(acc[k], acc[k + NumAccums]), acc[k + 2 * NumAccums]);
        #endif
    #endif
        while (start < end)
        {
            const std::ptrdiff_t i = *start++;
            const invec_t in = vec_set_32(load_as<std::int32_t>(input + i * sizeof(std::int32_t)));
            const auto    col =
              reinterpret_cast<const invec_t*>(&weights_cp[i * OutputDimensions * ChunkSize]);
            for (IndexType k = 0; k < NumAccums; ++k)
                vec_add_dpbusd_32(acc[k], in, col[k]);
        }

        outvec_t* outptr = reinterpret_cast<outvec_t*>(output);
        for (IndexType k = 0; k < NumAccums; ++k)
            outptr[k] = acc[k];

    #undef vec_set_32
    #undef vec_add_dpbusd_32
    #ifdef vec_add_32
        #undef vec_add_32
    #endif
#elif defined(USE_RVV)
        // The weights of an input are contiguous (ChunkSize 1), and nnzInfo has
        // the indices of the non-zero input bytes. We don't support larger
        // layers for now.
        static_assert(OutputDimensions <= 128 * 8 / 32);

    #define RVV_SPARSE_PROPAGATE(m1, m2, m4) \
        do \
        { \
            std::size_t    vl  = OutputDimensions; \
            vint32##m4##_t acc = __riscv_vle32_v_i32##m4(biases, vl); \
            IndexType      i   = 0; \
            for (; i + 2 <= nnzInfo.count; i += 2) \
            { \
                std::size_t   idx0 = nnzInfo.nnz[i + 0]; \
                std::size_t   idx1 = nnzInfo.nnz[i + 1]; \
                std::uint8_t  in0  = input[idx0]; \
                std::uint8_t  in1  = input[idx1]; \
                vint8##m1##_t w0   = __riscv_vle8_v_i8##m1(&weights[idx0 * OutputDimensions], vl); \
                vint8##m1##_t w1   = __riscv_vle8_v_i8##m1(&weights[idx1 * OutputDimensions], vl); \
                /* the inputs are in [0, 127] and the weights in [-128, 127], \
                 * so two products can be accumulated in 16 bits */ \
                vint16##m2##_t prod = __riscv_vwmulsu(w0, in0, vl); \
                prod                = __riscv_vwmaccus(prod, in1, w1, vl); \
                acc                 = __riscv_vwadd_wv(acc, prod, vl); \
            } \
            if (i < nnzInfo.count) \
            { \
                std::size_t    idx  = nnzInfo.nnz[i]; \
                std::uint8_t   in   = input[idx]; \
                vint8##m1##_t  w    = __riscv_vle8_v_i8##m1(&weights[idx * OutputDimensions], vl); \
                vint16##m2##_t prod = __riscv_vwmulsu(w, in, vl); \
                acc                 = __riscv_vwadd_wv(acc, prod, vl); \
            } \
            __riscv_vse32(output, acc, vl); \
        } while (0)

        // Select LMUL
        const std::size_t VL1 = __riscv_vsetvlmax_e32m1();
        if (VL1 >= OutputDimensions)
            RVV_SPARSE_PROPAGATE(mf4, mf2, m1);
        else if (VL1 * 2 >= OutputDimensions)
            RVV_SPARSE_PROPAGATE(mf2, m1, m2);
        else if (VL1 * 4 >= OutputDimensions)
            RVV_SPARSE_PROPAGATE(m1, m2, m4);
        else
            RVV_SPARSE_PROPAGATE(m2, m4, m8);

    #undef RVV_SPARSE_PROPAGATE
#else
        // Use dense implementation for the other architectures.
        affine_transform_non_ssse3<InputDimensions, PaddedInputDimensions, OutputDimensions>(
          output, weights, biases, input);
#endif
    }

   private:
    using BiasType   = OutputType;
    using WeightType = std::int8_t;

    alignas(CacheLineSize) BiasType biases[OutputDimensions];
    alignas(CacheLineSize) WeightType weights[OutputDimensions * PaddedInputDimensions];
};

}  // namespace Maskfish::Eval::NNUE::Layers

#endif  // #ifndef NNUE_LAYERS_AFFINE_TRANSFORM_SPARSE_INPUT_H_INCLUDED
