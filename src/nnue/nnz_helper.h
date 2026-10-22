/*
  Maskfish, a UCI jieqi engine
  Copyright (C) 2026 The Maskfish developers (see AUTHORS file)

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

#ifndef NNZ_HELPER_H_INCLUDED
#define NNZ_HELPER_H_INCLUDED

#include <array>
#include <cstdint>

#include "../misc.h"
#include "nnue_common.h"
#include "simd.h"

// Indices of the non-zero 32-bit chunks (4 bytes) of the output of the feature
// transformer. They are recorded by the feature
// transformer while it writes its output (FeatureTransformer::transform()),
// instead of in a second pass over the output by the sparse input layer
// (AffineTransformSparseInput::propagate()). Unlike upstream, they are always
// kept as a list of indices (upstream uses a bitset except with AVX-512 and
// RVV): with the layer stacks of Maskfish, the list is faster on AVX2.

namespace Maskfish::Eval::NNUE {

inline int nnz_popcount(unsigned m) {
#if defined(_MSC_VER)
    return int(__popcnt(m));
#else
    return __builtin_popcount(m);
#endif
}

#if !defined(USE_AVX512) && (defined(USE_SSSE3) || (USE_NEON >= 8) || defined(USE_LSX))
namespace Detail {

constexpr int nnz_lsb(unsigned b) {
    int i = 0;
    while (!(b & 1))
        b >>= 1, ++i;
    return i;
}

// The indices of the set bits of every byte, padded with zeros
struct NNZOffsetIndices {
    alignas(16) u16 offsets[256][8];

    constexpr NNZOffsetIndices() :
        offsets() {
        for (unsigned i = 0; i < 256; ++i)
        {
            unsigned j = i, k = 0;
            while (j)
            {
                offsets[i][k++] = u16(nnz_lsb(j));
                j &= j - 1;
            }
            while (k < 8)
                offsets[i][k++] = 0;
        }
    }
};

alignas(CacheLineSize) inline constexpr NNZOffsetIndices NNZLookup{};

}  // namespace Detail
#endif

template<usize Dimensions>
struct NNZInfo {
    static_assert(Dimensions % 64 == 0);

    // Number of entries of nnz, and indices of the non-zero 32-bit chunks (of
    // the non-zero bytes with RVV)
#if defined(USE_RVV)
    static constexpr usize Granularity = 1;
#else
    static constexpr usize Granularity = 4;
#endif
    unsigned count = 0;
    alignas(64) u16 nnz[Dimensions / Granularity];

#if defined(USE_AVX512)
    #ifdef USE_AVX512ICL
    // _mm512_packs_epi32() interleaves the 128-bit lanes of its two inputs
    alignas(64) static constexpr auto Indices = []() {
        std::array<std::array<u16, 32>, 2> indices{};
        for (int i = 0; i < 2; ++i)
        {
            indices[i] = {0, 1, 2,  3,  16, 17, 18, 19, 4,  5,  6,  7,  20, 21, 22, 23,
                          8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
            for (u16& m : indices[i])
                m += u16(i * Dimensions / 8);
        }
        return indices;
    }();
    #else
    alignas(64) static constexpr auto Indices = []() {
        std::array<std::array<u32, 16>, 2> indices{};
        for (int i = 0; i < 2; ++i)
        {
            indices[i] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
            for (u32& m : indices[i])
                m += u32(i * Dimensions / 8);
        }
        return indices;
    }();
    #endif

    struct NNZCursor {
        NNZInfo& info;
        __m512i  indices;
        unsigned count;

        NNZCursor(NNZInfo& info_, bool perspective, unsigned count_) :
            info(info_),
            count(count_) {
            indices = _mm512_load_si512(&Indices[perspective]);
        }

        void record2(SIMD::vec_t neurons1, SIMD::vec_t neurons2) {
    #if defined(USE_AVX512ICL)
            const __m512i increment = _mm512_set1_epi16(32);

            // Get a bitmask and gather non zero indices
            const __m512i   inputV01 = _mm512_packs_epi32(neurons1, neurons2);
            const __mmask32 nnzMask  = _mm512_test_epi16_mask(inputV01, inputV01);

            // Avoid _mm512_mask_compressstoreu_epi16() as it's 256 uOps on Zen4
            __m512i nnzIndices = _mm512_maskz_compress_epi16(nnzMask, indices);
            _mm512_storeu_si512(info.nnz + count, nnzIndices);

            count += nnz_popcount(nnzMask);
            indices = _mm512_add_epi16(indices, increment);
    #else
            const __m512i increment = _mm512_set1_epi32(16);

            for (auto neurons : {neurons1, neurons2})
            {
                // Get a bitmask and gather non zero indices
                const __mmask16 nnzMask = _mm512_test_epi32_mask(neurons, neurons);
                const __m512i   nnzV    = _mm512_maskz_compress_epi32(nnzMask, indices);
                _mm512_mask_cvtepi32_storeu_epi16(info.nnz + count, 0xFFFF, nnzV);

                count += nnz_popcount(nnzMask);
                indices = _mm512_add_epi32(indices, increment);
            }
    #endif
        }

        ~NNZCursor() { info.count = count; }
    };

    NNZCursor make_cursor(bool perspective) { return {*this, perspective, count}; }

#elif defined(USE_SSSE3) || (USE_NEON >= 8) || defined(USE_LSX)

    struct NNZCursor {
        NNZInfo&       info;
        SIMD::vec128_t base;
        unsigned       count;

        NNZCursor(NNZInfo& info_, bool perspective, unsigned count_) :
            info(info_),
            base(vec128_set_16(u16(perspective * Dimensions / 8))),
            count(count_) {}

        // Appends the indices of the set bits of an 8-bit mask of 8 chunks
        void record8(unsigned mask) {
            using namespace SIMD;
            const vec128_t offsets =
              vec128_load(reinterpret_cast<const vec128_t*>(&Detail::NNZLookup.offsets[mask]));
            vec128_storeu(reinterpret_cast<vec128_t*>(info.nnz + count), vec128_add(base, offsets));
            count += nnz_popcount(mask);
            base = vec128_add(base, vec128_set_16(8));
        }

        // Mask of the non-zero chunks of a vector
        static unsigned nnz_mask(SIMD::vec_t v) {
            using namespace SIMD;
    #if defined(USE_NEON)
            return unsigned(vec_nnz(vreinterpretq_u32_s16(v)));
    #else
            return unsigned(vec_nnz(v));
    #endif
        }

        void record2(SIMD::vec_t neurons1, SIMD::vec_t neurons2) {
            if constexpr (sizeof(SIMD::vec_t) == 32)
            {
                // 8 chunks per vector
                record8(nnz_mask(neurons1));
                record8(nnz_mask(neurons2));
            }
            else
            {
                // 4 chunks per vector
                static_assert(sizeof(SIMD::vec_t) == 16);
                record8(nnz_mask(neurons1) | nnz_mask(neurons2) << 4);
            }
        }

        ~NNZCursor() { info.count = count; }
    };

    NNZCursor make_cursor(bool perspective) { return {*this, perspective, count}; }

#elif defined(USE_RVV)

        // With RVV, FeatureTransformer::transform() records the indices of the
        // non-zero bytes itself, without a cursor

#else

    // No sparse input layer (it computes all the products)
    struct NNZCursor {
    #if defined(VECTOR)
        void record2(SIMD::vec_t, SIMD::vec_t) {}
    #endif
    };

    NNZCursor make_cursor(bool) { return {}; }
#endif
};

}  // namespace Maskfish::Eval::NNUE

#endif  // #ifndef NNZ_HELPER_H_INCLUDED
