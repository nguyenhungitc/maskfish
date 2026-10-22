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

#ifndef NNUE_SIMD_H_INCLUDED
#define NNUE_SIMD_H_INCLUDED

#if defined(USE_AVX2)
    #include <immintrin.h>

#elif defined(USE_SSE41)
    #include <smmintrin.h>

#elif defined(USE_SSSE3)
    #include <tmmintrin.h>

#elif defined(USE_SSE2)
    #include <emmintrin.h>

#elif defined(USE_NEON)
    #include <arm_neon.h>

#elif defined(USE_LASX)
    #include <lasxintrin.h>
    #include <lsxintrin.h>

#elif defined(USE_LSX)
    #include <lsxintrin.h>

#elif defined(USE_RVV)
    // The RISC-V vectors have no fixed size: VECTOR is not defined, the
    // feature transformer and the layers have their own RVV code
    #include <riscv_vector.h>
#endif

#include "../types.h"
#include "nnue_common.h"

namespace Maskfish::Eval::NNUE::SIMD {

// If vector instructions are enabled, we update and refresh the
// accumulator tile by tile such that each tile fits in the CPU's
// vector registers.
#define VECTOR

#ifdef USE_AVX512
using vec_t      = __m512i;
using psqt_vec_t = __m512i;
using vec_uint_t = __m512i;
    #define vec_load(a) _mm512_load_si512(a)
    #define vec_store(a, b) _mm512_store_si512(a, b)
    #define vec_zero() _mm512_setzero_epi32()
    #define vec_set_16(a) _mm512_set1_epi16(a)
    #define vec_load_psqt(a) _mm512_load_si512(a)
    #define vec_store_psqt(a, b) _mm512_store_si512(a, b)
    #define vec_add_psqt_32(a, b) _mm512_add_epi32(a, b)
    #define vec_sub_psqt_32(a, b) _mm512_sub_epi32(a, b)
    #define vec_zero_psqt() _mm512_setzero_epi32()

    #ifdef USE_AVX512BW
        #define vec_add_16(a, b) _mm512_add_epi16(a, b)
        #define vec_adds_16(a, b) _mm512_adds_epi16(a, b)
        #define vec_sub_16(a, b) _mm512_sub_epi16(a, b)
        #define vec_mulhi_16(a, b) _mm512_mulhi_epi16(a, b)
        #define vec_max_16(a, b) _mm512_max_epi16(a, b)
        #define vec_min_16(a, b) _mm512_min_epi16(a, b)
        #define vec_slli_16(a, b) _mm512_slli_epi16(a, b)
        // Inverse permuted at load time
        #define vec_packus_16(a, b) _mm512_packus_epi16(a, b)
    #else
        #define vec_op(op, a, b) \
            __builtin_shufflevector(op(__builtin_shufflevector(a, a, 0, 1, 2, 3), \
                                       __builtin_shufflevector(b, b, 0, 1, 2, 3)), \
                                    op(__builtin_shufflevector(a, a, 4, 5, 6, 7), \
                                       __builtin_shufflevector(b, b, 4, 5, 6, 7)), \
                                    0, 1, 2, 3, 4, 5, 6, 7)
        #define vec_add_16(a, b) vec_op(_mm256_add_epi16, a, b)
        #define vec_adds_16(a, b) vec_op(_mm256_adds_epi16, a, b)
        #define vec_sub_16(a, b) vec_op(_mm256_sub_epi16, a, b)
        #define vec_mulhi_16(a, b) vec_op(_mm256_mulhi_epi16, a, b)
        #define vec_max_16(a, b) vec_op(_mm256_max_epi16, a, b)
        #define vec_min_16(a, b) vec_op(_mm256_min_epi16, a, b)
        #define vec_slli_16(a, b) \
            __builtin_shufflevector( \
              _mm256_slli_epi16(__builtin_shufflevector(a, a, 0, 1, 2, 3), b), \
              _mm256_slli_epi16(__builtin_shufflevector(a, a, 4, 5, 6, 7), b), 0, 1, 2, 3, 4, 5, \
              6, 7)
        // Inverse permuted at load time
        #define vec_packus_16(a, b) vec_op(_mm256_packus_epi16, a, b)
    #endif

    #define NumRegistersSIMD 16
    #define MaxChunkSize 64

#elif USE_AVX2
using vec_t      = __m256i;
using vec128_t   = __m128i;
using psqt_vec_t = __m256i;
using vec_uint_t = __m256i;
    #define vec_load(a) _mm256_load_si256(a)
    #define vec_store(a, b) _mm256_store_si256(a, b)
    #define vec_add_16(a, b) _mm256_add_epi16(a, b)
    #define vec_adds_16(a, b) _mm256_adds_epi16(a, b)
    #define vec_sub_16(a, b) _mm256_sub_epi16(a, b)
    #define vec_mulhi_16(a, b) _mm256_mulhi_epi16(a, b)
    #define vec_zero() _mm256_setzero_si256()
    #define vec_set_16(a) _mm256_set1_epi16(a)
    #define vec_max_16(a, b) _mm256_max_epi16(a, b)
    #define vec_min_16(a, b) _mm256_min_epi16(a, b)
    #define vec_slli_16(a, b) _mm256_slli_epi16(a, b)
    // Inverse permuted at load time
    #define vec_packus_16(a, b) _mm256_packus_epi16(a, b)
    #define vec_load_psqt(a) _mm256_load_si256(a)
    #define vec_store_psqt(a, b) _mm256_store_si256(a, b)
    #define vec_add_psqt_32(a, b) _mm256_add_epi32(a, b)
    #define vec_sub_psqt_32(a, b) _mm256_sub_epi32(a, b)
    #define vec_zero_psqt() _mm256_setzero_si256()

    #ifdef USE_SSSE3
        #if defined(USE_VNNI) && !defined(USE_AVXVNNI)
            #define vec_nnz(a) _mm256_cmpgt_epi32_mask(a, _mm256_setzero_si256())
        #else
            #define vec_nnz(a) \
                _mm256_movemask_ps( \
                  _mm256_castsi256_ps(_mm256_cmpgt_epi32(a, _mm256_setzero_si256())))
        #endif
    #endif

    #define vec128_zero _mm_setzero_si128()
    #define vec128_set_16(a) _mm_set1_epi16(a)
    #define vec128_load(a) _mm_load_si128(a)
    #define vec128_storeu(a, b) _mm_storeu_si128(a, b)
    #define vec128_add(a, b) _mm_add_epi16(a, b)

    #define NumRegistersSIMD 16
    #define MaxChunkSize 32

#elif USE_SSE2
using vec_t      = __m128i;
using vec128_t   = __m128i;
using psqt_vec_t = __m128i;
using vec_uint_t = __m128i;
    #define vec_load(a) (*(a))
    #define vec_store(a, b) *(a) = (b)
    #define vec_add_16(a, b) _mm_add_epi16(a, b)
    #define vec_adds_16(a, b) _mm_adds_epi16(a, b)
    #define vec_sub_16(a, b) _mm_sub_epi16(a, b)
    #define vec_mulhi_16(a, b) _mm_mulhi_epi16(a, b)
    #define vec_zero() _mm_setzero_si128()
    #define vec_set_16(a) _mm_set1_epi16(a)
    #define vec_max_16(a, b) _mm_max_epi16(a, b)
    #define vec_min_16(a, b) _mm_min_epi16(a, b)
    #define vec_slli_16(a, b) _mm_slli_epi16(a, b)
    #define vec_packus_16(a, b) _mm_packus_epi16(a, b)
    #define vec_load_psqt(a) (*(a))
    #define vec_store_psqt(a, b) *(a) = (b)
    #define vec_add_psqt_32(a, b) _mm_add_epi32(a, b)
    #define vec_sub_psqt_32(a, b) _mm_sub_epi32(a, b)
    #define vec_zero_psqt() _mm_setzero_si128()

    #ifdef USE_SSSE3
        #define vec_nnz(a) \
            _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(a, _mm_setzero_si128())))
    #endif

    #define vec128_zero _mm_setzero_si128()
    #define vec128_set_16(a) _mm_set1_epi16(a)
    #define vec128_load(a) _mm_load_si128(a)
    #define vec128_storeu(a, b) _mm_storeu_si128(a, b)
    #define vec128_add(a, b) _mm_add_epi16(a, b)

    #define NumRegistersSIMD (Is64Bit ? 16 : 8)
    #define MaxChunkSize 16

#elif USE_NEON
using vec_i8x8_t __attribute__((may_alias))  = int8x8_t;
using vec_i16x8_t __attribute__((may_alias)) = int16x8_t;
using vec_i8x16_t __attribute__((may_alias)) = int8x16_t;
using vec_u16x8_t __attribute__((may_alias)) = uint16x8_t;
using vec_i32x4_t __attribute__((may_alias)) = int32x4_t;

using vec_t __attribute__((may_alias))      = int16x8_t;
using vec_i8_t __attribute__((may_alias))   = int8x16_t;
using psqt_vec_t __attribute__((may_alias)) = int32x4_t;
using vec128_t __attribute__((may_alias))   = uint16x8_t;
using vec_uint_t __attribute__((may_alias)) = uint32x4_t;
    #define vec_load(a) (*(a))
    #define vec_store(a, b) *(a) = (b)
    #define vec_add_16(a, b) vaddq_s16(a, b)
    #define vec_adds_16(a, b) vqaddq_s16(a, b)
    #define vec_sub_16(a, b) vsubq_s16(a, b)
    #define vec_mulhi_16(a, b) vqdmulhq_s16(a, b)
    #define vec_zero() vec_t{0}
    #define vec_set_16(a) vdupq_n_s16(a)
    #define vec_max_16(a, b) vmaxq_s16(a, b)
    #define vec_min_16(a, b) vminq_s16(a, b)
    #define vec_slli_16(a, b) vshlq_s16(a, vec_set_16(b))
    #define vec_packus_16(a, b) reinterpret_cast<vec_t>(vcombine_u8(vqmovun_s16(a), vqmovun_s16(b)))
    #define vec_load_psqt(a) (*(a))
    #define vec_store_psqt(a, b) *(a) = (b)
    #define vec_add_psqt_32(a, b) vaddq_s32(a, b)
    #define vec_sub_psqt_32(a, b) vsubq_s32(a, b)
    #define vec_zero_psqt() psqt_vec_t{0}

static constexpr std::uint32_t Mask[4] = {1, 2, 4, 8};
    #define vec_nnz(a) vaddvq_u32(vandq_u32(vtstq_u32(a, a), vld1q_u32(Mask)))
    #define vec128_zero vdupq_n_u16(0)
    #define vec128_set_16(a) vdupq_n_u16(a)
    #define vec128_load(a) vld1q_u16(reinterpret_cast<const std::uint16_t*>(a))
    #define vec128_storeu(a, b) vst1q_u16(reinterpret_cast<std::uint16_t*>(a), b)
    #define vec128_add(a, b) vaddq_u16(a, b)

    #define NumRegistersSIMD 16
    #define MaxChunkSize 16

#elif USE_LASX
// LoongArch Advanced SIMD eXtension (256 bits; the LSX instructions work on
// the low 128 bits of its registers)
using vec_t      = __m256i;
using vec128_t   = __m128i;
using psqt_vec_t = __m256i;
using vec_uint_t = __m256i;

inline __m256i lasx_packus_16(__m256i a, __m256i b) {
    #if defined(__clang__) && defined(__has_builtin) && __has_builtin(__builtin_lasx_xvssrani_bu_h)
    return (__m256i) __builtin_lasx_xvssrani_bu_h((v32i8) b, (v32i8) a, 0);
    #else
    return __lasx_xvssrani_bu_h(b, a, 0);
    #endif
}

inline int lasx_vec_nnz(__m256i a) {
    const __m256i cmp = __lasx_xvslt_w(__lasx_xvldi(0), a);
    const __m256i msk = __lasx_xvmskltz_w(cmp);
    return int(__lasx_xvpickve2gr_w(msk, 0)) | (int(__lasx_xvpickve2gr_w(msk, 4)) << 4);
}

    #define vec_load(a) __lasx_xvld(reinterpret_cast<const void*>(a), 0)
    #define vec_store(a, b) __lasx_xvst(b, reinterpret_cast<void*>(a), 0)
    #define vec_add_16(a, b) __lasx_xvadd_h(a, b)
    #define vec_adds_16(a, b) __lasx_xvsadd_h(a, b)
    #define vec_sub_16(a, b) __lasx_xvsub_h(a, b)
    #define vec_mulhi_16(a, b) __lasx_xvmuh_h(a, b)
    #define vec_zero() __lasx_xvldi(0)
    #define vec_set_16(a) __lasx_xvreplgr2vr_h(a)
    #define vec_max_16(a, b) __lasx_xvmax_h(a, b)
    #define vec_min_16(a, b) __lasx_xvmin_h(a, b)
    #define vec_slli_16(a, b) __lasx_xvslli_h(a, b)
    // Inverse permuted at load time (packs within 128-bit lanes, like AVX2)
    #define vec_packus_16(a, b) lasx_packus_16(a, b)
    #define vec_load_psqt(a) __lasx_xvld(reinterpret_cast<const void*>(a), 0)
    #define vec_store_psqt(a, b) __lasx_xvst(b, reinterpret_cast<void*>(a), 0)
    #define vec_add_psqt_32(a, b) __lasx_xvadd_w(a, b)
    #define vec_sub_psqt_32(a, b) __lasx_xvsub_w(a, b)
    #define vec_zero_psqt() __lasx_xvldi(0)
    #define vec_nnz(a) lasx_vec_nnz(a)

    #define vec128_zero __lsx_vldi(0)
    #define vec128_set_16(a) __lsx_vreplgr2vr_h(a)
    #define vec128_load(a) __lsx_vld(reinterpret_cast<const void*>(a), 0)
    #define vec128_storeu(a, b) __lsx_vst(b, reinterpret_cast<void*>(a), 0)
    #define vec128_add(a, b) __lsx_vadd_h(a, b)

    #define NumRegistersSIMD 32
    #define MaxChunkSize 32

#elif USE_LSX
// LoongArch SIMD eXtension (128 bits)
using vec_t      = __m128i;
using vec128_t   = __m128i;
using psqt_vec_t = __m128i;
using vec_uint_t = __m128i;

inline __m128i lsx_packus_16(__m128i a, __m128i b) {
    #if defined(__clang__) && defined(__has_builtin) && __has_builtin(__builtin_lsx_vssrani_bu_h)
    return (__m128i) __builtin_lsx_vssrani_bu_h((v16i8) b, (v16i8) a, 0);
    #else
    return __lsx_vssrani_bu_h(b, a, 0);
    #endif
}

inline int lsx_vec_nnz(__m128i a) {
    const __m128i cmp = __lsx_vslt_w(__lsx_vldi(0), a);
    const __m128i msk = __lsx_vmskltz_w(cmp);
    return int(__lsx_vpickve2gr_w(msk, 0));
}

    #define vec_load(a) (*(a))
    #define vec_store(a, b) *(a) = (b)
    #define vec_add_16(a, b) __lsx_vadd_h(a, b)
    #define vec_adds_16(a, b) __lsx_vsadd_h(a, b)
    #define vec_sub_16(a, b) __lsx_vsub_h(a, b)
    #define vec_mulhi_16(a, b) __lsx_vmuh_h(a, b)
    #define vec_zero() __lsx_vldi(0)
    #define vec_set_16(a) __lsx_vreplgr2vr_h(a)
    #define vec_max_16(a, b) __lsx_vmax_h(a, b)
    #define vec_min_16(a, b) __lsx_vmin_h(a, b)
    #define vec_slli_16(a, b) __lsx_vslli_h(a, b)
    #define vec_packus_16(a, b) lsx_packus_16(a, b)
    #define vec_load_psqt(a) (*(a))
    #define vec_store_psqt(a, b) *(a) = (b)
    #define vec_add_psqt_32(a, b) __lsx_vadd_w(a, b)
    #define vec_sub_psqt_32(a, b) __lsx_vsub_w(a, b)
    #define vec_zero_psqt() __lsx_vldi(0)
    #define vec_nnz(a) lsx_vec_nnz(a)

    #define vec128_zero __lsx_vldi(0)
    #define vec128_set_16(a) __lsx_vreplgr2vr_h(a)
    #define vec128_load(a) __lsx_vld(reinterpret_cast<const void*>(a), 0)
    #define vec128_storeu(a, b) __lsx_vst(b, reinterpret_cast<void*>(a), 0)
    #define vec128_add(a, b) __lsx_vadd_h(a, b)

    #define NumRegistersSIMD 32
    #define MaxChunkSize 16

#else
    #undef VECTOR

#endif

// On x86 with SSSE3 or better (AVX-512 with BW), two rows of 8-bit feature
// transformer weights are widened and summed into 16-bit lanes with one
// unpack and one maddubs (against a vector of ones), instead of one sign
// extension per row (paired int8 weight rows). The weights are permuted at
// load time to match (see
// FeatureTransformer::WideningPairOrder): vec_widen_i8() and
// vec_widen_pair_i8() turn sizeof(vec_t) weights into two consecutive
// vectors of the accumulator.
#if defined(USE_SSSE3) && (!defined(USE_AVX512) || defined(USE_AVX512BW))
    #define USE_PAIRED_WIDENING_ADD

    #if defined(USE_AVX512)
        #define vec_loadu_8(a) _mm512_loadu_si512(reinterpret_cast<const void*>(a))
        #define vec_set_8(a) _mm512_set1_epi8(a)
        #define vec_unpacklo_8(a, b) _mm512_unpacklo_epi8(a, b)
        #define vec_unpackhi_8(a, b) _mm512_unpackhi_epi8(a, b)
        #define vec_maddubs_16(a, b) _mm512_maddubs_epi16(a, b)
        #define vec_srai_16(a, b) _mm512_srai_epi16(a, b)
    #elif defined(USE_AVX2)
        #define vec_loadu_8(a) _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a))
        #define vec_set_8(a) _mm256_set1_epi8(a)
        #define vec_unpacklo_8(a, b) _mm256_unpacklo_epi8(a, b)
        #define vec_unpackhi_8(a, b) _mm256_unpackhi_epi8(a, b)
        #define vec_maddubs_16(a, b) _mm256_maddubs_epi16(a, b)
        #define vec_srai_16(a, b) _mm256_srai_epi16(a, b)
    #else
        #define vec_loadu_8(a) _mm_loadu_si128(reinterpret_cast<const __m128i*>(a))
        #define vec_set_8(a) _mm_set1_epi8(a)
        #define vec_unpacklo_8(a, b) _mm_unpacklo_epi8(a, b)
        #define vec_unpackhi_8(a, b) _mm_unpackhi_epi8(a, b)
        #define vec_maddubs_16(a, b) _mm_maddubs_epi16(a, b)
        #define vec_srai_16(a, b) _mm_srai_epi16(a, b)
    #endif

// The sizeof(vec_t) weights at p, sign-extended to 16 bits: lo then hi
inline void vec_widen_i8(const std::int8_t* p, vec_t& lo, vec_t& hi) {
    const vec_t w = vec_loadu_8(p);
    lo            = vec_srai_16(vec_unpacklo_8(vec_zero(), w), 8);
    hi            = vec_srai_16(vec_unpackhi_8(vec_zero(), w), 8);
}

// The sums of the sizeof(vec_t) weights at p0 and at p1, in 16 bits
inline void vec_widen_pair_i8(const std::int8_t* p0, const std::int8_t* p1, vec_t& lo, vec_t& hi) {
    const vec_t ones = vec_set_8(1);
    const vec_t w0   = vec_loadu_8(p0);
    const vec_t w1   = vec_loadu_8(p1);
    lo               = vec_maddubs_16(ones, vec_unpacklo_8(w0, w1));
    hi               = vec_maddubs_16(ones, vec_unpackhi_8(w0, w1));
}
#endif

#ifdef VECTOR
// Loads sizeof(vec_t) / 2 feature transformer weights of 8 bits and
// sign-extends them to 16-bit lanes
inline vec_t vec_load_i8_16(const std::int8_t* p) {
    #if defined(USE_AVX512)
        #if defined(USE_AVX512BW)
    return _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
        #else
    const __m256i lo =
      _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
    const __m256i hi =
      _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 16)));
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
        #endif
    #elif defined(USE_AVX2)
    return _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
    #elif defined(USE_SSE41)
    return _mm_cvtepi8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
    #elif defined(USE_SSE2)
    const __m128i x = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
    return _mm_srai_epi16(_mm_unpacklo_epi8(x, x), 8);
    #elif defined(USE_NEON)
    return vmovl_s8(vld1_s8(p));
    #elif defined(USE_LASX)
    // Sign extension of the 16 bytes of the low 128 bits
    const __m128i x = __lsx_vld(reinterpret_cast<const void*>(p), 0);
        #if defined(__has_builtin) && __has_builtin(__builtin_lasx_cast_128)
    return __lasx_vext2xv_h_b(__lasx_cast_128(x));
        #elif defined(__GNUC__) && !defined(__clang__)
    __m256i out;
    __asm__("vext2xv.h.b %u0, %u1" : "=f"(out) : "f"(x));
    return out;
        #else
    __m256i v = __lasx_xvldi(0);
    v         = __lasx_xvinsgr2vr_d(v, __lsx_vpickve2gr_d(x, 0), 0);
    v         = __lasx_xvinsgr2vr_d(v, __lsx_vpickve2gr_d(x, 1), 2);
    return __lasx_xvsllwil_h_b(v, 0);
        #endif
    #elif defined(USE_LSX)
    return __lsx_vsllwil_h_b(__lsx_vldrepl_d(reinterpret_cast<const void*>(p), 0), 0);
    #endif
}
#endif

struct Vec16Wrapper {
#ifdef VECTOR
    using type = vec_t;
    static type add(const type& lhs, const type& rhs) { return vec_add_16(lhs, rhs); }
    static type sub(const type& lhs, const type& rhs) { return vec_sub_16(lhs, rhs); }
#else
    using type = BiasType;
    static type add(const type& lhs, const type& rhs) { return lhs + rhs; }
    static type sub(const type& lhs, const type& rhs) { return lhs - rhs; }
#endif
};

struct Vec32Wrapper {
#ifdef VECTOR
    using type = psqt_vec_t;
    static type add(const type& lhs, const type& rhs) { return vec_add_psqt_32(lhs, rhs); }
    static type sub(const type& lhs, const type& rhs) { return vec_sub_psqt_32(lhs, rhs); }
#else
    using type = PSQTWeightType;
    static type add(const type& lhs, const type& rhs) { return lhs + rhs; }
    static type sub(const type& lhs, const type& rhs) { return lhs - rhs; }
#endif
};

enum UpdateOperation {
    Add,
    Sub
};

template<typename VecWrapper,
         UpdateOperation... ops,
         std::enable_if_t<sizeof...(ops) == 0, bool> = true>
typename VecWrapper::type fused(const typename VecWrapper::type& in) {
    return in;
}

template<typename VecWrapper,
         UpdateOperation update_op,
         UpdateOperation... ops,
         typename T,
         typename... Ts,
         std::enable_if_t<is_all_same_v<typename VecWrapper::type, T, Ts...>, bool> = true,
         std::enable_if_t<sizeof...(ops) == sizeof...(Ts), bool>                    = true>
typename VecWrapper::type
fused(const typename VecWrapper::type& in, const T& operand, const Ts&... operands) {
    switch (update_op)
    {
    case Add :
        return fused<VecWrapper, ops...>(VecWrapper::add(in, operand), operands...);
    case Sub :
        return fused<VecWrapper, ops...>(VecWrapper::sub(in, operand), operands...);
    default :
        static_assert(update_op == Add || update_op == Sub,
                      "Only Add and Sub are currently supported.");
        return typename VecWrapper::type();
    }
}

#if defined(USE_AVX512)

[[maybe_unused]] static int m512_hadd(__m512i sum, int bias) {
    return _mm512_reduce_add_epi32(sum) + bias;
}

[[maybe_unused]] static void m512_add_dpbusd_epi32(__m512i& acc, __m512i a, __m512i b) {

    #if defined(USE_VNNI)
    acc = _mm512_dpbusd_epi32(acc, a, b);
    #elif defined(USE_AVX512BW)
    __m512i product0 = _mm512_maddubs_epi16(a, b);
    product0         = _mm512_madd_epi16(product0, _mm512_set1_epi16(1));
    acc              = _mm512_add_epi32(acc, product0);
    #else
    acc = _mm512_add_epi32(
      acc, __builtin_shufflevector(
             _mm256_madd_epi16(_mm256_maddubs_epi16(__builtin_shufflevector(a, a, 0, 1, 2, 3),
                                                    __builtin_shufflevector(b, b, 0, 1, 2, 3)),
                               _mm256_set1_epi16(1)),
             _mm256_madd_epi16(_mm256_maddubs_epi16(__builtin_shufflevector(a, a, 4, 5, 6, 7),
                                                    __builtin_shufflevector(b, b, 4, 5, 6, 7)),
                               _mm256_set1_epi16(1)),
             0, 1, 2, 3, 4, 5, 6, 7));
    #endif
}

#endif

#if defined(USE_AVX2)

[[maybe_unused]] static int m256_hadd(__m256i sum, int bias) {
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
    sum128         = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_BADC));
    sum128         = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_CDAB));
    return _mm_cvtsi128_si32(sum128) + bias;
}

[[maybe_unused]] static void m256_add_dpbusd_epi32(__m256i& acc, __m256i a, __m256i b) {

    #if defined(USE_VNNI)
    acc = _mm256_dpbusd_epi32(acc, a, b);
    #else
    __m256i product0 = _mm256_maddubs_epi16(a, b);
    product0         = _mm256_madd_epi16(product0, _mm256_set1_epi16(1));
    acc              = _mm256_add_epi32(acc, product0);
    #endif
}

#endif

#if defined(USE_SSSE3)

[[maybe_unused]] static int m128_hadd(__m128i sum, int bias) {
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4E));  //_MM_PERM_BADC
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xB1));  //_MM_PERM_CDAB
    return _mm_cvtsi128_si32(sum) + bias;
}

[[maybe_unused]] static void m128_add_dpbusd_epi32(__m128i& acc, __m128i a, __m128i b) {

    __m128i product0 = _mm_maddubs_epi16(a, b);
    product0         = _mm_madd_epi16(product0, _mm_set1_epi16(1));
    acc              = _mm_add_epi32(acc, product0);
}

#endif

#if defined(USE_NEON_DOTPROD)

[[maybe_unused]] static void
dotprod_m128_add_dpbusd_epi32(int32x4_t& acc, int8x16_t a, int8x16_t b) {

    acc = vdotq_s32(acc, a, b);
}
#endif

#if defined(USE_NEON)

[[maybe_unused]] static int neon_m128_reduce_add_epi32(int32x4_t s) {
    #if USE_NEON >= 8
    return vaddvq_s32(s);
    #else
    return s[0] + s[1] + s[2] + s[3];
    #endif
}

[[maybe_unused]] static int neon_m128_hadd(int32x4_t sum, int bias) {
    return neon_m128_reduce_add_epi32(sum) + bias;
}

#endif

#if defined(USE_LASX)

[[maybe_unused]] static int lasx_m256_hadd(__m256i sum, int bias) {
    __m256i v  = sum;
    v          = __lasx_xvadd_w(v, __lasx_xvshuf4i_w(v, 0x4E));  // [C,D,A,B] per lane
    v          = __lasx_xvadd_w(v, __lasx_xvshuf4i_w(v, 0xB1));  // [B,A,D,C] per lane
    int lo_sum = int(__lasx_xvpickve2gr_w(v, 0));
    int hi_sum = int(__lasx_xvpickve2gr_w(v, 4));
    return lo_sum + hi_sum + bias;
}

// acc += the sums of the products of 4 unsigned bytes of a and 4 signed bytes
// of b (as _mm256_dpbusd_epi32: no saturation with inputs below 128)
[[maybe_unused]] static void lasx_m256_add_dpbusd_epi32(__m256i& acc, __m256i a, __m256i b) {
    __m256i tmp = __lasx_xvmulwev_h_bu_b(a, b);
    tmp         = __lasx_xvmaddwod_h_bu_b(tmp, a, b);
    acc         = __lasx_xvadd_w(acc, __lasx_xvhaddw_w_h(tmp, tmp));
}

#endif

#if defined(USE_LSX)

[[maybe_unused]] static int lsx_m128_hadd(__m128i sum, int bias) {
    sum = __lsx_vadd_w(sum, __lsx_vshuf4i_w(sum, 0x4E));  // [C,D,A,B]
    sum = __lsx_vadd_w(sum, __lsx_vshuf4i_w(sum, 0xB1));  // [B,A,D,C]
    return __lsx_vpickve2gr_w(sum, 0) + bias;
}

[[maybe_unused]] static void lsx_m128_add_dpbusd_epi32(__m128i& acc, __m128i a, __m128i b) {
    __m128i tmp = __lsx_vmulwev_h_bu_b(a, b);
    tmp         = __lsx_vmaddwod_h_bu_b(tmp, a, b);
    acc         = __lsx_vadd_w(acc, __lsx_vhaddw_w_h(tmp, tmp));
}

#endif

#if USE_NEON >= 8
[[maybe_unused]] static void neon_m128_add_dpbusd_epi32(int32x4_t& acc, int8x16_t a, int8x16_t b) {

    int16x8_t product0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    int16x8_t product1 = vmull_high_s8(a, b);
    int16x8_t sum      = vpaddq_s16(product0, product1);
    acc                = vpadalq_s16(acc, sum);
}
#endif


// Compute optimal SIMD register count for feature transformer accumulation.
template<IndexType TransformedFeatureWidth, IndexType HalfDimensions, IndexType PSQTBuckets>
class SIMDTiling {
#ifdef VECTOR
        // We use __m* types as template arguments, which causes GCC to emit warnings
        // about losing some attribute information. This is irrelevant to us as we
        // only take their size, so the following pragma are harmless.
    #if defined(__GNUC__)
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wignored-attributes"
    #endif

    template<typename SIMDRegisterType, typename LaneType, int NumLanes, int MaxRegisters>
    static constexpr int BestRegisterCount() {
        constexpr std::size_t RegisterSize = sizeof(SIMDRegisterType);
        constexpr std::size_t LaneSize     = sizeof(LaneType);

        static_assert(RegisterSize >= LaneSize);
        static_assert(MaxRegisters <= NumRegistersSIMD);
        static_assert(MaxRegisters > 0);
        static_assert(NumRegistersSIMD > 0);
        static_assert(RegisterSize % LaneSize == 0);
        static_assert((NumLanes * LaneSize) % RegisterSize == 0);

        const int ideal = (NumLanes * LaneSize) / RegisterSize;
        if (ideal <= MaxRegisters)
            return ideal;

        // Look for the largest divisor of the ideal register count that is smaller than MaxRegisters
        for (int divisor = MaxRegisters; divisor > 1; --divisor)
            if (ideal % divisor == 0)
                return divisor;

        return 1;
    }

    #if defined(__GNUC__)
        #pragma GCC diagnostic pop
    #endif

   public:
    static constexpr int NumRegs =
      BestRegisterCount<vec_t, BiasType, TransformedFeatureWidth, NumRegistersSIMD>();
    static constexpr int NumPsqtRegs =
      BestRegisterCount<psqt_vec_t, PSQTWeightType, PSQTBuckets, NumRegistersSIMD>();

    static constexpr IndexType TileHeight     = NumRegs * sizeof(vec_t) / 2;
    static constexpr IndexType PsqtTileHeight = NumPsqtRegs * sizeof(psqt_vec_t) / 4;

    static_assert(HalfDimensions % TileHeight == 0, "TileHeight must divide HalfDimensions");
    static_assert(PSQTBuckets % PsqtTileHeight == 0, "PsqtTileHeight must divide PSQTBuckets");
#endif
};
}

#endif
