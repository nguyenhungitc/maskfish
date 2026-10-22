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

// Constants used in NNUE evaluation function

#ifndef NNUE_COMMON_H_INCLUDED
#define NNUE_COMMON_H_INCLUDED

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <type_traits>

#include "../misc.h"

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
    #include <riscv_vector.h>
#endif

// The squared and the plain clipped ReLU of the layer stacks computed together
// with 256- or 512-bit vectors (SqrClippedReLU::propagate_pair(), "paired
// activations"); used by the layer stacks of make layers=2, whose
// activations have 32 inputs. Their outputs are in the lane order of the
// pack instructions, which the weights of the next layers are permuted to
// match at load time (AffineTransform with ScrambledInput).
#if defined(USE_AVX2)
    #define USE_PAIR_ACTIVATIONS
#endif

namespace Maskfish::Eval::NNUE {

// The feature transformer weights (HalfKA and threat features) have 8 bits
// like upstream, the accumulators 16 bits. Weights, biases and accumulators
// are in half units (scale 127, as in the file); the accumulators are doubled
// with saturation in FeatureTransformer::transform().
using BiasType         = std::int16_t;
using WeightType       = std::int8_t;
using PSQTWeightType   = std::int32_t;
using ThreatWeightType = std::int8_t;
using IndexType        = std::uint32_t;

// Version of the evaluation file (0x7AF32F20: 16-bit feature transformer
// weights, which trainer/serialize.py --upgrade converts)
constexpr std::uint32_t Version    = 0x7AF32F21u;
constexpr std::uint32_t OldVersion = 0x7AF32F20u;

// Constant used in evaluation value calculation
constexpr int OutputScale     = 16;
constexpr int WeightScaleBits = 6;

// Quantization (make quant=2: the alternative quantization).
//               quant=1 (default)                quant=2
// FT weights    scale 127, accumulators doubled  scale 255, not doubled
//               and clipped to 0..254            clipped to 0..FtMaxVal = 255
// FT output     0..127, 1.0 = 127               0..127, 1.0 = HiddenOneVal = 128
// fc_0          weights scale 64, 1.0 = 127*64   weights scale 128, 1.0 = 128*64*2
//               activations >> 6                 activations >> 7 (and sqr >> 21)
// fc_1          weights scale 64, >> 6           weights scale 64, >> 6 (1.0 = 128*64)
// fc_2 output   1.0 = 600 * OutputScale          1.0 = 128*64*2 (fc_0 units), the
//               direct term rescaled to it       sum with the direct term rescaled
#ifdef JIEQI_NNUE_QUANT2
constexpr bool UseQuant2 = true;
#else
constexpr bool UseQuant2 = false;
#endif
constexpr int FtMaxVal         = UseQuant2 ? 255 : 127 * 2;   // clip of the FT accumulators
constexpr int FtAccMultiplier  = UseQuant2 ? 1 : 2;           // applied to them first
constexpr int HiddenOneVal     = UseQuant2 ? 128 : 127;
constexpr int Fc0ScaleBits     = WeightScaleBits + (UseQuant2 ? 1 : 0);
// Xored into the hash of the layer stacks (networks of one quantization are
// rejected by builds of the other)
constexpr std::uint32_t Quant2ArchHash = 0x0A2C0000u;

// Size of cache line (in bytes)
constexpr std::size_t CacheLineSize = 64;

constexpr const char        Leb128MagicString[]   = "COMPRESSED_LEB128";
constexpr const std::size_t Leb128MagicStringSize = sizeof(Leb128MagicString) - 1;

// SIMD width (in bytes)
#if defined(USE_AVX2)
constexpr std::size_t SimdWidth = 32;

#elif defined(USE_SSE2)
constexpr std::size_t SimdWidth = 16;

#elif defined(USE_NEON)
constexpr std::size_t SimdWidth = 16;

#elif defined(USE_LASX)
constexpr std::size_t SimdWidth = 32;

#elif defined(USE_LSX)
constexpr std::size_t SimdWidth = 16;
#endif

constexpr std::size_t MaxSimdWidth = 32;

// Type of input feature after conversion
using TransformedFeatureType = std::uint8_t;

// Round n up to be a multiple of base
template<typename IntType>
constexpr IntType ceil_to_multiple(IntType n, IntType base) {
    return (n + base - 1) / base * base;
}


// Utility to read an integer (signed or unsigned, any size)
// from a stream in little-endian order. We swap the byte order after the read if
// necessary to return a result with the byte ordering of the compiling machine.
template<typename IntType>
inline IntType read_little_endian(std::istream& stream) {
    IntType result;

    if (IsLittleEndian)
        stream.read(reinterpret_cast<char*>(&result), sizeof(IntType));
    else
    {
        std::uint8_t                  u[sizeof(IntType)];
        std::make_unsigned_t<IntType> v = 0;

        stream.read(reinterpret_cast<char*>(u), sizeof(IntType));
        for (std::size_t i = 0; i < sizeof(IntType); ++i)
            v = (v << 8) | u[sizeof(IntType) - i - 1];

        std::memcpy(&result, &v, sizeof(IntType));
    }

    return result;
}


// Utility to write an integer (signed or unsigned, any size)
// to a stream in little-endian order. We swap the byte order before the write if
// necessary to always write in little-endian order, independently of the byte
// ordering of the compiling machine.
template<typename IntType>
inline void write_little_endian(std::ostream& stream, IntType value) {

    if (IsLittleEndian)
        stream.write(reinterpret_cast<const char*>(&value), sizeof(IntType));
    else
    {
        std::uint8_t                  u[sizeof(IntType)];
        std::make_unsigned_t<IntType> v = value;

        std::size_t i = 0;
        // if constexpr to silence the warning about shift by 8
        if constexpr (sizeof(IntType) > 1)
        {
            for (; i + 1 < sizeof(IntType); ++i)
            {
                u[i] = std::uint8_t(v);
                v >>= 8;
            }
        }
        u[i] = std::uint8_t(v);

        stream.write(reinterpret_cast<char*>(u), sizeof(IntType));
    }
}


// Read integers in bulk from a little-endian stream.
// This reads N integers from stream s and puts them in array out.
template<typename IntType>
inline void read_little_endian(std::istream& stream, IntType* out, std::size_t count) {
    if (IsLittleEndian)
        stream.read(reinterpret_cast<char*>(out), sizeof(IntType) * count);
    else
        for (std::size_t i = 0; i < count; ++i)
            out[i] = read_little_endian<IntType>(stream);
}


// Write integers in bulk to a little-endian stream.
// This takes N integers from array values and writes them on stream s.
template<typename IntType>
inline void write_little_endian(std::ostream& stream, const IntType* values, std::size_t count) {
    if (IsLittleEndian)
        stream.write(reinterpret_cast<const char*>(values), sizeof(IntType) * count);
    else
        for (std::size_t i = 0; i < count; ++i)
            write_little_endian<IntType>(stream, values[i]);
}


// Read N signed integers from the stream s, putting them in the array out.
// The stream is assumed to be compressed using the signed LEB128 format.
// See https://en.wikipedia.org/wiki/LEB128 for a description of the compression scheme.
template<typename IntType>
inline void read_leb_128(std::istream& stream, IntType* out, std::size_t count) {

    static_assert(std::is_signed_v<IntType>, "Not implemented for unsigned types");
    static_assert(sizeof(IntType) <= 4, "Not implemented for types larger than 32 bit");

    // Check the presence of our LEB128 magic string. A short or corrupted
    // file fails the stream instead of triggering an assert.
    char leb128MagicString[Leb128MagicStringSize];
    stream.read(leb128MagicString, Leb128MagicStringSize);
    if (stream.gcount() != std::streamsize(Leb128MagicStringSize))
    {
        stream.setstate(std::ios::failbit);
        return;
    }

    // Without the magic string, the block holds the values as plain
    // little-endian integers (no LEB128: the whole file is compressed with
    // zstd anyway; trainer: serialize.py --no-leb128). The
    // bytes already read are the start of the first values.
    if (strncmp(Leb128MagicString, leb128MagicString, Leb128MagicStringSize) != 0)
    {
        const std::size_t total = count * sizeof(IntType);
        if (total < Leb128MagicStringSize)
        {
            stream.setstate(std::ios::failbit);
            return;
        }
        auto bytes = std::make_unique<char[]>(total);
        std::memcpy(bytes.get(), leb128MagicString, Leb128MagicStringSize);
        stream.read(bytes.get() + Leb128MagicStringSize,
                    std::streamsize(total - Leb128MagicStringSize));
        if (stream.gcount() != std::streamsize(total - Leb128MagicStringSize))
        {
            stream.setstate(std::ios::failbit);
            return;
        }
        for (std::size_t i = 0; i < count; ++i)
        {
            std::make_unsigned_t<IntType> v = 0;
            for (std::size_t b = 0; b < sizeof(IntType); ++b)
                v |= std::make_unsigned_t<IntType>(
                       std::uint8_t(bytes[i * sizeof(IntType) + b]))
                  << (8 * b);
            out[i] = IntType(v);
        }
        return;
    }

    const std::uint32_t BUF_SIZE = 4096;
    std::uint8_t        buf[BUF_SIZE];

    auto bytes_left = read_little_endian<std::uint32_t>(stream);

    std::uint32_t buf_pos = 0, buf_end = 0;
    for (std::size_t i = 0; i < count; ++i)
    {
        // Accumulate into an unsigned type: for int32 values a 5-byte
        // encoding reaches shift 28, which overflows a signed int.
        std::uint32_t result = 0;
        size_t        shift  = 0;
        while (true)
        {
            if (buf_pos == buf_end)
            {
                if (bytes_left == 0)
                {
                    stream.setstate(std::ios::failbit);
                    return;
                }
                stream.read(reinterpret_cast<char*>(buf), std::min(bytes_left, BUF_SIZE));
                buf_pos = 0;
                buf_end = std::uint32_t(stream.gcount());

                // Short read: the file is truncated
                if (buf_end == 0)
                {
                    stream.setstate(std::ios::failbit);
                    return;
                }
            }

            std::uint8_t byte = buf[buf_pos++];
            --bytes_left;
            result |= std::uint32_t(byte & 0x7f) << (shift % 32);
            shift += 7;

            if ((byte & 0x80) == 0)
            {
                out[i] = IntType((sizeof(IntType) * 8 <= shift || (byte & 0x40) == 0)
                                   ? result
                                   : result | ~((std::uint32_t(1) << shift) - 1));
                break;
            }

            // Too many continuation bytes for this type: corrupted data
            if (shift >= sizeof(IntType) * 8)
            {
                stream.setstate(std::ios::failbit);
                return;
            }
        }
    }

    if (bytes_left != 0)
        stream.setstate(std::ios::failbit);
}


// Write signed integers to a stream with LEB128 compression.
// This takes N integers from array values, compresses them with
// the LEB128 algorithm and writes the result on the stream s.
// See https://en.wikipedia.org/wiki/LEB128 for a description of the compression scheme.
template<typename IntType>
inline void write_leb_128(std::ostream& stream, const IntType* values, std::size_t count) {

    // Write our LEB128 magic string
    stream.write(Leb128MagicString, Leb128MagicStringSize);

    static_assert(std::is_signed_v<IntType>, "Not implemented for unsigned types");

    std::uint32_t byte_count = 0;
    for (std::size_t i = 0; i < count; ++i)
    {
        IntType      value = values[i];
        std::uint8_t byte;
        do
        {
            byte = value & 0x7f;
            value >>= 7;
            ++byte_count;
        } while ((byte & 0x40) == 0 ? value != 0 : value != -1);
    }

    write_little_endian(stream, byte_count);

    const std::uint32_t BUF_SIZE = 4096;
    std::uint8_t        buf[BUF_SIZE];
    std::uint32_t       buf_pos = 0;

    auto flush = [&]() {
        if (buf_pos > 0)
        {
            stream.write(reinterpret_cast<char*>(buf), buf_pos);
            buf_pos = 0;
        }
    };

    auto write = [&](std::uint8_t b) {
        buf[buf_pos++] = b;
        if (buf_pos == BUF_SIZE)
            flush();
    };

    for (std::size_t i = 0; i < count; ++i)
    {
        IntType value = values[i];
        while (true)
        {
            std::uint8_t byte = value & 0x7f;
            value >>= 7;
            if ((byte & 0x40) == 0 ? value == 0 : value == -1)
            {
                write(byte);
                break;
            }
            write(byte | 0x80);
        }
    }

    flush();
}

}  // namespace Maskfish::Eval::NNUE

#endif  // #ifndef NNUE_COMMON_H_INCLUDED
