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

// Input features and network structure used in NNUE evaluation function

#ifndef NNUE_ARCHITECTURE_H_INCLUDED
#define NNUE_ARCHITECTURE_H_INCLUDED

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iosfwd>

#include "features/half_ka_v2_hm.h"
#include "features/jieqi_threats.h"
#include "layers/affine_transform.h"
#include "layers/affine_transform_sparse_input.h"
#include "layers/clipped_relu.h"
#include "layers/sqr_clipped_relu.h"
#include "nnue_common.h"

namespace Maskfish::Eval::NNUE {

// Input features used in evaluation function
using FeatureSet = Features::HalfKAv2_hm;

#ifdef JIEQI_NNUE_THREATS
// Threat input features (make threats=yes), added to the HalfKA ones. As in
// Pikafish, the feature transformer then has 1024 outputs instead of 2048.
using ThreatFeatureSet = Features::JieqiThreats;

// Number of input feature dimensions after conversion
constexpr IndexType TransformedFeatureDimensionsBig = 1024;
#else
// Number of input feature dimensions after conversion
constexpr IndexType TransformedFeatureDimensionsBig = 2048;
#endif
#ifdef JIEQI_NNUE_LAYERS2
constexpr int L2Big = 32;
#else
constexpr int L2Big = 15;
#endif
constexpr int L3Big = 32;

constexpr IndexType PSQTBuckets = 16;

// make psqt=no: networks without the PSQT part of the feature transformer
// (SFNNv17 of Pikafish). The material then goes through the layer stacks.
#ifdef JIEQI_NNUE_NO_PSQT
constexpr bool UsePSQT = false;
#else
constexpr bool UsePSQT = true;
#endif
constexpr IndexType LayerStacks = 16;

// make ctx=yes: the "inventory context" of AB-JChess. 16 values describing
// the uncertainty of the position (pool counts, numbers of dark pieces and of
// dark pieces captured without being identified, for both sides) go through
// a small layer (ContextTower, 16 -> 16, clipped ReLU) shared by the layer
// stacks, whose 16 outputs are additional inputs of fc0 in every layer stack.
#ifdef JIEQI_NNUE_CTX
constexpr bool UseContext = true;
#else
constexpr bool UseContext = false;
#endif
constexpr IndexType ContextInputs  = 16;
constexpr IndexType ContextOutputs = 16;
// Xored into the hash of the layer stacks when they have context inputs
constexpr std::uint32_t ContextArchHash = 0x0C7A0000u;

struct ContextTower {
    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t get_hash_value() { return 0x0C7A16C7u; }

    // Inputs (unsigned, 127 = 1.0), from the point of view of the side to move:
    // its pool counts by piece type (rook, advisor, cannon, pawn, knight,
    // bishop; divided by 2, 2, 2, 5, 2, 2), the opponent's, the numbers of dark
    // pieces of both sides on the board (divided by 16), and the numbers of
    // their dark pieces captured without being identified (pool count minus
    // dark pieces on the board, divided by 15). Defined in network.cpp.
    static void make_inputs(const Position& pos, std::uint8_t* inputs);

    bool read_parameters(std::istream& stream) {
        read_little_endian<std::int32_t>(stream, biases, ContextOutputs);
        read_little_endian<std::int8_t>(stream, &weights[0][0], ContextOutputs * ContextInputs);
        return !stream.fail();
    }

    bool write_parameters(std::ostream& stream) const {
        write_little_endian<std::int32_t>(stream, biases, ContextOutputs);
        write_little_endian<std::int8_t>(stream, &weights[0][0], ContextOutputs * ContextInputs);
        return !stream.fail();
    }

    // Affine transform and clipped ReLU, as the hidden layers (weights at
    // scale 64, biases at scale 64 * 127, outputs 127 = 1.0)
    void propagate(const std::uint8_t* inputs, std::uint8_t* outputs) const {
        for (IndexType o = 0; o < ContextOutputs; ++o)
        {
            std::int32_t sum = biases[o];
            for (IndexType i = 0; i < ContextInputs; ++i)
                sum += weights[o][i] * inputs[i];
            outputs[o] = std::uint8_t(std::clamp(sum >> WeightScaleBits, 0, 127));
        }
    }

    std::size_t get_content_hash() const {
        std::size_t h = 0;
        hash_combine(h, get_raw_data_hash(biases));
        hash_combine(h, get_raw_data_hash(weights));
        hash_combine(h, get_hash_value());
        return h;
    }

    std::int32_t biases[ContextOutputs];
    std::int8_t  weights[ContextOutputs][ContextInputs];
};

// If vector instructions are enabled, we update and refresh the
// accumulator tile by tile such that each tile fits in the CPU's
// vector registers.
static_assert(PSQTBuckets % 8 == 0,
              "Per feature PSQT values cannot be processed at granularity lower than 8 at a time.");

#ifdef JIEQI_NNUE_LAYERS2

// Layer stacks of make layers=2, after upstream's "Update nnue-architecture":
// fc_0 has L2 = 32 outputs, all of them activated, and the direct term is the
// difference of its last two outputs; fc_1 gets a squared activation too, and
// fc_2 is fed by all the activations (fc_0's and fc_1's, 128 inputs). The
// quantization is the one of the default layer stacks.
template<IndexType L1, int L2, int L3>
struct NetworkArchitecture {
    static constexpr IndexType TransformedFeatureDimensions = L1;
    static constexpr int       FC_0_OUTPUTS                 = L2;
    static constexpr int       FC_1_OUTPUTS                 = L3;

    static_assert(FC_0_OUTPUTS >= 2);

    Layers::AffineTransformSparseInput<TransformedFeatureDimensions, FC_0_OUTPUTS> fc_0;
    Layers::SqrClippedReLU<FC_0_OUTPUTS>                                           ac_sqr_0;
    Layers::ClippedReLU<FC_0_OUTPUTS>                                              ac_0;
    Layers::AffineTransform<FC_0_OUTPUTS * 2, FC_1_OUTPUTS>                        fc_1;
    Layers::SqrClippedReLU<FC_1_OUTPUTS>                                           ac_sqr_1;
    Layers::ClippedReLU<FC_1_OUTPUTS>                                              ac_1;
    Layers::AffineTransform<FC_0_OUTPUTS * 2 + FC_1_OUTPUTS * 2, 1>                fc_2;

    // make ctx=yes: weights of the context inputs of fc_0 (scale 64)
    static constexpr IndexType FC_0_CONTEXT_OUTPUTS = FC_0_OUTPUTS;
    std::int8_t fc_0_context[UseContext ? FC_0_CONTEXT_OUTPUTS : 1][ContextOutputs];

    void add_context(const std::uint8_t* context, std::int32_t* fc0Out) const {
        for (IndexType o = 0; o < FC_0_CONTEXT_OUTPUTS; ++o)
            for (IndexType i = 0; i < ContextOutputs; ++i)
                fc0Out[o] += fc_0_context[o][i] * context[i];
    }

    // Hash value embedded in the evaluation file (the squared activations are
    // not part of it, like upstream)
    static constexpr std::uint32_t get_hash_value() {
        // input slice hash
        std::uint32_t hashValue = 0xEC42E90Du;
        hashValue ^= TransformedFeatureDimensions * 2;

        hashValue = decltype(fc_0)::get_hash_value(hashValue);
        hashValue = decltype(ac_0)::get_hash_value(hashValue);
        hashValue = decltype(fc_1)::get_hash_value(hashValue);
        hashValue = decltype(ac_1)::get_hash_value(hashValue);
        hashValue = decltype(fc_2)::get_hash_value(hashValue);

        return UseContext ? hashValue ^ ContextArchHash : hashValue;
    }

    // Read network parameters
    bool read_parameters(std::istream& stream) {
        if (!fc_0.read_parameters(stream))
            return false;
        if constexpr (UseContext)
            read_little_endian<std::int8_t>(stream, &fc_0_context[0][0],
                                            FC_0_CONTEXT_OUTPUTS * ContextOutputs);
        return ac_0.read_parameters(stream)
            && fc_1.read_parameters(stream) && ac_1.read_parameters(stream)
            && fc_2.read_parameters(stream);
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) const {
        if (!fc_0.write_parameters(stream))
            return false;
        if constexpr (UseContext)
            write_little_endian<std::int8_t>(stream, &fc_0_context[0][0],
                                             FC_0_CONTEXT_OUTPUTS * ContextOutputs);
        return ac_0.write_parameters(stream)
            && fc_1.write_parameters(stream) && ac_1.write_parameters(stream)
            && fc_2.write_parameters(stream);
    }

    // context: the outputs of the ContextTower (make ctx=yes), else unused
    std::int32_t propagate(const TransformedFeatureType* transformedFeatures,
                           [[maybe_unused]] const std::uint8_t* context = nullptr) const {
        struct alignas(CacheLineSize) Buffer {
            alignas(CacheLineSize) typename decltype(fc_0)::OutputBuffer fc_0_out;
            // [sqr(fc_0), crelu(fc_0), sqr(fc_1), crelu(fc_1)]
            alignas(CacheLineSize) typename decltype(ac_sqr_0)::OutputType
              concat[ceil_to_multiple<IndexType>(FC_0_OUTPUTS * 2 + FC_1_OUTPUTS * 2, 32)];
            alignas(CacheLineSize) typename decltype(ac_0)::OutputBuffer ac_0_out;
            alignas(CacheLineSize) typename decltype(fc_1)::OutputBuffer fc_1_out;
            alignas(CacheLineSize) typename decltype(ac_1)::OutputBuffer ac_1_out;
            alignas(CacheLineSize) typename decltype(fc_2)::OutputBuffer fc_2_out;

            Buffer() { std::memset(concat, 0, sizeof(concat)); }
        };

        Buffer buffer;

        fc_0.propagate(transformedFeatures, buffer.fc_0_out);
        if constexpr (UseContext)
            add_context(context, buffer.fc_0_out);
        ac_sqr_0.propagate(buffer.fc_0_out, buffer.concat);
        ac_0.propagate(buffer.fc_0_out, buffer.ac_0_out);
        std::memcpy(buffer.concat + FC_0_OUTPUTS, buffer.ac_0_out,
                    FC_0_OUTPUTS * sizeof(typename decltype(ac_0)::OutputType));

        fc_1.propagate(buffer.concat, buffer.fc_1_out);
        ac_sqr_1.propagate(buffer.fc_1_out, buffer.concat + FC_0_OUTPUTS * 2);
        ac_1.propagate(buffer.fc_1_out, buffer.ac_1_out);
        std::memcpy(buffer.concat + FC_0_OUTPUTS * 2 + FC_1_OUTPUTS, buffer.ac_1_out,
                    FC_1_OUTPUTS * sizeof(typename decltype(ac_1)::OutputType));

        fc_2.propagate(buffer.concat, buffer.fc_2_out);

        // The direct term, like fc_0_out[15] of the default layer stacks: 1.0
        // is 127*(1<<WeightScaleBits) in quantized form, but we want 1.0 to be
        // equal to 600*OutputScale (64-bit product, see there)
        const std::int64_t skip = std::int64_t(buffer.fc_0_out[FC_0_OUTPUTS - 2])
                                - buffer.fc_0_out[FC_0_OUTPUTS - 1];
        std::int32_t fwdOut =
          std::int32_t(skip * (600 * OutputScale) / (127 * (1 << WeightScaleBits)));
        std::int32_t outputValue = buffer.fc_2_out[0] + fwdOut;
        return outputValue;
    }

    std::size_t get_content_hash() const {
        std::size_t h = 0;
        hash_combine(h, fc_0.get_content_hash());
        hash_combine(h, ac_sqr_0.get_content_hash());
        hash_combine(h, ac_0.get_content_hash());
        hash_combine(h, fc_1.get_content_hash());
        hash_combine(h, ac_sqr_1.get_content_hash());
        hash_combine(h, ac_1.get_content_hash());
        hash_combine(h, fc_2.get_content_hash());
        if constexpr (UseContext)
            hash_combine(h, get_raw_data_hash(fc_0_context));
        hash_combine(h, get_hash_value());
        return h;
    }
};

#else

template<IndexType L1, int L2, int L3>
struct NetworkArchitecture {
    static constexpr IndexType TransformedFeatureDimensions = L1;
    static constexpr int       FC_0_OUTPUTS                 = L2;
    static constexpr int       FC_1_OUTPUTS                 = L3;

    Layers::AffineTransformSparseInput<TransformedFeatureDimensions, FC_0_OUTPUTS + 1> fc_0;
    Layers::SqrClippedReLU<FC_0_OUTPUTS + 1>                                           ac_sqr_0;
    Layers::ClippedReLU<FC_0_OUTPUTS + 1>                                              ac_0;
    Layers::AffineTransform<FC_0_OUTPUTS * 2, FC_1_OUTPUTS>                            fc_1;
    Layers::ClippedReLU<FC_1_OUTPUTS>                                                  ac_1;
    Layers::AffineTransform<FC_1_OUTPUTS, 1>                                           fc_2;

    // make ctx=yes: weights of the context inputs of fc_0 (scale 64), also
    // of its last output (the direct term)
    static constexpr IndexType FC_0_CONTEXT_OUTPUTS = FC_0_OUTPUTS + 1;
    std::int8_t fc_0_context[UseContext ? FC_0_CONTEXT_OUTPUTS : 1][ContextOutputs];

    void add_context(const std::uint8_t* context, std::int32_t* fc0Out) const {
        for (IndexType o = 0; o < FC_0_CONTEXT_OUTPUTS; ++o)
            for (IndexType i = 0; i < ContextOutputs; ++i)
                fc0Out[o] += fc_0_context[o][i] * context[i];
    }

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t get_hash_value() {
        // input slice hash
        std::uint32_t hashValue = 0xEC42E90Du;
        hashValue ^= TransformedFeatureDimensions * 2;

        hashValue = decltype(fc_0)::get_hash_value(hashValue);
        hashValue = decltype(ac_0)::get_hash_value(hashValue);
        hashValue = decltype(fc_1)::get_hash_value(hashValue);
        hashValue = decltype(ac_1)::get_hash_value(hashValue);
        hashValue = decltype(fc_2)::get_hash_value(hashValue);

        return UseContext ? hashValue ^ ContextArchHash : hashValue;
    }

    // Read network parameters
    bool read_parameters(std::istream& stream) {
        if (!fc_0.read_parameters(stream))
            return false;
        if constexpr (UseContext)
            read_little_endian<std::int8_t>(stream, &fc_0_context[0][0],
                                            FC_0_CONTEXT_OUTPUTS * ContextOutputs);
        return ac_0.read_parameters(stream)
            && fc_1.read_parameters(stream) && ac_1.read_parameters(stream)
            && fc_2.read_parameters(stream);
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) const {
        if (!fc_0.write_parameters(stream))
            return false;
        if constexpr (UseContext)
            write_little_endian<std::int8_t>(stream, &fc_0_context[0][0],
                                             FC_0_CONTEXT_OUTPUTS * ContextOutputs);
        return ac_0.write_parameters(stream)
            && fc_1.write_parameters(stream) && ac_1.write_parameters(stream)
            && fc_2.write_parameters(stream);
    }

    // context: the outputs of the ContextTower (make ctx=yes), else unused
    std::int32_t propagate(const TransformedFeatureType* transformedFeatures,
                           [[maybe_unused]] const std::uint8_t* context = nullptr) const {
        struct alignas(CacheLineSize) Buffer {
            alignas(CacheLineSize) typename decltype(fc_0)::OutputBuffer fc_0_out;
            alignas(CacheLineSize) typename decltype(ac_sqr_0)::OutputType
              ac_sqr_0_out[ceil_to_multiple<IndexType>(FC_0_OUTPUTS * 2, 32)];
            alignas(CacheLineSize) typename decltype(ac_0)::OutputBuffer ac_0_out;
            alignas(CacheLineSize) typename decltype(fc_1)::OutputBuffer fc_1_out;
            alignas(CacheLineSize) typename decltype(ac_1)::OutputBuffer ac_1_out;
            alignas(CacheLineSize) typename decltype(fc_2)::OutputBuffer fc_2_out;

            Buffer() { std::memset(ac_sqr_0_out, 0, sizeof(ac_sqr_0_out)); }
        };

        Buffer buffer;

        fc_0.propagate(transformedFeatures, buffer.fc_0_out);
        if constexpr (UseContext)
            add_context(context, buffer.fc_0_out);
        ac_sqr_0.propagate(buffer.fc_0_out, buffer.ac_sqr_0_out);
        ac_0.propagate(buffer.fc_0_out, buffer.ac_0_out);
        std::memcpy(buffer.ac_sqr_0_out + FC_0_OUTPUTS, buffer.ac_0_out,
                    FC_0_OUTPUTS * sizeof(typename decltype(ac_0)::OutputType));
        fc_1.propagate(buffer.ac_sqr_0_out, buffer.fc_1_out);
        ac_1.propagate(buffer.fc_1_out, buffer.ac_1_out);
        fc_2.propagate(buffer.ac_1_out, buffer.fc_2_out);

        // buffer.fc_0_out[FC_0_OUTPUTS] is such that 1.0 is equal to 127*(1<<WeightScaleBits) in
        // quantized form, but we want 1.0 to be equal to 600*OutputScale. The
        // product is computed in 64 bits: in 32 bits it overflows as soon as
        // |fc_0_out| > 2^31 / 9600, about 27.5 in float (16 500 internal units).
        std::int32_t fwdOut = std::int32_t(std::int64_t(buffer.fc_0_out[FC_0_OUTPUTS])
                                           * (600 * OutputScale) / (127 * (1 << WeightScaleBits)));
        std::int32_t outputValue = buffer.fc_2_out[0] + fwdOut;

        return outputValue;
    }

    std::size_t get_content_hash() const {
        std::size_t h = 0;
        hash_combine(h, fc_0.get_content_hash());
        hash_combine(h, ac_sqr_0.get_content_hash());
        hash_combine(h, ac_0.get_content_hash());
        hash_combine(h, fc_1.get_content_hash());
        hash_combine(h, ac_1.get_content_hash());
        hash_combine(h, fc_2.get_content_hash());
        if constexpr (UseContext)
            hash_combine(h, get_raw_data_hash(fc_0_context));
        hash_combine(h, get_hash_value());
        return h;
    }
};

#endif

}  // namespace Maskfish::Eval::NNUE

template<Maskfish::Eval::NNUE::IndexType L1, int L2, int L3>
struct std::hash<Maskfish::Eval::NNUE::NetworkArchitecture<L1, L2, L3>> {
    std::size_t
    operator()(const Maskfish::Eval::NNUE::NetworkArchitecture<L1, L2, L3>& arch) const noexcept {
        return arch.get_content_hash();
    }
};

#endif  // #ifndef NNUE_ARCHITECTURE_H_INCLUDED
