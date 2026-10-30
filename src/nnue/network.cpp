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

#include "network.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

#include "../evaluate.h"
#include "../memory.h"
#include "../misc.h"
#include "../position.h"
#include "../types.h"
#include "nnue_architecture.h"
#include "nnue_common.h"
#include "nnue_misc.h"
#include "nnz_helper.h"

// The default network embedded in the binary (nnue/embedded_nnue.S): the
// zstd-compressed maskfish.nnue, or nothing (size 0) when the binary was built
// without it
extern "C" {
extern const unsigned char gEmbeddedNNUEData[];
extern const unsigned int  gEmbeddedNNUESize;
}

namespace Maskfish::Eval::NNUE {


namespace Detail {

// Read evaluation function parameters
template<typename T>
bool read_parameters(std::istream& stream, T& reference) {

    std::uint32_t header;
    header = read_little_endian<std::uint32_t>(stream);
    if (!stream || header != T::get_hash_value())
        return false;
    return reference.read_parameters(stream);
}

// Write evaluation function parameters
template<typename T>
bool write_parameters(std::ostream& stream, const T& reference) {

    write_little_endian<std::uint32_t>(stream, T::get_hash_value());
    return reference.write_parameters(stream);
}

}  // namespace Detail

template<typename Arch, typename Transformer>
void Network<Arch, Transformer>::load(const std::filesystem::path& rootDirectory,
                                      std::string                  evalfilePath) {
#if defined(DEFAULT_NNUE_DIRECTORY)
    std::vector<std::filesystem::path> dirs = {std::filesystem::path{}, rootDirectory,
                                               path_from_utf8(stringify(DEFAULT_NNUE_DIRECTORY))};
#else
    std::vector<std::filesystem::path> dirs = {std::filesystem::path{}, rootDirectory};
#endif

    if (evalfilePath.empty())
        evalfilePath = evalFile.defaultName;

    if (std::string(evalFile.current) == evalfilePath)
        return;

    // EvalFile not set (empty or the default name): the network embedded in
    // the binary only, never a file (verify() reports a binary without one).
    // A file named like the default network is loaded with a path, e.g.
    // ./maskfish.nnue.
    if (evalfilePath == std::string(evalFile.defaultName))
    {
        load_internal();
        return;
    }

    // EvalFile set: this file only, looked up as given (absolute, or relative
    // to the working directory), then in the directory of the binary
    for (const auto& directory : dirs)
    {
        if (std::string(evalFile.current) != evalfilePath)
        {
            load_user_net(directory, evalfilePath);
        }
    }
}


template<typename Arch, typename Transformer>
void Network<Arch, Transformer>::reload(const std::filesystem::path& rootDirectory,
                                        std::string                  evalfilePath) {
    // Not loaded until it loads again: a network that no longer fits the
    // options is then reported by verify() instead of being used
    evalFile.current = std::string("None");
    load(rootDirectory, evalfilePath);
}


template<typename Arch, typename Transformer>
bool Network<Arch, Transformer>::save(const std::optional<std::string>& filename) const {
    std::string actualFilename;
    std::string msg;

    if (filename.has_value())
        actualFilename = filename.value();
    else
    {
        if (std::string(evalFile.current) != std::string(evalFile.defaultName))
        {
            msg = "Failed to export a net. "
                  "A non-embedded net can only be saved if the filename is specified";

            sync_cout << msg << sync_endl;
            return false;
        }

        actualFilename = evalFile.defaultName;
    }

    std::ofstream stream(path_from_utf8(actualFilename), std::ios_base::binary);
    bool          saved = save(stream, evalFile.current, evalFile.netDescription);

    msg = saved ? "Network saved successfully to " + actualFilename : "Failed to export a net";

    sync_cout << msg << sync_endl;
    return saved;
}


template<typename Arch, typename Transformer>
NetworkOutput
Network<Arch, Transformer>::evaluate(const Position&                         pos,
                                     AccumulatorStack&                       accumulatorStack,
                                     AccumulatorCaches::Cache<FTDimensions>* cache) const {

    constexpr uint64_t alignment = CacheLineSize;

    alignas(alignment)
      TransformedFeatureType transformedFeatures[FeatureTransformer<FTDimensions>::BufferSize];

    ASSERT_ALIGNED(transformedFeatures, alignment);

    NNZInfo<FTDimensions> nnzInfo;

    const int  bucket = FeatureSet::make_layer_stack_bucket(pos);
    const auto psqt   = featureTransformer.transform(pos, accumulatorStack, cache,
                                                     transformedFeatures, psqt_bucket(bucket), nnzInfo);

    std::uint8_t context[ContextOutputs] = {};
    if constexpr (UseContext)
        compute_context(pos, context);

    auto positional = network[bucket].propagate(transformedFeatures, nnzInfo, context);

    // make lsblend=yes: interpolation with the next layer stack (the PSQT
    // bucket stays the floor one)
    if constexpr (UseLayerStackBlend)
    {
        const int blend = FeatureSet::make_layer_stack_selection(pos).blend;
        if (blend)
        {
            const auto next =
              network[bucket + 1].propagate(transformedFeatures, nnzInfo, context);
            positional =
              std::int32_t((std::int64_t(positional) * (256 - blend) + std::int64_t(next) * blend)
                           / 256);
        }
    }
    return {static_cast<Value>(psqt / OutputScale), static_cast<Value>(positional / OutputScale)};
}


template<typename Arch, typename Transformer>
void Network<Arch, Transformer>::compute_context(const Position& pos, std::uint8_t* context) const {
    std::uint8_t inputs[ContextInputs];
    ContextTower::make_inputs(pos, inputs);
    contextTower.propagate(inputs, context);
}


void ContextTower::make_inputs(const Position& pos, std::uint8_t* inputs) {
    // Divisors of the inputs (see the declaration), and q = round(127 * n / d)
    constexpr int PoolMax[6] = {2, 2, 2, 5, 2, 2};
    auto          q          = [](int n, int d) {
        return std::uint8_t(std::min(127, (2 * 127 * n + d) / (2 * d)));
    };

    const Color us = pos.side_to_move();
    int         k  = 0;
    int         pool[COLOR_NB] = {};
    for (Color c : {us, ~us})
        for (PieceType pt = ROOK; pt < KING; ++pt)
        {
            const int n = pos.rest_piece(make_piece(c, pt));
            pool[c] += n;
            inputs[k++] = q(n, PoolMax[pt - ROOK]);
        }
    const int dark[COLOR_NB] = {popcount(pos.pieces(WHITE, DARK)), popcount(pos.pieces(BLACK, DARK))};
    inputs[k++] = q(dark[us], 16);
    inputs[k++] = q(dark[~us], 16);
    inputs[k++] = q(std::max(0, pool[us] - dark[us]), 15);
    inputs[k++] = q(std::max(0, pool[~us] - dark[~us]), 15);
    assert(k == int(ContextInputs));
}


template<typename Arch, typename Transformer>
void Network<Arch, Transformer>::verify(std::string                                  evalfilePath,
                                        const std::function<void(std::string_view)>& f) const {
    if (evalfilePath.empty())
        evalfilePath = evalFile.defaultName;

    if (std::string(evalFile.current) != evalfilePath)
    {
        if (f)
        {
            const bool  embedded = evalfilePath == std::string(evalFile.defaultName);
            std::string msg1 =
              "Network evaluation parameters compatible with the engine must be available.";
            std::string msg2 = "The network file " + evalfilePath + " was not loaded successfully.";
            std::string msg3 = "The UCI option EvalFile might need to specify the full path, "
                               "including the directory name, to the network file.";
            if (embedded)
            {
                if (gEmbeddedNNUESize == 0)
                    msg2 = "This binary has no embedded network (it was built without the "
                           "network of the ref refs/assets/net, or with embed=no).";
                else
                    msg2 = "The network embedded in this binary is not compatible with the "
                           "engine (built with other network architecture options).";
                msg3 = "Set the UCI option EvalFile to a network file (with a path, e.g. ./"
                     + evalfilePath + ", for a file with the default name).";
            }
            std::string msg4 =
              "A Jieqi network (default name " + std::string(evalFile.defaultName)
              + ") can be trained with the trainer in the trainer/ directory of "
                  "https://github.com/nguyenhungitc/maskfish";
            std::string msg5 = "The engine will be terminated now.";
            if (oldVersionFile && !embedded)
                msg3 = "The network file has an old format (16-bit weights): convert it with "
                       "trainer/serialize.py --upgrade.";
            if (otherMidMirrorFile >= 0)
            {
                const auto b = [](bool v) { return std::string(v ? "true" : "false"); };
                msg3 = "The network was made for the UCI options MidMirrorPawnTieBreak="
                     + b(otherMidMirrorFile & 1) + " and MidMirrorNoKingFlag="
                     + b(otherMidMirrorFile & 2)
                     + " (as the data it was trained on, trainer options --mid-pawn-tiebreak "
                       "and --mid-no-king-flag): set them to these values.";
            }

            std::string msg = "ERROR: " + msg1 + '\n' + "ERROR: " + msg2 + '\n' + "ERROR: " + msg3
                            + '\n' + "ERROR: " + msg4 + '\n' + "ERROR: " + msg5 + '\n';

            f(msg);
        }

        exit(EXIT_FAILURE);
    }

    if (f)
    {
        size_t            size   = sizeof(featureTransformer) + sizeof(Arch) * LayerStacks;
        const std::string source = evalfilePath == std::string(evalFile.defaultName)
                                   ? evalfilePath + " (embedded)"
                                   : evalfilePath;
        f("NNUE evaluation using " + source + " (" + std::to_string(size / (1024 * 1024)) + "MiB, ("
          + std::to_string(featureTransformer.InputDimensions) + ", "
          + std::to_string(network[0].TransformedFeatureDimensions) + ", "
          + std::to_string(network[0].FC_0_OUTPUTS) + ", " + std::to_string(network[0].FC_1_OUTPUTS)
          + ", 1))");
    }
}


template<typename Arch, typename Transformer>
NnueEvalTrace
Network<Arch, Transformer>::trace_evaluate(const Position&                         pos,
                                           AccumulatorStack&                       accumulatorStack,
                                           AccumulatorCaches::Cache<FTDimensions>* cache) const {

    constexpr uint64_t alignment = CacheLineSize;

    alignas(alignment)
      TransformedFeatureType transformedFeatures[FeatureTransformer<FTDimensions>::BufferSize];

    ASSERT_ALIGNED(transformedFeatures, alignment);

    NnueEvalTrace t{};
    t.correctBucket = FeatureSet::make_layer_stack_bucket(pos);
    for (IndexType bucket = 0; bucket < LayerStacks; ++bucket)
    {
        NNZInfo<FTDimensions> nnzInfo;
        const auto            materialist =
          featureTransformer.transform(pos, accumulatorStack, cache, transformedFeatures,
                                       psqt_bucket(int(bucket)), nnzInfo);
        std::uint8_t context[ContextOutputs] = {};
        if constexpr (UseContext)
            compute_context(pos, context);
        const auto positional = network[bucket].propagate(transformedFeatures, nnzInfo, context);

        t.psqt[bucket]       = static_cast<Value>(materialist / OutputScale);
        t.positional[bucket] = static_cast<Value>(positional / OutputScale);
    }

    return t;
}


template<typename Arch, typename Transformer>
void Network<Arch, Transformer>::load_user_net(const std::filesystem::path& dir,
                                               const std::string&           evalfilePath) {
    // dir has no trailing separator (binary directory), so join with the
    // filesystem operator; an empty dir or an absolute evalfilePath give
    // evalfilePath unchanged. evalfilePath is UTF-8 (path_from_utf8() also
    // accepts the ANSI code page of old GUIs on Windows).
    std::stringstream stream = read_compressed_nnue(dir / path_from_utf8(evalfilePath));
    load_stream(stream, evalfilePath);
}


template<typename Arch, typename Transformer>
void Network<Arch, Transformer>::load_internal() {
    if (gEmbeddedNNUESize == 0)
        return;

    std::stringstream stream = read_compressed_nnue(gEmbeddedNNUEData, gEmbeddedNNUESize);
    load_stream(stream, std::string(evalFile.defaultName));
}


template<typename Arch, typename Transformer>
void Network<Arch, Transformer>::load_stream(std::stringstream& stream, const std::string& name) {
    auto description = load(stream);

    if (description.has_value())
    {
        evalFile.current        = name;
        evalFile.netDescription = description.value();
        oldVersionFile          = false;
        otherMidMirrorFile      = -1;
    }
    else
    {
        stream.clear();
        stream.seekg(0);
        const std::uint32_t version  = read_little_endian<std::uint32_t>(stream);
        const std::uint32_t fileHash = read_little_endian<std::uint32_t>(stream);
        oldVersionFile |= stream && version == OldVersion;
        // A file of this architecture made for other mid mirror settings
        if (stream && version == Version && fileHash != file_hash())
            for (bool tieBreak : {false, true})
                for (bool noKingFlag : {false, true})
                    if (fileHash == (hash ^ FeatureSet::runtime_hash(tieBreak, noKingFlag)))
                        otherMidMirrorFile = int(tieBreak) + 2 * int(noKingFlag);
    }
}


template<typename Arch, typename Transformer>
void Network<Arch, Transformer>::initialize() {
    initialized = true;
}


template<typename Arch, typename Transformer>
bool Network<Arch, Transformer>::save(std::ostream&      stream,
                                      const std::string& name,
                                      const std::string& netDescription) const {
    if (name.empty() || name == "None")
        return false;

    return write_parameters(stream, netDescription);
}


template<typename Arch, typename Transformer>
std::optional<std::string> Network<Arch, Transformer>::load(std::istream& stream) {
    initialize();
    std::string description;

    return read_parameters(stream, description) ? std::make_optional(description) : std::nullopt;
}


template<typename Arch, typename Transformer>
std::size_t Network<Arch, Transformer>::get_content_hash() const {
    if (!initialized)
        return 0;

    std::size_t h = 0;
    hash_combine(h, featureTransformer);
    if constexpr (UseContext)
        hash_combine(h, contextTower.get_content_hash());
    for (auto&& layerstack : network)
        hash_combine(h, layerstack);
    hash_combine(h, evalFile);
    return h;
}

// Read network header
template<typename Arch, typename Transformer>
bool Network<Arch, Transformer>::read_header(std::istream&  stream,
                                             std::uint32_t* hashValue,
                                             std::string*   desc) const {
    std::uint32_t version, size;

    version    = read_little_endian<std::uint32_t>(stream);
    *hashValue = read_little_endian<std::uint32_t>(stream);
    size       = read_little_endian<std::uint32_t>(stream);
    if (!stream || version != Version)
        return false;

    // Bound the description by what the stream actually holds rather than by
    // the declared size, which may be corrupted
    constexpr std::uint32_t Chunk = 4096;
    char                    buf[Chunk];

    desc->clear();
    for (std::uint32_t remaining = size; remaining > 0;)
    {
        const std::uint32_t want = std::min(remaining, Chunk);
        stream.read(buf, want);
        const std::uint32_t got = std::uint32_t(stream.gcount());
        desc->append(buf, got);
        if (got != want)
            return false;
        remaining -= want;
    }
    return !stream.fail();
}


// Write network header
template<typename Arch, typename Transformer>
bool Network<Arch, Transformer>::write_header(std::ostream&      stream,
                                              std::uint32_t      hashValue,
                                              const std::string& desc) const {
    write_little_endian<std::uint32_t>(stream, Version);
    write_little_endian<std::uint32_t>(stream, hashValue);
    write_little_endian<std::uint32_t>(stream, std::uint32_t(desc.size()));
    stream.write(&desc[0], desc.size());
    return !stream.fail();
}


template<typename Arch, typename Transformer>
bool Network<Arch, Transformer>::read_parameters(std::istream& stream,
                                                 std::string&  netDescription) {
    std::uint32_t hashValue;
    if (!read_header(stream, &hashValue, &netDescription))
        return false;
    if (hashValue != file_hash())
        return false;
    if (!Detail::read_parameters(stream, featureTransformer))
        return false;
    if constexpr (UseContext)
        if (!Detail::read_parameters(stream, contextTower))
            return false;
    for (std::size_t i = 0; i < LayerStacks; ++i)
    {
        if (!Detail::read_parameters(stream, network[i]))
            return false;
    }
    return stream && stream.peek() == std::ios::traits_type::eof();
}


template<typename Arch, typename Transformer>
bool Network<Arch, Transformer>::write_parameters(std::ostream&      stream,
                                                  const std::string& netDescription) const {
    if (!write_header(stream, file_hash(), netDescription))
        return false;
    if (!Detail::write_parameters(stream, featureTransformer))
        return false;
    if constexpr (UseContext)
        if (!Detail::write_parameters(stream, contextTower))
            return false;
    for (std::size_t i = 0; i < LayerStacks; ++i)
    {
        if (!Detail::write_parameters(stream, network[i]))
            return false;
    }
    return bool(stream);
}

// Explicit template instantiations

template class Network<NetworkArchitecture<TransformedFeatureDimensionsBig, L2Big, L3Big>,
                       FeatureTransformer<TransformedFeatureDimensionsBig>>;

}  // namespace Maskfish::Eval::NNUE
