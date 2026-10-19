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

#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include "../misc.h"
#include "../types.h"
#include "nnue_accumulator.h"
#include "nnue_architecture.h"
#include "nnue_common.h"
#include "nnue_feature_transformer.h"
#include "nnue_misc.h"

namespace Maskfish {
class Position;
}

namespace Maskfish::Eval::NNUE {

using NetworkOutput = std::tuple<Value, Value>;

// The network must be a trivial type, i.e. the memory must be in-line.
// This is required to allow sharing the network via shared memory, as
// there is no way to run destructors.
template<typename Arch, typename Transformer>
class Network {
    static constexpr IndexType FTDimensions = Arch::TransformedFeatureDimensions;

   public:
    Network(EvalFile file) :
        evalFile(file) {}

    Network(const Network& other) = default;
    Network(Network&& other)      = default;

    Network& operator=(const Network& other) = default;
    Network& operator=(Network&& other)      = default;

    // rootDirectory: the directory of the binary (a path, which does not
    // convert implicitly to a string on Windows); evalfilePath is UTF-8
    void load(const std::filesystem::path& rootDirectory, std::string evalfilePath);
    bool save(const std::optional<std::string>& filename) const;

    std::size_t get_content_hash() const;

    NetworkOutput evaluate(const Position&                         pos,
                           AccumulatorStack&                       accumulatorStack,
                           AccumulatorCaches::Cache<FTDimensions>* cache) const;


    void verify(std::string evalfilePath, const std::function<void(std::string_view)>&) const;
    NnueEvalTrace trace_evaluate(const Position&                         pos,
                                 AccumulatorStack&                       accumulatorStack,
                                 AccumulatorCaches::Cache<FTDimensions>* cache) const;

   private:
    // Outputs of the ContextTower for the position (make ctx=yes)
    void compute_context(const Position& pos, std::uint8_t* context) const;

    void load_user_net(const std::filesystem::path&, const std::string&);
    void load_internal();
    void load_stream(std::stringstream&, const std::string&);

    void initialize();

    bool                       save(std::ostream&, const std::string&, const std::string&) const;
    std::optional<std::string> load(std::istream&);

    bool read_header(std::istream&, std::uint32_t*, std::string*) const;
    bool write_header(std::ostream&, std::uint32_t, const std::string&) const;

    bool read_parameters(std::istream&, std::string&);
    bool write_parameters(std::ostream&, const std::string&) const;

    // Input feature converter
    Transformer featureTransformer;

    // Evaluation function
    Arch network[LayerStacks];

    // Inventory context (make ctx=yes), shared by the layer stacks
    ContextTower contextTower;

    EvalFile evalFile;

    bool initialized = false;

    // A file that failed to load had the version before the 8-bit feature
    // transformer weights (reported by verify())
    bool oldVersionFile = false;

    // Hash value of evaluation function structure
    static constexpr std::uint32_t hash = Transformer::get_hash_value() ^ Arch::get_hash_value();

    template<IndexType Size>
    friend struct AccumulatorCaches::Cache;

    friend class AccumulatorStack;
};

// Definitions of the network types
using BigFeatureTransformer  = FeatureTransformer<TransformedFeatureDimensionsBig>;
using BigNetworkArchitecture = NetworkArchitecture<TransformedFeatureDimensionsBig, L2Big, L3Big>;

using NetworkBig = Network<BigNetworkArchitecture, BigFeatureTransformer>;


struct Networks {
    Networks(std::unique_ptr<NetworkBig>&& nB) :
        big(std::move(*nB)) {}

    NetworkBig big;
};


}  // namespace Maskfish

template<typename ArchT, typename FeatureTransformerT>
struct std::hash<Maskfish::Eval::NNUE::Network<ArchT, FeatureTransformerT>> {
    std::size_t operator()(
      const Maskfish::Eval::NNUE::Network<ArchT, FeatureTransformerT>& network) const noexcept {
        return network.get_content_hash();
    }
};

template<>
struct std::hash<Maskfish::Eval::NNUE::Networks> {
    std::size_t operator()(const Maskfish::Eval::NNUE::Networks& networks) const noexcept {
        std::size_t h = 0;
        Maskfish::hash_combine(h, networks.big);
        return h;
    }
};

#endif
