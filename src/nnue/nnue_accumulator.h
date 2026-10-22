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

// Class for difference calculation of NNUE evaluation function

#ifndef NNUE_ACCUMULATOR_H_INCLUDED
#define NNUE_ACCUMULATOR_H_INCLUDED

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "../types.h"
#include "nnue_architecture.h"
#include "nnue_common.h"

namespace Maskfish {
class Position;
}

namespace Maskfish::Eval::NNUE {

template<IndexType Size>
struct alignas(CacheLineSize) Accumulator;

template<IndexType TransformedFeatureDimensions>
class FeatureTransformer;

// Class that holds the result of affine transformation of input features
template<IndexType Size>
struct alignas(CacheLineSize) Accumulator {
    std::int16_t               accumulation[COLOR_NB][Size];
    std::int32_t               psqtAccumulation[COLOR_NB][PSQTBuckets];
    std::array<bool, COLOR_NB> computed = {};
};


#ifdef JIEQI_NNUE_THREATS
// Accumulator of the threat features (make threats=yes), in half units (the
// threat weights are not doubled on load), with the sorted list of the threats
// of the position. It is updated lazily, from the nearest earlier state of the
// stack that has one, by the difference between the two lists of threats: this
// needs no per-move bookkeeping and is right whatever happened in between
// (reveals, chance nodes); a change of mirroring means a refresh.
template<IndexType Size>
struct alignas(CacheLineSize) ThreatAccumulator {
    std::int16_t               acc[COLOR_NB][Size];
    std::int32_t               psqt[COLOR_NB][PSQTBuckets];
    std::array<bool, COLOR_NB> computed  = {};
    std::array<bool, COLOR_NB> mirror    = {};     // mirroring acc was computed with
    std::uint16_t              features[COLOR_NB];  // number of active features
    bool                       listValid = false;  // list / count hold the threats
    std::uint16_t              count;
    ThreatFeatureSet::Threat   list[ThreatFeatureSet::MaxActiveDimensions];
};
#endif

// AccumulatorCaches struct provides per-thread accumulator caches, where each
// cache contains multiple entries for each of the possible king squares.
// When the accumulator needs to be refreshed, the cached entry is used to more
// efficiently update the accumulator, instead of rebuilding it from scratch.
// This idea, was first described by Luecx (author of Koivisto) and
// is commonly referred to as "Finny Tables".
struct AccumulatorCaches {

    // Entries are indexed by FeatureSet::make_cache_index()

    template<typename Networks>
    AccumulatorCaches(const Networks& networks) {
        clear(networks);
    }

    template<IndexType Size>
    struct alignas(CacheLineSize) Cache {

        struct alignas(CacheLineSize) Entry {
            BiasType       accumulation[Size];
            PSQTWeightType psqtAccumulation[PSQTBuckets];
            Bitboard       byColorBB[COLOR_NB];
            Bitboard       byTypeBB[PIECE_TYPE_NB + 1];
            int            restPieces[PIECE_NB];

            // To initialize a refresh entry, we set all its bitboards empty,
            // so we put the biases in the accumulation, without any weights on top
            void clear(const BiasType* biases) {

                std::memcpy(accumulation, biases, sizeof(accumulation));
                std::memset(reinterpret_cast<std::byte*>(this) + offsetof(Entry, psqtAccumulation),
                            0, sizeof(Entry) - offsetof(Entry, psqtAccumulation));
            }
        };

        template<typename Network>
        void clear(const Network& network) {
            for (auto& entries1D : entries)
                for (auto& entry : entries1D)
                    entry.clear(network.featureTransformer.biases);
        }

        std::array<Entry, COLOR_NB>& operator[](int index) { return entries[index]; }

        std::array<std::array<Entry, COLOR_NB>, FeatureSet::CacheEntriesPerColor> entries;
    };

    template<typename Networks>
    void clear(const Networks& networks) {
        big.clear(networks.big);
    }

    Cache<TransformedFeatureDimensionsBig> big;
};


struct AccumulatorState {
    Accumulator<TransformedFeatureDimensionsBig> accumulatorBig;
    DirtyPiece                                   dirtyPiece;
#ifdef JIEQI_NNUE_THREATS
    ThreatAccumulator<TransformedFeatureDimensionsBig> threats;
    // Threats added and removed by the move (and reveal) that led to this
    // state (UCI option ThreatsIncremental), filled by Position::do_move()
    // and do_flip(); not touched by reset()
    DirtyThreats dirtyThreats;
#endif
#ifdef JIEQI_NNUE_META
    // The meta features (make meta=yes) whose weights are in accumulatorBig
    // for each perspective (when it is computed): metaCount is 0 (none, after
    // a refresh) or MetaFeatures. They are those of the position of this state
    // only for the latest state of AccumulatorStack::evaluate(): the earlier
    // states keep the ones of the state they were updated from.
    IndexType    metaIndex[COLOR_NB][FeatureSet::MetaFeatures];
    std::uint8_t metaCount[COLOR_NB] = {};
#endif

    template<IndexType Size>
    auto& acc() noexcept {
        static_assert(Size == TransformedFeatureDimensionsBig, "Invalid size for accumulator");

        if constexpr (Size == TransformedFeatureDimensionsBig)
            return accumulatorBig;
    }

    template<IndexType Size>
    const auto& acc() const noexcept {
        static_assert(Size == TransformedFeatureDimensionsBig, "Invalid size for accumulator");

        if constexpr (Size == TransformedFeatureDimensionsBig)
            return accumulatorBig;
    }

    void reset(const DirtyPiece& dp) noexcept;
};


class AccumulatorStack {
   public:
    [[nodiscard]] const AccumulatorState& latest() const noexcept;

    void reset() noexcept;
    void push(const DirtyPiece& dirtyPiece) noexcept;
    void pop() noexcept;

#ifdef JIEQI_NNUE_THREATS
    // UCI option ThreatsIncremental: the threat accumulator is updated from
    // the threats recorded by do_move() and do_flip(), instead
    // of the difference between the lists of threats of the two positions
    static inline bool incrementalThreats = false;

    // Threats of the state that the next push() makes the latest one: filled
    // before the push (by do_move()), or again after a pop() (by do_flip())
    DirtyThreats& next_dirty_threats() noexcept { return accumulators[size].dirtyThreats; }
#endif

    template<IndexType Dimensions>
    void evaluate(const Position&                       pos,
                  const FeatureTransformer<Dimensions>& featureTransformer,
                  AccumulatorCaches::Cache<Dimensions>& cache) noexcept;

   private:
    [[nodiscard]] AccumulatorState& mut_latest() noexcept;

    template<Color Perspective, IndexType Dimensions>
    void evaluate_side(const Position&                       pos,
                       const FeatureTransformer<Dimensions>& featureTransformer,
                       AccumulatorCaches::Cache<Dimensions>& cache) noexcept;

    template<Color Perspective, IndexType Dimensions>
    [[nodiscard]] std::size_t find_last_usable_accumulator() const noexcept;

#ifdef JIEQI_NNUE_THREATS
    template<IndexType Dimensions>
    void evaluate_threats(const Position&                       pos,
                          const FeatureTransformer<Dimensions>& featureTransformer) noexcept;

    template<Color Perspective, IndexType Dimensions>
    void evaluate_threats_side(const Position&                       pos,
                               const FeatureTransformer<Dimensions>& featureTransformer) noexcept;

    template<Color Perspective>
    [[nodiscard]] std::size_t incremental_threats_source(bool mirror) const noexcept;

    template<IndexType Dimensions>
    bool evaluate_threats_incremental(const Position&                       pos,
                                      const FeatureTransformer<Dimensions>& featureTransformer) noexcept;
#endif

#ifdef JIEQI_NNUE_META
    template<Color Perspective, IndexType Dimensions>
    void update_meta(const FeatureTransformer<Dimensions>& featureTransformer,
                     const IndexType*                      metaIndex) noexcept;
#endif

    template<Color Perspective, IndexType Dimensions>
    void forward_update_incremental(const Position&                       pos,
                                    const FeatureTransformer<Dimensions>& featureTransformer,
                                    const std::size_t                     begin) noexcept;

    template<Color Perspective, IndexType Dimensions>
    void backward_update_incremental(const Position&                       pos,
                                     const FeatureTransformer<Dimensions>& featureTransformer,
                                     const std::size_t                     end) noexcept;

    std::array<AccumulatorState, MAX_PLY + 1> accumulators;
    std::size_t                               size = 1;
};

}  // namespace Maskfish::Eval::NNUE

#endif  // NNUE_ACCUMULATOR_H_INCLUDED
