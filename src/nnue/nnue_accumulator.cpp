/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "nnue_accumulator.h"

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <type_traits>

#include "../bitboard.h"
#include "../misc.h"
#include "../position.h"
#include "../types.h"
#include "nnue_architecture.h"
#include "nnue_feature_transformer.h"  // IWYU pragma: keep
#include "simd.h"

namespace Stockfish::Eval::NNUE {

using namespace SIMD;

namespace {

template<Color Perspective, IndexType TransformedFeatureDimensions>
void double_inc_update(const FeatureTransformer<TransformedFeatureDimensions>& featureTransformer,
                       const int                                               bucket,
                       const bool                                              mirror,
                       AccumulatorState&                                       middle_state,
                       AccumulatorState&                                       target_state,
                       const AccumulatorState&                                 computed);

template<Color Perspective, bool Forward, IndexType TransformedFeatureDimensions>
void update_accumulator_incremental(
  const FeatureTransformer<TransformedFeatureDimensions>& featureTransformer,
  const int                                               bucket,
  const bool                                              mirror,
  AccumulatorState&                                       target_state,
  const AccumulatorState&                                 computed);

template<Color Perspective, IndexType Dimensions>
void update_accumulator_refresh_cache(const FeatureTransformer<Dimensions>& featureTransformer,
                                      const Position&                       pos,
                                      AccumulatorState&                     accumulatorState,
                                      AccumulatorCaches::Cache<Dimensions>& cache);

}

void AccumulatorState::reset(const DirtyPiece& dp) noexcept {
    dirtyPiece = dp;
    accumulatorBig.computed.fill(false);
#ifdef JIEQI_NNUE_THREATS
    threats.computed.fill(false);
    threats.listValid = false;
#endif
}

const AccumulatorState& AccumulatorStack::latest() const noexcept { return accumulators[size - 1]; }

AccumulatorState& AccumulatorStack::mut_latest() noexcept { return accumulators[size - 1]; }

void AccumulatorStack::reset() noexcept {
    accumulators[0].reset({});
    size = 1;
}

void AccumulatorStack::push(const DirtyPiece& dirtyPiece) noexcept {
    assert(size < accumulators.size());
    accumulators[size].reset(dirtyPiece);
    size++;
}

void AccumulatorStack::pop() noexcept {
    assert(size > 1);
    size--;
}

template<IndexType Dimensions>
void AccumulatorStack::evaluate(const Position&                       pos,
                                const FeatureTransformer<Dimensions>& featureTransformer,
                                AccumulatorCaches::Cache<Dimensions>& cache) noexcept {

    evaluate_side<WHITE>(pos, featureTransformer, cache);
    evaluate_side<BLACK>(pos, featureTransformer, cache);
#ifdef JIEQI_NNUE_THREATS
    evaluate_threats(pos, featureTransformer);
#endif
}

#ifdef JIEQI_NNUE_THREATS
namespace {

// dst = src - sum of the `removed` rows + sum of the `added` rows of the threat
// weights (src == nullptr: zero), for the accumulator and the PSQT values. The
// accumulator is processed by tiles kept in registers, so that it is read and
// written once whatever the number of features.
template<IndexType Dimensions>
void apply_threats(std::int16_t* __restrict dst,
                   std::int32_t* __restrict dstPsqt,
                   const std::int16_t* __restrict src,
                   const std::int32_t* __restrict srcPsqt,
                   const FeatureTransformer<Dimensions>& featureTransformer,
                   const IndexType*                      removed,
                   int                                   nRemoved,
                   const IndexType*                      added,
                   int                                   nAdded) {

    const ThreatWeightType* weights = featureTransformer.threatWeights;

#if defined(__GNUC__)
    // GCC / clang vector extensions: 16 x int16 per vector (two registers on
    // 128-bit targets), sign extension of the 8-bit weights
    using I16 = std::int16_t __attribute__((vector_size(32)));
    using I8  = std::int8_t __attribute__((vector_size(16)));
    constexpr IndexType Lanes = 16, Regs = 8, Tile = Lanes * Regs;
    static_assert(Dimensions % Tile == 0);

    for (IndexType j = 0; j < Dimensions; j += Tile)
    {
        I16 r[Regs];
        for (IndexType i = 0; i < Regs; ++i)
            if (src)
                std::memcpy(&r[i], src + j + i * Lanes, sizeof(I16));
            else
                r[i] = I16{};

        for (int k = 0; k < nRemoved; ++k)
        {
            const ThreatWeightType* w = weights + std::size_t(removed[k]) * Dimensions + j;
            for (IndexType i = 0; i < Regs; ++i)
            {
                I8 v;
                std::memcpy(&v, w + i * Lanes, sizeof(I8));
                r[i] -= __builtin_convertvector(v, I16);
            }
        }
        for (int k = 0; k < nAdded; ++k)
        {
            const ThreatWeightType* w = weights + std::size_t(added[k]) * Dimensions + j;
            for (IndexType i = 0; i < Regs; ++i)
            {
                I8 v;
                std::memcpy(&v, w + i * Lanes, sizeof(I8));
                r[i] += __builtin_convertvector(v, I16);
            }
        }

        for (IndexType i = 0; i < Regs; ++i)
            std::memcpy(dst + j + i * Lanes, &r[i], sizeof(I16));
    }
#else
    for (IndexType j = 0; j < Dimensions; ++j)
        dst[j] = src ? src[j] : 0;
    for (int k = 0; k < nRemoved; ++k)
    {
        const ThreatWeightType* w = weights + std::size_t(removed[k]) * Dimensions;
        for (IndexType j = 0; j < Dimensions; ++j)
            dst[j] = std::int16_t(dst[j] - w[j]);
    }
    for (int k = 0; k < nAdded; ++k)
    {
        const ThreatWeightType* w = weights + std::size_t(added[k]) * Dimensions;
        for (IndexType j = 0; j < Dimensions; ++j)
            dst[j] = std::int16_t(dst[j] + w[j]);
    }
#endif

    for (IndexType b = 0; b < PSQTBuckets; ++b)
        dstPsqt[b] = srcPsqt ? srcPsqt[b] : 0;
    for (int k = 0; k < nRemoved; ++k)
        for (IndexType b = 0; b < PSQTBuckets; ++b)
            dstPsqt[b] -= featureTransformer.threatPsqtWeights[removed[k] * PSQTBuckets + b];
    for (int k = 0; k < nAdded; ++k)
        for (IndexType b = 0; b < PSQTBuckets; ++b)
            dstPsqt[b] += featureTransformer.threatPsqtWeights[added[k] * PSQTBuckets + b];
}

// Feature indices of a list of threats (those that are features). Branchless:
// the index is always written, and kept if it is a feature.
template<Color Perspective>
int threat_indices(const ThreatFeatureSet::Threat* threats, int n, bool mirror, IndexType* out) {
    int m = 0;
    for (int k = 0; k < n; ++k)
    {
        const IndexType index = ThreatFeatureSet::index<Perspective>(threats[k], mirror);
        out[m]                = index;
        m += index < ThreatFeatureSet::Dimensions;
    }
    return m;
}

}  // namespace

template<IndexType Dimensions>
void AccumulatorStack::evaluate_threats(
  const Position& pos, const FeatureTransformer<Dimensions>& featureTransformer) noexcept {

    auto& target = mut_latest().threats;
    if (target.computed[WHITE] && target.computed[BLACK])
        return;

    if (!target.listValid)
    {
        target.count     = std::uint16_t(ThreatFeatureSet::physical_threats(pos, target.list));
        target.listValid = true;
    }

    if (!target.computed[WHITE])
        evaluate_threats_side<WHITE>(pos, featureTransformer);
    if (!target.computed[BLACK])
        evaluate_threats_side<BLACK>(pos, featureTransformer);
}

template<Color Perspective, IndexType Dimensions>
void AccumulatorStack::evaluate_threats_side(
  const Position& pos, const FeatureTransformer<Dimensions>& featureTransformer) noexcept {

    constexpr int MaxActive = ThreatFeatureSet::MaxActiveDimensions;

    auto&      target = mut_latest().threats;
    const bool mirror = FeatureSet::make_feature_bucket(pos, Perspective).second;
    const int  count  = target.count;

    // Nearest earlier state of the stack with a computed threat accumulator
    // (with the same mirroring, otherwise every feature changes)
    const ThreatAccumulator<Dimensions>* source = nullptr;
    for (std::size_t i = size - 1; i-- > 0;)
        if (accumulators[i].threats.computed[Perspective])
        {
            if (accumulators[i].threats.mirror[Perspective] == mirror)
                source = &accumulators[i].threats;
            break;
        }

    // Difference between the two sorted lists of threats
    ThreatFeatureSet::Threat removed[MaxActive], added[MaxActive];
    int                      nRemoved = 0, nAdded = 0;
    if (source)
    {
        const auto* old  = source->list;
        const auto* list = target.list;
        const int   nOld = source->count;
        int         i = 0, j = 0;
        while (i < nOld || j < count)
        {
            if (j == count || (i < nOld && old[i] < list[j]))
                removed[nRemoved++] = old[i++];
            else if (i == nOld || list[j] < old[i])
                added[nAdded++] = list[j++];
            else
                ++i, ++j;
        }
        if (nRemoved + nAdded > count)
            source = nullptr;  // a refresh is cheaper
    }

    IndexType removedIdx[MaxActive], addedIdx[MaxActive];
    if (source)
    {
        nRemoved = threat_indices<Perspective>(removed, nRemoved, mirror, removedIdx);
        nAdded   = threat_indices<Perspective>(added, nAdded, mirror, addedIdx);
        apply_threats(target.acc[Perspective], target.psqt[Perspective],
                      source->acc[Perspective], source->psqt[Perspective], featureTransformer,
                      removedIdx, nRemoved, addedIdx, nAdded);
    }
    else
    {
        nAdded = threat_indices<Perspective>(target.list, count, mirror, addedIdx);
        apply_threats(target.acc[Perspective], target.psqt[Perspective], nullptr, nullptr,
                      featureTransformer, removedIdx, 0, addedIdx, nAdded);
    }

    target.computed[Perspective] = true;
    target.mirror[Perspective]   = mirror;

#ifndef NDEBUG
    // Debug builds: the incremental update must give the same accumulator as
    // a computation from scratch
    if (source)
    {
        auto      fresh = std::make_unique<ThreatAccumulator<Dimensions>>();
        const int n     = threat_indices<Perspective>(target.list, count, mirror, addedIdx);
        apply_threats(fresh->acc[Perspective], fresh->psqt[Perspective], nullptr, nullptr,
                      featureTransformer, removedIdx, 0, addedIdx, n);
        assert(!std::memcmp(fresh->acc[Perspective], target.acc[Perspective],
                            sizeof(target.acc[Perspective])));
        assert(!std::memcmp(fresh->psqt[Perspective], target.psqt[Perspective],
                            sizeof(target.psqt[Perspective])));
    }
#endif
}
#endif

template<Color Perspective, IndexType Dimensions>
void AccumulatorStack::evaluate_side(const Position&                       pos,
                                     const FeatureTransformer<Dimensions>& featureTransformer,
                                     AccumulatorCaches::Cache<Dimensions>& cache) noexcept {

    const auto last_usable_accum = find_last_usable_accumulator<Perspective, Dimensions>();

    if ((accumulators[last_usable_accum].template acc<Dimensions>()).computed[Perspective])
        forward_update_incremental<Perspective>(pos, featureTransformer, last_usable_accum);

    else
    {
        update_accumulator_refresh_cache<Perspective>(featureTransformer, pos, mut_latest(), cache);
        backward_update_incremental<Perspective>(pos, featureTransformer, last_usable_accum);
    }
}

// Find the earliest usable accumulator, this can either be a computed accumulator or the accumulator
// state just before a change that requires full refresh.
template<Color Perspective, IndexType Dimensions>
std::size_t AccumulatorStack::find_last_usable_accumulator() const noexcept {

    for (std::size_t curr_idx = size - 1; curr_idx > 0; curr_idx--)
    {
        if ((accumulators[curr_idx].template acc<Dimensions>()).computed[Perspective])
            return curr_idx;

        if (FeatureSet::requires_refresh(accumulators[curr_idx].dirtyPiece, Perspective))
            return curr_idx;
    }

    return 0;
}

template<Color Perspective, IndexType Dimensions>
void AccumulatorStack::forward_update_incremental(
  const Position&                       pos,
  const FeatureTransformer<Dimensions>& featureTransformer,
  const std::size_t                     begin) noexcept {

    assert(begin < accumulators.size());
    assert((accumulators[begin].acc<Dimensions>()).computed[Perspective]);

    auto [bucket, mirror] = FeatureSet::make_feature_bucket(pos, Perspective);

    for (std::size_t next = begin + 1; next < size; next++)
    {
        if (next + 1 < size)
        {
            DirtyPiece& dp1 = accumulators[next].dirtyPiece;
            DirtyPiece& dp2 = accumulators[next + 1].dirtyPiece;

            if (dp1.pc != DARK_PIECE && dp1.to == dp2.remove_sq)
            {
                const Square captureSq = dp1.to;
                dp1.to = dp2.remove_sq = SQ_NONE;
                double_inc_update<Perspective>(featureTransformer, bucket, mirror,
                                               accumulators[next], accumulators[next + 1],
                                               accumulators[next - 1]);
                dp1.to = dp2.remove_sq = captureSq;

                next++;
                continue;
            }

            if (dp1.pc == DARK_PIECE && dp1.add_sq == dp2.remove_sq)
            {
                const Square captureSq = dp1.add_sq;
                dp1.add_sq = dp2.remove_sq = SQ_NONE;
                double_inc_update<Perspective>(featureTransformer, bucket, mirror,
                                               accumulators[next], accumulators[next + 1],
                                               accumulators[next - 1]);
                dp1.add_sq = dp2.remove_sq = captureSq;

                next++;
                continue;
            }
        }
        update_accumulator_incremental<Perspective, true>(
          featureTransformer, bucket, mirror, accumulators[next], accumulators[next - 1]);
    }

    assert((latest().acc<Dimensions>()).computed[Perspective]);
}

template<Color Perspective, IndexType Dimensions>
void AccumulatorStack::backward_update_incremental(
  const Position&                       pos,
  const FeatureTransformer<Dimensions>& featureTransformer,
  const std::size_t                     end) noexcept {

    assert(end < accumulators.size());
    assert(end < size);
    assert((latest().acc<Dimensions>()).computed[Perspective]);

    auto [bucket, mirror] = FeatureSet::make_feature_bucket(pos, Perspective);

    for (std::int64_t next = std::int64_t(size) - 2; next >= std::int64_t(end); next--)
        update_accumulator_incremental<Perspective, false>(
          featureTransformer, bucket, mirror, accumulators[next], accumulators[next + 1]);

    assert((accumulators[end].acc<Dimensions>()).computed[Perspective]);
}

// Explicit template instantiations
template void AccumulatorStack::evaluate<TransformedFeatureDimensionsBig>(
  const Position&                                            pos,
  const FeatureTransformer<TransformedFeatureDimensionsBig>& featureTransformer,
  AccumulatorCaches::Cache<TransformedFeatureDimensionsBig>& cache) noexcept;


namespace {

template<typename VectorWrapper,
         IndexType Width,
         UpdateOperation... ops,
         typename ElementType,
         typename... Ts,
         std::enable_if_t<is_all_same_v<ElementType, Ts...>, bool> = true>
void fused_row_reduce(const ElementType* in, ElementType* out, const Ts* const... rows) {
    constexpr IndexType size = Width * sizeof(ElementType) / sizeof(typename VectorWrapper::type);

    auto* vecIn  = reinterpret_cast<const typename VectorWrapper::type*>(in);
    auto* vecOut = reinterpret_cast<typename VectorWrapper::type*>(out);

    for (IndexType i = 0; i < size; ++i)
        vecOut[i] = fused<VectorWrapper, ops...>(
          vecIn[i], reinterpret_cast<const typename VectorWrapper::type*>(rows)[i]...);
}

template<Color Perspective, IndexType Dimensions>
struct AccumulatorUpdateContext {
    const FeatureTransformer<Dimensions>& featureTransformer;
    const AccumulatorState&               from;
    AccumulatorState&                     to;

    AccumulatorUpdateContext(const FeatureTransformer<Dimensions>& ft,
                             const AccumulatorState&               accF,
                             AccumulatorState&                     accT) noexcept :
        featureTransformer{ft},
        from{accF},
        to{accT} {}

    template<UpdateOperation... ops,
             typename... Ts,
             std::enable_if_t<is_all_same_v<IndexType, Ts...>, bool> = true>
    void apply(const Ts... indices) {
        auto to_weight_vector = [&](const IndexType index) {
            return &featureTransformer.weights[index * Dimensions];
        };

        auto to_psqt_weight_vector = [&](const IndexType index) {
            return &featureTransformer.psqtWeights[index * PSQTBuckets];
        };

        fused_row_reduce<Vec16Wrapper, Dimensions, ops...>(
          (from.acc<Dimensions>()).accumulation[Perspective],
          (to.acc<Dimensions>()).accumulation[Perspective], to_weight_vector(indices)...);

        fused_row_reduce<Vec32Wrapper, PSQTBuckets, ops...>(
          (from.acc<Dimensions>()).psqtAccumulation[Perspective],
          (to.acc<Dimensions>()).psqtAccumulation[Perspective], to_psqt_weight_vector(indices)...);
    }
};

template<Color Perspective, IndexType Dimensions>
auto make_accumulator_update_context(const FeatureTransformer<Dimensions>& featureTransformer,
                                     const AccumulatorState&               accumulatorFrom,
                                     AccumulatorState&                     accumulatorTo) noexcept {
    return AccumulatorUpdateContext<Perspective, Dimensions>{featureTransformer, accumulatorFrom,
                                                             accumulatorTo};
}

template<Color Perspective, IndexType TransformedFeatureDimensions>
void double_inc_update(const FeatureTransformer<TransformedFeatureDimensions>& featureTransformer,
                       const int                                               bucket,
                       const bool                                              mirror,
                       AccumulatorState&                                       middle_state,
                       AccumulatorState&                                       target_state,
                       const AccumulatorState&                                 computed) {

    assert(computed.acc<TransformedFeatureDimensions>().computed[Perspective]);
    assert(!middle_state.acc<TransformedFeatureDimensions>().computed[Perspective]);
    assert(!target_state.acc<TransformedFeatureDimensions>().computed[Perspective]);

    FeatureSet::IndexList removed, added;
    FeatureSet::append_changed_indices<Perspective>(bucket, mirror, middle_state.dirtyPiece,
                                                    removed, added);
    // you can't capture a piece that was just involved in castling since the rook ends up
    // in a square that the king passed
    assert(added.size() < 2);
    FeatureSet::append_changed_indices<Perspective>(bucket, mirror, target_state.dirtyPiece,
                                                    removed, added);

    assert(added.size() == 1);
    assert(removed.size() == 2 || removed.size() == 3 || removed.size() == 4
           || removed.size() == 5);

    // Workaround compiler warning for uninitialized variables, replicated on
    // profile builds on windows with gcc 14.2.0.
    // TODO remove once unneeded
    sf_assume(added.size() == 1);
    sf_assume(removed.size() == 2 || removed.size() == 3 || removed.size() == 4
              || removed.size() == 5);

    auto updateContext =
      make_accumulator_update_context<Perspective>(featureTransformer, computed, target_state);

    if (removed.size() == 2)
    {
        updateContext.template apply<Add, Sub, Sub>(added[0], removed[0], removed[1]);
    }
    else if (removed.size() == 3)
    {
        updateContext.template apply<Add, Sub, Sub, Sub>(added[0], removed[0], removed[1],
                                                         removed[2]);
    }
    else if (removed.size() == 4)
    {
        updateContext.template apply<Add, Sub, Sub, Sub, Sub>(added[0], removed[0], removed[1],
                                                              removed[2], removed[3]);
    }
    else if (removed.size() == 5)
    {
        updateContext.template apply<Add, Sub, Sub, Sub, Sub, Sub>(
          added[0], removed[0], removed[1], removed[2], removed[3], removed[4]);
    }

    target_state.acc<TransformedFeatureDimensions>().computed[Perspective] = true;
}

template<Color Perspective, bool Forward, IndexType TransformedFeatureDimensions>
void update_accumulator_incremental(
  const FeatureTransformer<TransformedFeatureDimensions>& featureTransformer,
  const int                                               bucket,
  const bool                                              mirror,
  AccumulatorState&                                       target_state,
  const AccumulatorState&                                 computed) {

    assert((computed.acc<TransformedFeatureDimensions>()).computed[Perspective]);
    assert(!(target_state.acc<TransformedFeatureDimensions>()).computed[Perspective]);

    // The size must be enough to contain the largest possible update.
    // That might depend on the feature set and generally relies on the
    // feature set's update cost calculation to be correct and never allow
    // updates with more added/removed features than MaxActiveDimensions.
    // In this case, the maximum size of both feature addition and removal
    // is 2, since we are incrementally updating one move at a time.
    FeatureSet::IndexList removed, added;
    if constexpr (Forward)
        FeatureSet::append_changed_indices<Perspective>(bucket, mirror, target_state.dirtyPiece,
                                                        removed, added);
    else
        FeatureSet::append_changed_indices<Perspective>(bucket, mirror, computed.dirtyPiece, added,
                                                        removed);

    assert(added.size() == 1 || added.size() == 2 || added.size() == 3);
    assert(removed.size() == 1 || removed.size() == 2 || removed.size() == 3);
    assert((Forward && added.size() <= removed.size())
           || (!Forward && added.size() >= removed.size()));

    // Workaround compiler warning for uninitialized variables, replicated on
    // profile builds on windows with gcc 14.2.0.
    // TODO remove once unneeded
    sf_assume(added.size() == 1 || added.size() == 2 || added.size() == 3);
    sf_assume(removed.size() == 1 || removed.size() == 2 || removed.size() == 3);

    auto updateContext =
      make_accumulator_update_context<Perspective>(featureTransformer, computed, target_state);

    if ((Forward && removed.size() == 1) || (!Forward && added.size() == 1))
    {
        assert(added.size() == 1 && removed.size() == 1);
        updateContext.template apply<Add, Sub>(added[0], removed[0]);
    }
    else if (Forward && added.size() == 1 && removed.size() == 2)
    {
        updateContext.template apply<Add, Sub, Sub>(added[0], removed[0], removed[1]);
    }
    else if (Forward && added.size() == 1 && removed.size() == 3)
    {
        updateContext.template apply<Add, Sub, Sub, Sub>(added[0], removed[0], removed[1],
                                                         removed[2]);
    }
    else if (!Forward && removed.size() == 1 && added.size() == 2)
    {
        updateContext.template apply<Add, Add, Sub>(added[0], added[1], removed[0]);
    }
    else if (!Forward && removed.size() == 1 && added.size() == 3)
    {
        updateContext.template apply<Add, Add, Add, Sub>(added[0], added[1], added[2], removed[0]);
    }
    else
    {
        assert(added.size() == 2 && removed.size() == 2);
        updateContext.template apply<Add, Add, Sub, Sub>(added[0], added[1], removed[0],
                                                         removed[1]);
    }

    (target_state.acc<TransformedFeatureDimensions>()).computed[Perspective] = true;
}

template<Color Perspective, IndexType Dimensions>
void update_accumulator_refresh_cache(const FeatureTransformer<Dimensions>& featureTransformer,
                                      const Position&                       pos,
                                      AccumulatorState&                     accumulatorState,
                                      AccumulatorCaches::Cache<Dimensions>& cache) {

    using Tiling [[maybe_unused]] = SIMDTiling<Dimensions, Dimensions, PSQTBuckets>;

    auto [bucket, mirror] = FeatureSet::make_feature_bucket(pos, Perspective);

    auto cache_index = FeatureSet::make_cache_index(pos, Perspective);

    auto&                 entry = cache[cache_index][Perspective];
    FeatureSet::IndexList removed, added;

    for (Color c : {WHITE, BLACK})
    {
        for (PieceType pt = ROOK; pt <= DARK; ++pt)
        {
            const Piece piece = pt == DARK ? DARK_PIECE : make_piece(c, pt);
            Bitboard    oldBB = entry.byColorBB[c] & entry.byTypeBB[pt];
            Bitboard    newBB = pos.pieces(c, pt);
            if (pt != DARK)
            {
                oldBB &= ~entry.byTypeBB[DARK];
                newBB &= ~pos.pieces(DARK);

                while (entry.restPieces[piece] > pos.rest_piece(piece))
                {
                    removed.push_back(FeatureSet::make_dark_rest_index<Perspective>(piece, bucket));
                    --entry.restPieces[piece];
                }

                while (entry.restPieces[piece] < pos.rest_piece(piece))
                {
                    added.push_back(FeatureSet::make_dark_rest_index<Perspective>(piece, bucket));
                    ++entry.restPieces[piece];
                }
            }
            Bitboard toRemove = oldBB & ~newBB;
            Bitboard toAdd    = newBB & ~oldBB;

            while (toRemove)
            {
                Square sq = pop_lsb(toRemove);
                removed.push_back(FeatureSet::make_index<Perspective>(sq, piece, bucket, mirror));
            }
            while (toAdd)
            {
                Square sq = pop_lsb(toAdd);
                added.push_back(FeatureSet::make_index<Perspective>(sq, piece, bucket, mirror));
            }
        }
    }

    auto& accumulator                 = accumulatorState.acc<Dimensions>();
    accumulator.computed[Perspective] = true;

#ifdef VECTOR
    vec_t      acc[Tiling::NumRegs];
    psqt_vec_t psqt[Tiling::NumPsqtRegs];

    for (IndexType j = 0; j < Dimensions / Tiling::TileHeight; ++j)
    {
        auto* accTile =
          reinterpret_cast<vec_t*>(&accumulator.accumulation[Perspective][j * Tiling::TileHeight]);
        auto* entryTile = reinterpret_cast<vec_t*>(&entry.accumulation[j * Tiling::TileHeight]);

        for (IndexType k = 0; k < Tiling::NumRegs; ++k)
            acc[k] = entryTile[k];

        IndexType i = 0;
        for (; i < std::min(removed.size(), added.size()); ++i)
        {
            IndexType       indexR  = removed[i];
            const IndexType offsetR = Dimensions * indexR + j * Tiling::TileHeight;
            auto* columnR = reinterpret_cast<const vec_t*>(&featureTransformer.weights[offsetR]);
            IndexType       indexA  = added[i];
            const IndexType offsetA = Dimensions * indexA + j * Tiling::TileHeight;
            auto* columnA = reinterpret_cast<const vec_t*>(&featureTransformer.weights[offsetA]);

            for (IndexType k = 0; k < Tiling::NumRegs; ++k)
                acc[k] = fused<Vec16Wrapper, Add, Sub>(acc[k], columnA[k], columnR[k]);
        }
        for (; i < removed.size(); ++i)
        {
            IndexType       index  = removed[i];
            const IndexType offset = Dimensions * index + j * Tiling::TileHeight;
            auto* column = reinterpret_cast<const vec_t*>(&featureTransformer.weights[offset]);

            for (IndexType k = 0; k < Tiling::NumRegs; ++k)
                acc[k] = vec_sub_16(acc[k], column[k]);
        }
        for (; i < added.size(); ++i)
        {
            IndexType       index  = added[i];
            const IndexType offset = Dimensions * index + j * Tiling::TileHeight;
            auto* column = reinterpret_cast<const vec_t*>(&featureTransformer.weights[offset]);

            for (IndexType k = 0; k < Tiling::NumRegs; ++k)
                acc[k] = vec_add_16(acc[k], column[k]);
        }

        for (IndexType k = 0; k < Tiling::NumRegs; k++)
            vec_store(&entryTile[k], acc[k]);
        for (IndexType k = 0; k < Tiling::NumRegs; k++)
            vec_store(&accTile[k], acc[k]);
    }

    for (IndexType j = 0; j < PSQTBuckets / Tiling::PsqtTileHeight; ++j)
    {
        auto* accTilePsqt = reinterpret_cast<psqt_vec_t*>(
          &accumulator.psqtAccumulation[Perspective][j * Tiling::PsqtTileHeight]);
        auto* entryTilePsqt =
          reinterpret_cast<psqt_vec_t*>(&entry.psqtAccumulation[j * Tiling::PsqtTileHeight]);

        for (IndexType k = 0; k < Tiling::NumPsqtRegs; ++k)
            psqt[k] = entryTilePsqt[k];

        for (IndexType i = 0; i < removed.size(); ++i)
        {
            IndexType       index  = removed[i];
            const IndexType offset = PSQTBuckets * index + j * Tiling::PsqtTileHeight;
            auto*           columnPsqt =
              reinterpret_cast<const psqt_vec_t*>(&featureTransformer.psqtWeights[offset]);

            for (std::size_t k = 0; k < Tiling::NumPsqtRegs; ++k)
                psqt[k] = vec_sub_psqt_32(psqt[k], columnPsqt[k]);
        }
        for (IndexType i = 0; i < added.size(); ++i)
        {
            IndexType       index  = added[i];
            const IndexType offset = PSQTBuckets * index + j * Tiling::PsqtTileHeight;
            auto*           columnPsqt =
              reinterpret_cast<const psqt_vec_t*>(&featureTransformer.psqtWeights[offset]);

            for (std::size_t k = 0; k < Tiling::NumPsqtRegs; ++k)
                psqt[k] = vec_add_psqt_32(psqt[k], columnPsqt[k]);
        }

        for (IndexType k = 0; k < Tiling::NumPsqtRegs; ++k)
            vec_store_psqt(&entryTilePsqt[k], psqt[k]);
        for (IndexType k = 0; k < Tiling::NumPsqtRegs; ++k)
            vec_store_psqt(&accTilePsqt[k], psqt[k]);
    }

#else

    for (const auto index : removed)
    {
        const IndexType offset = Dimensions * index;
        for (IndexType j = 0; j < Dimensions; ++j)
            entry.accumulation[j] -= featureTransformer.weights[offset + j];

        for (std::size_t k = 0; k < PSQTBuckets; ++k)
            entry.psqtAccumulation[k] -= featureTransformer.psqtWeights[index * PSQTBuckets + k];
    }
    for (const auto index : added)
    {
        const IndexType offset = Dimensions * index;
        for (IndexType j = 0; j < Dimensions; ++j)
            entry.accumulation[j] += featureTransformer.weights[offset + j];

        for (std::size_t k = 0; k < PSQTBuckets; ++k)
            entry.psqtAccumulation[k] += featureTransformer.psqtWeights[index * PSQTBuckets + k];
    }

    // The accumulator of the refresh entry has been updated.
    // Now copy its content to the actual accumulator we were refreshing.

    std::memcpy(accumulator.accumulation[Perspective], entry.accumulation,
                sizeof(BiasType) * Dimensions);

    std::memcpy(accumulator.psqtAccumulation[Perspective], entry.psqtAccumulation,
                sizeof(int32_t) * PSQTBuckets);
#endif

    for (Color c : {WHITE, BLACK})
        entry.byColorBB[c] = pos.pieces(c);

    for (PieceType pt = ROOK; pt <= DARK; ++pt)
        entry.byTypeBB[pt] = pos.pieces(pt);
}

}

}
