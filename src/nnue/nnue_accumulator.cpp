/*
  Maskfish, a UCI jieqi engine
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

#include "nnue_accumulator.h"

#include <algorithm>
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

namespace Maskfish::Eval::NNUE {

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

#ifdef JIEQI_NNUE_META
    const auto& computed = (latest().template acc<Dimensions>()).computed;
    const bool  needMeta = !computed[WHITE] || !computed[BLACK];
#endif
    evaluate_side<WHITE>(pos, featureTransformer, cache);
    evaluate_side<BLACK>(pos, featureTransformer, cache);
#ifdef JIEQI_NNUE_THREATS
    evaluate_threats(pos, featureTransformer);
#endif
#ifdef JIEQI_NNUE_META
    // The meta features are added to the HalfKA accumulator: evaluate_side()
    // leaves in the latest state those of the state it was updated from (or
    // none after a refresh), replaced here by those of the position
    if (needMeta)
    {
        const int bucket[COLOR_NB] = {FeatureSet::make_feature_bucket(pos, WHITE).first,
                                      FeatureSet::make_feature_bucket(pos, BLACK).first};
        IndexType metaIndex[COLOR_NB][FeatureSet::MetaFeatures];
        FeatureSet::make_meta_indices(pos, bucket, metaIndex);
        update_meta<WHITE>(featureTransformer, metaIndex[WHITE]);
        update_meta<BLACK>(featureTransformer, metaIndex[BLACK]);
    }
#endif
}

#if defined(JIEQI_NNUE_THREATS) || defined(JIEQI_NNUE_META)
namespace {

    #if defined(USE_PAIRED_WIDENING_ADD) && defined(JIEQI_NNUE_THREATS)
// Rows of 8-bit threat weights to add to or to subtract from an accumulator
// (paired int8 weight rows). They are applied
// two at a time with vec_widen_pair_i8() (one shuffle per two rows instead of
// one sign extension per row), the last one alone if their number is odd. The
// 16-bit sums wrap around like the separate additions they replace, so the
// accumulator is the same whatever the order. A threat update has many rows
// of each sign, unlike an incremental HalfKA update (mostly one of each), for
// which this is not done.
template<int MaxRows>
class WeightRows {
   public:
    void push(const std::int8_t* row) {
        assert(count < MaxRows);
        rows[count++] = row;
    }

    // acc[0 .. 2 * NumVecs): the 16-bit lanes of the NumVecs * sizeof(vec_t)
    // weights at the offset j of every row
    template<int Sign, int NumVecs>
    void apply(IndexType j, vec_t* acc) const {
        static_assert(Sign == 1 || Sign == -1);
        auto update = [](vec_t& a, const vec_t& w) {
            if constexpr (Sign > 0)
                a = vec_add_16(a, w);
            else
                a = vec_sub_16(a, w);
        };

        int i = 0;
        for (; i + 1 < count; i += 2)
            for (int v = 0; v < NumVecs; ++v)
            {
                vec_t lo, hi;
                vec_widen_pair_i8(rows[i] + j + v * sizeof(vec_t),
                                  rows[i + 1] + j + v * sizeof(vec_t), lo, hi);
                update(acc[2 * v], lo);
                update(acc[2 * v + 1], hi);
            }
        if (i < count)
            for (int v = 0; v < NumVecs; ++v)
            {
                vec_t lo, hi;
                vec_widen_i8(rows[i] + j + v * sizeof(vec_t), lo, hi);
                update(acc[2 * v], lo);
                update(acc[2 * v + 1], hi);
            }
    }

   private:
    const std::int8_t* rows[MaxRows];
    int                count = 0;
};

// dst = src - the removed rows + the added rows (src == nullptr: zero), over
// the Dimensions 16-bit lanes of an accumulator, by tiles kept in registers
template<IndexType Dimensions, int MaxRows>
void apply_weight_rows(std::int16_t*              dst,
                       const std::int16_t*        src,
                       const WeightRows<MaxRows>& removed,
                       const WeightRows<MaxRows>& added) {
    // Accumulator registers of a tile: the widening needs 4 more registers
    // (two rows, ones, zero), so at most 12 of the 16 vector registers below
    // AVX-512 (16 of 32 with it)
    constexpr int NumRegs = []() {
        constexpr int Ideal   = int(Dimensions * sizeof(std::int16_t) / sizeof(vec_t));
        constexpr int MaxRegs = sizeof(vec_t) == 64 ? 16 : 12;
        int           n       = std::min(Ideal, MaxRegs);
        while (Ideal % n != 0 || n % 2 != 0)
            --n;
        return n;
    }();
    constexpr IndexType Height = NumRegs * sizeof(vec_t) / sizeof(std::int16_t);
    static_assert(NumRegs >= 2 && Dimensions % Height == 0);

    for (IndexType j = 0; j < Dimensions; j += Height)
    {
        vec_t acc[NumRegs];
        for (int k = 0; k < NumRegs; ++k)
            acc[k] = src ? vec_load(reinterpret_cast<const vec_t*>(src + j) + k) : vec_zero();

        removed.template apply<-1, NumRegs / 2>(j, acc);
        added.template apply<+1, NumRegs / 2>(j, acc);

        for (int k = 0; k < NumRegs; ++k)
            vec_store(reinterpret_cast<vec_t*>(dst + j) + k, acc[k]);
    }
}

// The 16-bit part of apply_rows() for weights permuted for vec_widen_pair_i8()
template<IndexType Dimensions>
void apply_rows_paired(std::int16_t*       dst,
                       const std::int16_t* src,
                       const std::int8_t*  weights,
                       const IndexType*    removed,
                       int                 nRemoved,
                       const IndexType*    added,
                       int                 nAdded) {
    constexpr int MaxRows = int(ThreatFeatureSet::MaxActiveDimensions);
    assert(nRemoved <= MaxRows && nAdded <= MaxRows);
    WeightRows<MaxRows> rowsRemoved, rowsAdded;
    for (int k = 0; k < nRemoved; ++k)
        rowsRemoved.push(weights + std::size_t(removed[k]) * Dimensions);
    for (int k = 0; k < nAdded; ++k)
        rowsAdded.push(weights + std::size_t(added[k]) * Dimensions);
    apply_weight_rows<Dimensions>(dst, src, rowsRemoved, rowsAdded);
}
    #else
// (no paired rows on this target: never instantiated)
template<IndexType Dimensions>
void apply_rows_paired(std::int16_t*,
                       const std::int16_t*,
                       const std::int8_t*,
                       const IndexType*,
                       int,
                       const IndexType*,
                       int) {
    static_assert(Dimensions == 0, "paired weight rows need SSSE3");
}
    #endif

// The 16-bit part of apply_rows(), weights in the order of the accumulator
template<IndexType Dimensions>
void apply_rows_16(std::int16_t*       dst,
                   const std::int16_t* src,
                   const std::int8_t* __restrict weights,
                   const IndexType* removed,
                   int              nRemoved,
                   const IndexType* added,
                   int              nAdded) {
#if defined(VECTOR)
    // The SIMD helpers of the feature transformer: one sign-extending load
    // (vpmovsxbw) per vector of weights, the accumulator kept in registers by
    // tiles of Regs vectors (both accumulators are aligned to the cache line)
    constexpr IndexType Lanes = sizeof(vec_t) / sizeof(std::int16_t);
    constexpr IndexType Regs  = 8;
    constexpr IndexType Tile  = Lanes * Regs;
    static_assert(Dimensions % Tile == 0);

    for (IndexType j = 0; j < Dimensions; j += Tile)
    {
        vec_t r[Regs];
        if (src)
            for (IndexType i = 0; i < Regs; ++i)
                r[i] = vec_load(reinterpret_cast<const vec_t*>(src + j) + i);
        else
            for (IndexType i = 0; i < Regs; ++i)
                r[i] = vec_zero();

        for (int k = 0; k < nRemoved; ++k)
        {
            const ThreatWeightType* w = weights + std::size_t(removed[k]) * Dimensions + j;
            for (IndexType i = 0; i < Regs; ++i)
                r[i] = vec_sub_16(r[i], vec_load_i8_16(w + i * Lanes));
        }
        for (int k = 0; k < nAdded; ++k)
        {
            const ThreatWeightType* w = weights + std::size_t(added[k]) * Dimensions + j;
            for (IndexType i = 0; i < Regs; ++i)
                r[i] = vec_add_16(r[i], vec_load_i8_16(w + i * Lanes));
        }

        for (IndexType i = 0; i < Regs; ++i)
            vec_store(reinterpret_cast<vec_t*>(dst + j) + i, r[i]);
    }
#elif defined(__GNUC__)
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
            const std::int8_t* w = weights + std::size_t(removed[k]) * Dimensions + j;
            for (IndexType i = 0; i < Regs; ++i)
            {
                I8 v;
                std::memcpy(&v, w + i * Lanes, sizeof(I8));
                r[i] -= __builtin_convertvector(v, I16);
            }
        }
        for (int k = 0; k < nAdded; ++k)
        {
            const std::int8_t* w = weights + std::size_t(added[k]) * Dimensions + j;
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
        const std::int8_t* w = weights + std::size_t(removed[k]) * Dimensions;
        for (IndexType j = 0; j < Dimensions; ++j)
            dst[j] = std::int16_t(dst[j] - w[j]);
    }
    for (int k = 0; k < nAdded; ++k)
    {
        const std::int8_t* w = weights + std::size_t(added[k]) * Dimensions;
        for (IndexType j = 0; j < Dimensions; ++j)
            dst[j] = std::int16_t(dst[j] + w[j]);
    }
#endif
}

// dst = src - sum of the `removed` rows + sum of the `added` rows of the given
// weights (src == nullptr: zero; src == dst: in place), for the accumulator
// and the PSQT values. The
// accumulator is processed by tiles kept in registers, so that it is read and
// written once whatever the number of features. Paired: the rows are permuted
// for vec_widen_pair_i8() (the threat weights with USE_PAIRED_WIDENING_ADD,
// see FeatureTransformer::WideningPairOrder).
template<IndexType Dimensions, bool Paired = false>
void apply_rows(std::int16_t*       dst,
                std::int32_t*       dstPsqt,
                const std::int16_t* src,
                const std::int32_t* srcPsqt,
                const std::int8_t* __restrict weights,
                const PSQTWeightType* __restrict psqtWeights,
                const IndexType* removed,
                int              nRemoved,
                const IndexType* added,
                int              nAdded) {

    if constexpr (Paired)
        apply_rows_paired<Dimensions>(dst, src, weights, removed, nRemoved, added, nAdded);
    else
        apply_rows_16<Dimensions>(dst, src, weights, removed, nRemoved, added, nAdded);

    if constexpr (!UsePSQT)
        return;

    for (IndexType b = 0; b < PSQTBuckets; ++b)
        dstPsqt[b] = srcPsqt ? srcPsqt[b] : 0;
    for (int k = 0; k < nRemoved; ++k)
        for (IndexType b = 0; b < PSQTBuckets; ++b)
            dstPsqt[b] -= psqtWeights[std::size_t(removed[k]) * PSQTBuckets + b];
    for (int k = 0; k < nAdded; ++k)
        for (IndexType b = 0; b < PSQTBuckets; ++b)
            dstPsqt[b] += psqtWeights[std::size_t(added[k]) * PSQTBuckets + b];
}

}  // namespace
#endif

#ifdef JIEQI_NNUE_THREATS
namespace {

// apply_rows() with the threat weights
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
    static_assert(sizeof(ThreatWeightType) == 1);
    // (with USE_PAIRED_WIDENING_ADD the threat weights are permuted for it)
    constexpr bool Paired =
    #if defined(USE_PAIRED_WIDENING_ADD)
      true;
    #else
      false;
    #endif
    apply_rows<Dimensions, Paired>(
      dst, dstPsqt, src, srcPsqt,
      reinterpret_cast<const std::int8_t*>(featureTransformer.threatWeights),
      featureTransformer.threatPsqtWeights, removed, nRemoved, added, nAdded);
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

    // UCI option ThreatsIncremental: from the threats recorded by the moves
    if (incrementalThreats && evaluate_threats_incremental(pos, featureTransformer))
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

// Nearest earlier state with a computed threat accumulator for the
// perspective, with the same mirroring, such that the threats of every move
// (and reveal) since then are recorded; size if there is none.
template<Color Perspective>
std::size_t AccumulatorStack::incremental_threats_source(bool mirror) const noexcept {
    for (std::size_t i = size - 1; i-- > 0;)
    {
        if (!accumulators[i + 1].dirtyThreats.valid)
            return size;
        if (accumulators[i].threats.computed[Perspective])
            return accumulators[i].threats.mirror[Perspective] == mirror ? i : size;
    }
    return size;
}

// Updates the threat accumulators of the latest state that are not computed
// from an earlier state by the threats recorded since then
// (DirtyThreats), both perspectives together when they start from the same
// state. Returns whether both perspectives are computed; a perspective that
// cannot be updated this way (no source, too many changes) is left to
// evaluate_threats_side().
template<IndexType Dimensions>
bool AccumulatorStack::evaluate_threats_incremental(
  const Position& pos, const FeatureTransformer<Dimensions>& featureTransformer) noexcept {

    // Threats recorded since the source: more than this, a refresh is cheaper
    constexpr int MaxRaw     = 2 * ThreatFeatureSet::MaxActiveDimensions;
    constexpr int MaxChanges = ThreatFeatureSet::MaxActiveDimensions;

    auto&      target    = mut_latest().threats;
    const bool mirror[2] = {FeatureSet::make_feature_bucket(pos, WHITE).second,
                            FeatureSet::make_feature_bucket(pos, BLACK).second};
    std::size_t src[2]   = {size, size};

    if (!target.computed[WHITE])
        src[WHITE] = incremental_threats_source<WHITE>(mirror[WHITE]);
    if (!target.computed[BLACK])
        src[BLACK] = incremental_threats_source<BLACK>(mirror[BLACK]);

    IndexType removed[2][MaxChanges], added[2][MaxChanges];
    int       nRemoved[2] = {}, nAdded[2] = {};
    bool      ok[2]       = {src[WHITE] < size, src[BLACK] < size};

    // The recorded threats since `begin`, without the ones that were added
    // and removed again (by the reveal, or by a later move): a threat is
    // recorded at most once in each direction in a row, so after sorting by
    // threat a pair of opposite signs cancels. Their feature indices for the
    // perspectives in (white, black), in one pass for both.
    auto collect = [&](std::size_t begin, bool white, bool black) {
        u32 raw[MaxRaw];
        int n = 0;
        for (std::size_t j = begin + 1; j < size; ++j)
        {
            const DirtyThreats& dts = accumulators[j].dirtyThreats;
            if (n + dts.size > MaxRaw)
            {
                ok[WHITE] &= !white, ok[BLACK] &= !black;
                return;
            }
            std::memcpy(raw + n, dts.list, dts.size * sizeof(u32));
            n += dts.size;
        }

        // Sort by threat (rotating the sign bit to the bottom), then drop
        // the pairs of the same threat with opposite signs
        for (int k = 0; k < n; ++k)
            raw[k] = raw[k] << 1 | raw[k] >> 31;
        std::sort(raw, raw + n);
        int m = 0;
        for (int k = 0; k < n; ++k)
        {
            if (m && (raw[m - 1] ^ raw[k]) == 1)
                --m;  // same threat, opposite signs
            else
                raw[m++] = raw[k];
        }

        for (int k = 0; k < m; ++k)
        {
            const bool add = raw[k] & 1;
            const auto th  = ThreatFeatureSet::Threat(raw[k] >> 1);
            for (Color c : {WHITE, BLACK})
            {
                if (!(c == WHITE ? white : black) || !ok[c])
                    continue;
                const IndexType idx = c == WHITE ? ThreatFeatureSet::index<WHITE>(th, mirror[c])
                                                 : ThreatFeatureSet::index<BLACK>(th, mirror[c]);
                if (idx >= ThreatFeatureSet::Dimensions)
                    continue;
                if (nRemoved[c] + nAdded[c] == MaxChanges)
                    ok[c] = false;
                else if (add)
                    added[c][nAdded[c]++] = idx;
                else
                    removed[c][nRemoved[c]++] = idx;
            }
        }
    };

    if (ok[WHITE] && ok[BLACK] && src[WHITE] == src[BLACK])
        collect(src[WHITE], true, true);
    else
    {
        if (ok[WHITE])
            collect(src[WHITE], true, false);
        if (ok[BLACK])
            collect(src[BLACK], false, true);
    }

    for (Color c : {WHITE, BLACK})
    {
        if (!ok[c])
            continue;

        // A refresh is cheaper than more changes than active features
        const ThreatAccumulator<Dimensions>& source = accumulators[src[c]].threats;
        const int features = source.features[c] + nAdded[c] - nRemoved[c];
        if (nRemoved[c] + nAdded[c] > features)
            continue;
        apply_threats(target.acc[c], target.psqt[c], source.acc[c], source.psqt[c],
                      featureTransformer, removed[c], nRemoved[c], added[c], nAdded[c]);
        target.computed[c] = true;
        target.mirror[c]   = mirror[c];
        target.features[c] = std::uint16_t(features);

#ifndef NDEBUG
        // Debug builds: the same accumulator as a computation from scratch
        ThreatFeatureSet::Threat list[ThreatFeatureSet::MaxActiveDimensions];
        IndexType                idx[ThreatFeatureSet::MaxActiveDimensions];
        const int                count = ThreatFeatureSet::physical_threats(pos, list);
        const int                n     = c == WHITE
                                         ? threat_indices<WHITE>(list, count, mirror[c], idx)
                                         : threat_indices<BLACK>(list, count, mirror[c], idx);
        auto fresh = std::make_unique<ThreatAccumulator<Dimensions>>();
        apply_threats(fresh->acc[c], fresh->psqt[c], nullptr, nullptr, featureTransformer, idx, 0,
                      idx, n);
        assert(!std::memcmp(fresh->acc[c], target.acc[c], sizeof(target.acc[c])));
        assert(!std::memcmp(fresh->psqt[c], target.psqt[c], sizeof(target.psqt[c])));
        assert(n == target.features[c]);
#endif
    }

    return target.computed[WHITE] && target.computed[BLACK];
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
    // (and with its list of threats, which the states updated with
    // ThreatsIncremental do not have)
    for (std::size_t i = size - 1; i-- > 0;)
        if (accumulators[i].threats.computed[Perspective])
        {
            if (accumulators[i].threats.mirror[Perspective] == mirror
                && accumulators[i].threats.listValid)
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
        target.features[Perspective] =
          std::uint16_t(source->features[Perspective] + nAdded - nRemoved);
    }
    else
    {
        nAdded = threat_indices<Perspective>(target.list, count, mirror, addedIdx);
        apply_threats(target.acc[Perspective], target.psqt[Perspective], nullptr, nullptr,
                      featureTransformer, nullptr, 0, addedIdx, nAdded);
        target.features[Perspective] = std::uint16_t(nAdded);
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
                      featureTransformer, nullptr, 0, addedIdx, n);
        assert(!std::memcmp(fresh->acc[Perspective], target.acc[Perspective],
                            sizeof(target.acc[Perspective])));
        assert(!std::memcmp(fresh->psqt[Perspective], target.psqt[Perspective],
                            sizeof(target.psqt[Perspective])));
    }
#endif
}
#endif

#ifdef JIEQI_NNUE_META
template<Color Perspective, IndexType Dimensions>
void AccumulatorStack::update_meta(const FeatureTransformer<Dimensions>& featureTransformer,
                                   const IndexType*                      metaIndex) noexcept {

    constexpr int N = FeatureSet::MetaFeatures;

    auto&      state = mut_latest();
    auto&      acc   = state.template acc<Dimensions>();
    IndexType* old   = state.metaIndex[Perspective];

    IndexType removed[N], added[N];
    int       nRemoved = 0, nAdded = 0;
    for (int k = 0; k < N; ++k)
        if (!state.metaCount[Perspective] || old[k] != metaIndex[k])
        {
            if (state.metaCount[Perspective])
                removed[nRemoved++] = old[k];
            added[nAdded++] = metaIndex[k];
        }

    if (nAdded)
        apply_rows<Dimensions>(acc.accumulation[Perspective], acc.psqtAccumulation[Perspective],
                               acc.accumulation[Perspective], acc.psqtAccumulation[Perspective],
                               featureTransformer.weights, featureTransformer.psqtWeights, removed,
                               nRemoved, added, nAdded);

    std::copy(metaIndex, metaIndex + N, old);
    state.metaCount[Perspective] = N;
}
#endif

template<Color Perspective, IndexType Dimensions>
void AccumulatorStack::evaluate_side(const Position&                       pos,
                                     const FeatureTransformer<Dimensions>& featureTransformer,
                                     AccumulatorCaches::Cache<Dimensions>& cache) noexcept {

    const auto last_usable_accum = find_last_usable_accumulator<Perspective, Dimensions>();

    if ((accumulators[last_usable_accum].template acc<Dimensions>()).computed[Perspective])
    {
        forward_update_incremental<Perspective>(pos, featureTransformer, last_usable_accum);
#ifdef JIEQI_NNUE_META
        // The updated states have the meta features of the first one
        const auto& from = accumulators[last_usable_accum];
        for (std::size_t i = last_usable_accum + 1; i < size; ++i)
        {
            std::copy(std::begin(from.metaIndex[Perspective]), std::end(from.metaIndex[Perspective]),
                      accumulators[i].metaIndex[Perspective]);
            accumulators[i].metaCount[Perspective] = from.metaCount[Perspective];
        }
#endif
    }
    else
    {
        update_accumulator_refresh_cache<Perspective>(featureTransformer, pos, mut_latest(), cache);
        backward_update_incremental<Perspective>(pos, featureTransformer, last_usable_accum);
#ifdef JIEQI_NNUE_META
        // A refresh computes the HalfKA features only
        for (std::size_t i = last_usable_accum; i < size; ++i)
            accumulators[i].metaCount[Perspective] = 0;
#endif
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
        // A capture that removes a piece from the pool (hidden capture
        // model) is updated on its own: the double update is limited to 5
        // removed features.
        if (next + 1 < size && accumulators[next].dirtyPiece.rest_remove_pc == NO_PIECE
            && accumulators[next + 1].dirtyPiece.rest_remove_pc == NO_PIECE)
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

// Same for the 16-bit accumulator and rows of 8-bit feature transformer
// weights, sign-extended on the fly
template<IndexType Width,
         UpdateOperation... ops,
         typename... Ts,
         std::enable_if_t<is_all_same_v<WeightType, Ts...>, bool> = true>
void fused_weight_row_reduce(const BiasType* in, BiasType* out, const Ts* const... rows) {
#ifdef VECTOR
    constexpr IndexType Lanes = sizeof(vec_t) / sizeof(BiasType);
    static_assert(Width % Lanes == 0);

    auto* vecIn  = reinterpret_cast<const vec_t*>(in);
    auto* vecOut = reinterpret_cast<vec_t*>(out);

    for (IndexType i = 0; i < Width / Lanes; ++i)
        vecOut[i] = fused<Vec16Wrapper, ops...>(vecIn[i], vec_load_i8_16(rows + i * Lanes)...);
#else
    for (IndexType i = 0; i < Width; ++i)
        out[i] = fused<Vec16Wrapper, ops...>(in[i], BiasType(rows[i])...);
#endif
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

        fused_weight_row_reduce<Dimensions, ops...>(
          (from.acc<Dimensions>()).accumulation[Perspective],
          (to.acc<Dimensions>()).accumulation[Perspective], to_weight_vector(indices)...);

        if constexpr (UsePSQT)
            fused_row_reduce<Vec32Wrapper, PSQTBuckets, ops...>(
              (from.acc<Dimensions>()).psqtAccumulation[Perspective],
              (to.acc<Dimensions>()).psqtAccumulation[Perspective],
              to_psqt_weight_vector(indices)...);
        else
            (void) to_psqt_weight_vector;
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
    mf_assume(added.size() == 1);
    mf_assume(removed.size() == 2 || removed.size() == 3 || removed.size() == 4
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
    // is 2, since we are incrementally updating one move at a time. (Jieqi: a
    // dark piece that captures a dark piece, is revealed, and whose capture
    // teaches the identity of the captured piece removes 4 features.)
    FeatureSet::IndexList removed, added;
    if constexpr (Forward)
        FeatureSet::append_changed_indices<Perspective>(bucket, mirror, target_state.dirtyPiece,
                                                        removed, added);
    else
        FeatureSet::append_changed_indices<Perspective>(bucket, mirror, computed.dirtyPiece, added,
                                                        removed);

    assert(added.size() >= 1 && added.size() <= 4);
    assert(removed.size() >= 1 && removed.size() <= 4);
    assert((Forward && added.size() <= removed.size())
           || (!Forward && added.size() >= removed.size()));

    // Workaround compiler warning for uninitialized variables, replicated on
    // profile builds on windows with gcc 14.2.0.
    // TODO remove once unneeded
    mf_assume(added.size() >= 1 && added.size() <= 4);
    mf_assume(removed.size() >= 1 && removed.size() <= 4);

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
    else if (Forward && added.size() == 1 && removed.size() == 4)
    {
        updateContext.template apply<Add, Sub, Sub, Sub, Sub>(added[0], removed[0], removed[1],
                                                              removed[2], removed[3]);
    }
    else if (!Forward && removed.size() == 1 && added.size() == 3)
    {
        updateContext.template apply<Add, Add, Add, Sub>(added[0], added[1], added[2], removed[0]);
    }
    else if (!Forward && removed.size() == 1 && added.size() == 4)
    {
        updateContext.template apply<Add, Add, Add, Add, Sub>(added[0], added[1], added[2],
                                                              added[3], removed[0]);
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
    // Weights per vector of the accumulator
    constexpr IndexType Lanes = sizeof(vec_t) / sizeof(BiasType);

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
            IndexType         indexR  = removed[i];
            const IndexType   offsetR = Dimensions * indexR + j * Tiling::TileHeight;
            const WeightType* columnR = &featureTransformer.weights[offsetR];
            IndexType         indexA  = added[i];
            const IndexType   offsetA = Dimensions * indexA + j * Tiling::TileHeight;
            const WeightType* columnA = &featureTransformer.weights[offsetA];

            for (IndexType k = 0; k < Tiling::NumRegs; ++k)
                acc[k] = fused<Vec16Wrapper, Add, Sub>(acc[k], vec_load_i8_16(columnA + k * Lanes),
                                                       vec_load_i8_16(columnR + k * Lanes));
        }
        for (; i < removed.size(); ++i)
        {
            IndexType         index  = removed[i];
            const IndexType   offset = Dimensions * index + j * Tiling::TileHeight;
            const WeightType* column = &featureTransformer.weights[offset];

            for (IndexType k = 0; k < Tiling::NumRegs; ++k)
                acc[k] = vec_sub_16(acc[k], vec_load_i8_16(column + k * Lanes));
        }
        for (; i < added.size(); ++i)
        {
            IndexType         index  = added[i];
            const IndexType   offset = Dimensions * index + j * Tiling::TileHeight;
            const WeightType* column = &featureTransformer.weights[offset];

            for (IndexType k = 0; k < Tiling::NumRegs; ++k)
                acc[k] = vec_add_16(acc[k], vec_load_i8_16(column + k * Lanes));
        }

        for (IndexType k = 0; k < Tiling::NumRegs; k++)
            vec_store(&entryTile[k], acc[k]);
        for (IndexType k = 0; k < Tiling::NumRegs; k++)
            vec_store(&accTile[k], acc[k]);
    }

    for (IndexType j = 0; UsePSQT && j < PSQTBuckets / Tiling::PsqtTileHeight; ++j)
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

        for (std::size_t k = 0; UsePSQT && k < PSQTBuckets; ++k)
            entry.psqtAccumulation[k] -= featureTransformer.psqtWeights[index * PSQTBuckets + k];
    }
    for (const auto index : added)
    {
        const IndexType offset = Dimensions * index;
        for (IndexType j = 0; j < Dimensions; ++j)
            entry.accumulation[j] += featureTransformer.weights[offset + j];

        for (std::size_t k = 0; UsePSQT && k < PSQTBuckets; ++k)
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
