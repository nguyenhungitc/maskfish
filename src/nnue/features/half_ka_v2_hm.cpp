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

// Definition of input features HalfKAv2_hm of NNUE evaluation function

#include "half_ka_v2_hm.h"

#include <algorithm>
#include <cstdint>
#include "../../attacks.h"

#include "../../position.h"
#include "../../types.h"
#include "../nnue_common.h"

namespace Maskfish::Eval::NNUE::Features {

using namespace Maskfish::Attacks;

// Index of a feature for a given king position and another piece on some square
template<Color Perspective>
inline IndexType HalfKAv2_hm::make_index(Square s, Piece pc, int bucket, bool mirror) {
    return IndexType(IndexMap[mirror][Perspective == BLACK][s] + PieceSquareIndex[Perspective][pc]
                     + PS_NB * bucket);
}

// Explicit template instantiations
template IndexType HalfKAv2_hm::make_index<WHITE>(Square s, Piece pc, int bucket, bool mirror);
template IndexType HalfKAv2_hm::make_index<BLACK>(Square s, Piece pc, int bucket, bool mirror);

template<Color Perspective>
inline IndexType HalfKAv2_hm::make_dark_rest_index(Piece pc, int bucket) {
    return IndexType(PS_NB * bucket + DarkRestIndex[Perspective][pc]);
}

// Explicit template instantiations
template IndexType HalfKAv2_hm::make_dark_rest_index<WHITE>(Piece pc, int bucket);
template IndexType HalfKAv2_hm::make_dark_rest_index<BLACK>(Piece pc, int bucket);

#ifdef JIEQI_NNUE_ATT
bool HalfKAv2_hm::requires_mid_mirror(const Position& pos, Color c) {
    return ((1ULL << 63) & pos.mid_encoding(c) & pos.mid_encoding(~c))
        && (pos.mid_encoding(c) < BalanceEncoding
            || (pos.mid_encoding(c) == BalanceEncoding && pos.mid_encoding(~c) < BalanceEncoding));
}

// Visible (revealed) attacking material of one side
int HalfKAv2_hm::make_attack_bucket(const Position& pos, Color c) {
    const Bitboard visible = pos.pieces(c) & ~pos.pieces(DARK);
    return 2 * bool(visible & pos.pieces(ROOK)) + bool(visible & pos.pieces(KNIGHT, CANNON));
}
#endif

std::pair<int, bool> HalfKAv2_hm::make_feature_bucket(const Position& pos, Color perspective) {
    const Square ksq  = pos.king_square(perspective);
    const Square oksq = pos.king_square(~perspective);
#ifdef JIEQI_NNUE_ATT
    const bool mirror = king_mirror(ksq, oksq, requires_mid_mirror(pos, perspective));
    return {KingBuckets[ksq][oksq].first * int(AttackBucketNB) + make_attack_bucket(pos, perspective),
            mirror};
#else
    return KingBuckets[ksq][oksq];
#endif
}

int HalfKAv2_hm::make_cache_index(const Position& pos, Color perspective) {
    // clang-format off
    static constexpr uint8_t KingCacheMaps[SQUARE_NB] = {
      0,  0,  0,  6,  0,  3,  0,  0,  0,
      0,  0,  0,  7,  1,  4,  0,  0,  0,
      0,  0,  0,  8,  2,  5,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  8,  2,  5,  0,  0,  0,
      0,  0,  0,  7,  1,  4,  0,  0,  0,
      0,  0,  0,  6,  0,  3,  0,  0,  0,
    };
    // clang-format on
    const auto [bucket, mirror] = make_feature_bucket(pos, perspective);
    int index = KingCacheMaps[pos.king_square(perspective)];
    if (index < 3 && mirror)
        index += 9;  // king on the middle file, mirrored
    return index * int(AttackBucketNB) + bucket % int(AttackBucketNB);
}

// Get layer stack bucket
IndexType HalfKAv2_hm::make_layer_stack_bucket(const Position& pos) {
#ifdef JIEQI_NNUE_BLEND
    return IndexType(make_layer_stack_selection(pos).floor);
#else
    return LayerStackBuckets[pos.count<ALL_PIECES>()];
#endif
}

HalfKAv2_hm::LayerStackSelection HalfKAv2_hm::make_layer_stack_selection(const Position& pos) {
    // Position of v in [lower, upper) as the fraction num / den (at most
    // (upper - lower - 1) / (upper - lower)), 0 beyond the last interval
    struct Fraction {
        int num, den;
    };
    auto residual = [](int v, const int* thresholds, int n) -> Fraction {
        int k = 0;
        while (k + 1 < n && thresholds[k + 1] <= v)
            ++k;
        if (k >= n - 1)
            return {0, 1};
        const int lower = thresholds[k], upper = thresholds[k + 1];
        return {std::min(upper - lower - 1, std::max(0, v - lower)), upper - lower};
    };

    // Piece count intervals [lower, upper) of the 16 buckets of LayerStackBuckets
    static constexpr int PieceIntervals[16][2] = {
      {2, 7},   {7, 9},   {9, 11},  {11, 13}, {13, 15}, {15, 17}, {17, 19}, {19, 21},
      {21, 23}, {23, 25}, {25, 27}, {27, 29}, {29, 31}, {31, 32}, {32, 33}, {33, 34}};
    static constexpr int DarkThresholds[]   = {0, 1, 5, 9, 17, 33};
    static constexpr int StrongThresholds[] = {0, 2, 5, 9, 19};

    const int pieces = std::clamp(pos.count<ALL_PIECES>(), 2, 32);
    const int dark   = popcount(pos.pieces(DARK));
    int       strong = 0;
    for (Color c : {WHITE, BLACK})
        strong += pos.rest_piece(make_piece(c, ROOK)) + pos.rest_piece(make_piece(c, CANNON))
                + pos.rest_piece(make_piece(c, KNIGHT));

    const int darkPhase   = dark >= 17 ? 4 : dark >= 9 ? 3 : dark >= 5 ? 2 : dark >= 1 ? 1 : 0;
    const int strongPhase = strong >= 9 ? 3 : strong >= 5 ? 2 : strong >= 2 ? 1 : 0;
    const int base        = LayerStackBuckets[pieces];
    const int floor       = std::clamp(base + darkPhase + strongPhase, 0, 15);
    if (floor == 15)
        return {15, 0};

    const int lower = PieceIntervals[base][0], upper = PieceIntervals[base][1];
    const Fraction r1 = {std::min(upper - lower - 1, std::max(0, pieces - lower)), upper - lower};
    const Fraction r2 = residual(dark, DarkThresholds, 6);
    const Fraction r3 = residual(strong, StrongThresholds, 5);

    // blend = round(255 * (r1 + r2 + r3) / 3), half up, exactly
    const std::int64_t den = std::int64_t(r1.den) * r2.den * r3.den;
    const std::int64_t num =
      85 * (std::int64_t(r1.num) * r2.den * r3.den + std::int64_t(r2.num) * r1.den * r3.den
            + std::int64_t(r3.num) * r1.den * r2.den);
    const int blend = int((2 * num + den) / (2 * den));
    if (blend >= 255)  // the next layer stack
        return {std::min(15, floor + 1), 0};
    return {floor, std::clamp(blend, 0, 254)};
}

// Get a list of indices for recently changed features
template<Color Perspective>
void HalfKAv2_hm::append_changed_indices(
  int bucket, bool mirror, const DirtyPiece& dp, IndexList& removed, IndexList& added) {
    removed.push_back(make_index<Perspective>(dp.from, dp.pc, bucket, mirror));

    if (dp.to != SQ_NONE)
        added.push_back(make_index<Perspective>(dp.to, dp.pc, bucket, mirror));
    else if (dp.pc == DARK_PIECE)
        removed.push_back(make_dark_rest_index<Perspective>(dp.add_pc, bucket));

    if (dp.add_sq != SQ_NONE)
        added.push_back(make_index<Perspective>(dp.add_sq, dp.add_pc, bucket, mirror));

    if (dp.remove_sq != SQ_NONE)
        removed.push_back(make_index<Perspective>(dp.remove_sq, dp.remove_pc, bucket, mirror));

    if (dp.rest_remove_pc != NO_PIECE)
        removed.push_back(make_dark_rest_index<Perspective>(dp.rest_remove_pc, bucket));
}

// Explicit template instantiations
template void HalfKAv2_hm::append_changed_indices<WHITE>(
  int bucket, bool mirror, const DirtyPiece& dp, IndexList& removed, IndexList& added);
template void HalfKAv2_hm::append_changed_indices<BLACK>(
  int bucket, bool mirror, const DirtyPiece& dp, IndexList& removed, IndexList& added);

bool HalfKAv2_hm::requires_refresh(const DirtyPiece& dirtyPiece, Color perspective) {
    return dirtyPiece.requires_refresh[perspective];
}

}  // namespace Maskfish::Eval::NNUE::Features
