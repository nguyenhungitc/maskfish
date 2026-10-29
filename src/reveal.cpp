/*
  Maskfish: Jieqi reveal heuristics.

  Copyright (C) 2026 The Maskfish developers (see AUTHORS file)

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "reveal.h"
#include "attacks.h"

#include <algorithm>
#include <cassert>
#include <numeric>

#include "bitboard.h"
#include "position.h"

namespace Maskfish::Reveal {

using namespace Maskfish::Attacks;

std::int64_t round_divide(std::int64_t numerator, std::int64_t denominator) {
    assert(denominator > 0);
    const std::int64_t quotient  = numerator / denominator;
    const std::int64_t remainder = numerator % denominator;
    const std::int64_t magnitude = remainder < 0 ? -remainder : remainder;
    return quotient + (magnitude * 2 >= denominator ? (numerator < 0 ? -1 : 1) : 0);
}

// Safety of the destination square of a quiet move of a dark piece, looking
// only at visible (revealed) pieces: dark attackers are unknown.
QuietSafety quiet_safety(const Position& pos, Move move) {
    assert(pos.move_dark(move));
    assert(!pos.capture(move));

    const Bitboard fromBB        = square_bb(move.from_sq());
    const Bitboard toBB          = square_bb(move.to_sq());
    const Bitboard occupiedAfter = (pos.pieces() ^ fromBB) | toBB;
    const Bitboard attackers = pos.attackers_to(move.to_sq(), occupiedAfter) & ~pos.pieces(DARK);
    const Bitboard enemy     = attackers & pos.pieces(~pos.side_to_move());
    const Bitboard friendly  = (attackers & pos.pieces(pos.side_to_move())) & ~fromBB;

    return !enemy ? QuietSafety::Safe : friendly ? QuietSafety::Contested : QuietSafety::Loose;
}

QuietFeatures
make_quiet_features(int globalDark, const std::array<int, 6>& poolCounts, QuietSafety safety) {
    const int phase = int(round_divide(std::int64_t(2 * globalDark - 30) * 1024, 30));

    const std::int64_t poolTotal = std::accumulate(poolCounts.begin(), poolCounts.end(), 0LL);
    int                highValue = 0;
    int                diversity = 0;

    if (poolTotal > 0)
    {
        // poolCounts order: rook, advisor, cannon, pawn, knight, bishop
        const std::int64_t highCount         = std::int64_t(poolCounts[0]) + poolCounts[2] + poolCounts[4];
        const std::int64_t centeredHighValue = 5 * highCount - 2 * poolTotal;
        highValue = int(round_divide(centeredHighValue * 1024,
                                     (centeredHighValue >= 0 ? 3 : 2) * poolTotal));

        std::int64_t squaredCounts = 0;
        for (int count : poolCounts)
            squaredCounts += std::int64_t(count) * count;
        diversity = std::clamp(
          int(round_divide((poolTotal * poolTotal - 5 * squaredCounts) * 256, poolTotal * poolTotal)),
          -1024, 1024);
    }

    const int s = safety == QuietSafety::Safe ? 1024 : safety == QuietSafety::Loose ? -1024 : 0;

    return {phase, s, highValue, diversity};
}

QuietFeatures make_quiet_features(const Position& pos, QuietSafety safety) {
    Position::RestPieceList restPieces;
    const int               n = pos.rest_pieces(pos.side_to_move(), restPieces);

    std::array<int, 6> poolCounts{};
    for (int i = 0; i < n; ++i)
    {
        const auto [piece, count] = restPieces[i];
        switch (type_of(piece))
        {
        case ROOK :
            poolCounts[0] += count;
            break;
        case ADVISOR :
            poolCounts[1] += count;
            break;
        case CANNON :
            poolCounts[2] += count;
            break;
        case PAWN :
            poolCounts[3] += count;
            break;
        case KNIGHT :
            poolCounts[4] += count;
            break;
        case BISHOP :
            poolCounts[5] += count;
            break;
        default :
            assert(false);
            break;
        }
    }

    return make_quiet_features(popcount(pos.pieces(DARK)), poolCounts, safety);
}

int quiet_bonus(const QuietParameters& p, const QuietFeatures& f) {
    if (!p.enabled())
        return 0;

    const std::int64_t weighted = std::int64_t(p.phase) * f.phase + std::int64_t(p.safety) * f.safety
                                + std::int64_t(p.highValue) * f.highValue
                                + std::int64_t(p.diversity) * f.diversity;
    const std::int64_t result = std::int64_t(p.base) + round_divide(weighted, 1024);
    return int(std::clamp<std::int64_t>(result, -192, 192));
}

int move_bonus(const Position& pos, Move move, const QuietParameters& params) {
    if (!params.enabled() || !pos.move_dark(move) || pos.capture(move))
        return 0;

    return quiet_bonus(params, make_quiet_features(pos, quiet_safety(pos, move)));
}

Value apply_bonus(Value rawValue, int bonus) {
    if (is_decisive(rawValue))
        return rawValue;

    return Value(std::clamp(int(rawValue) + bonus, int(VALUE_MATED_IN_MAX_PLY) + 1,
                            int(VALUE_MATE_IN_MAX_PLY) - 1));
}

Value remove_bonus(Value adjustedValue, int bonus) {
    if (is_decisive(adjustedValue))
        return adjustedValue;

    return Value(std::clamp(int(adjustedValue) - bonus, int(VALUE_MATED_IN_MAX_PLY) + 1,
                            int(VALUE_MATE_IN_MAX_PLY) - 1));
}

// Smallest raw (unadjusted) value whose adjusted value reaches the threshold.
Value first_raw(Value adjustedThreshold, int bonus) {
    constexpr int Lower  = VALUE_MATED_IN_MAX_PLY;
    constexpr int Upper  = VALUE_MATE_IN_MAX_PLY;
    const int     target = adjustedThreshold;

    if (target <= Lower || target >= Upper)
        return adjustedThreshold;

    if (target == Lower + 1)
        return Value(Lower + 1);

    const int candidate = std::max(Lower + 1, target - bonus);
    return Value(candidate <= Upper - 1 ? candidate : Upper);
}

Window child_window(Value alpha, Value beta, int bonus) {
    const Value rawAlpha = Value(first_raw(Value(alpha + 1), bonus) - 1);
    const Value rawBeta  = first_raw(beta, bonus);
    return {Value(-rawBeta), Value(-rawAlpha)};
}

}  // namespace Maskfish::Reveal
