/*
  Maskfish: Jieqi reveal heuristics.

  Ported (subset) from AB-JChess (https://github.com/lxsgx23/AB-JChess),
  GPL-3.0, by Huorongrong and Laoxu (Kouza).

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef REVEAL_H_INCLUDED
#define REVEAL_H_INCLUDED

#include <array>
#include <cstdint>
#include <utility>

#include "types.h"

namespace Maskfish {
class Position;
}

namespace Maskfish::Reveal {

// "Safe reveal" bonus: a score adjustment (in internal units) added to the
// value of a quiet move of a dark piece, depending on whether the square it
// lands on is safe, and on what the mover may turn out to be.
struct QuietParameters {
    int base      = 0;
    int phase     = 0;
    int safety    = 0;
    int highValue = 0;
    int diversity = 0;

    bool enabled() const { return base || phase || safety || highValue || diversity; }
};

enum class QuietSafety : std::int8_t {
    Safe,       // no visible enemy piece attacks the destination
    Contested,  // attacked, but also defended by a visible friendly piece
    Loose       // attacked and not defended
};

struct QuietFeatures {
    int phase;      // -1024 (no dark pieces left) .. 1024 (all 30 dark)
    int safety;     // 1024 safe, 0 contested, -1024 loose
    int highValue;  // share of rooks/cannons/knights in the mover's pool, centered
    int diversity;  // how mixed the mover's pool is
};

std::int64_t  round_divide(std::int64_t numerator, std::int64_t denominator);
QuietSafety   quiet_safety(const Position& pos, Move move);
QuietFeatures make_quiet_features(int globalDark, const std::array<int, 6>& poolCounts,
                                  QuietSafety safety);
QuietFeatures make_quiet_features(const Position& pos, QuietSafety safety);
int           quiet_bonus(const QuietParameters& params, const QuietFeatures& features);

// Bonus for 'move' in 'pos', or 0 when the move is not a quiet dark move.
int move_bonus(const Position& pos, Move move, const QuietParameters& params);

// Searching a child for a parent window [alpha, beta] when the parent value
// is (-child + bonus).
struct Window {
    Value alpha;
    Value beta;
};

Value  apply_bonus(Value rawValue, int bonus);
// Inverse of apply_bonus() (up to the clamping): the raw value of an adjusted one
Value  remove_bonus(Value adjustedValue, int bonus);
Value  first_raw(Value adjustedThreshold, int bonus);
Window child_window(Value alpha, Value beta, int bonus);

template<typename ChildSearch>
Value search_with_bonus(Value alpha, Value beta, int bonus, ChildSearch&& childSearch) {
    const Window w     = child_window(alpha, beta, bonus);
    const Value  child = std::forward<ChildSearch>(childSearch)(w.alpha, w.beta);
    return apply_bonus(Value(-child), bonus);
}

}  // namespace Maskfish::Reveal

#endif  // #ifndef REVEAL_H_INCLUDED
