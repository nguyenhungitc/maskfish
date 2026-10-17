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

#ifndef HISTORY_H_INCLUDED
#define HISTORY_H_INCLUDED

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>  // IWYU pragma: keep

#include "misc.h"
#include "position.h"

namespace Maskfish {

// Size of the pawn history table. Only half of it is used without the UCI
// option BigPawnHistory (see Search::Worker::pawn_history_index()).
constexpr int PAWN_HISTORY_SIZE        = 1024;   // has to be a power of 2
constexpr int CORRECTION_HISTORY_SIZE  = 32768;  // has to be a power of 2
constexpr int CORRECTION_HISTORY_LIMIT = 1024;
constexpr int LOW_PLY_HISTORY_SIZE     = 5;

static_assert((PAWN_HISTORY_SIZE & (PAWN_HISTORY_SIZE - 1)) == 0,
              "PAWN_HISTORY_SIZE has to be a power of 2");

static_assert((CORRECTION_HISTORY_SIZE & (CORRECTION_HISTORY_SIZE - 1)) == 0,
              "CORRECTION_HISTORY_SIZE has to be a power of 2");

enum PawnHistoryType {
    Normal,
    Correction
};

// Index of the pawn history (big: the whole table, otherwise half of it)
// or of the correction histories
template<PawnHistoryType T = Normal>
inline int pawn_structure_index(const Position& pos, bool big = false) {
    return pos.pawn_key()
         & ((T == Normal ? PAWN_HISTORY_SIZE / (big ? 1 : 2) : CORRECTION_HISTORY_SIZE) - 1);
}

inline int minor_piece_index(const Position& pos) {
    return pos.minor_piece_key() & (CORRECTION_HISTORY_SIZE - 1);
}

template<Color c>
inline int non_pawn_index(const Position& pos) {
    return pos.non_pawn_key(c) & (CORRECTION_HISTORY_SIZE - 1);
}

// StatsEntry is the container of various numerical statistics. We use a class
// instead of a naked value to directly call history update operator<<() on
// the entry. The first template parameter T is the base type of the array,
// and the second template parameter D limits the range of updates in [-D, D]
// when we update values with the << operator.
//
// The tables that the threads of a NUMA node may share (UCI option
// SharedHistory, see Search::HistoryTables) are read and written by several
// threads without locks: their entries (Shared = true) are relaxed atomics,
// which compile to plain loads and stores but make these races well defined,
// as in Stockfish and Pikafish. An update reads the entry once, so that a
// value written meanwhile by another thread cannot mix with it (the result
// then stays in [-D, D]; such an update may only be lost).
template<typename T, int D, bool Shared = false>
class StatsEntry {

    static_assert(std::is_arithmetic_v<T>, "Not an arithmetic type");
    static_assert(D <= std::numeric_limits<T>::max(), "D overflows T");

    std::conditional_t<Shared, RelaxedAtomic<T>, T> entry;

   public:
    StatsEntry& operator=(const T& v) {
        entry = v;
        return *this;
    }
    operator T() const { return entry; }

    void operator<<(int bonus) {
        // Make sure that bonus is in range [-D, D]
        const int clampedBonus = std::clamp(bonus, -D, D);
        const T   val          = entry;
        const T   newVal       = T(val + clampedBonus - val * std::abs(clampedBonus) / D);

        assert(std::abs(newVal) <= D);
        entry = newVal;
    }
};

enum StatsType {
    NoCaptures,
    Captures
};

template<typename T, int D, std::size_t... Sizes>
using Stats = MultiArray<StatsEntry<T, D>, Sizes...>;

// Tables that may be shared between threads (see StatsEntry)
template<typename T, int D, std::size_t... Sizes>
using SharedStats = MultiArray<StatsEntry<T, D, true>, Sizes...>;

// ButterflyHistory records how often quiet moves have been successful or unsuccessful
// during the current search, and is used for reduction and move ordering decisions.
// It uses 2 tables (one for each color) indexed by the move's from and to squares,
// see https://www.chessprogramming.org/Butterfly_Boards (~11 elo)
using ButterflyHistory = Stats<std::int16_t, 7183, COLOR_NB, 1 << 14>;

// LowPlyHistory is adressed by play and move's from and to squares, used
// to improve move ordering near the root
using LowPlyHistory = Stats<std::int16_t, 7183, LOW_PLY_HISTORY_SIZE, 1 << 14>;

// CapturePieceToHistory is addressed by a move's [piece][to][captured piece type]
using CapturePieceToHistory = Stats<std::int16_t, 10692, PIECE_NB, SQUARE_NB, PIECE_TYPE_NB>;

// PieceToHistory is like ButterflyHistory but is addressed by a move's [piece][to]
// (shared: it makes up the continuation histories)
using PieceToHistory = SharedStats<std::int16_t, 30000, PIECE_NB, SQUARE_NB>;

// ContinuationHistory is the combined history of a given pair of moves, usually
// the current one given a previous one. The nested history table is based on
// PieceToHistory instead of ButterflyBoards.
// (~63 elo)
using ContinuationHistory = MultiArray<PieceToHistory, PIECE_NB, SQUARE_NB>;

// PawnHistory is addressed by the pawn structure and a move's [piece][to]
using PawnHistory = SharedStats<std::int16_t, 8192, PAWN_HISTORY_SIZE, PIECE_NB, SQUARE_NB>;

// Correction histories record differences between the static evaluation of
// positions and their search score. It is used to improve the static evaluation
// used by some search heuristics.
// see https://www.chessprogramming.org/Static_Evaluation_Correction_History
enum CorrHistType {
    Pawn,          // By color and pawn structure
    Minor,         // By color and positions of minor pieces (Knight, Cannon)
    NonPawn,       // By non-pawn material positions and color
    PieceTo,       // By [piece][to] move
    Continuation,  // Combined history of move pairs
};

namespace Detail {

template<CorrHistType>
struct CorrHistTypedef {
    using type =
      SharedStats<std::int16_t, CORRECTION_HISTORY_LIMIT, CORRECTION_HISTORY_SIZE, COLOR_NB>;
};

template<>
struct CorrHistTypedef<PieceTo> {
    using type = SharedStats<std::int16_t, CORRECTION_HISTORY_LIMIT, PIECE_NB, SQUARE_NB>;
};

template<>
struct CorrHistTypedef<Continuation> {
    using type = MultiArray<CorrHistTypedef<PieceTo>::type, PIECE_NB, SQUARE_NB>;
};

template<>
struct CorrHistTypedef<NonPawn> {
    using type = SharedStats<std::int16_t,
                             CORRECTION_HISTORY_LIMIT,
                             CORRECTION_HISTORY_SIZE,
                             COLOR_NB,
                             COLOR_NB>;
};

}

template<CorrHistType T>
using CorrectionHistory = typename Detail::CorrHistTypedef<T>::type;

using TTMoveHistory = StatsEntry<std::int16_t, 8192>;

}  // namespace Maskfish

#endif  // #ifndef HISTORY_H_INCLUDED
