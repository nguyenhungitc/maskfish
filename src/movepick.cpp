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

#include "movepick.h"

#include <cassert>
#include <limits>

#include "attacks.h"
#include "bitboard.h"
#include "misc.h"
#include "position.h"

namespace Maskfish {

using Attacks::line_bb;

// Stages of the move picker. The kind of picker (main search, quiescence
// search, evasions, ProbCut) is the stage it starts with after the TT move,
// firstStage. The first stage generates and scores its moves in
// INIT_FIRST_STAGE: captures, or evasions when in check.
enum Stages : i8 {
    TT,                // the TT move, if any
    INIT_FIRST_STAGE,  // generate and score the moves of firstStage

    // main search
    GOOD_CAPTURE,
    GOOD_QUIET,
    BAD_MOVES,  // losing captures, then quiet moves with a low score

    // quiescence search: all captures
    QSEARCH,

    // in check: all evasions
    EVASION,

    // ProbCut: captures with SEE >= threshold
    PROBCUT,
};

namespace {

#ifdef USE_AVX512
// Load the Move, and the ExtMove value, into all lanes of 512-bit registers
static void splat_extmove(const ExtMove& m, __m512i& move, __m512i& value) {
    move  = _mm512_set1_epi32(m.raw());
    value = _mm512_set1_epi32(m.value);
}

// Sorts up to 16 moves.
struct MoveSorter {
    static constexpr int MAX_ELEMENTS = 16;
    __m512i              sortedValues, sortedMoves;

    explicit MoveSorter(const ExtMove& first) {
        splat_extmove(first, sortedMoves, sortedValues);

        // Set the uninitialized move values to INT_MIN, so that they sort less than any other move
        sortedValues = _mm512_mask_set1_epi32(sortedValues, ~1, std::numeric_limits<int>::min());
    }

    void insert(const ExtMove& m) {
        __m512i move, value;
        splat_extmove(m, move, value);

        // Mask of all elements except the insertion point
        assert(m.value != std::numeric_limits<int>::min());
        // (minus one: plain arithmetic rather than _kadd_mask16, which needs
        // AVX512DQ, not enabled by the x86-64-avx512 / -avx512bw targets)
        const uint16_t expand =
          uint16_t(_mm512_cmplt_epi32_mask(sortedValues, value) - 1);

        sortedValues = _mm512_mask_expand_epi32(value, expand, sortedValues);
        sortedMoves  = _mm512_mask_expand_epi32(move, expand, sortedMoves);
    }

    void write_sorted(ExtMove* moves, std::ptrdiff_t count) const {
        static_assert(sizeof(ExtMove) == 8);
        assert(count <= MAX_ELEMENTS);

        // Because values and moves are stored separately, we need to reassemble the ExtMoves
        auto write = [&](int offset, const __m512i indices) {
            const __m512i extMoves = _mm512_permutex2var_epi32(sortedMoves, indices, sortedValues);
            const std::ptrdiff_t storeCount = count - offset;

            if (storeCount > 0)
                _mm512_mask_storeu_epi64(moves + offset, (1 << storeCount) - 1, extMoves);
        };

        write(0, _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23));
        write(8, _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31));
    }
};
#endif

// Sort moves in descending order up to and including a given limit.
// The order of moves smaller than the limit is left unspecified.
void partial_insertion_sort(ExtMove* begin, ExtMove* end, int limit) {
    ExtMove *sortedEnd = begin, *p = begin + 1;

#ifdef USE_AVX512
    if (begin == end)
        return;

    MoveSorter sorter(*begin);
    for (; p < end; ++p)
    {
        if (p->value >= limit)
        {
            if (sortedEnd - begin + 1 >= MoveSorter::MAX_ELEMENTS)  // sorter full
                break;

            sorter.insert(*p);
            *p = *++sortedEnd;
        }
    }
    sorter.write_sorted(begin, sortedEnd - begin + 1);
    // Use scalar implementation for any remaining elements
#endif

    for (; p < end; ++p)
        if (p->value >= limit)
        {
            ExtMove tmp = *p, *q;
            *p          = *++sortedEnd;
            for (q = sortedEnd; q != begin && *(q - 1) < tmp; --q)
                *q = *(q - 1);
            *q = tmp;
        }
}

}  // namespace


// Constructors of the MovePicker class. As arguments, we pass information
// to decide which class of moves to emit, to help sorting the (presumably)
// good moves first, and how important move ordering is at the current node.

// MovePicker constructor for the main search and for the quiescence search
MovePicker::MovePicker(const Position&              p,
                       Move                         ttm,
                       Depth                        d,
                       const ButterflyHistory*      mh,
                       const LowPlyHistory*         lph,
                       const CapturePieceToHistory* cph,
                       const PieceToHistory**       ch,
                       const PawnHistoryEntry*      ph,
                       int                          pl) :
    pos(p),
    mainHistory(mh),
    lowPlyHistory(lph),
    captureHistory(cph),
    continuationHistory(ch),
    pawnHistory(ph),
    ttMove(ttm),
    depth(d),
    ply(pl) {

    if (pos.checkers())
        firstStage = EVASION;
    else
        firstStage = depth > 0 ? GOOD_CAPTURE : QSEARCH;

    stage = ttm && pos.legal(ttm) ? TT : INIT_FIRST_STAGE;
}

// MovePicker constructor for ProbCut: we generate captures with Static Exchange
// Evaluation (SEE) greater than or equal to the given threshold.
MovePicker::MovePicker(const Position& p, Move ttm, int th, const CapturePieceToHistory* cph) :
    pos(p),
    captureHistory(cph),
    ttMove(ttm),
    threshold(th) {
    assert(!pos.checkers());

    firstStage = PROBCUT;
    stage      = ttm && pos.legal(ttm) && pos.capture(ttm) && pos.see_ge(ttm, threshold)
                 ? TT
                 : INIT_FIRST_STAGE;
}

// Assigns a numerical value to each move in a list, used for sorting.
// Captures are ordered by Most Valuable Victim (MVV), preferring captures
// with a good history. Quiets moves are ordered using the history tables.
template<GenType Type>
ExtMove* MovePicker::generate_and_score() {

    static_assert(Type == CAPTURES || Type == QUIETS || Type == EVASIONS, "Wrong type");

    MoveList<Type> ml(pos);  // move generation

    Color us = pos.side_to_move();

    [[maybe_unused]] Bitboard threatByLesser[BISHOP + 1];
    if constexpr (Type == QUIETS)
    {
        if (Position::useThreatCacheMovePick)
        {
            // UCI option ThreatCacheMovePick: threats by lesser
            // pieces, from the threats cached in the StateInfo (x-raying our
            // king, except the cannons)
            threatByLesser[ADVISOR] = threatByLesser[BISHOP] = pos.threats_by(PAWN);
            threatByLesser[KNIGHT]                           = threatByLesser[CANNON] =
              pos.threats_by(ADVISOR) | pos.threats_by(BISHOP) | threatByLesser[ADVISOR];
            threatByLesser[ROOK] =
              pos.threats_by(KNIGHT) | pos.attacks_by<CANNON>(~us) | threatByLesser[KNIGHT];
        }
        else
        {
            threatByLesser[ADVISOR] = threatByLesser[BISHOP] = pos.attacks_by<PAWN>(~us);
            threatByLesser[KNIGHT]                           = threatByLesser[CANNON] =
              pos.attacks_by<ADVISOR>(~us) | pos.attacks_by<BISHOP>(~us) | threatByLesser[ADVISOR];
            threatByLesser[ROOK] =
              pos.attacks_by<KNIGHT>(~us) | pos.attacks_by<CANNON>(~us) | threatByLesser[KNIGHT];
        }
    }

    ExtMove* it = cur;
    for (auto move : ml)
    {
        ExtMove& m = *it++;
        m          = move;

        const Square    from          = m.from_sq();
        const Square    to            = m.to_sq();
        const Piece     pc            = pos.moved_piece(m);
        const PieceType pt            = type_of(pc);

        if constexpr (Type == CAPTURES)
            m.value = (*captureHistory)[pc][to][pos.capture_type(to)]
                    + 7 * int(pos.piece_value(to))
                    + (noCaptureCheckBonus ? 0 : 1024)
                        * bool((pt == CANNON
                                  ? pos.check_squares(pt) & ~line_bb(from, pos.king_square(~us))
                                  : pos.check_squares(pt))
                               & to);

        else if constexpr (Type == QUIETS)
        {
            // Use a local accumulator to eliminate redundant stores to m.value

            // histories
            int value = 2 * (*mainHistory)[us][m.from_to()];
            value += 2 * (*pawnHistory)[pc][to];
            value += (*continuationHistory[0])[pc][to];
            value += (*continuationHistory[1])[pc][to];
            value += (*continuationHistory[2])[pc][to];
            value += (*continuationHistory[3])[pc][to];
            value += (*continuationHistory[5])[pc][to];

            // bonus for checks
            value +=
              (bool((pt == CANNON ? pos.check_squares(pt) & ~line_bb(from, pos.king_square(~us))
                                  : pos.check_squares(pt))
                    & to)
               && pos.see_ge(m, -75))
              * 16384;

            // penalty for moving to a square threatened by a lesser piece
            // or bonus for escaping an attack by a lesser piece.
            // With the UCI option ThreatByLesserValue, in proportion to the
            // piece value, and no penalty when the piece is already attacked.
            if (pt != PAWN && pt <= BISHOP)
            {
                if (threatByLesserValue)
                    value += int(PieceValue[pt]) * 20
                           * (bool(threatByLesser[pt] & from) - bool(threatByLesser[pt] & to));
                else
                {
                    static constexpr int bonus[BISHOP + 1] = {0, 517, 144, 256, 0, 256, 144};
                    int v = threatByLesser[pt] & to ? -95 : 100 * bool(threatByLesser[pt] & from);
                    value += bonus[pt] * v;
                }
            }

            if (ply < LOW_PLY_HISTORY_SIZE)
                value += 8 * (*lowPlyHistory)[ply][m.from_to()] / (1 + ply);

            m.value = value;
        }

        else  // Type == EVASIONS
        {
            if (pos.capture(m))
                m.value = pos.piece_value(to) + (1 << 28);
            else
            {
                m.value = (*mainHistory)[us][m.from_to()] + (*continuationHistory[0])[pc][to];
                if (ply < LOW_PLY_HISTORY_SIZE)
                    m.value += 2 * (*lowPlyHistory)[ply][m.from_to()] / (1 + ply);
            }
        }
    }
    return it;
}

// Returns the next move satisfying a predicate function.
// This never returns the TT move, as it was emitted before.
template<typename Pred>
Move MovePicker::select(Pred filter) {

    for (; cur < endCur; ++cur)
        if (*cur != ttMove && filter())
            return *cur++;

    return Move::none();
}

// This is the most important method of the MovePicker class. We emit one
// new legal move on every call until there are no more moves left,
// picking the move with the highest score from a list of generated moves.
Move MovePicker::next_move() {

    constexpr int goodQuietThreshold = -14000;

dispatch:
    switch (stage)
    {

    case TT :
        stage = INIT_FIRST_STAGE;
        return ttMove;

    case INIT_FIRST_STAGE :
        cur = endBadMoves = moves;

        if (firstStage == EVASION)
            endCur = generate_and_score<EVASIONS>();
        else
            endCur = generate_and_score<CAPTURES>();

        partial_insertion_sort(cur, endCur, std::numeric_limits<int>::min());

        stage = firstStage;
        goto dispatch;

    case GOOD_CAPTURE :
        if (select([&]() {
                // Move losing capture to endBadMoves to be tried later
                if (pos.see_ge(*cur, -cur->value / 18))
                    return true;
                *endBadMoves++ = *cur;
                return false;
            }))
            return *(cur - 1);

        endBadCaptures = endBadMoves;

        if (!skipQuiets)
        {
            // Generate the quiet moves after the bad captures
            cur    = endBadMoves;
            endCur = generate_and_score<QUIETS>();

            partial_insertion_sort(cur, endCur, -3330 * depth);

            stage = GOOD_QUIET;
            [[fallthrough]];

        case GOOD_QUIET :
            if (select([&]() {
                    // Move quiet moves with a low score next to the bad
                    // captures, to be tried after them
                    if (cur->value > goodQuietThreshold)
                        return true;
                    *endBadMoves++ = *cur;
                    return false;
                }))
                return *(cur - 1);
        }

        // Prepare the pointers to loop over the bad captures and bad quiets
        cur    = moves;
        endCur = endBadMoves;

        stage = BAD_MOVES;
        [[fallthrough]];

    case BAD_MOVES :
    case QSEARCH :
    case EVASION :
        return select([]() { return true; });

    case PROBCUT :
        return select([&]() { return pos.see_ge(*cur, threshold); });

    default :
        assert(false);
        return Move::none();
    }
}

void MovePicker::skip_quiet_moves() {
    skipQuiets = true;

    // Drop the quiet moves not emitted yet: the good quiet moves left in
    // GOOD_QUIET and the bad quiet moves, which follow the bad captures
    if (stage == GOOD_QUIET || stage == BAD_MOVES)
    {
        endCur      = endBadCaptures;
        endBadMoves = endBadCaptures;
    }
}

}  // namespace Maskfish
