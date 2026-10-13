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

#ifndef ATTACKS_H_INCLUDED
#define ATTACKS_H_INCLUDED

#include <array>
#include <cassert>
#include <initializer_list>
#include <utility>

#include "bitboard.h"
#include "misc.h"
#include "types.h"

namespace Maskfish::Attacks {

#ifdef USE_PEXT
    #define IF_NOT_PEXT(...)
using MagicMask = u32;
#else
    #define IF_NOT_PEXT(...) __VA_ARGS__
using MagicMask = Bitboard;
#endif

// Magic holds all magic bitboards relevant data for a single square
struct Magic {
    Bitboard   mask;
    MagicMask* attacks;
    unsigned   shift;
#ifdef USE_PEXT
    Bitboard pseudoAttacks;
#else
    Bitboard magic;
#endif

    // Compute the attack's index using the 'magic bitboards' approach
    unsigned index(Bitboard occupied) const {

#ifdef USE_PEXT
        return unsigned(pext(occupied, mask, shift));
#else
        return unsigned(((occupied & mask) * magic) >> shift);
#endif
    }

    Bitboard attack_bb(Bitboard occupied) const {
#ifdef USE_PEXT
        return pdep(attacks[index(occupied)], pseudoAttacks);
#else
        return attacks[index(occupied)];
#endif
    }
};

void         init();
const Magic& magic(Square s, PieceType pt);

extern Bitboard LineBB[SQUARE_NB][SQUARE_NB];
extern Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];
extern Bitboard RayPassBB[SQUARE_NB][SQUARE_NB];
extern Bitboard LeaperPassBB[SQUARE_NB][SQUARE_NB];

inline Bitboard line_bb(Square s1, Square s2) {

    assert(is_ok(s1) && is_ok(s2));
    return LineBB[s1][s2];
}

inline Bitboard between_bb(Square s1, Square s2) {

    assert(is_ok(s1) && is_ok(s2));
    return BetweenBB[s1][s2];
}

inline Bitboard ray_pass_bb(Square s1, Square s2) {

    assert(is_ok(s1) && is_ok(s2));
    return RayPassBB[s1][s2];
}

inline Bitboard leaper_pass_bb(Square s1, Square s2) {

    assert(is_ok(s1) && is_ok(s2));
    return LeaperPassBB[s1][s2];
}

// Returns true if the squares s1, s2 and s3 are aligned either on a
// straight or on a diagonal line.
inline bool aligned(Square s1, Square s2, Square s3) { return bool(line_bb(s1, s2) & s3); }

// Returns the squares attacked by pawns of the given color
// from the given square.
template<Color C>
constexpr Bitboard pawn_attacks_bb(Square s) {
    Bitboard b      = square_bb(s);
    Bitboard attack = shift(b, C == WHITE ? NORTH : SOUTH);
    if ((C == WHITE && rank_of(s) > RANK_4) || (C == BLACK && rank_of(s) < RANK_5))
        attack |= shift(b, WEST) | shift(b, EAST);
    return attack;
}

// Returns the squares that if there is a pawn
// of the given color in there, it can attack the square s.
template<Color C>
constexpr Bitboard pawn_attacks_to_bb(Square s) {
    Bitboard b      = square_bb(s);
    Bitboard attack = shift(b, C == WHITE ? SOUTH : NORTH);
    if ((C == WHITE && rank_of(s) > RANK_4) || (C == BLACK && rank_of(s) < RANK_5))
        attack |= shift(b, WEST) | shift(b, EAST);
    return attack;
}

// Returns the bitboard of target square for the given step
// from the given square. If the step is off the board, returns empty bitboard.
constexpr Bitboard safe_destination(Square s, int step) {
    Square to = Square(s + step);
    return is_ok(to) && distance(s, to) <= 2 ? square_bb(to) : Bitboard(0);
}

template<PieceType pt>
constexpr Bitboard sliding_attack(Square sq, Bitboard occupied) {
    assert(pt == ROOK || pt == CANNON);
    Bitboard attack = 0;

    for (auto const& d : {NORTH, SOUTH, EAST, WEST})
    {
        bool hurdle = false;
        for (Square s = sq + d; is_ok(s) && distance(s - d, s) == 1; s += d)
        {
            if (pt == ROOK || hurdle)
                attack |= s;

            if (occupied & s)
            {
                if (pt == CANNON && !hurdle)
                    hurdle = true;
                else
                    break;
            }
        }
    }

    return attack;
}

template<PieceType pt>
constexpr Bitboard lame_leaper_path(Direction d, Square s) {
    Bitboard b  = 0;
    Square   to = s + d;
    if (!is_ok(to) || distance(s, to) > 3)
        return b;

    // If piece type is by knight attacks, swap the source and destination square
    if (pt == KNIGHT_TO)
    {
        std::swap(s, to);
        d = -d;
    }

    Direction dr = d > 0 ? NORTH : SOUTH;
    Direction df = (abs(d % NORTH) < NORTH / 2 ? d % NORTH : -(d % NORTH)) < 0 ? WEST : EAST;

    int diff = abs(file_of(to) - file_of(s)) - abs(rank_of(to) - rank_of(s));
    if (diff > 0)
        s += df;
    else if (diff < 0)
        s += dr;
    else
        s += df + dr;

    b |= s;
    return b;
}

template<PieceType pt>
constexpr auto get_directions() {
    std::array<Direction, 4> BishopDirections{2 * NORTH_EAST, 2 * SOUTH_EAST, 2 * SOUTH_WEST,
                                              2 * NORTH_WEST};
    std::array<Direction, 8> KnightDirections{2 * SOUTH + WEST, 2 * SOUTH + EAST, SOUTH + 2 * WEST,
                                              SOUTH + 2 * EAST, NORTH + 2 * WEST, NORTH + 2 * EAST,
                                              2 * NORTH + WEST, 2 * NORTH + EAST};
    if constexpr (pt == BISHOP)
        return BishopDirections;
    else
        return KnightDirections;
}

template<PieceType pt>
constexpr Bitboard lame_leaper_path(Square s) {
    Bitboard b = 0;
    for (const auto& d : get_directions<pt>())
        b |= lame_leaper_path<pt>(d, s);

    return b;
}

// Attacks of a knight or a bishop, blocked on the leg or the eye. Unlike
// xiangqi, bishops are not masked by the river: in Jieqi revealed bishops
// may cross it (and a dark bishop never leaves its starting square).
template<PieceType pt>
constexpr Bitboard lame_leaper_attack(Square s, Bitboard occupied) {
    Bitboard b = 0;
    for (const auto& d : get_directions<pt>())
    {
        Square to = s + d;
        if (is_ok(to) && distance(s, to) < 3 && !(lame_leaper_path<pt>(d, s) & occupied))
            b |= to;
    }
    return b;
}

// Attacks of the pieces on an empty board, by kind of table (instead of rows
// of one table indexed by piece types and offsets from them).
struct PseudoAttackTables {
    using Table = std::array<Bitboard, SQUARE_NB>;

    // Indexed by piece type: ROOK, ADVISOR, KNIGHT, BISHOP and KING (in the
    // palace only). The rows of the other piece types stay empty: cannon
    // attacks depend on the occupancy, pawn attacks on the color.
    std::array<Table, PIECE_TYPE_NB> byType{};

    std::array<Table, COLOR_NB> pawn{};               // Squares a pawn of the color attacks
    std::array<Table, COLOR_NB> pawnTo{};             // Squares from which such a pawn attacks
    Table                       unconstrainedKing{};  // King steps, ignoring the palace
};

inline constexpr PseudoAttackTables PseudoAttacks = [] {
    PseudoAttackTables t{};

    for (Square s1 = SQ_A0; s1 <= SQ_I9; ++s1)
    {
        t.pawn[WHITE][s1]   = pawn_attacks_bb<WHITE>(s1);
        t.pawn[BLACK][s1]   = pawn_attacks_bb<BLACK>(s1);
        t.pawnTo[WHITE][s1] = pawn_attacks_to_bb<WHITE>(s1);
        t.pawnTo[BLACK][s1] = pawn_attacks_to_bb<BLACK>(s1);

        t.byType[ROOK][s1]   = sliding_attack<ROOK>(s1, 0);
        t.byType[BISHOP][s1] = lame_leaper_attack<BISHOP>(s1, 0);
        t.byType[KNIGHT][s1] = lame_leaper_attack<KNIGHT>(s1, 0);

        // The king stays in the palace (its table is empty outside it).
        for (int step : {NORTH, SOUTH, WEST, EAST})
        {
            if (Palace & s1)
                t.byType[KING][s1] |= safe_destination(s1, step) & Palace;
            t.unconstrainedKing[s1] |= safe_destination(s1, step);
        }

        // Jieqi: revealed advisors may leave the palace. A dark advisor may
        // not: see Position::legal(), attackers_to() and attacks_by().
        for (int step : {NORTH_WEST, NORTH_EAST, SOUTH_WEST, SOUTH_EAST})
            t.byType[ADVISOR][s1] |= safe_destination(s1, step);
    }

    return t;
}();

// Returns the pseudo attacks of the given piece type
// assuming an empty board. Pawns (PAWN, PAWN_TO) need the color. A cannon
// attacks nothing on an empty board: use the version with the occupancy.
sf_always_inline Bitboard attacks_bb(PieceType pt, Square s, Color c = COLOR_NB) {

    assert(is_ok(s));
    if (pt == PAWN || pt == PAWN_TO)
    {
        assert(c == WHITE || c == BLACK);
        return pt == PAWN ? PseudoAttacks.pawn[c][s] : PseudoAttacks.pawnTo[c][s];
    }

    assert(pt == ROOK || pt == ADVISOR || pt == KNIGHT || pt == BISHOP || pt == KING);
    return PseudoAttacks.byType[pt][s];
}

// Returns the attacks by the given piece
// assuming the board is occupied according to the passed Bitboard.
// Sliding piece attacks do not continue passed an occupied square.
sf_always_inline Bitboard attacks_bb(PieceType pt, Square s, Bitboard occupied) {

    assert(pt != PAWN && pt != PAWN_TO && is_ok(s));

    switch (pt)
    {
    case ROOK :
    case CANNON :
    case BISHOP :
    case KNIGHT :
    case KNIGHT_TO :
        return magic(s, pt).attack_bb(occupied);
    default :
        assert(pt == ADVISOR || pt == KING);
        return PseudoAttacks.byType[pt][s];
    }
}

// Steps of a king or an advisor ignoring the palace. (Advisors are not
// confined to the palace in Jieqi, so they are their usual attacks.)
sf_always_inline Bitboard unconstrained_attacks_bb(PieceType pt, Square s) {

    assert((pt == KING || pt == ADVISOR) && is_ok(s));
    return pt == KING ? PseudoAttacks.unconstrainedKing[s] : PseudoAttacks.byType[ADVISOR][s];
}

}  // namespace Maskfish::Attacks

#endif  // #ifndef ATTACKS_H_INCLUDED
