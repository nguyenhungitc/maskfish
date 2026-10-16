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

#include "movegen.h"

#include <cassert>

#include "attacks.h"
#include "bitboard.h"
#include "position.h"

namespace Maskfish {

using namespace Attacks;

namespace {

// Squares where a piece of the side to move may land, given the occupancy
// without the moving piece and the checkers for that occupancy: every check
// must be blocked or the checker captured, and no piece may be put between an
// unobstructed cannon and the king (it would become the cannon's screen).
Bitboard legal_targets(const Position& pos, Bitboard occupied, Bitboard checkers) {
    Color    us     = pos.side_to_move();
    Square   ksq    = pos.king_square(us);
    Bitboard target = ~pos.pieces(us);

    // Every remaining check must be blocked or captured. The between squares
    // of a knight or bishop check are the leg or the eye of the checker.
    while (checkers)
    {
        Square s = pop_lsb(checkers);
        target &= type_of(pos.piece_on(s)) == CANNON ? (between_bb(ksq, s) & ~occupied) | s
                                                     : between_bb(ksq, s);
    }

    // Placing a piece between an unobstructed cannon and our king creates a check.
    Bitboard cannons = attacks_bb(ROOK, ksq, occupied) & pos.pieces(~us, CANNON);
    while (cannons)
    {
        Square s = pop_lsb(cannons);
        target &= ~(between_bb(ksq, s) ^ square_bb(s));
    }

    return target;
}

// Generates the legal moves of the pieces of type Pt. A piece that is not
// constrained (not a blocker of a check on its king, not the screen of a
// checking cannon) only has to land on safeTarget. A constrained piece gets
// its targets from the occupancy without it.
template<PieceType Pt, GenType Type>
Move* generate_moves(const Position& pos,
                     Move*           moveList,
                     Bitboard        target,
                     Color           us,
                     Bitboard        constrained,
                     Bitboard        safeTarget) {

    static_assert(Pt != KING, "Unsupported piece type in generate_moves()");

    Bitboard bb = pos.pieces(us, Pt);

    while (bb)
    {
        Square   from       = pop_lsb(bb);
        Bitboard b          = 0;
        Bitboard moveTarget = constrained & from ? target : safeTarget;
        if constexpr (Pt != CANNON)
            b = (Pt != PAWN ? attacks_bb(Pt, from, pos.pieces()) : attacks_bb(PAWN, from, us))
              & moveTarget;
        else
        {
            // Generate cannon capture moves.
            if (Type != QUIETS)
                b |= attacks_bb(CANNON, from, pos.pieces()) & pos.pieces(~us);

            // Generate cannon quiet moves.
            if (Type != CAPTURES)
                b |= attacks_bb(ROOK, from, pos.pieces()) & ~pos.pieces();

            b &= moveTarget;
        }

        // Jieqi: a dark advisor must stay in the palace
        if constexpr (Pt == ADVISOR)
            if (pos.is_dark(from))
                b &= Palace;

        if (b && (constrained & from))
        {
            Bitboard occupied = pos.pieces() ^ from;
            b &= legal_targets(pos, occupied, pos.checkers_to(~us, pos.king_square(us), occupied));
        }

        while (b)
            *moveList++ = Move(from, pop_lsb(b));
    }

    return moveList;
}

template<GenType Type>
Move* generate_all(const Position& pos, Move* moveList) {

    static_assert(Type != ALL, "Unsupported type in generate_all()");

    const Color  us     = pos.side_to_move();
    const Square ksq    = pos.king_square(us);
    Bitboard     target = Type == EVASIONS ? ~pos.pieces(us)
                        : Type == CAPTURES ? pos.pieces(~us)
                                           : ~pos.pieces();  // QUIETS

    // Blockers include the screens of cannons with two pieces in between, and
    // the legs and eyes of knights and bishops (see update_blockers()).
    Bitboard constrained = pos.blockers_for_king(us);
    // Moving a checking cannon's screen also changes the set of legal targets.
    Bitboard checkers = pos.checkers() & pos.pieces(~us, CANNON);
    while (checkers)
        constrained |= between_bb(ksq, pop_lsb(checkers)) & pos.pieces(us);

    Bitboard safeTarget = target & legal_targets(pos, pos.pieces(), pos.checkers());

    moveList = generate_moves<PAWN, Type>(pos, moveList, target, us, constrained, safeTarget);
    moveList = generate_moves<BISHOP, Type>(pos, moveList, target, us, constrained, safeTarget);
    moveList = generate_moves<ADVISOR, Type>(pos, moveList, target, us, constrained, safeTarget);
    moveList = generate_moves<KNIGHT, Type>(pos, moveList, target, us, constrained, safeTarget);
    moveList = generate_moves<CANNON, Type>(pos, moveList, target, us, constrained, safeTarget);
    moveList = generate_moves<ROOK, Type>(pos, moveList, target, us, constrained, safeTarget);

    // The king may not move to an attacked square. The flying general is
    // included in checkers_to(), and the king's own square is removed from
    // the occupancy so it cannot step back along the line of a slider check.
    Bitboard b = attacks_bb(KING, ksq) & target;
    if (b && Position::useThreatCache)
    {
        // UCI option ThreatCache: the same test with the squares attacked by
        // the opponent through the king, cached in the StateInfo (Pikafish)
#ifndef NDEBUG
        for (Bitboard t = b; t;)
        {
            const Square to = pop_lsb(t);
            assert(bool(to & pos.threats_by(ALL_PIECES))
                   == bool(pos.checkers_to(~us, to, (pos.pieces() ^ ksq) | to)));
        }
#endif
        b &= ~pos.threats_by(ALL_PIECES);
        while (b)
            *moveList++ = Move(ksq, pop_lsb(b));
    }
    else if (b)
    {
        const Bitboard occupied = pos.pieces() ^ ksq;
        while (b)
        {
            Square to = pop_lsb(b);
            if (!pos.checkers_to(~us, to, occupied | to))
                *moveList++ = Move(ksq, to);
        }
    }

    return moveList;
}

}  // namespace


// <CAPTURES> Generates all legal captures
// <QUIETS>   Generates all legal non-captures
// <EVASIONS> Generates all legal check evasions
//
// Returns a pointer to the end of the move list.
template<GenType Type>
Move* generate(const Position& pos, Move* moveList) {

    static_assert(Type != ALL, "Unsupported type in generate()");
    assert((Type == EVASIONS) == bool(pos.checkers()));

    return generate_all<Type>(pos, moveList);
}

// Explicit template instantiations
template Move* generate<CAPTURES>(const Position&, Move*);
template Move* generate<QUIETS>(const Position&, Move*);
template Move* generate<EVASIONS>(const Position&, Move*);

// generate<ALL> generates all the legal moves in the given position
template<>
Move* generate<ALL>(const Position& pos, Move* moveList) {
    return pos.checkers() ? generate<EVASIONS>(pos, moveList)
                          : generate<QUIETS>(pos, generate<CAPTURES>(pos, moveList));
}

}  // namespace Maskfish
