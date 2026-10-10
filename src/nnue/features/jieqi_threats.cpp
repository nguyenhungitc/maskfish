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

// Definition of the JieqiThreats input features (see jieqi_threats.h)

#include "jieqi_threats.h"
#include "../../attacks.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include "../../bitboard.h"
#include "../../position.h"
#include "half_ka_v2_hm.h"

namespace Maskfish::Eval::NNUE::Features {

using namespace Maskfish::Attacks;

namespace {

constexpr int Classes = JieqiThreats::ClassNb;

// Index tables: for (attacker class, from, to), the index of the first feature
// and the set of attacked classes that are features (bit per class)
std::uint32_t ThreatBase[Classes][SQUARE_NB][SQUARE_NB];
std::uint16_t ThreatMask[Classes][SQUARE_NB][SQUARE_NB];

// Kind of piece that starts on each square (dark pieces), white at the bottom
constexpr std::string_view DarkStart("RNBAKABNR"
                                     "........."
                                     ".C.....C."
                                     "P.P.P.P.P"
                                     "........."
                                     "........."
                                     "p.p.p.p.p"
                                     ".c.....c."
                                     "........."
                                     "rnbakabnr");

bool on_board(int f, int r) { return f >= 0 && f < FILE_NB && r >= 0 && r < RANK_NB; }
int  sq(int f, int r) { return r * FILE_NB + f; }
bool in_palace(int s) {
    const int f = s % FILE_NB, r = s / FILE_NB;
    return f >= 3 && f <= 5 && (r <= 2 || r >= 7);
}

PieceType dark_kind(int s) {
    switch (DarkStart[s])
    {
    case 'R' :
    case 'r' :
        return ROOK;
    case 'A' :
    case 'a' :
        return ADVISOR;
    case 'C' :
    case 'c' :
        return CANNON;
    case 'P' :
    case 'p' :
        return PAWN;
    case 'N' :
    case 'n' :
        return KNIGHT;
    case 'B' :
    case 'b' :
        return BISHOP;
    default :
        return NO_PIECE_TYPE;
    }
}

// Squares a class can stand on
bool class_square(int cls, int s) {
    const Color     c  = Color(cls >> 3);
    const PieceType pt = PieceType(cls & 7);
    if (pt == NO_PIECE_TYPE)  // dark: its own starting squares
        return dark_kind(s) != NO_PIECE_TYPE && (DarkStart[s] < 'a') == (c == WHITE);
    if (pt == KING)
        return in_palace(s) && ((s / FILE_NB) < 5) == (c == WHITE);
    return true;
}

// Every square the class could attack from s on some board, as a bitset over
// squares (sliders on an empty board, cannons with a screen next to them).
// Written with plain loops, like jieqi_features.py of nnue-pytorch, so that both
// enumerate exactly the same features.
void empty_targets(int cls, int s, bool out[SQUARE_NB]) {
    std::fill(out, out + SQUARE_NB, false);
    const Color c    = Color(cls >> 3);
    const bool  dark = (cls & 7) == 0;
    PieceType   pt   = dark ? dark_kind(s) : PieceType(cls & 7);
    const int   f = s % FILE_NB, r = s / FILE_NB;

    switch (pt)
    {
    case ROOK :
    case CANNON :
        for (auto [df, dr] : {std::pair{0, 1}, {0, -1}, {1, 0}, {-1, 0}})
            for (int k = (pt == CANNON ? 2 : 1); on_board(f + k * df, r + k * dr); ++k)
                out[sq(f + k * df, r + k * dr)] = true;
        break;
    case KNIGHT :
        for (auto [df, dr] :
             {std::pair{1, 2}, {-1, 2}, {1, -2}, {-1, -2}, {2, 1}, {2, -1}, {-2, 1}, {-2, -1}})
            if (on_board(f + df, r + dr))
                out[sq(f + df, r + dr)] = true;
        break;
    case BISHOP :
        for (auto [df, dr] : {std::pair{2, 2}, {2, -2}, {-2, 2}, {-2, -2}})
            if (on_board(f + df, r + dr))
                out[sq(f + df, r + dr)] = true;
        break;
    case ADVISOR :
        for (auto [df, dr] : {std::pair{1, 1}, {1, -1}, {-1, 1}, {-1, -1}})
            if (on_board(f + df, r + dr) && (!dark || in_palace(sq(f + df, r + dr))))
                out[sq(f + df, r + dr)] = true;
        break;
    case KING :
        for (auto [df, dr] : {std::pair{0, 1}, {0, -1}, {1, 0}, {-1, 0}})
            if (on_board(f + df, r + dr) && in_palace(sq(f + df, r + dr)))
                out[sq(f + df, r + dr)] = true;
        break;
    case PAWN : {
        const int fwd = c == WHITE ? 1 : -1;
        if (on_board(f, r + fwd))
            out[sq(f, r + fwd)] = true;
        if (c == WHITE ? r > 4 : r < 5)
            for (int df : {1, -1})
                if (on_board(f + df, r))
                    out[sq(f + df, r)] = true;
        break;
    }
    default :
        break;
    }
}

// Threats between pieces that always attack each other are kept only in the
// direction from > to
bool semi_excluded(int a, int b, int from, int to) {
    const int pa = a & 7, pb = b & 7;
    if (pa == 0 || pa != pb || pa == KNIGHT)
        return false;
    if (pa == PAWN)
    {
        const bool enemy    = (a >> 3) != (b >> 3);
        const bool sameFile = from % FILE_NB == to % FILE_NB;
        const bool sameRank = from / FILE_NB == to / FILE_NB;
        if (!((enemy && sameFile) || (!enemy && sameRank)))
            return false;
    }
    return !(from > to);
}

// Threats involving the kings that are not features: the king as an attacker
// (its position and its neighbours are known from the other features), an
// advisor next to its own king, and the checks of rooks, cannons and knights
// (the network does not evaluate positions in check). A dark piece counts as
// the kind of piece of its starting square.
bool king_excluded(int a, int b, int from) {
    const int pa = (a & 7) ? (a & 7) : dark_kind(from);
    if (pa == KING)
        return true;
    if ((b & 7) != KING)
        return false;
    const bool own = (a >> 3) == (b >> 3);
    return own ? pa == ADVISOR : (pa == ROOK || pa == CANNON || pa == KNIGHT);
}

int class_of(const Position& pos, Square s) {
    const Piece pc = pos.piece_on(s);
    return pos.is_dark(s) ? (color_of(pc) << 3) : int(pc);
}

}  // namespace

void JieqiThreats::init() {
    IndexType n = 0;
    bool      targets[SQUARE_NB];

    for (int a = 0; a < Classes; ++a)
        for (int from = 0; from < SQUARE_NB; ++from)
        {
            for (int to = 0; to < SQUARE_NB; ++to)
                ThreatMask[a][from][to] = 0, ThreatBase[a][from][to] = 0;

            if (!class_square(a, from))
                continue;

            empty_targets(a, from, targets);
            for (int to = 0; to < SQUARE_NB; ++to)
            {
                if (!targets[to])
                    continue;
                std::uint16_t m = 0;
                for (int b = 0; b < Classes; ++b)
                    if (class_square(b, to) && !semi_excluded(a, b, from, to)
                        && !king_excluded(a, b, from))
                        m |= std::uint16_t(1 << b);
                ThreatBase[a][from][to] = n;
                ThreatMask[a][from][to] = m;
                n += popcount(Bitboard(m));
            }
        }

    if (n != Dimensions)
    {
        std::cerr << "JieqiThreats: " << n << " features instead of " << Dimensions << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

IndexType JieqiThreats::make_index(int attacker, int from, int to, int attacked) {
    const unsigned m = ThreatMask[attacker][from][to];
    if (!((m >> attacked) & 1))
        return Dimensions;
    return ThreatBase[attacker][from][to] + popcount(Bitboard(m & ((1u << attacked) - 1)));
}

int JieqiThreats::physical_threats(const Position& pos, Threat* out) {
    const Bitboard occupied = pos.pieces();
    Bitboard       bb       = occupied;
    int            n        = 0;

    while (bb)
    {
        const Square    from = pop_lsb(bb);
        const Piece     pc   = pos.piece_on(from);
        const PieceType pt   = type_of(pc);
        const bool      dark = pos.is_dark(from);
        Bitboard        attacks =
          pt == PAWN ? attacks_bb(PAWN, from, color_of(pc)) : attacks_bb(pt, from, occupied);
        if (dark && pt == ADVISOR)
            attacks &= Palace;  // a dark advisor must stay in the palace
        attacks &= occupied;

        const Threat head = Threat(from) << 24 | Threat(dark ? (color_of(pc) << 3) : int(pc)) << 8;
        while (attacks)
        {
            const Square to = pop_lsb(attacks);
            assert(n < int(MaxActiveDimensions));
            out[n++] = head | Threat(to) << 16 | Threat(class_of(pos, to));
        }
    }
    return n;
}

template<Color Perspective>
IndexType JieqiThreats::index(Threat t, bool mirror) {
    const auto& map  = HalfKAv2_hm::IndexMap[mirror][Perspective == BLACK];
    const int   flip = Perspective == BLACK ? 8 : 0;
    return make_index(int((t >> 8) & 0xFF) ^ flip, map[t >> 24], map[(t >> 16) & 0xFF],
                      int(t & 0xFF) ^ flip);
}

template IndexType JieqiThreats::index<WHITE>(Threat, bool);
template IndexType JieqiThreats::index<BLACK>(Threat, bool);

}  // namespace Maskfish::Eval::NNUE::Features
