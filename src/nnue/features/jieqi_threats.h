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

// Threat input features of the Jieqi NNUE (engine built with `make threats=yes`).
//
// One feature per (attacker, from, to, attacked): the piece on `to` is attacked
// (or defended) by the piece on `from`, adapted to dark pieces:
//
// * pieces are grouped in 16 classes: the piece code for a revealed piece and
//   color * 8 for a dark piece;
// * a dark piece attacks like the kind of piece that starts on its square, a
//   dark advisor only inside the palace;
// * squares are seen from the perspective (rotated for black, mirrored like
//   the HalfKA features) and the colors of the classes are swapped for black;
// * threats between two pieces that always attack each other are counted once;
// * threats of a king, of an advisor to its own king, and checks of rooks,
//   cannons and knights are not features.
//
// trainer/jieqi_features.py implements exactly the same features ("Threat
// features" section); trainer/verify.py checks that both agree.

#ifndef NNUE_FEATURES_JIEQI_THREATS_H_INCLUDED
#define NNUE_FEATURES_JIEQI_THREATS_H_INCLUDED

#include <cstdint>

#include "../../misc.h"
#include "../../types.h"
#include "../nnue_common.h"

namespace Maskfish {
class Position;
}

namespace Maskfish::Eval::NNUE::Features {

class JieqiThreats {
   public:
    static constexpr const char* Name = "JieqiThreats";

    // Hash value embedded in the evaluation file (xored with the HalfKA one)
    // (earlier layouts: 0x5B1E7A39 with LEB128 threat weights, 0x5B1E7A3B
    // with the king features)
    static constexpr std::uint32_t HashValue = 0x5B1E7A3D;

    // Number of feature dimensions (checked when the tables are built)
    static constexpr IndexType Dimensions = 92711;

    // Maximum number of simultaneously active threats (of which the features
    // are a subset), i.e. of attacked occupied squares: at most 69 per side,
    // 138 in all. A dark piece attacks like the kind of its starting square,
    // so a side may have more than 2 rook, cannon or knight movers, but it has
    // at most 15 pieces besides its king (Position::pool_is_consistent()), at
    // most 2 revealed of a kind (5 pawns), and a dark piece attacks few
    // squares from its starting square. The 15 best attackers: 2 revealed
    // knights (8 squares each), 10 pieces of 4 squares (revealed rooks,
    // cannons, bishops, advisors, and dark cannons on b2 / h2), 3 of 3 (pawns
    // across the river, or dark knights on b0 / h0): 65, and the king 4.
    // physical_threats() only asserts this bound.
    static constexpr IndexType MaxActiveDimensions = 144;

    static constexpr int ClassNb = 16;

    // A threat on the board, packed as from << 24 | to << 16 | attacker class
    // << 8 | attacked class. physical_threats() lists them by ascending from
    // and to squares, that is in ascending order of the packed values.
    using Threat = std::uint32_t;

    // Builds the index tables (called once at startup)
    static void init();

    // All the threats of a position, in ascending order; returns their number
    // (at most MaxActiveDimensions)
    static int physical_threats(const Position& pos, Threat* out);

    // Feature index of a threat for a perspective (squares mapped like the
    // HalfKA features, classes color-swapped for black); Dimensions if the
    // threat is not a feature
    template<Color Perspective>
    static IndexType index(Threat t, bool mirror);

    // Index of a threat whose squares are mapped to the perspective and whose
    // classes are color-swapped for black; Dimensions if not a feature
    static IndexType make_index(int attacker, int from, int to, int attacked);
};

}  // namespace Maskfish::Eval::NNUE::Features

#endif  // #ifndef NNUE_FEATURES_JIEQI_THREATS_H_INCLUDED
