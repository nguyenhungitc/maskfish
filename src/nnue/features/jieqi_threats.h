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

// Threat input features of the Jieqi NNUE (engine built with `make threats=yes`).
//
// One feature per (attacker, from, to, attacked): the piece on `to` is attacked
// (or defended) by the piece on `from`. Idea from the "full threats" inputs of
// Stockfish and Pikafish, adapted to dark pieces:
//
// * pieces are grouped in 16 classes: the piece code for a revealed piece and
//   color * 8 for a dark piece;
// * a dark piece attacks like the kind of piece that starts on its square, a
//   dark advisor only inside the palace;
// * squares are seen from the perspective (rotated for black, mirrored like
//   the HalfKA features) and the colors of the classes are swapped for black;
// * threats between two pieces that always attack each other are counted once;
// * threats of a king, of an advisor to its own king, and checks of rooks,
//   cannons and knights are not features (Pikafish does the same).
//
// jieqi_features.py of nnue-pytorch implements exactly the same features ("Threat
// features" section); verify.py checks that both agree.

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
    // are a subset): at most 67 per side (2 rooks, 2 cannons x 4, 2 knights x 8,
    // 2 bishops, 2 advisors x 4, king x 4, 5 pawns x 3)
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
