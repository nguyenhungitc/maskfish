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

#pragma once
#include "position.h"

namespace jq {

// Indexed by PieceType (DARK unused). Constant: Position keeps the mean value of each urn (urnMean) from it.
extern const int PieceValue[9];
inline int urn_mean_value(const Position& pos, Color c) { return pos.urnMean[c]; }
Value classical_eval(const Position& pos);   // side-to-move perspective
Value evaluate(const Position& pos);         // the network if one is loaded, otherwise classical

}  // namespace jq
