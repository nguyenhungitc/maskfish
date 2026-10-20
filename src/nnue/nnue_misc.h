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

#ifndef NNUE_MISC_H_INCLUDED
#define NNUE_MISC_H_INCLUDED

#include <cstddef>
#include <memory>
#include <string>

#include "../misc.h"
#include "../types.h"
#include "nnue_architecture.h"

namespace Maskfish {

class Position;

namespace Eval::NNUE {

// EvalFile uses fixed string types because it's part of the network structure which must be trivial.
struct EvalFile {
    // Default net name, will use one of the EvalFileDefaultName* macros defined
    // in evaluate.h
    FixedString<256> defaultName;
    // Selected net name, either via uci option or default
    FixedString<256> current;
    // Net description extracted from the net file
    FixedString<256> netDescription;
};

struct NnueEvalTrace {
    // psqt[b]: the PSQT bucket of the layer stack b (psqt_bucket())
    Value       psqt[LayerStacks];
    Value       positional[LayerStacks];
    std::size_t correctBucket;
};

struct Networks;
struct AccumulatorCaches;

std::string trace(Position& pos, const Networks& networks, AccumulatorCaches& caches);

}  // namespace Maskfish::Eval::NNUE
}  // namespace Maskfish

template<>
struct std::hash<Maskfish::Eval::NNUE::EvalFile> {
    std::size_t operator()(const Maskfish::Eval::NNUE::EvalFile& evalFile) const noexcept {
        std::size_t h = 0;
        Maskfish::hash_combine(h, evalFile.defaultName);
        Maskfish::hash_combine(h, evalFile.current);
        Maskfish::hash_combine(h, evalFile.netDescription);
        return h;
    }
};

#endif  // #ifndef NNUE_MISC_H_INCLUDED
