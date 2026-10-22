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

#ifndef EVALUATE_H_INCLUDED
#define EVALUATE_H_INCLUDED

#include <string>

#include "types.h"

namespace Maskfish {

class Position;
class OptionsMap;

namespace Eval {

// The default net name. It MUST match the file name of the network in the
// hidden git ref refs/assets/net (maskfish.nnue), which scripts/net.sh fetches
// for the build (`make net`) and the build embeds in the binary. Do not change
// the name of the macro or the location where this macro is defined.
#define EvalFileDefaultNameBig "maskfish.nnue"

namespace NNUE {
struct Networks;
struct AccumulatorCaches;
class AccumulatorStack;
}

std::string trace(Position& pos, const Eval::NNUE::Networks& networks);

// Scaling of the network output in evaluate() (UCI option EvalScaling)
enum class EvalScaling {
    Complexity,  // |psqt - positional| and the major material (default)
    Alignment,   // agreement with simple_eval(), total material
    Correlation  // agreement with simple_eval_dark(), major material
};

// The value of the UCI option EvalScaling
EvalScaling eval_scaling(const OptionsMap& options);

Value evaluate(const NNUE::Networks&          networks,
               const Position&                pos,
               Eval::NNUE::AccumulatorStack&  accumulators,
               Eval::NNUE::AccumulatorCaches& caches,
               int                            optimism,
               bool                           jieqiMaterial  = false,
               EvalScaling                    scaling        = EvalScaling::Complexity,
               bool                           optimismSimple = false);
}  // namespace Eval

}  // namespace Maskfish

#endif  // #ifndef EVALUATE_H_INCLUDED
