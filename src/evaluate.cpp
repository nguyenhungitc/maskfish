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

#include "evaluate.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <tuple>

#include "bitboard.h"
#include "nnue/network.h"
#include "nnue/nnue_architecture.h"
#include "nnue/nnue_misc.h"
#include "position.h"
#include "types.h"
#include "uci.h"
#include "ucioption.h"
#include "nnue/nnue_accumulator.h"

namespace Maskfish {

namespace {

// Material balance from the side to move's point of view (a dark piece counts
// as the kind of piece that starts on its square, as in major_material())
int simple_eval(const Position& pos) {
    const Color c = pos.side_to_move();
    return PawnValue * (pos.count<PAWN>(c) - pos.count<PAWN>(~c))
         + AdvisorValue * (pos.count<ADVISOR>(c) - pos.count<ADVISOR>(~c))
         + BishopValue * (pos.count<BISHOP>(c) - pos.count<BISHOP>(~c))
         + pos.major_material(c) - pos.major_material(~c);
}

// Material of both sides, every piece but the kings
int total_material(const Position& pos) {
    return PawnValue * pos.count<PAWN>() + AdvisorValue * pos.count<ADVISOR>()
         + BishopValue * pos.count<BISHOP>() + pos.major_material();
}

// Material of the side to move minus that of the opponent (kings excluded), a
// dark piece counting at the average value of its side's pool
// (Position::dark_value()). Like simple_eval(), it stands for the
// PSQT part of the network in networks without one (make psqt=no) in
// EvalScaling Complexity, and is used by EvalScaling Correlation.
int simple_eval_dark(const Position& pos) {
    int material[COLOR_NB];
    for (Color c : {WHITE, BLACK})
    {
        // A dark piece counts as the kind of piece of its square here...
        material[c] = PawnValue * pos.count<PAWN>(c) + AdvisorValue * pos.count<ADVISOR>(c)
                    + BishopValue * pos.count<BISHOP>(c) + pos.major_material(c);

        // ...and is replaced by the expected value of its side's pool
        const Bitboard dark = pos.pieces(c, DARK);
        for (Bitboard b = dark; b;)
            material[c] -= PieceValue[pos.piece_on(pop_lsb(b))];
        material[c] += popcount(dark) * pos.dark_value(c);
    }

    const Color us = pos.side_to_move();
    return material[us] - material[~us];
}

}  // namespace

Eval::EvalScaling Eval::eval_scaling(const OptionsMap& options) {
    return options["EvalScaling"] == "Alignment"   ? EvalScaling::Alignment
         : options["EvalScaling"] == "Correlation" ? EvalScaling::Correlation
                                                   : EvalScaling::Complexity;
}

// Evaluate is the evaluator for the outer world. It returns a static evaluation
// of the position from the point of view of the side to move.
Value Eval::evaluate(const Eval::NNUE::Networks&    networks,
                     const Position&                pos,
                     Eval::NNUE::AccumulatorStack&  accumulators,
                     Eval::NNUE::AccumulatorCaches& caches,
                     int                            optimism,
                     bool                           jieqiMaterial,
                     EvalScaling                    scaling,
                     bool                           optimismSimple,
                     bool                           optimismMerged) {

    assert(!pos.checkers());

    auto [psqt, positional] = networks.big.evaluate(pos, accumulators, &caches.big);

    Value nnue = psqt + positional;
    int   v;

    if (scaling == EvalScaling::Alignment)
    {
        // EvalScaling Alignment (it does not use the psqt / positional split,
        // so it also suits networks without PSQT, make psqt=no). Normalize the
        // raw evaluations to [-1024, 1024] to measure their correlation: when
        // NNUE and material agree (positive alignment) the position is
        // straightforward, otherwise it involves complex compensation.
        const int se        = simple_eval(pos);
        const int se_norm   = (se * 1024) / (std::abs(se) + 1024);
        const int nnue_norm = int(std::int64_t(nnue) * 1024 / (std::abs(nnue) + 1024));
        const int align     = (se_norm * nnue_norm) / 512;

        // When winning, we favor easy positions, and vice versa. Optimism
        // has four times the weight of the evaluation in this term (UCI option
        // OptimismMerged: one division for both)
        const int base_eval =
          optimismMerged ? int(nnue + (std::int64_t(nnue) + 4 * optimism) * align / 65536)
                         : int(nnue + std::int64_t(nnue) * align / 65536
                               + std::int64_t(optimism) * align / 16384);

        // Scale the combined evaluation by the total material (JieqiMaterial
        // only applies to EvalScaling Complexity and Correlation)
        v = int(base_eval * std::int64_t(80030 + total_material(pos)) / 80030);
    }
    else if (scaling == EvalScaling::Correlation)
    {
        // EvalScaling Correlation: the same agreement of the network with the
        // material balance, but with dark pieces at the average value of their
        // pool (simple_eval_dark()) and the scaling by the major material of
        // EvalScaling Complexity. Optimism only enters through that agreement.
        // Modelled by the trainer: train.py --eval-scaling correlation
        // (trainer/eval_scale.py).
        const int material  = jieqiMaterial ? pos.jieqi_major_material() : pos.major_material();
        const int se        = simple_eval_dark(pos);
        const int se_norm   = (se * 1024) / (std::abs(se) + 1024);
        const int nnue_norm = (nnue * 1024) / (std::abs(nnue) + 1024);
        const int correl    = (se_norm * nnue_norm) / 512;

        const int base = optimismMerged
                         ? int(nnue + (std::int64_t(nnue) + 4 * optimism) * correl / 65536)
                         : nnue + (nnue * correl) / 65536 + (optimism * correl) / 16384;
        v              = int(std::int64_t(base) * (18560 + material) / 33331);
    }
    else
    {
        // EvalScaling Complexity (default): blend optimism and eval with nnue
        // complexity. Without PSQT (make psqt=no) psqt is 0 and |psqt -
        // positional| would be |nnue|: the material stands for the PSQT part
        // then, |psqt - (nnue - psqt)| with psqt = simple_eval_dark().
        int nnueComplexity;
        if constexpr (Eval::NNUE::UsePSQT)
            nnueComplexity = std::abs(psqt - positional);
        else
            nnueComplexity = std::abs(2 * simple_eval_dark(pos) - nnue);
        optimism += optimism * std::int64_t(nnueComplexity) / 488;
        nnue -= nnue * std::int64_t(nnueComplexity) / 11418;

        // UCI option OptimismSimple: the weight of optimism does not depend on
        // the material; the
        // constant is its value at the average major material of self-play games
        // (4990).
        const int material = jieqiMaterial ? pos.jieqi_major_material() : pos.major_material();
        const int optimismWeight = optimismSimple ? 2727 + 4990 : 2727 + material;
        v = int((nnue * std::int64_t(18560 + material) + optimism * std::int64_t(optimismWeight))
                / 33331);
    }

    // Damp down the evaluation linearly when shuffling. The counter is capped at
    // 80 plies (the game is drawn there): beyond 217 the factor would change the
    // sign of the evaluation, and the product could overflow an int. Unchanged
    // for every counter of a real game (see also trainer/eval_scale.py).
    v -= int(v * std::int64_t(std::min(pos.rule40_count(), 80)) / 217);

    // Guarantee evaluation does not hit the mate range
    v = std::clamp(v, VALUE_MATED_IN_MAX_PLY + 1, VALUE_MATE_IN_MAX_PLY - 1);

    return v;
}

// Like evaluate(), but instead of returning a value, it returns
// a string (suitable for outputting to stdout) that contains the detailed
// descriptions and values of each evaluation term. Useful for debugging.
// Trace scores are from white's point of view
std::string Eval::trace(Position& pos, const Eval::NNUE::Networks& networks) {

    if (pos.checkers())
        return "Final evaluation: none (in check)";

    auto accumulators = std::make_unique<Eval::NNUE::AccumulatorStack>();
    auto caches       = std::make_unique<Eval::NNUE::AccumulatorCaches>(networks);

    std::stringstream ss;
    ss << std::showpoint << std::noshowpos << std::fixed << std::setprecision(2);
    ss << '\n' << NNUE::trace(pos, networks, *caches) << '\n';

    ss << std::showpoint << std::showpos << std::fixed << std::setprecision(2) << std::setw(15);

    auto [psqt, positional] = networks.big.evaluate(pos, *accumulators, &caches->big);
    Value v                 = psqt + positional;
    v                       = pos.side_to_move() == WHITE ? v : -v;
    ss << "NNUE evaluation        " << 0.01 * UCIEngine::to_cp(v, pos) << " (white side)\n";

    v = evaluate(networks, pos, *accumulators, *caches, VALUE_ZERO);
    v = pos.side_to_move() == WHITE ? v : -v;
    ss << "Final evaluation       " << 0.01 * UCIEngine::to_cp(v, pos) << " (white side)";
    ss << " [with scaled NNUE, ...]";
    ss << "\n";

    return ss.str();
}

}  // namespace Maskfish
