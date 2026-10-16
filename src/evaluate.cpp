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
#include "nnue/nnue_accumulator.h"

namespace Maskfish {

namespace {

// Material of the side to move minus that of the opponent (kings excluded), a
// dark piece counting at the average value of its side's pool
// (Position::dark_value()). As Pikafish's simple_eval(), it stands for the
// PSQT part of the network in networks without one (make psqt=no).
int simple_eval(const Position& pos) {
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

// Evaluate is the evaluator for the outer world. It returns a static evaluation
// of the position from the point of view of the side to move.
Value Eval::evaluate(const Eval::NNUE::Networks&    networks,
                     const Position&                pos,
                     Eval::NNUE::AccumulatorStack&  accumulators,
                     Eval::NNUE::AccumulatorCaches& caches,
                     int                            optimism,
                     bool                           jieqiMaterial,
                     bool                           optimismSimple,
                     bool                           evalCorrelation) {

    assert(!pos.checkers());

    auto [psqt, positional] = networks.big.evaluate(pos, accumulators, &caches.big);

    Value nnue     = psqt + positional;
    int   material = jieqiMaterial ? pos.jieqi_major_material() : pos.major_material();
    int   v;

    // UCI option EvalCorrelation (Pikafish "Replace nnueComplexity with
    // simple_eval equivalent"): the agreement of the network with the
    // material balance (simple_eval(), dark pieces at the average value of
    // their pool) takes the place of the complexity blending. When they agree
    // the position is easy and the evaluation is boosted when winning; when
    // they disagree there is compensation, and the evaluation is damped.
    // Optimism only enters through that agreement. The scaling by material is
    // kept (the networks are trained through it, see trainer/eval_scale.py,
    // which does not model this option).
    if (evalCorrelation)
    {
        const int se        = simple_eval(pos);
        const int se_norm   = (se * 1024) / (std::abs(se) + 1024);
        const int nnue_norm = (nnue * 1024) / (std::abs(nnue) + 1024);
        const int correl    = (se_norm * nnue_norm) / 512;

        const int base = nnue + (nnue * correl) / 65536 + (optimism * correl) / 16384;
        v              = int(std::int64_t(base) * (18560 + material) / 33331);
    }
    else
    {
        // Blend optimism and eval with nnue complexity. Without PSQT (make
        // psqt=no) psqt is 0 and |psqt - positional| would be |nnue|: as in
        // Pikafish ("Replace nnueComplexity with simple_eval equivalent"), the
        // material stands for the PSQT part then, |psqt - (nnue - psqt)| with
        // psqt = simple_eval().
        int nnueComplexity;
        if constexpr (Eval::NNUE::UsePSQT)
            nnueComplexity = std::abs(psqt - positional);
        else
            nnueComplexity = std::abs(2 * simple_eval(pos) - nnue);
        optimism += optimism * std::int64_t(nnueComplexity) / 488;
        nnue -= nnue * std::int64_t(nnueComplexity) / 11418;

        // UCI option OptimismSimple (Pikafish "Simplify optimism scaling
        // formula"): the weight of optimism does not depend on the material; the
        // constant is its value at the average major material of self-play games
        // (4990).
        const int optimismWeight = optimismSimple ? 2727 + 4990 : 2727 + material;
        v = (nnue * std::int64_t(18560 + material) + optimism * std::int64_t(optimismWeight)) / 33331;
    }

    // Damp down the evaluation linearly when shuffling
    v -= (v * pos.rule40_count()) / 217;

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
