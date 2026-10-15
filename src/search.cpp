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

#include "search.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <initializer_list>
#include <string>
#include <utility>

#include "evaluate.h"
#include "history.h"
#include "misc.h"
#include "movegen.h"
#include "movepick.h"
#include "nnue/network.h"
#include "nnue/nnue_accumulator.h"
#include "position.h"
#include "thread.h"
#include "timeman.h"
#include "tt.h"
#include "uci.h"
#include "ucioption.h"

namespace Maskfish {

using namespace Search;

namespace {

// Conversion between scores and win rates, used to average the outcomes of
// chance nodes (flip_search, hidden_capture_search).
// The scaling is 360.83524 unless the UCI option ChanceScaling is set.
double flip_score_to_winrate(Value v, double scaling) { return 1.0 / (1.0 + std::exp(-v / scaling)); }

// Inverse of flip_score_to_winrate(), in non-decisive scores. The direction of
// rounding matters: a lower bound is rounded down, an upper bound up, and an
// estimate to the nearest score. (Truncation used to turn about 30% of the
// scores into a neighbouring one: a chance node whose outcomes all returned
// beta could return beta - 1.)
enum class Rounding {
    Nearest,
    Down,
    Up
};

Value flip_winrate_to_score(double winrate, double scaling, Rounding rounding = Rounding::Nearest) {
    constexpr double epsilon = 1e-9;
    winrate                  = std::clamp(winrate, epsilon, 1.0 - epsilon);
    const double q           = std::log(winrate / (1.0 - winrate)) * scaling;
    const double r           = rounding == Rounding::Down ? std::floor(q)
                             : rounding == Rounding::Up   ? std::ceil(q)
                                                          : std::round(q);
    return Value(
      std::clamp(r, double(VALUE_MATED_IN_MAX_PLY + 1), double(VALUE_MATE_IN_MAX_PLY - 1)));
}

constexpr int SEARCHEDLIST_CAPACITY = 32;
constexpr int MainHistoryDefault    = 51;

// History divisor of the lmrDepth adjustment by depth (UCI option
// LMRHistoryDivisor): Pikafish's formula (which replaced its tuned table of
// 16 values), scaled from its constant 3484 to this engine's 3626
constexpr int lmr_divisor(int depth) {
    const int d = std::min(depth, 16);
    return (3000 + 7 * (d - 8) * (d - 8)) * 3626 / 3484;
}
using SearchedList                  = ValueList<Move, SEARCHEDLIST_CAPACITY>;

// Reductions lookup table indexed by depth or move number, shared by all the
// workers
const std::array<int, std::max(MAX_PLY, MAX_MOVES) + 10> reductions = [] {
    std::array<int, std::max(MAX_PLY, MAX_MOVES) + 10> table{};
    for (size_t i = 1; i < table.size(); ++i)
        table[i] = int(2321 / (849 / 10.0) * std::log(i));
    return table;
}();

// (*Scalers):
// The values with Scaler asterisks have proven non-linear scaling.
// They are optimized to time controls of 180 + 1.8 and longer,
// so changing them or adding conditions that are similar requires
// tests at these types of time controls.

int correction_value(const Worker& w, const Position& pos, const Stack* const ss) {
    const Color us    = pos.side_to_move();
    const auto  m     = (ss - 1)->currentMove;
    const auto  pcv   = w.pawnCorrectionHistory[pawn_structure_index<Correction>(pos)][us];
    const auto  micv  = w.minorPieceCorrectionHistory[minor_piece_index(pos)][us];
    const auto  wnpcv = w.nonPawnCorrectionHistory[non_pawn_index<WHITE>(pos)][WHITE][us];
    const auto  bnpcv = w.nonPawnCorrectionHistory[non_pawn_index<BLACK>(pos)][BLACK][us];
    const auto  cntcv =
      m.is_ok()
        ? (*(ss - 2)->continuationCorrectionHistory)[pos.piece_on(m.to_sq())][m.to_sq()]
            + (w.contCorr4Ply
                 ? (*(ss - 4)->continuationCorrectionHistory)[pos.piece_on(m.to_sq())][m.to_sq()]
                 : 0)
        : 0;
    // 6 plies back (UCI option ContCorr6Ply), with a smaller weight
    const int cntcv6 =
      w.contCorr6Ply && m.is_ok()
        ? 6657 * (*(ss - 6)->continuationCorrectionHistory)[pos.piece_on(m.to_sq())][m.to_sq()]
        : 0;

    return 4897 * pcv + 4190 * micv + 8120 * (wnpcv + bnpcv) + 8324 * cntcv + cntcv6;
}

// Add correctionHistory value to raw staticEval and guarantee evaluation
// does not hit the tablebase range.
Value to_corrected_static_eval(const Value v, const int cv) {
    return std::clamp(v + cv / 131072, VALUE_MATED_IN_MAX_PLY + 1, VALUE_MATE_IN_MAX_PLY - 1);
}

void update_correction_history(const Position& pos,
                               Stack* const    ss,
                               Search::Worker& workerThread,
                               const int       bonus) {
    const Move  m  = (ss - 1)->currentMove;
    const Color us = pos.side_to_move();

    int nonPawnWeight = 129;

    workerThread.pawnCorrectionHistory[pawn_structure_index<Correction>(pos)][us]
      << bonus * 136 / 128;
    workerThread.minorPieceCorrectionHistory[minor_piece_index(pos)][us] << bonus * 88 / 128;
    workerThread.nonPawnCorrectionHistory[non_pawn_index<WHITE>(pos)][WHITE][us]
      << bonus * nonPawnWeight / 128;
    workerThread.nonPawnCorrectionHistory[non_pawn_index<BLACK>(pos)][BLACK][us]
      << bonus * nonPawnWeight / 128;

    if (m.is_ok())
    {
        const Square to = m.to_sq();
        const Piece  pc = pos.piece_on(to);
        (*(ss - 2)->continuationCorrectionHistory)[pc][to] << bonus * 114 / 128;

        if (workerThread.contCorr4Ply)
            (*(ss - 4)->continuationCorrectionHistory)[pc][to] << bonus * 64 / 128;

        if (workerThread.contCorr6Ply)
            (*(ss - 6)->continuationCorrectionHistory)[pc][to] << bonus * 35 / 128;
    }
}

// Add a small random component to draw evaluations to avoid 3-fold blindness
Value value_draw(size_t nodes) { return VALUE_DRAW - 1 + Value(nodes & 0x2); }
Value value_to_tt(Value v, int ply);
Value value_from_tt(Value v, int ply, int r40c);
void  update_pv(Move* pv, Move move, const Move* childPv);
void  update_continuation_histories(
  const Search::Worker& w, Stack* ss, Piece pc, Square to, int bonus);
void  update_quiet_histories(
   const Position& pos, Stack* ss, Search::Worker& workerThread, Move move, int bonus);
void update_all_stats(const Position& pos,
                      Stack*          ss,
                      Search::Worker& workerThread,
                      Move            bestMove,
                      Square          prevSq,
                      SearchedList&   quietsSearched,
                      SearchedList&   capturesSearched,
                      Depth           depth,
                      Move            TTMove,
                      int             moveCount,
                      bool            PvNode);

// Detect shuffling moves (a piece going back and forth) in order to limit
// search explosions. Captures are never shuffling; a reveal cannot be either,
// since a dark piece always stands on its starting square. This code was added
// upstream as a non-regression fix, so its parameters should not be tuned.
bool is_shuffling(Move move, Stack* const ss, const Position& pos) {
    if (pos.capture(move) || pos.rule40_count() < 10)
        return false;
    if (pos.state()->pliesFromNull <= 6 || ss->ply < 20)
        return false;
    return move.from_sq() == (ss - 2)->currentMove.to_sq()
        && (ss - 2)->currentMove.from_sq() == (ss - 4)->currentMove.to_sq();
}

}  // namespace

Search::Worker::Worker(SharedState&                    sharedState,
                       std::unique_ptr<ISearchManager> sm,
                       size_t                          threadId,
                       NumaReplicatedAccessToken       token) :
    sharedHistories(sharedState.threads.claim_shared_histories(token.get_numa_index(),
                                                               ownsSharedHistories)),
    continuationHistory(sharedHistories ? sharedHistories->continuationHistory
                                        : ownHistories.continuationHistory),
    pawnHistory(sharedHistories ? sharedHistories->pawnHistory : ownHistories.pawnHistory),
    pawnCorrectionHistory(sharedHistories ? sharedHistories->pawnCorrectionHistory
                                          : ownHistories.pawnCorrectionHistory),
    minorPieceCorrectionHistory(sharedHistories ? sharedHistories->minorPieceCorrectionHistory
                                                : ownHistories.minorPieceCorrectionHistory),
    nonPawnCorrectionHistory(sharedHistories ? sharedHistories->nonPawnCorrectionHistory
                                             : ownHistories.nonPawnCorrectionHistory),
    continuationCorrectionHistory(sharedHistories
                                    ? sharedHistories->continuationCorrectionHistory
                                    : ownHistories.continuationCorrectionHistory),
    // Unpack the SharedState struct into member variables
    threadIdx(threadId),
    numaAccessToken(token),
    manager(std::move(sm)),
    options(sharedState.options),
    threads(sharedState.threads),
    tt(sharedState.tt),
    networks(sharedState.networks),
    refreshTable(networks[token]) {
    clear();
}

void Search::Worker::ensure_network_replicated() {
    // Access once to force lazy initialization.
    // We do this because we want to avoid initialization during search.
    (void) (networks[numaAccessToken]);
}

void Search::Worker::start_searching() {

    accumulatorStack.reset();

    // Non-main threads go directly to iterative_deepening()
    if (!is_mainthread())
    {
        iterative_deepening();
        return;
    }

    main_manager()->tm.init(limits, rootPos.side_to_move(), rootPos.game_ply(), options,
                            main_manager()->originalTimeAdjust);
    tt.new_search();
    main_manager()->updates.onStart();

    // Whether the last PV sent to the GUI matches the final result
    bool uciPvSent = true;

    if (rootMoves.empty())
    {
        rootMoves.emplace_back(Move::none());
        main_manager()->updates.onUpdateNoMoves({0, {-VALUE_MATE, rootPos}});
    }
    else
    {
        threads.start_searching();               // start non-main threads
        uciPvSent = iterative_deepening();       // main thread start searching
    }

    // When we reach the maximum depth, we can arrive here without a raise of
    // threads.stop. However, if we are pondering or in an infinite search,
    // the UCI protocol states that we shouldn't print the best move before the
    // GUI sends a "stop" or "ponderhit" command. We therefore simply wait here
    // until the GUI sends one of those commands.
    while (!threads.stop && (main_manager()->ponder || limits.infinite))
    {}  // Busy wait for a stop or a ponder reset

    // Stop the threads if not already stopped (also raise the stop if
    // "ponderhit" just reset threads.ponder)
    threads.stop = true;

    // Wait until all threads have finished
    threads.wait_for_search_finished();

    // When playing in 'nodes as time' mode, subtract the searched nodes from
    // the available ones before exiting.
    if (limits.npmsec)
        main_manager()->tm.advance_nodes_time(threads.nodes_searched()
                                              - limits.inc[rootPos.side_to_move()]);

    Worker* bestThread = this;

    if (!limits.depth && rootMoves[0].pv[0] != Move::none())
        bestThread = threads.get_best_thread()->worker.get();

    main_manager()->bestPreviousScore        = bestThread->rootMoves[0].score;
    main_manager()->bestPreviousAverageScore = bestThread->rootMoves[0].averageScore;

    if (bestThread->rootMoves[0].pv.size() == 1
        && bestThread->rootMoves[0].extract_ponder_from_tt(tt, rootPos, rootObserver))
        uciPvSent = false;

    // Send PV info if it has changed since the last output in iterative_deepening(),
    // so that the last PV always matches bestmove and the ponder move
    if (!uciPvSent || bestThread != this)
        main_manager()->pv(*bestThread, threads, tt, bestThread->rootDepth);

    std::string ponder;

    // No ponder move after a move of a dark piece (the reply depends on the
    // reveal), and no ponder move that moves a dark piece itself: the GUI could
    // not send it back, since a move of a dark piece must name the revealed piece.
    // (pv[0] moves a piece of the side to move, so it does not change whether
    // the 'from' square of pv[1] holds a dark piece.)
    if (bestThread->rootMoves[0].pv.size() > 1
        && !rootPos.move_dark(bestThread->rootMoves[0].pv[0])
        && !rootPos.move_dark(bestThread->rootMoves[0].pv[1]))
        ponder = UCIEngine::move(bestThread->rootMoves[0].pv[1]);

    auto bestmove = UCIEngine::move(bestThread->rootMoves[0].pv[0]);
    main_manager()->updates.onBestmove(bestmove, ponder);
}

// Main iterative deepening loop. It calls search()
// repeatedly with increasing depth until the allocated thinking time has been
// consumed, the user stops the search, or the maximum search depth is reached.
bool Search::Worker::iterative_deepening() {

    SearchManager* mainThread = (is_mainthread() ? main_manager() : nullptr);

    Move pv[MAX_PLY + 1];

    // Best move, PV and score of the last completed iteration (except that a
    // shorter mate found earlier is kept, see forgottenMate below)
    std::vector<Move> lastBestMovePV;
    Depth             lastBestMoveDepth = 0;
    Value             lastBestMoveScore = -VALUE_INFINITE;
    bool              uciPvSent         = false;

    Value  alpha, beta;
    Value  bestValue     = -VALUE_INFINITE;
    Color  us            = rootPos.side_to_move();
    double timeReduction = 1, totBestMoveChanges = 0;

    // Jieqi information model and reveal heuristics (see search.h)
    rootObserver       = us;  // also when pondering: the root is then after the expected reply
    hiddenCaptureModel = bool(options["HiddenCaptureModel"]);
    probCutDarkCapture = bool(options["ProbCutDarkCapture"]);
    ttPenalize         = bool(options["TTPenalize"]);
    chanceBranchDepth  = bool(options["ChanceBranchDepth"]);
    contCorr4Ply       = bool(options["ContCorr4Ply"]);
    multiCutHistory    = bool(options["MultiCutHistory"]);
    cmhc               = bool(options["CMHC"]);
    seekMateOption     = bool(options["SeekMate"]);
    pruningTweaks      = bool(options["PruningTweaks"]);
    lmrTweaks          = bool(options["LMRTweaks"]);
    failHighRecoveryOption = bool(options["FailHighRecovery"]);
    aspirationTweaks = bool(options["AspirationTweaks"]);
    mainHistoryDecay = bool(options["MainHistoryDecay"]);
    malusDecay = bool(options["MalusDecay"]);
    bestMoveBonusScale = bool(options["BestMoveBonusScale"]);
    lmrHistoryDivisor = bool(options["LMRHistoryDivisor"]);
    simplifications        = bool(options["Simplifications"]);
    contHistOffset = bool(options["ContHistOffset"]);
    corrHistAsym = bool(options["CorrHistAsym"]);
    contCorr6Ply = bool(options["ContCorr6Ply"]);
    hindsightFirst = bool(options["HindsightFirst"]);
    followPVOption = bool(options["FollowPV"]);
    cutoffCntTweak = bool(options["CutoffCntTweak"]);
    nmpFailHighCount = bool(options["NMPFailHighCount"]);
    mateTTExtension = bool(options["MateTTExtension"]);
    stopEarly = bool(options["StopEarly"]);
    rootScoreEMA = bool(options["RootScoreEMA"]);
    standPatNoTT = bool(options["StandPatNoTT"]);
    bigPawnHistory = bool(options["BigPawnHistory"]);
    jieqiMaterial      = bool(options["JieqiMaterial"]);
    chanceScaling      = int(options["ChanceScaling"]) ? double(int(options["ChanceScaling"])) : 360.83524;
    chanceRisk         = int(options["ChanceRiskAversion"]) / 100.0;
    revealQuiet        = {int(options["RevealQuietBase"]), int(options["RevealQuietPhase"]),
                          int(options["RevealQuietSafety"]), int(options["RevealQuietHighValue"]),
                          int(options["RevealQuietDiversity"])};
    int    delta, iterIdx                        = 0;
    int    failHighRecovery                      = 0;  // UCI option FailHighRecovery

    // The search adds the safe reveal bonus to the value of a quiet move of a
    // dark piece: keep it out of the score reported for the root moves
    for (RootMove& rm : rootMoves)
        rm.revealBonus = Reveal::move_bonus(rootPos, rm.pv[0], revealQuiet);

    // Allocate stack with extra size to allow access from (ss - 7) to (ss + 2):
    // (ss - 7) is needed for update_continuation_histories(ss - 1) which accesses (ss - 6),
    // (ss + 2) is needed for initialization of cutOffCnt.
    Stack  stack[MAX_PLY + 10] = {};
    Stack* ss                  = stack + 7;

    for (int i = 7; i > 0; --i)
    {
        (ss - i)->continuationHistory =
          &this->continuationHistory[0][0][NO_PIECE][0];  // Use as a sentinel
        (ss - i)->continuationCorrectionHistory = &this->continuationCorrectionHistory[NO_PIECE][0];
        (ss - i)->staticEval                    = VALUE_NONE;
    }

    for (int i = 0; i <= MAX_PLY + 2; ++i)
        (ss + i)->ply = i;

    ss->pv = pv;

    if (mainThread)
    {
        if (mainThread->bestPreviousScore == VALUE_INFINITE)
            mainThread->iterValue.fill(VALUE_ZERO);
        else
            mainThread->iterValue.fill(mainThread->bestPreviousScore);
    }

    size_t multiPV = size_t(options["MultiPV"]);

    multiPV = std::min(multiPV, rootMoves.size());

    int searchAgainCounter = 0;

    lowPlyHistory.fill(102);

    // Main history of earlier searches is partly forgotten (UCI option
    // MainHistoryDecay)
    if (mainHistoryDecay)
        for (Color c : {WHITE, BLACK})
            for (auto& h : mainHistory[c])
                h = (h - MainHistoryDefault) * 3 / 4 + MainHistoryDefault;

    // Iterative deepening loop until requested to stop or the target depth is reached
    // (rootDepth always holds the last depth for which a search was started)
    while (rootDepth + 1 < MAX_PLY && !threads.stop
           && !(limits.depth && mainThread && rootDepth >= limits.depth))
    {
        rootDepth++;

        // Age out PV variability metric and signal the start of a new iteration
        if (mainThread)
        {
            totBestMoveChanges /= 2;
            uciPvSent = false;
        }

        // Save the last iteration's scores before the first PV line is searched and
        // all the move scores except the (new) PV are set to -VALUE_INFINITE.
        for (size_t i = 0; i < rootMoves.size(); ++i)
        {
            rootMoves[i].previousScore      = rootMoves[i].score;
            rootMoves[i].previousPV         = rootMoves[i].pv;
            rootMoves[i].previousScoreExact = i < multiPV;
        }

        size_t pvFirst = 0;
        pvLast         = rootMoves.size();

        if (!threads.increaseDepth)
            searchAgainCounter++;

        // MultiPV loop. We perform a full root search for each PV line
        for (pvIdx = 0; pvIdx < multiPV; ++pvIdx)
        {
            // Reset UCI info selDepth for each depth and each PV line
            selDepth = 0;

            if (followPVOption)
                followPVLine = rootMoves[pvIdx].previousPV;

            // Reset aspiration window starting size
            delta     = 10 + (aspirationTweaks ? int(threadIdx % 8) : 0)
                  + std::abs(rootMoves[pvIdx].meanSquaredScore) / 41924;
            Value avg = rootMoves[pvIdx].averageScore;
            alpha     = std::max(avg - delta, -VALUE_INFINITE);
            beta      = std::min(avg + delta, VALUE_INFINITE);

            // Adjust optimism based on root move's averageScore
            optimism[us]  = 74 * avg / (std::abs(avg) + 79);
            optimism[~us] = -optimism[us];

            // Start with a small aspiration window and, in the case of a fail
            // high/low, re-search with a bigger window until we don't fail
            // high/low anymore.
            int failedHighCnt = 0;
            if (!pvIdx)
                failHighRecovery = std::max(0, failHighRecovery - 2);
            while (true)
            {
                // Adjust the effective depth searched, but ensure at least one
                // effective increment for every four searchAgain steps (see issue #2717).
                Depth adjustedDepth = std::max(1, rootDepth - failedHighCnt - failHighRecovery
                                                    - 3 * (searchAgainCounter + 1) / 3);
                rootDelta = beta - alpha;
                bestValue = search<Root>(rootPos, ss, alpha, beta, adjustedDepth, false);

                // Bring the best move to the front. It is critical that sorting
                // is done with a stable algorithm because all the values but the
                // first and eventually the new best one is set to -VALUE_INFINITE
                // and we want to keep the same order for all the moves except the
                // new PV that goes to the front. Note that in the case of MultiPV
                // search the already searched PV lines are preserved.
                std::stable_sort(rootMoves.begin() + pvIdx, rootMoves.begin() + pvLast);

                // If search has been stopped, we break immediately. Sorting is
                // safe because RootMoves is still valid, although it refers to
                // the previous iteration.
                if (threads.stop)
                    break;

                // When failing high/low give some update before a re-search. To avoid
                // excessive output that could hang GUIs, only start at nodes > 10M
                // (rather than depth N, which can be reached quickly)
                if (mainThread && multiPV == 1 && (bestValue <= alpha || bestValue >= beta)
                    && nodes > 10000000)
                    main_manager()->pv(*this, threads, tt, rootDepth);

                // In case of failing low/high increase aspiration window and re-search,
                // otherwise exit the loop.
                if (bestValue <= alpha)
                {
                    beta  = aspirationTweaks ? alpha : (alpha + beta) / 2;
                    alpha = std::max(bestValue - delta, -VALUE_INFINITE);

                    failedHighCnt = 0;
                    if (mainThread)
                        mainThread->stopOnPonderhit = false;
                }
                else if (bestValue >= beta)
                {
                    if (aspirationTweaks)
                        alpha = std::max(beta - delta, alpha);
                    beta = std::min(bestValue + delta, VALUE_INFINITE);
                    ++failedHighCnt;
                }
                else
                    break;

                delta += delta / 3;

                assert(alpha >= -VALUE_INFINITE && beta <= VALUE_INFINITE);
            }

            // After a search that needed a reduced depth to resolve fail highs,
            // keep the depth reduced in the next iterations and increase it back
            // gradually (UCI option FailHighRecovery)
            if (failHighRecoveryOption && failedHighCnt > 0 && !pvIdx)
                failHighRecovery = (failedHighCnt + 1) / 2 + 2;

            if (threads.stop && pvIdx)
            {
                // In MultiPV analysis we do not let aborted searches spoil mated-in
                // scores from a completed search in an earlier PV line. Hence we
                // guard against an aborted pvIdx line overtaking pvIdx - 1 when
                // pvIdx - 1 is a proven loss. Moreover, we do not trust an exact
                // loss score from an aborted search.
                if ((is_loss(rootMoves[pvIdx - 1].score) && rootMoves[pvIdx] < rootMoves[pvIdx - 1])
                    || rootMoves[pvIdx].score_is_exact_loss())
                {
                    // If previousScore is exact and worse than pvIdx - 1, we can
                    // safely use it. If it is equal, we make sure it cannot overtake
                    // pvIdx - 1.
                    if (rootMoves[pvIdx].previousScore != -VALUE_INFINITE
                        && rootMoves[pvIdx].previousScoreExact
                        && rootMoves[pvIdx].previousScore <= rootMoves[pvIdx - 1].score)
                    {
                        rootMoves[pvIdx].score = rootMoves[pvIdx].uciScore =
                          rootMoves[pvIdx].previousScore;
                        rootMoves[pvIdx].previousScore = -VALUE_INFINITE;
                        rootMoves[pvIdx].pv            = rootMoves[pvIdx].previousPV;
                        rootMoves[pvIdx].unset_bound_flags();
                    }

                    // Otherwise, if we can, we cap the score to the best possible,
                    // and mark the score as a bound (also a valid excuse for the
                    // incomplete PV).
                    else
                    {
                        if (is_loss(rootMoves[pvIdx - 1].score))
                        {
                            rootMoves[pvIdx].score = rootMoves[pvIdx].uciScore =
                              rootMoves[pvIdx - 1].score;
                            rootMoves[pvIdx].previousScore = -VALUE_INFINITE;
                            rootMoves[pvIdx].pv.resize(1);
                            rootMoves[pvIdx].scoreUpperbound = true;
                        }
                        else
                            rootMoves[pvIdx].scoreUpperbound = false;

                        rootMoves[pvIdx].scoreLowerbound = !rootMoves[pvIdx].scoreUpperbound;
                    }
                }

                // Finally, we mark all loss scores of partially searched moves as bounds
                for (size_t i = pvIdx + 1; i < multiPV; ++i)
                    if (rootMoves[i].score_is_exact_loss())
                        rootMoves[i].scoreLowerbound = true;
            }

            // Sort the PV lines searched so far and update the GUI. The PV of a
            // stopped search is sent by start_searching(), after the checks below
            // and the best thread selection.
            std::stable_sort(rootMoves.begin() + pvFirst, rootMoves.begin() + pvIdx + 1);

            if (mainThread && !threads.stop && (pvIdx + 1 == multiPV || nodes > 10000000))
            {
                main_manager()->pv(*this, threads, tt, rootDepth);
                uciPvSent = (pvIdx + 1 == multiPV);
            }

            if (threads.stop)
                break;
        }

        if (!threads.stop)
            completedDepth = rootDepth;

        // A mate found in an earlier iteration that this one did not find again
        const bool forgottenMate = lastBestMoveScore != -VALUE_INFINITE
                                && is_decisive(lastBestMoveScore)
                                && (std::abs(rootMoves[0].score) < std::abs(lastBestMoveScore)
                                    || rootMoves[0].score_is_bound());

        if (!threads.stop)
        {
            if (lastBestMovePV.empty() || lastBestMovePV[0] != rootMoves[0].pv[0])
                lastBestMoveDepth = rootDepth;

            // Do not replace (shorter) mate scores from a previous iteration
            if (!forgottenMate)
            {
                lastBestMovePV    = rootMoves[0].pv;
                lastBestMoveScore = rootMoves[0].score;
            }
        }

        const bool abortedLossSearch = threads.stop && !pvIdx && rootMoves[0].score_is_exact_loss();

        // An exact mated-in score from an aborted search cannot be trusted: the
        // loss could be delayed or refuted upon exploring the remaining root moves.
        // Thus here we roll back to the score from the previous iteration. We do
        // the same if a search has failed to recover a mate score that was found
        // in a previous iteration.
        if (abortedLossSearch || (rootMoves[0].score != -VALUE_INFINITE && forgottenMate))
        {
            // Bring the last best move to the front for best thread selection
            if (!lastBestMovePV.empty())
            {
                Utility::move_to_front(rootMoves, [&lastPV = std::as_const(lastBestMovePV)](
                                                    const auto& rm) { return rm == lastPV[0]; });
                rootMoves[0].score = rootMoves[0].uciScore = lastBestMoveScore;
                rootMoves[0].pv                            = lastBestMovePV;
                rootMoves[0].unset_bound_flags();

                if (mainThread)
                    uciPvSent = false;
            }
            // For an aborted depth 1 search we label the loss score as a lower bound
            else if (abortedLossSearch)
                rootMoves[0].scoreLowerbound = true;
        }

        // Have we found a "mate in x" after a completed iteration? Any thread
        // can stop the search.
        if (limits.mate && !threads.stop
            && ((is_win(rootMoves[0].score) && VALUE_MATE - rootMoves[0].score <= 2 * limits.mate)
                || (is_loss(rootMoves[0].score)
                    && VALUE_MATE + rootMoves[0].score <= 2 * limits.mate)))
            threads.stop = true;

        if (!mainThread)
            continue;

        // Use part of the gained time from a previous stable move for the current move
        for (auto&& th : threads)
        {
            totBestMoveChanges += th->worker->bestMoveChanges;
            th->worker->bestMoveChanges = 0;
        }

        // Do we have time for the next iteration? Can we stop searching now?
        if (limits.use_time_management() && !threads.stop && !mainThread->stopOnPonderhit)
        {
            uint64_t nodesEffort =
              rootMoves[0].effort * 100000 / std::max(size_t(1), size_t(nodes));

            double fallingEval =
              (16.198 + 2.638 * (mainThread->bestPreviousAverageScore - bestValue)
               + 0.780 * (mainThread->iterValue[iterIdx] - bestValue))
              / 100.0;
            fallingEval = std::clamp(fallingEval, 0.6378, 1.8394);

            // If the bestMove is stable over several iterations, reduce time accordingly
            double k      = 0.516;
            double center = lastBestMoveDepth + 15;
            timeReduction = 0.8 + 0.85 / (1.089 + std::exp(-k * (completedDepth - center)));
            double reduction =
              (2.0380 + mainThread->previousTimeReduction) / (2.5240 * timeReduction);
            double bestMoveInstability = 0.9381 + 1.6401 * totBestMoveChanges / threads.size();

            double totalTime =
              mainThread->tm.optimum() * fallingEval * reduction * bestMoveInstability;

            auto elapsedTime = elapsed();

            if (completedDepth >= 9 && nodesEffort >= 81808 && elapsedTime > totalTime * 0.6519
                && !mainThread->ponder)
                threads.stop = true;

            // Stop the search if we have exceeded the totalTime or maximum, or with
            // StopEarly if we know that there are no better moves in the analysed
            // line(s): a single legal move, a mate in 1 or 2 in all the lines, or
            // a mated in 1.
            if (elapsedTime > std::min(totalTime, double(mainThread->tm.maximum()))
                || (stopEarly
                    && (rootMoves.size() == 1 || rootMoves[multiPV - 1].score >= mate_in(3)
                        || rootMoves[0].score == mated_in(2))))
            {
                // If we are allowed to ponder do not stop the search now but
                // keep pondering until the GUI sends "ponderhit" or "stop".
                if (mainThread->ponder)
                    mainThread->stopOnPonderhit = true;
                else
                    threads.stop = true;
            }
            else
                threads.increaseDepth = mainThread->ponder || elapsedTime <= totalTime * 0.264;
        }

        mainThread->iterValue[iterIdx] = bestValue;
        iterIdx                        = (iterIdx + 1) & 3;
    }

    if (!mainThread)
        return false;

    mainThread->previousTimeReduction = timeReduction;

    return uciPvSent;
}


// Prefetches the continuation correction history entries that the child node
// will read in correction_value() (not for a move of a dark piece: the entries
// depend on the revealed piece).
void Search::Worker::prefetch_correction_histories(const Stack* ss,
                                                   const Position& pos,
                                                   Move            move) const {
    if (pos.move_dark(move))
        return;

    const Piece  pc = pos.moved_piece(move);
    const Square to = move.to_sq();

    prefetch(&(*(ss - 1)->continuationCorrectionHistory)[pc][to]);
    if (contCorr4Ply)
        prefetch(&(*(ss - 3)->continuationCorrectionHistory)[pc][to]);
    if (contCorr6Ply)
        prefetch(&(*(ss - 5)->continuationCorrectionHistory)[pc][to]);
}

void Search::Worker::do_move(Position& pos, const Move move, StateInfo& st) {
    do_move(pos, move, st, pos.gives_check(move));
}

void Search::Worker::do_move(Position& pos, const Move move, StateInfo& st, const bool givesCheck) {
    // Prefetch the TT entry of the next position as soon as the move is known
    // (a move of a dark piece is completed by the reveal: no prefetch)
    if (!pos.move_dark(move))
        prefetch(tt.first_entry(observer_tt_key(pos.prefetch_key(move), rootObserver)));

    DirtyPiece dp = pos.do_move(move, st, givesCheck);
    // Preferable over fetch_add to avoid locking instructions (only this
    // thread writes its node count)
    nodes.store(nodes.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    accumulatorStack.push(dp);
}

void Search::Worker::do_null_move(Position& pos, StateInfo& st) {
    pos.do_null_move(st);
    prefetch(tt.first_entry(observer_tt_key(pos.key(), rootObserver)));
}

void Search::Worker::undo_move(Position& pos, const Move move) {
    pos.undo_move(move);
    accumulatorStack.pop();
}

void Search::Worker::undo_null_move(Position& pos) { pos.undo_null_move(); }


// Reset histories, usually before a new game
void Search::Worker::clear() {
    mainHistory.fill(MainHistoryDefault);
    captureHistory.fill(-560);
    ttMoveHistory = 0;

    // Shared histories are cleared once, by the Worker that owns them
    if (!sharedHistories || ownsSharedHistories)
    {
        pawnHistory.fill(-1083);
        pawnCorrectionHistory.fill(6);
        minorPieceCorrectionHistory.fill(0);
        nonPawnCorrectionHistory.fill(0);

        for (auto& to : continuationCorrectionHistory)
            for (auto& h : to)
                h.fill(7);

        for (bool inCheck : {false, true})
            for (StatsType c : {NoCaptures, Captures})
                for (auto& to : continuationHistory[inCheck][c])
                    for (auto& h : to)
                        h.fill(-548);
    }

    refreshTable.clear(networks[numaAccessToken]);
}


// Main search function for both PV and non-PV nodes
template<NodeType nodeType>
Value Search::Worker::search(
  Position& pos, Stack* ss, Value alpha, Value beta, Depth depth, bool cutNode) {

    constexpr bool PvNode   = nodeType != NonPV;
    constexpr bool rootNode = nodeType == Root;
    const bool     allNode  = !(PvNode || cutNode);

    // Set again below if the value returned derives from a chance estimate
    ss->estimated = false;

    // UCI option SeekMate: when the root score is high enough (relative to the
    // root depth) to suggest a forced mate, prune and extend less to find it.
    // Do not tune these values; they are not intended for playing strength.
    const bool seekMate = seekMateOption && rootDepth > 0
                       && std::abs(rootMoves[pvIdx].score)
                            >= 750 + 220000 / (rootDepth * rootDepth);

    // Dive into quiescence search when the depth reaches zero
    if (depth <= 0)
    {
        constexpr auto nt = PvNode ? PV : NonPV;
        return qsearch<nt>(pos, ss, alpha, beta);
    }

    // Dive into the hidden capture chance node when the root side has just
    // captured a dark piece whose identity it will learn
    if (hiddenCaptureModel && (ss - 1)->capturedDark && ~pos.side_to_move() == rootObserver)
    {
        constexpr auto nt = PvNode ? PV : NonPV;
        return hidden_capture_search<nt>(pos, ss, alpha, beta, false, depth, cutNode);
    }

    // Dive into flip search when the last move is moving a dark pieces
    if ((ss - 1)->currentMove.is_ok() && pos.is_dark((ss - 1)->currentMove.to_sq()))
    {
        constexpr auto nt = PvNode ? PV : NonPV;
        return flip_search<nt>(pos, ss, alpha, beta, false, depth, cutNode);
    }

    // Limit the depth if extensions made it too large
    depth = std::min(depth, MAX_PLY - 1);

    assert(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
    assert(PvNode || (alpha == beta - 1));
    assert(0 < depth && depth < MAX_PLY);
    assert(!(PvNode && cutNode));

    Move      pv[MAX_PLY + 1];
    StateInfo st;

    Key   posKey;
    Move  move, excludedMove, bestMove;
    Depth extension, newDepth;
    Value bestValue, value, eval, maxValue, probCutBeta;
    bool  givesCheck, priorCapture;
    bool  improving = false, opponentWorsening = false;
    bool  capture, ttCapture;
    bool  bestEstimated = false;  // bestValue derives from a chance estimate
    int   priorReduction;
    Piece movedPiece;

    SearchedList capturesSearched;
    SearchedList quietsSearched;

    // Step 1. Initialize node
    Worker* thisThread = this;
    ss->inCheck        = bool(pos.checkers());
    priorCapture       = pos.captured_piece();
    Color us           = pos.side_to_move();
    ss->moveCount      = 0;
    bestValue          = -VALUE_INFINITE;
    maxValue           = VALUE_INFINITE;

    // Is this node on the PV of the previous iteration? (UCI option FollowPV.
    // The outcomes of a chance node all follow it if the dark move does.)
    ss->followPV = followPVOption
                && (rootNode
                    || ((ss - 1)->followPV && size_t(ss->ply - 1) < followPVLine.size()
                        && (ss - 1)->currentMove == followPVLine[ss->ply - 1]));

    // Check for the available remaining time
    if (is_mainthread())
        main_manager()->check_time(*thisThread);

    // Used to send selDepth info to GUI (selDepth counts from 1, ply from 0)
    if (PvNode && thisThread->selDepth < ss->ply + 1)
        thisThread->selDepth = ss->ply + 1;

    if (!rootNode)
    {
        // Step 2. Check for aborted search and repetition
        Value result = VALUE_NONE;
        if (pos.rule_judge(result, ss->ply))
            return result == VALUE_DRAW ? value_draw(thisThread->nodes) : result;
        if (result != VALUE_NONE)
        {
            assert(result != VALUE_DRAW);

            // 2 fold result is mate for us, the only chance for the opponent is to get a draw
            // We can guarantee to get at least a draw score during searching for that line
            if (result > VALUE_DRAW)
                alpha = std::max(alpha, VALUE_DRAW - 1);
            // 2 fold result is mated for us, the only chance for us is to get a draw
            // We can guarantee to get no more than a draw score during searching for that line
            else
                beta = std::min(beta, VALUE_DRAW + 1);
        }

        if (threads.stop.load(std::memory_order_relaxed) || ss->ply >= MAX_PLY)
            return (ss->ply >= MAX_PLY && !ss->inCheck) ? evaluate(pos)
                                                        : value_draw(thisThread->nodes);

        // Step 3. Mate distance pruning. Even if we mate at the next move our score
        // would be at best mate_in(ss->ply + 1), but if alpha is already bigger because
        // a shorter mate was found upward in the tree then there is no need to search
        // because we will never beat the current alpha. Same logic but with reversed
        // signs apply also in the opposite condition of being mated instead of giving
        // mate. In this case, return a fail-high score.
        alpha = std::max(mated_in(ss->ply), alpha);
        beta  = std::min(mate_in(ss->ply + 1), beta);
        if (alpha >= beta)
            return alpha;
    }

    assert(0 <= ss->ply && ss->ply < MAX_PLY);

    Square prevSq  = ((ss - 1)->currentMove).is_ok() ? ((ss - 1)->currentMove).to_sq() : SQ_NONE;
    bestMove       = Move::none();
    priorReduction = (ss - 1)->reduction;
    (ss - 1)->reduction = 0;
    ss->statScore       = 0;
    ss->isPvNode        = PvNode;
    (ss + 2)->cutoffCnt = 0;
    (ss + 1)->priorNMPFailHigh = 0;

    // Computed before the TT lookup, to overlap with the TT prefetch
    const auto correctionValue = correction_value(*thisThread, pos, ss);

    // Step 4. Transposition table lookup
    excludedMove                   = ss->excludedMove;
    posKey                         = observer_tt_key(pos.key(), rootObserver);
    auto [ttHit, ttData, ttWriter] = tt.probe(posKey);
    // Need further processing of the saved data
    ss->ttHit    = ttHit;
    ttData.move  = rootNode ? thisThread->rootMoves[thisThread->pvIdx].pv[0]
                 : ttHit    ? ttData.move
                            : Move::none();
    ttData.value = ttHit ? value_from_tt(ttData.value, ss->ply, pos.rule40_count()) : VALUE_NONE;
    ss->ttPv     = excludedMove ? ss->ttPv : PvNode || (ttHit && ttData.is_pv);
    ttCapture    = ttData.move && pos.capture(ttData.move);

    // Step 5. Static evaluation of the position (a lambda, because it is done
    // before the TT cutoff with HindsightFirst and after it otherwise)
    Value unadjustedStaticEval = VALUE_NONE;
    // (references, since a lambda cannot capture structured bindings in C++17)
    const TTData& ttEntry           = ttData;
    TTWriter&     ttEntryWriter     = ttWriter;
    auto          static_evaluation = [&]() {
        if (ss->inCheck)
            ss->staticEval = eval = (ss - 2)->staticEval;
        else if (excludedMove)
            unadjustedStaticEval = eval = ss->staticEval;
        else if (ss->ttHit)
        {
            // Never assume anything about values stored in TT
            unadjustedStaticEval = ttEntry.eval;
            if (!is_valid(unadjustedStaticEval))
                unadjustedStaticEval = evaluate(pos);

            ss->staticEval = eval = to_corrected_static_eval(unadjustedStaticEval, correctionValue);

            // ttValue can be used as a better position evaluation
            if (is_valid(ttEntry.value)
                && (ttEntry.bound & (ttEntry.value > eval ? BOUND_LOWER : BOUND_UPPER)))
                eval = ttEntry.value;
        }
        else
        {
            unadjustedStaticEval = evaluate(pos);
            ss->staticEval = eval = to_corrected_static_eval(unadjustedStaticEval, correctionValue);

            // Static evaluation is saved as it was before adjustment by correction history
            ttEntryWriter.write(posKey, VALUE_NONE, ss->ttPv, BOUND_NONE, DEPTH_UNSEARCHED,
                           Move::none(), unadjustedStaticEval, tt.generation());
        }
    };

    // Set up the improving flag, which is true if current static evaluation is
    // bigger than the previous static evaluation at our turn (if we were in
    // check at our previous move we go back until we weren't in check) and is
    // false otherwise. The improving flag is used in various pruning heuristics.
    // Then the hindsight adjustment of the depth, based on the reduction of the
    // previous move and on the static evaluation difference.
    auto hindsight_adjustments = [&](int minPriorReduction) {
        improving         = ss->staticEval > (ss - 2)->staticEval;
        opponentWorsening = ss->staticEval > -(ss - 1)->staticEval;

        if (priorReduction >= 3 && !opponentWorsening)
            depth++;
        if (priorReduction >= minPriorReduction && depth >= 2
            && ss->staticEval + (ss - 1)->staticEval > 181)
            depth--;
    };

    // With HindsightFirst, both are done before the TT cutoff, which then uses
    // the adjusted depth (also in check)
    if (hindsightFirst)
    {
        static_evaluation();
        hindsight_adjustments(2);
    }

    // At this point, if excluded, skip straight to step 5, static eval. However,
    // to save indentation, we list the condition in all code between here and there.

    // At non-PV nodes we check for an early TT cutoff. The validity of the TT
    // value is always checked, because of access races in probe().
    if (!PvNode && !excludedMove && is_valid(ttData.value)
        && ttData.depth > depth - (ttData.value <= beta))
    {
        // Case A: the TT entry can produce a cutoff
        if ((ttData.bound & (ttData.value >= beta ? BOUND_LOWER : BOUND_UPPER))
            && (cutNode == (ttData.value >= beta) || depth > 5))
        {
            // If ttMove is quiet, update move sorting heuristics on TT hit
            if (ttData.move && ttData.value >= beta)
            {
                // Bonus for a quiet ttMove that fails high
                if (!ttCapture)
                    update_quiet_histories(pos, ss, *this, ttData.move,
                                           simplifications ? 114 * depth
                                                           : std::min(98 * depth - 58, 1488));

                // Extra penalty for early quiet moves of the previous ply
                if (prevSq != SQ_NONE && (ss - 1)->moveCount <= 2 && !priorCapture)
                    update_continuation_histories(*this, ss - 1, (ss - 1)->movedPiece, prevSq,
                                                  -2387);
            }

            // Partial workaround for the graph history interaction problem
            // For high rule40 counts don't produce transposition table cutoffs.
            if (pos.rule40_count() < 70)
            {
                if (depth >= 8 && ttData.move && pos.pseudo_legal(ttData.move)
                    && pos.legal(ttData.move) && !is_decisive(ttData.value)
                    && !pos.move_dark(ttData.move))
                {
                    do_move(pos, ttData.move, st);
                    Key nextPosKey = observer_tt_key(pos.key(), rootObserver);
                    auto [ttHitNext, ttDataNext, ttWriterNext] = tt.probe(nextPosKey);
                    undo_move(pos, ttData.move);

                    // Check that the ttValue after the tt move would also trigger a cutoff
                    if (!is_valid(ttDataNext.value))
                        return ttData.value;
                    if ((ttData.value >= beta) == (-ttDataNext.value >= beta))
                        return ttData.value;
                }
                else
                    return ttData.value;
            }
        }
        // Case B: no cutoff, but the entry is deep enough, so its bound points
        // the wrong way: make it easier to replace. (Idea from AB-JChess, UCI
        // option TTPenalize.) Unlike upstream, the bound can also be BOUND_NONE
        // here (the estimate of a chance node, see search()), which is left alone.
        else if (ttPenalize && depth > 5 && ttData.bound != BOUND_NONE)
        {
            assert(!(ttData.bound & (ttData.value >= beta ? BOUND_LOWER : BOUND_UPPER)));

            ttWriter.penalize(1);
        }
    }

    if (ss->inCheck)
    {
        // Skip early pruning when in check
        if (!hindsightFirst)
        {
            static_evaluation();
            improving = false;
        }
        goto moves_loop;
    }

    if (!hindsightFirst)
        static_evaluation();

    // Use static evaluation difference to improve quiet move ordering
    if (((ss - 1)->currentMove).is_ok() && !(ss - 1)->inCheck && !priorCapture && !ttHit)
    {
        int bonus = std::clamp(-19 * int((ss - 1)->staticEval + ss->staticEval), -1271, 1725) + 379;
        thisThread->mainHistory[~us][((ss - 1)->currentMove).from_to()] << bonus * 1269 / 1024;
        if (type_of((ss - 1)->movedPiece) != PAWN)
            thisThread->pawnHistory[thisThread->pawn_history_index(pos)][(ss - 1)->movedPiece][prevSq]
              << bonus * 1267 / 1024;
    }

    if (!hindsightFirst)
        hindsight_adjustments(1);

    // Step 6. Razoring
    // If eval is really low, skip search entirely and return the qsearch value.
    // For PvNodes, we must have a guard against mates being returned.
    // (With Simplifications, the margin is linear in depth, as in Pikafish.)
    if ((pruningTweaks ? allNode : !PvNode)
        && eval < alpha
                    - (simplifications ? (pruningTweaks ? 511 : 720) * depth
                                       : 1130 + 252 * depth * depth))
        return qsearch<NonPV>(pos, ss, alpha, beta);

    // Step 7. Futility pruning: child node
    // The depth condition is important for mate finding.
    {
        auto futility_margin = [&](Depth d) {
            Value futilityMult = 101 - 37 * (cutNode && !ss->ttHit);

            return futilityMult * d                      //
                 - improving * futilityMult * 2          //
                 - opponentWorsening * futilityMult / 3  //
                 + (ss - 1)->statScore / 132             //
                 + std::abs(correctionValue) / 115091;
        };

        if (!ss->ttPv && depth < (seekMate ? 6 : 13) && eval - futility_margin(depth) >= beta
            && eval >= beta
            && (!ttData.move || ttCapture) && !is_loss(beta) && !is_win(eval))
            return beta + (eval - beta) / 3;
    }

    // Step 8. Null move search with verification search
    if (cutNode && (ss - 1)->currentMove != Move::null() && eval >= beta
        && ss->staticEval + (nmpFailHighCount ? 50 * ss->priorNMPFailHigh : 0)
             >= beta - 9 * depth - 50 * (pruningTweaks && improving) + 201
        && !excludedMove && pos.major_material(us) && ss->ply >= thisThread->nmpMinPly
        && (pruningTweaks ? beta >= -2000 : !is_loss(beta)))
    {
        assert(eval - beta >= 0);

        // Null move dynamic reduction based on depth (and on the static
        // evaluation with PruningTweaks)
        Depth R = 7 + depth / 3 + (pruningTweaks ? std::max((ss->staticEval - beta) / 256, 0) : 0);

        // When seeking a mate (UCI option SeekMate), reduce less
        R -= seekMate * R / 2;

        ss->currentMove                   = Move::null();
        ss->movedPiece                    = NO_PIECE;
        ss->continuationHistory           = &thisThread->continuationHistory[0][0][NO_PIECE][0];
        ss->continuationCorrectionHistory = &thisThread->continuationCorrectionHistory[NO_PIECE][0];

        ss->capturedDark = false;
        do_null_move(pos, st);

        Value nullValue = -search<NonPV>(pos, ss + 1, -beta, -beta + 1, depth - R, false);

        undo_null_move(pos);

        // Do not return unproven mate
        if (nullValue >= beta && !is_win(nullValue))
        {
            if (thisThread->nmpMinPly || depth < 16)
            {
                ++ss->priorNMPFailHigh;
                return nullValue;
            }

            assert(!thisThread->nmpMinPly);  // Recursive verification is not allowed

            // Do verification search at high depths, with null move pruning disabled
            // until ply exceeds nmpMinPly.
            thisThread->nmpMinPly = ss->ply + 3 * (depth - R) / 4;

            Value v = search<NonPV>(pos, ss, beta - 1, beta, depth - R, false);

            thisThread->nmpMinPly = 0;

            if (v >= beta)
            {
                ++ss->priorNMPFailHigh;
                return nullValue;
            }
        }
    }

    improving |= ss->staticEval >= beta + 108;

    // Step 9. Internal iterative reductions
    // For PV nodes without a ttMove as well as for deep enough cutNodes, we decrease depth.
    // (*Scaler) Especially if they make IIR less aggressive.
    if (!ss->followPV && !allNode && depth >= 5 && !ttData.move)
        depth--;

    // Step 10. ProbCut
    // If we have a good enough capture and a reduced search
    // returns a value much above beta, we can (almost) safely prune the previous move.
    probCutBeta = beta + 259 - 65 * improving;
    if (depth >= 3
        && !is_decisive(beta)
        // If value from transposition table is lower than probCutBeta, don't attempt
        // probCut there
        && !(is_valid(ttData.value) && ttData.value < probCutBeta))
    {
        assert(probCutBeta < VALUE_INFINITE && probCutBeta > beta);

        MovePicker mp(pos, ttData.move, probCutBeta - ss->staticEval, &thisThread->captureHistory);
        Depth      probCutDepth = std::max(depth - 5 - (pruningTweaks && improving), 0);

        while ((move = mp.next_move()) != Move::none())
        {
            assert(move.is_ok());

            if (move == excludedMove || !pos.legal(move))
                continue;

            assert(pos.capture(move));

            movedPiece = pos.moved_piece(move);

            ss->capturedDark = pos.is_dark(move.to_sq());

            // Captures of dark pieces are less predictable: optionally ask for a
            // larger margin and search them shallower. (Idea from AB-JChess,
            // UCI option ProbCutDarkCapture.)
            const bool  darkCut  = probCutDarkCapture && ss->capturedDark;
            const Value moveBeta = darkCut ? std::min(probCutBeta + 64, VALUE_INFINITE - 1) : probCutBeta;
            const Depth moveDepth = darkCut ? std::max(depth - 6, 0) : probCutDepth;

            do_move(pos, move, st);

            ss->currentMove = move;
            ss->movedPiece  = movedPiece;
            ss->continuationHistory =
              &this->continuationHistory[ss->inCheck][true][movedPiece][move.to_sq()];
            ss->continuationCorrectionHistory =
              &this->continuationCorrectionHistory[movedPiece][move.to_sq()];

            // Perform a preliminary qsearch to verify that the move holds
            value = -qsearch<NonPV>(pos, ss + 1, -moveBeta, -moveBeta + 1);

            // If the qsearch held, perform the regular search
            if (value >= moveBeta && moveDepth > 0)
                value = -search<NonPV>(pos, ss + 1, -moveBeta, -moveBeta + 1, moveDepth, !cutNode);

            undo_move(pos, move);

            if (value >= moveBeta)
            {
                // Save ProbCut data into transposition table
                ttWriter.write(posKey, value_to_tt(value, ss->ply), ss->ttPv, BOUND_LOWER,
                               moveDepth + 1, move, unadjustedStaticEval, tt.generation());

                if (!is_decisive(value))
                    return value - (moveBeta - beta);
            }
        }
    }

moves_loop:  // When in check, search starts here

    // Step 11. A small Probcut idea
    probCutBeta = beta + 430;
    if ((ttData.bound & BOUND_LOWER) && ttData.depth >= depth - 4 && ttData.value >= probCutBeta
        && !is_decisive(beta) && is_valid(ttData.value) && !is_decisive(ttData.value) && !seekMate)
        return probCutBeta;

    const PieceToHistory* contHist[] = {
      (ss - 1)->continuationHistory, (ss - 2)->continuationHistory, (ss - 3)->continuationHistory,
      (ss - 4)->continuationHistory, (ss - 5)->continuationHistory, (ss - 6)->continuationHistory};


    MovePicker mp(pos, ttData.move, depth, &thisThread->mainHistory, &thisThread->lowPlyHistory,
                  &thisThread->captureHistory, contHist, &thisThread->pawnHistory, ss->ply,
                  thisThread->pawn_history_index(pos));

    value = bestValue;

    int moveCount = 0;

    // Step 12. Loop through all pseudo-legal moves until no moves remain
    // or a beta cutoff occurs.
    while ((move = mp.next_move()) != Move::none())
    {
        assert(move.is_ok());

        if (move == excludedMove)
            continue;

        // Check for legality
        if (!pos.legal(move))
            continue;

        // At root obey the "searchmoves" option and skip moves not listed in Root Move List.
        // In MultiPV mode we also skip PV moves that have been already searched.
        if (rootNode
            && !std::count(thisThread->rootMoves.begin() + thisThread->pvIdx,
                           thisThread->rootMoves.begin() + thisThread->pvLast, move))
            continue;

        ss->moveCount = ++moveCount;

        if (rootNode && is_mainthread() && nodes > 10000000)
        {
            main_manager()->updates.onIter(
              {depth, UCIEngine::move(move), moveCount + thisThread->pvIdx});
        }
        if (PvNode)
            (ss + 1)->pv = nullptr;

        extension  = 0;
        capture    = pos.capture(move);
        movedPiece = pos.moved_piece(move);
        givesCheck = pos.gives_check(move);

        (ss + 1)->quietMoveStreak = (!capture && !givesCheck) ? (ss->quietMoveStreak + 1) : 0;

        // Calculate new depth for this move
        newDepth = depth - 1;

        int delta = beta - alpha;

        Depth r = reduction(improving, depth, moveCount, delta);

        // With Simplifications, the reduction does not depend on the window
        // (delta, see reduction()), but is smaller at PV nodes
        if (simplifications && PvNode)
            r -= 605;

        // Increase reduction for ttPv nodes (*Scaler)
        // Smaller or even negative value is better for short time controls
        // Bigger value is better for long time controls
        if (ss->ttPv)
            r += 910;

        // Step 13. Pruning at shallow depth.
        // Depth conditions are important for mate finding.
        if (!rootNode && pos.major_material(us) && !is_loss(bestValue))
        {
            // Skip quiet moves if movecount exceeds our FutilityMoveCount threshold
            if (moveCount >= (3 + depth * depth) / (2 - improving))
                mp.skip_quiet_moves();

            // Reduced depth of the next LMR search
            int lmrDepth = newDepth - r / 811;

            if (capture || givesCheck)
            {
                int captHist =
                  thisThread->captureHistory[movedPiece][move.to_sq()][pos.capture_type(move.to_sq())];

                // Futility pruning for captures
                if (!givesCheck && lmrDepth < 18 && !ss->inCheck)
                {
                    Value futilityValue = ss->staticEval + 307 + 365 * lmrDepth
                                        + pos.piece_value(move.to_sq()) + 105 * captHist / 491
                                        + 275 * (pruningTweaks && move.to_sq() == prevSq);
                    if (futilityValue <= alpha)
                        continue;
                }

                // SEE based pruning for captures and checks
                int seeHist = std::clamp(captHist / 30, -204 * depth, 186 * depth);
                if (!pos.see_ge(move, -225 * depth - seeHist))
                    continue;
            }
            else if (!ss->followPV || !PvNode)
            {
                int history =
                  (*contHist[0])[movedPiece][move.to_sq()]
                  + (*contHist[1])[movedPiece][move.to_sq()]
                  + thisThread->pawnHistory[thisThread->pawn_history_index(pos)][movedPiece][move.to_sq()];

                // Continuation history based pruning
                if (history < -3491 * depth)
                    continue;

                history += 60 * thisThread->mainHistory[us][move.from_to()] / 28;

                lmrDepth += history / (lmrHistoryDivisor ? lmr_divisor(depth) : 3626);

                Value baseFutility = (bestMove && !simplifications ? 48 : 275);
                Value futilityValue =
                  ss->staticEval + baseFutility + 126 * lmrDepth + 103 * (ss->staticEval > alpha);

                // Futility pruning: parent node
                // (*Scaler): Generally, more frequent futility pruning
                // scales well with respect to time and threads
                if (!ss->inCheck && lmrDepth < 8 && futilityValue <= alpha)
                {
                    if (bestValue <= futilityValue && !is_decisive(bestValue)
                        && !is_win(futilityValue))
                        bestValue = futilityValue;
                    continue;
                }

                lmrDepth = std::max(lmrDepth, 0);

                // Prune moves with negative SEE
                if (!pos.see_ge(move, -38 * lmrDepth * lmrDepth))
                    continue;
            }
        }

        // Step 14. Extensions
        // Singular extension search. If all moves but one
        // fail low on a search of (alpha-s, beta-s), and just one fails high on
        // (alpha, beta), then that move is singular and should be extended. To
        // verify this we do a reduced search on the position excluding the ttMove
        // and if the result is lower than ttValue minus a margin, then we will
        // extend the ttMove. Recursive singular search is avoided.

        // (*Scaler) Generally, higher singularBeta (i.e closer to ttValue)
        // and lower extension margins scale well.

        if (!rootNode && move == ttData.move && !excludedMove
            && depth >= 5 - (thisThread->completedDepth > 32) + ss->ttPv && is_valid(ttData.value)
            && !is_decisive(ttData.value) && (ttData.bound & BOUND_LOWER)
            && ttData.depth >= depth - 3 && !is_shuffling(move, ss, pos) && !seekMate)
        {
            Value singularBeta  = ttData.value - (48 + 72 * (ss->ttPv && !PvNode)) * depth / 59;
            Depth singularDepth = newDepth / 2;

            ss->excludedMove = move;
            value = search<NonPV>(pos, ss, singularBeta - 1, singularBeta, singularDepth, cutNode);
            ss->excludedMove = Move::none();

            if (value < singularBeta)
            {
                int corrValAdj   = std::abs(correctionValue) / 285633;
                int doubleMargin = -4 + 252 * PvNode - 152 * !ttCapture - corrValAdj
                                 - 1091 * ttMoveHistory / 144928
                                 - (ss->ply > thisThread->rootDepth) * 49;
                int tripleMargin = 85 + 303 * PvNode - 228 * !ttCapture + 88 * ss->ttPv - corrValAdj
                                 - (ss->ply * 2 > thisThread->rootDepth * 3) * 61;

                extension =
                  1 + (value < singularBeta - doubleMargin) + (value < singularBeta - tripleMargin);

                depth++;
            }

            // Multi-cut pruning
            // Our ttMove is assumed to fail high based on the bound of the TT entry,
            // and if after excluding the ttMove with a reduced search we fail high
            // over the original beta, we assume this expected cut-node is not
            // singular (multiple moves fail high), and we can prune the whole
            // subtree by returning a softbound.
            else if (value >= beta && !is_decisive(value))
            {
                if (multiCutHistory)
                {
                    // The ttMove was not singular: penalize it in ttMoveHistory,
                    // and use the multi-cut value to correct the static eval.
                    ttMoveHistory << -396 - 104 * depth;

                    if (!ss->inCheck && value > ss->staticEval)
                    {
                        const int bonus =
                          std::clamp(int(value - ss->staticEval) * 660 / 1024,
                                     -CORRECTION_HISTORY_LIMIT / 4, CORRECTION_HISTORY_LIMIT / 4);
                        update_correction_history(pos, ss, *this, bonus);
                    }
                }

                return value;
            }

            // Negative extensions
            // If other moves failed high over (ttValue - margin) without the
            // ttMove on a reduced search, but we cannot do multi-cut because
            // (ttValue - margin) is lower than the original beta, we do not know
            // if the ttMove is singular or can do a multi-cut, so we reduce the
            // ttMove in favor of other moves based on some conditions:

            // If the ttMove is assumed to fail high over current beta
            else if (ttData.value >= beta)
                extension = -3;

            // If we are on a cutNode but the ttMove is not assumed to fail high
            // over current beta
            else if (cutNode)
                extension = simplifications ? -3 : -2;
        }

        // Bonus for a safe reveal (a quiet move of a dark piece), computed on
        // the position before the move
        const int revealBonus = Reveal::move_bonus(pos, move, revealQuiet);

        // Step 15. Make the move
        uint64_t nodeCount = rootNode ? uint64_t(nodes) : 0;
        ss->capturedDark   = pos.is_dark(move.to_sq());
        prefetch_correction_histories(ss, pos, move);
        do_move(pos, move, st, givesCheck);

        // Add extension to new depth
        newDepth += extension;

        // Update the current move (this must be done after singular extension search)
        ss->currentMove = move;
        ss->movedPiece  = movedPiece;
        ss->continuationHistory =
          &thisThread->continuationHistory[ss->inCheck][capture][movedPiece][move.to_sq()];
        ss->continuationCorrectionHistory =
          &thisThread->continuationCorrectionHistory[movedPiece][move.to_sq()];

        // Decrease reduction for PvNodes (*Scaler)
        if (ss->ttPv)
            r -= 2082 + PvNode * 1139 + (ttData.value > alpha) * 915
               + (ttData.depth >= depth) * (1037 + cutNode * 1023);

        // These reduction adjustments have no proven non-linear scaling

        r += 328;  // Base reduction offset to compensate for other tweaks
        r -= moveCount * 61;
        r -= std::abs(correctionValue) / 32919;

        // Increase reduction for cut nodes
        if (cutNode)
            r += 3065 + 998 * !ttData.move;

        // Increase reduction if ttMove is a capture
        if (ttCapture)
            r += 1333 + (depth < 8) * 1130;

        // Increase reduction if next ply has a lot of fail high
        if (cutoffCntTweak)
        {
            if ((ss + 1)->cutoffCnt > 1)
                r += ((ss + 1)->cutoffCnt > 2 ? 1449 : 295) + allNode * 968;
        }
        else if ((ss + 1)->cutoffCnt > 2)
            r += 1449 + allNode * 968;

        r += (ss + 1)->quietMoveStreak * 49;

        // For first picked move (ttMove) reduce reduction
        if (move == ttData.move)
            r -= 2753;

        if (capture)
            ss->statScore =
              786 * int(pos.captured_value()) / 100
              + thisThread->captureHistory[movedPiece][move.to_sq()][pos.captured_type()]
              - 4998;
        else
            ss->statScore = 2 * thisThread->mainHistory[us][move.from_to()]
                          + (*contHist[0])[movedPiece][move.to_sq()]
                          + (*contHist[1])[movedPiece][move.to_sq()] - 3804;

        // Decrease/increase reduction for moves with a good/bad history
        r -= ss->statScore * 955 / 9646;

        if (lmrTweaks)
        {
            // Reduce quiet moves less when the static eval is well above alpha
            // (loose alpha window), more when it is below
            if (!capture && !is_decisive(alpha))
                r += 3 * std::clamp(alpha - eval, -64, 96);

            // Scale up reductions for expected ALL nodes
            if (allNode)
                r += r / (depth + 1);
        }

        // When seeking a mate (UCI option SeekMate), reduce non-checking moves
        // more and checking moves less
        if (seekMate && r > 0)
            r += (givesCheck ? -r : r) / 4;

        // Step 16. Late moves reduction / extension (LMR)
        if (depth >= 2 && moveCount > 1)
        {
            // In general we want to cap the LMR depth search at newDepth, but when
            // reduction is negative, we allow this move a limited search extension
            // beyond the first move depth. To avoid search explosion, extensions
            // are not allowed deep, relative to rootDepth, in the search tree.
            const bool notTooDeep   = ss->ply < 2 * thisThread->rootDepth;
            const int  maxExtension = notTooDeep ? !allNode + (PvNode && !bestMove) : 0;
            Depth d = std::max(1, newDepth + std::min(-r / 1090, maxExtension))
                    + (ss - 1)->isPvNode;

            ss->reduction = newDepth - d;
            value         = search_child<NonPV>(pos, ss + 1, alpha, alpha + 1, d, true, revealBonus);
            ss->reduction = 0;

            // Do a full-depth search when reduced LMR search fails high
            // (*Scaler) Usually doing more shallower searches
            // doesn't scale well to longer TCs
            if (value > alpha && d < newDepth)
            {
                // Adjust full-depth search based on LMR results - if the result was
                // good enough search deeper, if it was bad enough search shallower.
                // Searching deeper gives the child the depth of this node: like the
                // extensions above, not deep in the tree, otherwise chains of such
                // re-searches (one per ply, frequent with the noisy values of the
                // chance nodes) do not reduce the depth at all.
                const bool doDeeperSearch =
                  notTooDeep && value > (bestValue + 52 + 2 * newDepth);
                const bool doShallowerSearch = value < bestValue + 9;

                newDepth += doDeeperSearch - doShallowerSearch;

                if (newDepth > d)
                    value = search_child<NonPV>(pos, ss + 1, alpha, alpha + 1, newDepth, !cutNode,
                                                revealBonus);

                // Post LMR continuation history updates
                update_continuation_histories(*this, ss, movedPiece, move.to_sq(), 1533);
            }
            else if (value > alpha && value < bestValue + 9)
                newDepth--;
        }

        // Step 17. Full-depth search when LMR is skipped
        else if (!PvNode || moveCount > 1)
        {
            // Increase reduction if ttMove is not present
            if (!ttData.move)
                r += 875;

            r -= ttMoveHistory / 8;

            // Note that if expected reduction is high, we reduce search depth here
            value = search_child<NonPV>(pos, ss + 1, alpha, alpha + 1,
                                        newDepth - (r > 4170) - (r > 5563 && newDepth > 2),
                                        !cutNode, revealBonus);
        }

        // For PV nodes only, do a full PV search on the first move or after a fail high,
        // otherwise let the parent node fail low with value <= alpha and try another move.
        if (PvNode && (moveCount == 1 || value > alpha))
        {
            (ss + 1)->pv    = pv;
            (ss + 1)->pv[0] = Move::none();

            // Extend move from transposition table if we are about to dive into qsearch.
            // With MateTTExtension, always for a decisive TT value (qsearch would
            // ignore it), otherwise only for a deep enough TT entry.
            if (move == ttData.move
                && (mateTTExtension
                      ? (is_valid(ttData.value) && is_decisive(ttData.value) && ttData.depth > 0)
                          || (ttData.depth > 1 && thisThread->rootDepth > 8)
                      : thisThread->rootDepth > 8))
                newDepth = std::max(newDepth, 1);

            value = search_child<PV>(pos, ss + 1, alpha, beta, newDepth, false, revealBonus);
        }

        // Step 18. Undo move
        undo_move(pos, move);

        assert(value > -VALUE_INFINITE && value < VALUE_INFINITE);

        // Step 19. Check for a new best move
        // Finished searching the move. If a stop occurred, the return value of
        // the search cannot be trusted, and we return immediately without updating
        // best move, principal variation nor transposition table.
        if (threads.stop.load(std::memory_order_relaxed))
            return VALUE_ZERO;

        if (rootNode)
        {
            RootMove& rm =
              *std::find(thisThread->rootMoves.begin(), thisThread->rootMoves.end(), move);

            const uint64_t moveNodes = nodes - nodeCount;
            rm.effort += moveNodes;

            if (rootScoreEMA)
            {
                // Weight of the new score (of 32): larger when the new search took
                // more nodes compared with the earlier ones, from 37.5% to 75% (50%
                // at most for the mean squared score).
                const uint64_t prevEffort = std::max(uint64_t(1), rm.effort - moveNodes);
                const int64_t  w          = int64_t(
                  std::clamp(32 * moveNodes * 2 / (moveNodes * 2 + 3 * prevEffort), uint64_t(12),
                                       uint64_t(24)));
                const int64_t wSquared = std::min(w, int64_t(16));

                rm.averageScore =
                  rm.averageScore != -VALUE_INFINITE
                    ? Value((int64_t(value) * w + int64_t(rm.averageScore) * (32 - w)) / 32)
                    : value;

                rm.meanSquaredScore =
                  rm.meanSquaredScore != -VALUE_INFINITE * VALUE_INFINITE
                    ? Value((int64_t(value) * std::abs(value) * wSquared
                             + int64_t(rm.meanSquaredScore) * (32 - wSquared))
                            / 32)
                    : value * std::abs(value);
            }
            else
            {
                rm.averageScore =
                  rm.averageScore != -VALUE_INFINITE ? (value + rm.averageScore) / 2 : value;

                rm.meanSquaredScore = rm.meanSquaredScore != -VALUE_INFINITE * VALUE_INFINITE
                                        ? (value * std::abs(value) + rm.meanSquaredScore) / 2
                                        : value * std::abs(value);
            }

            // PV move or new best move?
            if (moveCount == 1 || value > alpha)
            {
                rm.score = rm.uciScore = value;
                rm.selDepth            = thisThread->selDepth;
                rm.scoreLowerbound = rm.scoreUpperbound = false;

                if (value >= beta)
                {
                    rm.scoreLowerbound = true;
                    rm.uciScore        = beta;
                }
                else if (value <= alpha)
                {
                    rm.scoreUpperbound = true;
                    rm.uciScore        = alpha;
                }

                rm.pv.resize(1);

                assert((ss + 1)->pv);

                for (Move* m = (ss + 1)->pv; *m != Move::none(); ++m)
                    rm.pv.push_back(*m);

                // We record how often the best move has been changed in each iteration.
                // This information is used for time management. In MultiPV mode,
                // we must take care to only do this for the first PV line.
                if (moveCount > 1 && !thisThread->pvIdx)
                    ++thisThread->bestMoveChanges;
            }
            else
                // All other moves but the PV, are set to the lowest value: this
                // is not a problem when sorting because the sort is stable and the
                // move position in the list is preserved - just the PV is pushed up.
                rm.score = -VALUE_INFINITE;
        }

        // In case we have an alternative move equal in eval to the current bestmove,
        // promote it to bestmove by pretending it just exceeds alpha (but not beta).
        int inc = (value == bestValue && ss->ply + 2 >= thisThread->rootDepth
                   && (int(nodes) & 15) == 0 && !is_win(std::abs(value) + 1));

        if (value + inc > bestValue)
        {
            bestValue     = value;
            bestEstimated = (ss + 1)->estimated;

            if (value + inc > alpha)
            {
                bestMove = move;

                if (PvNode && !rootNode)  // Update pv even in fail-high case
                    update_pv(ss->pv, move, (ss + 1)->pv);

                if (value >= beta)
                {
                    // (* Scaler) Especially if they make cutoffCnt increment more often.
                    ss->cutoffCnt += (extension < 2) || PvNode;
                    assert(value >= beta);  // Fail high
                    break;
                }

                // Reduce other moves if we have found at least one score improvement
                if (depth > 2 && depth < 11 && !is_decisive(value))
                    depth -= 2;

                assert(depth > 0);
                alpha = value;  // Update alpha! Always alpha < beta
            }
        }

        // If the move is worse than some previously searched move,
        // remember it, to update its stats later.
        if (move != bestMove && moveCount <= SEARCHEDLIST_CAPACITY)
        {
            if (capture)
                capturesSearched.push_back(move);
            else
                quietsSearched.push_back(move);
        }
    }

    // Step 20. Check for mate
    // All legal moves have been searched and if there are no legal moves,
    // it must be a mate. If we are in a singular extension search then
    // return a fail low score.

    assert(moveCount || !ss->inCheck || excludedMove || !MoveList<LEGAL>(pos).size());

    // Adjust best value for fail high cases
    if (bestValue >= beta && !is_decisive(bestValue) && !is_decisive(alpha))
        bestValue = (bestValue * depth + beta) / (depth + 1);

    if (!moveCount)
        bestValue = excludedMove ? alpha : mated_in(ss->ply);

    // If there is a move that produces search value greater than alpha,
    // we update the stats of searched moves.
    else if (bestMove)
    {
        update_all_stats(pos, ss, *this, bestMove, prevSq, quietsSearched, capturesSearched, depth,
                         ttData.move, moveCount, PvNode);
        if (!PvNode)
            ttMoveHistory << (bestMove == ttData.move ? 774 : -844);
    }

    // Bonus for prior quiet countermove that caused the fail low
    else if (!priorCapture && prevSq != SQ_NONE)
    {
        int bonusScale = -251;
        bonusScale += std::min(-(ss - 1)->statScore / 76, 256);
        bonusScale += std::min(54 * depth, 549);
        bonusScale += 138 * ((ss - 1)->moveCount > 12);
        bonusScale += 70 * (!ss->inCheck && bestValue <= ss->staticEval - 184);
        bonusScale += 154 * (!(ss - 1)->inCheck && bestValue <= -(ss - 1)->staticEval - 132);

        bonusScale = std::max(bonusScale, 0);

        // scaledBonus ranges from 0 to roughly 2.1M, so the products below
        // are close to the int range: compute the largest one in 64 bits.
        const int scaledBonus = std::min(153 * depth - 93, 2290) * bonusScale;

        update_continuation_histories(*thisThread, ss - 1, (ss - 1)->movedPiece, prevSq,
                                      scaledBonus * 384 / 32768);

        thisThread->mainHistory[~us][((ss - 1)->currentMove).from_to()]
          << scaledBonus * 221 / 32768;

        if (type_of((ss - 1)->movedPiece) != PAWN)
            thisThread->pawnHistory[thisThread->pawn_history_index(pos)][(ss - 1)->movedPiece][prevSq]
              << int(std::int64_t(scaledBonus) * 1001 / 32768);
    }

    // Bonus for prior capture countermove that caused the fail low
    else if (prevSq != SQ_NONE)
    {
        assert(priorCapture);
        assert(pos.captured_piece() != NO_PIECE);
        thisThread->captureHistory[(ss - 1)->movedPiece][prevSq][pos.captured_type()] << 958;
    }

    if (PvNode)
        bestValue = std::min(bestValue, maxValue);

    // If no good move is found and the previous position was ttPv, then the previous
    // opponent move is probably good and the new position is added to the search tree.
    if (bestValue <= alpha)
        ss->ttPv = ss->ttPv || (ss - 1)->ttPv;

    // An exact value (PV node, within the window) that derives from the
    // estimate of a chance node is not proven: report it to the parent, and
    // do not let it produce TT cutoffs (its move is still worth storing).
    ss->estimated = PvNode && bestMove && bestValue < beta && bestEstimated;

    // Write gathered information in transposition table. Note that the
    // static evaluation is saved as it was before correction history.
    if (!excludedMove && !(rootNode && thisThread->pvIdx))
        ttWriter.write(posKey, value_to_tt(bestValue, ss->ply), ss->ttPv,
                       bestValue >= beta    ? BOUND_LOWER
                       : ss->estimated      ? BOUND_NONE
                       : PvNode && bestMove ? BOUND_EXACT
                                            : BOUND_UPPER,
                       moveCount != 0 ? depth : std::min(MAX_PLY - 1, depth + 6), bestMove,
                       unadjustedStaticEval, tt.generation());

    // Adjust correction history
    if (!ss->inCheck && !(bestMove && pos.capture(bestMove))
        && ((bestValue < ss->staticEval && bestValue < beta)  // negative correction & no fail high
            || (bestValue > ss->staticEval && bestMove)))     // positive correction & no fail low
    {
        // With CorrHistAsym, positive corrections are smaller
        const int divisor = 8 + (corrHistAsym ? 2 * (bestValue > ss->staticEval) : 0);
        auto      bonus   = std::clamp(int(bestValue - ss->staticEval) * depth / divisor,
                                       -CORRECTION_HISTORY_LIMIT / 4, CORRECTION_HISTORY_LIMIT / 4);
        update_correction_history(pos, ss, *thisThread, bonus);
    }

    assert(bestValue > -VALUE_INFINITE && bestValue < VALUE_INFINITE);

    return bestValue;
}


// Quiescence search function, which is called by the main search function with
// depth zero, or recursively with further decreasing depth. With depth <= 0, we
// "should" be using static eval only, but tactical moves may confuse the static eval.
// To fight this horizon effect, we implement this qsearch of tactical moves.
// See https://www.chessprogramming.org/Horizon_Effect
// and https://www.chessprogramming.org/Quiescence_Search
template<NodeType nodeType>
Value Search::Worker::qsearch(Position& pos, Stack* ss, Value alpha, Value beta) {

    static_assert(nodeType != Root);
    constexpr bool PvNode = nodeType == PV;

    // Set again below if the value returned derives from a chance estimate
    ss->estimated = false;

    // Dive into the hidden capture chance node (see search())
    if (hiddenCaptureModel && (ss - 1)->capturedDark && ~pos.side_to_move() == rootObserver)
    {
        constexpr auto nt = PvNode ? PV : NonPV;
        return hidden_capture_search<nt>(pos, ss, alpha, beta, true, 0, false);
    }

    // Dive into flip search when the last move is moving a dark pieces
    if ((ss - 1)->currentMove.is_ok() && pos.is_dark((ss - 1)->currentMove.to_sq()))
    {
        constexpr auto nt = PvNode ? PV : NonPV;
        return flip_search<nt>(pos, ss, alpha, beta);
    }

    assert(alpha >= -VALUE_INFINITE && alpha < beta && beta <= VALUE_INFINITE);
    assert(PvNode || (alpha == beta - 1));

    Move      pv[MAX_PLY + 1];
    StateInfo st;

    Key   posKey;
    Move  move, bestMove;
    Value bestValue, value, futilityBase;
    bool  pvHit, givesCheck, capture;
    bool  bestEstimated = false;  // bestValue derives from a chance estimate
    int   moveCount;

    // Step 1. Initialize node
    if (PvNode)
    {
        (ss + 1)->pv = pv;
        ss->pv[0]    = Move::none();
    }

    Worker* thisThread = this;
    bestMove           = Move::none();
    ss->inCheck        = bool(pos.checkers());
    moveCount          = 0;

    // Used to send selDepth info to GUI (selDepth counts from 1, ply from 0)
    if (PvNode && thisThread->selDepth < ss->ply + 1)
        thisThread->selDepth = ss->ply + 1;

    // Step 2. Check for repetition or maximum ply reached
    Value result = VALUE_NONE;
    if (pos.rule_judge(result, ss->ply))
        return result;
    if (result != VALUE_NONE)
    {
        assert(result != VALUE_DRAW);

        // 2 fold result is mate for us, the only chance for the opponent is to get a draw
        // We can guarantee to get at least a draw score during searching for that line
        if (result > VALUE_DRAW)
            alpha = std::max(alpha, VALUE_DRAW);
        // 2 fold result is mated for us, the only chance for us is to get a draw
        // We can guarantee to get no more than a draw score during searching for that line
        else
            beta = std::min(beta, VALUE_DRAW);

        if (alpha >= beta)
            return alpha;
    }

    if (ss->ply >= MAX_PLY)
        return !ss->inCheck ? evaluate(pos) : VALUE_DRAW;

    assert(0 <= ss->ply && ss->ply < MAX_PLY);

    // Step 3. Transposition table lookup
    posKey                         = observer_tt_key(pos.key(), rootObserver);
    auto [ttHit, ttData, ttWriter] = tt.probe(posKey);
    // Need further processing of the saved data
    ss->ttHit    = ttHit;
    ttData.move  = ttHit ? ttData.move : Move::none();
    ttData.value = ttHit ? value_from_tt(ttData.value, ss->ply, pos.rule40_count()) : VALUE_NONE;
    pvHit        = ttHit && ttData.is_pv;

    // At non-PV nodes we check for an early TT cutoff
    if (!PvNode && ttData.depth >= DEPTH_QS
        && is_valid(ttData.value)  // Can happen when !ttHit or when access race in probe()
        && (ttData.bound & (ttData.value >= beta ? BOUND_LOWER : BOUND_UPPER)))
        return ttData.value;

    // Step 4. Static evaluation of the position
    Value unadjustedStaticEval = VALUE_NONE;
    if (ss->inCheck)
        bestValue = futilityBase = -VALUE_INFINITE;
    else
    {
        const auto correctionValue = correction_value(*thisThread, pos, ss);

        if (ss->ttHit)
        {
            // Never assume anything about values stored in TT
            unadjustedStaticEval = ttData.eval;
            if (!is_valid(unadjustedStaticEval))
                unadjustedStaticEval = evaluate(pos);
            ss->staticEval = bestValue =
              to_corrected_static_eval(unadjustedStaticEval, correctionValue);

            // ttValue can be used as a better position evaluation
            if (is_valid(ttData.value) && !is_decisive(ttData.value)
                && (ttData.bound & (ttData.value > bestValue ? BOUND_LOWER : BOUND_UPPER)))
                bestValue = ttData.value;
        }
        else
        {
            unadjustedStaticEval = evaluate(pos);

            ss->staticEval = bestValue =
              to_corrected_static_eval(unadjustedStaticEval, correctionValue);
        }

        // Stand pat. Return immediately if static value is at least beta
        if (bestValue >= beta)
        {
            // bestValue is the clamped static evaluation or a TT value that
            // is not decisive
            assert(!is_decisive(bestValue));
            bestValue = (bestValue + beta) / 2;

            if (!ss->ttHit)
                ttWriter.write(posKey, standPatNoTT ? VALUE_NONE : value_to_tt(bestValue, ss->ply),
                               false, BOUND_LOWER, DEPTH_UNSEARCHED, Move::none(),
                               unadjustedStaticEval, tt.generation());
            return bestValue;
        }

        if (bestValue > alpha)
            alpha = bestValue;

        futilityBase = ss->staticEval + 159;
    }

    const PieceToHistory* contHist[] = {(ss - 1)->continuationHistory,
                                        (ss - 2)->continuationHistory};

    Square prevSq = ((ss - 1)->currentMove).is_ok() ? ((ss - 1)->currentMove).to_sq() : SQ_NONE;

    // Initialize a MovePicker object for the current position, and prepare to search
    // the moves. We presently use two stages of move generator in quiescence search:
    // captures, or evasions only when in check.
    MovePicker mp(pos, ttData.move, DEPTH_QS, &thisThread->mainHistory, &thisThread->lowPlyHistory,
                  &thisThread->captureHistory, contHist, &thisThread->pawnHistory, ss->ply,
                  thisThread->pawn_history_index(pos));

    // Step 5. Loop through all pseudo-legal moves until no moves remain or a beta
    // cutoff occurs.
    while ((move = mp.next_move()) != Move::none())
    {
        assert(move.is_ok());

        if (!pos.legal(move))
            continue;

        givesCheck = pos.gives_check(move);
        capture    = pos.capture(move);

        moveCount++;

        // Step 6. Pruning
        if (!is_loss(bestValue))
        {
            // Futility pruning and moveCount pruning
            if (!givesCheck && move.to_sq() != prevSq && !is_loss(futilityBase))
            {
                if (moveCount > 2)
                    continue;

                Value futilityValue = futilityBase + pos.piece_value(move.to_sq());

                // If static eval + value of piece we are going to capture is
                // much lower than alpha, we can prune this move.
                if (futilityValue <= alpha)
                {
                    bestValue = std::max(bestValue, futilityValue);
                    continue;
                }

                // If static exchange evaluation is low enough
                // we can prune this move.
                if (!pos.see_ge(move, alpha - futilityBase))
                {
                    bestValue = std::max(bestValue, std::min(alpha, futilityBase));
                    continue;
                }
            }

            // Continuation history based pruning
            if (!capture
                && (*contHist[0])[pos.moved_piece(move)][move.to_sq()]
                       + thisThread->pawnHistory[thisThread->pawn_history_index(pos)][pos.moved_piece(move)]
                                                [move.to_sq()]
                     <= 2861)
                continue;

            // Do not search moves with bad enough SEE values. A move that
            // reveals or captures a dark piece is a chance node of up to 6
            // (36 for both) outcomes: search it only if it wins material
            // on average, otherwise even exchanges of dark pieces make the
            // quiescence search explode.
            if (!pos.see_ge(move, pos.move_dark(move) || pos.is_dark(move.to_sq()) ? 1 : -114))
                continue;
        }

        // Step 7. Make and search the move
        Piece movedPiece = pos.moved_piece(move);

        ss->capturedDark = pos.is_dark(move.to_sq());
        prefetch_correction_histories(ss, pos, move);
        do_move(pos, move, st, givesCheck);

        // Update the current move
        ss->currentMove = move;
        ss->movedPiece  = movedPiece;
        ss->continuationHistory =
          &thisThread->continuationHistory[ss->inCheck][capture][movedPiece][move.to_sq()];
        ss->continuationCorrectionHistory =
          &thisThread->continuationCorrectionHistory[movedPiece][move.to_sq()];

        value = -qsearch<nodeType>(pos, ss + 1, -beta, -alpha);
        undo_move(pos, move);

        assert(value > -VALUE_INFINITE && value < VALUE_INFINITE);

        // Step 8. Check for a new best move
        if (value > bestValue)
        {
            bestValue     = value;
            bestEstimated = (ss + 1)->estimated;

            if (value > alpha)
            {
                bestMove = move;

                if (PvNode)  // Update pv even in fail-high case
                    update_pv(ss->pv, move, (ss + 1)->pv);

                if (value < beta)  // Update alpha here!
                    alpha = value;
                else
                    break;  // Fail high
            }
        }
    }

    // Step 9. Check for mate
    // All legal moves have been searched. A special case: if no legal
    // moves were found, it is checkmate.
    if (bestValue == -VALUE_INFINITE || (!moveCount && [&] {
            for (const auto& m : MoveList<QUIETS>(pos))
                if (pos.legal(m))
                    return false;
            return true;
        }()))
    {
        assert(!MoveList<LEGAL>(pos).size());
        return mated_in(ss->ply);  // Plies to mate from the root
    }

    if (!is_decisive(bestValue) && bestValue > beta)
        bestValue = (bestValue + beta) / 2;

    // Save gathered info in transposition table. The static evaluation
    // is saved as it was before adjustment by correction history.
    // A value within the window that derives from a chance estimate is not a
    // proven bound (see search())
    ss->estimated = PvNode && bestMove && bestValue < beta && bestEstimated;

    ttWriter.write(posKey, value_to_tt(bestValue, ss->ply), pvHit,
                   bestValue >= beta ? BOUND_LOWER
                   : ss->estimated   ? BOUND_NONE
                                     : BOUND_UPPER,
                   DEPTH_QS, bestMove,
                   unadjustedStaticEval, tt.generation());

    assert(bestValue > -VALUE_INFINITE && bestValue < VALUE_INFINITE);

    return bestValue;
}


// Averages the outcomes of a chance node. A chance node is only decisive when
// *every* outcome is decisive for the same side; otherwise the outcomes are
// averaged in win-rate space. With ChanceRiskAversion, the standard deviation
// of the win rates is added (from the point of view of the side to move here,
// i.e. subtracted for the side whose move caused the chance event): positive
// values make that side avoid uncertain reveals/captures, negative values
// make it seek them. (Risk-sensitive aggregation idea from AB-JChess.) The
// average always lies between the smallest and the largest outcome.
//
// With kind == Lower (resp. Upper), 'values' are lower (resp. upper) bounds of
// the outcomes, and the result is a lower (resp. upper) bound of the value
// the exact outcomes would give. This relies on the aggregate being
// non-decreasing in every outcome (with ChanceRiskAversion, the standard
// deviation, at most 0.5, is bounded separately).
Value Search::Worker::aggregate_chance(
  const Value* values, const int* counts, int n, int total, ChanceBound kind) const {

    assert(n > 0 && total > 0);

    bool  allWin = true, allLoss = true;
    Value vmin = values[0], vmax = values[0];
    for (int i = 0; i < n; ++i)
    {
        allWin &= is_win(values[i]);
        allLoss &= is_loss(values[i]);
        vmin = std::min(vmin, values[i]);
        vmax = std::max(vmax, values[i]);
    }

    // All wins: the guaranteed mate is the slowest one (minimum value).
    // All losses: we are mated at the latest by the slowest one (maximum value).
    // For bounds, the other rule gives a bound as well: if all lower bounds are
    // losses, the true value is at least the largest of them (whether or not
    // the true outcomes are all losses), and symmetrically for upper bounds.
    if (allWin)
        return vmin;
    if (allLoss)
        return vmax;

    double winrateSum = 0.0, winrateSqSum = 0.0;
    for (int i = 0; i < n; ++i)
    {
        const double w = flip_score_to_winrate(values[i], chanceScaling);
        winrateSum += w * counts[i];
        winrateSqSum += w * w * counts[i];
    }

    double expected = winrateSum / total;

    if (kind == ChanceBound::Exact)
    {
        if (chanceRisk != 0.0)
        {
            const double variance = std::max(0.0, winrateSqSum / total - expected * expected);
            expected += chanceRisk * std::sqrt(variance);
        }
        return std::clamp(flip_winrate_to_score(expected, chanceScaling), vmin, vmax);
    }

    if (kind == ChanceBound::Lower)
        return std::max(vmin, flip_winrate_to_score(expected + std::min(chanceRisk, 0.0) * 0.5,
                                                    chanceScaling, Rounding::Down));

    return std::min(vmax, flip_winrate_to_score(expected + std::max(chanceRisk, 0.0) * 0.5,
                                                chanceScaling, Rounding::Up));
}


// Searches the outcomes of a chance node and returns their aggregate.
// searchOutcome(i, alpha, beta) searches outcome i, from the point of view of
// the side to move at the chance node. The outcomes are sorted from the most
// to the least likely.
//
// Every outcome is searched with the window of the node, so a value returned
// by an outcome is only a lower bound if it failed high, an upper bound if it
// failed low. The proven bounds of every outcome are kept, and the node fails
// high (low) as soon as the aggregate of the lower (upper) bounds, where an
// outcome not searched yet counts as a loss (win), proves it (Star1, Ballard
// 1983). This also skips the remaining outcomes when the first ones are
// conclusive. When some outcomes failed high and others low, the aggregate of
// the values returned is only an estimate: it is kept within the proven
// bounds. (Searching such outcomes again until the result is exact multiplies
// the size of the tree by 20 to 35 at depths 10-12, even in PV nodes only.)
// Such an estimate is flagged in ss->estimated, so that the callers do not
// store it, or a value derived from it, as BOUND_EXACT. Every outcome is
// searched at ss: the flag of an outcome is read right after its search.
template<typename SearchOutcome>
Value Search::Worker::chance_node(Stack*          ss,
                                  const int*      counts,
                                  int             n,
                                  int             total,
                                  Value           alpha,
                                  Value           beta,
                                  SearchOutcome&& searchOutcome) {

    assert(n > 0 && n <= Position::MaxRestPieceTypes && total > 0);

    // A single outcome: its value, and its flag, are those of the node
    if (n == 1)
        return searchOutcome(0, alpha, beta);

    std::array<Value, Position::MaxRestPieceTypes> lo, hi, values;
    lo.fill(-VALUE_MATE);
    hi.fill(VALUE_MATE);

    // Some outcome returned, within the window, a value that is an estimate
    bool outcomeEstimated = false;

    for (int i = 0; i < n; ++i)
    {
        const Value v = values[i] = searchOutcome(i, alpha, beta);

        // The search was stopped: the remaining outcomes would only return
        // meaningless values, and the caller discards this result anyway.
        if (threads.stop.load(std::memory_order_relaxed))
        {
            ss->estimated = false;
            return VALUE_ZERO;
        }

        if (v <= alpha)
            hi[i] = v;
        else if (v >= beta)
            lo[i] = v;
        else
        {
            lo[i] = hi[i] = v;
            outcomeEstimated |= ss->estimated;
        }

        // Bounds proven by the outcomes (outside the window: never exact)
        const Value upper = aggregate_chance(hi.data(), counts, n, total, ChanceBound::Upper);
        if (upper <= alpha)
        {
            ss->estimated = false;
            return upper;
        }

        const Value lower = aggregate_chance(lo.data(), counts, n, total, ChanceBound::Lower);
        if (lower >= beta)
        {
            ss->estimated = false;
            return lower;
        }

        if (i == n - 1)
        {
            // Exact only if every outcome was exact; otherwise an estimate
            // within the proven bounds
            ss->estimated = outcomeEstimated || lower != upper;
            return lower == upper
                   ? lower
                   : std::clamp(aggregate_chance(values.data(), counts, n, total, ChanceBound::Exact),
                                lower, upper);
        }
    }

    return VALUE_ZERO;  // Unreachable
}


template<NodeType nodeType>
Value Search::Worker::search_child(
  Position& pos, Stack* ss, Value alpha, Value beta, Depth depth, bool cutNode, int revealBonus) {

    // Keep the usual negamax call when there is no bonus
    if (revealBonus == 0)
        return -search<nodeType>(pos, ss, -beta, -alpha, depth, cutNode);

    return Reveal::search_with_bonus(alpha, beta, revealBonus, [&](Value a, Value b) {
        return search<nodeType>(pos, ss, a, b, depth, cutNode);
    });
}


// Sorts the possible identities of a dark piece from the most to the least
// likely (ties: more valuable piece first), and returns their total count.
// The aggregate of a chance node does not depend on the order, but the TT and
// the histories filled by the first, most representative outcome help the
// following ones, and the first outcomes are the ones most likely to decide
// the node on their own (see chance_node()). (Idea from AB-JChess.)
static int sort_outcomes(Position::RestPieceList&                      restPieces,
                         int                                           restCount,
                         std::array<int, Position::MaxRestPieceTypes>& counts) {
    std::stable_sort(restPieces.begin(), restPieces.begin() + restCount,
                     [](const auto& a, const auto& b) {
                         return a.second != b.second ? a.second > b.second
                                                     : PieceValue[a.first] > PieceValue[b.first];
                     });
    int total = 0;
    for (int i = 0; i < restCount; ++i)
        total += counts[i] = restPieces[i].second;
    return total;
}


// Chance node over the identity of a dark piece that the root side (the
// observer) has just captured. In Jieqi the capturer is shown the captured
// piece, so the observer will know which piece left the pool, while its own
// dark pieces captured by the opponent stay unknown (they remain in the pool,
// which is already the correct probability model). Each possible identity is
// searched with that piece removed from its owner's pool, and the results are
// aggregated by chance_node(). (Idea from AB-JChess.)
template<NodeType nodeType>
Value Search::Worker::hidden_capture_search(
  Position& pos, Stack* ss, Value alpha, Value beta, bool isQsearch, Depth depth, bool cutNode) {

    // The captured piece belongs to the side to move
    Position::RestPieceList                        restPieces;
    std::array<int, Position::MaxRestPieceTypes> counts{};
    const int restCount = pos.rest_pieces(pos.side_to_move(), restPieces);
    const int total     = sort_outcomes(restPieces, restCount, counts);

    // Do not enumerate again when the children are searched
    (ss - 1)->capturedDark = false;

    // Pool empty (can only come from an inconsistent input): plain search
    if (total == 0)
    {
        Value v = isQsearch ? qsearch<nodeType>(pos, ss, alpha, beta)
                            : search<nodeType>(pos, ss, alpha, beta, depth, cutNode);
        (ss - 1)->capturedDark = true;
        return v;
    }

    // The capture also takes the piece out of the pool: one more feature
    // removed by the incremental update of the NNUE accumulator (every outcome
    // sets rest_remove_pc, which is cleared again at the end)
    DirtyPiece dp = accumulatorStack.latest().dirtyPiece;
    assert(dp.remove_sq != SQ_NONE);

    ChanceBranchReduction branchReduction((ss - 1)->reduction);

    auto searchOutcome = [&](int i, Value a, Value b) {
        const Piece piece = restPieces[i].first;

        branchReduction.restore();
        accumulatorStack.pop();
        pos.remove_rest_piece(piece);
        dp.rest_remove_pc = piece;
        accumulatorStack.push(dp);

        const Value value = isQsearch ? qsearch<nodeType>(pos, ss, a, b)
                                      : search<nodeType>(pos, ss, a, b, depth, cutNode);

        pos.restore_rest_piece(piece);
        return value;
    };

    const Value value =
      chance_node(ss, counts.data(), restCount, total, alpha, beta, searchOutcome);

    // Leave the capture as the parent made it: its pool is restored, so the
    // top of the accumulator stack must not remove a piece from it any more
    accumulatorStack.pop();
    dp.rest_remove_pc = NO_PIECE;
    accumulatorStack.push(dp);

    (ss - 1)->capturedDark = true;
    return value;
}


template<NodeType nodeType>
Value Search::Worker::flip_search(
    Position& pos, Stack* ss, Value alpha, Value beta, bool isQsearch, Depth depth, bool cutNode) {

    Position::RestPieceList                        restPieces;
    std::array<int, Position::MaxRestPieceTypes> counts{};
    const int restCount = pos.rest_pieces(~pos.side_to_move(), restPieces);
    const int total     = sort_outcomes(restPieces, restCount, counts);

    // In a consistent position a dark piece always has a candidate in its
    // owner's pool. An inconsistent position can only come from the input (it
    // is reported by Engine::set_position()), so do not assert here, but never
    // return an out-of-range value.
    if (total == 0)
        return VALUE_DRAW;

    const Square     flipSq = (ss - 1)->currentMove.to_sq();
    const DirtyPiece baseDp = accumulatorStack.latest().dirtyPiece;

    // Give every outcome the same view of the parent's LMR reduction.
    ChanceBranchReduction branchReduction((ss - 1)->reduction);

    auto searchOutcome = [&](int i, Value a, Value b) {
        const Piece piece = restPieces[i].first;

        branchReduction.restore();
        // Prefetch the TT entry of the outcome as early as possible
        const Key flipKey = pos.flip_key(flipSq, piece);
        prefetch(tt.first_entry(observer_tt_key(flipKey, rootObserver)));

        DirtyPiece dp = baseDp;  // do_flip may add refresh flags for this outcome only
        accumulatorStack.pop();
        const Piece flippedPiece = pos.do_flip(flipSq, piece, &dp);
        accumulatorStack.push(dp);
        assert(pos.key() == flipKey);

        // Optionally search the less likely outcomes one ply shallower
        const Value value =
          isQsearch ? qsearch<nodeType>(pos, ss, a, b)
                    : search<nodeType>(pos, ss, a, b,
                                       chanceBranchDepth && i > 0 && depth > 1 ? depth - 1 : depth,
                                       cutNode);

        pos.undo_flip(flipSq, flippedPiece);
        return value;
    };

    const Value value = chance_node(ss, counts.data(), restCount, total, alpha, beta, searchOutcome);

    // Leave the move of the dark piece as the parent made it, without the
    // reveal (and the refresh flags) of the last outcome: a second search of
    // the same move (e.g. after LMR) starts again from it
    accumulatorStack.pop();
    accumulatorStack.push(baseDp);

    return value;
}


Depth Search::Worker::reduction(bool i, Depth d, int mn, int delta) const {
    int reductionScale = reductions[d] * reductions[mn];
    return reductionScale - (simplifications ? 0 : delta * 1333 / rootDelta)
         + !i * reductionScale * 122 / 246 + 2378;
}

// elapsed() returns the time elapsed since the search started. If the
// 'nodestime' option is enabled, it will return the count of nodes searched
// instead. This function is called to check whether the search should be
// stopped based on predefined thresholds like time limits or nodes searched.
//
// elapsed_time() returns the actual time elapsed since the start of the search.
// This function is intended for use only when printing PV outputs, and not used
// for making decisions within the search algorithm itself.
TimePoint Search::Worker::elapsed() const {
    return main_manager()->tm.elapsed([this]() { return threads.nodes_searched(); });
}

TimePoint Search::Worker::elapsed_time() const { return main_manager()->tm.elapsed_time(); }

Value Search::Worker::evaluate(const Position& pos) {
    return Eval::evaluate(networks[numaAccessToken], pos, accumulatorStack, refreshTable,
                          optimism[pos.side_to_move()], jieqiMaterial);
}

namespace {
// Adjusts a mate from "plies to mate from the root" to
// "plies to mate from the current position". Standard scores are unchanged.
// The function is called before storing a value in the transposition table.
Value value_to_tt(Value v, int ply) { return is_win(v) ? v + ply : is_loss(v) ? v - ply : v; }


// Inverse of value_to_tt(): it adjusts a mate score from the transposition
// table (which refers to the plies to mate/be mated from current position) to
// "plies to mate/be mated from the root". However, to avoid potentially false
// mate scores related to the 40 moves rule and the graph history interaction,
// we return the highest non-mate score instead.
Value value_from_tt(Value v, int ply, int r40c) {

    if (!is_valid(v))
        return VALUE_NONE;

    // Handle win
    if (is_win(v))
        // Downgrade a potentially false mate score
        return VALUE_MATE - v > 80 - r40c ? VALUE_MATE_IN_MAX_PLY - 1 : v - ply;

    // Handle loss
    if (is_loss(v))
        // Downgrade a potentially false mate score.
        return VALUE_MATE + v > 80 - r40c ? VALUE_MATED_IN_MAX_PLY + 1 : v + ply;

    return v;
}


// Adds current move and appends child pv[]
void update_pv(Move* pv, Move move, const Move* childPv) {

    for (*pv++ = move; childPv && *childPv != Move::none();)
        *pv++ = *childPv++;
    *pv = Move::none();
}


// Updates stats at the end of search() when a bestMove is found
void update_all_stats(const Position& pos,
                      Stack*          ss,
                      Search::Worker& workerThread,
                      Move            bestMove,
                      Square          prevSq,
                      SearchedList&   quietsSearched,
                      SearchedList&   capturesSearched,
                      Depth           depth,
                      Move            ttMove,
                      int             moveCount,
                      bool            PvNode) {

    CapturePieceToHistory& captureHistory = workerThread.captureHistory;
    Piece                  movedPiece     = pos.moved_piece(bestMove);
    PieceType              capturedPiece;

    int bonus = std::min(144 * depth - 106, 1803) + 273 * (bestMove == ttMove);

    // The more moves were searched before the best one, the more significant
    // it is (UCI option BestMoveBonusScale). 64-bit product: no overflow.
    if (workerThread.bestMoveBonusScale && !PvNode)
        bonus += int(std::int64_t(bonus) * (quietsSearched.size() + capturesSearched.size()) / 256);
    int malus = std::min(1012 * depth - 213, 2437) - (workerThread.malusDecay ? 0 : 31 * moveCount);

    if (!pos.capture(bestMove))
    {
        update_quiet_histories(pos, ss, workerThread, bestMove, bonus * 1077 / 1024);

        // Decrease stats for all non-best quiet moves (with MalusDecay, less and
        // less for the later ones)
        int quietMalus = malus * 1555 / 1024;
        for (Move move : quietsSearched)
        {
            if (workerThread.malusDecay)
                quietMalus = quietMalus * 950 / 1024;
            update_quiet_histories(pos, ss, workerThread, move, -quietMalus);
        }
    }
    else
    {
        // Increase stats for the best move in case it was a capture move
        capturedPiece = pos.capture_type(bestMove.to_sq());
        captureHistory[movedPiece][bestMove.to_sq()][capturedPiece] << bonus * 1264 / 1024;
    }

    // Extra penalty for a quiet early move that was not a TT move in
    // previous ply when it gets refuted.
    if (prevSq != SQ_NONE && ((ss - 1)->moveCount == 1 + (ss - 1)->ttHit) && !pos.captured_piece())
        update_continuation_histories(workerThread, ss - 1, (ss - 1)->movedPiece, prevSq,
                                      -malus * 681 / 1024);

    // Decrease stats for all non-best capture moves
    for (Move move : capturesSearched)
    {
        movedPiece    = pos.moved_piece(move);
        capturedPiece = pos.capture_type(move.to_sq());
        captureHistory[movedPiece][move.to_sq()][capturedPiece] << -malus * 1184 / 1024;
    }
}


// Updates histories of the move pairs formed by moves at ply -1 to -6 (not -5
// with the UCI option Simplifications) with the current move.
void update_continuation_histories(
  const Search::Worker& w, Stack* ss, Piece pc, Square to, int bonus) {
    static constexpr std::array<ConthistBonus, 6> conthist_bonuses = {
      {{1, 1028}, {2, 583}, {3, 282}, {4, 529}, {5, 130}, {6, 501}}};

    // Counter-Move History Continuity (UCI option CMHC): scale the update
    // up when the continuation histories updated so far already agree
    // (are positive), in 1/128 units.
    static constexpr int CMHCMultipliers[] = {96, 100, 100, 100, 115, 118, 129};
    int                  positiveCount     = 0;

    for (const auto [i, weight] : conthist_bonuses)
    {
        // Only update the first 2 continuation histories if we are in check
        if (ss->inCheck && i > 2)
            break;
        // No update of the move 5 plies back with Simplifications
        if (i == 5 && w.simplifications)
            continue;
        if (((ss - i)->currentMove).is_ok())
        {
            auto& entry = (*(ss - i)->continuationHistory)[pc][to];
            // UCI option ContHistOffset: offset for the move one ply back
            const int offset = w.contHistOffset && i == 1 ? 88 : 0;
            if (w.cmhc)
            {
                positiveCount += entry > 0;
                entry << int(std::int64_t(bonus) * weight * CMHCMultipliers[positiveCount]
                             / (1024 * 128))
                           + offset;
            }
            else
                entry << bonus * weight / 1024 + offset;
        }
    }
}

// Updates move sorting heuristics

void update_quiet_histories(
  const Position& pos, Stack* ss, Search::Worker& workerThread, Move move, int bonus) {

    Color us = pos.side_to_move();
    workerThread.mainHistory[us][move.from_to()] << bonus;  // Untuned to prevent duplicate effort

    if (ss->ply < LOW_PLY_HISTORY_SIZE)
        workerThread.lowPlyHistory[ss->ply][move.from_to()] << bonus * 839 / 1024;

    update_continuation_histories(workerThread, ss, pos.moved_piece(move), move.to_sq(),
                                  bonus * (bonus > 0 ? 1064 : 824) / 1024);

    // With BigPawnHistory, stronger updates, scaled like upstream's (691 / 468
    // -> 800 / 500)
    int pIndex = workerThread.pawn_history_index(pos);
    workerThread.pawnHistory[pIndex][pos.moved_piece(move)][move.to_sq()]
      << bonus
           * (workerThread.bigPawnHistory ? (bonus > 0 ? 873 : 491) : (bonus > 0 ? 754 : 460))
           / 1024;
}

}

// Used to print debug info and, more importantly, to detect
// when we are out of available time and thus stop the search.
void SearchManager::check_time(Search::Worker& worker) {
    if (--callsCnt > 0)
        return;

    // When using nodes, ensure checking rate is not lower than 0.1% of nodes
    callsCnt = worker.limits.nodes ? std::min(512, int(worker.limits.nodes / 1024)) : 512;

    static TimePoint lastInfoTime = now();

    TimePoint elapsed = tm.elapsed([&worker]() { return worker.threads.nodes_searched(); });
    TimePoint tick    = worker.limits.startTime + elapsed;

    if (tick - lastInfoTime >= 1000)
    {
        lastInfoTime = tick;
        dbg_print();
    }

    // We should not stop pondering until told so by the GUI
    if (ponder)
        return;

    if (
      // Later we rely on the fact that we can at least use the mainthread previous
      // root-search score and PV in a multithreaded environment to prove mated-in scores.
      worker.completedDepth >= 1
      && ((worker.limits.use_time_management() && (elapsed > tm.maximum() || stopOnPonderhit))
          || (worker.limits.movetime && elapsed >= worker.limits.movetime)
          || (worker.limits.nodes && worker.threads.nodes_searched() >= worker.limits.nodes)))
        worker.threads.stop = true;
}

void SearchManager::pv(const Search::Worker&     worker,
                       const ThreadPool&         threads,
                       const TranspositionTable& tt,
                       Depth                     depth) const {

    const auto  nodes     = threads.nodes_searched();
    const auto& rootMoves = worker.rootMoves;
    const auto& pos       = worker.rootPos;
    size_t      multiPV   = std::min(size_t(worker.options["MultiPV"]), rootMoves.size());

    for (size_t i = 0; i < multiPV; ++i)
    {
        bool usePreviousScore = rootMoves[i].score == -VALUE_INFINITE;

        if (depth == 1 && usePreviousScore && i > 0)
            continue;

        Depth d = usePreviousScore ? std::max(1, depth - 1) : depth;
        Value v = usePreviousScore ? rootMoves[i].previousScore : rootMoves[i].uciScore;

        // The root move's own reveal bonus is a preference of the search, not
        // part of the value of the position (bounds are shifted the same way)
        v = v == -VALUE_INFINITE ? VALUE_ZERO : Reveal::remove_bonus(v, rootMoves[i].revealBonus);

        std::string pv;
        Position    tempPos;
        StateInfo   st;
        // StateInfo objects must outlive every do_move() on tempPos, because each
        // new state keeps a pointer to the previous one.
        std::deque<StateInfo> pvStates;
        tempPos.set(pos, &st);

        for (Move m : usePreviousScore ? rootMoves[i].previousPV : rootMoves[i].pv)
        {
            pv += UCIEngine::move(m);

            // Stop at the first move of a dark piece: what follows depends on
            // the revealed piece, which is not known yet.
            if (tempPos.move_dark(m))
                break;

            tempPos.do_move(m, pvStates.emplace_back());

            pv += " ";
        }

        // Remove last whitespace
        if (!pv.empty() && pv.back() == ' ')
            pv.pop_back();

        auto wdl   = worker.options["UCI_ShowWDL"] ? UCIEngine::wdl(v, pos) : "";
        auto bound = rootMoves[i].scoreLowerbound
                     ? "lowerbound"
                     : (rootMoves[i].scoreUpperbound ? "upperbound" : "");

        InfoFull info;

        info.depth    = d;
        info.selDepth = rootMoves[i].selDepth;
        info.multiPV  = i + 1;
        info.score    = {v, pos};
        info.wdl      = wdl;

        // Previous scores are exact, even though their bound flags may say otherwise.
        // A move searched in this iteration keeps its bound flags even when it is not
        // the current pvIdx (partially searched, or new best move of a stopped search).
        if (!usePreviousScore)
            info.bound = bound;

        TimePoint time = std::max(TimePoint(1), tm.elapsed_time());
        info.timeMs    = time;
        info.nodes     = nodes;
        info.nps       = nodes * 1000 / time;
        info.tbHits    = 0;
        info.pv        = pv;
        info.hashfull  = tt.hashfull();

        updates.onUpdateFull(info);
    }
}

// Called in case we have no ponder move before exiting the search,
// for instance, in case we stop the search during a fail high at root.
// We try hard to have a ponder move to return to the GUI,
// otherwise in case of 'ponder on' we have nothing to think about.
bool RootMove::extract_ponder_from_tt(const TranspositionTable& tt, Position& pos, Color observer) {

    StateInfo st;

    assert(pv.size() == 1);
    if (pv[0] == Move::none() || pos.move_dark(pv[0]))
        return false;

    pos.do_move(pv[0], st);

    // No ponder move if the best move ends the game (e.g. a draw by repetition)
    Value result;
    if (!pos.rule_judge(result, 1))
    {
        auto [ttHit, ttData, ttWriter] = tt.probe(observer_tt_key(pos.key(), observer));
        if (ttHit && MoveList<LEGAL>(pos).contains(ttData.move))
            pv.push_back(ttData.move);
    }

    pos.undo_move(pv[0]);
    return pv.size() > 1;
}


}  // namespace Maskfish
