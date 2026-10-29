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

#include "engine.h"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <deque>
#include <iostream>
#include <memory>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

#include "evaluate.h"
#include "misc.h"
#include "movepick.h"
#include "nnue/features/half_ka_v2_hm.h"
#include "nnue/network.h"
#include "nnue/nnue_accumulator.h"
#include "nnue/nnue_common.h"
#include "nnue/nnue_misc.h"
#include "numa.h"
#include "perft.h"
#include "position.h"
#include "score.h"
#include "search.h"
#include "shm.h"
#include "types.h"
#include "uci.h"
#include "ucioption.h"

namespace Maskfish {

namespace NN = Eval::NNUE;

int MaxThreads = std::max(1024, 4 * int(get_hardware_concurrency()));

// The default configuration will attempt to group L3 domains up to 32 threads.
// This size was found to be a good balance between the Elo gain of increased
// history sharing and the speed loss from more cross-cache accesses (see
// PR#6526). The user can always explicitly override this behavior.
constexpr NumaAutoPolicy DefaultNumaPolicy = BundledL3Policy{32};

Engine::Engine(std::optional<std::filesystem::path> path) :
    binaryDirectory(path ? CommandLine::get_binary_directory(*path) : std::filesystem::path{}),
    numaContext(NumaConfig::from_system(DefaultNumaPolicy)),
    states(new std::deque<StateInfo>(1)),
    threads(),
    networks(numaContext, get_default_networks()) {

    pos.set(StartFEN, &states->back());

    options.add(  //
      "Debug Log File", Option("", [](const Option& o) {
          start_logger(path_from_utf8(std::string(o)));
          return std::nullopt;
      }));

    options.add(  //
      "NumaPolicy", Option("auto", [this](const Option& o) {
          if (!set_numa_config_from_option(o))
              return "NumaPolicy: invalid value '" + std::string(o) + "', keeping previous config.";
          return numa_config_information_as_string() + "\n"
               + thread_allocation_information_as_string();
      }));

    options.add(  //
      "Threads", Option(1, 1, MaxThreads, [this](const Option&) {
          resize_threads();
          return thread_allocation_information_as_string();
      }));

    options.add(  //
      "Hash", Option(16, 1, MaxHashMB, [this](const Option& o) {
          set_tt_size(o);
          return std::nullopt;
      }));

    options.add(  //
      "Clear Hash", Option([this](const Option&) {
          search_clear();
          return std::nullopt;
      }));

    options.add(  //
      "Ponder", Option(false));

    options.add(  //
      "MultiPV", Option(1, 1, MAX_MOVES));

    options.add("Move Overhead", Option(10, 0, 5000));

    options.add("nodestime", Option(0, 0, 10000));

    options.add("UCI_ShowWDL", Option(false));

    // Report scores in internal units (for training data generation)
    options.add("RawScore", Option(false, [](const Option& o) {
                    Score::rawInternal = bool(o);
                    return std::nullopt;
                }));

    // Jieqi: chance node over the identity of a dark piece captured by the
    // side the engine searches for (see Search::Worker::hiddenCaptureModel).
    options.add("HiddenCaptureModel", Option(true));

    // Jieqi: bonus for safe reveals (quiet moves of dark pieces). Set them all
    // to 0 to disable.
    options.add("RevealQuietBase", Option(-12, -96, 96));
    options.add("RevealQuietPhase", Option(0, -96, 96));
    options.add("RevealQuietSafety", Option(24, -192, 192));
    options.add("RevealQuietHighValue", Option(-12, -96, 96));
    options.add("RevealQuietDiversity", Option(0, -96, 96));

    // Experimental, off by default: test with SPRT.
    // ProbCut asks a larger margin (+64) and one ply less for dark captures.
    options.add("ProbCutDarkCapture", Option(false));
    // Lower the depth of TT entries that are deep enough but cannot cut.
    options.add("TTPenalize", Option(false));
    // Search the less likely outcomes of a reveal one ply shallower.
    options.add("ChanceBranchDepth", Option(false));
    // Reset the correction histories at the start of every search (they are
    // otherwise kept until ucinewgame).
    options.add("CorrHistReset", Option(false));
    // Check the time every 64 nodes instead of 512.
    options.add("FrequentTimeCheck", Option(false));
    // Let the threads vote for the best move only when MultiPV is 1: in
    // MultiPV analysis the main thread's lines are kept.
    options.add("VoteSinglePVOnly", Option(false));
    // Also give the safe reveal bonus to quiet moves of dark pieces that
    // evade a check in the quiescence search.
    options.add("RevealQsearchEvasions", Option(false));

    // Experimental search ideas, off by default: test
    // with SPRT (trainer/match.py --opt-a Name=true).
    // Continuation correction history also from the move 4 plies back.
    options.add("ContCorr4Ply", Option(false));
    // On a multi-cut, penalize the ttMove in ttMoveHistory and use the cut
    // value to update the correction histories.
    options.add("MultiCutHistory", Option(false));
    // Counter-Move History Continuity: larger continuation history updates
    // when the previous continuation histories already agree.
    options.add("CMHC", Option(false));
    // When the root score suggests a forced mate, futility-prune only at low
    // depth, skip singular extensions and the small ProbCut, reduce null moves
    // less, and reduce checks less and other moves more, to find the mate faster.
    options.add("SeekMate", Option(false));
    // With SeekMate: decide whether to seek a mate once per PV line and
    // iteration, from the score of the line in the previous iteration, instead
    // of at every node from the current score of the root move (which is
    // -VALUE_INFINITE for the MultiPV lines after the first until searched).
    options.add("SeekMatePrevScore", Option(false));
    // Small pruning changes: razoring only at expected ALL nodes, more null
    // move pruning when improving and a reduction growing with the static eval
    // margin, deeper probcut reduction when improving, and a larger capture
    // futility margin when recapturing the piece that just moved.
    options.add("PruningTweaks", Option(false));
    // LMR: reduce quiet moves depending on the static eval vs alpha, and scale
    // up reductions at expected ALL nodes.
    options.add("LMRTweaks", Option(false));
    // After the root search failed high (and was re-searched at a reduced
    // depth), increase the depth back gradually in the next iterations.
    options.add("FailHighRecovery", Option(false));
    // Aspiration windows: after a fail low the new beta is alpha, after a fail
    // high the new alpha is beta - delta, and helper threads start with
    // slightly different windows.
    options.add("AspirationTweaks", Option(false));
    // At each new search, move the main history 1/4 of the way back to its
    // initial value.
    options.add("MainHistoryDecay", Option(false));
    // History malus of the quiet moves searched before the best move decreasing
    // geometrically with their number (instead of the move count term).
    options.add("MalusDecay", Option(false));
    // At non-PV nodes, scale the history bonus of the best move with the
    // number of moves searched before it.
    options.add("BestMoveBonusScale", Option(false));
    // Pruning at shallow depth: divide the history of quiet moves by a
    // depth dependent value (instead of a constant) when reducing lmrDepth.
    options.add("LMRHistoryDivisor", Option(false));
    // Simplifications of the search: razoring margin
    // linear in depth, LMR not depending on the window but smaller at PV nodes,
    // no continuation history update 5 plies back, bonus of a quiet TT move
    // linear in depth, a futility margin that does not depend on the best move,
    // the same negative extension at cut nodes as for TT values above beta.
    options.add("Simplifications", Option(false));
    // Add a constant offset to the continuation history updates of the move
    // one ply back.
    options.add("ContHistOffset", Option(false));
    // Correction history: smaller updates when the search result is above the
    // static eval than when it is below.
    options.add("CorrHistAsym", Option(false));
    // Continuation correction history also from the move 6 plies back.
    options.add("ContCorr6Ply", Option(false));
    // Compute the static eval and the hindsight depth adjustments before the
    // TT cutoff (which then uses the adjusted depth).
    options.add("HindsightFirst", Option(false));
    // On the PV of the previous iteration, no internal iterative reduction and
    // no shallow pruning of quiet moves at PV nodes.
    options.add("FollowPV", Option(false));
    // Increase the reduction already when the next ply has two fail highs
    options.add("CutoffCntTweak", Option(false));
    // Make null move pruning easier after null move fail highs of previous siblings
    options.add("NMPFailHighCount", Option(false));
    // Null move pruning: the threads with an odd index reduce one ply less, so
    // that the threads search more different trees (only matters with
    // Threads > 1)
    options.add("NMPOddThreads", Option(false));
    // Before qsearch, extend the TT move for a decisive TT value, otherwise
    // only for a TT depth above 1
    options.add("MateTTExtension", Option(false));
    // Under time management, stop as soon as no better move can be found: a single
    // legal move, mate in 1 or 2 in all lines, or mated in 1
    options.add("StopEarly", Option(false));
    // Average the scores of a root move with a weight depending on the effort
    options.add("RootScoreEMA", Option(false));
    // In qsearch, do not store the stand pat value in the TT (only the static evaluation)
    options.add("StandPatNoTT", Option(false));
    // Restrict the secondary aging of TT entries to decisive values
    options.add("DecisiveTTAging", Option(false, [this](const Option& o) {
                    tt.set_decisive_aging(bool(o));
                    return std::nullopt;
                }));
    // Compare 21 bits of the keys inside a TT cluster instead of 16 (fewer
    // false matches, same memory). Set it before the search (or clear the hash).
    options.add("TTKey21", Option(false, [this](const Option& o) {
                    tt.set_key_bits(bool(o) ? 21 : 16);
                    return std::nullopt;
                }));
    // Pawn history indexed by 10 bits of the pawn key instead of 9 (the table
    // always has the size for 10), with stronger updates
    options.add("BigPawnHistory", Option(false));
    // Share the continuation, pawn and correction histories (not the
    // continuation correction history) between the threads of a NUMA node;
    // the pawn and correction histories then grow with the
    // number of threads (only matters with Threads > 1). Changing it recreates
    // the threads.
    options.add("SharedHistory", Option(false, [this](const Option&) {
                    resize_threads();
                    return std::nullopt;
                }));
    // LMR: statScore without the constant subtracted from it (the base reduction
    // is raised to compensate for quiet moves)
    options.add("StatScoreNoOffset", Option(false));
    // Full-depth search when LMR is skipped: no ttMoveHistory term in the
    // reduction, and lower thresholds for reducing the depth (3744 and 5150
    // instead of 4170 and 5563)
    options.add("NoTTMoveHistoryLMR", Option(false));
    // LMR: the extra ply of the reduced search is given at PV nodes (instead of
    // at nodes whose parent is a PV node)
    options.add("LMRPvNodeSelf", Option(false));
    // LMR: no reduction term for the number of consecutive quiet moves leading
    // to the node; the base reduction is raised by 286 instead
    options.add("NoQuietMoveStreak", Option(false));
    // LMR: the reduction decrease of the TT move is not applied when the next
    // ply has many fail highs, and does not take the reduction below 0
    options.add("TTMoveReductionFloor", Option(false));
    // LMR re-search: the same branch for every fail high of the reduced search
    // (also when it was not reduced), searching deeper above bestValue + 60
    options.add("SimpleLMRResearch", Option(false));
    // Singular extensions: the minimum depth does not decrease once the
    // completed depth exceeds 32
    options.add("SENoCompletedDepth", Option(false));
    // Singular extensions: the triple extension margin is lowered beyond the
    // root depth (instead of 1.5 times the root depth), like the double margin
    options.add("TripleMarginPly", Option(false));
    // Futility pruning of captures also when the side to move is in check
    options.add("CaptureFutilityInCheck", Option(false));
    // SEE pruning of captures and checks: the capture history term of the
    // margin is not clamped
    options.add("SimpleSeeMargin", Option(false));
    // The node is also considered improving when the static evaluation is at
    // least beta (instead of beta + 108)
    options.add("ImprovingAtBeta", Option(false));
    // ProbCut verifies captures at depth - 3, or at depth - 5 when the node is
    // improving (instead of depth - 5)
    options.add("ProbCutImprovingDepth", Option(false));
    // No upper bound on the term from the statScore of the previous move in the
    // bonus scale of a quiet move that caused a fail low
    options.add("BonusScaleNoCap", Option(false));
    // The history bonus of the best move also depends on the statScore of the
    // previous move (statScore / 32)
    options.add("BonusPrevStatScore", Option(false));
    // Continuation histories of quiet moves are updated with one scale (944/1024)
    // for bonuses and maluses instead of 1064/1024 and 824/1024
    options.add("ContHistSingleScale", Option(false));
    // The static evaluation difference also updates the main history of the
    // previous move on a TT hit
    options.add("EvalDiffOnTTHit", Option(false));
    // The correction history is updated upwards only when a best move was found
    // and downwards only when none was
    options.add("CorrHistBestMoveSign", Option(false));
    // Captures are ordered without the bonus for captures giving check
    options.add("NoCaptureCheckBonus", Option(false, [](const Option& o) {
                    MovePicker::noCaptureCheckBonus = bool(o);
                    return std::nullopt;
                }));
    // Bonus for escaping and penalty for moving into an attack by a lesser piece
    // of a quiet move: 20 * piece value, the two cancelling out
    options.add("ThreatByLesserValue", Option(false, [](const Option& o) {
                    MovePicker::threatByLesserValue = bool(o);
                    return std::nullopt;
                }));
    // Quiet evasions are ordered without the low ply history
    options.add("NoEvasionLowPly", Option(false, [](const Option& o) {
                    MovePicker::noEvasionLowPly = bool(o);
                    return std::nullopt;
                }));
    // ProbCut tries the TT move first when it is a capture, without testing its
    // SEE against the ProbCut threshold
    options.add("ProbCutTTNoSee", Option(false, [](const Option& o) {
                    MovePicker::probCutTTNoSee = bool(o);
                    return std::nullopt;
                }));
    // Time management: the share of the nodes spent on the best move scales the
    // time of the move smoothly instead of stopping the search early
    options.add("TMEffortFactor", Option(false));
    // Time management: the time reduction for a stable best move is linear in
    // the number of iterations it has been stable, between 0.67 and 1.44
    options.add("TMReductionInterp", Option(false));
    // History bonus from the static evaluation difference as the clamped
    // difference times a constant (24 main history, 23 pawn history)
    options.add("SimpleEvalDiffBonus", Option(false));
    // Count dark pieces at their expected value in the material scaling of
    // the evaluation.
    options.add("JieqiMaterial", Option(false));
    // Scaling of the network output in the evaluation: Complexity (default,
    // |psqt - positional| and the major material), Alignment (agreement of
    // the network with the material
    // balance, and the total material; suits networks without PSQT) or
    // Correlation (the same agreement, with dark pieces at the average value
    // of their pool, and
    // the major material). The trainer must use the same (train.py
    // --eval-scaling); it does not model Correlation.
    options.add("EvalScaling", Option("Complexity var Complexity var Alignment var Correlation",
                                      "Complexity"));
    // Time management: the move horizon in whole
    // moves, at most 50 (instead of centi-moves, at most 60).
    options.add("SimpleMtg", Option(false));
    // Simpler optimism scaling: the weight of optimism in
    // the evaluation does not depend on the material (EvalScaling Complexity).
    options.add("OptimismSimple", Option(false));
    // EvalScaling Alignment and Correlation: the evaluation and optimism terms
    // of the agreement boost in one division, (nnue + 4 * optimism) * a / 65536,
    // instead of two (only the rounding differs)
    options.add("OptimismMerged", Option(false));
    // Alternative futility pruning (margin, depth limit, returned value).
    options.add("AltFutility", Option(false));
    // Simpler thread selection: votes by score only (Threads > 1).
    options.add("SimpleThreadVote", Option(false));
    // The squares attacked by the opponent are cached in the StateInfo
    // and used to generate and test king moves. No functional change (bench
    // identical): compare the speed with and without.
#ifdef JIEQI_NNUE_ATT
    // Mid mirror (make nnue=att): tell apart the pawn pairs with the same sum
    // of squares. The network and the prepared training data must use the
    // same setting (trainer option --mid-pawn-tiebreak).
    options.add("MidMirrorPawnTieBreak", Option(false, [](const Option& o) {
                    Eval::NNUE::FeatureSet::midPawnTieBreak = bool(o);
                    return std::nullopt;
                }));
    // Mid mirror: do not test the king flag (bit 63) of the balance encodings,
    // which the borrows of unbalanced positions can change (trainer option
    // --mid-no-king-flag).
    options.add("MidMirrorNoKingFlag", Option(false, [](const Option& o) {
                    Eval::NNUE::FeatureSet::midNoKingFlag = bool(o);
                    return std::nullopt;
                }));
#endif
    options.add("ThreatCache", Option(false, [](const Option& o) {
                    Position::useThreatCache = bool(o);
                    return std::nullopt;
                }));
    // Threats by lesser pieces of the quiet move ordering from the cached
    // threats (x-raying our king): changes the move ordering.
    options.add("ThreatCacheMovePick", Option(false, [](const Option& o) {
                    Position::useThreatCacheMovePick = bool(o);
                    return std::nullopt;
                }));
#ifdef JIEQI_NNUE_THREATS
    // Incremental threat features (DirtyThreats, both perspectives
    // together): the moves record the threats
    // they add and remove, and the threat accumulator is updated from them
    // instead of from the difference of the lists of threats of the two
    // positions. No functional change (bench identical): compare the speed.
    options.add("ThreatsIncremental", Option(false, [](const Option& o) {
                    Eval::NNUE::AccumulatorStack::incrementalThreats = bool(o);
                    return std::nullopt;
                }));
#endif
    // Score -> win rate scale used to average chance nodes (0 = built-in
    // 360.83524). trainer/fit_scaling.py fits it from self-play data.
    options.add("ChanceScaling", Option(0, 0, 3000));
    // Risk preference of the side causing a chance event, in percent of the
    // standard deviation of the outcomes' win rates: > 0 avoids uncertain
    // reveals and captures, < 0 seeks them.
    options.add("ChanceRiskAversion", Option(0, -100, 100));
    // How the outcomes of a chance node are averaged: Winrate
    // (default), Tables (score <-> mass tables of ChanceTables) or Linear
    // (average of the scores clamped to +-2000). Tables
    // without loaded tables falls back to Winrate.
    options.add("ChanceModel", Option("Winrate var Winrate var Tables var Linear", "Winrate"));
    // Directory of the tables of ChanceModel Tables (made by
    // trainer/chance_tables.py)
    options.add("ChanceTables", Option("", [](const Option& o) {
                    return std::optional<std::string>(Search::load_chance_tables(std::string(o)));
                }));

    options.add(  //
      "EvalFile", Option(EvalFileDefaultNameBig, [this](const Option& o) {
          load_big_network(std::string(o));
          return std::nullopt;
      }));

    threads.clear();
    threads.ensure_network_replicated();
    resize_threads();
}

// Move generation does not need the network: perft also works without one
// (no Jieqi network is distributed with the engine, see tests/perft.sh)
u64 Engine::perft(const std::string& fen, Depth depth) { return Benchmark::perft(fen, depth); }

void Engine::go(Search::LimitsType& limits) {
    assert(limits.perft == 0);

    // An invalid position is not searched (no root moves): it needs no network
    if (positionValid)
        verify_networks();
    else
        sync_cout << "info string Invalid position, not searched" << sync_endl;

    threads.start_thinking(pos, states, limits, positionValid);
}
void Engine::stop() { threads.stop = true; }

void Engine::search_clear() {
    wait_for_search_finished();

    tt.clear(threads);
    threads.clear();
}

void Engine::set_on_update_no_moves(std::function<void(const Engine::InfoShort&)>&& f) {
    updateContext.onUpdateNoMoves = std::move(f);
}

void Engine::set_on_update_full(std::function<void(const Engine::InfoFull&)>&& f) {
    updateContext.onUpdateFull = std::move(f);
}

void Engine::set_on_iter(std::function<void(const Engine::InfoIter&)>&& f) {
    updateContext.onIter = std::move(f);
}

void Engine::set_on_bestmove(std::function<void(std::string_view, std::string_view)>&& f) {
    updateContext.onBestmove = std::move(f);
}

void Engine::set_on_start(std::function<void()>&& f) { updateContext.onStart = std::move(f); }

void Engine::set_on_verify_networks(std::function<void(std::string_view)>&& f) {
    onVerifyNetworks = std::move(f);
}

void Engine::wait_for_search_finished() { threads.main_thread()->wait_for_search_finished(); }

void Engine::set_position(const std::string& fen, const std::vector<std::string>& moves) {
    constexpr std::string_view PieceToChar(" RACPNBK racpnbkXx");

    // Drop the old state and create a new one
    states = StateListPtr(new std::deque<StateInfo>(1));
    pos.set(fen, &states->back());
    positionValid = true;

    // A position that fails this check may not be handled by move generation,
    // the search or the network (e.g. too many pieces): ignore the moves, and
    // 'go' will not search
    if (std::string reason; !pos.pool_is_consistent(&reason))
    {
        sync_cout << "info string Invalid position: " << reason << sync_endl;
        positionValid = false;
        return;
    }

    // Parses one character naming a revealed piece of color c, which must still
    // be in the pool. Returns NO_PIECE if the character is not acceptable.
    auto revealed = [&](char ch, Color c) {
        size_t idx = PieceToChar.find(ch);
        if (idx == std::string_view::npos || idx == 0 || idx >= size_t(PIECE_NB))
            return NO_PIECE;
        Piece pc = Piece(idx);
        if (type_of(pc) == NO_PIECE_TYPE || type_of(pc) == KING || color_of(pc) != c
            || pos.rest_piece(pc) <= 0)
            return NO_PIECE;
        return pc;
    };

    // Moves are given in the usual coordinate notation, followed by
    //  - the piece revealed by the move, if a dark piece was moved (mandatory),
    //  - then the identity of the captured piece, if a dark piece was captured
    //    (optional: the owner of the captured piece may not know it).
    // e.g. "a3a4P", "b2b9Cn" (cannon revealed, captured dark piece was a knight).
    for (const auto& move : moves)
    {
        auto m = UCIEngine::to_move(pos, move);

        if (m == Move::none())
        {
            sync_cout << "info string Invalid move " << move
                      << ": illegal move, the position will not be searched" << sync_endl;
            positionValid = false;
            break;
        }

        const Color us          = pos.side_to_move();
        const bool  moveDark    = pos.move_dark(m);
        const bool  captureDark = pos.capture(m) && pos.is_dark(m.to_sq());

        Piece       flipped = NO_PIECE, captured = NO_PIECE;
        size_t      i       = 4;
        std::string error;

        if (moveDark)
        {
            if (move.size() <= i)
                error = "missing the revealed piece";
            else if ((flipped = revealed(move[i++], us)) == NO_PIECE)
                error = "invalid revealed piece";
        }

        if (error.empty() && captureDark && move.size() > i
            && (captured = revealed(move[i++], ~us)) == NO_PIECE)
            error = "invalid captured piece";

        if (error.empty() && move.size() > i)
            error = "unexpected characters";

        if (!error.empty())
        {
            sync_cout << "info string Invalid move " << move << ": " << error
                      << ", the position will not be searched" << sync_endl;
            positionValid = false;
            break;
        }

        states->emplace_back();
        pos.do_move(m, states->back());

        if (moveDark)
            pos.do_flip(m.to_sq(), flipped);

        if (captured != NO_PIECE)
            pos.remove_rest_piece(captured);
    }
}

// modifiers

bool Engine::set_numa_config_from_option(const std::string& o) {
    if (o == "auto" || o == "system")
    {
        numaContext.set_numa_config(NumaConfig::from_system(DefaultNumaPolicy));
    }
    else if (o == "hardware")
    {
        // Don't respect affinity set in the system.
        numaContext.set_numa_config(NumaConfig::from_system(DefaultNumaPolicy, false));
    }
    else if (o == "none")
    {
        numaContext.set_numa_config(NumaConfig{});
    }
    else
    {
        auto parsed = NumaConfig::from_string(o);
        if (!parsed.has_value())
            return false;
        numaContext.set_numa_config(std::move(*parsed));
    }

    // Force reallocation of threads in case affinities need to change.
    resize_threads();
    threads.ensure_network_replicated();
    return true;
}

void Engine::resize_threads() {
    threads.wait_for_search_finished();
    threads.set(numaContext.get_numa_config(), {options, threads, tt, networks},
                updateContext);

    // Reallocate the hash with the new threadpool size
    set_tt_size(options["Hash"]);
    threads.ensure_network_replicated();
}

void Engine::set_tt_size(usize mb) {
    wait_for_search_finished();
    tt.resize(mb, threads);
}

void Engine::set_ponderhit(bool b) { threads.main_manager()->ponder = b; }

// network related

void Engine::verify_networks() const {
    networks->big.verify(options["EvalFile"], onVerifyNetworks);

    auto statuses = networks.get_status_and_errors();
    for (size_t i = 0; i < statuses.size(); ++i)
    {
        const auto [status, error] = statuses[i];
        std::string message        = "Network replica " + std::to_string(i + 1) + ": ";
        if (status == SystemWideSharedConstantAllocationStatus::NoAllocation)
        {
            message += "No allocation.";
        }
        else if (status == SystemWideSharedConstantAllocationStatus::LocalMemory)
        {
            message += "Local memory.";
        }
        else if (status == SystemWideSharedConstantAllocationStatus::SharedMemory)
        {
            message += "Shared memory.";
        }
        else
        {
            message += "Unknown status.";
        }

        if (error.has_value())
        {
            message += " " + *error;
        }

        onVerifyNetworks(message);
    }
}

// The network with the default file name (EvalFile default value), loaded
// before it is shared: replicating it is then done only once
std::unique_ptr<NN::Networks> Engine::get_default_networks() const {

    // Heap-allocate because sizeof(NN::Networks) is large
    auto networks_ = std::make_unique<NN::Networks>(
      std::make_unique<NN::NetworkBig>(NN::EvalFile{EvalFileDefaultNameBig, "None", ""}));

    networks_->big.load(binaryDirectory, "");

    return networks_;
}

void Engine::load_big_network(const std::string& file) {
    networks.modify_and_replicate(
      [this, &file](NN::Networks& networks_) { networks_.big.load(binaryDirectory, file); });
    threads.clear();
    threads.ensure_network_replicated();
}

void Engine::save_network(const std::pair<std::optional<std::string>, std::string> files) {
    networks.modify_and_replicate(
      [&files](NN::Networks& networks_) { networks_.big.save(files.first); });
}

// utility functions

void Engine::trace_eval() const {
    StateListPtr trace_states(new std::deque<StateInfo>(1));
    Position     p;
    p.set(pos.fen(), &trace_states->back());

    if (!positionValid)
    {
        sync_cout << "info string Invalid position, not evaluated" << sync_endl;
        return;
    }

    if (std::string reason; !p.pool_is_consistent(&reason))
    {
        sync_cout << "info string Invalid position: " << reason << sync_endl;
        return;
    }

    verify_networks();

    sync_cout << "\n" << Eval::trace(p, *networks) << sync_endl;
}

// Prints the raw network output for the current position, in the units of
// the network (before evaluate() scaling): used to check that a network
// exported by the trainer is read exactly like the trainer computes it.
void Engine::eval_raw() const {
    StateListPtr evalStates(new std::deque<StateInfo>(1));
    Position     p;
    p.set(pos.fen(), &evalStates->back());

    if (!positionValid)
    {
        sync_cout << "info string Invalid position, not evaluated" << sync_endl;
        return;
    }

    // The network needs a valid position (at most 16 pieces per side, ...)
    if (std::string reason; !p.pool_is_consistent(&reason))
    {
        sync_cout << "info string Invalid position: " << reason << sync_endl;
        return;
    }

    verify_networks();

    // Heap-allocated: the accumulator stack is large
    auto accumulators = std::make_unique<Eval::NNUE::AccumulatorStack>();
    auto caches       = std::make_unique<Eval::NNUE::AccumulatorCaches>(*networks);
    auto [psqt, positional] = networks->big.evaluate(p, *accumulators, &caches->big);

    // Static evaluation after the scaling of Eval::evaluate() (optimism 0, the
    // JieqiMaterial and EvalScaling options as set): what the search scores, and so the training
    // data, are made of. The trainer applies the same scaling to the network
    // output when it computes the loss (trainer/eval_scale.py).
    std::string finalEval = "none";  // evaluate() is not defined in check
    if (!p.checkers())
        finalEval = std::to_string(int(Eval::evaluate(*networks, p, *accumulators, *caches,
                                                      VALUE_ZERO, bool(options["JieqiMaterial"]),
                                                      Eval::eval_scaling(options),
                                                      bool(options["OptimismSimple"]))));

    sync_cout << "evalraw psqt " << int(psqt) << " positional " << int(positional) << " bucket "
              << Eval::NNUE::FeatureSet::make_layer_stack_bucket(p) << " final " << finalEval
              << " fen " << p.fen() << sync_endl;
}

// Prints "info string rule <verdict>", what the rules of the search
// (Position::rule_judge()) decide in the current position, for the side to
// move: "none" (the game goes on), "draw", "win" or "loss". Covers the third
// occurrence of a position (draw, perpetual check or perpetual chase) and the
// 40-move rule (a draw, or a loss for a side that has no legal move). Referees
// (fishtest, trainer/match.py) ask it to apply the same rules as the engine.
void Engine::rule() {
    wait_for_search_finished();

    if (!positionValid)
    {
        sync_cout << "info string rule invalid" << sync_endl;
        return;
    }

    Value      result = VALUE_NONE;
    const bool ended  = pos.rule_judge(result, 0);

    sync_cout << "info string rule "
              << (!ended                  ? "none"
                  : result == VALUE_DRAW  ? "draw"
                  : result > VALUE_DRAW   ? "win"
                                          : "loss")
              << sync_endl;
}

const OptionsMap& Engine::get_options() const { return options; }
OptionsMap&       Engine::get_options() { return options; }

std::string Engine::fen() const { return pos.fen(); }

// Position::flip() sets up the flipped position, which must be one that the
// engine can handle (one king per side, ...): a position that is not (see
// Engine::set_position()) is left as it is
void Engine::flip() {
    if (std::string reason; !pos.pool_is_consistent(&reason))
    {
        sync_cout << "info string Invalid position: " << reason << ", not flipped" << sync_endl;
        return;
    }

    pos.flip();
}

std::string Engine::visualize() const {
    std::stringstream ss;
    ss << pos;
    return ss.str();
}

int Engine::get_hashfull(int maxAge) const { return tt.hashfull(maxAge); }

std::vector<std::pair<usize, usize>> Engine::get_bound_thread_count_by_numa_node() const {
    auto                                 counts = threads.get_bound_thread_count_by_numa_node();
    const NumaConfig&                    cfg    = numaContext.get_numa_config();
    std::vector<std::pair<usize, usize>> ratios;
    NumaIndex                            n = 0;
    for (; n < counts.size(); ++n)
        ratios.emplace_back(counts[n], cfg.num_cpus_in_numa_node(n));
    if (!counts.empty())
        for (; n < cfg.num_numa_nodes(); ++n)
            ratios.emplace_back(0, cfg.num_cpus_in_numa_node(n));
    return ratios;
}

std::string Engine::get_numa_config_as_string() const {
    return numaContext.get_numa_config().to_string();
}

std::string Engine::numa_config_information_as_string() const {
    auto cfgStr = get_numa_config_as_string();
    return "Available processors: " + cfgStr;
}

std::string Engine::thread_binding_information_as_string() const {
    auto              boundThreadsByNode = get_bound_thread_count_by_numa_node();
    std::stringstream ss;
    if (boundThreadsByNode.empty())
        return ss.str();

    bool isFirst = true;

    for (auto&& [current, total] : boundThreadsByNode)
    {
        if (!isFirst)
            ss << ":";
        ss << current << "/" << total;
        isFirst = false;
    }

    return ss.str();
}

std::string Engine::thread_allocation_information_as_string() const {
    std::stringstream ss;

    usize threadsSize = threads.size();
    ss << "Using " << threadsSize << (threadsSize > 1 ? " threads" : " thread");

    auto boundThreadsByNodeStr = thread_binding_information_as_string();
    if (boundThreadsByNodeStr.empty())
        return ss.str();

    ss << " with NUMA node thread binding: ";
    ss << boundThreadsByNodeStr;

    return ss.str();
}
}
