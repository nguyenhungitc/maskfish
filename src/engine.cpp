/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "engine.h"

#include <algorithm>
#include <cassert>
#include <deque>
#include <iosfwd>
#include <memory>
#include <ostream>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

#include "evaluate.h"
#include "misc.h"
#include "nnue/network.h"
#include "nnue/nnue_accumulator.h"
#include "nnue/features/half_ka_v2_hm.h"
#include "score.h"
#include "nnue/nnue_common.h"
#include "nnue/nnue_misc.h"
#include "numa.h"
#include "perft.h"
#include "position.h"
#include "search.h"
#include "shm.h"
#include "types.h"
#include "uci.h"
#include "ucioption.h"

namespace Stockfish {

namespace NN = Eval::NNUE;

constexpr auto StartFEN =
  "xxxxkxxxx/9/1x5x1/x1x1x1x1x/9/9/X1X1X1X1X/1X5X1/9/XXXXKXXXX w R2A2C2P5N2B2r2a2c2p5n2b2 0 1";
constexpr int MaxHashMB  = Is64Bit ? 33554432 : 2048;
int           MaxThreads = std::max(1024, 4 * int(get_hardware_concurrency()));

Engine::Engine(std::optional<std::string> path) :
    binaryDirectory(path ? CommandLine::get_binary_directory(*path) : ""),
    numaContext(NumaConfig::from_system()),
    states(new std::deque<StateInfo>(1)),
    threads(),
    networks(numaContext, get_default_networks()) {

    pos.set(StartFEN, &states->back());

    options.add(  //
      "Debug Log File", Option("", [](const Option& o) {
          start_logger(o);
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

    // Jieqi: bonus for safe reveals (quiet moves of dark pieces). The defaults
    // are the values used by AB-JChess; set them all to 0 to disable.
    options.add("RevealQuietBase", Option(-12, -96, 96));
    options.add("RevealQuietPhase", Option(0, -96, 96));
    options.add("RevealQuietSafety", Option(24, -192, 192));
    options.add("RevealQuietHighValue", Option(-12, -96, 96));
    options.add("RevealQuietDiversity", Option(0, -96, 96));

    // Experimental (ideas from AB-JChess), off by default: test with SPRT.
    // ProbCut asks a larger margin (+64) and one ply less for dark captures.
    options.add("ProbCutDarkCapture", Option(false));
    // Lower the depth of TT entries that are deep enough but cannot cut.
    options.add("TTPenalize", Option(false));
    // Search the less likely outcomes of a reveal one ply shallower.
    options.add("ChanceBranchDepth", Option(false));

    // Experimental search ideas ported from Pikafish, off by default: test
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
    // depth and skip singular extensions, to find the mate faster.
    options.add("SeekMate", Option(false));
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
    // Share the continuation, pawn and correction histories between the
    // threads of a NUMA node (only matters with Threads > 1). Changing it
    // recreates the threads.
    options.add("SharedHistory", Option(false, [this](const Option&) {
                    resize_threads();
                    return std::nullopt;
                }));
    // Count dark pieces at their expected value in the material scaling of
    // the evaluation.
    options.add("JieqiMaterial", Option(false));
    // Score -> win rate scale used to average chance nodes (0 = built-in
    // 360.83524). trainer/fit_scaling.py fits it from self-play data.
    options.add("ChanceScaling", Option(0, 0, 3000));
    // Risk preference of the side causing a chance event, in percent of the
    // standard deviation of the outcomes' win rates: > 0 avoids uncertain
    // reveals and captures, < 0 seeks them.
    options.add("ChanceRiskAversion", Option(0, -100, 100));

    options.add(  //
      "EvalFile", Option(EvalFileDefaultNameBig, [this](const Option& o) {
          load_big_network(o);
          return std::nullopt;
      }));

    threads.clear();
    threads.ensure_network_replicated();
    resize_threads();
}

std::uint64_t Engine::perft(const std::string& fen, Depth depth) {
    verify_networks();

    return Benchmark::perft(fen, depth);
}

void Engine::go(Search::LimitsType& limits) {
    assert(limits.perft == 0);
    verify_networks();

    threads.start_thinking(pos, states, limits);
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

void Engine::set_on_verify_networks(std::function<void(std::string_view)>&& f) {
    onVerifyNetworks = std::move(f);
}

void Engine::wait_for_search_finished() { threads.main_thread()->wait_for_search_finished(); }

void Engine::set_position(const std::string& fen, const std::vector<std::string>& moves) {
    constexpr std::string_view PieceToChar(" RACPNBK racpnbkXx");

    // Drop the old state and create a new one
    states = StateListPtr(new std::deque<StateInfo>(1));
    pos.set(fen, &states->back());

    if (std::string reason; !pos.pool_is_consistent(&reason))
        sync_cout << "info string Invalid position: " << reason << sync_endl;

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
            break;

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
                      << ", ignoring this and the following moves" << sync_endl;
            break;
        }

        states->emplace_back();
        pos.do_move(m, states->back());

        if (moveDark)
            pos.do_flip(m.to_sq(), flipped);

        if (captured != NO_PIECE)
            pos.reveal_captured(captured);
    }
}

// modifiers

bool Engine::set_numa_config_from_option(const std::string& o) {
    if (o == "auto" || o == "system")
    {
        numaContext.set_numa_config(NumaConfig::from_system());
    }
    else if (o == "hardware")
    {
        // Don't respect affinity set in the system.
        numaContext.set_numa_config(NumaConfig::from_system(false));
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
    threads.set(numaContext.get_numa_config(), {options, threads, tt, networks}, updateContext);

    // Reallocate the hash with the new threadpool size
    set_tt_size(options["Hash"]);
    threads.ensure_network_replicated();
}

void Engine::set_tt_size(size_t mb) {
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

    verify_networks();

    // Heap-allocated: the accumulator stack is large
    auto accumulators = std::make_unique<Eval::NNUE::AccumulatorStack>();
    auto caches       = std::make_unique<Eval::NNUE::AccumulatorCaches>(*networks);
    auto [psqt, positional] = networks->big.evaluate(p, *accumulators, &caches->big);

    sync_cout << "evalraw psqt " << int(psqt) << " positional " << int(positional) << " bucket "
              << Eval::NNUE::FeatureSet::make_layer_stack_bucket(p) << " fen " << p.fen()
              << sync_endl;
}

const OptionsMap& Engine::get_options() const { return options; }
OptionsMap&       Engine::get_options() { return options; }

std::string Engine::fen() const { return pos.fen(); }

void Engine::flip() { pos.flip(); }

std::string Engine::visualize() const {
    std::stringstream ss;
    ss << pos;
    return ss.str();
}

int Engine::get_hashfull(int maxAge) const { return tt.hashfull(maxAge); }

std::vector<std::pair<size_t, size_t>> Engine::get_bound_thread_count_by_numa_node() const {
    auto                                   counts = threads.get_bound_thread_count_by_numa_node();
    const NumaConfig&                      cfg    = numaContext.get_numa_config();
    std::vector<std::pair<size_t, size_t>> ratios;
    NumaIndex                              n = 0;
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

    size_t threadsSize = threads.size();
    ss << "Using " << threadsSize << (threadsSize > 1 ? " threads" : " thread");

    auto boundThreadsByNodeStr = thread_binding_information_as_string();
    if (boundThreadsByNodeStr.empty())
        return ss.str();

    ss << " with NUMA node thread binding: ";
    ss << boundThreadsByNodeStr;

    return ss.str();
}
}
