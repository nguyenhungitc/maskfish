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

#ifndef ENGINE_H_INCLUDED
#define ENGINE_H_INCLUDED

#include <functional>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "misc.h"
#include "nnue/network.h"
#include "nnue/nnue_misc.h"
#include "numa.h"
#include "position.h"
#include "search.h"
#include "thread.h"
#include "tt.h"
#include "types.h"
#include "ucioption.h"

namespace Maskfish {

constexpr int MaxHashMB = Is64Bit ? 33554432 : 2048;
extern int    MaxThreads;

class Engine {
   public:
    using InfoShort = Search::InfoShort;
    using InfoFull  = Search::InfoFull;
    using InfoIter  = Search::InfoIteration;

    Engine(std::optional<std::filesystem::path> path = std::nullopt);

    // Cannot be movable due to components holding backreferences to fields
    Engine(const Engine&)            = delete;
    Engine(Engine&&)                 = delete;
    Engine& operator=(const Engine&) = delete;
    Engine& operator=(Engine&&)      = delete;

    ~Engine() { wait_for_search_finished(); }

    u64 perft(const std::string& fen, Depth depth);

    // non blocking call to start searching
    void go(Search::LimitsType&);
    // non blocking call to stop searching
    void stop();

    // blocking call to wait for search to finish
    void wait_for_search_finished();
    // set a new position, moves are in UCI format
    // Moves of dark pieces carry the revealed piece (see README). Invalid input
    // is reported with "info string" and the rest of the moves is ignored.
    void set_position(const std::string& fen, const std::vector<std::string>& moves);

    // modifiers

    bool set_numa_config_from_option(const std::string& o);
    void resize_threads();
    void set_tt_size(usize mb);
    void set_ponderhit(bool);
    void search_clear();

    void set_on_update_no_moves(std::function<void(const InfoShort&)>&&);
    void set_on_update_full(std::function<void(const InfoFull&)>&&);
    void set_on_iter(std::function<void(const InfoIter&)>&&);
    void set_on_bestmove(std::function<void(std::string_view, std::string_view)>&&);
    void set_on_start(std::function<void()>&&);
    void set_on_verify_networks(std::function<void(std::string_view)>&&);

    // network related

    void                                  verify_networks() const;
    std::unique_ptr<Eval::NNUE::Networks> get_default_networks() const;
    void                                  load_big_network(const std::string& file);
    void save_network(const std::pair<std::optional<std::string>, std::string> files);

    // utility functions

    void trace_eval() const;
    void eval_raw() const;

    const OptionsMap& get_options() const;
    OptionsMap&       get_options();

    int get_hashfull(int maxAge = 0) const;

    std::string                          fen() const;
    void                                 flip();
    std::string                          visualize() const;
    std::vector<std::pair<usize, usize>> get_bound_thread_count_by_numa_node() const;
    std::string                          get_numa_config_as_string() const;
    std::string                          numa_config_information_as_string() const;
    std::string                          thread_allocation_information_as_string() const;
    std::string                          thread_binding_information_as_string() const;

   private:
    const std::filesystem::path binaryDirectory;

    NumaReplicationContext numaContext;

    Position     pos;
    StateListPtr states;

    OptionsMap                                        options;
    ThreadPool                                        threads;
    TranspositionTable                                tt;
    LazyNumaReplicatedSystemWide<Eval::NNUE::Networks> networks;

    Search::SearchManager::UpdateContext  updateContext;
    std::function<void(std::string_view)> onVerifyNetworks;
};

}  // namespace Maskfish


#endif  // #ifndef ENGINE_H_INCLUDED
