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

// Alpha-beta search with chance nodes for face-down movers.
#pragma once
#include "position.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace jq {

struct Limits {
    int depth = 0;
    int64_t nodes = 0;
    int movetime = 0;          // ms
    int time[2] = {0, 0};      // ms remaining per colour
    int inc[2] = {0, 0};
    int movestogo = 0;
    bool infinite = false;
    bool ponder = false;       // go ponder: no time limit applies until ponderhit
    int mate = 0;              // go mate N: stop at a mate in at most N moves (depth capped at 2 N plies)
    std::vector<Move> searchMoves;  // root restricted to these moves when not empty
    int multiPV = 1;                // root lines reported per iteration (capped by the number of root moves)
};

struct SearchResult {
    Move best = MOVE_NONE;
    Value score = VALUE_NONE;
    int depth = 0;
    uint64_t nodes = 0;
    int timeMs = 0;
    std::vector<Move> pv;
};

// Tunable integer parameters exposed as UCI spin options (for SPSA later).
struct Param {
    const char* name;
    int* value;
    int lo, hi;
};
std::vector<Param>& params();

class Searcher {
public:
    explicit Searcher(int threads = 1, std::size_t hashMb = 16);
    ~Searcher();
    Searcher(const Searcher&) = delete;
    Searcher& operator=(const Searcher&) = delete;
    void set_threads(int n);
    int threads() const;
    std::size_t resize_hash(std::size_t mb);  // returns the size the table got, in bytes (smaller when memory is short)
    void clear();  // histories and transposition table
    void stop();   // may be called from another thread
    // Clears the stop flag before go() is started on another thread (so an early stop() holds). pondering: the
    // coming go() is a `go ponder`, so a ponderhit that arrives before it starts must not be lost either.
    void prepare(bool pondering = false);
    void ponderhit();  // the pondered move was played: the clock of the search starts now (from another thread)
    // Runs synchronously; `info` receives UCI info lines from the main thread. With resetStop = false the
    // caller must have called prepare().
    SearchResult go(const Position& root, const Limits& limits, const std::function<void(const std::string&)>& info,
                    bool resetStop = true);
    int hashfull() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

namespace search {

void init();  // tables and LMR (call again after changing LMR parameters)
uint64_t perft(Position& pos, int depth);   // base moves (one representative reveal per dark move)
double perft_chance(Position& pos, int depth, uint64_t& leaves);  // expands reveals, returns probability mass
// Static exchange evaluation of a pseudo-legal base move: does the side to move net >= threshold on its
// target square? Face-down movers and victims count at their side's expected identity value.
bool see_ge(const Position& pos, Move m, int threshold);
int see_value(const Position& pos, Move m);  // exact SEE value (search over thresholds)
// A face-down piece of the side to move on s that the opponent attacks. Returns 0 if s is not attacked, 1 if it is
// attacked and 2 if the ThreatRev options count it as threatened; *loss is the opponent's expected gain from the
// exchange on s, over the identities the piece may have. darkV: both sides' urn means, if the caller has them.
int threat_reveal(const Position& pos, int s, int* loss, const int* darkV = nullptr);

}  // namespace search
}  // namespace jq
