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

#ifndef TT_H_INCLUDED
#define TT_H_INCLUDED

#include <cstddef>
#include <cstdint>
#include <tuple>

#include "memory.h"
#include "types.h"

namespace Maskfish {

class ThreadPool;
struct Cluster;

// There is only one global hash table for the engine and all its threads. For chess in particular, we even allow racy
// updates between threads to and from the TT, as taking the time to synchronize access would cost thinking time and
// thus elo. As a hash table, collisions are possible and may cause chess playing issues (bizarre blunders, faulty mate
// reports, etc). Fixing these also loses elo; however such risk decreases quickly with larger TT size.
//
// `probe` is the primary method: given a board position, we lookup its entry in the table, and return a tuple of:
//   1) whether the entry already has this position
//   2) a copy of the prior data (if any) (may be inconsistent due to read races)
//   3) a writer object to this entry
// The copied data and the writer are separated to maintain clear boundaries between local vs global objects.


// A copy of the data already in the entry (possibly collided). `probe` may be racy, resulting in inconsistent data.
struct TTData {
    Move  move;
    Value value, eval;
    Depth depth;
    Bound bound;
    bool  is_pv;

    TTData() = delete;
    TTData(Move m, Value v, Value ev, Depth d, Bound b, bool pv) :
        move(m),
        value(v),
        eval(ev),
        depth(d),
        bound(b),
        is_pv(pv) {};
};


// This is used to make racy writes to the global TT.
struct TTWriter {
   public:
    void write(Key k, Value v, bool pv, Bound b, Depth d, Move m, Value ev, uint8_t generation8);
    // Lowers the depth of the entry, so that it is replaced sooner
    void penalize(int penalty);

   private:
    friend class TranspositionTable;
    Cluster* cluster;
    int      index;
    bool     decisiveAging;
    uint64_t keyMask;
    TTWriter(Cluster* cl, int idx, bool decisiveAging, uint64_t keyMask);
};


class TranspositionTable {

   public:
    ~TranspositionTable() { aligned_large_pages_free(table); }

    void resize(size_t mbSize, ThreadPool& threads);  // Set TT size
    void clear(ThreadPool& threads);                  // Re-initialize memory, multithreaded
    int  hashfull(int maxAge = 0)
      const;  // Approximate what fraction of entries (permille) have been written to during this root search

    void
    new_search();  // This must be called at the beginning of each root search to track entry aging
    uint8_t generation() const;  // The current age, used when writing new data to the TT
    std::tuple<bool, TTData, TTWriter>
    probe(const Key key) const;  // The main method, whose retvals separate local vs global objects
    Cluster* get_cluster(const Key key)
      const;  // This is the hash function; its only external use is memory prefetching.

    // Restrict the secondary aging of entries to decisive values (UCI option
    // DecisiveTTAging)
    void set_decisive_aging(bool b) { decisiveAging = b; }

    // Key bits compared inside a cluster: 21 with the UCI option TTKey21,
    // otherwise 16 (the entries keep 21-bit fields either way)
    void set_key_bits(int bits) { keyMask = (uint64_t(1) << bits) - 1; }

   private:
    size_t   clusterCount;
    Cluster* table = nullptr;

    uint8_t generation8 = 0;  // Size must be not bigger than TTEntry::genBound8

    bool     decisiveAging = false;
    uint64_t keyMask       = 0xFFFF;
};

}  // namespace Maskfish

#endif  // #ifndef TT_H_INCLUDED
