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

#ifndef PERFT_H_INCLUDED
#define PERFT_H_INCLUDED

#include <cstdint>

#include "movegen.h"
#include "position.h"
#include "types.h"
#include "uci.h"

namespace Maskfish::Benchmark {

// Utility to verify move generation. All the leaf nodes up
// to the given depth are generated and counted, and the sum is returned.
//
// Moving a dark piece reveals it. Such a move is a chance node: it is expanded
// into one child per kind of piece that can be revealed (each kind is counted
// once, regardless of how many of that kind are left in the pool). Without
// this the dark piece would stay hidden on a non-starting square and the check
// information of the child position would never be computed.
template<bool Root>
uint64_t perft(Position& pos, Depth depth) {

    StateInfo st;

    uint64_t   cnt, nodes = 0;
    const bool leaf = (depth == 2);

    auto count = [&]() { return leaf ? MoveList<LEGAL>(pos).size() : perft<false>(pos, depth - 1); };

    for (const auto& m : MoveList<LEGAL>(pos))
    {
        if (Root && depth <= 1)
            cnt = 1, nodes++;
        else
        {
            pos.do_move(m, st);

            if (pos.is_dark(m.to_sq()))
            {
                cnt = 0;
                for (const auto& [pc, num] : pos.rest_pieces(~pos.side_to_move()))
                {
                    Piece fromPc = pos.do_flip(m.to_sq(), pc);
                    cnt += count();
                    pos.undo_flip(m.to_sq(), fromPc);
                }
            }
            else
                cnt = count();

            nodes += cnt;
            pos.undo_move(m);
        }
        if (Root)
            sync_cout << UCIEngine::move(m) << ": " << cnt << sync_endl;
    }
    return nodes;
}

inline uint64_t perft(const std::string& fen, Depth depth) {
    StateListPtr states(new std::deque<StateInfo>(1));
    Position     p;
    p.set(fen, &states->back());

    // Move generation needs exactly one king per side (see Engine::set_position())
    if (p.count<KING>(WHITE) != 1 || p.count<KING>(BLACK) != 1)
        return 0;

    return perft<true>(p, depth);
}
}

#endif  // PERFT_H_INCLUDED
