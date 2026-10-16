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
// once, regardless of how many of that kind are left in the pool), at every
// ply including the last one. Without this the dark piece would stay hidden on
// a non-starting square and the check information of the child position would
// never be computed.

// Number of children of a move: the kinds of piece that a move of a dark piece
// can reveal, 1 for any other move
inline u64 perft_children(const Position& pos, Move m) {
    if (!pos.move_dark(m))
        return 1;

    Position::RestPieceList restPieces;
    return u64(pos.rest_pieces(pos.side_to_move(), restPieces));
}

template<bool Root>
u64 perft(Position& pos, Depth depth) {

    StateInfo st;

    u64        cnt, nodes = 0;
    const bool leaf = (depth == 2);

    auto count = [&]() {
        if (!leaf)
            return perft<false>(pos, depth - 1);

        u64 n = 0;
        for (const auto& m : MoveList<LEGAL>(pos))
            n += perft_children(pos, m);
        return n;
    };

    for (const auto& m : MoveList<LEGAL>(pos))
    {
        if (Root && depth <= 1)
            cnt = perft_children(pos, m), nodes += cnt;
        else
        {
            [[maybe_unused]] const bool mayCheck = pos.gives_check(m);

            pos.do_move(m, st);

            if (pos.is_dark(m.to_sq()))
            {
                [[maybe_unused]] bool anyCheck = false;

                cnt = 0;
                for (const auto& [pc, num] : pos.rest_pieces(~pos.side_to_move()))
                {
                    Piece fromPc = pos.do_flip(m.to_sq(), pc);
                    anyCheck |= bool(pos.checkers());
                    cnt += count();
                    pos.undo_flip(m.to_sq(), fromPc);
                }

                // gives_check() of a move of a dark piece: some reveal checks
                assert(mayCheck == anyCheck);
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

inline u64 perft(const std::string& fen, Depth depth) {
    StateInfo st;
    Position  p;
    p.set(fen, &st);

    // Move generation needs a playable board (see Position::is_playable())
    if (!p.is_playable())
        return 0;

    return perft<true>(p, depth);
}
}

#endif  // PERFT_H_INCLUDED
