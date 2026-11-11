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

// The network evaluation: the HNN evaluator of src/hnn/ (hnnue.h: inputs, head and file format), trained by
// train/train.py. EvalFile takes a raw JQHNNUE1 net (.hnn) or a JQHNNL01 / JQHNNZ01 compressed one (.nnue),
// recognised by its magic. Without a net the search uses the classical evaluation (eval.h).
#pragma once
#include "hnn/hnnue.h"
#include "position.h"

#include <string>

namespace jq::nnue {

struct Accumulator {
    hnn::Acc hnn;  // int16 accumulators, PSQT, material and no-capture count
};

// Feature changes caused by one move (computed from the position before the move). A default Dirty is a null
// move. A king that changes bucket or mirror is refreshed inside hnn::update, from its per-thread cache.
struct Dirty {
    hnn::Delta hd;
};

bool load(const std::string& path, std::string* error = nullptr);
bool load_buffer(const uint8_t* data, std::size_t size, std::string* error = nullptr);  // a net file's bytes
void unload();
void set_flush_denormals();  // call once per thread that evaluates
bool loaded();
std::string net_desc();  // one line about the loaded net, for the UCI info string

// The value carries the 80-ply damping and a material scale already (hnn::runtime_value), so the search does
// not shrink it again (Rule40Div applies to the classical evaluation only).
void refresh(const Position& pos, Accumulator& acc);
void make_dirty(const Position& pos, Move m, Dirty& d);  // m carries the reveal for a dark mover
void update(const Accumulator& prev, Accumulator& next, const Dirty& d);
Value evaluate(const Accumulator& acc, Color stm);
Value evaluate(const Position& pos);  // full refresh; for tools and the root

}  // namespace jq::nnue
