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

#include "nnue.h"

#include <cstdint>
#include <cstdio>
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#endif

namespace jq::nnue {

bool load(const std::string& path, std::string* error) {
    if (FILE* f = std::fopen(path.c_str(), "rb")) std::fclose(f);
    else {
        if (error) *error = "cannot open the file";
        return false;
    }
    if (!hnn::is_hnn_file(path)) {
        if (error) *error = "not a network this engine reads (JQHNNUE1, JQHNNL01 or JQHNNZ01)";
        return false;
    }
    if (hnn::load(path, error)) return true;
    hnn::unload();  // the loader may have stopped part-way: no net is usable now
    return false;
}

bool load_buffer(const uint8_t* data, std::size_t size, std::string* error) {
    if (hnn::load_buffer(data, size, error)) return true;
    hnn::unload();
    return false;
}

void unload() { hnn::unload(); }
bool loaded() { return hnn::loaded(); }
// Denormals flushed to zero for this thread: FTZ | DAZ of MXCSR on x86, FZ of FPCR on arm64.
void set_flush_denormals() {
#if defined(__x86_64__) || defined(__i386__)
    _mm_setcsr(_mm_getcsr() | 0x8040);
#elif defined(__aarch64__)
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    __asm__ volatile("msr fpcr, %0" : : "r"(fpcr | (uint64_t(1) << 24)));
#endif
}

std::string net_desc() {
    if (!hnn::loaded()) return "none";
    const hnn::LoadInfo& li = hnn::last_load();
    return "HNN width " + std::to_string(hnn::width()) + ", king buckets scheme " + std::to_string(hnn::kb_scheme()) +
           ", head " + hnn::head_desc() + ", " + (li.fmt == "raw" ? std::string("raw") : "compressed " + li.fmt) + " " +
           std::to_string(li.fileBytes) + " bytes, md5 " + li.rawMd5;
}

void refresh(const Position& pos, Accumulator& acc) { hnn::refresh(pos, acc.hnn); }
void make_dirty(const Position& pos, Move m, Dirty& d) { hnn::make_delta(pos, m, d.hd); }
void update(const Accumulator& prev, Accumulator& next, const Dirty& d) { hnn::update(prev.hnn, next.hnn, d.hd); }
Value evaluate(const Accumulator& acc, Color stm) { return hnn::evaluate(acc.hnn, stm); }
int evaluate_base(const Accumulator& acc, Color stm) { return hnn::evaluate_base(acc.hnn, stm); }
Value value_at(int base, int noCapture) { return hnn::value_at(base, noCapture); }

Value evaluate(const Position& pos) {
    static thread_local Accumulator acc;
    refresh(pos, acc);
    return evaluate(acc, pos.stm);
}

}  // namespace jq::nnue
