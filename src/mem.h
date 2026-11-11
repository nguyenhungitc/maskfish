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

// Aligned allocation for the large tables.
#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstring>
#if defined(__linux__)
#include <sys/mman.h>
#endif

// Aligned blocks for the HNN evaluator (src/hnn/). The size is rounded up to a multiple of the alignment, which
// C11 requires of aligned_alloc.
inline void* jq_aligned_alloc(std::size_t align, std::size_t size) {
    return std::aligned_alloc(align, (size ? size + align - 1 : align) / align * align);
}
inline void jq_aligned_free(void* p) { std::free(p); }

namespace jq {

// Zeroed block of at least `bytes`, aligned to 64. A block of half a megabyte or more is aligned to 2 MB, padded to a
// multiple of 2 MB and offered to the kernel as transparent huge pages (Linux keeps them in "madvise" mode by
// default): a probe into a table of hundreds of megabytes then costs a cache miss, not a cache miss plus a page walk.
// Release with std::free.
inline void* alloc_large(std::size_t bytes) {
    constexpr std::size_t HUGE = std::size_t(2) << 20;
    const bool big = bytes >= HUGE / 4;
    const std::size_t align = big ? HUGE : 64;
    const std::size_t size = ((bytes ? bytes : 1) + align - 1) / align * align;
    void* p = std::aligned_alloc(align, size);
    if (!p) return nullptr;
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    if (big) madvise(p, size, MADV_HUGEPAGE);
#endif
    std::memset(p, 0, size);
    return p;
}

}  // namespace jq
