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

#ifndef UNIVERSAL_SELECT_H_INCLUDED
#define UNIVERSAL_SELECT_H_INCLUDED

// Universal binary (make ARCH=x86-64-universal, arm64-universal): one build
// of the engine per instruction set, each in its own namespace
// (Maskfish_<arch>), with its static initializers collected in the section
// <arch>_init instead of .init_array, so that only those of the selected
// build run. main() selects the best build for the CPU.
//
// The environment variable MASKFISH_UNIVERSAL_ARCH=<arch> (e.g. x86-64-avx2)
// selects a build by name instead, for testing: it must be supported by the
// CPU. MASKFISH_UNIVERSAL_ARCH=list prints the builds of this binary.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace {

struct Build {
    const char* name;
    int (*entry)(int, char**);
};

// The build named by MASKFISH_UNIVERSAL_ARCH, if any. Exits after "list" or
// an unknown name.
template<int N>
const Build* forced_build(const Build (&builds)[N]) {
    const char* name = getenv("MASKFISH_UNIVERSAL_ARCH");
    if (!name || !*name)
        return nullptr;
    for (const Build& b : builds)
        if (!strcmp(b.name, name))
            return &b;
    if (strcmp(name, "list"))
        fprintf(stderr, "MASKFISH_UNIVERSAL_ARCH: unknown build '%s'\n", name);
    fprintf(stderr, "builds of this binary:");
    for (const Build& b : builds)
        fprintf(stderr, " %s", b.name);
    fprintf(stderr, "\n");
    exit(strcmp(name, "list") ? EXIT_FAILURE : EXIT_SUCCESS);
}

}  // namespace

#endif  // #ifndef UNIVERSAL_SELECT_H_INCLUDED
