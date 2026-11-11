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

#include <iostream>
#ifdef MASKFISH_UNIVERSAL
#include "universal.h"
#endif
#ifdef MASKFISH_UNIVERSAL_SLICE
#include "embedded.h"
#endif

namespace jq {
int uci_loop(int argc, char** argv);
}

int main(int argc, char** argv) {
#ifdef MASKFISH_UNIVERSAL
    jq::universal::dispatch(argv);  // the best build for the CPU; returns when this one (the baseline) is to run
#endif
#ifdef MASKFISH_UNIVERSAL_SLICE
    jq::embedded_net();  // takes the network the universal binary passed, and closes its descriptor
#endif
    std::ios::sync_with_stdio(false);
    return jq::uci_loop(argc, argv);
}
