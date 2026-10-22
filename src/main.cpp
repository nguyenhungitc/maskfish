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
#include <memory>
#include <utility>

#include "attacks.h"
#include "misc.h"
#include "nnue/features/jieqi_threats.h"
#include "position.h"
#include "tune.h"
#include "uci.h"

using namespace Maskfish;

#ifdef UNIVERSAL_BINARY
// One build of a universal binary (see universal/select.h): Maskfish is
// defined to Maskfish_<arch>, and the entry point selects the main() of a
// build.
namespace Maskfish {

int main(int argc, char* argv[]);  // silence 'no previous declaration'

__attribute__((used))  // keep main alive
#endif

int main(int argc, char* argv[]) {
    std::cout << engine_info() << std::endl;

    Attacks::init();
    Position::init();
#ifdef JIEQI_NNUE_THREATS
    Eval::NNUE::Features::JieqiThreats::init();
#endif

    auto cli = CommandLine(argc, argv);
    auto uci = std::make_unique<UCIEngine>(std::move(cli));

    Tune::init(uci->engine_options());

    uci->loop();

    return 0;
}

#ifdef UNIVERSAL_BINARY
}  // namespace Maskfish

    #ifdef UNIVERSAL_NEEDS_MAIN_SHIM
int main(int argc, char* argv[]) { return Maskfish::main(argc, argv); }
    #endif
#endif
