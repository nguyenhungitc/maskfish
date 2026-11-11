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

// embedded_net() of a build of a universal binary: the binary embeds the network once and passes it in a memfd
// whose descriptor is in MASKFISH_UNIVERSAL_NET_FD (src/universal.cpp). The first call reads it and closes the
// descriptor (main() calls it at startup).

#include "embedded.h"

#include <cstdlib>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace jq {

EmbeddedNet embedded_net() {
    static std::vector<uint8_t> buf;
    static bool done = false;
    if (!done) {
        done = true;
        if (const char* e = std::getenv(EMBEDDED_NET_FD_ENV); e && *e) {
            const int fd = std::atoi(e);
            struct stat st;
            if (fd > 2 && fstat(fd, &st) == 0 && st.st_size > 0) {
                buf.resize(std::size_t(st.st_size));
                std::size_t got = 0;
                while (got < buf.size()) {
                    const ssize_t n = pread(fd, buf.data() + got, buf.size() - got, off_t(got));
                    if (n <= 0) break;
                    got += std::size_t(n);
                }
                if (got != buf.size()) buf.clear();
            }
            if (fd > 2) close(fd);
            unsetenv(EMBEDDED_NET_FD_ENV);
        }
    }
    return EmbeddedNet{buf.empty() ? nullptr : buf.data(), buf.size()};
}

}  // namespace jq
