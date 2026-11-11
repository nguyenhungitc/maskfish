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

#include "universal.h"

#include "embedded.h"

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

#if defined(__x86_64__)
#include <cpuid.h>
#elif defined(__aarch64__)
#include <sys/auxv.h>
#else
#error "the universal binary is for Linux x86-64 and arm64"
#endif

#ifndef MFD_EXEC
#define MFD_EXEC 0x0010U  // Linux 6.3: the memfd may be executed (without it newer kernels log a warning)
#endif

extern char** environ;

namespace jq::universal {

namespace {

// Whether the CPU (and the OS, for the AVX state) runs a build. The names and their compiler flags are those of
// CMakeLists.txt; the features checked are those the flags let the compiler use.
#if defined(__x86_64__)
struct Cpu {
    bool v2 = false, v3 = false, avxvnni = false, v4 = false, icl = false;
};

Cpu detect() {
    Cpu r;
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d)) return r;
    const unsigned c1 = c;
    unsigned b7 = 0, c7 = 0, a71 = 0;
    if (__get_cpuid_max(0, nullptr) >= 7) {
        __cpuid_count(7, 0, a, b, c, d);
        b7 = b, c7 = c;
        __cpuid_count(7, 1, a, b, c, d);
        a71 = a;
    }
    unsigned c81 = 0;
    if (__get_cpuid(0x80000001, &a, &b, &c, &d)) c81 = c;
    unsigned long long xcr0 = 0;
    if (c1 >> 27 & 1) {  // OSXSAVE: the OS saves the extended state, XGETBV says which
        unsigned lo, hi;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        xcr0 = (unsigned long long)hi << 32 | lo;
    }
    auto bit = [](unsigned x, int k) { return (x >> k & 1) != 0; };
    const bool ymm = (xcr0 & 0x6) == 0x6, zmm = (xcr0 & 0xE6) == 0xE6;
    // x86-64-v2: SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT, CMPXCHG16B, LAHF/SAHF
    r.v2 = bit(c1, 0) && bit(c1, 9) && bit(c1, 19) && bit(c1, 20) && bit(c1, 23) && bit(c1, 13) && bit(c81, 0);
    // x86-64-v3: AVX, AVX2, BMI1, BMI2, F16C, FMA, LZCNT, MOVBE
    r.v3 = r.v2 && ymm && bit(c1, 28) && bit(b7, 5) && bit(b7, 3) && bit(b7, 8) && bit(c1, 29) && bit(c1, 12) &&
           bit(c81, 5) && bit(c1, 22);
    r.avxvnni = r.v3 && bit(a71, 4);
    // x86-64-v4: AVX-512 F, BW, CD, DQ, VL
    r.v4 = r.v3 && zmm && bit(b7, 16) && bit(b7, 30) && bit(b7, 28) && bit(b7, 17) && bit(b7, 31);
    // Ice Lake and later, Zen 4 and later: x86-64-v4 with AVX-512 VNNI and VBMI2
    r.icl = r.v4 && bit(c7, 11) && bit(c7, 6);
    return r;
}

bool supported(const char* name) {
    static const Cpu cpu = detect();
    if (!std::strcmp(name, "x86-64")) return true;
    if (!std::strcmp(name, "x86-64-v2")) return cpu.v2;
    if (!std::strcmp(name, "x86-64-v3")) return cpu.v3;
    if (!std::strcmp(name, "x86-64-avxvnni")) return cpu.avxvnni;
    if (!std::strcmp(name, "x86-64-v4")) return cpu.v4;
    if (!std::strcmp(name, "x86-64-avx512icl")) return cpu.icl;
    return false;
}
#else
bool supported(const char* name) {
    if (!std::strcmp(name, "armv8")) return true;
    if (!std::strcmp(name, "armv8-dotprod")) {
        // armv8.2-a+dotprod: the dot product instructions, and the LSE atomics and RDM of ARMv8.1
        const unsigned long hw = getauxval(AT_HWCAP);
        return (hw & (1UL << 20)) && (hw & (1UL << 8)) && (hw & (1UL << 12));  // ASIMDDP, ATOMICS, ASIMDRDM
    }
    return false;
}
#endif

// Executes the embedded build s in this process; returns only if it could not be started.
void run(const Slice& s, char** argv) {
    int fd = memfd_create(s.name, MFD_CLOEXEC | MFD_EXEC);
    if (fd < 0 && errno == EINVAL) fd = memfd_create(s.name, MFD_CLOEXEC);  // kernels before 6.3
    if (fd < 0) {
        std::fprintf(stderr, "maskfish: memfd_create failed (%s)\n", std::strerror(errno));
        return;
    }
    for (const unsigned char* p = s.begin; p < s.end;) {
        const ssize_t n = write(fd, p, std::size_t(s.end - p));
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            std::fprintf(stderr, "maskfish: cannot write the %s build (%s)\n", s.name, std::strerror(errno));
            close(fd);
            return;
        }
        p += n;
    }
    // The embedded network, once in the binary, goes to the build in a memfd it inherits (src/embedded_fd.cpp).
    int netFd = -1;
    if (const EmbeddedNet e = embedded_net(); e.size) {
        netFd = memfd_create("maskfish.nnue", 0);
        bool ok = netFd >= 0;
        for (const uint8_t* p = e.data; ok && p < e.data + e.size;) {
            const ssize_t n = write(netFd, p, std::size_t(e.data + e.size - p));
            if (n < 0 && errno == EINTR) continue;
            ok = n > 0;
            if (ok) p += n;
        }
        if (ok) setenv(EMBEDDED_NET_FD_ENV, std::to_string(netFd).c_str(), 1);
        else if (netFd >= 0) close(netFd), netFd = -1;
    }
    // The build runs from memory: it finds the files next to the binary (the network) through this path.
    char self[PATH_MAX];
    const ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
    if (n > 0) {
        self[n] = 0;
        setenv("MASKFISH_UNIVERSAL_EXE", self, 1);
    }
    // By its /proc path first: tools that follow exec (Intel SDE, Pin) follow this one; then by the descriptor.
    char path[64];
    std::snprintf(path, sizeof path, "/proc/self/fd/%d", fd);
    execve(path, argv, environ);
    fexecve(fd, argv, environ);
    std::fprintf(stderr, "maskfish: cannot start the %s build (%s)\n", s.name, std::strerror(errno));
    close(fd);
    if (netFd >= 0) close(netFd);
    unsetenv(EMBEDDED_NET_FD_ENV);
    unsetenv("MASKFISH_UNIVERSAL_EXE");
}

}  // namespace

void dispatch(char** argv) {
    const char* want = std::getenv("MASKFISH_UNIVERSAL_ARCH");
    if (want && !*want) want = nullptr;
    int best = -1;  // index in kSlices, -1 = the baseline
    for (int i = 0; i < kNumSlices; ++i)
        if (supported(kSlices[i].name)) best = i;
    if (want && !std::strcmp(want, "list")) {
        std::printf("%-18s this CPU\n", "build");
        std::printf("%-18s yes%s\n", kBaseline, best < 0 ? "       selected" : "");
        for (int i = 0; i < kNumSlices; ++i)
            std::printf("%-18s %s\n", kSlices[i].name,
                        !supported(kSlices[i].name) ? "no" : i == best ? "yes       selected" : "yes");
        std::exit(0);
    }
    if (want) {
        if (!std::strcmp(want, kBaseline)) return;
        best = -2;
        for (int i = 0; i < kNumSlices; ++i)
            if (!std::strcmp(want, kSlices[i].name)) best = i;
        if (best == -2) {
            std::fprintf(stderr, "maskfish: no build %s in this binary (MASKFISH_UNIVERSAL_ARCH=list lists them)\n", want);
            std::exit(1);
        }
        if (!supported(want)) {
            std::fprintf(stderr, "maskfish: this CPU does not run the %s build\n", want);
            std::exit(1);
        }
    }
    if (best < 0) return;
    run(kSlices[best], argv);
    std::fprintf(stderr, "maskfish: running the %s build instead\n", kBaseline);
}

}  // namespace jq::universal
