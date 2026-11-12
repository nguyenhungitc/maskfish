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

#include "embedded.h"
#include "eval.h"
#include "nnue.h"
#include "search.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits.h>
#include <unistd.h>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace jq {

void selfplay(std::istringstream& is);

namespace {

std::mutex outMutex;
void out(const std::string& s) {
    std::lock_guard<std::mutex> lock(outMutex);
    std::cout << s << std::endl;
}

const char* BENCH_FENS[] = {
    "xxxxkxxxx/9/1x5x1/x1x1x1x1x/9/9/X1X1X1X1X/1X5X1/9/XXXXKXXXX w R2A2C2P5N2B2r2a2c2p5n2b2 0 1",
    "xxxxkxxxx/9/1x5x1/x1x1x1x1x/9/4P4/X1X3X1X/1X5X1/9/XXXXKXXXX b R2A2C2P4N2B2r2a2c2p5n2b2 1 1",
    "r1xxkxxxx/9/1x5c1/x1x1x1x1x/9/9/X1X1X1X1X/1C5X1/9/XXXXKXXXX w R2A2C1P5N2B2r1a2c1p5n2b2 2 2",
    "2xakxxxx/4r4/1x5c1/x1x1x1x1p/9/2P6/X3X1X1X/1C5X1/9/XXXXKXXXX w R2A2C1P4N2B2r1a1c1p4n2b2 0 4",  // was X1X1X1X1X: 17 red pieces
    "4k4/9/9/9/9/9/9/9/4A4/3RK4 w - 0 1",
    "3ak4/4a4/9/9/9/9/9/9/9/3RK4 w - 0 1",
};

// The default value of EvalFile. The network is, in this order: the file EvalFile names (a relative path in the
// current directory, else next to the engine), then the network embedded in the binary (src/embedded.h) when
// EvalFile is empty, is the default name and no such file exists, or names a file that cannot be loaded. Without
// either the engine searches with the classical evaluation and reports an error.
constexpr const char* DEFAULT_NET = "maskfish.nnue";

#ifndef MASKFISH_BUILD
#define MASKFISH_BUILD "unknown"
#endif
// The compiler and the instruction set of this build (`compiler`).
std::string build_info() {
#if defined(__clang__)
    std::string c = "clang " __clang_version__;
#elif defined(__GNUC__)
    std::string c = "g++ " __VERSION__;
#else
    std::string c = "unknown";
#endif
    std::string b = MASKFISH_BUILD;
#if defined(MASKFISH_UNIVERSAL) || defined(MASKFISH_UNIVERSAL_SLICE)
    b += " (universal binary; MASKFISH_UNIVERSAL_ARCH=list lists its builds)";
#endif
    const EmbeddedNet e = embedded_net();
#ifdef MASKFISH_BITBOARD
    b += ", bitboard move generation";
#endif
    return "compiler " + c + "\nbuild " + b + "\nembedded network " +
           (e.size ? std::string(DEFAULT_NET) + ", " + std::to_string(e.size) + " bytes" : std::string("none"));
}

std::string exe_dir() {  // directory of the running executable with a trailing '/', "" if unknown
    std::string p;
    // A build of a universal binary runs from memory (src/universal.cpp): the binary gives its own path.
    if (const char* u = std::getenv("MASKFISH_UNIVERSAL_EXE"); u && *u) p = u;
    else {
        char buf[PATH_MAX];
        const ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
        if (n <= 0) return "";
        p.assign(buf, std::size_t(n));
    }
    const std::size_t k = p.rfind('/');
    return k == std::string::npos ? std::string() : p.substr(0, k + 1);
}
bool file_exists(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fclose(f);
    return true;
}
// A net path as given: an absolute path as it is; a relative one in the current directory, then the engine's.
std::string resolve_net(const std::string& name) {
    if (name.empty() || name[0] == '/' || file_exists(name)) return name;
    const std::string d = exe_dir();
    return !d.empty() && file_exists(d + name) ? d + name : name;
}

struct Engine {
    Position pos;
    Searcher searcher{1, 64};
    std::thread thread;
    std::vector<std::string> netNote;  // what the first isready reports about the network loaded at startup
    bool firstReady = true;
    int multiPV = 1;
    bool ponderOpt = false;  // UCI option Ponder: bestmove then names a move to ponder on
    // The last `position` command was refused, whole (a bad fen) or from one of its moves on: the position held
    // is not the one the GUI means, and `go` does not search it.
    bool posRefused = false;

    Engine() {
        pos.set_fen(START_FEN);
        netNote = load_net(DEFAULT_NET);
    }
    // Loads the network for the value of EvalFile (see DEFAULT_NET) and returns the info strings that say what
    // was loaded. "EvalFile load failed" marks a file that was named and could not be loaded (the tools look
    // for it), "ERROR" the lack of any network.
    static std::vector<std::string> load_net(const std::string& value) {
        std::vector<std::string> lines;
        std::string err, why = "EvalFile is empty";
        if (!(value.empty() || value == "<empty>")) {
            const std::string path = resolve_net(value);
            why = value + " is neither in the current directory nor next to the engine";
            if (file_exists(path) || value != DEFAULT_NET) {
                if (nnue::load(path, &err)) {
                    lines.push_back("EvalFile loaded " + path + " (" + nnue::net_desc() + ")");
                    return lines;
                }
                lines.push_back("EvalFile load failed: " + path + ": " + err);
                why = "EvalFile " + path + " cannot be loaded";
            }
        }
        const EmbeddedNet e = embedded_net();
        if (e.size) {
            if (nnue::load_buffer(e.data, e.size, &err)) {
                lines.push_back(std::string("embedded network ") + DEFAULT_NET + " loaded (" + nnue::net_desc() + ")");
                return lines;
            }
            lines.push_back("embedded network failed to load: " + err);
        }
        lines.push_back("ERROR: no network: " + why + (e.size ? "" : " and none is embedded in this binary") +
                        "; set EvalFile (the hand-written evaluation is used until then)");
        return lines;
    }
    void cmd_isready() {
        if (firstReady)
            for (const std::string& l : netNote) out("info string " + l);
        firstReady = false;
        out("readyok");
    }
    ~Engine() { wait(); }
    void wait() {
        if (thread.joinable()) {
            searcher.stop();
            thread.join();
        }
    }

    void cmd_uci() {
        out("id name Maskfish 0.1");
        out("id author the Maskfish developers (see AUTHORS file)");
        out("option name Hash type spin default 64 min 1 max 16384");
        out("option name Threads type spin default 1 min 1 max 256");
        out(std::string("option name EvalFile type string default ") + DEFAULT_NET);
        out("option name Clear Hash type button");
        out("option name MultiPV type spin default 1 min 1 max " + std::to_string(MAX_MOVES));
        out("option name Ponder type check default false");
        out("option name HNNScale type spin default 1000 min 100 max 4000");
        for (const Param& p : params())
            out("option name " + std::string(p.name) + " type spin default " + std::to_string(*p.value) +
                " min " + std::to_string(p.lo) + " max " + std::to_string(p.hi));
        out("uciok");
    }

    void cmd_setoption(std::istringstream& is) {
        std::string tok, name, value;
        is >> tok;  // "name"
        while (is >> tok && tok != "value") name += (name.empty() ? "" : " ") + tok;
        while (is >> tok) value += (value.empty() ? "" : " ") + tok;
        wait();
        // A spin option's value, clamped to [lo, hi]; a value that is not a number is refused.
        int v = 0;
        auto spin = [&](int lo, int hi) {
            long long n = 0;
            const char* b = value.data();
            const auto [end, ec] = std::from_chars(b, b + value.size(), n);
            if (value.empty() || end != b + value.size() || (ec != std::errc() && ec != std::errc::result_out_of_range)) {
                out("info string bad value " + value + " for option " + name);
                return false;
            }
            if (ec == std::errc::result_out_of_range) n = value[0] == '-' ? lo : hi;
            v = int(std::clamp<long long>(n, lo, hi));
            return true;
        };
        // Option names are matched without regard to case (UCI): the canonical name replaces the one sent.
        auto named = [&](const char* canonical) {
            if (name.size() != std::strlen(canonical)) return false;
            for (std::size_t i = 0; i < name.size(); ++i)
                if (std::tolower(static_cast<unsigned char>(name[i])) != std::tolower(static_cast<unsigned char>(canonical[i])))
                    return false;
            name = canonical;
            return true;
        };
        if (named("Hash")) {
            if (spin(1, 16384)) {
                const std::size_t got = searcher.resize_hash(std::size_t(v)) >> 20;
                if (got * 2 <= std::size_t(v))  // below the power of two the request rounds down to: memory was short
                    out("info string Hash " + std::to_string(v) + " MB cannot be allocated; the table has " +
                        std::to_string(got) + " MB");
            }
        }
        else if (named("Threads")) { if (spin(1, 256)) searcher.set_threads(v); }
        else if (named("Clear Hash")) searcher.clear();
        else if (named("MultiPV")) { if (spin(1, int(MAX_MOVES))) multiPV = v; }
        else if (named("Ponder")) ponderOpt = value == "true";  // the engine ponders when it is sent go ponder
        else if (named("HNNScale")) { if (spin(100, 4000)) { hnn::g_scale = v; searcher.clear(); } }  // permille of the net's value
        else if (named("EvalFile")) {
            firstReady = false;  // the answer below says what is loaded
            for (const std::string& l : load_net(value)) out("info string " + l);
            searcher.clear();
        } else {
            for (Param& p : params())
                if (named(p.name)) {
                    if (spin(p.lo, p.hi)) {
                        *p.value = v;
                        search::init();
                    }
                    return;
                }
            out("info string unknown option " + name);
        }
    }

    void cmd_position(std::istringstream& is) {
        std::string tok, fen;
        is >> tok;
        if (tok == "startpos") {
            fen = START_FEN;
            is >> tok;
        } else if (tok == "fen") {
            while (is >> tok && tok != "moves") fen += (fen.empty() ? "" : " ") + tok;
        } else
            return;
        std::string err;
        Position p;
        if (!p.set_fen(fen, &err)) {
            out("info string bad fen: " + err);
            posRefused = true;
            return;
        }
        posRefused = false;
        if (tok == "moves")
            while (is >> tok)
                if (!p.apply_token(tok, &err)) {
                    out("info string bad move " + tok + ": " + err);
                    posRefused = true;
                    break;
                }
        pos = p;
    }

    void cmd_go(std::istringstream& is) {
        Limits lim;
        std::string tok;
        // A word of `go` that ends the move list of searchmoves.
        static const char* const WORDS[] = {"depth", "nodes", "movetime", "wtime", "btime", "winc", "binc",
                                            "movestogo", "infinite", "ponder", "mate", "searchmoves"};
        bool held = false;  // tok holds the word that ended searchmoves
        std::vector<std::string> searchTokens;  // the words after searchmoves, checked once the position is known
        while (held || is >> tok) {
            held = false;
            if (tok == "depth") is >> lim.depth;
            else if (tok == "nodes") is >> lim.nodes;
            // A clock or move time of 0 or less (a GUI sends it when the clock has run out, an increment still to
            // come) is the smallest budget: the engine answers at once. Left at 0 it would mean no limit.
            else if (tok == "movetime") { is >> lim.movetime; lim.movetime = std::max(lim.movetime, 1); }
            else if (tok == "wtime") { is >> lim.time[RED]; lim.time[RED] = std::max(lim.time[RED], 1); }
            else if (tok == "btime") { is >> lim.time[BLACK]; lim.time[BLACK] = std::max(lim.time[BLACK], 1); }
            else if (tok == "winc") is >> lim.inc[RED];
            else if (tok == "binc") is >> lim.inc[BLACK];
            else if (tok == "movestogo") is >> lim.movestogo;
            else if (tok == "infinite") lim.infinite = true;
            else if (tok == "ponder") lim.ponder = true;
            else if (tok == "mate") is >> lim.mate;
            else if (tok == "searchmoves")
                while (is >> tok) {
                    if (std::find(std::begin(WORDS), std::end(WORDS), tok) != std::end(WORDS)) {
                        held = true;  // the limits after the moves are read, not taken for moves
                        break;
                    }
                    searchTokens.push_back(tok);
                }
        }
        lim.multiPV = multiPV;
        wait();
        if (posRefused) {  // the GUI's position is not known: a move found in another one would mislead it
            out("info string no search: the last position command was refused");
            out("bestmove (none)");
            return;
        }
        // searchmoves: the legal moves among those named. A word that is not one is said and left out; when none
        // is left the engine does not search (the whole list would be a move the GUI did not allow).
        if (!searchTokens.empty()) {
            Move legal[MAX_MOVES];
            const int n = pos.gen_legal(legal);
            for (const std::string& t : searchTokens) {
                const Move m = parse_move_coords(t);
                if (m != MOVE_NONE && std::find(legal, legal + n, m) != legal + n) {
                    if (std::find(lim.searchMoves.begin(), lim.searchMoves.end(), m) == lim.searchMoves.end())
                        lim.searchMoves.push_back(m);
                } else
                    out("info string searchmoves: " + t + " is not a legal move, left out");
            }
            if (lim.searchMoves.empty() && n > 0) {
                out("info string no search: no legal move in searchmoves");
                out("bestmove (none)");
                return;
            }
        }
        Position root = pos;
        searcher.prepare(lim.ponder);  // a stop or ponderhit that arrives before the search thread runs must not be lost
        const bool offerPonder = ponderOpt;
        thread = std::thread([this, root, lim, offerPonder] {
            static const std::function<void(const std::string&)> info = [](const std::string& s) { out(s); };
            const SearchResult r = searcher.go(root, lim, info, false);
            // No legal move: the side to move has lost (stalemate included), reported as mate 0.
            if (r.best == MOVE_NONE) out("info depth 0 score mate 0");
            // With Ponder on, the second move of the line is offered for pondering (as a base move: what a
            // face-down mover turns out to be is for the GUI to tell).
            out(r.best == MOVE_NONE ? "bestmove (none)"
                                    : "bestmove " + move_str(r.best) +
                                          (offerPonder && r.pv.size() > 1 ? " ponder " + move_str(base_move(r.pv[1])) : ""));
        });
    }

    void cmd_bench(std::istringstream& is) {
        int depth = 7;
        if (!(is >> depth) || depth < 1) depth = 7;  // a depth of 0 or less would search no further than the root
        wait();
        uint64_t nodes = 0;
        const auto t0 = std::chrono::steady_clock::now();
        searcher.clear();
        for (const char* f : BENCH_FENS) {
            Position p;
            std::string err;
            if (!p.set_fen(f, &err)) {  // a half-parsed position would be searched with undefined behaviour
                out("info string bench: bad fen " + std::string(f) + ": " + err);
                continue;
            }
            Limits lim;
            lim.depth = depth;
            const SearchResult r = searcher.go(p, lim, [](const std::string&) {});
            nodes += r.nodes;
            out("bench " + std::string(f) + " -> " + move_str(r.best) + " score " + std::to_string(r.score) +
                " nodes " + std::to_string(r.nodes));
        }
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
        out("Nodes searched: " + std::to_string(nodes));
        out("Nodes/second  : " + std::to_string(nodes * 1000 / uint64_t(std::max<int64_t>(1, ms))));
        out("Time (ms)     : " + std::to_string(ms));
    }
};

}  // namespace

int uci_loop(int argc, char** argv) {
    Engine engine;
    std::string line;
    // Command-line arguments are executed as one command (e.g. `maskfish bench 8`), then exit.
    const bool once = argc > 1;
    if (once) {
        for (int i = 1; i < argc; ++i) line += (i > 1 ? " " : "") + std::string(argv[i]);
    }
    while (once || std::getline(std::cin, line)) {
        std::istringstream is(line);
        std::string cmd;
        is >> cmd;
        if (cmd == "uci") engine.cmd_uci();
        else if (cmd == "isready") engine.cmd_isready();
        else if (cmd == "ponderhit") engine.searcher.ponderhit();
        else if (cmd == "ucinewgame") { engine.wait(); engine.searcher.clear(); }
        else if (cmd == "setoption") engine.cmd_setoption(is);
        else if (cmd == "position") { engine.wait(); engine.cmd_position(is); }
        else if (cmd == "go") engine.cmd_go(is);
        else if (cmd == "stop") engine.wait();
        else if (cmd == "quit") break;
        else if (cmd == "d") out(engine.pos.pretty());
        else if (cmd == "moves") {  // legal moves of the current position (diagnostics)
            Move ms[MAX_MOVES];
            const int n = engine.pos.gen_legal(ms);
            std::string line2 = "moves";
            for (int i = 0; i < n; ++i) line2 += " " + move_str(ms[i]);
            out(line2);
        }
        else if (cmd == "eval") out("classical " + std::to_string(classical_eval(engine.pos)) + " eval " + std::to_string(evaluate(engine.pos)));
        else if (cmd == "perft") {
            int d = 1;
            is >> d;
            Position p = engine.pos;
            const auto t0 = std::chrono::steady_clock::now();
            const uint64_t n = search::perft(p, d);
            const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
            out("perft " + std::to_string(d) + " " + std::to_string(n) + " (" + std::to_string(ms) + " ms)");
        } else if (cmd == "see") {
            std::string mv;
            is >> mv;
            const Move m = parse_move_coords(mv);
            out("see " + mv + " " + std::to_string(search::see_value(engine.pos, m)));
        } else if (cmd == "perftc") {
            int d = 1;
            is >> d;
            Position p = engine.pos;
            uint64_t leaves = 0;
            const double mass = search::perft_chance(p, d, leaves);
            char buf[128];
            std::snprintf(buf, sizeof buf, "perftc %d leaves %llu expected %.6f", d, (unsigned long long)leaves, mass);
            out(buf);
        } else if (cmd == "nnbench") {
            if (!nnue::loaded()) { out("info string no net"); continue; }
            Position p = engine.pos;
            Move ml[MAX_MOVES];
            const int n = p.gen_legal(ml);
            static nnue::Accumulator a0, a1;
            if (is.str().find("ftz") != std::string::npos) nnue::set_flush_denormals();
            nnue::Dirty d;
            nnue::refresh(p, a0);
            const int iters = 1000000;
            auto t0 = std::chrono::steady_clock::now();
            int64_t sink = 0;
            for (int i = 0; i < iters; ++i) sink += nnue::evaluate(a0, p.stm);
            auto t1 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) {
                Move m = ml[i % n];
                if (p.is_dark_move(m)) m = make_move(from_sq(m), to_sq(m), SOLDIER);
                nnue::make_dirty(p, m, d);
                nnue::update(a0, a1, d);
                sink += int64_t(a1.hnn.v[0][0]);
            }
            auto t2 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters / 10; ++i) { nnue::refresh(p, a1); sink += int64_t(a1.hnn.v[1][1]); }
            auto t3 = std::chrono::steady_clock::now();
            auto ns = [&](auto x, auto y, int k) { return std::chrono::duration<double, std::nano>(y - x).count() / k; };
            out("nnbench evaluate " + std::to_string(ns(t0, t1, iters)) + " ns, update " + std::to_string(ns(t1, t2, iters)) +
                " ns, refresh " + std::to_string(ns(t2, t3, iters / 10)) + " ns (sink " + std::to_string(sink & 1) + ")");
        } else if (cmd == "bench") engine.cmd_bench(is);
        else if (cmd == "selfplay") { engine.wait(); selfplay(is); }
        else if (cmd == "compiler") out(build_info());
        else if (!cmd.empty()) out("info string unknown command " + cmd);
        if (once) {
            engine.wait();
            break;
        }
    }
    engine.wait();
    return 0;
}

}  // namespace jq
