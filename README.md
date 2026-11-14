# Maskfish - *luck is a skill too*

Maskfish is a free UCI engine for **Jieqi** (揭棋): a xiangqi variant in which
all pieces except the kings start face down, shuffled, and are revealed when
they first move.

It is written in C++20 and has:

* a board representation for face-down ("dark") pieces and for each side's
  **urn**, the identities its face-down pieces can still be;
* chance nodes in the search: a move of a face-down piece is evaluated over
  every identity it can turn out to be, weighted by how many of each are left
  in the urn;
* an NNUE-style network (HNN) with features for face-down pieces and for the
  contents of the urns, evaluated with integer arithmetic so that every build
  gives the same values;
* an independent rules referee (`tools/native/`), which judges every game of a
  match and against which the engine's rules are tested.

Maskfish speaks UCI, with a FEN and move notation for face-down pieces (see
[UCI protocol](#uci-protocol)). It needs a GUI or another program that
knows the Jieqi rules and tells the engine which piece was revealed after each
reveal; the repository has a [local board](#playing-against-it) in the
browser for that.

The repository also contains:

| Directory | |
| --- | --- |
| [`src/`](src/) | the engine; [`src/hnn/`](src/hnn/) the network evaluator |
| [`tools/`](tools/) | the rules referee ([`tools/native/`](tools/native/), C++, and [`tools/jieqi/`](tools/jieqi/), its Python side), refereed matches and SPRT, SPSA, the local board, training-record tools, the network packer ([`tools/hnn/`](tools/hnn/)), `check_all.sh` |
| [`train/`](train/) | the PyTorch trainer of the network |
| [`scripts/`](scripts/) | network fetch / publish, packaging of the release binary, and the training automation ([README](scripts/README.md)) |
| [`fishtest/`](fishtest/) | a single-machine fishtest to test patches with an SPRT or tune them with SPSA ([README](fishtest/README.md)) |
| [`tests/`](tests/) | the differential test against the referee, smoke tests and the test network (see [Testing your change](./CONTRIBUTING.md#testing-your-change)) |
| [`.github/`](.github/) | CI, binaries, releases and the fishtest image (GitHub Actions) |

## Network

The engine evaluates with an HNN network: a raw `.hnn` file (magic
`JQHNNUE1`) or a compressed `.nnue` (`JQHNNL01`, `JQHNNZ01`), recognised by its
first bytes whatever the file name.

The official network, `maskfish.nnue`, is published as the single file of the
hidden git ref `refs/assets/net` of this repository (by
`scripts/push_net.sh`), apart from the history of the code.
`scripts/net.sh` fetches it to `nets/maskfish.nnue` (a local copy, ignored by
git), where the checks, the tools, fishtest, the build (which embeds it) and the
release archive take it:

```sh
sh scripts/net.sh                       # nets/maskfish.nnue from refs/assets/net
sh scripts/push_net.sh path/to/net      # publish another network as refs/assets/net
build/jqnz_tool z best.hnn best.nnue    # a trained .hnn packed into the compressed form
```

The engine loads its network in this order:

1. The file `EvalFile` names: by default `maskfish.nnue` in the current
   directory, or else in the engine's own directory (where the release archive
   puts it); a relative path is looked for the same way.
2. The network embedded in the binary (CMake `MASKFISH_EMBED_NET`, below; the
   published binaries embed the published network), when `EvalFile` is empty,
   is the default name and no such file exists, or names a file that cannot be
   loaded. In the last case the engine first prints `info string EvalFile load
   failed: <path>: <reason>`, so that a mistyped path does not pass unseen;
   `tools/match.py` voids a game whose engine printed it.
3. Without either, `info string ERROR: no network: <reason>`, and the engine
   plays with a much weaker hand-written evaluation.

The first `isready` says which network was loaded (`EvalFile loaded <path>
(...)` or `embedded network maskfish.nnue loaded (...)`, with the md5 of the
decoded network), and so does every `setoption name EvalFile`. `compiler` says
whether a network is embedded and its size. The `selfplay` command's `evalfile`
does not fall back: a file it cannot load is an error.

## Rules implemented

The rules are the common rules of Jieqi, with these conventions where the
published rules are silent:

* Board and setup as in xiangqi, red moves first. The kings stand face up on
  e0 and e9; each side's other 15 pieces (2 chariots, 2 horses, 2 elephants,
  2 advisors, 2 cannons, 5 soldiers) are shuffled face down on its 15 other
  starting points.
* A face-down piece moves and captures like the piece whose starting point it
  stands on, and is revealed at its destination; there is no reveal without a
  move. A face-down advisor's move stays in the palace. A face-down piece is
  therefore always on a starting point of its own side.
* Revealed advisors move one diagonal step anywhere, revealed elephants two
  diagonal steps (eye empty) anywhere, both crossing the river. A soldier
  moves forward on its own half (even when revealed on its back rank) and also
  sideways beyond the river, never backward. Kings stay in the palace and may
  not face each other on an open file.
* Reveals are public; a face-down piece that is captured is shown only to the
  side that captured it.
* A side with no legal move (checkmate or stalemate) loses. The third
  occurrence of a position is a draw, except that the side giving a perpetual
  check, or a perpetual chase, loses. After 80 plies without a capture
  (reveals do not reset the count) the game is drawn, unless the side to move
  has no legal move.

The referee (`tools/native/`) implements these rules; `maskfish_difftest`
compares the engine's move generation, check, chase and repetition verdicts
with it.

## UCI protocol

The engine speaks UCI with the following additions.

### FEN

A FEN has five fields, `board side urns no-capture-plies move-number`:

```
xxxxkxxxx/9/1x5x1/x1x1x1x1x/9/9/X1X1X1X1X/1X5X1/9/XXXXKXXXX w R2A2C2P5N2B2r2a2c2p5n2b2 0 1
```

* Board: from rank 9 to rank 0, the xiangqi letters `K R N B A C P`
  (uppercase for Red, the side moving first, lowercase for Black), plus
  `X` / `x` for a face-down piece, which is accepted only on a starting point
  of its own side other than the king's.
* Side to move: `w` (or `r`) for Red, `b` for Black.
* **Urns**: for each kind of piece, how many of each side's face-down pieces
  can still be of that kind, as `<letter><count>` pairs, e.g.
  `R2A2C2P5N2B2r2a2c2p5n2b2` at the start. The engine writes them in the
  order `R A C P N B` and reads them in any order; kinds that are
  not listed count as 0, and `-` means both urns are empty. A side's urn also
  holds the identities of its face-down pieces that were captured without its
  seeing them, so it must hold at least as many pieces as the side has
  face-down pieces on the board.
* No-capture counter in plies (0 to 80; a larger number counts as 80); the
  move number is read and ignored.
  The urns, the counter and the move number may be left out (`-`, 0, 1).

`position startpos` uses the FEN above. A FEN that cannot be read (a bad
character, a rank without 9 files, a face-down piece off its starting point, a
missing king or a second one, a king out of its palace, more than 16 pieces a
side, more pieces of a kind on the board and in the urn than at the start, an
urn smaller than the side's face-down pieces, a counter that is not a number
or is negative) is refused with `info string bad fen: ...`, and the engine
keeps its previous position. `tests/invalid_input.py` checks these refusals
and that no input crashes the engine.

### Moves

Moves use coordinate notation (`h2e2`): files `a`–`i`, ranks `0`–`9` from
Red's side. Moves sent to the engine carry extra letters for what the side
that sends them saw:

| Situation | Format | Example |
|---|---|---|
| Move of a revealed piece or a king | `<from><to>` | `h2e2` |
| A face-down piece moves: the revealed identity is **required**, in the mover's case | `<from><to><piece>` | `a3a4P` |
| A face-down piece is captured and the capturer saw it: its identity, in the victim's case | `<from><to><captured>` | `b9d9a` |
| Both | `<from><to><piece><captured>` | `b2b9Pr` |

In `b2b9Pr`, a face-down piece on Red's cannon point captures on b9, turns out
to be a soldier, and the piece it took was a chariot. Only the capturer's
record of the move names the captured piece; the other side's record leaves it
out. An `x` or `X` in the suffix, which some programs write after a lone reveal,
is ignored. The revealed or captured identity must still be in
the corresponding urn.

If a move is illegal or its letters are invalid, the engine prints
`info string bad move <move>: ...` and ignores that move and the ones after
it; the position is the one before the bad move. After a `position` command
that was refused (a bad fen, a bad move), `go` does not search: the engine
answers `info string no search: ...` and `bestmove (none)` until a `position`
command is accepted.

```
position startpos moves a3a4P
position startpos moves b2b9Pr a9b9r
```

### Engine output

* `bestmove` is always a plain coordinate move. When it moves a face-down
  piece, the GUI reveals the piece and sends its identity back in the next
  `position` command.
* A PV in `info` lines stops after the first move of a face-down piece,
  because the continuation depends on the piece that will be revealed.
* With `Ponder` on, `bestmove` also names the second move of the PV as the
  move to ponder on, when there is one (there is none after a face-down move).
* Scores of moves of face-down pieces are expected values over the possible
  identities (`score cp`, or `score mate N`). With no legal move the engine
  answers `info depth 0 score mate 0` and `bestmove (none)`.

`go` takes `wtime btime winc binc movestogo`, `movetime`, `depth`, `nodes`,
`infinite`, `ponder`, `mate N` and `searchmoves`. A `go ponder` search has no
time limit until `ponderhit`, from which its clock runs; a pondering or
infinite search answers only after `ponderhit` or `stop`. `go mate N` stops at
a mate in N moves or fewer. A word after `searchmoves` that is not a legal
move is left out (`info string searchmoves: <word> is not a legal move, left
out`); when none of them is legal the engine does not search and answers
`info string no search: no legal move in searchmoves` and `bestmove (none)`.

### UCI options

| Option | Default | Meaning |
| --- | --- | --- |
| `Hash` | `64` | Transposition table size in MB (1 to 16384) |
| `Threads` | `1` | Search threads (Lazy SMP, 1 to 256) |
| `EvalFile` | `maskfish.nnue` | The network file; `<empty>` (or a file that cannot be loaded) for the embedded network (see [Network](#network)) |
| `Clear Hash` | | Clears the transposition table and the search histories |
| `MultiPV` | `1` | Number of lines searched and reported |
| `Ponder` | `false` | `bestmove` names a move to ponder on |
| `HNNScale` | `1000` | Scale of the network's value in permille (100 to 4000) |
| `NodesTime` | `0` | For tests on a loaded host: the clock runs on nodes, N per millisecond (0: wall clock) |

Every search parameter is also a spin option (`ChanceK`, `ChanceMix`,
`LmrDiv`, `RfpMargin`, `ChaseRule`, ...): the plain ints of the `params()`
table in [`src/search.cpp`](src/search.cpp), with the measured configuration as
their defaults; `uci` lists them all with their ranges. SPSA
(`tools/spsa.py`, fishtest) tunes through them. An option the engine does not
have is answered with `info string unknown option <name>`. Option names are
matched without regard to case (`setoption name hash value 32`). A spin value out of
its range is clamped to it; one that is not a number is refused with `info
string bad value <value> for option <name>`.

### Debugging commands

* `d` prints the board (`X` / `x` for face-down pieces) and the FEN.
* `moves` lists the legal moves; `see MOVE` prints the static exchange value of
  a move.
* `eval` prints the hand-written evaluation (`classical`) and the evaluation
  in use (`eval`, the network's when one is loaded).
* `perft N` counts move paths, reveals not expanded (44, 1,926 and 83,907 from
  the start for N = 1 to 3); `perftc N` expands the chance outcomes and prints
  the leaves and the expected mass.
* `bench [DEPTH]` (default 7) searches a fixed set of positions and prints the
  node count, the signature of the build (61,977 with the test network, see
  [Contributing](#contributing)); `nnbench` times the network's evaluate,
  update and refresh.
* `compiler` prints the compiler and the instruction set of the build (see
  [Universal binary](#universal-binary)).
* `selfplay` plays games and writes training records (see
  [Training](#training)).

Arguments on the command line run one command and exit, as in
`build/maskfish perft 3`.

## Compiling

With CMake 3.20 or later and a C++20 compiler:

```sh
cmake -S . -B build
cmake --build build -j4
```

This builds the engine (`build/maskfish`), the differential test
(`build/maskfish_difftest`) and the referee's library
(`build/lib/libjieqi_rules.so`), which `tools/match.py` and `tools/play.py`
load from there (or from `$JIEQI_NATIVE_LIBRARY`); the target `jqnz_tool`
(`--target jqnz_tool`) builds the network packer.

| CMake option | Default | |
| --- | --- | --- |
| `CMAKE_BUILD_TYPE` | `Release` | `-O3` |
| `MASKFISH_NATIVE` | `ON` | `-march=native`; `OFF` gives a portable build (the same evaluations, slower) |
| `MASKFISH_SANITIZE` | `OFF` | AddressSanitizer and UndefinedBehaviorSanitizer |
| `MASKFISH_LTO` | `OFF` | link-time optimisation (it measured no faster here) |
| `MASKFISH_BITBOARD` | `OFF` | move generation, legality, checks and the exchange evaluation on bitboards (`src/bitboard.h`): the same moves in the same order and the same `bench` signature, found faster; off until a test at a time control shows the gain |
| `MASKFISH_PEXT` | `OFF` | with `MASKFISH_BITBOARD`, gathers a file's occupancy with PEXT (for CPUs where it is fast) |
| `MASKFISH_UNIVERSAL` | `OFF` | a universal binary (below; Linux x86-64 and arm64) |
| `MASKFISH_EMBED_NET` | `ON` | embed the network `MASKFISH_NET_FILE` in the binary, if that file exists |
| `MASKFISH_NET_FILE` | `nets/maskfish.nnue` | the network to embed (`sh scripts/net.sh` fetches the published one there) |

The network to embed is looked for when CMake configures: fetch it with
`sh scripts/net.sh` first (or run `cmake` again after), otherwise CMake warns
and the binary embeds none. A file that is not a network is an error. The
embedded network is used when `EvalFile` gives none (see [Network](#network));
`tests/evalfile_smoke.py BIN [--net NET]` checks the order, and with `--net`
that the embedded network is `NET`.

### Universal binary

```sh
cmake -S . -B build -DMASKFISH_UNIVERSAL=ON
cmake --build build -j4 --target maskfish
```

builds one binary that runs on every Linux x86-64 or arm64 CPU at the speed of
the matching build. It is the engine built for the baseline of the
architecture, with the engine built for every higher instruction set embedded
in it; at startup it runs the highest build the CPU supports (from memory,
`memfd_create` and `fexecve`, in the same process):

| Architecture | Builds (lowest first) |
| --- | --- |
| x86-64 | `x86-64` (SSE2), `x86-64-v2` (SSSE3, SSE4.1, POPCNT), `x86-64-v3` (AVX2, BMI2), `x86-64-avxvnni` (AVX-VNNI), `x86-64-v4` (AVX-512 BW), `x86-64-avx512icl` (AVX-512 with VNNI and VBMI2) |
| arm64 | `armv8` (NEON), `armv8-dotprod` (dot product instructions) |

* Every build gives the same evaluations and the same bench signature (the
  network's arithmetic is integer). `tests/universal_smoke.py BIN` checks every
  build the CPU runs: perft, the bench signature, the network evaluator, the
  embedded network, and that the network next to the binary is found.
* The network is embedded once, in the binary, which hands it to the build it
  starts through an inherited memfd (`MASKFISH_UNIVERSAL_NET_FD`).
* The builds the CPU lacks can be checked under Intel SDE (`sde64 -adl --
  BIN` runs the `x86-64-avxvnni` build, `-icx` the `x86-64-avx512icl` one,
  ...): the binary starts its build through `/proc/self/fd`, which SDE
  follows.
* `compiler` prints the build that runs. `MASKFISH_UNIVERSAL_ARCH=list` in the
  environment lists the builds and the one this CPU gets;
  `MASKFISH_UNIVERSAL_ARCH=<build>` forces one (the CPU must support it), e.g.
  to compare them on one machine.
* When a build cannot be started from memory (a kernel or a sandbox that
  forbids it), the binary says so on stderr and runs its baseline build.
* Every build is compiled separately (`build/universal/`), so the build takes
  about as many times longer as there are builds. It does not go with
  `MASKFISH_SANITIZE` or a `-march` in `CMAKE_CXX_FLAGS`;
  `-DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc"` links the C++
  runtime statically, as the published binaries do.

The release archives (`scripts/package_binary.sh`, made by the GitHub
workflows from `pro`), `maskfish-ubuntu-x86-64-universal.tar` and
`maskfish-ubuntu-arm64-universal.tar`, hold the universal binary (with the
published network embedded), the same network as `maskfish.nnue`, this README,
AUTHORS and the licence.

## Search and evaluation

### Information model

Each side's urn holds the identities still unaccounted for among its
face-down pieces, as seen by the side the engine plays for. The opponent's urn
is known exactly; our own also holds those of our face-down pieces that were
captured without our seeing them. By exchangeability, a face-down mover
reveals identity t with probability `urn[t] / urnTotal`. In search, capturing a
face-down piece leaves the urn unchanged, and moving one is a chance node with
up to six outcomes.

### Search ([`src/search.cpp`](src/search.cpp))

* PVS with a transposition table, null move, reverse futility, futility and
  late-move pruning, LMR, SEE pruning, killers, counter-moves, butterfly,
  continuation and capture history, check extension, aspiration windows and
  Lazy SMP.
* Chance nodes average their outcomes in expected-score space,
  `sigmoid(v / ChanceK)`, with values saturating at ±10000.
  * At an expanded chance node (`ChanceMix 6`, the default) every identity gets
    an open-window search at depth − 6 and the most numerous identity a
    full-depth search; the value is the probability-weighted average with the
    full-depth value in place of that identity's shallow one. `ChanceMix 0`
    searches every outcome at full depth with the parent's window instead.
  * `ChanceXDepth 1` (the default): every chance node of the main search with
    depth 1 or more is expanded this way, however many chance nodes lie above
    it on the line. Elsewhere (in quiescence, and at depth 0 once the line has
    used its expanded chance layers, `MaxChanceLayers`) the move plays one
    representative identity, the one whose value is closest to the urn's mean.
* Options that are off by default: `ThreatRevRoot`, `ThreatRevScale`,
  `ThreatRevLmr` and `ThreatRevOrder` treat a face-down piece the opponent can
  win material on (by `ThreatRevMode`, `ThreatRevSee`, `ThreatRevProb`)
  differently at the root, in LMR and in move ordering; `ChanceMixLow` and
  `ChanceMixLowR` probe unlikely identities of a ChanceMix node less deeply; a
  negative `RootDarkPen` is a bonus for face-down moves at the root. None has
  been tested for strength here.
* The rules in search are those of the referee: no legal move loses; at a
  repetition the side responsible for a perpetual check or, with
  `ChaseRule 1` (the default), a perpetual chase loses (a port of the
  referee's chase classification); otherwise a repetition and the 80-ply
  no-capture rule draw.

### Network ([`src/hnn/`](src/hnn/))

* Inputs per perspective: king buckets (by the scheme of the net: none, three
  by king file, or two or six with the board mirrored by the king's file) × 16
  piece kinds (own and opponent's `K R N B A C P` and face-down) on 90 points,
  oriented to the perspective's home rank, and 30 urn thresholds
  `urn[c][t] >= j`.
* Layers: inputs → W (256 for the published net) int16 accumulators and 16
  PSQT columns, the same weights for both perspectives, updated incrementally;
  the pairwise product of each half → W bytes; one of 16 layer stacks (by the
  number of pieces and whether any is face down): W → 16 → [square, clipped]
  30 → 32 → 1, with a skip from the 16th L1 output and the PSQT column of the
  stack. A version-3 file has another head (L1 16 or 32, L2 32 or 64, an
  optional squared L2).
* The value is scaled by material and damped by the no-capture count inside
  the evaluator (the search's own `Rule40Div` shrink applies to the
  hand-written evaluation only). Kernels for AVX-512, AVX2, SSE and NEON;
  integer arithmetic (and the engine is compiled with `-ffp-contract=off`), so
  every build and CPU gives the same values.

## Training

`train/train.py` (PyTorch, `train/requirements.txt`; CUDA, or `--device cpu`)
trains on TrainRecords, the 112-byte records of the engine's `selfplay`
command (`src/selfplay.cpp`, `train/records.py`), and writes `best.hnn` and
`last.hnn` with their `.pt` states. `--init` takes a net file, such as the
published `maskfish.nnue`, and continues from its weights exactly (with its
width, king buckets and head), so the next network starts from the current
one:

```sh
printf 'selfplay evalfile nets/maskfish.nnue games 20000 nodes 5000 threads 8 out data/gen0.bin\nquit\n' | build/maskfish
printf 'selfplay evalfile nets/maskfish.nnue games 500 nodes 5000 threads 8 seed 7 out data/val0.bin\nquit\n' | build/maskfish
python3 train/train.py data/gen0.bin --val-file data/val0.bin --init nets/maskfish.nnue \
    --wt 0 --ws 0.7 --wr 0.3 --epochs 10 --out runs/gen1
```

* `selfplay` takes `games` (default 100), `nodes` (5000) or `depth`, `threads`
  (1), `seed` (1), `random` (random opening plies, 6), `hash` (MB, 16),
  `maxply` (600), `out` (`selfplay.bin`) and `evalfile`.
* Training target, in win-probability space: `wt · sigmoid(teacher / K) +
  ws · sigmoid(score / K) + wr · (result + 1) / 2`, with `K` fitted on the data
  (`--k` fixes it). The teacher (`--teacher`) is an HNN net with the standard
  head, evaluated exactly as the engine does (`train/hnn_ref.py`); `--wt 0`
  trains without one. Quantisation-aware in the last quarter of the steps
  (`--qat-from`).
* A new network from scratch: `--width`, `--kb-scheme` (0 to 3, the schemes
  above), `--l1`/`--l2`/`--sq2`. `train/hnn_file.py` reads and writes net
  files as weights (numpy only).

A new network must then win an SPRT against the current one
(`tools/match.py`). `scripts/pipeline.sh` does all of this for one generation
and `scripts/autotune.sh` repeats it until the Elo stops rising
([scripts/README.md](scripts/README.md)).

## Playing against it

```sh
python3 tools/play.py            # then open http://localhost:8765
```

A local board in the browser (`tools/play.py` and `tools/play.html`, Python
standard library only). The server listens on 127.0.0.1 and talks to one
local engine process; it connects to nothing else. Options: `--port`,
`--engine` (default `build/maskfish`, or the newest `bin/maskfish-v*` if there
is one), `--net`, and `--threads` and `--hash` as defaults for the new-game
dialog.

* New game: your colour, a game clock (minutes and seconds per move) or a
  fixed time per engine move, threads, hash, and whether the engine thinks on
  your time.
* The referee is the rules library of `tools/native/`, the deal is random, and
  the browser is told only what a player at the board sees: reveals are
  public, a face-down piece you capture is shown to you alone, and a face-down
  piece the engine captures from you stays unknown to you until the game is
  over. The engine receives its own observation, exactly as in `match.py`.
* Finished games are appended to `runs/play/games.jsonl` (seed, moves,
  result).

## Tools

* `tools/match.py`: refereed matches on seeded deals in colour-swapped pairs;
  each engine receives only its own observation, as move tokens of the
  notation above. It reports Elo with trinomial and pentanomial intervals. Engines are
  given as `--engine name=path[:Opt=val,...]`. `--sprt E0 E1` stops at a generalized SPRT decision
  on the pairs (`--sprt-model logistic`, the default, or `normalized` for nElo
  bounds); `--scores` records the engines' scores for training.
* `tools/sprt.py GAMES --engine NAME --elo0 E0 --elo1 E1`: the pentanomial
  GSPRT (exact maximum likelihood, logistic or normalized Elo)
  with Elo, nElo and LOS, for the `games.jsonl` of any run, finished or not.
* `tools/spsa.py`: SPSA tuning of search parameters on game pairs.
* `tools/games2rec.py`: training records from refereed games (`match.py
  --scores`); `tools/label.py`: re-labelling of records with another engine's
  scores; `scripts/datatool.py`: record files (count, score scale, rescale,
  cat, stats).
* `scripts/pipeline.sh`, `scripts/autotune.sh`: the training automation
  ([scripts/README.md](scripts/README.md)).
* `fishtest/`: a local fishtest that builds branches and plays SPRT and SPSA
  tests through `tools/match.py` ([fishtest/README.md](fishtest/README.md)).

## Contributing

See the [Contributing Guide](./CONTRIBUTING.md). Changes to the search or
evaluation should be validated by testing against the current version of
Maskfish, e.g. with the local [fishtest](fishtest/README.md) or
`tools/match.py`; any claim about the engine's strength must give the match
command and its records.

`bash tools/check_all.sh` checks a clean clone: the engine's rules against the
referee (`maskfish_difftest`), perft, the `bench` signature (61,977 nodes with
the test network of `tests/testnet.py`), the network evaluator, pondering, the
referee's library, the local board and the trainer. A change that is meant to
keep the search as it is must keep the bench signature.

## Terms of use

Maskfish is free and distributed under the
[**GNU General Public License version 3**](./Copying.txt) (GPL v3), or any
later version. Its authors, and the authors and licences of the projects it
contains code from, are in [AUTHORS](AUTHORS). To cite it, see [CITATION.cff](CITATION.cff).

Essentially, this means you are free to do almost exactly what you want with
the program, including distributing it among your friends, making it
available for download from your website, selling it (either by itself or as
part of some bigger software package), or using it as the starting point for
a software project of your own.

The only real limitation is that whenever you distribute Maskfish in some way,
you MUST always include the license and the full source code (or a pointer to
where the source code can be found) to generate the exact binary you are
distributing. If you make any changes to the source code, these changes must
also be made available under GPL v3.
