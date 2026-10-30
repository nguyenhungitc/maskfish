# Maskfish - *luck is a skill too*

Maskfish adapts Pikafish, a free UCI xiangqi engine derived
from Stockfish, to **Jieqi**: a xiangqi variant in which all
pieces except the kings start face down and are revealed when they first move.

Maskfish shares the search framework, NNUE architecture and UCI interface of
Pikafish, and adds:

* a board representation for hidden ("dark") pieces and for the pool of pieces
  that have not been revealed yet;
* chance nodes in the search: a move of a dark piece is evaluated over every
  piece it can turn out to be, weighted by how many of each are left;
* an NNUE feature set with features for dark pieces and for the contents of the
  pool;
* extensions of the FEN and UCI move notation to describe hidden pieces and
  reveals (see [UCI protocol](#uci-protocol)).

Like Pikafish, Maskfish **does not include a graphical user interface**. It
needs a GUI or another program that knows the Jieqi rules and the notation
described below, and that tells the engine which piece was revealed after each
reveal.

The repository also contains:

| Directory | |
| --- | --- |
| [`src/`](src/) | the engine |
| [`trainer/`](trainer/) | PyTorch trainer for the Jieqi networks, data generation, SPRT matches ([README](trainer/README.md)), and scripts that automate the training ([README](trainer/scripts/README.md)) |
| [`fishtest/`](fishtest/) | a single-machine fishtest to test patches with an SPRT ([README](fishtest/README.md)) |
| [`tests/`](tests/) | regression tests (see [Contributing](#contributing)) |
| [`scripts/`](scripts/) | network fetch / publish and packaging scripts |

## Network

The engine needs a Jieqi NNUE network (default file name `maskfish.nnue`).
The official network is published as the single file of the hidden git ref
`refs/assets/net` of this repository (by `scripts/push_net.sh`); the build
fetches it and embeds it in the binary (see [Compiling](#compiling)).

## Rules implemented

* At the start, the 15 non-king pieces of each side stand face down on the usual
  xiangqi starting squares, shuffled. Kings are never hidden.
* A dark piece moves like the xiangqi piece whose starting square it occupies
  (a dark piece on a rook square moves like a rook, and so on). A dark advisor
  must stay in the palace.
* Moving a dark piece reveals it. A dark piece is therefore always on its
  starting square.
* Revealed pieces move as in xiangqi, except that revealed advisors may leave
  the palace and revealed bishops may cross the river. A revealed pawn that has
  not crossed the river (including one revealed on its back rank) only moves
  forward.
* Kings, checks, the flying-general rule, and the repetition rules are those of
  Pikafish: the third occurrence of a position is a draw, except that the side
  giving a perpetual check or a perpetual chase loses.
* A game is drawn after 40 moves (80 plies) without a capture (a side that has
  no legal move at that point loses). Reveals do not reset this counter.

## UCI protocol

The engine speaks UCI with the following additions.

### FEN

```
xxxxkxxxx/9/1x5x1/x1x1x1x1x/9/9/X1X1X1X1X/1X5X1/9/XXXXKXXXX w R2A2C2P5N2B2r2a2c2p5n2b2 0 1
```

* Board: the usual xiangqi FEN letters (`R A C P N B K`, uppercase for the side
  moving first, lowercase for the other side), plus `X` / `x` for a dark piece
  of either side. A dark piece is only accepted on a starting square of its own
  side other than the king's; anything else is ignored.
* Side to move: `w` or `b` (anything other than `b`, e.g. `r`, is read as
  `w`).
* **Pool**: for each kind of piece, the number of pieces of that kind that are
  still unknown, as `<letter><digit>` pairs, e.g. `R2A2C2P5N2B2r2a2c2p5n2b2`.
  Kinds that are not listed count as 0; an empty pool can be written `-`
  (the engine itself always prints every kind, e.g. `R0A0...`). The pool also
  contains captured dark pieces whose identity was never reported. Each side
  needs at least as many pieces in its pool as it has dark pieces on the
  board.
* The position is checked: exactly one king per side, in its palace; for every
  kind of piece, the revealed pieces on the board plus the pool do not exceed
  the initial number (2 rooks, 5 pawns, ...); the pool condition above; and the
  side not to move is not in check. Otherwise the engine prints
  `info string Invalid position: ...`, ignores the moves that follow the FEN,
  and does not search (`go` answers `bestmove (none)`).
* No-capture counter in plies (0 to 80), and the move number.

`position startpos` uses the FEN above.

### Moves

Moves use coordinate notation (`a3a4`), files `a`–`i`, ranks `0`–`9` from the
first player's side. Moves sent to the engine carry extra letters for revealed
pieces:

| Situation | Format | Example |
|---|---|---|
| Move of a revealed piece or a king | `<from><to>` | `e0e1` |
| A dark piece is moved: the revealed piece is **required** | `<from><to><piece>` | `a3a4P` |
| A revealed piece captures a dark piece: its identity is **optional** | `<from><to><captured>` | `b9d9a` |
| Both | `<from><to><piece><captured>` | `b2b9Cn` |

The revealed piece is written with the case of the moving side, the captured
piece with the case of the captured side. The piece must still be in the
corresponding pool. The identity of a captured dark piece is optional because
the owner of that piece may not be told what it was.

If a move is illegal or its letters are invalid, the engine prints
`info string Invalid move ...`, ignores that move and all following ones, and
does not search the position (`go` answers `bestmove (none)`): searching the
position before the invalid move would return a move for the wrong side.

Examples:

```
position startpos moves a3a4P
position startpos moves b2b9Cn a9b9r
position startpos moves b2b9Cn a6a5p b9d9a
```

In the second line, the dark piece on b2 (moving like a cannon) captures the
dark piece on b9, is revealed as a cannon, and the captured piece was a knight;
then the dark piece on a9 captures that cannon and is revealed as a rook.

### Engine output

* `bestmove` is always a plain coordinate move. When it moves a dark piece, the
  GUI must reveal the piece and send it back in the next `position` command.
* A PV in `info` lines stops after the first move of a dark piece, because the
  continuation depends on the piece that will be revealed. With
  `HiddenCaptureModel`, it also stops after a capture of a dark piece by the
  side the engine searches for, when the pool of its owner holds more than one
  kind of piece (the continuation depends on the identity of the captured
  piece).
* No `ponder` move is given when the best move reveals a piece, or when the
  expected reply would reveal one (the GUI could not send it back without the
  revealed piece), or when the PV stops after the best move as above.
* Scores of moves that reveal pieces are expected values over the possible
  outcomes. A mate score is only reported when every outcome leads to mate.

### UCI options

The usual options of Pikafish are available: `Threads`, `Hash`, `Clear Hash`,
`Ponder`, `MultiPV`, `Move Overhead`, `nodestime`, `UCI_ShowWDL`,
`NumaPolicy`, `Debug Log File` and `EvalFile`. Options specific to Jieqi:

| Option | Default | Meaning |
| --- | --- | --- |
| `HiddenCaptureModel` | `true` | In the search, a capture of a dark piece by the side the engine searches for is a chance node over the identity of the captured piece (any kind left in its owner's pool) |
| `ChanceModel` | `Winrate` | How the outcomes of a chance node are averaged: `Winrate` (win rates), `Tables` (score ↔ mass tables of `ChanceTables`; without tables, `Winrate`) or `Linear` (scores clamped to ±2000) |
| `ChanceScaling` | `0` | Score → win rate scale used by `Winrate` (0: built-in 360.835); `trainer/fit_scaling.py` fits it |
| `ChanceRiskAversion` | `0` | −100…100, in % of the standard deviation of the outcomes' win rates: > 0 avoids uncertain reveals and captures, < 0 seeks them |
| `ChanceTables` | empty | Directory of the tables of `ChanceModel Tables` (made by `trainer/chance_tables.py`) |
| `RevealQuietBase`, `RevealQuietPhase`, `RevealQuietSafety`, `RevealQuietHighValue`, `RevealQuietDiversity` | −12, 0, 24, −12, 0 | Bonus for safe reveals (quiet moves of dark pieces); all 0 disables it. Not tuned yet |
| `EvalScaling` | `Complexity` | Scaling of the network output in the evaluation: `Complexity`, `Alignment` (suits networks without PSQT) or `Correlation`. The network must be trained for it (`train.py --eval-scaling`) |
| `JieqiMaterial` | `false` | Count dark pieces at their expected value in the material scaling of the evaluation (the training data must be prepared with `--jieqi-material`) |
| `RawScore` | `false` | `info ... score cp` in internal units instead of normalized centipawns (used by the trainer) |

The other options (`ProbCutDarkCapture`, `TTPenalize`, `ChanceBranchDepth`,
`SeekMate`, `PruningTweaks`, `ThreatCache`, ...) are experimental search
changes, off by default, to be tested with an SPRT; each one is described by
a comment in [`src/engine.cpp`](src/engine.cpp).

### Debugging commands

* `d` prints the board (`X` / `x` for dark pieces), the FEN and the hash key.
* `go perft <depth>` counts leaf nodes; a move of a dark piece is expanded into
  one child per kind of piece left in the mover's pool (each kind counted
  once), also at the last ply. It does not need a network.
* `eval` prints the evaluation of the current position, term by term.
* `evalraw` prints the raw network output (PSQT and positional parts, in
  internal units), the layer stack used, and `final`: the static evaluation
  after the scaling of `evaluate()` (optimism 0; `none` in check). With the
  option `RawScore`, `info ... score cp` reports internal units instead of
  normalized centipawns. Both are used by the [trainer](trainer/).
* `rule` prints `info string rule <verdict>`: what the rules of the search
  decide in the current position for the side to move, `none` (the game goes
  on), `draw`, `win` or `loss` (third occurrence of a position, perpetual check
  or chase, 40-move rule), or `invalid` for an invalid position. The referees
  of [fishtest](fishtest/) and `trainer/match.py` use it to end games with the
  engine's rules.
* `bench [hash threads limit fenfile limittype]` searches a fixed set of
  positions (default `16 1 13 default depth`) and prints the node count, the
  signature of the build; `speedtest` measures the speed. Both need a network.
* `flip` swaps the colors of the position; `export_net [file]` writes the
  network in use; `compiler` prints the compiler and the build options.

## Compiling

On Unix-like systems, use the Makefile in `src`:

```
cd src
make -j build
```

Run `make help` for the list of targets and architectures (e.g.
`make -j build ARCH=x86-64-avx2`). On Windows, build with MSYS2 (`make` and
`g++` of MinGW-w64).

Every build first runs `make net` (`scripts/net.sh`), which fetches the
network of the hidden ref `refs/assets/net` (from `origin`, then from this
repository) into `src/maskfish.nnue`, and embeds it in the binary (the build
prints `embedded net: maskfish.nnue`), which then needs no network file.
`make -j build embed=no` builds a binary without it, which needs the `EvalFile`
option.

If the fetch fails (no connection), the build uses the existing copy of
`src/maskfish.nnue`. Without one (the source is not a git clone, e.g. a
downloaded archive), it goes on without a network: the build prints `embedded net: none`, and the `EvalFile` option must
be set to a network at run time.

`make profile-build` also runs a benchmark to collect profile data, which needs
a compatible network (it stops before compiling when there is none). When the
binary cannot run natively (cross compilation), `RUN_PREFIX` gives the command
used to run it, e.g. `make profile-build COMP=mingw RUN_PREFIX=wine`
(`WINE_PATH` is still accepted but deprecated).

The network architecture is chosen at build time; the experimental
architectures (`nnue=att`, `threats=yes`, `layers=2`, `psqt=no`, `ctx=yes`,
`lsblend=yes`, `meta=yes`, `quant=2`, `stacks=32`) need a network trained for
them (see the [trainer](trainer/README.md#7-network-architectures-and-training-options)).
Run `make clean` when changing them.

### Universal binary

`make -j build ARCH=x86-64-universal` (as Pikafish) builds one binary that runs
on every x86-64 CPU at the speed of the matching build: it holds the builds
`x86-64`, `x86-64-sse41-popcnt`, `x86-64-avx2`, `x86-64-bmi2`,
`x86-64-avxvnni`, `x86-64-avx512bw`, `x86-64-vnni512` and `x86-64-avx512icl`,
and selects the best one for the CPU when it starts (`compiler` prints the
selected one; AMD Excavator and Zen 1/2, whose BMI2 is slow, get `x86-64-avx2`).
`ARCH=arm64-universal` does the same with `armv8` / `armv8-dotprod` (Linux,
Android, Windows; not macOS, where `ARCH=apple-silicon` is the build to use).
`make macos-lipo` makes a macOS binary for both Intel (universal) and Apple
silicon:

```
make -j build ARCH=x86-64-universal COMP=gcc      # Linux
make -j build ARCH=x86-64-universal COMP=mingw    # Windows, from Linux (MinGW-w64)
make -j macos-lipo COMP=clang                     # macOS
make -j build ARCH=arm64-universal COMP=ndk       # Android (NDK r27c or later)
```

* Every build is compiled separately in `src/temp_builds/` (about eight times
  the time of one build); `make clean` deletes them. The architecture options
  of the network (`threats=yes`, `layers=2`, ...) apply to all of them. They
  are all made again when a source file, the Makefile or a build option
  (compiler, network options, PGO or not, flags) changes, and only then.
* `make -j profile-build ARCH=x86-64-universal` runs the PGO bench of every
  build, which needs a CPU that runs all of them, or
  `RUN_PREFIX="/path/to/sde -future --"` (Intel SDE). Cross builds use
  `make build` (no PGO): under Wine the engine stops at startup (NUMA
  detection, as Pikafish's), so a Windows binary is checked on Windows.
* It needs `optimize=yes` and `debug=no` (it uses the LTO-optimized object of
  every build). On Linux the C++ runtime is linked statically; with
  `COMP=clang UNIVERSAL_LIBCXX=yes` it is libc++ (as Pikafish's releases).
  Windows and Android binaries are static.
* The network is embedded as in the other builds (`embed=yes`), once: every
  build of the binary uses the same copy.
* `MASKFISH_UNIVERSAL_ARCH=<build>` in the environment forces a build (the
  CPU must support it), e.g. to compare them on one machine;
  `MASKFISH_UNIVERSAL_ARCH=list` lists the builds of the binary.

## Contributing

See the [Contributing Guide](./CONTRIBUTING.md). Changes to the search or
evaluation should be validated by testing against the current version of
Maskfish, e.g. with the local [fishtest](fishtest/README.md) or
`trainer/match.py`.

Useful checks when changing Jieqi-specific code (the scripts in `tests/` are
run from `src`, after building):

* build with `make -j build debug=yes sanitize="address undefined"` and run
  searches from positions with dark pieces: `python3 ../tests/random_games.py`
  plays random games with reveals and hidden captures and searches along the
  way (it needs a network), and `../tests/invalid_input.sh` checks that
  invalid positions and moves are rejected and never searched (no network
  needed);
* compare `go perft` counts before and after changes to move generation or to
  `Position` (`../tests/perft.sh` checks a few recorded counts; it does not
  need a network);
* compare the key printed by `d` for the same position reached through
  `position startpos moves ...` and set directly with `position fen ...`; they
  must be equal;
* `../tests/signature.sh` prints the `bench` signature and
  `../tests/reprosearch.sh` checks that searches are reproducible (both need a
  network).

## Terms of use

Maskfish is free and distributed under the
[**GNU General Public License version 3**](./Copying.txt) (GPL v3). Essentially,
this means you are free to do almost exactly what you want with the program,
including distributing it among your friends, making it available for download
from your website, selling it (either by itself or as part of some bigger
software package), or using it as the starting point for a software project of
your own.

The only real limitation is that whenever you distribute Maskfish in some way,
you MUST always include the license and the full source code (or a pointer to
where the source code can be found) to generate the exact binary you are
distributing. If you make any changes to the source code, these changes must
also be made available under GPL v3.
