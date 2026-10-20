## Overview

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

> [!IMPORTANT]
> Maskfish needs a **Jieqi network**. The binaries of the
> [latest release](https://github.com/nguyenhungitc/maskfish/releases/latest)
> embed one, and each archive also holds it as `maskfish.nnue`.
> Xiangqi networks (such as Pikafish's `master-net`) are **not compatible**
> (they use a different feature set), see [Network](#network).

## Binaries

The binaries are **universal binaries**: one file per system, which selects
the best build for the CPU when it starts (see
[Universal binary](#universal-binary)). They are built by the
[Binaries](.github/workflows/binaries.yml) workflow and attached to the
releases published since then (and to every run of the workflow, under
Actions):

| File | System |
| --- | --- |
| `maskfish-ubuntu-x86-64-universal.tar` | Linux x86-64 (any distribution with glibc; the C++ runtime is linked in) |
| `maskfish-windows-x86-64-universal.zip` | Windows x86-64 |
| `maskfish-macos-universal.tar` | macOS, Intel and Apple silicon |
| `maskfish-android-arm64-universal.tar` | Android arm64 (static) |

Each archive holds the binary (with the current network when it was built
embedded in it), `maskfish.nnue` (the same network, as a file), `README.md`,
`Copying.txt` and `AUTHORS`. `scripts/get_native_properties.sh` prints the
best architecture of the machine (for `make ARCH=...`) and the name of the
archive for it.

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
* Kings, checks, the flying-general rule, and the repetition rules (perpetual
  check and chase) are those of Pikafish.
* A game is drawn after 40 moves (80 plies) without a capture. Reveals do not
  reset this counter.

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
* Side to move: `w` or `b`.
* **Pool**: for each kind of piece, the number of pieces of that kind that are
  still unknown, as `<letter><digit>` pairs, e.g. `R2A2C2P5N2B2r2a2c2p5n2b2`.
  The pool also contains captured dark pieces whose identity was never
  reported. Each side needs at least as many pieces in its pool as it has dark
  pieces on the board.
* The position is checked: exactly one king per side, in its palace; for every
  kind of piece, the revealed pieces on the board plus the pool do not exceed
  the initial number (2 rooks, 5 pawns, ...); the pool condition above; and the
  side not to move is not in check. Otherwise the engine prints
  `info string Invalid position: ...`, ignores the moves that follow the FEN,
  and does not search (`go` answers `bestmove (none)`).
* No-capture counter in plies, and the move number.

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

### Jieqi search options

| Option | Default | Meaning |
| --- | --- | --- |
| `HiddenCaptureModel` | `true` | When the side the engine searches for captures a dark piece, the search branches over every identity that piece can have (it will be shown to the capturer), removing it from the pool in each branch. Dark pieces of that side captured by the opponent stay in the pool, since their identity is never learned. TT entries are kept separate per observing side. |
| `RevealQuietBase` | `-12` | Safe reveal bonus: constant part, added to the value of every quiet move of a dark piece. |
| `RevealQuietSafety` | `24` | Weight of the safety of the destination square (safe = +1, defended = 0, loose = -1, using visible pieces only). |
| `RevealQuietHighValue` | `-12` | Weight of the share of rooks, cannons and knights in the mover's pool. |
| `RevealQuietPhase` | `0` | Weight of the number of dark pieces still on the board. |
| `RevealQuietDiversity` | `0` | Weight of how mixed the mover's pool is. |
| `ProbCutDarkCapture` | `false` | ProbCut asks a larger margin (+64) and searches one ply less for captures of dark pieces. |
| `TTPenalize` | `false` | Lowers by one the depth of a TT entry that is deep enough but whose bound cannot produce a cutoff, so it is replaced sooner. |
| `ChanceBranchDepth` | `false` | In a reveal, the outcomes after the most likely one are searched one ply shallower. |
| `JieqiMaterial` | `false` | The material term of the evaluation scaling counts each dark piece at the expected major-piece value of its owner's pool (instead of the value of the piece that starts on its square). |
| `ChanceScaling` | `0` | Score → win-rate scale used to average chance nodes; 0 means the built-in 360.8. Fit it with [`trainer/fit_scaling.py`](trainer/fit_scaling.py). |
| `ChanceRiskAversion` | `0` | Risk preference, in percent of the standard deviation of the outcomes, of the side whose move causes a chance event: positive avoids uncertain reveals/captures, negative seeks them. |

The bonus is in internal score units and clamped to -192..192. Setting all
`RevealQuiet*` options to 0 disables it. The defaults are those of AB-JChess
and have not been re-tuned for Maskfish's networks; both features should be
validated with SPRT tests against a version where they are disabled.

The engine also has boolean options for search ideas under test, all off by
default and described next to their definition in `src/engine.cpp`:
`ContCorr4Ply`, `ContCorr6Ply`, `MultiCutHistory`, `CMHC`, `SeekMate`,
`PruningTweaks`, `LMRTweaks`, `FailHighRecovery`, `AspirationTweaks`,
`MainHistoryDecay`, `MalusDecay`, `BestMoveBonusScale`, `LMRHistoryDivisor`,
`ContHistOffset`, `CorrHistAsym`, `HindsightFirst`, `FollowPV`,
`CutoffCntTweak`, `NMPFailHighCount`, `MateTTExtension`, `StopEarly`,
`RootScoreEMA`, `StandPatNoTT`, `DecisiveTTAging`, `BigPawnHistory` and
`SharedHistory`. Test them with `trainer/match.py --opt-a Name=true`; the ones
that fail should be removed rather than kept as dead options.

#### Options ported from AB-JChess

Further ideas of [AB-JChess](https://github.com/lxsgx23/AB-JChess) (GPL-3.0),
each behind an option that is off by default (the default search is unchanged,
same bench signature). They have not been tested for Maskfish yet: validate
each one with SPRT before changing its default.

| Option | Default | Meaning |
| --- | --- | --- |
| `CorrHistReset` | `false` | Reset the correction histories at the start of every search (otherwise they are kept until `ucinewgame`). |
| `FrequentTimeCheck` | `false` | Check the time every 64 nodes instead of 512, to overrun short time limits less. |
| `VoteSinglePVOnly` | `false` | With several threads, choose the best thread by vote only when `MultiPV` is 1; in MultiPV analysis the main thread's lines are reported. |
| `RevealQsearchEvasions` | `false` | Also add the safe reveal bonus (`RevealQuiet*`) to quiet moves of dark pieces that evade a check in the quiescence search. |
| `ChanceModel` | `Winrate` | How the outcomes of a chance node are averaged. `Winrate`: in win-rate space (`ChanceScaling`). `Tables`: as AB-JChess, with score ↔ mass lookup tables (`ChanceTables`; without tables, `Winrate` is used). `Linear`: average of the scores clamped to ±2000, AB-JChess's `AggressiveLevel` 10 when combined with `ChanceRiskAversion` 3. `ChanceRiskAversion` applies to every model. |
| `ChanceTables` | `<empty>` | Directory holding the tables of `ChanceModel Tables` (`probability_score_to_mass.i32le` and `probability_mass_to_score.i32le`, AB-JChess's format). Make them with [`trainer/chance_tables.py`](trainer/chance_tables.py): `fit` from self-play data, `logistic` (equivalent to `Winrate`) or `extract` from an AB-JChess network. |

### Debugging commands

`d` prints the board (`X` / `x` for dark pieces), the FEN and the hash key.
`go perft <depth>` counts leaf nodes; a move of a dark piece is expanded into
one child per kind of piece left in the mover's pool (each kind counted once).
It does not need a network.
`evalraw` prints the raw network output (PSQT and layer parts, in internal
units), the layer stack used, and `final`: the static evaluation after the
scaling of `evaluate()` (optimism 0; `none` in check). The option `RawScore`
makes `info ... score cp` report internal units instead of normalized
centipawns. Both are used by the [trainer](trainer/).

## Network

The default network is embedded in the binary when it is built (see
[Compiling](#compiling)), as in Stockfish. The engine loads its network as
follows:

1. `EvalFile` set to a file: this file only (as given, absolute or relative to
   the working directory, then in the directory of the binary).
2. `EvalFile` not set (default `maskfish.nnue`, or empty): the network embedded
   in the binary, never a file.
3. Otherwise (no embedded network, or a file that cannot be loaded): the engine
   reports an error and exits at the first `go`.

A file named like the default network is loaded with a path, e.g.
`setoption name EvalFile value ./maskfish.nnue`.

The network of a release is the `maskfish.nnue` of its archives (the binaries
embed it). The build gets the current network automatically (see
[Compiling](#compiling)).

### Publishing a network

On GitHub, the network is only in the hidden git ref `refs/assets/net` (a
commit holding only `maskfish.nnue`, apart from the history of the code): every
build and every workflow takes it from there. The ref does not show on GitHub
(files, branches, tags) nor in `git branch -a`, and `git clone` does not fetch
it (`git ls-remote` lists it).

To publish a network, put it at `net/maskfish.nnue` (a local directory at the
root of the repository, ignored by git) and run:

```
scripts/push_net.sh
```

(`scripts/push_net.sh path/to/file.nnue` publishes another file.)

A release is made by hand: Actions > Release > Run workflow (on the branch to
release). The [Release workflow](.github/workflows/release.yml) builds the
binaries with the network of the ref and makes a release tagged `yymmdd` (date
in Vietnam; `yymmdd-2`, ... for another one the same day; or a given tag) with
their archives only: each binary embeds the network and each archive holds it,
so there is no separate network file.

Maskfish uses its own feature set (`HalfKAv2_hm` with dark-piece and pool
features, feature hash `0x0d17b100`). Networks trained for Pikafish master
(xiangqi), including the one published as `master-net`, use a different feature
set and are rejected when loaded; the engine then prints an error and exits at
the first `go`.

### Training a network

The trainer for Maskfish's networks is in the [`trainer`](trainer/) directory of
this repository: a PyTorch toolkit (self-play data generation, training,
export, and a bit-exact check against the engine). See
[`trainer/README.md`](trainer/README.md).

## Compiling

On Unix-like systems, use the Makefile in `src`:

```
cd src
make -j build
```

Run `make help` for the list of targets and architectures.

Every build target first runs `make net`, which fetches the network of the
ref `refs/assets/net` (see [Publishing a network](#publishing-a-network)) of
`origin`, then of this repository, into `src/maskfish.nnue`: a local copy,
replaced whenever the network of the ref changes. This needs a git clone of
the source.

The network is embedded in the binary (the build prints
`embedded net: maskfish.nnue`), which then needs no network file.
`make -j build embed=no` builds a binary without it, which needs the `EvalFile`
option.

If the fetch fails (no connection), the build uses the existing copy; without
one (not a git clone, no network published yet), it goes on without a network
(and the binary has no embedded network): set the `EvalFile` option to a
network at run time.

`make profile-build` also runs a benchmark to collect profile data, which needs
a compatible network. When the binary cannot run natively (cross compilation),
`RUN_PREFIX` gives the command used to run it, e.g.
`make profile-build COMP=mingw RUN_PREFIX=wine` (`WINE_PATH` is still accepted
but deprecated).

### Universal binary

`make -j build ARCH=x86-64-universal` (as Pikafish) builds one binary that runs
on every x86-64 CPU at the speed of the matching build: it holds the builds
`x86-64`, `x86-64-sse41-popcnt`, `x86-64-avx2`, `x86-64-bmi2`,
`x86-64-avxvnni`, `x86-64-avx512bw`, `x86-64-vnni512` and `x86-64-avx512icl`,
and selects the best one for the CPU when it starts (`compiler` prints the
selected one; AMD Zen 1/2, whose BMI2 is slow, get `x86-64-avx2`).
`ARCH=arm64-universal` does the same with `armv8` / `armv8-dotprod` (Linux,
Android, Windows; not macOS, where `ARCH=apple-silicon` is the build to use).
`make macos-lipo` makes a macOS binary for both Intel (universal) and Apple
silicon. The published [binaries](#binaries) are made this way:

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
* The network is not embedded (as in the other builds).
* `MASKFISH_UNIVERSAL_ARCH=<build>` in the environment forces a build (the
  CPU must support it), e.g. to compare them on one machine;
  `MASKFISH_UNIVERSAL_ARCH=list` lists the builds of the binary.

### Experimental network architecture

`make -j build nnue=att` builds the engine for a different network (not
compatible with the default one, and rejected by a default build):

* **attack buckets**: the feature transformer has 4 sets of weights per king
  bucket, selected by the visible attacking material of each side (has a
  revealed rook? a revealed knight or cannon?), 30240 inputs instead of 7560;
* **mid mirror**: when both kings are on the middle file, the board is
  mirrored according to its left/right balance, so that the network sees a
  canonical orientation (dark pieces count as the kind of their square).

Both ideas come from Pikafish (xiangqi) and are also used by AB-JChess. Train
such a network with `--arch att` in the trainer, and compare it with a default
network trained on the same data before adopting it.

`make -j build threats=yes` (possibly with `nnue=att`) adds **threat
inputs**, after the "full threats" networks of Stockfish and Pikafish: one
feature per (attacking piece, from square, to square, attacked piece) present
on the board, 92711 features (not those of the kings), where a dark piece attacks like the kind of
piece of its starting square. The feature transformer has 1024 outputs
instead of 2048. The engine is about a third slower and needs about 90 MB
more memory for the network. Train such a network with `--arch threats` (or
`att-threats`); see section 7.3 of the [trainer README](trainer/README.md).

`make -j build layers=2` (combinable with the options above) uses the layer
stacks of Pikafish's "Update nnue-architecture": the first layer has 32
outputs, all of them activated, the second layer gets a squared activation
too, the last layer is fed by all the activations of both, and the direct
term is the difference of the last two outputs of the first layer. The engine
is about 10% slower. Train such a network with `train.py --layers 2` (section
7.4 of the [trainer README](trainer/README.md)); it can start from the feature transformer of a
network with the current layer stacks.

`make -j build psqt=no` (combinable with the options above) builds the engine
for a network **without PSQT outputs**, as Pikafish's SFNNv17: the feature
transformer keeps only its L1 outputs and the evaluation is the layer-stack
output alone. Such networks have a different hash (`0x5E0D0000` is xored into
the feature-transformer hash), so they are rejected by a build with PSQT and
vice versa. Train such a network with `train.py --no-psqt` (section 7.5 of
the [trainer README](trainer/README.md)), like AB-JChess's networks, which
have no PSQT part either. The default build keeps the PSQT outputs.

`make -j build ctx=yes` (combinable with the options above) adds the
**inventory context** of AB-JChess: 16 values describing the uncertainty of
the position, from the side to move (its pool counts by piece type, the
opponent's, the numbers of dark pieces of both sides on the board, and the
numbers of their dark pieces captured without being identified), go through
a small layer (16 → 16, clipped ReLU) shared by the layer stacks, whose
outputs are 16 more inputs of the first layer of every layer stack. The cost
is negligible. Train such a network with `train.py --ctx` (section 7.6 of the
[trainer README](trainer/README.md)).

`make -j build lsblend=yes` (combinable with the options above) selects the
layer stacks as **AB-JChess**: the bucket of the piece count is raised by an
"uncertainty phase" (numbers of dark pieces on the board and of rooks,
cannons and knights left in the pools), and the evaluation is interpolated
between that layer stack and the next one, according to where the three
counts are within their intervals. About half of the positions then use two
layer stacks (the engine is about 10% slower). The prepared data changes:
see section 7.7 of the [trainer README](trainer/README.md).

`make -j build meta=yes` (combinable with the options above) adds the **meta
features** of AB-JChess to the inputs, 9 per side among 88 per feature
transformer bucket (7560 + 528 inputs): the numbers of rooks, cannons and
knights and of all the pieces in each pool, the number of dark pieces on the
board, the numbers of dark pieces of each side captured without being
identified, and the numbers of revealed pieces of each side attacked by
revealed enemy pieces. They change with almost every move and need the
attacks of the revealed pieces at each evaluation: the engine is about 30%
slower. The prepared data changes: see section 7.8 of the
[trainer README](trainer/README.md).

## Contributing

See the [Contributing Guide](./CONTRIBUTING.md). Changes to the search or
evaluation should be validated by testing against the current version of
Maskfish.

Useful checks when changing Jieqi-specific code:

* build with `make -j build debug=yes sanitize="address undefined"` and run
  searches from positions with dark pieces;
* compare `go perft` counts before and after changes to move generation or to
  `Position` (`tests/perft.sh`, run from `src`, checks a few recorded counts);
* compare the key printed by `d` for the same position reached through
  `position startpos moves ...` and set directly with `position fen ...`; they
  must be equal.

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

## Acknowledgements

The hidden capture model, the safe reveal bonus (`src/reveal.cpp`) and several
chance-node improvements are ported from [AB-JChess](https://github.com/lxsgx23/AB-JChess)
(GPL-3.0).


Maskfish is based on the `jieqi` branch of
[Pikafish](https://github.com/official-pikafish/Pikafish), which is derived from
[Stockfish](https://github.com/official-stockfish/Stockfish). Pikafish networks
are trained on
[data provided by the Pika Xiangqi Zero project](https://www.kaggle.com/datasets/pikacat/px0data),
which is made available under the
[Open Database License](https://opendatacommons.org/licenses/odbl/odbl-10.txt) (ODbL).
