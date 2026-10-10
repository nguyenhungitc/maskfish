# Maskfish

Maskfish is a free and strong UCI engine for **Jieqi**.

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
> Maskfish needs a **Jieqi network**. Xiangqi networks (such as Pikafish's
> `master-net`) are **not compatible** (they use a different feature set), see
> [Network](#network).

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
  pieces on the board; otherwise the engine prints
  `info string Invalid position: ...`. A board without exactly one king per
  side is not searched at all (`go` answers `bestmove (none)`) and the moves
  that follow the FEN are ignored.
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
`info string Invalid move ...` and ignores that move and all following ones.

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
  continuation depends on the piece that will be revealed.
* No `ponder` move is given when the best move reveals a piece, or when the
  expected reply would reveal one (the GUI could not send it back without the
  revealed piece).
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
| `ChanceScaling` | `0` | Score → win-rate scale used to average chance nodes; 0 means the built-in 360.8. Fit it with `fit_scaling.py` of [nnue-pytorch](https://github.com/official-maskfish/nnue-pytorch). |
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
`SharedHistory`. Test them with `match.py --opt-a Name=true` of nnue-pytorch; the ones
that fail should be removed rather than kept as dead options.

### Debugging commands

`d` prints the board (`X` / `x` for dark pieces), the FEN and the hash key.
`go perft <depth>` counts leaf nodes; a move of a dark piece is expanded into
one child per kind of piece left in the mover's pool (each kind counted once).
`evalraw` prints the raw network output (PSQT and layer parts, in internal
units) and the layer stack used. The option `RawScore` makes `info ... score cp`
report internal units instead of normalized centipawns. Both are used by the
trainer ([nnue-pytorch](https://github.com/official-maskfish/nnue-pytorch)).

## Network

The network file is set with the `EvalFile` option (default `maskfish.nnue`,
looked up next to the binary and in the working directory).

Maskfish uses its own feature set (`HalfKAv2_hm` with dark-piece and pool
features, feature hash `0x0d17b100`). Networks trained for Pikafish master
(xiangqi), including the one published as `master-net`, use a different feature
set and are rejected when loaded; the engine then prints an error and exits at
the first `go`.

### Training a network

The trainer for Maskfish's network is a separate repository,
[nnue-pytorch](https://github.com/official-maskfish/nnue-pytorch): a PyTorch trainer (self-play data generation,
training, export, and a bit-exact check against the engine). See its README.

## Compiling

On Unix-like systems, use the Makefile in `src`:

```
cd src
make -j build
```

Run `make help` for the list of targets and architectures.

Every build target first runs `make net`, which only checks whether
`src/maskfish.nnue` exists (nothing is downloaded). Put a Jieqi network at
`src/maskfish.nnue`, or point the `EvalFile` option to one at run time.

`make profile-build` also runs a benchmark to collect profile data, which needs
a compatible network.

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
such a network with `--arch att` in nnue-pytorch, and compare it with a default
network trained on the same data before adopting it.

`make -j build threats=yes` (possibly with `nnue=att`) adds **threat
inputs**, after the "full threats" networks of Stockfish and Pikafish: one
feature per (attacking piece, from square, to square, attacked piece) present
on the board, 92711 features (not those of the kings), where a dark piece attacks like the kind of
piece of its starting square. The feature transformer has 1024 outputs
instead of 2048. The engine is about a third slower and needs about 90 MB
more memory for the network. Train such a network with `--arch threats` (or
`att-threats`); see section 7.3 of the nnue-pytorch README.

`make -j build layers=2` (combinable with the options above) uses the layer
stacks of Pikafish's "Update nnue-architecture": the first layer has 32
outputs, all of them activated, the second layer gets a squared activation
too, the last layer is fed by all the activations of both, and the direct
term is the difference of the last two outputs of the first layer. The engine
is about 10% slower. Train such a network with `train.py --layers 2` (section
7.4 of the nnue-pytorch README); it can start from the feature transformer of a
network with the current layer stacks.

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
