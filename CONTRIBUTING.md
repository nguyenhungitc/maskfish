# Contributing to Maskfish

Welcome to the Maskfish project! Maskfish is a free UCI engine for **Jieqi**
(揭棋, xiangqi with face-down pieces). Our goal is simple: **luck is a skill too**.

## About this project (please read)

Maskfish was started by a Jieqi enthusiast who is **not a professional C++
programmer** and who develops much of the code together with an AI assistant
(Claude). That means:

- The project has real ideas (chance nodes over the urn of face-down
  identities, urn features in the network, an independent rules referee), but
  it also has **rough edges, untested assumptions and probably bugs**.
  Reviews, critiques and corrections are very welcome.
- You do **not** need to be a C++ expert to help. Strong Jieqi players, people
  who know NNUE/ML, testers and writers are all needed just as much as
  programmers.
- If you are an experienced engine developer, your review of the search and
  evaluation code is among the most valuable things you can give us.

## Table of Contents

- [Ways to help](#ways-to-help)
- [Wanted list](#wanted-list)
- [Building Maskfish](#building-maskfish)
- [Reporting issues](#reporting-issues)
- [Submitting pull requests](#submitting-pull-requests)
- [Testing your change](#testing-your-change)
- [AI-assisted contributions](#ai-assisted-contributions)
- [Code style](#code-style)
- [Be kind](#be-kind)
- [License](#license)

## Ways to help

### I love Jieqi, but I don't write code

- **Play against Maskfish** (with the local board, `python3 tools/play.py`, or
  in a GUI that supports Jieqi and the notation described in the
  [README](./README.md#uci-protocol)) and tell us where it plays badly: reveals
  it should not make, captures it should not take, endgames it misjudges. Send
  us the game (moves and the identities of the revealed pieces; the local board
  saves its games in `runs/play/games.jsonl`).
- **Check the rules.** Jieqi has several regional rule variants. The rules we
  implement are listed in the README. If something differs from the rules you
  play by (advisors and elephants after a reveal, soldiers revealed on the back
  rank, the 80-ply no-capture rule, repetition and chase rules), open an issue
  and tell us which rules are the common ones. Where the published rules are
  silent, the referee (`tools/native/`) follows the conventions listed in the
  README; a report that it disagrees with another implementation or a
  published ruling should name the rule and give the game or position.
- **Share positions.** Interesting openings, tricky reveal situations,
  well-known endgames, and positions where you believe the engine's
  evaluation is wrong are all useful test material.
- **Translate and document.** Tutorials, FAQs and README translations help
  Jieqi players who are new to UCI engines.
- **Spread the word** to GUI authors and Jieqi communities, so that more
  people can play with and against the engine.

### I know AI, machine learning or NNUE

Maskfish needs a strong Jieqi network, and the network is where a lot of the
strength will come from. You can help with:

- **Network training and data.** Self-play data generation (`selfplay`),
  training (`train/train.py`) and the automated loop (`scripts/pipeline.sh`,
  `scripts/autotune.sh`, see [scripts/README.md](./scripts/README.md)) are in
  this repository. Better data, better training settings and better
  architectures are all welcome.
- **Feature design.** The current feature set (king-bucketed piece-square
  features with a face-down piece kind, plus 30 urn thresholds) is a first
  attempt. Ideas for representing hidden information better are especially
  interesting.
- **Other architectures** (`train/train.py --width`, `--kb-scheme`, `--l1`,
  `--l2`, `--sq2`, see the README): compare them against the default network,
  trained on the same data, and report the results.
- **Handling hidden information.** Chance nodes, how their outcomes are
  averaged and searched (`ChanceK`, `ChanceMix`, `ChanceXDepth`, ...) and
  how the urn is modelled are open research questions. If you have experience
  with search under uncertainty, we would like to hear it.

### I write C++

- Fix bugs, improve speed, add tests.
- Review the Jieqi-specific code (`src/position.cpp`: move generation,
  repetition and chase rule; the chance nodes of `src/search.cpp`; the
  referee in `tools/native/`) for correctness.
- Make the code easier to read and maintain.

## Wanted list

Concrete tasks where help is welcome. Open an issue before starting a large
one so that we don't duplicate work.

1. **Tune the search parameters.** Every search parameter is a UCI spin option
   (the `params()` table of `src/search.cpp`), and many were set by hand. SPSA
   runs with `tools/spsa.py` or the local fishtest.
2. **Validate the experimental options** (`ThreatRevRoot`, `ThreatRevScale`,
   `ThreatRevLmr`, `ThreatRevOrder`, `ChanceMixLow`, `RootDarkPen`, ...).
   Each one should be proven by an SPRT test or removed. Options that fail
   should be deleted, not kept as dead code.
3. **Check `ChanceK`**, the scale at which chance outcomes are averaged,
   against the score scale of the current network (`scripts/datatool.py
   fit-k` on its self-play data), and report what you find.
4. **A shared testing setup.** Maskfish has a local, single-machine
   [fishtest](./fishtest/README.md), but no distributed testing framework yet.
   A simple, reliable way for volunteers to donate CPU time to SPRT tests
   would speed up progress a lot.
5. **Test suites and benchmarks** of Jieqi positions, including positions with
   many face-down pieces, so that regressions are noticed.
6. **GUI and tool support.** Help GUI authors support the Jieqi FEN and move
   notation (the extra letters for reveals and captured face-down pieces) and
   report any protocol problems.
7. **Cross-engine comparisons.** Matches against other Jieqi engines with
   `tools/match.py` (refereed, colour-swapped pairs on seeded deals), with
   clearly stated conditions (time control, hardware, network, deals). An
   earlier measurement (the search at v0.15g, with a network of an older
   format, 500 pairs against each of six engines) has its records and report in
   the history of the repository, at commit `cba367f` and before.

## Building Maskfish

See the [Compiling](./README.md#compiling) section of the README. In short:

```
sh scripts/net.sh
cmake -S . -B build
cmake --build build -j4
```

`sh scripts/net.sh` fetches the **Jieqi network** of the hidden ref
`refs/assets/net` into `nets/maskfish.nnue`, which the build embeds in the
binary (CMake `MASKFISH_EMBED_NET`; without the file CMake warns and the binary
embeds none). The engine loads the file `EvalFile` names first (by default
`maskfish.nnue` in the working directory or next to the binary) and falls back
on the embedded network (see [Network](./README.md#network)). Networks of other
engines and xiangqi networks do not work.

The default build is for the build machine (`-march=native`). The published
binaries are universal binaries for Linux x86-64 and arm64
(`-DMASKFISH_UNIVERSAL=ON`, see [Universal
binary](./README.md#universal-binary)): one build per instruction set, the
best one chosen at startup. The network evaluator has kernels for AVX-512,
AVX-VNNI, AVX2, SSE and NEON and must give the same values with every one of
them: a change to `src/hnn/` or one that is specific to an instruction set
should be checked with the builds it affects, e.g. with
`python3 tests/universal_smoke.py build/maskfish` on a universal binary (every
build the CPU runs: perft, the bench signature, the network evaluator, the
embedded network), with `MASKFISH_UNIVERSAL_ARCH=<build>`, or for builds the
CPU lacks under Intel SDE (`sde64 -adl -- build/maskfish ...`).
`UNIVERSAL=1 bash tools/check_all.sh` also builds and checks a universal
binary.

## Reporting issues

If you find a bug, please open an issue on the
[issue tracker][issue-tracker-link]. The more of the following you include,
the faster we can reproduce it:

- Operating system, CPU, compiler and CMake options.
- The network file used (name or hash; the first `isready` prints it).
- The exact `position` and `go` commands. For Jieqi, remember that moves that
  reveal a piece carry an extra letter (for example `a3a4P`), and include the
  urns part of the FEN.
- What you expected, and what happened instead.
- For playing-strength problems: the game moves and revealed pieces.

Feature ideas, rule questions and "why does the engine play X?" questions are
welcome in the issue tracker too.

## Submitting pull requests

- **Functional changes** (search, evaluation, move generation) must be tested
  against the current version of Maskfish, with an SPRT test run with
  [`tools/match.py`][match-link] or the local [fishtest](./fishtest/README.md).
  Include the test command, its results and the new bench signature in the
  pull request description. Any claim about the engine's strength must come
  with the match command and its records.
- **Changes to move generation, to `Position` or to the referee** must pass
  `maskfish_difftest` and keep the perft counts (44, 1,926, 83,907).
- **Non-functional changes** (refactoring, code style, documentation,
  translations) do not need to be tested for strength, but a change to the
  engine must keep the bench signature, and one that might impact performance
  should say so.
- Keep pull requests **small and focused**: one idea per PR is much easier to
  review and test.
- Describe clearly what you changed and why.

_First time contributors should add their name to [AUTHORS](./AUTHORS)._

Don't worry if you are new to GitHub or to engine testing. Open a draft pull
request or an issue, and we will help you get the change into shape.

## Testing your change

One command builds the engine and the referee and checks them (it needs
CMake 3.20 or later, a C++20 compiler and Python 3.11 or later):

```
bash tools/check_all.sh
```

It fetches the network, builds in `build/check` (at nice 19 with 4 jobs;
`NICE`, `JOBS` and `BUILD` change that, and the script's header lists the
other settings), and checks:

- the engine's moves, check, chase and repetition verdicts against the
  referee's at every position of 1,000 seeded random games
  (`build/maskfish_difftest N SEED`);
- perft 1 to 3 from the start position (44, 1,926 and 83,907);
- the `bench` signature: 61,977 nodes with the test network of
  `tests/testnet.py` (a fixed seeded net, independent of the published one);
- invalid input (`tests/invalid_input.py`): positions, moves and option values
  the engine must refuse, odd arguments, and seeded fuzzing; the engine must
  refuse them, keep a sound position and never crash (run it on a sanitizer
  build too: the CI does, with more fuzzing);
- reproducible search (`tests/reprosearch.py`): the same `go nodes N`
  searches give the same lines after `ucinewgame` and in a new engine;
- random games (`tests/random_games.py`): random legal moves with random
  reveals and captures of face-down pieces, short searches along them; most
  useful on a sanitizer build with `-DJQ_VERIFY_LEGAL` (the CI's), which checks
  the legality and check of every move against a full test;
- the network evaluator against reference values, which network the engine
  loads (`EvalFile`, else the embedded one, else an error:
  `tests/evalfile_smoke.py`), pondering, the referee's
  library from Python, the local board, and the trainer on the CPU (when
  PyTorch is installed).

It prints a summary and fails if any step fails. The bench signature by hand:

```
python3 tests/testnet.py testnet.hnn
printf 'setoption name EvalFile value %s\nbench\nquit\n' "$PWD/testnet.hnn" | build/maskfish
```

A change to the search changes the signature; a change that is meant to keep
the search as it is (a refactoring, a speedup) must keep it. A change of the
signature on purpose updates `BENCH_NODES` in `tools/check_all.sh` and the
numbers in the README and in this file.

Other useful checks when changing Jieqi-specific code:

- Build with `-DMASKFISH_SANITIZE=ON` (AddressSanitizer and
  UndefinedBehaviorSanitizer, as the CI does) and run searches from positions
  with face-down pieces.
- Compare the FEN printed by `d` for the same position reached through
  `position startpos moves ...` and set directly with `position fen ...`.
- For strength changes, run an SPRT test against the unmodified version, and
  state the time control, number of games, deals and result.

Some files must stay in agreement:

| What | Where | Checked by |
|---|---|---|
| The rules | `tools/native/` (the referee) and `src/position.cpp` | `maskfish_difftest` |
| The C interface | `tools/native/include/jieqi/c_api.h` and `tools/jieqi/native.py` | the ABI version (5), checked on load: change it with the interface |
| The deals | `tools/jieqi/setup.py` | fixed seeds |
| The statistics | `tools/match.py` (`summarize`) and `tools/sprt.py` | |
| The network file | `src/hnn/hnnue.cpp` (loader), `train/train.py` (export), `train/hnn_file.py` | `tests/train_smoke.py`, `tests/hnn_smoke.py` |
| The network messages | `src/uci.cpp` (`load_net`: `EvalFile loaded`, `EvalFile load failed`, `embedded network ... loaded`) and the tools that read them (`tools/match.py`, `tests/*_smoke.py`, `scripts/pipeline.sh`, `fishtest/`) | `tests/evalfile_smoke.py`, `tests/train_smoke.py` |

## AI-assisted contributions

We use AI tools ourselves, so AI-assisted pull requests are welcome, with a few
rules:

- **You are responsible for what you submit.** Read and understand the code
  before sending it.
- **Test it.** AI-generated code that has not been built, run, and checked
  with `tools/check_all.sh` will not be merged.
- **Say so** in the pull request if a significant part was generated by an AI
  tool. This is not a penalty; it just tells reviewers where to look closely.
- Do not paste code from other projects unless its license is compatible with
  GPL-3.0, and mention its origin.

## Code style

There is no formatter configuration: follow the style of the code around your
change (C++20, 4-space indentation, lines up to about 120 columns, comments
that say why). Text files use LF line endings (`.gitattributes`). The match
harness, the referee's Python side and the checks use the
Python standard library only; the trainer (`train/`, PyTorch and NumPy) and the
record tools (`tools/label.py`, `scripts/datatool.py`, NumPy) are the
exceptions.

## Be kind

Maskfish is a hobby project run by volunteers across different countries and
backgrounds. Please be patient and respectful: assume good intentions, give
constructive feedback, and remember that every expert was once a beginner.

## License

By contributing to Maskfish, you agree that your contributions will be licensed
under the GNU General Public License v3.0. See [Copying.txt](./Copying.txt) for
more details.

Thank you for contributing to Maskfish and helping us make it even better!

[issue-tracker-link]: https://github.com/nguyenhungitc/maskfish/issues
[match-link]: ./tools/match.py
