# Contributing to Maskfish

*luck is a skill too*

Welcome to the Maskfish project! Maskfish is a free UCI engine for **Jieqi**
(揭棋, "hidden xiangqi"). Our goal is simple: **make the strongest open Jieqi
engine there is, together with the community**.

## About this project (please read)

Maskfish was started by a Jieqi enthusiast who is **not a professional C++
programmer** and who develops much of the code together with an AI assistant
(Claude). That means:

- The project has real ideas (chance nodes, dark-piece NNUE features, a hidden
  capture model), but it also has **rough edges, untested assumptions and
  probably bugs**. Reviews, critiques and corrections are very welcome.
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

- **Play against Maskfish** (in a GUI that supports Jieqi and the notation
  described in the [README](./README.md#uci-protocol)) and tell us where it
  plays badly: reveals it should not make, captures it should not take,
  endgames it misjudges. Send us the game (moves and the identities of the
  revealed pieces).
- **Check the rules.** Jieqi has several regional rule variants. The rules we
  implement are listed in the README. If something differs from the rules you
  play by (advisors and bishops after a reveal, pawns revealed on the back
  rank, the 40-move / 80-ply rule, repetition and chase rules), open an issue
  and tell us which rules are the common ones.
- **Share positions.** Interesting openings, tricky reveal situations,
  well-known endgames, and positions where you believe the engine's
  evaluation is wrong are all useful test material.
- **Translate and document.** Tutorials, FAQs and README translations help
  Jieqi players who are new to UCI engines.
- **Spread the word** to GUI authors and Jieqi communities, so that more
  people can play with and against the engine.

### I know AI, machine learning or NNUE

Maskfish needs a Jieqi network, and the network is where a lot of the strength
will come from. You can help with:

- **Network training and data.** Self-play data generation, training, export
  and verification are done with the
  [trainer](./trainer/) of this repository. Better
  data, better training settings and better architectures are all welcome.
- **Feature design.** The current feature set (`HalfKAv2_hm` plus dark-piece
  and pool features) is a first attempt. Ideas for representing hidden
  information better are especially interesting.
- **Experimental architectures** (`nnue=att`, `threats=yes`, `layers=2`,
  see the README): compare them against the default network, trained on the
  same data, and report the results.
- **Handling hidden information.** Chance nodes, the hidden capture model,
  risk aversion and expected-value scoring are open research questions. If you
  have experience with search under uncertainty, we would like to hear it.

### I write C++

- Fix bugs, improve speed, add tests.
- Review the Jieqi-specific code (`Position`, move generation, chance nodes,
  `src/reveal.cpp`) for correctness.
- Make the code easier to read and maintain.

## Wanted list

Concrete tasks where help is welcome. Open an issue before starting a large
one so that we don't duplicate work.

1. **Tune the reveal bonus.** The `RevealQuiet*` defaults have **not been
   tuned for Maskfish's networks**.
2. **Validate the experimental options** (`ChanceBranchDepth`,
   `HiddenCaptureModel`, `ChanceRiskAversion`, `JieqiMaterial`, ...). Each one
   should be proven by an SPRT test or removed. Options that fail should be
   deleted, not kept as dead code.
3. **Fit `ChanceScaling`** with `trainer/fit_scaling.py` and report
   the value you get.
4. **A shared testing setup.** Maskfish does not have a distributed testing
   framework like Fishtest yet. A simple, reliable way for volunteers to donate
   CPU time to SPRT tests would speed up progress a lot.
5. **Test suites and benchmarks** of Jieqi positions, including positions with
   many dark pieces, so that regressions are noticed.
6. **GUI and tool support.** Help GUI authors support the Jieqi FEN and move
   notation (the extra letters for reveals) and report any protocol problems.
7. **Cross-engine comparisons.** Matches against other Jieqi engines, with
   clearly stated conditions (time control, hardware, network, opening set).

## Building Maskfish

See the [Compiling](./README.md#compiling) section of the README. In short:

```
cd src
make -j build
```

The build fetches the **Jieqi network** from the hidden ref `refs/assets/net`
and embeds it in the binary (see [Publishing a
network](./README.md#publishing-a-network)); otherwise set the `EvalFile`
option. Xiangqi networks do not work.

The published binaries are universal binaries (`make -j build
ARCH=x86-64-universal`, see [Universal binary](./README.md#universal-binary)):
a change that is specific to an instruction set (SIMD code, `#ifdef USE_...`)
should be checked with the builds it affects, e.g.
`MASKFISH_UNIVERSAL_ARCH=x86-64 ./maskfish` for the baseline build of a
universal binary.

## Reporting issues

If you find a bug, please open an issue on the
[issue tracker][issue-tracker-link]. The more of the following you include,
the faster we can reproduce it:

- Operating system, CPU, compiler and build options.
- The network file used (name or hash).
- The exact `position` and `go` commands. For Jieqi, remember that moves that
  reveal a piece carry an extra letter (for example `a3a4P`), and include the
  pool part of the FEN.
- What you expected, and what happened instead.
- For playing-strength problems: the game moves and revealed pieces.

Feature ideas, rule questions and "why does the engine play X?" questions are
welcome in the issue tracker too.

## Submitting pull requests

- **Functional changes** (search, evaluation, move generation) must be tested
  against the current version of Maskfish, for example with an SPRT test run
  with [`trainer/match.py`][trainer-link]. Include the test
  results and the new bench in the pull request description.
- **Changes to move generation or to `Position`** must keep the `go perft`
  counts checked by `tests/perft.sh` unchanged.
- **Non-functional changes** (refactoring, code style, documentation,
  translations) do not need to be tested, unless they might impact
  performance.
- Keep pull requests **small and focused**: one idea per PR is much easier to
  review and test.
- Describe clearly what you changed and why.

_First time contributors should add their name to [AUTHORS](./AUTHORS)._

Don't worry if you are new to GitHub or to engine testing. Open a draft pull
request or an issue, and we will help you get the change into shape.

## Testing your change

Useful checks when changing Jieqi-specific code:

- Build with `make -j build debug=yes sanitize="address undefined"` and run
  searches from positions with dark pieces.
- Compare `go perft` counts before and after your change
  (`tests/perft.sh`, run from `src`).
- Compare the key printed by `d` for the same position reached through
  `position startpos moves ...` and set directly with `position fen ...`; they
  must be equal.
- For strength changes, run an SPRT test against the unmodified version, and
  state the time control, number of games, opening set and result.

## AI-assisted contributions

We use AI tools ourselves, so AI-assisted pull requests are welcome, with a few
rules:

- **You are responsible for what you submit.** Read and understand the code
  before sending it.
- **Test it.** AI-generated code that has not been built, run, and checked
  against `perft` and the other checks above will not be merged.
- **Say so** in the pull request if a significant part was generated by an AI
  tool. This is not a penalty; it just tells reviewers where to look closely.
- Do not paste code from other projects unless its license is compatible with
  GPL-3.0, and mention its origin.

## Code style

Changes to Maskfish C++ code should respect our coding style defined by
[.clang-format](.clang-format). You can format your changes by running
`make format`. This requires clang-format version 20 to be installed on your
system.

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
[trainer-link]: ./trainer/match.py
