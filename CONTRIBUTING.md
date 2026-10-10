# Contributing to Maskfish

Welcome to the Maskfish project! We are excited that you are interested in
contributing. This document outlines the guidelines and steps to follow when
making contributions to Maskfish.

## Table of Contents

- [Building Maskfish](#building-maskfish)
- [Making Contributions](#making-contributions)
  - [Reporting Issues](#reporting-issues)
  - [Submitting Pull Requests](#submitting-pull-requests)
- [Code Style](#code-style)
- [License](#license)

## Building Maskfish

See the [Compiling](./README.md#compiling) section of the README. In short:

```
cd src
make -j build
```

## Making Contributions

### Reporting Issues

If you find a bug, please open an issue on the
[issue tracker][issue-tracker-link]. Be sure to include relevant information
like your operating system, build environment, the network used, and a detailed
description of the problem (ideally the `position` and `go` commands that
reproduce it).

### Submitting Pull Requests

- Functional changes (search, evaluation, move generation) need to be tested
  against the current version of Maskfish, for example with an SPRT test run
  with `match.py` of [nnue-pytorch][nnue-pytorch-link]. The accompanying pull
  request should include the test results and the new bench.

- Changes to move generation or to `Position` should keep the `go perft`
  counts checked by `tests/perft.sh` unchanged.

- Non-functional changes (e.g. refactoring, code style, documentation) do not
  need to be tested, unless they might impact performance.

- Provide a clear and concise description of the changes in the pull request
  description.

_First time contributors should add their name to [AUTHORS](./AUTHORS)._

## Code Style

Changes to Maskfish C++ code should respect our coding style defined by
[.clang-format](.clang-format). You can format your changes by running
`make format`. This requires clang-format version 20 to be installed on your system.

## License

By contributing to Maskfish, you agree that your contributions will be licensed
under the GNU General Public License v3.0. See [Copying.txt](./Copying.txt) for
more details.

Thank you for contributing to Maskfish and helping us make it even better!

[issue-tracker-link]: https://github.com/official-maskfish/maskfish/issues
[nnue-pytorch-link]: https://github.com/official-maskfish/nnue-pytorch
