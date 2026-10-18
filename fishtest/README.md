# Local fishtest for Maskfish

A web interface to queue tests of Maskfish patches and play them with an SPRT,
modeled on the fishtest of [Stockfish](https://tests.stockfishchess.org) and
[Pikafish](https://test.pikafish.org), but for **a single machine**: there are
no distributed workers, user accounts or approvals. The server builds both
engines, plays the games on the local cores, and stops the test when the SPRT
decides.

* Tests list with the usual results boxes (`LLR`, `Total W L D`, `Ptnml(0-2)`),
  green when the SPRT passes, red when it fails, yellow while running.
* New test form: repository (any fork, `owner/repo` or a local path), test and
  base branch / tag / commit, networks, UCI options, make options
  (`nnue=att`, `threats=yes`, `layers=2`), time control, SPRT bounds or a fixed
  number of games, priority.
* Test page: Elo, nElo, LOS, LLR graph, build log and benches, stop / resume /
  rerun / delete, diff (local, and a link to GitHub), the games as text.
* Networks page: upload networks exported by the trainer, download the latest
  released one.
* Optional **SPSA tuning**, as on fishtest.
* **Settings page**: pause / resume the runner, games at a time, build options,
  SPSA on/off, form defaults, password, cached builds, without restarting.

## Running

```
cd fishtest
pip install -r requirements.txt        # Flask
python fishtest.py
```

Then open <http://127.0.0.1:6543>. Requirements: Python 3.9+, git, make and
a C++17 compiler (the same as to build Maskfish). Linux, macOS and WSL are
tested; on Windows the builds need MSYS2 `make` and `g++` in the `PATH`.

Everything except the address and the data directory can then be changed on
the **Settings** page (see below). The command-line options give the starting
values (`python fishtest.py --help`):

| Option | Default | Meaning |
| --- | --- | --- |
| `--concurrency N` | cores − 1 | games played at the same time (divided by the threads of the test) |
| `--jobs N` | cores | `make -j` of the builds |
| `--arch` | `native` | `ARCH` of the builds |
| `--repo` | this repository on GitHub | default repository in the form |
| `--base-branch` | the repository's HEAD | default base branch in the form |
| `--port`, `--host` | `6543`, `127.0.0.1` | where the web server listens |
| `--password` | none | asked (HTTP basic auth, any user name) for every change; also `FISHTEST_PASSWORD` |
| `--data` | `fishtest/data` | database, git cache, builds, networks, games |
| `--timemargin` | `200` | ms an engine may exceed its clock before losing on time |
| `--spsa` | off | enable SPSA tuning tests; also `FISHTEST_SPSA=1` |
| `--no-runner` / `--runner-only` | | run only the web server / only the runner |

With `--host 0.0.0.0` the page is reachable from the local network: set a
password, since anyone could otherwise queue builds of any repository.

## How a test is run

1. **Submit.** The branches are fetched into a cache (`data/repo.git`) and
   resolved to commits, so a test always plays the commits it was submitted
   with.
2. **Build.** The runner takes the pending test with the highest priority (then
   the oldest), and builds each commit with `make build ARCH=... <make
   options>`. Builds are cached per commit, options and architecture. It then
   checks that each engine starts and searches with its network, and runs
   `bench` (shown on the test page; with a Jieqi network the bench depends on
   the network too, so it is not a commit signature as on fishtest).
3. **Play.** Game pairs run in parallel. The referee is the one of
   [`trainer/match.py`](../trainer/match.py) (its helpers are reused), with
   real clocks added:
   * both games of a pair use the **same hidden identities and random
     opening** (2–6 random plies by default), colors swapped: Jieqi luck
     mostly cancels within a pair;
   * each engine only knows what its side knows: reveals are public, a
     captured dark piece is only told to the capturer (option *full
     information* to tell both);
   * games end by mate / no legal move, adjudication (mate score, or both
     sides agreeing on |score| ≥ 3000 internal units for 4 moves each),
     threefold repetition (perpetual check loses), the 40-move rule, or 400
     plies;
   * time controls: `10+0.1` (seconds + increment), `nodes=N`, `depth=N`,
     `movetime=MS`. A crash, a timeout or a loss on time loses the game and is
     counted on the test page.
4. **Stop.** After every pair the LLR is updated; the test stops when it
   reaches a bound, when the number of games is reached, or when it is
   stopped from the page. A stopped test can be resumed: it keeps its games.

Only one test runs at a time, on all the configured cores. A test with a
higher priority starts when the current one ends (it does not interrupt it).
If the server is restarted, a test that was running goes back to the queue
and continues where it was.

## Docker

```
docker run -d --name fishtest -p 6543:6543 -v fishtest-data:/data nguyenhungitc/maskfish-fishtest
```

The image contains Python, git, make and g++: the engines are compiled in the
container when a test starts (`ARCH=native`, so for the CPU of the machine
running it), and everything is kept in the `/data` volume. Options of
`fishtest.py` can be appended (`... maskfish-fishtest --concurrency 15`), and
most of them changed later on the Settings page. The page listens on all
addresses in the container: set a password on the Settings page (or
`-e FISHTEST_PASSWORD=...`) if the port is reachable by others.

To build it yourself, from the root of the repository:

```
docker build -f fishtest/Dockerfile -t maskfish-fishtest .
```

The workflow `.github/workflows/fishtest-docker.yml` builds the image
(linux/amd64 and linux/arm64), checks that the pages answer, and pushes it to
Docker Hub as `nguyenhungitc/maskfish-fishtest:latest` and `:sha-<commit>`
whenever `fishtest/` (or the two trainer helpers it uses) changes on `pro`.
It needs the repository secret `DOCKERHUB_TOKEN` (a Docker Hub access token
with write access). In a fork, set the repository variable
`DOCKERHUB_USERNAME` (and, for another image name, `DOCKERHUB_IMAGE`).

## Settings page

* **Runner**: *Pause now* puts the current test back in the queue (its games
  are kept) and starts nothing until *Resume testing*; *Pause after the
  current test* lets it finish first. A banner on every page shows when the
  runner is paused. Tests can be submitted, stopped and resumed meanwhile
  (the tests list also has *Resume* buttons for stopped tests).
* **Games and builds**: games at a time, `make -j`, `ARCH`, time margin, SPSA
  on/off. A new concurrency applies at once: the current test's games are
  restarted with the new number of processes. A new `ARCH` applies from the
  next build.
* **Defaults of the New test form**: repository, base branch, URL of the
  default network.
* **Password**: set or remove the password asked for every change (stored
  hashed in the database; it replaces `--password`).
* **Storage**: sizes, and a button to delete the cached builds.

Values saved on this page win over the command line (the page shows which
ones differ); *Reset to the command-line values* forgets them.

## SPSA tuning

With SPSA enabled (`--spsa`, or the checkbox on the Settings page), the form offers a third test type, *SPSA tuning*. SPSA
tunes several numeric parameters at once: for every game pair, each parameter
is shifted up or down at random by `c_k`, one engine plays with `theta + c_k`,
the other with `theta - c_k`, and each parameter moves toward the side that
won the pair. Both sides are the test commit; the parameters are UCI spin
options, set before every pair. The gains and the defaults are those of
fishtest (A ratio 0.1, alpha 0.602, gamma 0.101); see `spsa.py`.

Parameters, one per line:

```
RevealQuietBase
RevealQuietSafety,24,-192,192,12,0.002
```

* `name,start,min,max,c_end,r_end`: the format of fishtest, which is also the
  one the engine prints at startup for the variables wrapped in `TUNE()`
  (`src/tune.cpp`), so these lines can be pasted as they are;
* or just the name of an existing UCI option, such as `RevealQuietBase`,
  `ChanceScaling` or `ChanceRiskAversion`: its default, min and max are read
  from the engine, with `c_end = (max - min) / 20` and `r_end = 0.002`.

The number of games sets the length of the tuning (one iteration per pair);
SPSA needs many games, so use a short time control. The test page shows the
current value of each parameter, a graph of their evolution (in % of their
range), the rounded values, and a button *SPRT with these values* that fills
the form with an SPRT of the tuned values (as UCI options) against the same
commit with its defaults.

When SPSA is disabled the form does not offer it and the server rejects SPSA
tests; SPSA tests already in the database can still be viewed, and queued
ones are still run.

## Statistics

As on fishtest, the results of a pair (0, ½, 1, 1½ or 2 points) are counted in
the pentanomial `Ptnml(0-2)`, and the SPRT is the generalized SPRT on these
counts, with α = β = 0.05 (bounds ±2.94).

* The bounds are in **normalized Elo** by default (nElo, as fishtest:
  `nElo = (score − ½) / σ_game × 800 / ln 10`), so that `[0, 2]` and
  `[-1.75, 0.25]` mean roughly the same as on fishtest. Jieqi games are
  noisier than xiangqi games, so a test needs more games than a Pikafish test
  with the same bounds.
* **Logistic Elo** bounds can be chosen instead (the model of
  `trainer/match.py`, e.g. `[0, 5]`).

The presets of the form follow fishtest: STC 10+0.1 with `[0, 2]`, LTC
60+0.6 with `[0.5, 2.5]`, non-regression `[-1.75, 0.25]`. The speed of
Maskfish (chance nodes) and of the machine are not those of the Stockfish
workers, so choose the time controls for your machine.

## Files

| File | |
| --- | --- |
| `fishtest.py` | entry point: starts the runner process and the web server |
| `app.py`, `templates/`, `static/` | web interface (Flask) |
| `runner.py` | the local worker: queue, builds, game pool, SPRT |
| `games.py` | referee: games, clocks, crashes |
| `builder.py` | git cache, commit resolution, builds, bench |
| `stats.py` | GSPRT (normalized / logistic) and Elo estimates |
| `spsa.py` | SPSA: parameters, gains, perturbations, updates |
| `db.py` | SQLite storage (`data/fishtest.db`) |
| `config.py` | settings of the Settings page, runner control, password |

`GET /api/runs/<id>` returns a test as JSON (for scripts).

Not included, compared to fishtest: distributed workers, user accounts and
approvals, and opening books (openings are random plies with
random hidden identities, as in `match.py`).
