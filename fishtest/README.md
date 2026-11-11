# Local fishtest for Maskfish

A web interface to queue tests of Maskfish patches and play them with an SPRT,
on **a single machine**: there are no distributed workers, user accounts or approvals. The server builds both
engines, plays the games on the local cores, and stops the test when the SPRT
decides.

* Tests list with the usual results boxes (`LLR`, `Total W L D`, `Ptnml(0-2)`),
  green when the SPRT passes, red when it fails, yellow while running.
* New test form: repository (any fork, `owner/repo` or a local path), test and
  base branch / tag / commit, networks, UCI options, build options
  (`NAME=value` items given to CMake as `-DNAME=value`, such as `MASKFISH_LTO=ON`),
  time control, SPRT bounds or a fixed number of games, priority.
* Test page: Elo, nElo, LOS, LLR graph, build log and benches, stop / resume /
  rerun / delete, diff (local, and a link to GitHub), the games as text.
* Networks page: upload networks (`.hnn` from `train/train.py`, or compressed
  `.nnue`), fetch the default
  one (`maskfish.nnue` of the hidden ref `refs/assets/net`, published by
  `scripts/push_net.sh`).
* Optional **SPSA tuning**.
* **Settings page**: pause / resume the runner, games at a time, build options,
  SPSA on/off, form defaults, password, cached builds, without restarting.

## Running

```
pip install -r fishtest/requirements.txt        # Flask
python fishtest/fishtest.py
```

Then open <http://127.0.0.1:6543>. Requirements: Python 3.11+, git, CMake 3.20+
and a C++20 compiler (the same as to build Maskfish). fishtest runs from a clone of
the repository: it plays the games with `tools/match.py` and the referee of
`tools/native` and `tools/jieqi`, whose library it takes from `$JIEQI_NATIVE_LIBRARY`, else `build/lib` of the clone,
else compiles it once into the data directory (`g++` or `$CXX`). Tested on Linux.

Everything except the address and the data directory can then be changed on
the **Settings** page (see below). The command-line options give the starting
values (`python fishtest.py --help`):

| Option | Default | Meaning |
| --- | --- | --- |
| `--concurrency N` | cores − 1 | games played at the same time (divided by the threads of the test) |
| `--jobs N` | cores | parallel jobs of the builds (`cmake --build -j`) |
| `--arch` | `native` | architecture of the builds: `native` (`-march=native`), `portable` (no `-march`) or a `-march` value (`x86-64-v3`, ...) |
| `--repo` | this repository on GitHub | default repository in the form |
| `--base-branch` | the repository's HEAD | default base branch in the form |
| `--net-repo` (`--net-url`) | this repository on GitHub | repository whose hidden ref `refs/assets/net` holds the default network (URL, `owner/repo` or local path) |
| `--port`, `--host` | `6543`, `127.0.0.1` | where the web server listens |
| `--password` | none | asked (HTTP basic auth, any user name) for every change; also `FISHTEST_PASSWORD` |
| `--data` | `fishtest/data` | database, git cache, builds, networks, games |
| `--timemargin` | `200` | ms an engine may exceed its clock before losing on time |
| `--spsa` | off | enable SPSA tuning tests; also `FISHTEST_SPSA=1` |
| `--no-runner` / `--runner-only` | | run only the web server / only the runner |

With `--host 0.0.0.0` the page is reachable from the local network: set a
password, since anyone could otherwise queue builds of any repository.

Queuing a test builds and runs the code of a repository, so every change is
protected against requests sent by other web sites (cross-site request
forgery): a form is only accepted with the token of the browser session and
from the page's own origin. On a loopback address (the default `127.0.0.1`)
the server also answers only to the host names `localhost`, `127.0.0.1` and
`::1`, which blocks DNS rebinding. Repositories must be `https://` (or
`http://`, `ssh://`, `git://`, `file://`) URLs, `owner/repo`, `user@host:path`
or an existing local directory, and the architecture a plain word.

## How a test is run

1. **Submit.** The branches are fetched into a cache (`data/repo.git`) and
   resolved to commits, so a test always plays the commits it was submitted
   with.
2. **Build.** The runner takes the pending test with the highest priority (then
   the oldest), and builds each commit with CMake (target `maskfish`, Release, the
   architecture and the build options of the test). The engine may be at the
   root of the commit or in `engine/` (the layout before it moved to the root),
   so older commits such as v0.15g can be tested. Builds are cached per commit,
   options and architecture. It then checks that each engine starts and
   searches with its network, and runs `bench` (shown on the test page; it
   depends on the network too, so it is a commit signature only with the same
   network). A UCI option of the test that an engine does not have (Maskfish answers
   `info string unknown option`) or a network it cannot load stops the test
   with an error instead of being ignored.
3. **Play.** Game pairs run in parallel, each played by `play_game` of
   [`tools/match.py`](../tools/match.py), with the rules referee of
   [`tools/native`](../tools/native) and [`tools/jieqi`](../tools/jieqi):
   * both games of a pair use the **same hidden deal** (`Setup.from_seed`) and
     the **same random opening** (2–6 random legal plies by default, chosen by
     the referee), colours swapped: Jieqi luck mostly cancels within a pair;
   * each engine only knows what its side knows: reveals are public, a
     captured face-down piece is only told to the capturer (option *full
     information* to tell both);
   * the referee is the same for both engines and is not part of either
     build, so a patch that changes the engine's rules cannot judge its own
     games. It ends a game by mate or no legal move, repetition (a draw, or a
     loss for perpetual check or chase), the 80-ply no-capture rule, or 400
     plies, and an illegal move loses (counted on the test page);
   * adjudication (off by default, a value in centipawns): the side that
     reports a mate against itself resigns, and a game ends when both sides'
     last four scores are at least that value for the same side;
   * time controls: `10+0.1` (seconds + increment), `nodes=N`, `depth=N`,
     `movetime=MS`. A crash, a timeout or a loss on time (beyond the time
     margin) loses the game and is counted on the test page. Each game starts
     both engines afresh.
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

The image contains Python, git, CMake and g++, the referee (its library built
into the image), `tools/match.py` and `tools/sprt.py`: the engines are compiled in the container
when a test starts (`-march=native`, so for the CPU of the machine running it),
and everything else is kept in the `/data` volume. Options of
`fishtest.py` can be appended (`... maskfish-fishtest --concurrency 15`), and
most of them changed later on the Settings page. The page listens on all
addresses in the container: set a password on the Settings page (or
`-e FISHTEST_PASSWORD=...`) if the port is reachable by others.

To build it yourself, from the root of the repository:

```
docker build -f fishtest/Dockerfile -t maskfish-fishtest .
```

The image is for `linux/amd64` and `linux/arm64` (Docker picks the one of the
machine). The workflow `.github/workflows/fishtest-docker.yml` builds each on a
runner of that architecture, checks that the pages answer, that the referee's
library loads and that the image compiles the engine (perft 3), and pushes it to
Docker Hub as `nguyenhungitc/maskfish-fishtest:latest` whenever `fishtest/`,
`tools/match.py`, `tools/sprt.py` or the referee changes on `pro`, or when it is started
by hand. Only `latest` is kept: the workflow deletes the other tags of the
image.
It needs the repository secret `DOCKERHUB_TOKEN` (a Docker Hub access token
with write access). In a fork, set the repository variable
`DOCKERHUB_USERNAME` (and, for another image name, `DOCKERHUB_IMAGE`).

## Settings page

* **Runner**: *Pause now* puts the current test back in the queue (its games
  are kept) and starts nothing until *Resume testing*; *Pause after the
  current test* lets it finish first. A banner on every page shows when the
  runner is paused. Tests can be submitted, stopped and resumed meanwhile
  (the tests list also has *Resume* buttons for stopped tests).
* **Games and builds**: games at a time, build jobs, architecture, time margin,
  SPSA on/off. A new concurrency applies at once: the current test's games are
  restarted with the new number of processes. A new architecture applies from
  the next build.
* **Defaults of the New test form**: repository, base branch, repository of
  the default network (its ref `refs/assets/net`).
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
ChanceK
RootDarkPen,0,-300,300,15,0.002
```

* `name,start,min,max,c_end,r_end`;
* or just the name of a UCI spin option of Maskfish, such as `ChanceK`, `DarkLmr`
  or `RootDarkPen` (`uci` lists them all): its default, min and max are read
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

The results of a pair (0, ½, 1, 1½ or 2 points) are counted in
the pentanomial `Ptnml(0-2)`, and the SPRT is the generalized SPRT on these
counts, with α = β = 0.05 (bounds ±2.94).

* The bounds are in **normalized Elo** by default (nElo:
  `nElo = (score − ½) / σ_game × 800 / ln 10`), so that bounds such as
  `[0, 2]` and `[-1.75, 0.25]` mean the same sample size at any draw rate.
  Jieqi games are noisy (the deal and the reveals), so a test needs many
  games.
* **Logistic Elo** bounds can be chosen instead (the model of
  `tools/match.py --sprt`, e.g. `[0, 5]`).

The presets of the form: STC 10+0.1 with `[0, 2]`, LTC 60+0.6 with
`[0.5, 2.5]`, non-regression `[-1.75, 0.25]`. Choose the time controls for
the speed of your machine.

## Files

| File | |
| --- | --- |
| `fishtest.py` | entry point: starts the runner process and the web server |
| `app.py`, `templates/`, `static/` | web interface (Flask) |
| `runner.py` | the local worker: queue, builds, game pool, SPRT |
| `games.py` | game pairs through `tools/match.py` and the referee, engine checks |
| `builder.py` | git cache, commit resolution, CMake builds, bench, default network, referee library |
| `stats.py` | GSPRT (normalized / logistic) and Elo estimates |
| `spsa.py` | SPSA: parameters, gains, perturbations, updates |
| `db.py` | SQLite storage (`data/fishtest.db`) |
| `config.py` | settings of the Settings page, runner control, password |
| `Dockerfile`, `requirements.txt` | Docker image (built from the root of the repository), Python packages |

`GET /api/runs/<id>` returns a test as JSON (for scripts).

Not included, compared to fishtest: distributed workers, user accounts and
approvals, and opening books (openings are random plies on a seeded hidden
deal, as in `tools/match.py --random-plies`).
