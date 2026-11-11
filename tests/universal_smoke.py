#!/usr/bin/env python3
"""Checks a universal binary (CMake -DMASKFISH_UNIVERSAL=ON, src/universal.h):

  python3 tests/universal_smoke.py BIN

- `MASKFISH_UNIVERSAL_ARCH=list` lists the builds, and the build the binary starts by itself (`compiler`) is the
  one it marks as selected;
- every build this CPU runs, forced with MASKFISH_UNIVERSAL_ARCH: perft 1 to 3 from the start (44, 1,926,
  83,907), the bench signature with the test network (tests/testnet.py; BENCH_NODES of tools/check_all.sh) and
  the network evaluator against its reference values (tests/hnn_smoke.py);
- every build gets the network embedded in the binary (passed through a memfd), the same one, and its bench with
  that network gives the same nodes as every other build's;
- a build started from memory finds the network next to the binary: a copy of the binary with the test network
  as maskfish.nnue beside it, run from another directory, loads that file.

Builds this CPU does not run are listed as skipped. Python standard library only.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(binary, commands, arch=None, cwd=None):
    env = dict(os.environ)
    env.pop("MASKFISH_UNIVERSAL_ARCH", None)
    if arch:
        env["MASKFISH_UNIVERSAL_ARCH"] = arch
    p = subprocess.run([binary], input="".join(c + "\n" for c in commands) + "quit\n", capture_output=True,
                       text=True, env=env, cwd=cwd, timeout=600)
    return p.stdout + p.stderr


def build_of(out):
    m = re.search(r"^build (\S+)", out, re.M)
    return m.group(1) if m else None


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: universal_smoke.py BIN")
    binary = str(Path(sys.argv[1]).resolve())
    bench_nodes = re.search(r"^BENCH_NODES=(\d+)", (ROOT / "tools/check_all.sh").read_text(), re.M).group(1)
    env = dict(os.environ, MASKFISH_UNIVERSAL_ARCH="list")
    listing = subprocess.run([binary], capture_output=True, text=True, env=env, timeout=60).stdout
    print(listing, end="")
    rows = [line.split() for line in listing.splitlines()[1:] if line.strip()]
    builds = [(r[0], r[1] == "yes") for r in rows]
    selected = [r[0] for r in rows if r[-1] == "selected"]
    if not builds or len(selected) != 1:
        sys.exit("universal_smoke: FAIL, no build list from MASKFISH_UNIVERSAL_ARCH=list")
    failures = []
    auto = build_of(run(binary, ["compiler"]))
    print(f"started by itself: {auto}")
    if auto != selected[0]:
        failures.append(f"the binary started {auto}, the list selects {selected[0]}")

    with tempfile.TemporaryDirectory() as tmp:
        empty = Path(tmp) / "empty"
        empty.mkdir()
        embedded = "embedded network none" not in run(binary, ["compiler"])
        embedded_bench = set()
        testnet = Path(tmp) / "testnet.hnn"
        subprocess.run([sys.executable, str(ROOT / "tests/testnet.py"), str(testnet)], check=True,
                       capture_output=True)
        for name, ok in builds:
            if not ok:
                print(f"{name:18s} skipped (this CPU does not run it)")
                continue
            out = run(binary, ["compiler", "position startpos", "perft 1", "perft 2", "perft 3",
                               f"setoption name EvalFile value {testnet}", "bench"], arch=name)
            got_build = build_of(out)
            perft = " ".join(re.findall(r"^perft \d (\d+)", out, re.M))
            nodes = re.search(r"^Nodes searched: (\d+)", out, re.M)
            env = dict(os.environ, MASKFISH_UNIVERSAL_ARCH=name)
            smoke = subprocess.run([sys.executable, str(ROOT / "tests/hnn_smoke.py"), binary], capture_output=True,
                                   text=True, env=env, timeout=600)
            net = "-"
            if embedded:  # started where there is no maskfish.nnue: the embedded network
                eout = run(binary, ["isready", "bench"], arch=name, cwd=empty)
                m = re.search(r"^info string embedded network \S+ loaded .*md5 ([0-9a-f]{32})", eout, re.M)
                b = re.search(r"^Nodes searched: (\d+)", eout, re.M)
                net = f"{m.group(1)[:8]}, bench {b and b.group(1)}" if m else "NOT loaded"
                embedded_bench.add(net)
            line = (f"{name:18s} build {got_build}, perft {perft}, "
                    f"bench {nodes and nodes.group(1)}, hnn_smoke {'pass' if smoke.returncode == 0 else 'FAIL'}, "
                    f"embedded network {net}")
            print(line)
            if got_build != name or perft != "44 1926 83907" or not nodes or \
                    nodes.group(1) != bench_nodes or smoke.returncode != 0 or net == "NOT loaded":
                failures.append(line + (" " + smoke.stdout.strip() if smoke.returncode else ""))

        # The network next to the binary, found by the build the binary starts (from memory)
        home, elsewhere = Path(tmp) / "bin", Path(tmp) / "cwd"
        home.mkdir()
        elsewhere.mkdir()
        shutil.copy2(binary, home / "maskfish")
        shutil.copy2(testnet, home / "maskfish.nnue")
        out = run(str(home / "maskfish"), ["isready"], cwd=elsewhere)
        want = f"EvalFile loaded {home / 'maskfish.nnue'} ("
        print(f"network next to the binary: {'found' if want in out else 'NOT found'}")
        if want not in out:
            failures.append("the network next to the binary is not loaded: " + out.strip()[:300])

    if len(embedded_bench) > 1:
        failures.append(f"the builds differ with the embedded network (md5, bench): {sorted(embedded_bench)}")
    if failures:
        sys.exit("universal_smoke: FAIL\n  " + "\n  ".join(failures))
    print(f"universal_smoke: pass, {sum(ok for _, ok in builds)} of {len(builds)} builds checked")


if __name__ == "__main__":
    main()
