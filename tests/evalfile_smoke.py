#!/usr/bin/env python3
"""Checks which network the engine loads (src/uci.cpp, load_net):

  python3 tests/evalfile_smoke.py BIN [--net NET]

In this order: the file EvalFile names (the default maskfish.nnue: in the current directory, else next to the
binary), then the network embedded in the binary (CMake MASKFISH_EMBED_NET) when EvalFile is empty, is the default
name and there is no such file, or names a file that cannot be loaded ("EvalFile load failed" is printed); without
either, "ERROR: no network". With --net, the embedded network must be NET (same md5 of the decoded net). Runs a copy
of BIN in a temporary directory. Python standard library only.
"""
import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(binary, commands, cwd):
    p = subprocess.run([str(binary)], input="".join(c + "\n" for c in commands) + "quit\n", capture_output=True,
                       text=True, cwd=cwd, timeout=120)
    return [line[12:] for line in p.stdout.splitlines() if line.startswith("info string ")], p.stdout


def md5(line):
    m = re.search(r"md5 ([0-9a-f]{32})", line)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("--net", help="the network that must be the embedded one")
    args = ap.parse_args()
    failures = []

    def check(what, ok, lines):
        print(f"{'ok  ' if ok else 'FAIL'} {what}: {' | '.join(lines)[:220]}")
        if not ok:
            failures.append(what)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        home, cwd = tmp / "bin", tmp / "cwd"
        home.mkdir()
        cwd.mkdir()
        exe = home / "maskfish"
        shutil.copy2(args.binary, exe)
        testnet = tmp / "testnet.hnn"
        subprocess.run([sys.executable, str(ROOT / "tests/testnet.py"), str(testnet)], check=True, capture_output=True)
        (cwd / "garbage.nnue").write_text("not a network\n")

        _, out = run(exe, ["compiler"], cwd)
        embedded = "embedded network none" not in out
        print(f"embedded network: {'yes' if embedded else 'none'}")
        fallback = "embedded network maskfish.nnue loaded" if embedded else "ERROR: no network"

        def falls_back(lines):
            return any(line.startswith(fallback) for line in lines)

        lines, _ = run(exe, ["isready"], cwd)
        check("startup, no maskfish.nnue anywhere", len(lines) == 1 and falls_back(lines), lines)
        if embedded and args.net:
            want, _ = run(exe, [f"setoption name EvalFile value {Path(args.net).resolve()}"], cwd)
            same = md5(lines[0]) is not None and md5(lines[0]) == md5(want[0])
            check(f"the embedded network is {args.net}", same, [lines[0], want[0]])

        shutil.copy2(testnet, home / "maskfish.nnue")
        lines, _ = run(exe, ["isready"], cwd)
        check("startup, maskfish.nnue next to the binary",
              lines == [l for l in lines if l.startswith(f"EvalFile loaded {home / 'maskfish.nnue'} (")] and lines, lines)
        (home / "maskfish.nnue").unlink()
        shutil.copy2(testnet, cwd / "maskfish.nnue")
        lines, _ = run(exe, ["isready"], cwd)
        check("startup, maskfish.nnue in the current directory",
              len(lines) == 1 and lines[0].startswith("EvalFile loaded maskfish.nnue ("), lines)
        (cwd / "maskfish.nnue").unlink()

        lines, _ = run(exe, [f"setoption name EvalFile value {testnet}"], cwd)
        check("EvalFile names a network", len(lines) == 1 and lines[0].startswith(f"EvalFile loaded {testnet} ("), lines)
        for what, value in (("a missing file", str(tmp / "missing.nnue")), ("a file that is not a network", "garbage.nnue")):
            lines, _ = run(exe, [f"setoption name EvalFile value {value}"], cwd)
            check(f"EvalFile names {what}",
                  len(lines) == 2 and lines[0].startswith("EvalFile load failed") and falls_back(lines[1:]), lines)
        lines, _ = run(exe, ["setoption name EvalFile value <empty>"], cwd)
        check("EvalFile empty", len(lines) == 1 and falls_back(lines), lines)

    if failures:
        sys.exit("evalfile_smoke: FAIL: " + "; ".join(failures))
    print("evalfile_smoke: pass")


if __name__ == "__main__":
    main()
