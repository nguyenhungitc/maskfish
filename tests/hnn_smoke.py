#!/usr/bin/env python3
"""Smoke test of the HNN evaluator (src/hnn/): writes a small net of pseudo-random weights in the raw JQHNNUE1
format, loads it and compares the static evaluation of the positions of one game with reference values for the
same file, computed by the evaluator's original implementation. With --print the values are printed instead of
checked.

    python3 tests/hnn_smoke.py build/maskfish
"""
import os
import struct
import subprocess
import sys
import tempfile

W, SCHEME = 128, 3          # FT width, king-bucket scheme (6 buckets, mirrored)
NFEAT = 6 * 1440 + 30

# A recorded game (the first 70 plies of a long-clock game), as the tokens of an observer who sees every reveal.
TOKENS = ("a3a4P i6i5r h0g2P h9g7a b0c2N c6c5a c3c4P e6e5n e3e4C f9e8n c4c5 e9f9 f0e1C e5c4 c2d4 c4b2 h2b2N "
          "i5f5 g0e2A h7h0p e1f1 f5c5 b2d3 h0i0 d3f4 e8f6 f4e6 c5f5 e4f4 f5f4 e6f4 f9e9 f4g6 d9e8b g6f4 c9e7c "
          "d0e1P b9d8p f1f6 g7f6 f4h5 f6e5 d4f3 e5f4 h5f4 b7b1p g3g4B i9i3p f3d2 i0h0 d2b1 h0g0 g4e6 e8c6 e2f1 "
          "c6a4 a0a4R a6a5b f4d5 e7f7 f1g0 g9i7p e6g4 a9a7c a4a5 a7d7 d5f6 d7d3 a5e5 e9f9").split()

# Evaluations of the start position and of the position after each token, side to move's view.
# (EXPECTED, at the end of the file.)


def lcg(seed):
    x = seed
    while True:
        x = (x * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        yield x >> 33


def write_net(path):
    g = lcg(20261031)

    def ints(n, lo, hi):
        span = hi - lo + 1
        return [lo + next(g) % span for _ in range(n)]

    with open(path, "wb") as f:
        f.write(b"JQHNNUE1" + struct.pack("<8I", 1, W, NFEAT, SCHEME, 16, 16, 16, 32))
        f.write(struct.pack(f"<{W}h", *ints(W, -30, 119)))
        for _ in range(NFEAT):
            f.write(struct.pack(f"<{W}h", *ints(W, -24, 24)))
        f.write(struct.pack(f"<{NFEAT * 16}i", *ints(NFEAT * 16, -3000, 3000)))
        for _ in range(16):
            f.write(struct.pack("<16i", *ints(16, -4000, 4000)))
            f.write(struct.pack(f"<{W * 16}b", *ints(W * 16, -40, 40)))
            f.write(struct.pack("<32i", *ints(32, -3000, 3000)))
            f.write(struct.pack("<1024b", *ints(1024, -50, 50)))
            f.write(struct.pack("<i", *ints(1, -3000, 3000)))
            f.write(struct.pack("<32b", *ints(32, -80, 80)))


def evals(engine, net):
    cmds = [f"setoption name EvalFile value {net}"]
    for k in range(len(TOKENS) + 1):
        cmds += ["position startpos" + (" moves " + " ".join(TOKENS[:k]) if k else ""), "eval"]
    out = subprocess.run([engine], input="\n".join(cmds + ["quit"]) + "\n", capture_output=True, text=True).stdout
    if "EvalFile load failed" in out or "bad move" in out:
        sys.exit("hnn_smoke: " + [l for l in out.splitlines() if "failed" in l or "bad move" in l][0])
    return [int(l.split(" eval ")[1]) for l in out.splitlines() if l.startswith("classical ")]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    with tempfile.TemporaryDirectory() as d:
        net = os.path.join(d, "smoke.hnn")
        write_net(net)
        got = evals(sys.argv[1], net)
    if "--print" in sys.argv:
        print(got)
        return
    bad = [(k, e, g) for k, (e, g) in enumerate(zip(EXPECTED, got)) if e != g]
    if len(got) != len(EXPECTED) or bad:
        sys.exit(f"hnn_smoke: FAIL, {len(got)} values, {len(bad)} differ; first (ply, expected, got): {bad[:3]}")
    print(f"hnn_smoke: pass, {len(got)} evaluations equal to the reference values")


EXPECTED = [
    -957, -393, -946, -597, 397, -1309, -44, -1996, 627, -1636, 1132, 53, 1774, -834, 1419, -824, 1704,
    -213, 1572, -997, 2396, -423, 2646, -2255, 2729, -1622, 3762, -2500, 3128, -2407, 3663, -795, 3380,
    -2007, -1120, -1325, -901, -1136, -904, -1558, -1452, -480, -1779, -776, -1211, -1150, -1075, 135, 968,
    -245, 104, -9, 341, 1073, -516, 1279, 590, 722, 832, 1000, -78, -631, 92, -425, -2087, -703, -1128,
    -679, -1303, -772, -480,
]

if __name__ == "__main__":
    main()
