#!/usr/bin/env python3
"""The test network: a fixed HNN net (raw JQHNNUE1, width 256, king-bucket scheme 2, the standard head) made from
a seeded generator, so that the bench signature does not depend on the published network.
Its PSQT holds piece values and its other weights are small pseudo-random integers: it plays like a material
counter with noise, which is enough to exercise every part of the evaluator and the search.

    python3 tests/testnet.py OUT.hnn      # writes the net and prints its sha256

Python 3 standard library only. The file is always the same (TESTNET_SHA256).
"""
import hashlib
import struct
import sys

W, SCHEME, KB = 256, 2, 2
NFEAT = KB * 16 * 90 + 30
L1, L2, L2IN, NST = 16, 32, 32, 16
VALUES = {2: 943, 3: 438, 4: 168, 5: 190, 6: 491, 7: 101, 8: 330}  # R N B A C P (as the trainer starts), face-down
TESTNET_SHA256 = "a7c33205b3860763bc1ac12bd2ec5c2b887f3bd7eff705fcff12b7b6dc04fa2e"


def lcg(seed):
    x = seed
    while True:
        x = (x * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        yield x >> 33


def build():
    g = lcg(20261111)

    def ints(n, lo, hi):
        span = hi - lo + 1
        return [lo + next(g) % span for _ in range(n)]

    out = [b"JQHNNUE1", struct.pack("<8I", 1, W, NFEAT, SCHEME, 16, NST, L1, L2)]
    out.append(struct.pack(f"<{W}h", *ints(W, -20, 100)))
    out.append(struct.pack(f"<{NFEAT * W}h", *ints(NFEAT * W, -12, 12)))
    psqt = []
    for f in range(NFEAT):
        kind = (f % (16 * 90)) // 90 if f < KB * 16 * 90 else -1   # 0..15: own K R N B A C P D, then the opponent's
        v = VALUES.get(kind % 8 + 1, 0) if kind >= 0 else 0
        v = -v if kind >= 8 else v
        noise = ints(16, -40, 40)
        psqt += [16 * v + n for n in noise]
    out.append(struct.pack(f"<{NFEAT * 16}i", *psqt))
    for _ in range(NST):
        out.append(struct.pack(f"<{L1}i", *ints(L1, -1000, 1000)))
        out.append(struct.pack(f"<{W * L1}b", *ints(W * L1, -8, 8)))
        out.append(struct.pack(f"<{L2}i", *ints(L2, -1500, 1500)))
        l2 = []
        for _ in range(L2IN // 4):          # file layout [L2IN/4][L2][4]; inputs 30 and 31 are padding (zero)
            for _ in range(L2):
                l2 += ints(4, -25, 25)
        for c in range(L2IN // 4 - 1, L2IN // 4):
            for o in range(L2):
                for k in range(4):
                    if 4 * c + k >= 2 * (L1 - 1):
                        l2[(c * L2 + o) * 4 + k] = 0
        out.append(struct.pack(f"<{L2 * L2IN}b", *l2))
        out.append(struct.pack("<i", *ints(1, -100, 100)))
        out.append(struct.pack(f"<{L2}b", *ints(L2, -10, 10)))
    return b"".join(out)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    data = build()
    h = hashlib.sha256(data).hexdigest()
    if h != TESTNET_SHA256:
        sys.exit(f"testnet: sha256 {h}, expected {TESTNET_SHA256}")
    open(sys.argv[1], "wb").write(data)
    print(h)


if __name__ == "__main__":
    main()
