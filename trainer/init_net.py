"""Writes a material-only starting network (no Jieqi network exists yet).

    python init_net.py gen0.nnue [--arch att]

The PSQT weights hold the piece values (a hidden piece counts as the average
value of the 15 pieces of a side) and the layer stacks output 0, so the
engine plays with a material evaluation. Use it to generate the first data,
train, then generate again with the trained network, and so on.
"""

import argparse

import torch

import jieqi_features as F
import serialize as S
from model import NNUE


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--arch", default="default", choices=sorted(F.ARCHS))
    args = ap.parse_args()
    F.set_arch(args.arch)
    if not F.PSQT:
        raise SystemExit("a material-only network needs the PSQT buckets: make one for the "
                         "architecture without -nopsqt, then train the -nopsqt network from "
                         "it (train.py --no-psqt --init-net, which takes its feature transformer)")
    torch.manual_seed(0)
    model = NNUE(init_psqt=True)
    with torch.no_grad():
        model.stacks.l1.weight.zero_()
        model.stacks.l1.bias.zero_()
        model.stacks.l2.weight.zero_()
        model.stacks.l2.bias.zero_()
        model.stacks.output.weight.zero_()
        model.stacks.output.bias.zero_()
    S.write_nnue(S.quantize(model), args.out, f"Jieqi material-only starting net ({F.ARCH})")
    torch.save({"model": model.state_dict(), "arch": F.ARCH}, args.out + ".pt")
    print(f"written {args.out} (arch {F.ARCH})")


if __name__ == "__main__":
    main()
