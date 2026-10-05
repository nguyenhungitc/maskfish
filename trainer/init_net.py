"""Writes a material-only starting network (no Jieqi network exists yet).

    python init_net.py gen0.nnue

The PSQT weights hold the piece values (a hidden piece counts as the average
value of the 15 pieces of a side) and the layer stacks output 0, so the
engine plays with a material evaluation. Use it to generate the first data,
train, then generate again with the trained network, and so on.
"""

import sys

import torch

import serialize as S
from model import NNUE


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    torch.manual_seed(0)
    model = NNUE(init_psqt=True)
    with torch.no_grad():
        model.stacks.l1.weight.zero_()
        model.stacks.l1.bias.zero_()
        model.stacks.l2.weight.zero_()
        model.stacks.l2.bias.zero_()
        model.stacks.output.weight.zero_()
        model.stacks.output.bias.zero_()
    S.write_nnue(S.quantize(model), sys.argv[1], "Jieqi material-only starting net")
    torch.save({"model": model.state_dict()}, sys.argv[1] + ".pt")
    print("written", sys.argv[1])


if __name__ == "__main__":
    main()
