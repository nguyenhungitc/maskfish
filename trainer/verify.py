"""Checks that a network exported by serialize.py is read by the engine exactly
as the trainer computes it.

    python verify.py --engine ../src/pikafish [--net out.nnue] [--positions 200]

Without --net, a randomly initialized model is exported to a temporary file.
For each position (random Jieqi games played with real hidden identities),
it compares the engine's `evalraw` output with:
  * the bit-exact integer reference (serialize.quantized_eval): must be equal;
  * the float PyTorch model: must be close (quantization error only).
"""

from __future__ import annotations

import argparse
import os
import random
import tempfile

import jieqi_features as F
import jieqi_uci as U
import serialize as S


def random_positions(engine: U.Engine, count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    fens = []
    while len(fens) < count:
        ids = U.HiddenIdentities(rng)
        moves = []
        for _ in range(rng.randint(0, 120)):
            engine.set_position(moves)
            fen, in_check = engine.board_info()
            if not in_check:
                fens.append(fen)
            legal = engine.legal_moves()
            if not legal:
                break
            moves.append(ids.annotate(rng.choice(legal)))
            if len(fens) >= count:
                break
    return fens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--net", help="network to check (default: a random model)")
    ap.add_argument("--checkpoint", help="checkpoint to compare the float model with")
    ap.add_argument("--positions", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--arch", default="default", choices=sorted(F.ARCHS),
                    help="architecture of the random test network (ignored with --net)")
    args = ap.parse_args()
    F.set_arch(S.detect_arch(args.net) if args.net else args.arch)
    print(f"arch {F.ARCH}")

    import torch
    from model import NNUE, NNUE2SCORE

    model = None
    tmp = None
    if args.net:
        q = S.read_nnue(args.net)
        net = args.net
        if args.checkpoint:
            ck = torch.load(args.checkpoint, map_location="cpu")
            model = NNUE.from_state_dict(ck["model"] if "model" in ck else ck)
    else:
        torch.manual_seed(args.seed)
        # A factorized model, so that the coalescing done at export is tested too
        model = NNUE(factorized=True, l1_fact=True, kb_factor=True)
        with torch.no_grad():   # make every layer matter in the test
            for p in model.stacks.parameters():
                p.uniform_(-0.5, 0.5)
            model.stacks.l1_fact.weight.uniform_(-0.2, 0.2)
            model.ft.weight[:, :2048].uniform_(-0.1, 0.1)
            model.ft.weight[:, 2048:].uniform_(-0.2, 0.2)   # also test every PSQT row
            model.ft.weight[F.PAD_INDEX].zero_()
            model.ft_ps.weight.uniform_(-0.03, 0.03)
            model.ft_kb.weight.uniform_(-0.03, 0.03)
        model.clip_weights()
        q = S.quantize(model)
        tmp = tempfile.NamedTemporaryFile(suffix=".nnue", delete=False)
        tmp.close()
        S.write_nnue(q, tmp.name, "verify")
        net = tmp.name

    engine = U.Engine(args.engine, {"EvalFile": os.path.abspath(net)})
    fens = random_positions(engine, args.positions, args.seed)

    exact, worst = 0, 0.0
    for fen in fens:
        engine.send(f"position fen {fen}")
        e_psqt, e_pos, e_bucket = engine.evalraw()
        r_psqt, r_pos, r_bucket = S.quantized_eval(q, fen)
        if (e_psqt, e_pos, e_bucket) == (r_psqt, r_pos, r_bucket):
            exact += 1
        else:
            print("MISMATCH", fen, "engine", (e_psqt, e_pos, e_bucket),
                  "reference", (r_psqt, r_pos, r_bucket))
        if model is not None:
            s, n, b, _ = F.fen_to_features(fen)
            pad = lambda l: l + [F.PAD_INDEX] * (F.MAX_ACTIVE - len(l))
            with torch.no_grad():
                v = model(torch.tensor([pad(s)]), torch.tensor([pad(n)]),
                          torch.tensor([b])).item() * NNUE2SCORE
            worst = max(worst, abs(v - (e_psqt + e_pos)))
    engine.quit()
    if tmp:
        os.unlink(tmp.name)

    print(f"{exact}/{len(fens)} positions bit-exact with the engine")
    if model is not None:
        print(f"largest float/quantized difference: {worst:.1f} internal units")
    raise SystemExit(0 if exact == len(fens) else 1)


if __name__ == "__main__":
    main()
