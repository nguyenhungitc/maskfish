"""Checks that a network exported by serialize.py is read by the engine exactly
as the trainer computes it.

    python verify.py --engine ../src/maskfish [--net out.nnue] [--positions 200]

Without --net, a randomly initialized model is exported to a temporary file.
For each position (random Jieqi games played with real hidden identities, seen
from the side to move as gen_data.py writes them: the identity of a captured
dark piece is known to its capturer only), it compares the engine's `evalraw`
output with:
  * the bit-exact integer reference (serialize.quantized_eval): must be equal;
  * the evaluate() scaling of that output (eval_scale.evaluate_int, the
    function train.py applies in the loss): must be equal to the engine's;
  * the float PyTorch model: must be close (quantization error only).
"""

from __future__ import annotations

import argparse
import os
import random
import tempfile

import eval_scale as E
import jieqi_features as F
import jieqi_uci as U
import serialize as S


def random_positions(engine: U.Engine, count: int, seed: int) -> list[str]:
    """Positions of random games, each from the view of its side to move (pools
    then also hold captured dark pieces of unknown identity, as in the data).
    Captures are preferred half of the time to reach every layer stack."""
    rng = random.Random(seed)
    fens = []
    while len(fens) < count:
        ids = U.HiddenIdentities(rng)
        views: tuple[list[str], list[str]] = ([], [])
        full_info = rng.random() < 0.3
        for ply in range(rng.randint(0, 160)):
            engine.set_position(views[ply % 2])
            fen, in_check = engine.board_info()
            if not in_check:
                fens.append(fen)
            legal = engine.legal_moves()
            if not legal or len(fens) >= count:
                break
            captures = [m for m in legal if U.is_capture(fen, m)]
            move = rng.choice(captures if captures and rng.random() < 0.5 else legal)
            white_view, black_view = U.annotate_views(ids, move, full_info)
            views[0].append(white_view)
            views[1].append(black_view)
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
    ap.add_argument("--jieqi-material", action="store_true",
                    help="check the evaluate() scaling with the engine option JieqiMaterial=true")
    args = ap.parse_args()
    F.set_arch(S.detect_arch(args.net) if args.net else args.arch)
    print(f"arch {F.ARCH}")

    import torch
    from model import NNUE, NNUE2SCORE, threat_inputs

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
            L1 = F.L1
            model.ft.weight[:, :L1].uniform_(-0.1, 0.1)
            model.ft.weight[:, L1:].uniform_(-0.2, 0.2)   # also test every PSQT row
            if model.ft_thr is not None:
                model.ft_thr.weight[:, :L1].uniform_(-0.1, 0.1)
                model.ft_thr.weight[:, L1:].uniform_(-0.2, 0.2)
            model.ft.weight[F.PAD_INDEX].zero_()
            model.ft_ps.weight.uniform_(-0.03, 0.03)
            model.ft_kb.weight.uniform_(-0.03, 0.03)
        model.clip_weights()
        q = S.quantize(model)
        tmp = tempfile.NamedTemporaryFile(suffix=".nnue", delete=False)
        tmp.close()
        S.write_nnue(q, tmp.name, "verify")
        net = tmp.name

    engine = U.Engine(args.engine, {"EvalFile": os.path.abspath(net),
                                    "JieqiMaterial": str(args.jieqi_material).lower()})
    fens = random_positions(engine, args.positions, args.seed)

    exact, worst, scaled, no_final = 0, 0.0, 0, 0
    for fen in fens:
        engine.send(f"position fen {fen}")
        e_psqt, e_pos, e_bucket, e_final = engine.evalraw_full()
        r_psqt, r_pos, r_bucket = S.quantized_eval(q, fen)
        ok = (e_psqt, e_pos, e_bucket) == (r_psqt, r_pos, r_bucket)
        if not ok:
            print("MISMATCH", fen, "engine", (e_psqt, e_pos, e_bucket),
                  "reference", (r_psqt, r_pos, r_bucket))
        if e_final is None:
            no_final += 1
        else:
            material, rule40 = E.position_scale_inputs(F.parse_fen(fen), args.jieqi_material)
            r_final = E.evaluate_int(e_psqt, e_pos, material, rule40)
            if r_final == e_final:
                scaled += 1
            else:
                ok = False
                print("SCALE MISMATCH", fen, "engine evaluate()", e_final, "reference", r_final,
                      f"(material {material}, rule40 {rule40})")
        exact += ok
        if model is not None:
            s, n, b, _ = F.fen_to_features(fen)
            pad = lambda l: l + [F.PAD_INDEX] * (F.MAX_ACTIVE - len(l))
            thr = threat_inputs(list(F.fen_to_threats(fen))) if F.THREATS else None
            with torch.no_grad():
                v = model(torch.tensor([pad(s)]), torch.tensor([pad(n)]),
                          torch.tensor([b]), thr).item() * NNUE2SCORE
            worst = max(worst, abs(v - (e_psqt + e_pos)))
    engine.quit()
    if tmp:
        os.unlink(tmp.name)

    print(f"{exact}/{len(fens)} positions bit-exact with the engine")
    if no_final == len(fens):
        print("warning: the engine does not print the evaluate() value in evalraw "
              "(older engine): the scaling used in the loss was not checked")
    else:
        print(f"evaluate() scaling: {scaled}/{len(fens) - no_final} positions bit-exact")
    if model is not None:
        print(f"largest float/quantized difference: {worst:.1f} internal units")
    raise SystemExit(0 if exact == len(fens) else 1)


if __name__ == "__main__":
    main()
