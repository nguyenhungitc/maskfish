"""Trains the Jieqi NNUE.

    python train.py data/train --val data/val --epochs 400 --epoch-size 50000000 \
        --batch-size 16384 --out runs/exp1 [--resume runs/exp1/last.pt | --init-net old.nnue]

After every epoch: runs/exp1/epoch_N.pt, runs/exp1/last.pt and
runs/exp1/epoch_N.nnue (ready for the engine's EvalFile option).

Loss (as in nnue-pytorch): the network score s (internal units) and the data
score are both mapped to a win probability with sigmoid(x / scaling); the
target mixes the search score and the game result with --lambda:
    target = lambda * sigmoid(score / scaling) + (1 - lambda) * (result + 1) / 2
    loss   = mean(|sigmoid(s / scaling) - target| ^ 2.5)
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import torch

import jieqi_features as F
import serialize as S
from model import NNUE, NNUE2SCORE


class Data:
    def __init__(self, prefix: str):
        self.features = np.load(prefix + ".features.npy", mmap_mode="r")
        self.meta = np.load(prefix + ".meta.npy", mmap_mode="r")
        self.n = len(self.meta)

    def batches(self, batch_size: int, count: int, rng: np.random.Generator, device):
        """'count' batches; contiguous chunks from random offsets (the data
        is already shuffled), which keeps memory-mapped reads fast."""
        for _ in range(count):
            start = int(rng.integers(0, max(1, self.n - batch_size + 1)))
            yield self.get(start, min(self.n, start + batch_size), device)

    def get(self, a: int, b: int, device):
        f = torch.from_numpy(np.ascontiguousarray(self.features[a:b]).astype(np.int64))
        m = self.meta[a:b]
        return (f[:, 0].to(device, non_blocking=True), f[:, 1].to(device, non_blocking=True),
                torch.from_numpy(m["bucket"].astype(np.int64)).to(device),
                torch.from_numpy(m["score"].astype(np.float32)).to(device),
                torch.from_numpy(m["result"].astype(np.float32)).to(device))


def loss_fn(model, batch, scaling: float, lam: float):
    stm, nstm, bucket, score, result = batch
    out = model(stm, nstm, bucket).squeeze(1) * NNUE2SCORE
    q = torch.sigmoid(out / scaling)
    p = torch.sigmoid(score / scaling)
    target = lam * p + (1.0 - lam) * (result + 1.0) * 0.5
    return torch.pow(torch.abs(q - target), 2.5).mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("train", help="prefix written by prepare_data.py")
    ap.add_argument("--val", help="validation prefix")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--epoch-size", type=int, default=20_000_000, help="positions per epoch")
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=8.75e-4)
    ap.add_argument("--gamma", type=float, default=0.992, help="lr decay per epoch")
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0)
    ap.add_argument("--scaling", type=float, default=361.0)
    ap.add_argument("--resume", help="checkpoint to resume from (model + optimizer)")
    ap.add_argument("--init-net", help=".nnue file to start from")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    train = Data(args.train)
    val = Data(args.val) if args.val else None
    print(f"train: {train.n} positions" + (f", val: {val.n}" if val else ""), flush=True)

    model = NNUE()
    if args.init_net:
        S.load_into_model(S.read_nnue(args.init_net), model)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-7)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.gamma)
    first_epoch = 1
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
            sched.load_state_dict(ck["scheduler"])
        first_epoch = ck.get("epoch", 0) + 1

    steps = max(1, args.epoch_size // args.batch_size)
    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for i, batch in enumerate(train.batches(args.batch_size, steps, rng, device)):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model, batch, args.scaling, args.lam)
            loss.backward()
            opt.step()
            model.clip_weights()
            total += loss.item()
            if (i + 1) % 100 == 0:
                print(f"epoch {epoch} step {i + 1}/{steps} loss {total / (i + 1):.6f}", flush=True)
        sched.step()

        msg = f"epoch {epoch}: train loss {total / steps:.6f}"
        if val:
            model.eval()
            vl, vn = 0.0, 0
            with torch.no_grad():
                for a in range(0, min(val.n, 1_000_000), args.batch_size):
                    b = min(val.n, a + args.batch_size)
                    vl += loss_fn(model, val.get(a, b, device), args.scaling, args.lam).item() * (b - a)
                    vn += b - a
            msg += f", val loss {vl / max(1, vn):.6f}"
        msg += f", lr {sched.get_last_lr()[0]:.2e}, {time.time() - t0:.0f}s"
        print(msg, flush=True)

        ck = {"model": model.state_dict(), "optimizer": opt.state_dict(),
              "scheduler": sched.state_dict(), "epoch": epoch, "args": vars(args)}
        torch.save(ck, os.path.join(args.out, f"epoch_{epoch}.pt"))
        torch.save(ck, os.path.join(args.out, "last.pt"))
        model_cpu = NNUE(init_psqt=False)
        model_cpu.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
        S.write_nnue(S.quantize(model_cpu), os.path.join(args.out, f"epoch_{epoch}.nnue"),
                     f"Jieqi NNUE epoch {epoch}")


if __name__ == "__main__":
    main()
