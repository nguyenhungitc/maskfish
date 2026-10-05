"""Trains the Jieqi NNUE.

    python train.py data/train0 data/train1 --val data/val --epochs 400 \
        --epoch-size 50000000 --batch-size 16384 --out runs/exp1 \
        [--resume runs/exp1/last.pt | --init-net old.nnue]

Each data argument is a prefix written by prepare_data.py (all its shards are
used); several prefixes can be given, e.g. the data of several generations.

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
import glob
import math
import os
import time

import numpy as np
import torch

import jieqi_features as F
import serialize as S
from model import NNUE, NNUE2SCORE


def data_arch(prefix: str) -> str:
    try:
        with open(prefix + ".arch") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "default"


def shard_prefixes(prefix: str) -> list[str]:
    """Shards of a prefix: prefix-NNNNN.*, or prefix.* (older single-file format)."""
    if os.path.exists(prefix + ".features.npy"):
        return [prefix]
    shards = sorted(p[:-len(".features.npy")] for p in glob.glob(glob.escape(prefix) + "-*.features.npy"))
    if not shards:
        raise SystemExit(f"no data found for prefix {prefix!r}")
    return shards


class Data:
    """All the shards of one or more prefixes, memory-mapped."""

    def __init__(self, prefixes: list[str]):
        archs = {data_arch(p) for p in prefixes}
        if len(archs) != 1:
            raise SystemExit(f"data prefixes have different architectures: {sorted(archs)}")
        self.arch = archs.pop()
        self.shards = []
        for prefix in prefixes:
            for sp in shard_prefixes(prefix):
                f = np.load(sp + ".features.npy", mmap_mode="r")
                m = np.load(sp + ".meta.npy", mmap_mode="r")
                if len(m):
                    self.shards.append((f, m))
        self.sizes = np.array([len(m) for _, m in self.shards], dtype=np.int64)
        self.n = int(self.sizes.sum())
        if self.n == 0:
            raise SystemExit("empty data set")

    def batches(self, batch_size: int, count: int, rng: np.random.Generator, device):
        """'count' batches. Each one is a contiguous chunk at a random offset of
        a random shard (chosen in proportion to its size); shards are already
        shuffled, and contiguous reads keep memory-mapped access fast."""
        probs = self.sizes / self.n
        for _ in range(count):
            k = int(rng.choice(len(self.shards), p=probs))
            size = int(self.sizes[k])
            start = int(rng.integers(0, max(1, size - batch_size + 1)))
            yield self._get(k, start, min(size, start + batch_size), device)

    def sequential(self, batch_size: int, limit: int, device):
        """Batches over the beginning of the data (for validation)."""
        done = 0
        for k, size in enumerate(self.sizes.tolist()):
            for a in range(0, size, batch_size):
                if done >= limit:
                    return
                b = min(size, a + batch_size, a + limit - done)
                done += b - a
                yield b - a, self._get(k, a, b, device)

    def _get(self, k: int, a: int, b: int, device):
        feats, meta = self.shards[k]
        f = torch.from_numpy(np.ascontiguousarray(feats[a:b]).astype(np.int64))
        m = meta[a:b]
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
    ap.add_argument("train", nargs="+", help="prefix(es) written by prepare_data.py")
    ap.add_argument("--val", nargs="+", help="validation prefix(es)")
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
    if val and val.arch != train.arch:
        raise SystemExit(f"train data is arch '{train.arch}' but validation data is '{val.arch}'")
    F.set_arch(train.arch)
    if args.init_net and S.detect_arch(args.init_net) != F.ARCH:
        raise SystemExit(f"--init-net is arch '{S.detect_arch(args.init_net)}', data is '{F.ARCH}'")
    print(f"arch {F.ARCH}, train: {train.n} positions" + (f", val: {val.n}" if val else ""),
          flush=True)

    model = NNUE()
    if args.init_net:
        S.load_into_model(S.read_nnue(args.init_net), model)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-7)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.gamma)
    first_epoch = 1
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        if ck.get("arch", "default") != F.ARCH:
            raise SystemExit(f"checkpoint is arch '{ck.get('arch', 'default')}', data is '{F.ARCH}'")
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
                for n, batch in val.sequential(args.batch_size, 1_000_000, device):
                    vl += loss_fn(model, batch, args.scaling, args.lam).item() * n
                    vn += n
            msg += f", val loss {vl / max(1, vn):.6f}"
        msg += f", lr {sched.get_last_lr()[0]:.2e}, {time.time() - t0:.0f}s"
        print(msg, flush=True)

        ck = {"model": model.state_dict(), "optimizer": opt.state_dict(),
              "scheduler": sched.state_dict(), "epoch": epoch, "args": vars(args),
              "arch": F.ARCH}
        torch.save(ck, os.path.join(args.out, f"epoch_{epoch}.pt"))
        torch.save(ck, os.path.join(args.out, "last.pt"))
        S.write_nnue(S.quantize(model), os.path.join(args.out, f"epoch_{epoch}.nnue"),
                     f"Jieqi NNUE ({F.ARCH}) epoch {epoch}")


if __name__ == "__main__":
    main()
