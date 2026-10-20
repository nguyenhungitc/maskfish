"""Trains the Jieqi NNUE.

    python train.py data/train0 data/train1 --val data/val --epochs 400 \
        --epoch-size 50000000 --batch-size 16384 --out runs/exp1 \
        [--resume runs/exp1/last.pt | --init-net old.nnue]

Each data argument is a prefix written by prepare_data.py (all its shards are
used); several prefixes can be given, e.g. the data of several generations.
The data gives the input features (architecture); --layers 2 trains the
layer stacks of `make layers=2` on them (architecture "<arch>-l2"), and
--ctx the inventory context inputs of AB-JChess (`make ctx=yes`, "<arch>-ctx")
and --no-psqt a network without PSQT buckets (`make psqt=no`, "<arch>-nopsqt");
each can start from the feature transformer of a network without them
(--init-net).

Several GPUs: the same command started by torchrun (see Distributed):

    torchrun --standalone --nproc_per_node 4 train.py data/train0 --out runs/exp1 ...

After every epoch: runs/exp1/epoch_N.pt, runs/exp1/last.pt and
runs/exp1/epoch_N.nnue (ready for the engine's EvalFile option), each
written under a temporary name and renamed when complete; the metrics are
appended to runs/exp1/metrics.jsonl (and runs/exp1/tb with --tensorboard).
Batches are read ahead in a background thread (--prefetch). With
--checkpoint-every K, epoch_N.pt is kept only every K epochs (and for the last
one): checkpoints of the threat architectures take about 1.3 GB each.

Loss (as in nnue-pytorch): the network score s (internal units) and the data
score are both mapped to a win probability with sigmoid(x / scaling); the
target mixes the search score and the game result with --lambda:
    target = lambda * sigmoid(score / score_scaling) + (1 - lambda) * (result + 1) / 2
    loss   = mean(|sigmoid(s / scaling) - target| ^ power)
score_scaling is --score-scaling (default: --scaling), power --loss-power
(2.5). AB-JChess uses 410 for the data scores, 361 for the network and the
power 2 (with which mixing the targets is the same as mixing the two losses).
With --lambda-end, lambda goes linearly from --lambda (first epoch) to
--lambda-end (last epoch), as AB-JChess's lambda schedule (1.0 -> 0.8): the
game results count more and more.
With --scaling-table (fit_scaling.py --by-material), scaling depends on the
total material of the position (like the win rate model of Stockfish, which
depends on the material) instead of being one constant: the table gives it at
a few material values, linearly interpolated (score_scaling is then scaled by
the same factor). It needs the field allmat of the prepared data.

s is the network output after the scaling of the engine's evaluate() (the
data scores are search scores, made of evaluate() values; see eval_scale.py):
    s = evaluate(psqt, positional, material, rule40)
so that, once exported, the engine's evaluation of the network matches the
scores it was trained on, and the scale of the evaluation does not shrink
from one generation to the next. --no-eval-scale compares the raw network
output instead (the behaviour of older versions of the trainer).

Training-only techniques (ideas from nnue-pytorch / AB-JChess, none of them
changes the exported network format):
* virtual input features and a shared fc0 part (see model.py), on by
  default (--no-factorize, --no-l1-fact to disable; --factorize-kb adds the
  king bucket rows that AB-JChess also uses);
* the output layer learns at --output-lr-scale (0.1) times the base rate;
* --optimizer ranger21: the optimizer of AB-JChess and nnue-pytorch
  (ranger21.py), instead of Adam;
* --stack-balance CAP weights each sample by the inverse frequency of its
  layer stack (capped at CAP, weights normalized to an average of 1 per
  stack), so that rare stacks (endgames) are not drowned out. Off by default.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import queue
import threading
import time

import numpy as np
import torch
from torch import nn

import eval_scale as E
import jieqi_features as F
import serialize as S
from model import NNUE, NNUE2SCORE


def data_arch(prefix: str) -> str:
    try:
        with open(prefix + ".arch") as f:
            return F.feature_arch(f.read().strip())
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
        self.has_threats = F.ARCHS[self.arch].threats
        self.has_blend = F.ARCHS[self.arch].blend
        self.shards = []
        self.threats = []   # per shard: (flat indices, counts (N, 2), position starts) or None
        for prefix in prefixes:
            for sp in shard_prefixes(prefix):
                f = np.load(sp + ".features.npy", mmap_mode="r")
                m = np.load(sp + ".meta.npy", mmap_mode="r")
                thr = self._load_threats(sp, len(m))
                if len(m):
                    self.shards.append((f, m))
                    self.threats.append(thr)
        self.sizes = np.array([len(m) for _, m in self.shards], dtype=np.int64)
        # material / rule40 for the evaluate() scaling (prepare_data.py since that change)
        # se / allmat for EvalScaling Alignment (prepare_data.py since that option)
        self.has_align = all("se" in m.dtype.names and "allmat" in m.dtype.names
                             for _, m in self.shards)
        self.has_scale = all("material" in m.dtype.names and "rule40" in m.dtype.names
                             for _, m in self.shards)
        kinds = set()
        for prefix in prefixes:
            try:
                with open(prefix + ".material") as fm:
                    kinds.add(fm.read().strip())
            except FileNotFoundError:
                kinds.add("none")
        self.material_kinds = sorted(kinds)
        self.n = int(self.sizes.sum())
        if self.n == 0:
            raise SystemExit("empty data set")

    def _load_threats(self, sp: str, n: int):
        if not self.has_threats:
            return None
        try:
            flat = np.load(sp + ".threats.npy", mmap_mode="r")
            counts = np.load(sp + ".thrcount.npy")
        except FileNotFoundError:
            raise SystemExit(f"{sp}: no threat data (prepare the data again with --arch {self.arch})")
        if len(counts) != n:
            raise SystemExit(f"{sp}: threat data does not match the positions")
        starts = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(counts.astype(np.int64).sum(axis=1), out=starts[1:])
        if starts[-1] != len(flat):
            raise SystemExit(f"{sp}: threat data does not match the positions")
        return flat, counts, starts

    def batches(self, batch_size: int, count: int, rng: np.random.Generator, device,
                prefetch: int = 0):
        """'count' batches. Each one is a contiguous chunk at a random offset of
        a random shard (chosen in proportion to its size); shards are already
        shuffled, and contiguous reads keep memory-mapped access fast.
        prefetch > 0 reads that many batches ahead in a background thread (the
        same batches, in the same order, as without it)."""
        pin = torch.device(device).type == "cuda"

        def host():
            probs = self.sizes / self.n
            for _ in range(count):
                k = int(rng.choice(len(self.shards), p=probs))
                size = int(self.sizes[k])
                start = int(rng.integers(0, max(1, size - batch_size + 1)))
                yield self._host(k, start, min(size, start + batch_size), pin)

        for h in prefetched(host(), prefetch):
            yield to_device(h, device)

    def sequential(self, batch_size: int, limit: int, device, prefetch: int = 0):
        """Batches over the beginning of the data (for validation)."""
        pin = torch.device(device).type == "cuda"

        def host():
            done = 0
            for k, size in enumerate(self.sizes.tolist()):
                for a in range(0, size, batch_size):
                    if done >= limit:
                        return
                    b = min(size, a + batch_size, a + limit - done)
                    done += b - a
                    yield b - a, self._host(k, a, b, pin)

        for n, h in prefetched(host(), prefetch):
            yield n, to_device(h, device)

    def _host(self, k: int, a: int, b: int, pin: bool) -> dict:
        """Positions a..b of shard k as CPU tensors in their stored types
        (int16 features, ...): 4 to 8 times fewer bytes to copy to the GPU
        than int64 / float32, converted there by to_device(). With pin, the
        tensors are in page-locked memory (asynchronous copies to the GPU)."""
        feats, meta = self.shards[k]
        m = meta[a:b]
        h = {"feats": host_tensor(feats[a:b], pin),
             "bucket": host_tensor(m["bucket"], pin),
             "score": host_tensor(m["score"], pin),
             "result": host_tensor(m["result"], pin)}
        if self.threats[k] is not None:
            # EmbeddingBag input: bags stm, nstm of each position
            flat, counts, starts = self.threats[k]
            off = np.zeros(2 * (b - a), dtype=np.int64)
            np.cumsum(counts[a:b].reshape(-1)[:-1], out=off[1:])
            h["thr_idx"] = host_tensor(flat[starts[a]:starts[b]], pin, np.int32)
            h["thr_off"] = host_tensor(off, pin)
        if self.has_scale:
            h["material"] = host_tensor(m["material"], pin)
            h["rule40"] = host_tensor(m["rule40"], pin)
        if self.has_align:
            h["se"] = host_tensor(m["se"], pin)
            h["allmat"] = host_tensor(m["allmat"], pin)
        if self.has_blend:
            h["blend"] = host_tensor(m["blend"], pin)
        return h


def host_tensor(a: np.ndarray, pin: bool, dtype=None) -> torch.Tensor:
    """CPU tensor holding a copy of the array (a memory-mapped slice, or a
    strided field of the meta records), in page-locked memory with pin."""
    dtype = np.dtype(dtype or a.dtype)
    t = torch.empty(a.shape, dtype=torch.from_numpy(np.empty(0, dtype)).dtype, pin_memory=pin)
    t.numpy()[...] = a
    return t


def to_device(h: dict, device):
    """Batch tuple used by loss_fn / model_score:
    (stm, nstm, bucket, score, result, threats, material, rule40, blend, se, allmat),
    converted on the device to the types the model expects (int64 indices,
    float32 values), exactly the values of the stored data."""
    def dev(name):
        return h[name].to(device, non_blocking=True)

    f = dev("feats").long()
    threats = None
    if "thr_idx" in h:
        threats = (dev("thr_idx").long(), dev("thr_off"))
    material = dev("material").float() if "material" in h else None
    rule40 = dev("rule40").float() if "rule40" in h else None
    blend = dev("blend").long() if "blend" in h else None
    se = dev("se").float() if "se" in h else None
    allmat = dev("allmat").float() if "allmat" in h else None
    return (f[:, 0], f[:, 1], dev("bucket").long(), dev("score").float(), dev("result").float(),
            threats, material, rule40, blend, se, allmat)


_DONE = object()


def prefetched(it, depth: int):
    """Iterates over 'it' in a background thread, 'depth' items ahead, so that
    reading the memory-mapped data overlaps the training step (numpy releases
    the GIL while copying). depth 0: no thread. The items and their order are
    those of 'it'; an exception in the thread is raised here."""
    if depth <= 0:
        yield from it
        return
    q: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()

    def put(x) -> bool:
        while not stop.is_set():
            try:
                q.put(x, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def run():
        try:
            for x in it:
                if not put(x):
                    return
            put(_DONE)
        except BaseException as e:      # handed over to the consumer
            put(e)

    t = threading.Thread(target=run, name="prefetch", daemon=True)
    t.start()
    try:
        while True:
            x = q.get()
            if x is _DONE:
                return
            if isinstance(x, BaseException):
                raise x
            yield x
    finally:
        stop.set()
        t.join()


def scaling_mode(eval_scale) -> str | None:
    """None (raw network output), "complexity" (True: the engine's default
    EvalScaling) or "alignment" (EvalScaling Alignment)."""
    if not eval_scale:
        return None
    return "complexity" if eval_scale is True else str(eval_scale)


def model_score(model, batch, eval_scale=True, qat: bool = False) -> torch.Tensor:
    """Score of the model for a batch, internal units: after the engine's
    evaluate() scaling when the batch has its inputs and eval_scale is on
    (True / "complexity", or "alignment": the EvalScaling option), otherwise the
    raw network output. qat: the score of the quantized network, as the engine
    computes it (model.forward_parts_qat())."""
    stm, nstm, bucket, _, _, threats = batch[:6]
    material, rule40 = (batch[6], batch[7]) if len(batch) >= 8 else (None, None)
    blend = batch[8] if len(batch) >= 9 else None
    se, allmat = (batch[9], batch[10]) if len(batch) >= 11 else (None, None)
    mode = scaling_mode(eval_scale)
    if material is None or (mode == "alignment" and se is None):
        mode = None
    extra = {"scaling": mode, "se": se, "allmat": allmat} if mode else {}
    if qat:
        psqt, positional = model.forward_parts_qat(stm, nstm, bucket, threats, blend)
        psqt, positional = psqt.squeeze(1), positional.squeeze(1)
        if mode:
            return E.evaluate_qat(psqt, positional, material, rule40, **extra)
        return psqt + positional
    if mode:
        psqt, positional = model.forward_parts(stm, nstm, bucket, threats, blend)
        return E.evaluate_float(psqt.squeeze(1) * NNUE2SCORE, positional.squeeze(1) * NNUE2SCORE,
                                material, rule40, **extra)
    return model(stm, nstm, bucket, threats, blend).squeeze(1) * NNUE2SCORE


class Scorer(nn.Module):
    """model_score() as a module: what DistributedDataParallel wraps (it only
    synchronizes the gradients of what goes through its forward())."""

    def __init__(self, model: nn.Module, eval_scale=True):
        super().__init__()
        self.model = model
        self.eval_scale = eval_scale
        self.qat = False

    def forward(self, *batch) -> torch.Tensor:
        return model_score(self.model, batch, self.eval_scale, self.qat)


class ScalingTable:
    """Score -> win probability scale as a function of the total material
    (allmat) of the position, from fit_scaling.py --by-material: linear
    between the points of the table, constant beyond its ends."""

    def __init__(self, path: str):
        with open(path) as f:
            t = json.load(f)
        self.material = torch.tensor(t["material"], dtype=torch.float32)
        self.scale = torch.tensor(t["scale"], dtype=torch.float32)
        if len(self.material) < 2 or len(self.material) != len(self.scale) \
                or bool((self.material[1:] <= self.material[:-1]).any()) \
                or bool((self.scale <= 0).any()):
            raise SystemExit(f"{path}: not a scaling table of fit_scaling.py --by-material")

    def __call__(self, allmat: torch.Tensor) -> torch.Tensor:
        x = self.material.to(allmat.device)
        s = self.scale.to(allmat.device)
        i = torch.clamp(torch.bucketize(allmat, x), 1, len(x) - 1)
        t = torch.clamp((allmat - x[i - 1]) / (x[i] - x[i - 1]), 0.0, 1.0)
        return s[i - 1] + (s[i] - s[i - 1]) * t

    def describe(self) -> str:
        return ", ".join(f"{m:.0f}: {v:.0f}" for m, v in zip(self.material.tolist(),
                                                           self.scale.tolist()))


def loss_fn(model, batch, scaling: float, lam: float, stack_weights=None, eval_scale=True,
            score_scaling: float | None = None, power: float = 2.5, qat: bool = False,
            scaling_table: ScalingTable | None = None):
    """model: an NNUE, or a Scorer (possibly wrapped in DDP), which then
    computes the score itself (with its own eval_scale and qat).
    scaling_table: scaling (and score_scaling, by the same factor) from the
    total material of each position, batch[10]."""
    bucket, score, result = batch[2], batch[3], batch[4]
    if isinstance(model, NNUE):
        out = model_score(model, batch, eval_scale, qat)
    else:
        out = model(*batch)
    if scaling_table is not None:
        ratio = (score_scaling or scaling) / scaling
        scaling = scaling_table(batch[10])
        score_scaling = scaling * ratio
    q = torch.sigmoid(out / scaling)
    p = torch.sigmoid(score / (score_scaling if score_scaling is not None else scaling))
    target = lam * p + (1.0 - lam) * (result + 1.0) * 0.5
    loss = torch.pow(torch.abs(q - target), power)
    if stack_weights is None:
        return loss.mean()
    w = stack_weights[bucket]
    return (loss * w).sum() / w.sum()


def stack_balance_weights(data: "Data", cap: float) -> tuple[np.ndarray, np.ndarray]:
    """Capped inverse-frequency weight of each layer stack, normalized to sum
    to the number of stacks (AB-JChess's head balancing), and the number of
    positions of each layer stack."""
    counts = np.zeros(F.NUM_LAYER_STACKS, dtype=np.float64)
    for _, meta in data.shards:
        counts += np.bincount(np.asarray(meta["bucket"], dtype=np.int64),
                              minlength=F.NUM_LAYER_STACKS)[:F.NUM_LAYER_STACKS]
    total = counts.sum()
    raw = np.minimum(cap, total / np.maximum(counts, 1.0))
    return raw * F.NUM_LAYER_STACKS / raw.sum(), counts


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
    ap.add_argument("--lambda-end", type=float, default=None,
                    help="lambda of the last epoch: lambda goes linearly from --lambda to it "
                         "(AB-JChess: 1.0 -> 0.8); default: constant --lambda")
    ap.add_argument("--scaling", type=float, default=361.0)
    ap.add_argument("--score-scaling", type=float, default=None,
                    help="scale of the sigmoid of the data scores (default: --scaling; "
                         "AB-JChess: 410 with --scaling 361)")
    ap.add_argument("--scaling-table", metavar="FILE",
                    help="scaling as a function of the total material, from fit_scaling.py "
                         "--by-material (instead of the constant --scaling)")
    ap.add_argument("--loss-power", type=float, default=2.5,
                    help="exponent of the loss (AB-JChess: 2)")
    ap.add_argument("--resume", help="checkpoint to resume from (model + optimizer)")
    ap.add_argument("--init-net", help=".nnue file to start from")
    ap.add_argument("--layers", type=int, choices=(1, 2), default=1,
                    help="layer stacks: 1 (default) or 2 (make layers=2, architecture *-l2)")
    ap.add_argument("--ctx", action="store_true",
                    help="inventory context inputs of the layer stacks (make ctx=yes, *-ctx)")
    ap.add_argument("--quant2", action="store_true",
                    help="quantization of current Pikafish (make quant=2, architecture *-q2)")
    ap.add_argument("--no-psqt", action="store_true",
                    help="no PSQT buckets (make psqt=no, architecture *-nopsqt)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-factorize", action="store_true",
                    help="train without virtual input features")
    ap.add_argument("--factorize-kb", action="store_true",
                    help="also a virtual row per king bucket (AB-JChess); off by default")
    ap.add_argument("--no-l1-fact", action="store_true",
                    help="train without the fc0 part shared by the layer stacks")
    ap.add_argument("--optimizer", choices=("adam", "ranger21"), default="adam",
                    help="adam (default) or ranger21 (AB-JChess, nnue-pytorch: AdamW with "
                         "lookahead)")
    ap.add_argument("--output-lr-scale", type=float, default=0.1,
                    help="learning rate of the output layer relative to --lr")
    ap.add_argument("--checkpoint-every", type=int, default=1, metavar="K",
                    help="write epoch_N.pt every K epochs only (last.pt and the .nnue "
                         "files are always written)")
    ap.add_argument("--stack-balance", type=float, default=0.0, metavar="CAP",
                    help="weight samples by inverse layer stack frequency, capped at CAP "
                         "(e.g. 8); 0 = off")
    ap.add_argument("--eval-scaling", choices=E.SCALINGS, default="complexity",
                    help="the engine's EvalScaling option the network is trained for: "
                         "complexity (default) or alignment (needs data prepared with se / allmat)")
    ap.add_argument("--no-eval-scale", action="store_true",
                    help="compare the raw network output with the data scores, without the "
                         "engine's evaluate() scaling (old behaviour; the scale of the "
                         "evaluation then shrinks from one generation to the next)")
    ap.add_argument("--qat-start", type=int, default=0, metavar="EPOCH",
                    help="quantization-aware training from this epoch on (1: every epoch; "
                         "0: off): the loss is that of the exported integer network")
    ap.add_argument("--val-qat", action="store_true",
                    help="before --qat-start, also print the validation loss of the quantized "
                         "network (\"quantized net loss\"): what the engine will play with")
    ap.add_argument("--prefetch", type=int, default=4, metavar="N",
                    help="batches read ahead in a background thread (0: none)")
    ap.add_argument("--tensorboard", action="store_true",
                    help="also write TensorBoard logs to <out>/tb (needs the tensorboard package)")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    dist = Distributed(args.device)
    device = dist.device
    main_rank = dist.rank == 0
    torch.manual_seed(args.seed)
    # One random stream per process (several GPUs, see Distributed); a single
    # process draws exactly the batches of the trainer before DDP
    rng = np.random.default_rng(args.seed if dist.world == 1 else (args.seed, dist.rank))
    os.makedirs(args.out, exist_ok=True)
    if args.batch_size % dist.world:
        raise SystemExit(f"--batch-size {args.batch_size} is not a multiple of the "
                         f"{dist.world} processes")
    rank_batch = args.batch_size // dist.world   # --batch-size is the global batch
    say = print if main_rank else (lambda *a, **k: None)
    if dist.world > 1:
        say(f"distributed: {dist.world} processes ({dist.backend}), {rank_batch} positions "
            f"per process and step (global batch {args.batch_size})", flush=True)

    train = Data(args.train)
    val = Data(args.val) if args.val else None
    if val and val.arch != train.arch:
        raise SystemExit(f"train data is arch '{train.arch}' but validation data is '{val.arch}'")
    eval_scale = False if args.no_eval_scale else args.eval_scaling
    if eval_scale:
        for name, d in (("train", train), ("validation", val)):
            if d is not None and not d.has_scale:
                raise SystemExit(f"the {name} data has no material / rule40 (prepared by an older "
                                 "prepare_data.py): prepare it again from the text files, or "
                                 "train with --no-eval-scale")
            if d is not None and eval_scale == "alignment" and not d.has_align:
                raise SystemExit(f"the {name} data has no se / allmat (prepared by an older "
                                 "prepare_data.py): prepare it again from the text files to "
                                 "train with --eval-scaling alignment")
        kinds = sorted(set(train.material_kinds) | set(val.material_kinds if val else []))
        if len(kinds) > 1:
            say(f"warning: data prepared with different material settings {kinds}", flush=True)
    say("loss on: " + (f"evaluate() scaling of the network output (EvalScaling {eval_scale})"
                       if eval_scale else "raw network output"), flush=True)
    F.set_arch(F.model_arch(train.arch, layers=args.layers, ctx=args.ctx, psqt=not args.no_psqt,
                            quant2=args.quant2))
    init_q, init_ft_only = None, False
    if args.init_net:
        # A network that converts into the trained architecture (adding -meta,
        # -ctx or -blend, see serialize.convert()) is used whole; a network with
        # the same feature set gives its feature transformer only (with zero
        # rows for the meta features when it has none)
        target = F.ARCH
        init_arch = S.detect_arch(args.init_net)
        F.set_arch(init_arch)
        init_q = S.read_nnue(args.init_net)
        ib, ifl = F.split_arch(init_arch)
        tb, tfl = F.split_arch(target)
        if S.convertible(init_arch, target, requantize=True) is None:
            # (with another quantization, load_into_model() dequantizes it
            # with its own scales and the weights are clipped into this one's)
            init_q = S.convert(init_q, init_arch, target, requantize=True)
        elif ib == tb and not (ifl["meta"] and not tfl["meta"]):
            init_ft_only = True
            if tfl["meta"] and not ifl["meta"]:
                init_q = S.convert(init_q, init_arch, F.arch_name(ib, **{**ifl, "meta": True}))
        else:
            raise SystemExit(f"--init-net is arch '{init_arch}', training '{target}' "
                             f"(data '{train.arch}'): not the same feature set")
        F.set_arch(target)
        if init_ft_only:
            say(f"--init-net is arch '{init_arch}': only its feature transformer is used",
                flush=True)
        elif init_arch != target:
            say(f"--init-net is arch '{init_arch}': converted into '{target}'", flush=True)
    say(f"arch {F.ARCH}, train: {train.n} positions" + (f", val: {val.n}" if val else ""),
        flush=True)

    ck = None
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        if ck.get("arch", "default") != F.ARCH:
            raise SystemExit(f"checkpoint is arch '{ck.get('arch', 'default')}', training "
                             f"'{F.ARCH}' (data '{train.arch}')")
        # keep the factorization of the run being resumed
        model = NNUE.from_state_dict(ck["model"])
    else:
        model = NNUE(factorized=not args.no_factorize, l1_fact=not args.no_l1_fact,
                     kb_factor=args.factorize_kb)
        if init_q is not None:
            S.load_into_model(init_q, model, ft_only=init_ft_only)
            if init_q.get("quant2", False) != F.QUANT2:
                # another quantization: clip into this one's ranges now, and say
                # how much of the feature transformer that changes
                lim, L1 = model.s.max_ft_weight, F.L1
                w = model.ft.weight[:F.NUM_FEATURES, :L1]
                out = int((w.abs() > lim + 1e-6).sum())
                model.clip_weights()
                say(f"--init-net has the other quantization: requantized, {out} of {w.numel()} "
                    f"feature transformer weights ({100 * out / w.numel():.3f}%) clipped to "
                    f"+-{lim:.3f}", flush=True)
    say(f"factorized inputs: {model.factorized} (king bucket rows: {model.ft_kb is not None}), "
        f"shared fc0: {model.has_l1_fact}", flush=True)
    model.to(device)

    output_params = list(model.stacks.output.parameters())
    output_ids = {id(p) for p in output_params}
    groups = [{"params": [p for p in model.parameters() if id(p) not in output_ids], "lr": args.lr},
              {"params": output_params, "lr": args.lr * args.output_lr_scale}]
    steps = max(1, args.epoch_size // args.batch_size)
    if args.optimizer == "ranger21":
        # The settings of AB-JChess (and of nnue-pytorch): the AdamW core with
        # lookahead, without warmup / warmdown (the learning rate follows
        # --lr and --gamma), gradient centralization, clipping or decay
        from ranger21 import Ranger21
        opt = Ranger21(groups, lr=args.lr, betas=(0.9, 0.999), eps=1e-7,
                       num_batches_per_epoch=steps, num_epochs=args.epochs, use_warmup=False,
                       warmdown_active=False, using_gc=False, using_normgc=False,
                       use_adaptive_gradient_clipping=False, softplus=False,
                       pnm_momentum_factor=0.0, weight_decay=0.0, logging_active=False)
    else:
        opt = torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-7)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.gamma)
    first_epoch = 1
    if ck is not None:
        restored = False
        if ck.get("optimizer_name", "adam") != args.optimizer:
            say(f"warning: the checkpoint was trained with --optimizer "
                f"{ck.get('optimizer_name', 'adam')}: starting a new {args.optimizer} optimizer",
                flush=True)
        else:
            try:
                opt.load_state_dict(ck["optimizer"])
                sched.load_state_dict(ck["scheduler"])
                restored = True
            except (KeyError, ValueError) as e:
                say(f"warning: optimizer state not restored ({e}); starting a new one", flush=True)
        first_epoch = ck.get("epoch", 0) + 1
        if not restored:
            # A new optimizer keeps the learning rate where the run was
            for _ in range(first_epoch - 1):
                sched.step()
        del ck

    stack_weights = None
    if args.stack_balance > 0:
        w, counts = stack_balance_weights(train, args.stack_balance)
        stack_weights = torch.tensor(w, dtype=torch.float32, device=device)
        say("layer stack counts: " + " ".join(str(int(c)) for c in counts), flush=True)
        say("layer stack weights: " + " ".join(f"{x:.2f}" for x in w), flush=True)

    def epoch_lambda(epoch: int) -> float:
        if args.lambda_end is None or args.epochs <= 1:
            return args.lam
        t = min(1.0, max(0.0, (epoch - 1) / (args.epochs - 1)))
        return args.lam + (args.lambda_end - args.lam) * t

    for name, v in (("--lambda", args.lam), ("--lambda-end", args.lambda_end)):
        if v is not None and not 0.0 <= v <= 1.0:
            raise SystemExit(f"{name} must be in [0, 1]")
    loss_args = {"score_scaling": args.score_scaling, "power": args.loss_power}
    if args.scaling_table:
        if not train.has_align or (val and not val.has_align):
            raise SystemExit("--scaling-table needs the total material of the positions (field "
                             "allmat): prepare the data again with prepare_data.py")
        loss_args["scaling_table"] = ScalingTable(args.scaling_table)
        say(f"scaling by material (allmat: scale): {loss_args['scaling_table'].describe()}",
            flush=True)

    # Several processes: the gradients are averaged by DDP, which needs the
    # score to go through its forward(); every process applies the same
    # optimizer step and clipping, so the models stay identical
    scorer = Scorer(model, eval_scale)
    if dist.world > 1:
        scorer = dist.wrap(scorer)
    log = MetricsLog(args.out, args.tensorboard) if main_rank else None
    acc_dtype = torch.float32 if device.type == "mps" else torch.float64   # (no float64 on MPS)

    if args.qat_start:
        say(f"quantization-aware training from epoch {args.qat_start}", flush=True)
    for epoch in range(first_epoch, args.epochs + 1):
        lam = epoch_lambda(epoch)
        qat = 0 < args.qat_start <= epoch
        (scorer.module if dist.world > 1 else scorer).qat = qat
        model.train()
        t0 = time.time()
        # running sum on the device: no synchronization with the GPU at every step
        total = torch.zeros((), dtype=acc_dtype, device=device)
        batches = train.batches(rank_batch, steps, rng, device, prefetch=args.prefetch)
        for i, batch in enumerate(batches):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(scorer, batch, args.scaling, lam, stack_weights, **loss_args)
            loss.backward()
            opt.step()
            model.clip_weights()
            total += loss.detach().to(acc_dtype)
            if (i + 1) % 100 == 0:
                avg = dist.mean(total) / (i + 1)
                say(f"epoch {epoch} step {i + 1}/{steps} loss {avg:.6f}", flush=True)
                if log:
                    log.step(epoch, (epoch - 1) * steps + i + 1, avg,
                             (i + 1) * args.batch_size / (time.time() - t0))
        sched.step()
        train_loss = dist.mean(total) / steps
        train_time = time.time() - t0

        if main_rank:
            msg = f"epoch {epoch}: train loss {train_loss:.6f}"
            val_loss = qval_loss = None
            if val:
                model.eval()

                def val_pass(quantized: bool) -> float:
                    vl, vn = 0.0, 0
                    with torch.no_grad():
                        for n, batch in val.sequential(args.batch_size, 1_000_000, device,
                                                       prefetch=args.prefetch):
                            vl += loss_fn(model, batch, args.scaling, lam, eval_scale=eval_scale,
                                          qat=quantized, **loss_args).item() * n
                            vn += n
                    return vl / max(1, vn)

                val_loss = val_pass(qat)
                msg += f", val loss {val_loss:.6f}"
                if args.val_qat and not qat:
                    # (not "val loss": scripts/pipeline.sh reads the number after that)
                    qval_loss = val_pass(True)
                    msg += f", quantized net loss {qval_loss:.6f}"
            if args.lambda_end is not None:
                msg += f", lambda {lam:.3f}"
            if qat:
                msg += ", qat"
            lr = sched.get_last_lr()[0]
            speed = steps * args.batch_size / train_time
            msg += f", lr {lr:.2e}, {time.time() - t0:.0f}s, {speed / 1000:.1f}k pos/s"
            # (validation loss is unweighted: the real distribution of positions)
            print(msg, flush=True)
            log.epoch(epoch, epoch * steps, train_loss, val_loss, lr, lam, speed,
                      time.time() - t0, qat, val_loss if qat else qval_loss)

            ck = {"model": model.state_dict(), "optimizer": opt.state_dict(),
                  "optimizer_name": args.optimizer,
                  "scheduler": sched.state_dict(), "epoch": epoch, "args": vars(args),
                  "arch": F.ARCH}
            # Written under a temporary name and renamed: an interrupted run never
            # leaves a truncated last.pt, and a watcher (watch_nets.py) never
            # reads a half-written network
            if epoch % max(1, args.checkpoint_every) == 0 or epoch == args.epochs:
                atomic_write(os.path.join(args.out, f"epoch_{epoch}.pt"),
                             lambda path: torch.save(ck, path))
            atomic_write(os.path.join(args.out, "last.pt"), lambda path: torch.save(ck, path))
            atomic_write(os.path.join(args.out, f"epoch_{epoch}.nnue"),
                         lambda path: S.write_nnue(S.quantize(model), path,
                                                   f"Jieqi NNUE ({F.ARCH}) epoch {epoch}"))
            del ck
        dist.barrier()          # the other processes wait for the validation and the files

    if log:
        log.close()
    dist.close()


def atomic_write(path: str, write) -> None:
    """write(tmp) then rename tmp to path (atomic on the same file system)."""
    tmp = path + ".tmp"
    write(tmp)
    os.replace(tmp, path)


class MetricsLog:
    """Training metrics of the main process: <out>/metrics.jsonl (one JSON
    object per line, appended to across --resume), and TensorBoard scalars in
    <out>/tb with --tensorboard."""

    def __init__(self, out: str, tensorboard: bool):
        self.f = open(os.path.join(out, "metrics.jsonl"), "a", encoding="utf-8")
        self.tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as e:
                raise SystemExit(f"--tensorboard needs the tensorboard package "
                                 f"(pip install tensorboard): {e}")
            self.tb = SummaryWriter(os.path.join(out, "tb"))

    def _write(self, rec: dict):
        rec["time"] = datetime.datetime.now().isoformat(timespec="seconds")
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()

    def step(self, epoch: int, gstep: int, loss: float, speed: float):
        self._write({"type": "step", "epoch": epoch, "step": gstep, "loss": loss,
                     "pos_per_s": round(speed)})
        if self.tb:
            self.tb.add_scalar("train/loss_running", loss, gstep)
            self.tb.add_scalar("train/positions_per_s", speed, gstep)

    def epoch(self, epoch: int, gstep: int, train_loss: float, val_loss, lr: float, lam: float,
              speed: float, seconds: float, qat: bool = False, quantized_val_loss=None):
        self._write({"type": "epoch", "epoch": epoch, "step": gstep, "train_loss": train_loss,
                     "val_loss": val_loss, "quantized_val_loss": quantized_val_loss,
                     "lr": lr, "lambda": lam, "qat": qat,
                     "pos_per_s": round(speed), "seconds": round(seconds, 1)})
        if self.tb:
            self.tb.add_scalar("epoch/train_loss", train_loss, epoch)
            if val_loss is not None:
                self.tb.add_scalar("epoch/val_loss", val_loss, epoch)
            if quantized_val_loss is not None:
                self.tb.add_scalar("epoch/quantized_val_loss", quantized_val_loss, epoch)
            self.tb.add_scalar("epoch/lr", lr, epoch)
            self.tb.add_scalar("epoch/lambda", lam, epoch)
            self.tb.add_scalar("epoch/positions_per_s", speed, epoch)
            self.tb.flush()

    def close(self):
        self.f.close()
        if self.tb:
            self.tb.close()


class Distributed:
    """Training on several GPUs (or CPU processes) with DistributedDataParallel
    when the script is started by torchrun:

        torchrun --standalone --nproc_per_node 4 train.py data/train --out runs/x ...

    Each process draws its own random batches of --batch-size / processes
    positions; DDP averages the gradients, so a step is the same as one step
    on a single device with --batch-size (same hyperparameters, same number of
    steps per epoch). Only process 0 prints, validates and writes files.
    Without torchrun (WORLD_SIZE unset or 1) nothing changes."""

    def __init__(self, device: str):
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.backend = None
        if self.world <= 1:
            self.world, self.rank = 1, 0
            self.device = torch.device(device)
            return
        import torch.distributed as td
        self.td = td
        local = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.device(device).type == "cuda":
            self.device = torch.device("cuda", local)
            torch.cuda.set_device(self.device)
            self.backend = "nccl"
        else:
            self.device = torch.device(device)
            self.backend = "gloo"
        # long timeout: process 0 validates and writes checkpoints (1.3 GB with
        # threats) while the others wait in barrier()
        self.td.init_process_group(self.backend, timeout=datetime.timedelta(hours=2))

    def wrap(self, module: nn.Module) -> nn.Module:
        from torch.nn.parallel import DistributedDataParallel
        ids = [self.device.index] if self.device.type == "cuda" else None
        return DistributedDataParallel(module, device_ids=ids, gradient_as_bucket_view=True)

    def mean(self, t: torch.Tensor) -> float:
        """Mean over the processes of a scalar tensor (its value with one)."""
        if self.world > 1:
            t = t.clone()
            self.td.all_reduce(t)
            t /= self.world
        return t.item()

    def barrier(self):
        if self.world > 1:
            self.td.barrier()

    def close(self):
        if self.world > 1:
            self.td.destroy_process_group()


if __name__ == "__main__":
    main()
