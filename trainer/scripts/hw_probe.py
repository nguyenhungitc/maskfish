"""Hardware probe for pipeline.sh: CPU, RAM, disk, GPUs, and the largest
training batch that fits on the GPU. Prints KEY=VALUE lines prefixed with
"HW_" (parsed by pipeline.sh) plus a readable summary.

    python scripts/hw_probe.py [--arch default] [--batch 16384] [--probe-batch] [--path .]

Works without torch (then no GPU information). With --probe-batch and CUDA,
it really runs a few training steps (model + Adam, synthetic batch) on the
GPU with the most free memory, halving the batch size until it fits.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys


def read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def cpu_count() -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    # cgroup CPU quota (containers, Slurm, Docker --cpus ...)
    q = read("/sys/fs/cgroup/cpu.max")                       # v2: "quota period"
    if q and not q.startswith("max"):
        quota, period = (int(x) for x in q.split()[:2])
        n = min(n, max(1, quota // period))
    else:
        quota, period = read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if quota and period and int(quota) > 0:
            n = min(n, max(1, int(quota) // int(period)))
    return max(1, n)


def memory_mb() -> tuple[int, int]:
    """(total, available) in MB, cgroup limits included."""
    total = avail = 0
    info = read("/proc/meminfo")
    if info:
        kv = {l.split(":")[0]: int(l.split()[1]) for l in info.splitlines() if len(l.split()) >= 2}
        total = kv.get("MemTotal", 0) // 1024
        avail = kv.get("MemAvailable", kv.get("MemFree", 0)) // 1024
    else:
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2**20
        except (ValueError, OSError, AttributeError):
            total = 4096
        avail = int(total * 0.6)                               # macOS: no cheap "available"
    lim = read("/sys/fs/cgroup/memory.max") or read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if lim and lim.isdigit() and int(lim) < 2**60:
        lim_mb = int(lim) // 2**20
        used = read("/sys/fs/cgroup/memory.current") or read("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        used_mb = int(used) // 2**20 if used and used.isdigit() else 0
        total = min(total, lim_mb)
        avail = min(avail, max(0, lim_mb - used_mb))
    return total, avail


def probe_batch(torch, arch: str, target: int, headroom: float) -> tuple[int, int]:
    """Largest batch size (target, target/2, ... >= 1024) whose training step
    fits in the free memory of the current GPU. Returns (batch, peak MB)."""
    import numpy as np
    import jieqi_features as F
    F.set_arch(arch)
    from model import NNUE
    from train import loss_fn

    dev = torch.device("cuda")
    model = NNUE(factorized=True, l1_fact=True).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-7)
    rng = np.random.default_rng(0)
    bs = target
    loss = batch = feats = thr = None
    while bs >= 1024:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            free0, _ = torch.cuda.mem_get_info()
            reserved0 = torch.cuda.memory_reserved()   # model (+ Adam state) already held by torch
            for _ in range(3):                       # Adam state is allocated at the 1st step
                feats = np.full((2, bs, F.MAX_ACTIVE), F.PAD_INDEX, dtype=np.int64)
                feats[:, :, :32] = rng.integers(0, F.NUM_FEATURES, (2, bs, 32))
                thr = None
                if F.THREATS:
                    idx = torch.from_numpy(rng.integers(0, F.THREAT_DIMS, 2 * bs * 40)).to(dev)
                    off = torch.arange(0, 2 * bs * 40, 40, device=dev)
                    thr = (idx, off)
                batch = (torch.from_numpy(feats[0]).to(dev), torch.from_numpy(feats[1]).to(dev),
                         torch.from_numpy(rng.integers(0, F.NUM_LAYER_STACKS, bs)).to(dev),
                         torch.from_numpy(rng.normal(0, 400, bs).astype(np.float32)).to(dev),
                         torch.from_numpy(rng.integers(-1, 2, bs).astype(np.float32)).to(dev), thr)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model, batch, 361.0, 1.0)
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_reserved()
            if peak <= headroom * (free0 + reserved0):
                return bs, peak // 2**20
        except RuntimeError as e:                    # torch.cuda.OutOfMemoryError is a RuntimeError
            if "out of memory" not in str(e).lower():
                raise
        loss = batch = feats = thr = None            # free the failed attempt before retrying
        opt.zero_grad(set_to_none=True)
        bs //= 2
    return 0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="default")
    ap.add_argument("--batch", type=int, default=16384, help="target (maximum) batch size")
    ap.add_argument("--probe-batch", action="store_true")
    ap.add_argument("--path", default=".")
    ap.add_argument("--headroom", type=float, default=0.85)
    args = ap.parse_args()

    out = {}
    out["CPUS"] = cpu_count()
    out["RAM_TOTAL_MB"], out["RAM_AVAIL_MB"] = memory_mb()
    out["DISK_FREE_MB"] = shutil.disk_usage(args.path).free // 2**20

    try:
        import torch
        out["TORCH"] = torch.__version__
        cuda = torch.cuda.is_available()
    except Exception:                                # noqa: BLE001  (torch missing or broken)
        torch, cuda = None, False
    out["CUDA"] = int(cuda)
    if cuda:
        gpus = []
        for i in range(torch.cuda.device_count()):
            try:
                free, total = torch.cuda.mem_get_info(i)
            except RuntimeError:
                continue
            gpus.append((free, total, i, torch.cuda.get_device_name(i)))
        out["GPU_COUNT"] = len(gpus)
        if gpus:
            free, total, idx, name = max(gpus)       # the GPU with the most free memory
            torch.cuda.set_device(idx)
            out["GPU_INDEX"] = idx
            out["GPU_NAME"] = name.replace(" ", "_")
            out["GPU_TOTAL_MB"] = total // 2**20
            out["GPU_FREE_MB"] = free // 2**20
            if args.probe_batch:
                # the trainer modules (jieqi_features, model, train) are in the parent directory
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                try:
                    bs, peak = probe_batch(torch, args.arch, args.batch, args.headroom)
                    out["BATCH_FIT"] = bs
                    out["BATCH_PEAK_MB"] = peak
                except Exception as e:               # noqa: BLE001  (report, keep the other values)
                    print(f"batch probe failed: {e!r}", file=sys.stderr)

    for k, v in out.items():
        print(f"HW_{k}={v}")


if __name__ == "__main__":
    main()
