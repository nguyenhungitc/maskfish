#!/usr/bin/env python3
"""Smoke test of the trainer (train/train.py) on the CPU: the path that makes the next network.

  python3 tests/train_smoke.py BUILD/maskfish [--jqnz BUILD/jqnz_tool] [--require]

1. The test network (tests/testnet.py) is read as float weights (train.py --init NET) and exported again: the file
   must come back byte for byte, also from its compressed form (with --jqnz), so that training continues exactly
   from a published net.
2. The engine plays a few self-play games with it (selfplay, 112-byte TrainRecords).
3. train.py trains one short epoch from the net on those records, on the CPU.
4. The engine loads the trained net and runs its bench.

Needs numpy, and PyTorch for steps 3 and 4 and for the trainer's side of step 1 (the CPU build is enough). Without
PyTorch it checks the net file reader (train/hnn_file.py) instead in step 1, says so and passes, unless --require.
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        sys.exit(f"train_smoke: {' '.join(map(str, cmd))} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout


def engine(binary, *cmds):
    return run([binary], input="\n".join(cmds + ("quit",)) + "\n")


def main():
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        sys.exit(__doc__)
    binary = str(Path(args[0]).resolve())
    jqnz = str(Path(args[args.index("--jqnz") + 1]).resolve()) if "--jqnz" in args else ""
    torch = importlib.util.find_spec("torch") is not None
    if not torch and "--require" in args:
        sys.exit("train_smoke: PyTorch is not installed")
    if importlib.util.find_spec("numpy") is None:
        print("train_smoke: skipped (no numpy)")
        return
    sys.path.insert(0, str(ROOT / "train"))
    import hnn_file as HF  # noqa: E402  (train/hnn_file.py)
    TR = None
    if torch:
        import train as TR  # noqa: E402  (train/train.py)

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        net = d / "net.hnn"
        run([sys.executable, str(ROOT / "tests" / "testnet.py"), str(net)])
        orig = net.read_bytes()
        nets = [net]
        if jqnz:
            run([jqnz, "z", str(net), str(d / "net.nnue")])
            nets.append(d / "net.nnue")
            os.environ["HNN_JQNZ_TOOL"] = jqnz
        for n in nets:
            if TR:
                TR.HNN.from_net(str(n)).export(d / "back.hnn")
            else:
                raw = n
                if n.suffix == ".nnue":
                    run([jqnz, "d", str(n), str(d / "dec.hnn")])
                    raw = d / "dec.hnn"
                HF.write(d / "back.hnn", HF.read(raw))
            if (d / "back.hnn").read_bytes() != orig:
                sys.exit(f"train_smoke: FAIL, {n.name} read and exported again is not the same net")
        print(f"train_smoke: {', '.join(n.name for n in nets)} read and exported again "
              f"({'train.py' if TR else 'hnn_file.py'}): the same bytes")

        for name, games, seed in (("data.bin", 8, 1), ("val.bin", 3, 2)):
            out = engine(binary, f"selfplay evalfile {net} games {games} nodes 200 threads 2 seed {seed} out {d / name}")
            if not (d / name).exists() or "selfplay done" not in out:
                sys.exit(f"train_smoke: FAIL, no self-play records:\n{out}")
        print(f"train_smoke: {(d / 'data.bin').stat().st_size // 112} training records")
        if not TR:
            print("train_smoke: pass without training (no PyTorch: train.py not run)")
            return

        log = run([sys.executable, str(ROOT / "train" / "train.py"), str(d / "data.bin"), "--val-file", str(d / "val.bin"),
                   "--out", str(d / "run"), "--init", str(net), "--device", "cpu", "--threads", "2", "--epochs", "1",
                   "--batch", "64", "--wt", "0", "--ws", "0.7", "--wr", "0.3"], cwd=str(d))
        if '{"epoch": 1' not in log or not (d / "run" / "best.hnn").exists():
            sys.exit(f"train_smoke: FAIL, the trainer wrote no epoch / best.hnn:\n{log[-2000:]}")
        out = engine(binary, f"setoption name EvalFile value {d / 'run' / 'best.hnn'}", "bench")
        if "EvalFile loaded" not in out or "Nodes searched" not in out:
            sys.exit(f"train_smoke: FAIL, the engine cannot use the trained net:\n{out[-2000:]}")
    print("train_smoke: pass (trained from the net on the CPU, the engine loads the result)")


if __name__ == "__main__":
    main()
