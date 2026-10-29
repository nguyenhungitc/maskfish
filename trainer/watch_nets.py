"""Measures every network a training run writes, while it trains, and keeps
the best one (the idea of nnue-pytorch's run_games.py, with this trainer's
match.py instead of c-chess-cli + ordo).

    python watch_nets.py runs/gen2 --engine ../src/maskfish --ref nets/gen1.nnue \\
        --pairs 200 --nodes 5000 --workers 6 --follow [--every 5] [--tensorboard]

Every epoch_N.nnue of the run directory (one or several) plays --pairs game
pairs against the reference network --ref, with match.py (same engine, same
options, paired games with colors swapped). All networks use the same
openings and hidden identities (--seed), so their results are comparable:
the ranking is the Elo of each network against the reference.

* Results are saved in <run>/watch.json after every network, and the script
  can be stopped and restarted at any time (measured networks are skipped;
  a larger --pairs tops them up).
* The best network (highest Elo with at least --min-pairs pairs) is copied to
  <run>/best.nnue whenever it changes.
* --follow keeps watching for new networks (every --poll seconds) until the
  network of --until-epoch has been measured, or until no new network
  appeared for --idle minutes; without it, the script measures what exists
  and exits.
* --tensorboard adds the Elo of every network (match/elo_vs_ref, at its
  epoch) to the TensorBoard logs of the run (<run>/tb, see train.py).

The engine must be built for the architecture of the networks (make
nnue=att / threats=yes / layers=2 ...). Training uses the GPU and the
matches use CPU cores: leave the trainer the cores it needs (train.py
--threads, the data prefetch thread) when choosing --workers.

The Elo of a few hundred pairs has an error of about +/- 15-30: use the
ranking to pick candidates, then confirm the best one with an SPRT
(match.py --sprt), as in step 6 of the README.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time

import match as M

NET_RE = re.compile(r"^epoch_(\d+)\.nnue$")
RESULTS = "watch.json"


def list_nets(run: str, every: int) -> list[tuple[int, str]]:
    """(epoch, path) of the finished networks of a run, by epoch."""
    nets = []
    try:
        names = os.listdir(run)
    except FileNotFoundError:
        return nets
    now = time.time()
    for name in names:
        m = NET_RE.match(name)
        if not m:
            continue
        epoch, path = int(m.group(1)), os.path.join(run, name)
        if epoch % every:
            continue
        # train.py writes networks atomically; the age check also covers
        # files copied in by other means while they are being written
        try:
            if now - os.path.getmtime(path) < 2.0:
                continue
        except FileNotFoundError:
            continue
        nets.append((epoch, path))
    return sorted(nets)


def load_results(run: str) -> dict:
    try:
        with open(os.path.join(run, RESULTS), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_results(run: str, results: dict):
    path = os.path.join(run, RESULTS)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, sort_keys=True)
    os.replace(path + ".tmp", path)


def match_args(args, net: str) -> argparse.Namespace:
    """match.py arguments for 'net' (A) against the reference (B)."""
    m = M.build_parser().parse_args([])
    m.engine = args.engine
    m.engine_b = args.engine_ref
    m.net_a, m.net_b = net, args.ref
    m.opt_a = list(args.opt)
    m.opt_b = list(args.opt_ref if args.opt_ref is not None else args.opt)
    m.nodes, m.depth, m.movetime = args.nodes, args.depth, args.movetime
    m.workers, m.hash = args.workers, args.hash
    m.random_plies, m.random_plies_min = args.random_plies, args.random_plies_min
    m.max_plies, m.adjudicate, m.full_info = args.max_plies, args.adjudicate, args.full_info
    return M.finalize_args(m)


def measure(args, net: str, previous: dict | None) -> dict:
    """Plays the pairs that 'net' still needs against the reference.
    Previous results are topped up with the next pairs of the same sequence
    (pair i has the same opening and identities for every network)."""
    penta = list(previous["penta"]) if previous else [0] * 5
    done = sum(penta)
    secs = (previous or {}).get("seconds", 0.0)
    if args.pairs > done:
        t0 = time.time()
        pairs = list(M.play_pairs(match_args(args, net), args.pairs - done, args.seed, start=done))
        for i, n in enumerate(M.pentanomial(pairs)):
            penta[i] += n
        secs += time.time() - t0
    elo, err = M.elo_estimate(penta)
    return {"penta": penta, "pairs": sum(penta), "elo": round(elo, 1),
            "err": round(err, 1) if err != float("inf") else None,
            "seconds": round(secs, 1)}


def ranking(all_results: list[tuple[str, int, dict]], min_pairs: int) -> list:
    ok = [r for r in all_results if r[2]["pairs"] >= min_pairs]
    return sorted(ok, key=lambda r: (-r[2]["elo"], r[1]))


def print_ranking(rows, top: int):
    print(f"{'rank':>4}  {'Elo':>7}  {'+/-':>6}  {'pairs':>5}  network", flush=True)
    for i, (path, epoch, r) in enumerate(rows[:top], 1):
        err = f"{r['err']:.1f}" if r["err"] is not None else "-"
        print(f"{i:>4}  {r['elo']:>+7.1f}  {err:>6}  {r['pairs']:>5}  {path}", flush=True)


def copy_best(run: str, src: str):
    dst = os.path.join(run, "best.nnue")
    shutil.copyfile(src, dst + ".tmp")
    os.replace(dst + ".tmp", dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="training output directories (train.py --out)")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--engine-ref", help="engine of the reference (default: --engine)")
    ap.add_argument("--ref", required=True, help="reference network (e.g. the current one)")
    ap.add_argument("--opt", nargs="*", default=[], help="UCI options, Name=Value")
    ap.add_argument("--opt-ref", nargs="*", default=None,
                    help="UCI options of the reference (default: --opt)")
    ap.add_argument("--pairs", type=int, default=200, help="game pairs per network")
    ap.add_argument("--min-pairs", type=int, default=0,
                    help="pairs needed to be ranked (default: --pairs)")
    ap.add_argument("--every", type=int, default=1, metavar="K", help="only epochs multiple of K")
    ap.add_argument("--nodes", type=int, default=5000)
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--movetime", type=int, default=0, help="ms per move")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--hash", type=int, default=16)
    ap.add_argument("--random-plies", type=int, default=6)
    ap.add_argument("--random-plies-min", type=int, default=2)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--adjudicate", type=int, default=3000, help="internal units")
    ap.add_argument("--full-info", action="store_true")
    ap.add_argument("--seed", type=int, default=1,
                    help="openings / identities, the same for every network")
    ap.add_argument("--follow", action="store_true", help="keep watching for new networks")
    ap.add_argument("--poll", type=float, default=60.0, help="seconds between scans (--follow)")
    ap.add_argument("--until-epoch", type=int, default=0,
                    help="--follow: stop once this epoch has been measured")
    ap.add_argument("--idle", type=float, default=0.0,
                    help="--follow: stop after this many minutes without a new network (0: never)")
    ap.add_argument("--top", type=int, default=10, help="rows of the ranking printed")
    ap.add_argument("--no-best", action="store_true", help="do not write <run>/best.nnue")
    ap.add_argument("--tensorboard", action="store_true",
                    help="write match/elo_vs_ref to <run>/tb (needs the tensorboard package)")
    args = ap.parse_args()
    args.engine_ref = args.engine_ref or args.engine
    args.ref = os.path.abspath(args.ref)
    args.min_pairs = args.min_pairs or args.pairs
    if not os.path.exists(args.ref):
        raise SystemExit(f"reference network {args.ref} not found")
    go = (f"depth {args.depth}" if args.depth else
          f"movetime {args.movetime}" if args.movetime else f"nodes {args.nodes}")
    print(f"reference {args.ref}, {args.pairs} pairs per network, go {go}, "
          f"{args.workers} workers", flush=True)

    writers = {}

    def tb_writer(run):
        if not args.tensorboard:
            return None
        if run not in writers:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as e:
                raise SystemExit(f"--tensorboard needs the tensorboard package: {e}")
            writers[run] = SummaryWriter(os.path.join(run, "tb"))
        return writers[run]

    results = {run: load_results(run) for run in args.runs}
    best = {run: None for run in args.runs}
    last_new = time.time()
    try:
        while True:
            measured_any = False
            for run in args.runs:
                for epoch, path in list_nets(run, args.every):
                    name = os.path.basename(path)
                    prev = results[run].get(name)
                    if prev and (prev.get("ref"), prev.get("go")) != (args.ref, go):
                        prev = None      # another reference or search limit: measure again
                    if prev and prev["pairs"] >= args.pairs:
                        continue
                    r = measure(args, path, prev)
                    r.update({"epoch": epoch, "ref": args.ref, "go": go})
                    results[run][name] = r
                    save_results(run, results[run])
                    measured_any = True
                    last_new = time.time()
                    err = f"{r['err']:.1f}" if r["err"] is not None else "-"
                    print(f"{path}: Elo {r['elo']:+.1f} +/- {err} vs {args.ref} "
                          f"({r['pairs']} pairs, penta {r['penta']}, {r['seconds']:.0f}s)",
                          flush=True)
                    w = tb_writer(run)
                    if w:
                        w.add_scalar("match/elo_vs_ref", r["elo"], epoch)
                        w.flush()

            rows = ranking([(os.path.join(run, name), r["epoch"], r)
                            for run in args.runs for name, r in results[run].items()
                            if (r.get("ref"), r.get("go")) == (args.ref, go)], args.min_pairs)
            if measured_any and rows:
                print_ranking(rows, args.top)
            if not args.no_best:
                for run in args.runs:
                    own = [x for x in rows if os.path.dirname(x[0]) == run]
                    if own and own[0][0] != best[run]:
                        best[run] = own[0][0]
                        copy_best(run, own[0][0])
                        print(f"{run}/best.nnue <- {own[0][0]} (Elo {own[0][2]['elo']:+.1f})",
                              flush=True)

            if not args.follow:
                break
            if args.until_epoch and any(r["epoch"] >= args.until_epoch
                                        for run in args.runs for r in results[run].values()
                                        if (r.get("ref"), r.get("go")) == (args.ref, go)):
                print(f"epoch {args.until_epoch} measured: done", flush=True)
                break
            if args.idle and time.time() - last_new > args.idle * 60:
                print(f"no new network for {args.idle:g} minutes: done", flush=True)
                break
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("interrupted (results so far are in watch.json)", flush=True)
    finally:
        for w in writers.values():
            w.close()


if __name__ == "__main__":
    main()
