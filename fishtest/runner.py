"""The runner: a single local "worker" that takes the pending tests one at a
time (highest priority first, then oldest), builds both engines, and plays
game pairs on all the configured cores until the SPRT decides, the number of
games is reached, or the test is stopped from the web interface.

It runs in its own process (started by fishtest.py) and only communicates
with the web server through the database.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import random
import shlex
import time
import traceback
from pathlib import Path

import builder as B
import config as C
import games as G
import spsa as P
import stats as S
from db import ACTIVE, DB

MAX_LLR_POINTS = 1000


class Stopped(Exception):
    pass


class Requeued(Exception):
    """The test goes back to the queue (runner paused, concurrency changed)."""


def parse_options(text: str) -> dict:
    """'Hash=64 Threads=2 Foo=bar' -> dict (values without spaces)."""
    opts = {}
    for item in shlex.split(text or ""):
        if "=" not in item:
            raise ValueError(f"bad option {item!r}: expected Name=Value")
        k, v = item.split("=", 1)
        opts[k] = v
    return opts


class Runner:
    def __init__(self, settings: dict):
        self.base = settings
        self.data = Path(settings["data_dir"])
        self.db = DB(str(self.data / "fishtest.db"))
        self.repo = B.Repo(self.data / "repo.git")
        self.builds = self.data / "builds"
        self.nets = self.data / "nets"
        self.games_dir = self.data / "games"
        for d in (self.builds, self.nets, self.games_dir):
            d.mkdir(parents=True, exist_ok=True)

    def cfg(self) -> dict:
        """Current settings: command line + values saved on the Settings page."""
        return C.effective(self.db, self.base)

    def heartbeat(self, state: str, run_id=None, workers=0):
        self.db.set_config("runner", {"state": state, "run": run_id, "workers": workers,
                                      "time": time.time(), "pid": os.getpid()})

    def loop(self):
        if not B.is_network(self.nets / B.NET_NAME):
            try:
                B.fetch_default_net(self.nets, self.cfg()["net_url"])
                print(f"Downloaded the default network to {self.nets / B.NET_NAME}", flush=True)
            except Exception as e:
                print(f"Could not download the default network ({e}); the first build "
                      "will try again, or upload one on the Networks page.", flush=True)
        # Tests left active by a previous runner (crash, restart) go back to the queue
        for r in self.db.runs(ACTIVE):
            self.db.update_run(r["id"], status="pending")
            self.db.append_log(r["id"], "Runner restarted: test put back in the queue.\n")
        while True:
            mode = C.runner_mode(self.db)
            if mode != "running":
                if mode == "pause_after":            # no test running any more
                    C.set_runner_mode(self.db, "paused")
                self.heartbeat("paused")
                time.sleep(2)
                continue
            run = self.db.next_pending()
            if not run:
                self.heartbeat("idle")
                time.sleep(2)
                continue
            try:
                self.execute(run)
            except Stopped:
                self.db.append_log(run["id"], "Stopped.\n")
            except Requeued as e:
                self.db.update_run(run["id"], only_if_status=ACTIVE, status="pending")
                self.db.append_log(run["id"], f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} {e}: "
                                              "test put back in the queue\n")
            except Exception as e:
                msg = str(e) if isinstance(e, (B.BuildError, ValueError)) else traceback.format_exc()
                self.db.update_run(run["id"], only_if_status=ACTIVE, status="failed",
                                   error=msg, finished=time.time())
                self.db.append_log(run["id"], f"ERROR: {msg}\n")

    # ------------------------------------------------------------ one test

    def workers_for(self, threads: int) -> int:
        return max(1, self.cfg()["concurrency"] // max(1, threads))

    def check_status(self, run_id, workers=None, threads=1):
        """Stops when the test was stopped from the web page, requeues it when
        the runner is paused or the concurrency changed."""
        r = self.db.get_run(run_id)
        if not r or r["status"] not in ACTIVE:
            raise Stopped()
        if C.runner_mode(self.db) == "paused":
            raise Requeued("runner paused")
        if workers is not None and self.workers_for(threads) != workers:
            raise Requeued("concurrency changed")

    def net_path(self, name: str) -> Path:
        name = name or B.NET_NAME
        p = self.nets / name
        if not B.is_network(p):
            raise ValueError(f"network {name} not found in {self.nets} "
                             "(upload it on the Networks page)")
        return p.resolve()

    def execute(self, run):
        rid = run["id"]
        if not self.db.update_run(rid, only_if_status="pending", status="building", error=""):
            return
        log = lambda text: self.db.append_log(rid, text)
        self.heartbeat("building", rid)
        log(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} building\n")

        c = self.cfg()
        binaries = {}
        for side in ("new", "base"):
            binaries[side] = B.build(self.repo, run[f"{side}_sha"], run[f"{side}_make"],
                                     c["arch"], c["jobs"], self.builds, self.nets, log)
            self.check_status(rid)

        sides = {}
        for side in ("new", "base"):
            net = self.net_path(run[f"{side}_net"])
            sides[side] = {"binary": str(binaries[side].resolve()), "net": str(net),
                           "options": parse_options(run[f"{side}_options"])}
            err = G.check_engine(sides[side]["binary"], sides[side], run["threads"], run["hash"])
            if err:
                raise ValueError(f"the {side} engine does not work:\n{err}")
            if not run[f"{side}_bench"]:
                b = B.bench(binaries[side], net)
                if b:
                    self.db.update_run(rid, **{f"{side}_bench": f"{b['nodes']} ({b['nps']} nps)"})
                    log(f"{side} bench: {b['nodes']} nodes, {b['nps']} nps\n")
            self.check_status(rid)

        if run["test_type"] == "spsa" and not run["spsa"].get("params"):
            params = P.resolve(P.parse_params(run["spsa"].get("text", "")),
                               G.spin_options(sides["new"]["binary"]))
            state = P.init_state(params, run["num_games"] // 2, run["spsa"]["A_ratio"],
                                 run["spsa"]["alpha"], run["spsa"]["gamma"])
            self.db.update_run(rid, spsa={**run["spsa"], **state})
            log("SPSA parameters:\n" + P.params_text(params) + "\n")

        workers = self.workers_for(run["threads"])
        cfg = {
            "new": sides["new"], "base": sides["base"],
            "threads": run["threads"], "hash": run["hash"],
            "tc": G.parse_tc(run["tc"]), "timemargin": c["timemargin"],
            "move_timeout": 600, "max_plies": 400,
            "adjudicate": run["adjudicate"], "full_info": bool(run["full_info"]),
            "plies_min": run["plies_min"], "plies_max": run["plies_max"],
        }
        if not self.db.update_run(rid, only_if_status="building", status="running",
                                  started=run["started"] or time.time()):
            raise Stopped()
        log(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} playing with {workers} workers\n")
        self.heartbeat("running", rid, workers)
        self.play(rid, cfg, workers)

    def play(self, rid, cfg, workers):
        run = self.db.get_run(rid)
        penta = list(run["penta"])
        w, l, d = run["wins"], run["losses"], run["draws"]
        crashes, time_losses = run["crashes"], run["time_losses"]
        history = list(run["llr_history"])
        sprt = run["test_type"] == "sprt"
        spsa = run["spsa"] if run["test_type"] == "spsa" else None
        pairs_done = sum(penta)
        rng = random.Random(run["seed"] * 7919 + pairs_done)     # SPSA flips
        steps = {}
        target = run["num_games"] // 2 if run["num_games"] else None
        if target is not None and pairs_done >= target:
            self.finish(rid, "")
            return

        ctx = mp.get_context("spawn")
        results: queue.Queue = queue.Queue()
        pool = ctx.Pool(workers, initializer=G.worker_init, initargs=(cfg,))
        games_file = open(self.games_dir / f"run-{rid}.txt", "a", encoding="utf-8")
        next_index, inflight, errors = pairs_done, 0, 0
        play_t0, base_seconds = time.time(), run["play_seconds"]
        last_check, last_beat = 0.0, 0.0
        try:
            while True:
                while inflight < 2 * workers and (target is None or next_index < target):
                    seed = (run["seed"] * 1000003 + next_index) % (2 ** 63)
                    args = (next_index, seed)
                    if spsa:
                        step = P.perturb(spsa, next_index + 1, rng)
                        steps[next_index] = step
                        args += (step["plus"], step["minus"])
                    pool.apply_async(G.play_pair, args,
                                     callback=results.put,
                                     error_callback=lambda e: results.put(e))
                    next_index += 1
                    inflight += 1
                if inflight == 0:
                    self.finish(rid, "")
                    return
                try:
                    r = results.get(timeout=1)
                except queue.Empty:
                    r = None
                now = time.time()
                if now - last_check > 2:
                    self.check_status(rid, workers, run["threads"])
                    last_check = now
                if now - last_beat > 10:
                    self.heartbeat("running", rid, workers)
                    last_beat = now
                if r is None:
                    continue
                inflight -= 1
                if isinstance(r, BaseException):
                    errors += 1
                    self.db.append_log(rid, f"worker error: {r}\n")
                    if errors >= 10:
                        raise ValueError(f"too many worker errors, the last one: {r}")
                    continue
                errors = 0
                r1, r2 = r["scores"]
                penta[(r1 + 1) + (r2 + 1)] += 1
                for x in (r1, r2):
                    w, l, d = w + (x == 1), l + (x == -1), d + (x == 0)
                crashes += r["crashes"]
                time_losses += r["time_losses"]
                games_file.write("".join(r["games"]))
                games_file.flush()
                fields = {"penta": penta, "wins": w, "losses": l, "draws": d,
                          "crashes": crashes, "time_losses": time_losses,
                          "play_seconds": base_seconds + time.time() - play_t0}
                decision = None
                if spsa:
                    step = steps.pop(r["index"], None)
                    if step:
                        P.update(spsa, step, r1 + r2, sum(penta))
                    fields["spsa"] = spsa
                if sprt:
                    v, _, _, decision = S.sprt_state(penta, run["elo0"], run["elo1"], run["elo_model"],
                                                     run["alpha"], run["beta"])
                    fields["llr"] = v
                    history.append([2 * sum(penta), round(v, 4)])
                    if len(history) > MAX_LLR_POINTS:
                        history = history[::2] + ([history[-1]] if len(history) % 2 == 0 else [])
                    fields["llr_history"] = history
                if not self.db.update_run(rid, only_if_status="running", **fields):
                    raise Stopped()
                if decision:
                    self.finish(rid, decision)
                    return
                if target is not None and sum(penta) >= target:
                    self.finish(rid, "")
                    return
        finally:
            pool.terminate()
            pool.join()
            games_file.close()

    def finish(self, rid, result):
        if self.db.update_run(rid, only_if_status="running", status="finished",
                              result=result, finished=time.time()):
            self.db.append_log(rid, f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} finished"
                                    f"{': ' + result if result else ''}\n")


def runner_main(settings: dict):
    import signal

    def on_term(*_):
        raise SystemExit(0)            # runs the finally blocks: the pool is terminated
    signal.signal(signal.SIGTERM, on_term)
    try:
        Runner(settings).loop()
    except (KeyboardInterrupt, SystemExit):
        pass


def new_seed() -> int:
    return random.SystemRandom().randrange(1, 2 ** 31)
