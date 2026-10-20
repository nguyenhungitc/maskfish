#!/usr/bin/env python3
"""Local fishtest for Maskfish: a web page to queue engine tests and a runner
that plays them on this machine with an SPRT, like the fishtest of Stockfish
and Pikafish but without distributed workers.

    python fishtest.py                      # http://127.0.0.1:6543
    python fishtest.py --concurrency 14 --port 8000
    python fishtest.py --host 0.0.0.0 --password secret

See README.md.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import secrets
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import builder as B  # noqa: E402
from db import DB    # noqa: E402

DEFAULT_REPO = "https://github.com/nguyenhungitc/maskfish"


def default_branch(url: str) -> str:
    try:
        out = subprocess.run(["git", "ls-remote", "--symref", url, "HEAD"], capture_output=True,
                             text=True, timeout=20).stdout
        for line in out.splitlines():
            if line.startswith("ref: refs/heads/"):
                return line.split()[1][len("refs/heads/"):]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "master"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="address to listen on (0.0.0.0: all)")
    ap.add_argument("--port", type=int, default=6543)
    ap.add_argument("--data", default=str(HERE / "data"), help="directory of the database, builds, nets")
    ap.add_argument("--concurrency", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="cores used for games (default: all but one)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="make -j for builds")
    ap.add_argument("--arch", default="native", help="ARCH of the builds (make help)")
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help="default repository of the tests (URL, owner/repo or local path)")
    ap.add_argument("--base-branch", help="default base branch (default: the repository's HEAD)")
    ap.add_argument("--net-repo", "--net-url", dest="net_url", default=B.DEFAULT_NET_REPO,
                    help="repository of the default network, which is the maskfish.nnue of its "
                         "hidden ref refs/assets/net (URL, owner/repo or local path)")
    ap.add_argument("--timemargin", type=int, default=200,
                    help="ms an engine may exceed its clock before losing on time")
    ap.add_argument("--password", default=os.environ.get("FISHTEST_PASSWORD"),
                    help="password asked for every change (user name is free); also FISHTEST_PASSWORD")
    ap.add_argument("--spsa", action="store_true", default=os.environ.get("FISHTEST_SPSA") == "1",
                    help="enable SPSA tuning tests (off by default); also FISHTEST_SPSA=1")
    ap.add_argument("--no-runner", action="store_true", help="web interface only")
    ap.add_argument("--runner-only", action="store_true", help="runner only, no web interface")
    args = ap.parse_args()

    data = Path(args.data).resolve()
    data.mkdir(parents=True, exist_ok=True)
    repo = B.normalize_repo(args.repo)
    if os.path.exists(repo):
        repo = str(Path(repo).resolve())
    settings = {
        "data_dir": str(data), "concurrency": max(1, args.concurrency), "jobs": max(1, args.jobs),
        "arch": args.arch, "repo": repo, "base_branch": args.base_branch or default_branch(repo),
        "net_url": args.net_url, "timemargin": args.timemargin, "password": args.password,
        "spsa": args.spsa, "host": args.host, "port": args.port,
    }
    secret_file = data / "secret"
    if not secret_file.exists():
        secret_file.write_text(secrets.token_hex(32))
    settings["secret"] = secret_file.read_text().strip()
    DB(str(data / "fishtest.db"))       # creates the schema

    if args.runner_only:
        import runner
        runner.runner_main(settings)
        return

    proc = None
    if not args.no_runner:
        import runner
        proc = mp.get_context("spawn").Process(target=runner.runner_main, args=(settings,),
                                               name="fishtest-runner")
        proc.start()

    from app import create_app
    app = create_app(settings)
    print(f"Maskfish fishtest: http://{args.host}:{args.port}  (data: {data}, "
          f"{settings['concurrency']} games at a time, default base {settings['base_branch']})",
          flush=True)
    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        if proc and proc.is_alive():
            proc.terminate()
            proc.join(5)


if __name__ == "__main__":
    main()
