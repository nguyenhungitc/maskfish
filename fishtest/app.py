"""Web interface: the list of tests, the form to submit a test, the page of a
test, and the networks. Modeled on fishtest (Stockfish / Pikafish), for a
single machine: there are no workers, users or approvals.
"""

from __future__ import annotations

import functools
import hashlib
import html
import math
import os
import re
import shutil
import time
from pathlib import Path

from flask import (Flask, Response, abort, flash, jsonify, redirect, render_template, request,
                   send_file, url_for)
from markupsafe import Markup

import builder as B
import config as C
import games as G
import runner as R
import spsa as P
import stats as S
from db import ACTIVE, DB

PER_PAGE = 25

DEFAULTS = {
    "tc": "10+0.1", "threads": 1, "hash": 16, "test_type": "sprt", "elo0": 0.0, "elo1": 2.0,
    "elo_model": "normalized", "alpha": 0.05, "beta": 0.05, "num_games": 0, "priority": 0,
    "plies_min": 2, "plies_max": 6, "adjudicate": 3000, "full_info": 0,
    "base_net": B.NET_NAME, "new_net": B.NET_NAME, "base_options": "", "new_options": "",
    "base_make": "", "new_make": "", "info": "",
    "spsa_params": "", "spsa_A": P.DEFAULT_A_RATIO, "spsa_alpha": P.DEFAULT_ALPHA,
    "spsa_gamma": P.DEFAULT_GAMMA,
}
CHART_COLORS = ["#2463eb", "#e5484d", "#30a46c", "#f5a623", "#8e4ec6", "#12a594", "#d6409f",
                "#6b7280", "#a2711d", "#0090ff"]


def create_app(settings: dict) -> Flask:
    app = Flask(__name__)
    app.secret_key = settings.get("secret") or hashlib.sha256(
        str(Path(settings["data_dir"]).resolve()).encode()).hexdigest()
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024
    data = Path(settings["data_dir"])
    db = DB(str(data / "fishtest.db"))
    repo = B.Repo(data / "repo.git")
    nets_dir = data / "nets"
    nets_dir.mkdir(parents=True, exist_ok=True)
    games_dir = data / "games"
    builds_dir = data / "builds"

    def eff() -> dict:
        """Command-line settings with the values saved on the Settings page."""
        return C.effective(db, settings)

    # ------------------------------------------------------------ helpers

    def protected(view):
        """Actions that change something ask for the password, if one is set."""
        @functools.wraps(view)
        def wrapper(*a, **kw):
            if C.password_required(db, settings):
                auth = request.authorization
                if not C.check_password(db, settings, auth.password if auth else None):
                    return Response("Password required", 401,
                                    {"WWW-Authenticate": 'Basic realm="fishtest"'})
            return view(*a, **kw)
        return wrapper

    def list_nets():
        out = []
        for p in sorted(nets_dir.glob("*.nnue")):
            st = p.stat()
            out.append({"name": p.name, "size": st.st_size, "mtime": st.st_mtime,
                        "valid": B.is_network(p)})
        return out

    def runner_state():
        st = db.get_config("runner")
        if not st:
            return {"label": "runner not started", "cls": "off"}
        mode = C.runner_mode(db)
        if time.time() - st["time"] > 60:
            return {"label": "runner offline", "cls": "off", "mode": mode}
        if st["state"] == "paused":
            return {"label": "runner paused", "cls": "paused", "mode": mode}
        if st["state"] == "idle":
            return {"label": "runner idle", "cls": "idle", "mode": mode}
        lbl = f"{st['state']} #{st['run']}"
        if st["state"] == "running":
            n = st["workers"]
            lbl += f" · {n} game{'s' if n != 1 else ''} at a time"
        if mode == "pause_after":
            lbl += " · pausing after this test"
        return {"label": lbl, "cls": "busy", "run": st["run"], "mode": mode}

    @app.context_processor
    def inject():
        return {"runner": runner_state(), "settings": eff()}

    # ------------------------------------------------------------ filters

    @app.template_filter("ago")
    def ago(t):
        if not t:
            return ""
        d = time.time() - t
        for unit, sec in (("d", 86400), ("h", 3600), ("min", 60)):
            if d >= sec:
                return f"{int(d // sec)}{unit} ago"
        return "just now"

    @app.template_filter("datetime")
    def fmt_datetime(t):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) if t else ""

    @app.template_filter("duration")
    def duration(s):
        s = int(s or 0)
        h, m = divmod(s // 60, 60)
        return f"{h}h {m:02d}min" if h else f"{m}min {s % 60:02d}s"

    @app.template_filter("size")
    def size(n):
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024 or unit == "GB":
                return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
            n /= 1024

    def commit_url(run, sha):
        web = B.github_web(run["repo"])
        return f"{web}/commit/{sha}" if web else None

    def compare_url(run):
        web = B.github_web(run["repo"])
        return f"{web}/compare/{run['base_sha'][:10]}...{run['new_sha'][:10]}" if web else None

    app.jinja_env.globals.update(commit_url=commit_url, compare_url=compare_url)

    # ------------------------------------------------------------ results

    def results(run) -> dict:
        """What the results box shows, as in fishtest."""
        penta = run["penta"]
        games = run["wins"] + run["losses"] + run["draws"]
        es = S.elo_stats(penta)
        lines = []
        cls = run["status"]
        if run["test_type"] == "spsa":
            n = run["num_games"] // 2
            lines.append(f"SPSA: {sum(penta)}/{n} iterations")
        elif run["test_type"] == "sprt":
            lo, hi = S.sprt_bounds(run["alpha"], run["beta"])
            lines.append(f"LLR: {run['llr']:.2f} ({lo:.2f},{hi:.2f}) "
                         f"<{run['elo0']:.2f},{run['elo1']:.2f}>"
                         + ("" if run["elo_model"] == "normalized" else " logistic"))
            if run["status"] == "finished":
                cls = run["result"] or "finished"
        else:
            ci = "inf" if math.isinf(es["elo_ci"]) else f"{es['elo_ci']:.2f}"
            lines.append(f"Elo: {es['elo']:.2f} ± {ci} (95%) LOS: {es['los'] * 100:.1f}%")
            if run["status"] == "finished" and games:
                if es["elo"] - es["elo_ci"] > 0:
                    cls = "passed"
                elif es["elo"] + es["elo_ci"] < 0:
                    cls = "failed"
                else:
                    cls = "finished"
        lines.append(f"Total: {games} W: {run['wins']} L: {run['losses']} D: {run['draws']}")
        lines.append("Ptnml(0-2): " + ", ".join(str(c) for c in penta))
        if run["crashes"] or run["time_losses"]:
            lines.append(f"Crashes: {run['crashes']} Time losses: {run['time_losses']}")
        return {"lines": lines, "cls": cls, "games": games, "elo": es}

    app.jinja_env.globals.update(results=results)

    def llr_chart(run) -> Markup:
        hist = run["llr_history"]
        if run["test_type"] != "sprt" or len(hist) < 2:
            return Markup("")
        lo, hi = S.sprt_bounds(run["alpha"], run["beta"])
        xs = [h[0] for h in hist]
        ys = [h[1] for h in hist]
        ymin, ymax = min(lo, min(ys)) - 0.3, max(hi, max(ys)) + 0.3
        W, H, P = 640, 220, 34
        xmax = max(xs[-1], 1)
        X = lambda x: P + (W - P - 8) * x / xmax
        Y = lambda y: 8 + (H - 30) * (ymax - y) / (ymax - ymin)
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))
        parts = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="LLR">']
        for v, c in ((hi, "pos"), (0, "zero"), (lo, "neg")):
            parts.append(f'<line class="b-{c}" x1="{P}" x2="{W - 8}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                         f'<text x="{P - 4}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:.2f}</text>')
        parts.append(f'<polyline class="llr" points="{pts}"/>')
        parts.append(f'<text x="{W - 8}" y="{H - 6}" text-anchor="end">{xmax} games</text>')
        parts.append(f'<text x="{P}" y="{H - 6}">0</text></svg>')
        return Markup("".join(parts))

    def spsa_chart(run) -> Markup:
        st = run["spsa"]
        hist = st.get("history", [])
        if len(hist) < 2:
            return Markup("")
        params = st["params"]
        W, H, PAD = 640, 220, 40
        xmax = max(hist[-1][0], 1)
        X = lambda x: PAD + (W - PAD - 8) * x / xmax
        Y = lambda v: 8 + (H - 30) * (100 - v) / 100       # % of [min, max]
        parts = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="SPSA parameters">']
        for v in (0, 50, 100):
            parts.append(f'<line class="b-zero" x1="{PAD}" x2="{W - 8}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                         f'<text x="{PAD - 4}" y="{Y(v) + 4:.1f}" text-anchor="end">{v}%</text>')
        for i, p in enumerate(params):
            span = p["max"] - p["min"]
            pts = " ".join(f"{X(h[0]):.1f},{Y(100 * (h[1][i] - p['min']) / span):.1f}" for h in hist)
            color = CHART_COLORS[i % len(CHART_COLORS)]
            parts.append(f'<polyline fill="none" stroke-width="1.6" stroke="{color}" points="{pts}">'
                         f'<title>{html.escape(p["name"])}</title></polyline>')
        parts.append(f'<text x="{W - 8}" y="{H - 6}" text-anchor="end">{xmax} iterations</text>')
        parts.append(f'<text x="{PAD}" y="{H - 6}">0</text></svg>')
        legend = "".join(f'<span><i style="background:{CHART_COLORS[i % len(CHART_COLORS)]}"></i>'
                         f'{html.escape(p["name"])}</span>' for i, p in enumerate(params))
        return Markup("".join(parts) + f'<div class="legend">{legend}</div>')

    def spsa_rows(run) -> list[dict]:
        st = run["spsa"]
        if not st.get("params"):
            return []
        k = max(1, sum(run["penta"]))
        rows = []
        for p, (c, r) in zip(st["params"], P.gains(st, k)):
            rows.append({**p, "c_k": c, "r_k": r})
        return rows

    # ------------------------------------------------------------ pages

    @app.route("/")
    def index():
        return redirect(url_for("tests"))

    @app.route("/tests")
    def tests():
        page = max(1, request.args.get("page", 1, type=int))
        finished_states = ("finished", "stopped", "failed")
        ctx = {
            "active": db.runs(ACTIVE),
            "pending": db.runs(("pending",)),
            "finished": db.runs(finished_states, order="COALESCE(finished, updated) DESC, id DESC",
                                limit=PER_PAGE, offset=(page - 1) * PER_PAGE),
            "page": page,
            "total": db.count(finished_states),
        }
        ctx.update({
            "pages": max(1, math.ceil(ctx["total"] / PER_PAGE)),
        })
        if request.args.get("partial"):
            return render_template("_runs.html", **ctx)
        return render_template("tests.html", **ctx)

    @app.route("/tests/view/<int:rid>")
    def view(rid):
        run = db.get_run(rid) or abort(404)
        ctx = {"run": run,
               "chart": spsa_chart(run) if run["test_type"] == "spsa" else llr_chart(run),
               "spsa_rows": spsa_rows(run) if run["test_type"] == "spsa" else [],
               "games_file": (games_dir / f"run-{rid}.txt").exists()}
        if request.args.get("partial"):
            return render_template("_live.html", **ctx)
        return render_template("view.html", **ctx)

    @app.route("/api/runs/<int:rid>")
    def api_run(rid):
        run = db.get_run(rid) or abort(404)
        run.pop("log", None)
        r = results(run)
        run.update(elo=r["elo"], games=r["games"])
        return jsonify(run)

    @app.route("/tests/diff/<int:rid>")
    def diff(rid):
        run = db.get_run(rid) or abort(404)
        try:
            stat, full = repo.diff(run["base_sha"], run["new_sha"])
            commits = repo.log(run["base_sha"], run["new_sha"])
        except B.BuildError as e:
            stat, full, commits = str(e), "", []
        lines = []
        for line in full.splitlines():
            cls = ("add" if line.startswith("+") and not line.startswith("+++") else
                   "del" if line.startswith("-") and not line.startswith("---") else
                   "hunk" if line.startswith("@@") else
                   "file" if line.startswith(("diff --git", "+++", "---", "index ")) else "")
            lines.append(f'<span class="{cls}">{html.escape(line)}</span>' if cls else html.escape(line))
        return render_template("diff.html", run=run, stat=stat, commits=commits,
                               diff=Markup("\n".join(lines)), diff_lines=len(lines))

    @app.route("/tests/games/<int:rid>")
    def games(rid):
        p = games_dir / f"run-{rid}.txt"
        if not p.exists():
            abort(404)
        return send_file(p.resolve(), mimetype="text/plain", as_attachment=True,
                         download_name=f"maskfish-test-{rid}-games.txt")

    # ------------------------------------------------------------ new test

    def form_values(source: dict) -> dict:
        v = dict(DEFAULTS)
        cur = eff()
        v.update(repo=cur["repo"], base_ref=cur["base_branch"], new_ref="",
                 username=request.cookies.get("username", ""))
        v.update({k: source[k] for k in source if k in v})
        return v

    @app.route("/tests/run", methods=["GET"])
    def run_form():
        src = {}
        clone = request.args.get("clone", type=int)
        if clone:
            old = db.get_run(clone)
            if old:
                src = {k: old[k] for k in list(DEFAULTS) + ["repo", "base_ref", "new_ref", "username"]
                       if k in old}
                sp = old["spsa"] or {}
                if old["test_type"] == "spsa":
                    src.update(spsa_params=P.params_text(sp["params"]) if sp.get("params")
                               else sp.get("text", ""),
                               spsa_A=sp.get("A_ratio"), spsa_alpha=sp.get("alpha"),
                               spsa_gamma=sp.get("gamma"))
                if request.args.get("tuned") and sp.get("params"):
                    # SPRT of the tuned values against the same commit with its defaults
                    tuned = " ".join(f"{k}={v}" for k, v in P.current_values(sp).items())
                    src.update(test_type="sprt", base_ref=old["new_sha"][:10],
                               new_options=(old["new_options"] + " " + tuned).strip(),
                               base_options=old["new_options"], base_net=old["new_net"],
                               base_make=old["new_make"], new_ref=old["new_sha"][:10],
                               info=f"SPSA #{old['id']} tuned values: {tuned}")
        return render_template("run.html", v=form_values(src), nets=list_nets(), errors=[])

    @app.route("/tests/run", methods=["POST"])
    @protected
    def run_submit():
        f = request.form
        v = form_values({k: f.get(k, "").strip() for k in f})
        v["full_info"] = 1 if f.get("full_info") else 0
        errors = []

        def num(key, kind, lo=None, hi=None):
            try:
                x = kind(v[key])
            except (TypeError, ValueError):
                errors.append(f"{key}: not a number")
                return DEFAULTS.get(key, 0)
            if (lo is not None and x < lo) or (hi is not None and x > hi):
                errors.append(f"{key}: must be between {lo} and {hi}")
            return x

        fields = {k: num(k, int, *r) for k, r in {
            "threads": (1, 256), "hash": (1, 65536), "num_games": (0, 10 ** 7),
            "priority": (-100, 100), "plies_min": (0, 60), "plies_max": (0, 60),
            "adjudicate": (100, 100000)}.items()}
        fields.update({k: num(k, float) for k in ("elo0", "elo1", "alpha", "beta")})
        fields["test_type"] = v["test_type"] if v["test_type"] in ("numgames", "spsa") else "sprt"
        spsa_field = {}
        if fields["test_type"] == "spsa":
            if not eff().get("spsa"):
                errors.append("SPSA is disabled on this server (start it with --spsa)")
            try:
                P.parse_params(v["spsa_params"])
            except P.SpsaError as e:
                errors.append(f"SPSA parameters: {e}")
            sp = {k: num(f"spsa_{k}", float) for k in ("A", "alpha", "gamma")}
            if sp["A"] < 0 or not (0 < sp["alpha"] <= 1) or not (0 < sp["gamma"] <= 1):
                errors.append("SPSA: A ratio >= 0, 0 < alpha <= 1, 0 < gamma <= 1")
            spsa_field = {"text": v["spsa_params"], "A_ratio": sp["A"], "alpha": sp["alpha"],
                          "gamma": sp["gamma"]}
            # Both sides are the same engine, with perturbed parameters
            for k in ("ref", "net", "options", "make"):
                v[f"base_{k}"] = v[f"new_{k}"]
        fields["elo_model"] = "logistic" if v["elo_model"] == "logistic" else "normalized"
        if fields["test_type"] == "sprt" and fields["elo1"] <= fields["elo0"]:
            errors.append("SPRT: elo1 must be greater than elo0")
        if fields["test_type"] in ("numgames", "spsa") and fields["num_games"] < 2:
            errors.append("number of games: at least 2")
        if not (0 < fields["alpha"] < 0.5 and 0 < fields["beta"] < 0.5):
            errors.append("alpha and beta must be between 0 and 0.5")
        fields["num_games"] -= fields["num_games"] % 2           # games are played in pairs
        if fields["plies_max"] < fields["plies_min"]:
            errors.append("random opening: the maximum is below the minimum")
        try:
            G.parse_tc(v["tc"])
        except ValueError as e:
            errors.append(str(e))
        for side in ("new", "base"):
            try:
                R.parse_options(v[f"{side}_options"])
            except ValueError as e:
                errors.append(f"{side} options: {e}")
            if not re.fullmatch(r"[\w=. +-]*", v[f"{side}_make"]):
                errors.append(f"{side} make options: only NAME=value items")
            # The default network may still be missing: the first build downloads it
            if v[f"{side}_net"] != B.NET_NAME and not B.is_network(nets_dir / v[f"{side}_net"]):
                errors.append(f"{side} network {v[f'{side}_net']!r} not found or invalid")
        if not v["new_ref"]:
            errors.append("the test branch / commit is required")

        url = B.normalize_repo(v["repo"] or eff()["repo"])
        shas = {}
        if not errors:
            for side in ("new", "base"):
                try:
                    shas[side] = repo.resolve(url, v[f"{side}_ref"])
                except B.BuildError as e:
                    errors.append(str(e))
        if errors:
            return render_template("run.html", v=v, nets=list_nets(), errors=errors), 400

        info = {s: repo.commit_info(shas[s]) for s in shas}
        rid = db.insert_run({
            **fields,
            "username": v["username"][:60], "info": v["info"][:2000], "repo": url,
            "tc": v["tc"].strip(), "full_info": v["full_info"],
            "base_ref": v["base_ref"], "base_sha": shas["base"], "base_msg": info["base"]["subject"],
            "new_ref": v["new_ref"], "new_sha": shas["new"], "new_msg": info["new"]["subject"],
            **{f"{s}_{k}": v[f"{s}_{k}"] for s in ("new", "base") for k in ("net", "options", "make")},
            "seed": R.new_seed(), "spsa": spsa_field,
        })
        resp = redirect(url_for("view", rid=rid))
        if v["username"]:
            resp.set_cookie("username", v["username"], max_age=3 * 365 * 86400)
        return resp

    # ------------------------------------------------------------ actions

    @app.route("/tests/stop/<int:rid>", methods=["POST"])
    @protected
    def stop(rid):
        if db.update_run(rid, only_if_status=("pending",) + ACTIVE, status="stopped",
                         finished=time.time()):
            db.append_log(rid, f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} stopped from the web page\n")
        return redirect(request.referrer or url_for("view", rid=rid))

    @app.route("/tests/resume/<int:rid>", methods=["POST"])
    @protected
    def resume(rid):
        run = db.get_run(rid) or abort(404)
        if run["status"] in ("stopped", "failed") or (run["status"] == "finished" and
                                                      run["test_type"] == "numgames"):
            db.update_run(rid, status="pending", finished=None, error="", result="")
            db.append_log(rid, f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} resumed\n")
        return redirect(url_for("view", rid=rid))

    @app.route("/tests/delete/<int:rid>", methods=["POST"])
    @protected
    def delete(rid):
        run = db.get_run(rid) or abort(404)
        if run["status"] in ACTIVE:
            flash("Stop the test before deleting it.")
            return redirect(url_for("view", rid=rid))
        db.delete_run(rid)
        (games_dir / f"run-{rid}.txt").unlink(missing_ok=True)
        flash(f"Test #{rid} deleted.")
        return redirect(url_for("tests"))

    @app.route("/tests/modify/<int:rid>", methods=["POST"])
    @protected
    def modify(rid):
        run = db.get_run(rid) or abort(404)
        try:
            priority = int(request.form.get("priority", run["priority"]))
            num_games = int(request.form.get("num_games", run["num_games"]))
        except ValueError:
            flash("Priority and games must be numbers.")
            return redirect(url_for("view", rid=rid))
        num_games = max(0, num_games - num_games % 2)
        db.update_run(rid, priority=max(-100, min(100, priority)), num_games=num_games,
                      info=request.form.get("info", run["info"])[:2000])
        flash("Test modified" + (" (the runner reads the number of games when the test starts "
                                 "or resumes)" if run["status"] in ACTIVE else "") + ".")
        return redirect(url_for("view", rid=rid))

    # ------------------------------------------------------------ networks

    @app.route("/nets")
    def nets():
        return render_template("nets.html", nets=list_nets())

    @app.route("/nets/upload", methods=["POST"])
    @protected
    def nets_upload():
        f = request.files.get("network")
        if not f or not f.filename:
            flash("Choose a file.")
            return redirect(url_for("nets"))
        name = Path(f.filename).name
        if request.form.get("rename") == "hash":
            name = None
        if name is not None and not re.fullmatch(r"[\w.-]+\.nnue", name):
            flash("The name must end with .nnue and contain only letters, digits, '.', '_' and '-'.")
            return redirect(url_for("nets"))
        tmp = nets_dir / ".upload.tmp"
        f.save(tmp)
        if not B.is_network(tmp):
            tmp.unlink(missing_ok=True)
            flash("This is not a Maskfish network (it must be a zstd-compressed .nnue file).")
            return redirect(url_for("nets"))
        if name is None:
            h = hashlib.sha256(tmp.read_bytes()).hexdigest()[:12]
            name = f"nn-{h}.nnue"
        if (nets_dir / name).exists():
            tmp.unlink(missing_ok=True)
            flash(f"{name} already exists.")
            return redirect(url_for("nets"))
        tmp.rename(nets_dir / name)
        flash(f"Uploaded {name}.")
        return redirect(url_for("nets"))

    @app.route("/nets/fetch", methods=["POST"])
    @protected
    def nets_fetch():
        try:
            B.fetch_default_net(repo, nets_dir, eff()["net_url"], replace=True)
            flash(f"Fetched the network of {B.NET_REF} as {B.NET_NAME}.")
        except Exception as e:
            flash(f"Fetch failed: {e}")
        return redirect(url_for("nets"))

    @app.route("/nets/delete/<name>", methods=["POST"])
    @protected
    def nets_delete(name):
        p = nets_dir / Path(name).name
        used = db.query("SELECT id FROM runs WHERE status IN ('pending', 'building', 'running') "
                        "AND (new_net = ? OR base_net = ?)", (p.name, p.name))
        if used:
            flash(f"{p.name} is used by test #{used[0]['id']}.")
        elif p.exists():
            p.unlink()
            flash(f"Deleted {p.name}.")
        return redirect(url_for("nets"))

    @app.route("/nets/download/<name>")
    def nets_download(name):
        p = nets_dir / Path(name).name
        if not p.exists():
            abort(404)
        return send_file(p.resolve(), as_attachment=True, download_name=p.name)

    # ------------------------------------------------------------ settings

    def dir_size(path: Path) -> int:
        total = 0
        if path.exists():
            for root, _, files in os.walk(path):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
        return total

    @app.route("/settings")
    def settings_page():
        cur = eff()
        return render_template(
            "settings.html", cur=cur, saved=C.saved(db), defaults=settings, errors=[],
            cpus=os.cpu_count() or 1, builds_size=dir_size(builds_dir),
            games_size=dir_size(games_dir), password_cli=bool(settings.get("password")),
            password_web=bool(db.get_config("password_hash")))

    @app.route("/settings", methods=["POST"])
    @protected
    def settings_save():
        if request.form.get("action") == "reset":
            C.reset(db)
            flash("Settings reset to the command-line values.")
            return redirect(url_for("settings_page"))
        values, errors = C.parse(request.form, settings)
        if errors:
            for e in errors:
                flash(e)
            return redirect(url_for("settings_page"))
        values["repo"] = B.normalize_repo(values["repo"])
        if os.path.exists(values["repo"]):
            values["repo"] = str(Path(values["repo"]).resolve())
        # Only keep the values that differ from the command line
        C.reset(db)
        C.save(db, {k: v for k, v in values.items() if settings.get(k) != v})
        flash("Settings saved. The runner uses them from the next build or game batch; "
              "a new concurrency restarts the current test's games.")
        return redirect(url_for("settings_page"))

    @app.route("/settings/runner/<mode>", methods=["POST"])
    @protected
    def settings_runner(mode):
        if mode in C.RUNNER_MODES:
            C.set_runner_mode(db, mode)
            flash({"running": "Runner resumed.",
                   "paused": "Runner paused: the current test goes back to the queue.",
                   "pause_after": "The runner pauses when the current test ends."}[mode])
        return redirect(request.referrer or url_for("settings_page"))

    @app.route("/settings/password", methods=["POST"])
    @protected
    def settings_password():
        if request.form.get("action") == "remove":
            C.set_password(db, "")
            flash("Password removed" + (" (the --password option still applies)."
                                        if settings.get("password") else "."))
            return redirect(url_for("settings_page"))
        pw, pw2 = request.form.get("password", ""), request.form.get("password2", "")
        if len(pw) < 4:
            flash("The password needs at least 4 characters.")
        elif pw != pw2:
            flash("The two passwords differ.")
        else:
            C.set_password(db, pw)
            flash("Password set. The browser asks for it at the next change (any user name).")
        return redirect(url_for("settings_page"))

    @app.route("/settings/clear-builds", methods=["POST"])
    @protected
    def settings_clear_builds():
        st = db.get_config("runner") or {}
        if st.get("state") in ("building", "running") and time.time() - st.get("time", 0) < 60:
            flash("Pause the runner (or wait for the current test) before deleting the builds.")
        else:
            shutil.rmtree(builds_dir, ignore_errors=True)
            builds_dir.mkdir(parents=True, exist_ok=True)
            flash("Cached builds deleted.")
        return redirect(url_for("settings_page"))

    return app
