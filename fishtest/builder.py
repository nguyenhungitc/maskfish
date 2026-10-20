"""Git and build helpers: a local cache of the repositories used by the tests,
resolution of branches / tags / commits to SHAs, diffs, and compiled engines
cached per (commit, make options, architecture).
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import threading
from pathlib import Path

NET_NAME = "maskfish.nnue"
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


class BuildError(Exception):
    pass


def normalize_repo(url: str) -> str:
    """'owner/repo' -> GitHub URL; local paths and full URLs are kept."""
    url = url.strip().rstrip("/")
    if re.fullmatch(r"[\w.-]+/[\w.-]+", url) and not os.path.exists(url):
        return f"https://github.com/{url}"
    if url.startswith("https://github.com/") and url.endswith(".git"):
        url = url[:-4]
    return url


def github_web(url: str) -> str | None:
    m = re.fullmatch(r"https://github\.com/[\w.-]+/[\w.-]+", url)
    return url if m else None


class Repo:
    """A bare repository that caches the commits of every repository used."""

    _lock = threading.Lock()

    def __init__(self, path: Path):
        self.path = Path(path)
        if not (self.path / "HEAD").exists():
            self.path.mkdir(parents=True, exist_ok=True)
            self.git("init", "--bare", "-q")

    def git(self, *args, timeout=600, binary=False, check=True):
        r = subprocess.run(["git", "--git-dir", str(self.path), *args], capture_output=True,
                           timeout=timeout, text=not binary)
        if check and r.returncode != 0:
            err = r.stderr if not binary else r.stderr.decode(errors="replace")
            raise BuildError(f"git {' '.join(args[:2])} failed:\n{err.strip()}")
        return r

    @staticmethod
    def key(url: str) -> str:
        return hashlib.sha1(url.encode()).hexdigest()[:10]

    def fetch(self, url: str):
        k = self.key(url)
        with self._lock:
            self.git("fetch", "--force", "--prune", "--no-tags", url,
                     f"+refs/heads/*:refs/ft/{k}/heads/*",
                     f"+refs/tags/*:refs/ft/{k}/tags/*")

    def _verify(self, rev: str) -> str | None:
        r = self.git("rev-parse", "--verify", "--quiet", rev + "^{commit}", check=False)
        return r.stdout.strip() if r.returncode == 0 else None

    def resolve(self, url: str, ref: str) -> str:
        """Branch, tag or commit of the repository -> full SHA."""
        ref = ref.strip()
        if not ref or ref.startswith("-") or " " in ref:
            raise BuildError(f"invalid branch or commit {ref!r}")
        self.fetch(url)
        k = self.key(url)
        for cand in (f"refs/ft/{k}/heads/{ref}", f"refs/ft/{k}/tags/{ref}"):
            sha = self._verify(cand)
            if sha:
                return sha
        if re.fullmatch(r"[0-9a-fA-F]{4,40}", ref):
            sha = self._verify(ref)
            if not sha and len(ref) == 40:   # a commit that is on no branch (e.g. a PR)
                with self._lock:
                    self.git("fetch", "--no-tags", url, ref, check=False)
                sha = self._verify(ref)
            if sha:
                return sha
        raise BuildError(f"{ref!r} is not a branch, tag or commit of {url}")

    def commit_info(self, sha: str) -> dict:
        out = self.git("log", "-1", "--format=%s%x00%an%x00%cI%x00%B", sha).stdout
        subject, author, date, body = (out.split("\x00") + ["", "", "", ""])[:4]
        return {"subject": subject.strip(), "author": author, "date": date, "body": body.strip()}

    def log(self, base: str, new: str, limit: int = 50) -> list[dict]:
        out = self.git("log", f"--max-count={limit}", "--format=%H%x00%s%x00%an",
                       f"{base}..{new}", check=False).stdout
        return [dict(zip(("sha", "subject", "author"), l.split("\x00")))
                for l in out.splitlines() if l]

    def diff(self, base: str, new: str, max_bytes: int = 2_000_000) -> tuple[str, str]:
        stat = self.git("diff", "--stat=120", base, new, check=False).stdout
        full = self.git("diff", base, new, check=False).stdout
        if len(full) > max_bytes:
            full = full[:max_bytes] + "\n... (diff truncated)\n"
        return stat, full

    def export(self, sha: str, dest: Path):
        data = self.git("archive", "--format=tar", sha, binary=True, timeout=300).stdout
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            try:
                tar.extractall(dest, filter="data")
            except TypeError:                      # Python < 3.12
                tar.extractall(dest)


def is_network(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == ZSTD_MAGIC
    except OSError:
        return False


# The default network is only in the hidden ref refs/assets/net of the
# repository (a commit holding only maskfish.nnue, published by
# scripts/push_net.sh): the releases no longer hold it as a separate file.
DEFAULT_NET_REPO = "https://github.com/nguyenhungitc/maskfish"
NET_REF = "refs/assets/net"


def net_repo(url: str | None) -> str:
    """Repository of the default network. A setting of the time the network was
    downloaded from the releases (a URL of a release file) gives the default."""
    url = (url or "").strip()
    if not url or "/releases/" in url or url.endswith(".nnue"):
        return DEFAULT_NET_REPO
    return normalize_repo(url)


def fetch_default_net(repo: Repo, nets_dir: Path, url: str | None = None,
                      replace: bool = False) -> Path:
    """Fetches maskfish.nnue of the ref refs/assets/net of the repository url (by
    default the one of Maskfish) as nets_dir/maskfish.nnue."""
    dest = Path(nets_dir) / NET_NAME
    if is_network(dest) and not replace:
        return dest
    url = net_repo(url)
    local = f"refs/ft/{Repo.key(url)}/assets/net"
    with repo._lock:
        repo.git("fetch", "--force", "--no-tags", url, f"+{NET_REF}:{local}")
    data = repo.git("cat-file", "blob", f"{local}:{NET_NAME}", binary=True).stdout
    Path(nets_dir).mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".download")
    tmp.write_bytes(data)
    if not is_network(tmp):
        tmp.unlink(missing_ok=True)
        raise BuildError(f"{NET_NAME} of {NET_REF} of {url} is not a network")
    os.replace(tmp, dest)
    return dest


def binary_name() -> str:
    return "maskfish.exe" if os.name == "nt" else "maskfish"


def build(repo: Repo, sha: str, make_args: str, arch: str, jobs: int, builds_dir: Path,
          nets_dir: Path, log) -> Path:
    """Compiles the commit (cached). log(text) receives the progress."""
    args = shlex.split(make_args or "")
    for a in args:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=[A-Za-z0-9_.+-]*", a):
            raise BuildError(f"bad make option {a!r}: expected NAME=value")
    key = sha[:12] + "-" + hashlib.sha1(f"{arch} {' '.join(args)}".encode()).hexdigest()[:8]
    out_dir = builds_dir / key
    binary = out_dir / binary_name()
    if binary.exists():
        log(f"Using the cached build of {sha[:10]} {make_args}\n")
        return binary

    tmp = builds_dir / f"tmp-{key}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        log(f"Exporting {sha[:10]}\n")
        repo.export(sha, tmp)
        src = tmp / "src"
        if not (src / "Makefile").exists():
            raise BuildError("no src/Makefile in this commit")
        # Give the build the default network: the exported tree is not a git
        # clone, so `make net` cannot fetch the ref refs/assets/net (it keeps
        # this copy, which the build embeds), and older commits would download
        # it from the releases, which no longer hold it
        cached_net = nets_dir / NET_NAME
        if is_network(cached_net):
            shutil.copy(cached_net, src / NET_NAME)
        date = repo.git("show", "-s", "--date=format:%Y%m%d", "--format=%cd", sha).stdout.strip()
        cmd = ["make", f"-j{jobs}", "build", f"ARCH={arch}", f"GIT_SHA={sha[:8]}",
               f"GIT_DATE={date}", *args]
        log(f"$ {' '.join(cmd)}\n")
        r = subprocess.run(cmd, cwd=src, capture_output=True, text=True, timeout=3600)
        output = (r.stdout + r.stderr).strip().splitlines()
        if r.returncode != 0 or not (src / binary_name()).exists():
            raise BuildError("build failed:\n" + "\n".join(output[-60:]))
        log("\n".join(output[-3:]) + "\nBuild OK\n")
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / binary_name(), binary)
        return binary
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def bench(binary: Path, net: Path | None, timeout: int = 600) -> dict:
    """Runs the benchmark: {'nodes': int, 'nps': int} (empty when it fails)."""
    cmds = (f"setoption name EvalFile value {net}\n" if net else "") + "bench\nquit\n"
    try:
        r = subprocess.run([str(binary)], input=cmds, capture_output=True, text=True,
                           timeout=timeout, cwd=binary.parent)
    except (subprocess.TimeoutExpired, OSError):
        return {}
    text = r.stdout + r.stderr
    nodes = re.search(r"Nodes searched\s*:\s*(\d+)", text)
    nps = re.search(r"Nodes/second\s*:\s*(\d+)", text)
    if not nodes:
        return {}
    return {"nodes": int(nodes.group(1)), "nps": int(nps.group(1)) if nps else 0}
