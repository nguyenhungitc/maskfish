"""Git and build helpers: a local cache of the repositories used by the tests,
resolution of branches / tags / commits to SHAs, diffs, compiled engines cached
per (commit, CMake options, architecture), the default network and the
referee's library.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
from pathlib import Path

NET_NAME = "maskfish.nnue"       # the default network: maskfish.nnue of the hidden ref refs/assets/net
# Networks the engine reads (EvalFile): HNN nets, raw (.hnn) or compressed (.nnue)
NET_MAGICS = (b"JQHNNUE1", b"JQHNNL01", b"JQHNNZ01")
NET_SUFFIXES = (".hnn", ".nnue")


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


# Repositories git is allowed to fetch from: URLs of the usual transports,
# scp-like "user@host:path", and existing local directories. Anything else is
# refused before it reaches git: in particular a value starting with "-" would
# be read as an option of git (e.g. --upload-pack=<command> runs a command),
# and the "ext::" transport runs a command as well.
_URL_SCHEMES = ("https://", "http://", "ssh://", "git://", "file://")
_SCP_LIKE = re.compile(r"[\w.-]+@[\w.-]+:[\w./~-]+")


def check_repo(url: str) -> str:
    """Returns the (normalized) repository, or raises BuildError."""
    url = normalize_repo(url or "")
    if not url:
        raise BuildError("the repository is required")
    if url.startswith("-") or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in url):
        raise BuildError(f"invalid repository {url!r}")
    if url.startswith(_URL_SCHEMES):
        if not re.fullmatch(r"[a-z]+://[^\s'\"`;|&$<>\\]+", url):
            raise BuildError(f"invalid repository URL {url!r}")
        return url
    if "::" in url:
        raise BuildError(f"unsupported git transport in {url!r}")
    if _SCP_LIKE.fullmatch(url):
        return url
    if os.path.isdir(url):
        return url
    raise BuildError(f"{url!r} is not a URL (https://...), owner/repo or an existing local directory")


_ARCH = re.compile(r"[A-Za-z0-9_.+-]+")


def check_arch(arch: str) -> str:
    """ARCH of the builds: "native" (-march=native), "portable" (no -march), or
    a -march value of the compiler (x86-64-v3, haswell, armv8.2-a, ...)."""
    if not _ARCH.fullmatch(arch or ""):
        raise BuildError(f"invalid ARCH {arch!r}")
    return arch


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
        url = check_repo(url)
        k = self.key(url)
        with self._lock:
            self.git("fetch", "--force", "--prune", "--no-tags", "--end-of-options", url,
                     f"+refs/heads/*:refs/ft/{k}/heads/*",
                     f"+refs/tags/*:refs/ft/{k}/tags/*")

    def _verify(self, rev: str) -> str | None:
        r = self.git("rev-parse", "--verify", "--quiet", rev + "^{commit}", check=False)
        return r.stdout.strip() if r.returncode == 0 else None

    def resolve(self, url: str, ref: str) -> str:
        """Branch, tag or commit of the repository -> full SHA."""
        ref = ref.strip()
        if not ref or ref.startswith("-") or any(c.isspace() or ord(c) < 32 for c in ref):
            raise BuildError(f"invalid branch or commit {ref!r}")
        url = check_repo(url)
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
                    self.git("fetch", "--no-tags", "--end-of-options", url, ref, check=False)
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
            head = f.read(8)
    except OSError:
        return False
    return any(head.startswith(m) for m in NET_MAGICS)


# The default network is only in the hidden ref refs/assets/net of the
# repository (a commit holding only maskfish.nnue, published by
# scripts/push_net.sh).
DEFAULT_NET_REPO = "https://github.com/nguyenhungitc/maskfish"
NET_REF = "refs/assets/net"


def net_repo(url: str | None) -> str:
    """Repository of the default network. A setting of the time the network was
    downloaded from the releases (a URL of a release file) gives the default."""
    url = (url or "").strip()
    if not url or "/releases/" in url or url.endswith((".nnue", ".hnn")):
        return DEFAULT_NET_REPO
    return normalize_repo(url)


def fetch_default_net(repo: Repo, nets_dir: Path, url: str | None = None,
                      replace: bool = False) -> Path:
    """Fetches maskfish.nnue of the ref refs/assets/net of the repository url (by
    default the one of Maskfish) as nets_dir/maskfish.nnue."""
    dest = Path(nets_dir) / NET_NAME
    if is_network(dest) and not replace:
        return dest
    url = check_repo(net_repo(url))
    local = f"refs/ft/{Repo.key(url)}/assets/net"
    with repo._lock:
        repo.git("fetch", "--force", "--no-tags", "--end-of-options", url, f"+{NET_REF}:{local}")
    data = repo.git("cat-file", "blob", f"{local}:{NET_NAME}", binary=True).stdout
    Path(nets_dir).mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".download")
    tmp.write_bytes(data)
    if not is_network(tmp):
        tmp.unlink(missing_ok=True)
        raise BuildError(f"{NET_NAME} of {NET_REF} of {url} is not a network this engine reads (HNN)")
    os.replace(tmp, dest)
    return dest


def binary_name() -> str:
    return "maskfish.exe" if os.name == "nt" else "maskfish"


def cmake_args(arch: str, options: str) -> list[str]:
    """-D arguments of a build: the architecture, then the options of the test
    (NAME=value items, e.g. MASKFISH_LTO=ON), given to CMake as -DNAME=value."""
    arch = check_arch(arch)
    out = ["-DCMAKE_BUILD_TYPE=Release"]
    if arch == "native":
        out.append("-DMASKFISH_NATIVE=ON")
    elif arch == "portable":
        out.append("-DMASKFISH_NATIVE=OFF")
    else:
        out += ["-DMASKFISH_NATIVE=OFF", f"-DCMAKE_CXX_FLAGS=-march={arch}"]
    for a in shlex.split(options or ""):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=[A-Za-z0-9_.+-]*", a):
            raise BuildError(f"bad build option {a!r}: expected NAME=value")
        out.append("-D" + a)
    return out


def build(repo: Repo, sha: str, make_args: str, arch: str, jobs: int, builds_dir: Path,
          nets_dir: Path, log) -> Path:
    """Compiles the commit with CMake (cached). make_args: the test's build
    options, NAME=value items. log(text) receives the progress."""
    args = cmake_args(arch, make_args)
    key = sha[:12] + "-" + hashlib.sha1(" ".join(args).encode()).hexdigest()[:8]
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
        # The engine at the root, or in engine/ (the layout before the engine was moved to the root)
        src_root = next((d for d in (tmp, tmp / "engine")
                         if (d / "CMakeLists.txt").exists() and (d / "src").is_dir()), None)
        if src_root is None:
            raise BuildError("no CMakeLists.txt and src/ at the root or in engine/ of this commit")
        bdir = tmp / "build"
        for cmd in (["cmake", "-S", str(src_root), "-B", str(bdir), *args],
                    ["cmake", "--build", str(bdir), f"-j{jobs}", "--target", "maskfish"]):
            log(f"$ {' '.join(cmd)}\n")
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            output = (r.stdout + r.stderr).strip().splitlines()
            if r.returncode != 0:
                raise BuildError("build failed:\n" + "\n".join(output[-60:]))
        built = bdir / binary_name()
        if not built.exists():
            raise BuildError(f"build failed: no {binary_name()} in the build directory")
        log("\n".join(output[-3:]) + "\nBuild OK\n")
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built, binary)
        return binary
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def referee_library(data_dir: Path) -> Path:
    """The referee's shared library (tools/native of this checkout): the
    one named by $JIEQI_NATIVE_LIBRARY, else build/lib of the repository,
    else one compiled into data_dir/referee (once). Sets
    $JIEQI_NATIVE_LIBRARY, which the game processes inherit."""
    root = Path(__file__).resolve().parent.parent
    env = os.environ.get("JIEQI_NATIVE_LIBRARY")
    for cand in ([Path(env)] if env else []) + [root / "build/lib/libjieqi_rules.so",
                                                 root / "build/lib/libjieqi_rules.dylib"]:
        if cand.is_file():
            os.environ["JIEQI_NATIVE_LIBRARY"] = str(cand)
            return cand
    nat = root / "tools" / "native"
    out = Path(data_dir) / "referee" / ("libjieqi_rules.dylib" if sys.platform == "darwin" else "libjieqi_rules.so")
    srcs = [nat / f for f in ("c_api.cpp", "rules.cpp", "repetition.cpp", "chase.cpp")]
    newest = max(p.stat().st_mtime for p in srcs)
    if not out.is_file() or out.stat().st_mtime < newest:
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [os.environ.get("CXX", "g++"), "-std=c++20", "-O2", "-fPIC", "-shared", "-DJIEQI_BUILD_DLL",
               f"-I{nat / 'include'}", *map(str, srcs), "-o", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise BuildError("cannot build the referee library:\n" + (r.stdout + r.stderr)[-3000:])
    os.environ["JIEQI_NATIVE_LIBRARY"] = str(out)
    return out


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
