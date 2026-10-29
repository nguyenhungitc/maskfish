"""Settings that can be changed from the web page (Settings), stored in the
database. The command-line options give the defaults; a value saved on the
web page wins over them until it is reset.

Also the runner control (running / paused / pause after the current test)
and the password.
"""

from __future__ import annotations

from werkzeug.security import check_password_hash, generate_password_hash

# name: (type, minimum, maximum) -- None for no bound
EDITABLE = {
    "concurrency": (int, 1, 1024),
    "jobs": (int, 1, 1024),
    "arch": (str, None, None),
    "timemargin": (int, 0, 60000),
    "spsa": (bool, None, None),
    "repo": (str, None, None),
    "base_branch": (str, None, None),
    "net_url": (str, None, None),
}

RUNNER_MODES = ("running", "paused", "pause_after")


def saved(db) -> dict:
    return db.get_config("settings") or {}


def effective(db, base: dict) -> dict:
    """The command-line settings with the values saved on the web page."""
    s = dict(base)
    s.update({k: v for k, v in saved(db).items() if k in EDITABLE})
    s["password_set"] = bool(db.get_config("password_hash") or base.get("password"))
    return s


def save(db, values: dict):
    cur = saved(db)
    cur.update({k: v for k, v in values.items() if k in EDITABLE})
    db.set_config("settings", cur)


def reset(db):
    db.set_config("settings", {})


def parse(form, base: dict) -> tuple[dict, list[str]]:
    """Form values -> (settings, errors)."""
    out, errors = {}, []
    for key, (kind, lo, hi) in EDITABLE.items():
        if kind is bool:
            out[key] = form.get(key) in ("on", "1", "true")
            continue
        raw = (form.get(key) or "").strip()
        if kind is str:
            if not raw:
                errors.append(f"{key}: required")
            elif any(c.isspace() for c in raw) and key != "repo":
                errors.append(f"{key}: no spaces")
            out[key] = raw
            continue
        try:
            v = kind(raw)
        except ValueError:
            errors.append(f"{key}: not a number")
            continue
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            errors.append(f"{key}: must be between {lo} and {hi}")
        out[key] = v
    return out, errors


# ---------------------------------------------------------------- runner control

def runner_mode(db) -> str:
    m = db.get_config("runner_mode", "running")
    return m if m in RUNNER_MODES else "running"


def set_runner_mode(db, mode: str):
    if mode in RUNNER_MODES:
        db.set_config("runner_mode", mode)


# ---------------------------------------------------------------- password

def check_password(db, base: dict, password: str | None) -> bool:
    """True when no password is set, or when it matches."""
    h = db.get_config("password_hash")
    if h:
        return bool(password) and check_password_hash(h, password)
    if base.get("password"):
        return password == base["password"]
    return True


def password_required(db, base: dict) -> bool:
    return bool(db.get_config("password_hash") or base.get("password"))


def set_password(db, password: str):
    """An empty password removes the password set on the web page (the
    --password option, if any, applies again)."""
    if password:
        db.set_config("password_hash", generate_password_hash(password))
    else:
        db.execute("DELETE FROM config WHERE key = 'password_hash'")
