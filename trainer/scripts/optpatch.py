#!/usr/bin/env python3
"""Source transforms for autotune.sh: list the engine's UCI options and rewrite
the source so that an option becomes the default, gets a new default value, or
disappears from the code.

    optpatch.py list   --src SRC_DIR
    optpatch.py engine --src SRC_DIR --option NAME --action flip|set|remove [--value V] [--no-fold]
    optpatch.py trainer --file train.py --flag --no-factorize --action set|remove [--value V]

engine actions
  flip    bool option: default becomes the opposite (the option stays)
  set     default becomes --value (int / bool)
          (flip / set of a bool option whose callback sets a static member,
          X::member = bool(o): the member's initial value changes too)
  remove  the option disappears: registration + reading removed, the member
          variable becomes `static constexpr` with the CURRENT default, then
          (unless --no-fold) dead branches are folded where it is provably safe:
          if (X) / if (!X) [+ else], X ? a : b, `X && `, ` || X` ...
          Remaining uses keep compiling against the constexpr (the compiler
          removes the dead code).

The caller (autotune.sh) builds the patched engine and checks that `bench` gives
exactly the node count expected, so a wrong transform is never committed.

trainer actions (train.py argparse)
  set     the flag's default becomes --value (store_true flags: True)
  remove  the add_argument call is removed and every args.<dest> is replaced by
          the literal default
Only the standard library is used.
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys

# ------------------------------------------------------------------ C++ scanning helpers


def skip_ws_comments(s: str, i: int) -> int:
    n = len(s)
    while i < n:
        if s[i].isspace():
            i += 1
        elif s.startswith("//", i):
            j = s.find("\n", i)
            i = n if j < 0 else j + 1
        elif s.startswith("/*", i):
            j = s.find("*/", i + 2)
            i = n if j < 0 else j + 2
        else:
            break
    return i


def match_close(s: str, i: int) -> int:
    """s[i] is an opening bracket; returns the index of the matching closing one."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack = [pairs[s[i]]]
    j = i + 1
    n = len(s)
    while j < n:
        c = s[j]
        if s.startswith("//", j):
            k = s.find("\n", j)
            j = n if k < 0 else k
            continue
        if s.startswith("/*", j):
            k = s.find("*/", j + 2)
            j = n if k < 0 else k + 2
            continue
        if c in "\"'":
            k = j + 1
            while k < n and s[k] != c:
                k += 2 if s[k] == "\\" else 1
            j = k + 1
            continue
        if c in pairs:
            stack.append(pairs[c])
        elif c in ")]}":
            if not stack or stack[-1] != c:
                raise ValueError(f"unbalanced bracket at {j}")
            stack.pop()
            if not stack:
                return j
        j += 1
    raise ValueError("no matching bracket")


def statement_end(s: str, i: int) -> int:
    """Index just past the ';' that ends the statement starting at i (depth 0)."""
    n = len(s)
    j = i
    while j < n:
        c = s[j]
        if c in "([{":
            j = match_close(s, j) + 1
            continue
        if c in "\"'":
            k = j + 1
            while k < n and s[k] != c:
                k += 2 if s[k] == "\\" else 1
            j = k + 1
            continue
        if s.startswith("//", j):
            k = s.find("\n", j)
            j = n if k < 0 else k
            continue
        if c == ";":
            return j + 1
        j += 1
    raise ValueError("no end of statement")


def line_start(s: str, i: int) -> int:
    return s.rfind("\n", 0, i) + 1


def line_end(s: str, i: int) -> int:
    j = s.find("\n", i)
    return len(s) if j < 0 else j + 1


# ------------------------------------------------------------------ options in engine.cpp

ADD_RE = re.compile(r'options\.add\(\s*(?://[^\n]*\n\s*)?"([^"]+)"\s*,\s*Option\(')


def find_option_add(src: str, name: str):
    """(start, end, option_args_start, option_args_end) of the options.add statement."""
    for m in ADD_RE.finditer(src):
        if m.group(1) != name:
            continue
        start = m.start()
        paren = src.index("(", start)
        close = match_close(src, paren)
        end = statement_end(src, close)
        oparen = m.end() - 1
        oclose = match_close(src, oparen)
        return start, end, oparen + 1, oclose
    return None


def split_top_args(s: str):
    """Split a C++ argument list at depth-0 commas."""
    out, depth, cur, i = [], 0, [], 0
    while i < len(s):
        c = s[i]
        if c in "([{":
            j = match_close(s, i)
            cur.append(s[i:j + 1])
            i = j + 1
            continue
        if c == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
        i += 1
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def list_options(src_dir: str):
    src = open(os.path.join(src_dir, "engine.cpp"), encoding="utf-8").read()
    res = []
    for m in ADD_RE.finditer(src):
        name = m.group(1)
        oparen = m.end() - 1
        args = split_top_args(src[oparen + 1:match_close(src, oparen)])
        if not args:
            kind, default, lo, hi = "button", "", "", ""
        elif args[0] in ("true", "false"):
            kind, default, lo, hi = "bool", args[0], "", ""
        elif re.fullmatch(r"-?\d+", args[0]) and len(args) >= 3 and re.fullmatch(r"-?\d+", args[1]) \
                and re.fullmatch(r"-?\d+", args[2]):
            kind, default, lo, hi = "int", args[0], args[1], args[2]
        else:
            kind, default, lo, hi = "other", args[0].replace(" ", ""), "", ""
        res.append((name, kind, default, lo, hi))
    return res


def set_default(src: str, name: str, value: str) -> str:
    f = find_option_add(src, name)
    if not f:
        raise SystemExit(f"option {name} not found in engine.cpp")
    _, _, a0, a1 = f
    args_txt = src[a0:a1]
    first = split_top_args(args_txt)[0]
    k = args_txt.index(first)
    new_args = args_txt[:k] + value + args_txt[k + len(first):]
    return src[:a0] + new_args + src[a1:]


def sync_callback_statics(src_dir: str, src: str, name: str, value: str) -> list:
    """A bool option whose callback sets a static member (X::member = bool(o)):
    the callback only runs when the option is set, so the member keeps its own
    initial value. Gives the declaration `static inline bool member = ...;` the
    new default too. Returns the names of the files changed."""
    f = find_option_add(src, name)
    if not f:
        return []
    members = re.findall(r"(?:\w+::)*(\w+)\s*=\s*bool\(\s*o\s*\)", src[f[2]:f[3]])
    changed = []
    for member in members:
        decl = re.compile(r"(static\s+inline\s+bool\s+%s\s*=\s*)(?:true|false)(\s*;)" % re.escape(member))
        hits = []
        for fn in sorted(os.listdir(src_dir)):
            if fn.endswith((".cpp", ".h")):
                p = os.path.join(src_dir, fn)
                if decl.search(open(p, encoding="utf-8").read()):
                    hits.append(p)
        for sub in ("nnue", os.path.join("nnue", "features")):
            d = os.path.join(src_dir, sub)
            for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
                p = os.path.join(d, fn)
                if fn.endswith((".cpp", ".h")) and decl.search(open(p, encoding="utf-8").read()):
                    hits.append(p)
        if len(hits) != 1:
            raise SystemExit(f"static member {member} of option {name}: {len(hits)} declarations")
        s = open(hits[0], encoding="utf-8").read()
        s = decl.sub(lambda m: m.group(1) + value + m.group(2), s, count=1)
        open(hits[0], "w", encoding="utf-8").write(s)
        changed.append(os.path.relpath(hits[0], src_dir))
    return changed


KEEP_COMMENT = re.compile(r"Experimental|off by default|test\s+with SPRT|match\.py|ideas? (from|ported)", re.I)


def remove_option_add(src: str, name: str) -> str:
    f = find_option_add(src, name)
    if not f:
        raise SystemExit(f"option {name} not found in engine.cpp")
    start, end, _, _ = f
    ls = line_start(src, start)
    le = line_end(src, end)
    # the comment lines right above describe this option, unless they introduce a group
    cstart = ls
    while cstart > 0:
        prev = line_start(src, cstart - 1)
        line = src[prev:cstart].strip()
        if line.startswith("//") and not KEEP_COMMENT.search(line):
            cstart = prev
        else:
            break
    return src[:cstart] + src[le:]


# ------------------------------------------------------------------ remove: member -> constexpr, fold

def literal(kind: str, value: str) -> str:
    if kind == "bool":
        return "true" if value in ("true", "1") else "false"
    return value


def remove_option_engine(src_dir: str, name: str, fold: bool):
    files = {}
    for fn in sorted(os.listdir(src_dir)):
        if fn.endswith((".cpp", ".h")):
            p = os.path.join(src_dir, fn)
            files[p] = open(p, encoding="utf-8").read()
    eng = os.path.join(src_dir, "engine.cpp")
    opts = {o[0]: o for o in list_options(src_dir)}
    if name not in opts:
        raise SystemExit(f"option {name} not found")
    _, kind, default, _, _ = opts[name]
    if kind not in ("bool", "int"):
        raise SystemExit(f"option {name} has kind {kind}: cannot remove automatically")
    lit = literal(kind, default)
    files[eng] = remove_option_add(files[eng], name)

    # IDENT = bool(options["NAME"]);  -> member variable
    assign = re.compile(r'^[ \t]*(\w+)\s*=\s*(?:bool|int)\(\s*(?:[\w.]+\.)?options\["%s"\]\s*\)\s*;[ \t]*\n'
                        % re.escape(name), re.M)
    idents = []
    for p, s in list(files.items()):
        for m in list(assign.finditer(s)):
            idents.append(m.group(1))
        files[p] = assign.sub("", s)
    # every other read becomes the literal
    read = re.compile(r'(?:bool|int)\(\s*(?:[\w.]+\.)?options\["%s"\]\s*\)' % re.escape(name))
    for p, s in files.items():
        files[p] = read.sub(lit, s)
    ctype = "bool" if kind == "bool" else "int"
    for ident in idents:
        decl = re.compile(r'^([ \t]*)(?:bool|int|double)(\s+)%s(\s*)(?:=\s*[^;]+)?;' % re.escape(ident), re.M)
        n = 0
        for p, s in files.items():
            s2, k = decl.subn(lambda m: f"{m.group(1)}static constexpr {ctype}{m.group(2)}{ident}{m.group(3)}= {lit};", s)
            files[p] = s2
            n += k
        if n != 1:
            raise SystemExit(f"declaration of member {ident} found {n} times (expected 1)")
    if fold and kind == "bool":
        for ident in idents:
            for p in files:
                files[p] = fold_bool(files[p], ident, lit == "true")
            # no use left: drop the constexpr declaration
            uses = sum(len(re.findall(r"\b%s\b" % re.escape(ident), mask(s)[0])) for s in files.values())
            if uses == 1:
                d = re.compile(r'^[ \t]*static constexpr bool\s+%s\s*= (?:true|false);[ \t]*\n' % re.escape(ident), re.M)
                for p in files:
                    files[p] = d.sub("", files[p])
    changed = []
    for p, s in files.items():
        old = open(p, encoding="utf-8").read()
        if s != old:
            open(p, "w", encoding="utf-8").write(s)
            changed.append(os.path.basename(p))
    print(f"removed {name} ({kind}, hard-coded {lit}); members {idents or '-'}; changed {changed}")


ATOM = r"(?:\b\w+(?:\.|->))*%s\b"
# where a boolean operand may start: after '(' (not a cast/call check needed: the
# operand is the whole parenthesized expression's first term), '&&', '||', ',',
# '?', ':', or an assignment '=' (not ==, !=, <=, >=)
LB_BASE = r"(?<=\()|(?<=\|\|)|(?<=,)|(?<=\?)|(?<=[^:]:)|(?<=[^=!<>]=)|(?<=\breturn)"
LB = r"(?:%s|(?<=&&))" % LB_BASE
LB_NO_AND = r"(?:%s)" % LB_BASE        # 'x && LIT || E': LIT belongs to the && term


def mask(s: str):
    """Replaces comments and string / char literals by placeholders, so that
    the folding only touches code. Returns (masked, restore list)."""
    out, saved, i, n = [], [], 0, len(s)
    while i < n:
        if s.startswith("//", i):
            j = s.find("\n", i)
            j = n if j < 0 else j
        elif s.startswith("/*", i):
            j = s.find("*/", i + 2)
            j = n if j < 0 else j + 2
        elif s[i] in "\"'" and not (s[i] == "'" and i > 0 and s[i - 1].isalnum()):   # 1'000 digit separators
            q = s[i]
            j = i + 1
            while j < n and s[j] != q:
                j += 2 if s[j] == "\\" else 1
            j += 1
        else:
            out.append(s[i])
            i += 1
            continue
        saved.append(s[i:j])
        out.append(f"\x01{len(saved) - 1}\x02")
        i = j
    return "".join(out), saved


def unmask(s: str, saved) -> str:
    return re.sub(r"\x01(\d+)\x02", lambda m: saved[int(m.group(1))], s)


def operand_chain_end(s: str, i: int, stop_ops) -> int:
    """From i, skip operands joined by anything but the operators in stop_ops,
    up to a depth-0 ')', ';', ',', '?', ':' or one of stop_ops."""
    n = len(s)
    j = i
    while j < n:
        c = s[j]
        if c in "([{":
            j = match_close(s, j) + 1
            continue
        if c in ")]};,?" or (c == ":" and s[j + 1:j + 2] != ":" and s[j - 1] != ":"):
            return j
        if any(s.startswith(op, j) for op in stop_ops):
            return j
        j += 1
    return j


def fold_bool(s: str, ident: str, value: bool) -> str:
    """Safe boolean simplifications for a compile-time constant member
    (comments and strings are left untouched)."""
    s, saved = mask(s)
    s = fold_bool_masked(s, ident, value)
    s = fold_local_consts(s)
    s = tidy_arith(s)
    return unmask(s, saved)


def fold_bool_masked(s: str, ident: str, value: bool) -> str:
    atom = ATOM % re.escape(ident)
    tv, fv = ("true", "false") if value else ("false", "true")
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"!\s*(%s)(?!\s*(?:\.|->|\(|\[))" % atom, fv, s)
        s = re.sub(atom + r"(?=\s*(?:[)&|;,?]|:(?!:)))", tv, s)
        # identities, valid whatever the folded value: true && E, E && true,
        # false || E, E || false -> E ; then absorption: false && E, true || E
        s = re.sub(LB + r"(\s*)true\s*&&\s*", r"\1", s)
        s = re.sub(r"\s*&&\s*true(?=\s*(?:[)&|;,?]|:(?!:)))", "", s)
        s = re.sub(LB_NO_AND + r"(\s*)false\s*\|\|\s*", r"\1", s)
        s = re.sub(r"\s*\|\|\s*false(?=\s*(?:[)&|;,?]|:(?!:)))", "", s)
        for absorb, pat, stop in (("false", re.compile(LB + r"(\s*)false\s*&&"), ("||",)),
                                  ("true", re.compile(LB_NO_AND + r"(\s*)true\s*\|\|"), ())):
            pos = 0
            while True:
                m = pat.search(s, pos)
                if not m:
                    break
                e = operand_chain_end(s, m.end(), stop)
                s = s[:m.start()] + m.group(1) + absorb + s[e:]
                pos = m.start() + 1
        s = re.sub(r"(\b\w+\s*)?(?<![\]>)])\(\s*(true|false)\s*\)(?=\s*(?:[)&|;,?]|:(?!:)))",
                   lambda m: m.group(0) if m.group(1) else m.group(2), s)
        s = fold_ternaries(s)
        s = fold_ifs(s)
    return s


def fold_ternaries(s: str) -> str:
    """(true ? A : B) / (false ? A : B) where the literal directly follows '(' or '='."""
    out = s
    for lit in ("true", "false"):
        pos = 0
        while True:
            m = re.compile(r"(?:(?<=\()|(?<=,)|(?<=[^=!<>]=)|(?<=\breturn)|(?<=\?)|(?<=[^:]:))(\s*)%s\s*\?" % lit).search(out, pos)
            if not m:
                break
            a_start = m.end()
            # find ':' at depth 0 (nested ternaries counted), then the end of B
            j, depth_q, n = a_start, 0, len(out)
            colon = None
            while j < n:
                c = out[j]
                if c in "([{":
                    j = match_close(out, j) + 1
                    continue
                if c in ")]};,":
                    break
                if c == "?":
                    depth_q += 1
                elif c == ":" and not out.startswith("::", j) and out[j - 1] != ":":
                    if depth_q == 0:
                        colon = j
                        break
                    depth_q -= 1
                j += 1
            if colon is None:
                pos = m.end()
                continue
            j = colon + 1
            while j < n:
                c = out[j]
                if c in "([{":
                    j = match_close(out, j) + 1
                    continue
                if c in ")]};,":
                    break
                if c == "?":       # a ternary inside B without parentheses: give up
                    j = -1
                    break
                j += 1
            if j < 0:
                pos = m.end()
                continue
            a = out[a_start:colon].strip()
            b = out[colon + 1:j].strip()
            keep = a if lit == "true" else b
            out = out[:m.start()] + m.group(1) + keep + out[j:]
            pos = m.start()
    return out


def stmt_end(s: str, i: int) -> int:
    """End (exclusive) of the statement starting at i: blocks, if/else chains,
    for/while loops and simple statements."""
    i = skip_ws_comments(s, i)
    if s[i] == "{":
        return match_close(s, i) + 1
    m = re.match(r"(if(?:\s+constexpr)?|for|while|switch)\b\s*\(", s[i:])
    if m:
        paren = i + m.end() - 1
        j = stmt_end(s, match_close(s, paren) + 1)
        if m.group(1).startswith("if"):
            k = skip_ws_comments(s, j)
            if re.match(r"else\b", s[k:]):
                j = stmt_end(s, k + 4)
        return j
    return statement_end(s, i)


def fold_ifs(s: str) -> str:
    """if (true) S [else T] -> S ; if (false) S [else T] -> T or nothing.
    T may itself be an if/else chain."""
    pos = 0
    pat = re.compile(r"\bif\s*\(\s*(true|false)\s*\)")
    while True:
        m = pat.search(s, pos)
        if not m:
            return s
        body_start = skip_ws_comments(s, m.end())
        body_end = stmt_end(s, body_start)
        k = skip_ws_comments(s, body_end)
        else_body = None
        end = body_end
        if re.match(r"else\b", s[k:]):
            eb = skip_ws_comments(s, k + 4)
            end = stmt_end(s, eb)
            else_body = (eb, end)
        taken = m.group(1) == "true"
        before = s[:m.start()].rstrip()
        is_else_branch = re.search(r"\belse$", before) is not None
        if taken:
            repl = s[body_start:body_end]
        elif else_body:
            repl = s[else_body[0]:else_body[1]]
        else:
            repl = ""
        if repl == "" and is_else_branch:
            cut = before.rfind("else")
            s = s[:cut].rstrip() + s[end:]
            pos = cut
            continue
        if repl == "":
            ls = line_start(s, m.start())
            if s[ls:m.start()].strip() == "" and s[end:line_end(s, end)].strip() == "":
                s = s[:ls] + s[line_end(s, end):]
                pos = ls
            else:
                s = s[:m.start()] + ";" + s[end:]
                pos = m.start()
            continue
        s = s[:m.start()] + repl + s[end:]
        pos = m.start()


def fold_local_consts(s: str) -> str:
    """const bool X = true|false;  -> fold X in the rest of its block, drop it if unused."""
    pat = re.compile(r"^[ \t]*const\s+bool\s+(\w+)\s*=\s*(true|false)\s*;[ \t]*\n", re.M)
    pos = 0
    while True:
        m = pat.search(s, pos)
        if not m:
            return s
        # end of the enclosing block: first unmatched '}'
        j, n = m.end(), len(s)
        while j < n:
            if s[j] in "([{":
                j = match_close(s, j) + 1
                continue
            if s[j] == "}":
                break
            j += 1
        region = fold_bool_masked(s[m.end():j], m.group(1), m.group(2) == "true")
        if re.search(r"\b%s\b" % re.escape(m.group(1)), region):
            s = s[:m.end()] + region + s[j:]
            pos = m.end()
        else:
            s = s[:m.start()] + region + s[j:]
            pos = m.start()


def tidy_arith(s: str) -> str:
    """Cosmetic: (0) / (literal) parentheses, '+ 0' / '- 50 * false' terms."""
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"(\b\w+\s*)?(?<![\]>)])\(\s*(-?\d+|true|false)\s*\)",
                   lambda m: m.group(0) if m.group(1) else m.group(2), s)
        # additive zero terms: operand end before, no * / % after
        s = re.sub(r"(?<=[\w)\]])(\s*)[+-]\s*(?:\d+\s*\*\s*)?(?:0|false)\b(?!\s*[*/%.(\[])", "", s)
    return s


# ------------------------------------------------------------------ trainer (train.py argparse)

def find_add_argument(src: str, flag: str):
    for m in re.finditer(r"ap\.add_argument\(", src):
        close = match_close(src, m.end() - 1)
        call = src[m.start():close + 1]
        if re.search(r'["\']%s["\']' % re.escape(flag), call):
            return m.start(), close + 1, call
    raise SystemExit(f"flag {flag} not found")


def flag_info(call: str, flag: str):
    tree = ast.parse(call.replace("ap.add_argument", "f", 1))
    c = tree.body[0].value
    kw = {k.arg: k.value for k in c.keywords}
    dest = ast.literal_eval(kw["dest"]) if "dest" in kw else flag.lstrip("-").replace("-", "_")
    action = ast.literal_eval(kw["action"]) if "action" in kw and isinstance(kw["action"], ast.Constant) else None
    if "default" in kw:
        default = ast.literal_eval(kw["default"])
    else:
        default = False if action == "store_true" else None
    return dest, action, default


def _const_expr(node) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp):
        return _const_expr(node.operand)
    if isinstance(node, ast.BoolOp):
        return all(_const_expr(v) for v in node.values)
    if isinstance(node, ast.Compare):
        return _const_expr(node.left) and all(_const_expr(c) for c in node.comparators)
    return False


def fold_python_ifs(src: str) -> str:
    """`if <constant expression>:` blocks without elif/else: drop a false one,
    dedent a true one (text edit, comments are kept)."""
    while True:
        tree = ast.parse(src)
        target = None
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and not node.orelse and _const_expr(node.test):
                target = node
                break
        if target is None:
            return src
        lines = src.splitlines(keepends=True)
        first, last = target.lineno - 1, target.end_lineno      # if-line .. last body line
        if eval(compile(ast.Expression(target.test), "<c>", "eval"), {"__builtins__": {}}):
            ind = len(lines[first]) - len(lines[first].lstrip())
            body_ind = len(lines[first + 1]) - len(lines[first + 1].lstrip())
            shift = body_ind - ind
            body = [l[shift:] if l.strip() else l for l in lines[first + 1:last]]
            lines[first:last] = body
        else:
            del lines[first:last]
        src = "".join(lines)


def trainer_patch(path: str, flag: str, action: str, value: str | None):
    src = open(path, encoding="utf-8").read()
    start, end, call = find_add_argument(src, flag)
    dest, act, default = flag_info(call, flag)
    if action == "set":
        if act == "store_true":
            new = call.replace('action="store_true"', 'action="store_true", default=True', 1)
        else:
            v = ast.literal_eval(value)
            if re.search(r"default\s*=", call):
                new = re.sub(r"default\s*=\s*[^,)]+", f"default={v!r}", call, count=1)
            else:
                new = call[:-1] + f", default={v!r})"
        src = src[:start] + new + src[end:]
        print(f"{flag}: default {default!r} -> {True if act == 'store_true' else value}")
    elif action == "remove":
        ls, le = line_start(src, start), line_end(src, end)
        src = src[:ls] + src[le:]
        n = len(re.findall(r"\bargs\.%s\b" % re.escape(dest), src))
        src = re.sub(r"\bargs\.%s\b" % re.escape(dest), repr(default), src)
        print(f"{flag}: removed, args.{dest} -> {default!r} ({n} uses)")
        src = re.sub(r"\bnot (True|False)\b", lambda m: "False" if m.group(1) == "True" else "True", src)
        src = fold_python_ifs(src)
    ast.parse(src)
    open(path, "w", encoding="utf-8").write(src)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--src", required=True)
    p = sub.add_parser("engine")
    p.add_argument("--src", required=True)
    p.add_argument("--option", required=True)
    p.add_argument("--action", choices=("flip", "set", "remove"), required=True)
    p.add_argument("--value")
    p.add_argument("--no-fold", action="store_true")
    p = sub.add_parser("trainer")
    p.add_argument("--file", required=True)
    p.add_argument("--flag", required=True)
    p.add_argument("--action", choices=("set", "remove"), required=True)
    p.add_argument("--value")
    a = ap.parse_args()

    if a.cmd == "list":
        for o in list_options(a.src):
            print("\t".join(o))
        return
    if a.cmd == "trainer":
        trainer_patch(a.file, a.flag, a.action, a.value)
        return
    eng = os.path.join(a.src, "engine.cpp")
    if a.action == "remove":
        remove_option_engine(a.src, a.option, not a.no_fold)
        return
    opts = {o[0]: o for o in list_options(a.src)}
    if a.option not in opts:
        raise SystemExit(f"option {a.option} not found")
    _, kind, default, lo, hi = opts[a.option]
    if a.action == "flip":
        if kind != "bool":
            raise SystemExit("flip needs a bool option")
        value = "false" if default == "true" else "true"
    else:
        value = literal(kind, a.value)
        if kind == "int" and not (int(lo) <= int(value) <= int(hi)):
            raise SystemExit(f"{value} outside [{lo}, {hi}]")
    src = open(eng, encoding="utf-8").read()
    s = set_default(src, a.option, value)
    open(eng, "w", encoding="utf-8").write(s)
    synced = sync_callback_statics(a.src, src, a.option, value) if kind == "bool" else []
    print(f"{a.option}: default {default} -> {value}"
          + (f" (static members in {', '.join(synced)})" if synced else ""))


if __name__ == "__main__":
    sys.exit(main())
