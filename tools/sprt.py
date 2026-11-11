#!/usr/bin/env python3
"""Generalized SPRT on pentanomial pair results, standard library only.

match.py plays every deal twice with colours swapped, so the unit of a test is the pair. Its score is one of
0, 1/4, 1/2, 3/4, 1, and the five counts are the pentanomial. The test is the generalized SPRT (GSPRT):
under each hypothesis the pair distribution is the one closest to the observed frequencies (maximum likelihood)
among those that meet the hypothesis, and the log-likelihood ratio is

    LLR = N * sum_i p_i log(q1_i / q0_i)

with p the observed frequencies (an empty count gets a small weight) and N the number of pairs.

Two Elo models:
  logistic    the expected pair score is 1 / (1 + 10^(-elo / 400)); Elo bounds as in the earlier tests;
  normalized  (mu - 1/2) / sigma_game = nelo / (800 / ln 10) with sigma_game = sqrt(2) sigma_pair; the bounds
              then mean the same sample size at any draw or decisive rate.

Usage:
  python3 tools/sprt.py RUN/games.jsonl --engine NAME --elo0 0 --elo1 5 [--model normalized]
"""
import argparse
import json
import math

NELO_PER_NT = 800 / math.log(10)  # nElo per unit of (mu - 1/2) / sigma_game
SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)
EMPTY = 1e-3  # weight of an empty pentanomial count


def pentanomial(results):
    """results: (pair_index, a_is_red, score_for_a) per game -> counts of complete pairs by pair score."""
    pairs = {}
    for pi, _red, sc in results:
        pairs.setdefault(pi, []).append(sc)
    counts = [0] * 5
    for v in pairs.values():
        if len(v) == 2:
            counts[int(round((v[0] + v[1]) * 2))] += 1
    return counts


def _freq(counts):
    n = sum(counts)
    w = [c if c > 0 else EMPTY for c in counts]
    t = sum(w)
    return n, [x / t for x in w]


def _mean_var(q):
    mu = sum(qi * a for qi, a in zip(q, SCORES))
    return mu, sum(qi * (a - mu) ** 2 for qi, a in zip(q, SCORES))


def _solve(p, phi):
    """q_i = p_i / (1 + x phi_i) with E_q[phi] = 0: the closest distribution to p that meets the linear
    constraint. None when no distribution does (phi of one sign)."""
    if max(phi) <= 0 or min(phi) >= 0:
        return None
    lo, hi = -1 / max(phi), -1 / min(phi)  # 1 + x phi_i > 0 for every i
    # sum p_i phi_i / (1 + x phi_i) falls monotonically from +inf to -inf on (lo, hi)
    f = lambda x: sum(pi * f_ / (1 + x * f_) for pi, f_ in zip(p, phi))
    a, b = lo + (hi - lo) * 1e-12, hi - (hi - lo) * 1e-12
    for _ in range(200):
        m = (a + b) / 2
        if f(m) > 0:
            a = m
        else:
            b = m
    x = (a + b) / 2
    q = [pi / (1 + x * f_) for pi, f_ in zip(p, phi)]
    s = sum(q)
    return [qi / s for qi in q]


def mle_logistic(p, elo):
    s = 1 / (1 + 10 ** (-elo / 400))
    return _solve(p, [a - s for a in SCORES])


def mle_normalized(p, nelo):
    """Closest distribution with (mu - 1/2) / sigma_pair = t. The constraint is not linear in q; it is linearized
    at the current mu and sigma (phi below has E_q[phi] = mu - 1/2 - t sigma and the constraint's gradient up to a
    constant) and solved again until q stops moving."""
    t = nelo / NELO_PER_NT * math.sqrt(2)
    q = list(p)
    for _ in range(100):
        mu, var = _mean_var(q)
        sigma = math.sqrt(max(var, 1e-12))
        phi = [a - 0.5 - t * (sigma / 2 + (a - mu) ** 2 / (2 * sigma)) for a in SCORES]
        q2 = _solve(p, phi)
        if q2 is None:
            return None
        if max(abs(x - y) for x, y in zip(q, q2)) < 1e-12:
            return q2
        q = q2
    return q


def llr(counts, elo0, elo1, model="logistic"):
    n, p = _freq(counts)
    if n == 0:
        return 0.0
    mle = mle_normalized if model == "normalized" else mle_logistic
    q0, q1 = mle(p, elo0), mle(p, elo1)
    if q0 is None or q1 is None:
        return 0.0
    return n * sum(pi * math.log(b / a) for pi, a, b in zip(p, q0, q1))


def bounds(alpha=0.05, beta=0.05):
    return math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)


def elo_stats(counts):
    """Elo and nElo with 95 % intervals, and the likelihood of superiority, from the pentanomial."""
    n = sum(counts)
    if n < 2:
        return None
    p = [c / n for c in counts]
    mu, var = _mean_var(p)
    se = math.sqrt(var / n)
    elo = lambda s: -400 * math.log10(1 / min(max(s, 1e-9), 1 - 1e-9) - 1)
    sg = math.sqrt(2 * var) if var > 0 else 0.0
    nelo = (mu - 0.5) / sg * NELO_PER_NT if sg else 0.0
    nse = NELO_PER_NT / math.sqrt(2 * n)
    los = 0.5 * (1 + math.erf((mu - 0.5) / se / math.sqrt(2))) if se else (1.0 if mu > 0.5 else 0.0)
    return dict(pairs=n, score=mu, elo=elo(mu), elo_lo=elo(mu - 1.96 * se), elo_hi=elo(mu + 1.96 * se),
                nelo=nelo, nelo_lo=nelo - 1.96 * nse, nelo_hi=nelo + 1.96 * nse, los=los)


def state(counts, elo0, elo1, model="logistic", alpha=0.05, beta=0.05):
    """(llr, lower, upper, decision): decision 'H1', 'H0' or None."""
    v = llr(counts, elo0, elo1, model)
    lo, hi = bounds(alpha, beta)
    return v, lo, hi, "H1" if v >= hi else "H0" if v <= lo else None


def results_from_games(path, engine):
    out = []
    for line in open(path):
        g = json.loads(line)
        if engine not in (g["red"], g["black"]):
            continue
        red = g["red"] == engine
        out.append((g["pair"], red, g["result"] if red else 1 - g["result"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("games", help="games.jsonl of match.py")
    ap.add_argument("--engine", required=True, help="the engine under test (its name in the match)")
    ap.add_argument("--elo0", type=float, default=0.0)
    ap.add_argument("--elo1", type=float, default=5.0)
    ap.add_argument("--model", choices=("logistic", "normalized"), default="logistic")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    a = ap.parse_args()
    counts = pentanomial(results_from_games(a.games, a.engine))
    st = elo_stats(counts)
    v, lo, hi, dec = state(counts, a.elo0, a.elo1, a.model, a.alpha, a.beta)
    print(f"pentanomial {counts}")
    if st:
        print(f"Elo {st['elo']:+.1f} [{st['elo_lo']:+.1f}, {st['elo_hi']:+.1f}]  "
              f"nElo {st['nelo']:+.1f} [{st['nelo_lo']:+.1f}, {st['nelo_hi']:+.1f}]  LOS {100 * st['los']:.1f} %")
    print(f"LLR {v:.2f} ({lo:.2f}, {hi:.2f}) [{a.elo0:g}, {a.elo1:g}] {a.model}: {dec or 'continue'}")


if __name__ == "__main__":
    main()
