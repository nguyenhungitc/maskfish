"""Statistics of a test, as in fishtest: generalized SPRT on the pentanomial
results of game pairs, and Elo estimates.

The SPRT bounds are in normalized Elo (nElo, the default of fishtest) or in
logistic Elo. The log-likelihood ratio is the generalized one (GSPRT): for each
hypothesis, the pentanomial distribution closest to the observed one (maximum
likelihood) that satisfies the hypothesis is computed, and the LLR is the
difference of the log-likelihoods of the results under these two distributions.

    nElo = (mu - 1/2) / sigma_pg * 800 / ln(10)

where mu is the expected score of a game and sigma_pg the standard deviation of
the score per game (for game pairs, sigma_pg = sqrt(2) * sigma of the average
score of a pair).
"""

from __future__ import annotations

import math

NELO_DIVIDED_BY_NT = 800 / math.log(10)     # 347.43...
PAIR_SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)   # average score of a pair, per outcome
EPSILON = 1e-3                              # pseudo-count added to empty outcomes


def L(elo: float) -> float:
    return 1 / (1 + 10 ** (-elo / 400))


def elo_from_score(s: float) -> float:
    s = min(max(s, 1e-9), 1 - 1e-9)
    return -400 * math.log10(1 / s - 1)


def sprt_bounds(alpha: float = 0.05, beta: float = 0.05) -> tuple[float, float]:
    return math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)


def _pdf(penta):
    counts = [max(EPSILON, float(c)) for c in penta]
    n = sum(counts)
    return n, [c / n for c in counts]


def _mean_var(p):
    mu = sum(pi * a for pi, a in zip(p, PAIR_SCORES))
    var = sum(pi * (a - mu) ** 2 for pi, a in zip(p, PAIR_SCORES))
    return mu, var


def _solve(p, phi):
    """The distribution q_i = p_i / (1 + x phi_i) with sum q_i phi_i = 0, i.e.
    the maximum likelihood distribution under the linear constraint
    E_q[phi] = 0. None if the constraint cannot be met."""
    hi_phi, lo_phi = max(phi), min(phi)
    if hi_phi <= 0 or lo_phi >= 0:
        return None
    lo, hi = -1 / hi_phi, -1 / lo_phi       # 1 + x phi_i > 0 for every i
    f = lambda x: sum(pi * f_ / (1 + x * f_) for pi, f_ in zip(p, phi))  # decreasing
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    x = (lo + hi) / 2
    q = [pi / (1 + x * f_) for pi, f_ in zip(p, phi)]
    s = sum(q)
    return [qi / s for qi in q]


def mle_logistic(p, s):
    """Closest distribution to p with expected pair score s."""
    return _solve(p, [a - s for a in PAIR_SCORES])


def mle_normalized(p, t):
    """Closest distribution to p with (mu - 1/2) / sigma = t, where sigma is
    the standard deviation of the pair score. Solved by fixed-point
    iteration on the linearized constraint."""
    q = list(p)
    for _ in range(500):
        mu, var = _mean_var(q)
        sigma = math.sqrt(max(var, 1e-12))
        phi = [a - 0.5 - t * (sigma / 2 + (a - mu) ** 2 / (2 * sigma)) for a in PAIR_SCORES]
        nq = _solve(p, phi)
        if nq is None:
            return None
        if max(abs(x - y) for x, y in zip(q, nq)) < 1e-10:
            return nq
        q = nq
    return q


def llr(penta, elo0: float, elo1: float, model: str = "normalized") -> float:
    """Generalized log-likelihood ratio of H1 (elo1) against H0 (elo0)."""
    if sum(penta) == 0:
        return 0.0
    n, p = _pdf(penta)
    if model == "logistic":
        # The Elo of a game; a pair has the same expected score per game
        q0, q1 = mle_logistic(p, L(elo0)), mle_logistic(p, L(elo1))
    else:
        # t per game -> t of the pair average (sigma_pg = sqrt(2) sigma_pair)
        t0, t1 = (math.sqrt(2) * e / NELO_DIVIDED_BY_NT for e in (elo0, elo1))
        q0, q1 = mle_normalized(p, t0), mle_normalized(p, t1)
    if q0 is None or q1 is None:
        return 0.0
    return n * sum(pi * math.log(b / a) for pi, a, b in zip(p, q0, q1))


def elo_stats(penta) -> dict:
    """Elo, nElo with 95% intervals, and LOS, from the pentanomial results."""
    n = sum(penta)
    out = {"elo": 0.0, "elo_ci": 0.0, "nelo": 0.0, "nelo_ci": 0.0, "los": 0.5, "score": 0.5}
    if n == 0:
        return out
    p = [c / n for c in penta]
    mu, var = _mean_var(p)
    out["score"] = mu
    out["elo"] = elo_from_score(mu)
    if n < 2 or var <= 0:
        out["elo_ci"] = out["nelo_ci"] = float("inf")
        return out
    se = math.sqrt(var / n)
    out["elo_ci"] = (elo_from_score(mu + 1.96 * se) - elo_from_score(mu - 1.96 * se)) / 2
    sigma_pg = math.sqrt(2 * var)
    out["nelo"] = (mu - 0.5) / sigma_pg * NELO_DIVIDED_BY_NT
    out["nelo_ci"] = 1.96 * se / sigma_pg * NELO_DIVIDED_BY_NT
    out["los"] = 0.5 * (1 + math.erf((mu - 0.5) / se / math.sqrt(2)))
    return out


def sprt_state(penta, elo0, elo1, model="normalized", alpha=0.05, beta=0.05):
    """(llr, lower bound, upper bound, decision) with decision None, 'passed'
    or 'failed'."""
    v = llr(penta, elo0, elo1, model)
    lo, hi = sprt_bounds(alpha, beta)
    decision = "passed" if v >= hi else "failed" if v <= lo else None
    return v, lo, hi, decision


if __name__ == "__main__":
    # A few values to compare with fishtest
    for penta, b in (([1000, 10000, 20000, 10000, 1000], (0, 2)),
                     ([900, 10000, 20000, 10100, 1000], (0, 2)),
                     ([800, 9800, 20000, 10400, 1000], (0, 2)),
                     ([1200, 10200, 20000, 9800, 800], (-1.75, 0.25))):
        print(penta, b, f"LLR {llr(penta, *b):.3f}", f"logistic {llr(penta, *b, 'logistic'):.3f}",
              {k: round(v, 2) for k, v in elo_stats(penta).items()})
