"""Backtests of VaR and Expected Shortfall forecasts."""
import numpy as np
from scipy.stats import binom, chi2


def kupiec_pof(exceptions, n, p):
    """Kupiec (1995) proportion-of-failures likelihood ratio test of unconditional coverage."""
    x = exceptions
    phat = x / n
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll1 = (n - x) * np.log(1 - phat) + (x * np.log(phat) if x > 0 else 0.0)
    lr = -2 * (ll0 - ll1)
    return lr, chi2.sf(lr, 1)


def christoffersen(hits, p):
    """Christoffersen (1998) independence and conditional coverage tests (first-order Markov)."""
    h = np.asarray(hits, int)
    n00 = np.sum((h[:-1] == 0) & (h[1:] == 0)); n01 = np.sum((h[:-1] == 0) & (h[1:] == 1))
    n10 = np.sum((h[:-1] == 1) & (h[1:] == 0)); n11 = np.sum((h[:-1] == 1) & (h[1:] == 1))
    pi01 = n01 / (n00 + n01); pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def safe(a, b):
        return a * np.log(b) if a > 0 else 0.0
    ll_ind0 = safe(n00 + n10, 1 - pi) + safe(n01 + n11, pi)
    ll_ind1 = safe(n00, 1 - pi01) + safe(n01, pi01) + safe(n10, 1 - pi11) + safe(n11, pi11)
    lr_ind = -2 * (ll_ind0 - ll_ind1)
    lr_uc, _ = kupiec_pof(h.sum(), len(h), p)
    lr_cc = lr_uc + lr_ind
    return {"lr_ind": lr_ind, "p_ind": chi2.sf(lr_ind, 1), "lr_cc": lr_cc, "p_cc": chi2.sf(lr_cc, 2),
            "p_exception_after_exception": pi11, "p_exception_after_no_exception": pi01}


def traffic_light(hits, window=250):
    """Basel traffic light on rolling 250-day windows for 99% VaR: green 0-4, yellow 5-9, red 10+."""
    h = np.asarray(hits, int)
    counts = np.convolve(h, np.ones(window, int), "valid")
    return {"green": np.mean(counts <= 4), "yellow": np.mean((counts >= 5) & (counts <= 9)),
            "red": np.mean(counts >= 10), "max_exceptions_in_window": counts.max()}


def traffic_light_cutoffs(window=250, p=0.01):
    """Cumulative binomial probabilities behind the Basel (1996) zones for 250 days at 99%:
    the green zone ends at 4 exceptions (P(X <= 4) = 89.22%), the yellow zone at 9 (P(X <= 9) = 99.97%)."""
    return {k: binom.cdf(k, window, p) for k in (4, 9)}


def acerbi_szekely_z2(returns, var, es, alpha=0.025):
    """Acerbi-Szekely (2014) Test 2: Z2 = sum(X_t I_t / ES_t) / (T alpha) + 1, with X_t the P&L,
    I_t the exceedance of the VaR at the ES level, and VaR and ES positive. Expected value 0
    under a correct model; negative values mean ES is underestimated. Suggested critical value
    about -0.70 at 5% (Acerbi & Szekely, 2014), which can vary with the model."""
    x = np.asarray(returns)
    hit = x < -var
    return np.sum(x * hit / es) / (len(x) * alpha) + 1


def mcneil_frey(returns, es, var, sigma, n_boot=20_000, seed=0):
    """McNeil-Frey (2000) test: on days when the loss exceeds VaR, the scaled excess over ES
    should have mean zero. One-sided bootstrap p-value (H1: ES too low)."""
    loss = -np.asarray(returns)
    hit = loss > var
    e = (loss[hit] - es[hit]) / sigma[hit]
    if len(e) < 5:
        return np.nan, np.nan, len(e)
    rng = np.random.default_rng(seed)
    centred = e - e.mean()
    boot = rng.choice(centred, (n_boot, len(e)), replace=True).mean(axis=1)
    return e.mean(), np.mean(boot >= e.mean()), len(e)
