"""Portfolio returns and one-day-ahead VaR / Expected Shortfall models.

Conventions: returns r_t; losses are -r_t; VaR and ES are reported as positive loss
fractions. VaR at 99% and ES at 97.5% (the Basel FRTB confidence level).
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import norm, t as student_t


# ---------------- Portfolio ----------------
def par_bond_price(coupon, maturity, y):
    """Price per 100 of a bond paying semi-annual coupons, at yield y (bond-equivalent)."""
    n = np.ceil(maturity * 2 - 1e-9)
    frac = maturity * 2 - (n - 1)                 # time to first coupon, in half-years
    k = np.arange(n)
    times = frac + k
    df = (1 + y / 2) ** (-times)
    return 100 * (coupon / 2 * df.sum() + df[-1])


def bond_returns(yields_10y, dt=1 / 252):
    """Daily total return of a constant-maturity 10-year par bond: buy at par, reprice one day later."""
    y = yields_10y.values
    out = np.full(len(y), np.nan)
    for i in range(1, len(y)):
        # Discounting the remaining cash flows gives the full (dirty) price, which already
        # includes the coupon accrued over the day: no separate accrual is added.
        price = par_bond_price(y[i - 1], 10 - dt, y[i])
        out[i] = price / 100 - 1
    return pd.Series(out, index=yields_10y.index)


# ---------------- GARCH(1,1) ----------------
def garch_filter(r, omega, alpha, beta, s2_0=None):
    s2 = np.empty(len(r) + 1)
    s2[0] = np.var(r) if s2_0 is None else s2_0
    for i in range(len(r)):
        s2[i + 1] = omega + alpha * r[i] ** 2 + beta * s2[i]
    return s2                                          # s2[i] is the variance for r[i]; s2[-1] is next day


def garch_t_negloglik(params, r):
    omega, alpha, beta, nu = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1 or nu <= 2.05:
        return 1e10
    s2 = garch_filter(r, omega, alpha, beta)[:-1]
    z2 = r ** 2 / s2
    # Student-t standardised to unit variance
    ll = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
          - 0.5 * np.log(s2) - (nu + 1) / 2 * np.log1p(z2 / (nu - 2)))
    return -ll.sum()


def fit_garch_t(r, start=None):
    """Maximum likelihood GARCH(1,1) with standardised Student-t innovations (zero mean)."""
    v = np.var(r)
    x0 = start if start is not None else [v * 0.05, 0.08, 0.90, 8.0]
    res = minimize(garch_t_negloglik, x0, args=(r,), method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-8, "maxiter": 20_000, "maxfev": 20_000})
    return res.x, -res.fun


def t_var_es(nu, alpha_var=0.99, alpha_es=0.975):
    """VaR and ES multipliers of a unit-variance Student-t distribution (positive loss)."""
    sc = np.sqrt((nu - 2) / nu)
    q_var = student_t.ppf(alpha_var, nu) * sc
    q = student_t.ppf(alpha_es, nu)
    es = sc * student_t.pdf(q, nu) * (nu + q ** 2) / ((nu - 1) * (1 - alpha_es))
    return q_var, es


# ---------------- Rolling forecasts ----------------
def rolling_forecasts(r, start, hs_window=500, refit_every=21, garch_window=1000,
                      lam=0.94, alpha_var=0.99, alpha_es=0.975):
    """One-day-ahead VaR (99%) and ES (97.5%) for four models, using only past data."""
    r = np.asarray(r)
    n = len(r)
    out = {m: {"var": np.full(n, np.nan), "var975": np.full(n, np.nan), "es": np.full(n, np.nan),
               "sigma": np.full(n, np.nan)}
           for m in ["Historical simulation", "EWMA normal", "GARCH-t", "Filtered historical simulation"]}
    z_var, z_es = norm.ppf(alpha_var), norm.pdf(norm.ppf(alpha_es)) / (1 - alpha_es)

    # EWMA variance (RiskMetrics), initialised on the first 250 days
    ew = np.empty(n)
    ew[0] = np.var(r[:250])
    for i in range(1, n):
        ew[i] = lam * ew[i - 1] + (1 - lam) * r[i - 1] ** 2

    params, s2_next, std_resid = None, None, None
    for i in range(start, n):
        past = r[i - hs_window:i]
        losses = -past
        v = np.quantile(losses, alpha_var)
        out["Historical simulation"]["var"][i] = v
        q975 = np.quantile(losses, alpha_es)
        out["Historical simulation"]["var975"][i] = q975
        out["Historical simulation"]["es"][i] = losses[losses >= q975].mean()
        out["Historical simulation"]["sigma"][i] = past.std(ddof=1)

        out["EWMA normal"]["var"][i] = z_var * np.sqrt(ew[i])
        out["EWMA normal"]["var975"][i] = norm.ppf(alpha_es) * np.sqrt(ew[i])
        out["EWMA normal"]["es"][i] = z_es * np.sqrt(ew[i])
        out["EWMA normal"]["sigma"][i] = np.sqrt(ew[i])

        if (i - start) % refit_every == 0:                      # refit GARCH-t on the trailing window
            win = r[i - garch_window:i]
            params, _ = fit_garch_t(win, params)
            s2 = garch_filter(win, *params[:3])
            std_resid = win / np.sqrt(s2[:-1])
            s2_next = s2[-1]
        else:
            s2_next = params[0] + params[1] * r[i - 1] ** 2 + params[2] * s2_prev
            std_resid = np.append(std_resid[1:], r[i - 1] / np.sqrt(s2_prev))
        s2_prev = s2_next
        sig = np.sqrt(s2_next)
        qv, qe = t_var_es(params[3], alpha_var, alpha_es)
        out["GARCH-t"]["var"][i] = sig * qv
        out["GARCH-t"]["es"][i] = sig * qe
        out["GARCH-t"]["var975"][i] = sig * student_t.ppf(alpha_es, params[3]) * np.sqrt((params[3] - 2) / params[3])
        out["GARCH-t"]["sigma"][i] = sig
        zl = -std_resid[-hs_window:]
        out["Filtered historical simulation"]["var"][i] = sig * np.quantile(zl, alpha_var)
        qz = np.quantile(zl, alpha_es)
        out["Filtered historical simulation"]["var975"][i] = sig * qz
        out["Filtered historical simulation"]["es"][i] = sig * zl[zl >= qz].mean()
        out["Filtered historical simulation"]["sigma"][i] = sig
    return out
