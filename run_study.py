"""Market risk study: VaR and Expected Shortfall of a 60/40 portfolio, 1990-2022, with backtesting."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import (acerbi_szekely_z2, christoffersen, kupiec_pof, mcneil_frey, traffic_light,
                          traffic_light_cutoffs)
from src.models import bond_returns, fit_garch_t, garch_filter, rolling_forecasts

ROOT = Path(__file__).parent
TABLES, FIGS = ROOT / "results" / "tables", ROOT / "results" / "figures"
for folder in (TABLES, FIGS):
    folder.mkdir(parents=True, exist_ok=True)
NAVY, RED, GREY, GOLD, GREEN = "#1F3864", "#C0392B", "#7F8C8D", "#D4A017", "#2E7D32"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

W_EQUITY, START = 0.60, 1000
MODELS = ["Historical simulation", "EWMA normal", "GARCH-t", "Filtered historical simulation"]
COLORS = dict(zip(MODELS, [GREY, GOLD, NAVY, GREEN]))


def simulate_garch_t(n, omega, alpha, beta, nu, rng):
    z = rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu)
    r, s2 = np.empty(n), omega / (1 - alpha - beta)
    for i in range(n):
        r[i] = np.sqrt(s2) * z[i]
        s2 = omega + alpha * r[i] ** 2 + beta * s2
    return r


def main():
    # 1. Data and portfolio: 60% equal-weighted 20 US stocks, 40% constant-maturity 10-year Treasury
    stocks = pd.read_csv(ROOT / "data" / "sp500_20_stocks.csv.gz", parse_dates=["Date"], index_col="Date")
    ust = pd.read_csv(ROOT / "data" / "ust_raw.csv", parse_dates=["date"], index_col="date")
    y10 = ust["BC_10YEAR"].replace(0, np.nan).dropna() / 100
    idx = stocks.index.intersection(y10.index)
    stocks, y10 = stocks.loc[idx], y10.loc[idx]
    eq = stocks.pct_change().mean(axis=1)
    bd = bond_returns(y10)
    port = (W_EQUITY * eq + (1 - W_EQUITY) * bd).dropna()
    dates, r = port.index, port.values
    data_summary = pd.DataFrame({"series": ["Equities (equal-weighted)", "10y Treasury", "60/40 portfolio"],
                                 "annual_return": [eq.mean() * 252, bd.mean() * 252, port.mean() * 252],
                                 "annual_volatility": [eq.std() * np.sqrt(252), bd.std() * np.sqrt(252), port.std() * np.sqrt(252)],
                                 "worst_day": [eq.min(), bd.min(), port.min()],
                                 "correlation_with_equities": [1.0, eq.corr(bd), eq.corr(port)]})
    data_summary.to_csv(TABLES / "data_summary.csv", index=False)

    # 2. GARCH-t validation: (a) against the arch library on the full sample, (b) on simulated data
    from arch import arch_model
    mine, ll_mine = fit_garch_t(r)
    am = arch_model(r * 100, mean="Zero", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    ref = am.params
    garch_val = pd.DataFrame({
        "parameter": ["omega (x 1e4)", "alpha", "beta", "nu", "log-likelihood"],
        "this_project": [mine[0] * 1e4, mine[1], mine[2], mine[3], ll_mine],
        "arch_library": [ref["omega"], ref["alpha[1]"], ref["beta[1]"], ref["nu"], am.loglikelihood + len(r) * np.log(100)],
    })
    rng = np.random.default_rng(7)
    true = (2e-6, 0.08, 0.90, 6.0)
    sim = simulate_garch_t(20_000, *true, rng)
    est, _ = fit_garch_t(sim)
    sim_val = pd.DataFrame({"parameter": ["omega", "alpha", "beta", "nu"], "true": true, "estimated": est})
    garch_val.to_csv(TABLES / "garch_validation_vs_arch.csv", index=False)
    sim_val.to_csv(TABLES / "garch_validation_simulated.csv", index=False)

    # 3. Rolling one-day-ahead forecasts (only past data), from the 1,000th day on
    fc = rolling_forecasts(r, START)
    test = slice(START, len(r))
    rt, dt = r[test], dates[test]
    pd.DataFrame({"date": dt, "return": rt,
                  **{f"{m} | VaR99": fc[m]["var"][test] for m in MODELS},
                  **{f"{m} | ES97.5": fc[m]["es"][test] for m in MODELS}}).to_csv(TABLES / "forecasts.csv", index=False)

    # 4. Backtests
    rows, hits_all = [], {}
    n = len(rt)
    for m in MODELS:
        var, var975, es, sig = (fc[m][k][test] for k in ("var", "var975", "es", "sigma"))
        hits = (-rt > var).astype(int)
        hits_all[m] = hits
        lr, p_uc = kupiec_pof(hits.sum(), n, 0.01)
        cc = christoffersen(hits, 0.01)
        tl = traffic_light(hits)
        z2 = acerbi_szekely_z2(rt, var975, es, 0.025)
        mf_mean, mf_p, mf_n = mcneil_frey(rt, es, var975, sig)
        rows.append({"model": m, "days": n, "exceptions": hits.sum(), "expected": 0.01 * n,
                     "exception_rate": hits.mean(), "kupiec_p": p_uc, "christoffersen_ind_p": cc["p_ind"],
                     "christoffersen_cc_p": cc["p_cc"],
                     "p_exception_after_exception": cc["p_exception_after_exception"],
                     "traffic_green": tl["green"], "traffic_yellow": tl["yellow"], "traffic_red": tl["red"],
                     "max_exceptions_250d": tl["max_exceptions_in_window"],
                     "es_acerbi_szekely_z2": z2, "es_mcneil_frey_mean": mf_mean, "es_mcneil_frey_p": mf_p,
                     "average_var99": var.mean(), "average_es975": es.mean()})
    bt = pd.DataFrame(rows)
    bt.to_csv(TABLES / "backtest_results.csv", index=False)

    # 5. Crisis windows
    crises = {"Global financial crisis (Sep 2008 - Mar 2009)": ("2008-09-01", "2009-03-31"),
              "Covid crash (Feb - Apr 2020)": ("2020-02-15", "2020-04-30")}
    crisis_rows = []
    for cname, (a, b) in crises.items():
        mask = (dt >= a) & (dt <= b)
        for m in MODELS:
            crisis_rows.append({"period": cname, "model": m, "days": mask.sum(), "exceptions": hits_all[m][mask].sum(),
                                "expected": 0.01 * mask.sum()})
    crisis = pd.DataFrame(crisis_rows)
    crisis.to_csv(TABLES / "crisis_exceptions.csv", index=False)

    # 6. Stressed ES (Basel 2.5 / FRTB idea): historical ES on the 2008-2009 stress window
    stress_win = (dates >= "2008-09-01") & (dates <= "2009-08-31")
    sl = -r[stress_win]
    stressed_es = sl[sl >= np.quantile(sl, 0.975)].mean()
    recent_l = -r[-250:]
    current_es = recent_l[recent_l >= np.quantile(recent_l, 0.975)].mean()
    stressed = pd.DataFrame([{"stressed_es_2008_09": stressed_es, "current_es_last_250d": current_es,
                              "ratio": stressed_es / current_es}])
    stressed.to_csv(TABLES / "stressed_es.csv", index=False)
    cutoffs = traffic_light_cutoffs()

    # 7. Charts
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(dt, -rt * 100, color="#BBBBBB", lw=0.4, label="Daily loss")
    for m in ["Historical simulation", "GARCH-t"]:
        ax.plot(dt, fc[m]["var"][test] * 100, color=COLORS[m], lw=0.9, label=f"VaR 99%, {m}")
    h = hits_all["Historical simulation"].astype(bool)
    ax.scatter(dt[h], -rt[h] * 100, color=RED, s=6, zorder=3, label="Exception (historical simulation)")
    ax.set_ylim(bottom=0); ax.set_ylabel("Loss (% of portfolio)")
    ax.set_title("60/40 portfolio: daily losses vs one-day 99% VaR, 1994-2022")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(FIGS / "var_vs_losses.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, (cname, (a, b)) in zip(axes, crises.items()):
        mask = (dt >= pd.Timestamp(a) - pd.Timedelta(days=60)) & (dt <= b)
        ax.bar(dt[mask], -rt[mask] * 100, color="#BBBBBB", width=1.0, label="Daily loss")
        for m in MODELS:
            ax.plot(dt[mask], fc[m]["var"][test][mask] * 100, color=COLORS[m], lw=1.1, label=m)
        ax.set_title(cname, fontsize=9); ax.set_ylim(bottom=0); ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    axes[0].set_ylabel("Loss / VaR 99% (%)")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(FIGS / "crises_zoom.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.8))
    for m in MODELS:
        counts = np.convolve(hits_all[m], np.ones(250, int), "valid")
        ax.plot(dt[249:], counts, color=COLORS[m], lw=1, label=m)
    ax.axhspan(0, 4.5, color=GREEN, alpha=0.08); ax.axhspan(4.5, 9.5, color=GOLD, alpha=0.10)
    ax.axhspan(9.5, ax.get_ylim()[1] + 5, color=RED, alpha=0.08)
    ax.set_ylabel("Exceptions in last 250 days"); ax.set_title("Basel traffic light over time (green 0-4, yellow 5-9, red 10+)")
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout(); fig.savefig(FIGS / "traffic_light.png", dpi=150); plt.close(fig)

    # Console summary
    pd.set_option("display.width", 220)
    print(f"Portfolio days {len(r)} ({dates[0].date()} to {dates[-1].date()}); backtest days {n} from {dt[0].date()}")
    print("\nData summary\n" + data_summary.round(4).to_string(index=False))
    print("\nGARCH-t vs arch\n" + garch_val.round(5).to_string(index=False))
    print("\nGARCH-t on simulated data\n" + sim_val.round(5).to_string(index=False))
    print("\nBacktests\n" + bt.round(4).T.to_string())
    print("\nCrisis exceptions\n" + crisis.to_string(index=False))
    print("\nStressed ES\n" + stressed.round(4).to_string(index=False))
    print(f"\nTraffic light binomial cut-offs (250 days, p = 1%): P(X<=4) = {cutoffs[4]:.4f}, P(X<=9) = {cutoffs[9]:.5f}")


if __name__ == "__main__":
    main()
