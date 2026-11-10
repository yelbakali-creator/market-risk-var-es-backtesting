"""Validation tests: bond pricing, Basel traffic-light cut-offs, Kupiec test and Student-t risk measures."""
import numpy as np
from scipy.stats import t as student_t

from src.backtest import kupiec_pof, traffic_light_cutoffs
from src.models import par_bond_price, t_var_es


def test_par_bond_prices_at_100_and_earns_carry():
    assert abs(par_bond_price(0.03, 10, 0.03) - 100) < 1e-9
    one_day = par_bond_price(0.03, 10 - 1 / 252, 0.03) / 100 - 1
    assert abs(one_day - 0.03 / 252) < 2e-6


def test_basel_traffic_light_cutoffs():
    cut = traffic_light_cutoffs()
    assert round(cut[4] * 100, 2) == 89.22 and round(cut[9] * 100, 2) == 99.97


def test_kupiec_does_not_reject_exact_coverage():
    _, p = kupiec_pof(70, 7000, 0.01)
    assert p > 0.99


def test_student_t_es_by_simulation():
    nu = 6
    _, es = t_var_es(nu, 0.99, 0.975)
    z = -student_t.rvs(nu, size=2_000_000, random_state=1) * np.sqrt((nu - 2) / nu)
    q = np.quantile(z, 0.975)
    assert abs(z[z >= q].mean() / es - 1) < 0.01
