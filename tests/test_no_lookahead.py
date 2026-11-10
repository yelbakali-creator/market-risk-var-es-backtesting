"""No look-ahead test: a forecast made for day k may only use returns up to day k - 1."""
import numpy as np

from src.models import rolling_forecasts


def test_forecasts_use_only_past_data():
    rng = np.random.default_rng(0)
    r = rng.standard_t(5, 1700) * 0.01
    start, k = 1000, 1300
    base = rolling_forecasts(r, start)
    shocked = r.copy()
    shocked[k:] = shocked[k:] * 5 + 0.05          # change day k and every later day
    alt = rolling_forecasts(shocked, start)
    for model in base:
        # Forecasts for days start..k are made before day k is observed: they must not change
        assert np.allclose(base[model]["var"][start:k + 1], alt[model]["var"][start:k + 1])
        assert np.allclose(base[model]["es"][start:k + 1], alt[model]["es"][start:k + 1])
        # Forecasts after day k must react to the new data
        assert not np.allclose(base[model]["var"][k + 1:k + 5], alt[model]["var"][k + 1:k + 5])


if __name__ == "__main__":
    test_forecasts_use_only_past_data()
    print("No look-ahead: OK")
