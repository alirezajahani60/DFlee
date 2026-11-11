"""Regression tests for the review of hecras2dflee: speed closure, missing values, gauge period."""
import pytest

# Optional dependencies of flee.preprocessing.hecras2dflee (pip install flee[hydro]).
pytest.importorskip("shapely")
pytest.importorskip("scipy")
pytest.importorskip("h5py")

from datetime import datetime as D, timedelta

import numpy as np

from flee.preprocessing.hecras2dflee.criteria import passable, speed_closed, speed_factor
from flee.preprocessing.hecras2dflee.rasread import clean_nonfinite
from flee.preprocessing.hecras2dflee import stage

PREG = {"type": "pregnolato2017"}          # closure at 300 mm
CRIT = {"type": "depth", "max_depth": 0.5}  # passable deeper than the speed function allows


def test_pregnolato_closure_closes_the_link():
    d = np.array([0.29, 0.30, 0.31, 0.6])
    assert speed_closed(d, PREG).tolist() == [False, False, True, True]
    g = speed_factor(d, PREG)
    assert g[0] > 0 and g[1] > 0          # passable up to and including the limit
    # the floor applies only to passable depths; above the closure the link is closed
    assert (speed_factor(d[:2], PREG) >= 0.1).all()


def test_link_states_respect_the_speed_closure():
    shapely = pytest.importorskip("shapely")
    from shapely.geometry import LineString
    import geopandas as gpd
    from flee.preprocessing.hecras2dflee.coupling import Domain, link_states
    from flee.preprocessing.hecras2dflee.rasread import MeshResults
    t0 = D(2020, 1, 1)
    times = [t0 + timedelta(hours=h) for h in range(3)]
    xs = np.arange(5.0, 100.0, 10.0)
    centres = np.c_[xs, np.full_like(xs, 5.0)]
    depth = np.array([[0.1] * len(xs), [0.4] * len(xs), [0.1] * len(xs)])   # 0.4 m: above 300 mm
    m = MeshResults("a", times, centres, depth, np.zeros_like(depth),
                    np.array([[0, 0], [100, 0], [100, 10], [0, 10]], float), None, np.zeros(len(xs)))
    links = gpd.GeoDataFrame(dict(name1=["A"], name2=["B"]), geometry=[LineString([(5, 5), (95, 5)])])
    rows, _ = link_states(Domain([m]), links, {"vehicle": {"criterion": CRIT, "speed": PREG}},
                          spacing=10.0, start=t0, step_hours=1.0, n_steps=3)
    assert [r["accessible"] for r in rows] == [1, 0, 1]
    assert rows[1]["speed_factor"] == 0.0
    assert passable(np.array([0.4]), np.array([0.0]), CRIT)[0]    # the criterion alone would allow it


def test_nonfinite_values_are_reported_or_rejected():
    d = np.array([[0.2, np.nan, 0.0]]); v = np.array([[0.5, 0.1, np.inf]])
    with pytest.warns(UserWarning, match="1 depth and 1 velocity"):
        dd, vv, counts = clean_nonfinite(d, v, "a")
    assert counts == dict(depth=1, velocity=1, values=3)
    assert dd.tolist() == [[0.2, 0.0, 0.0]] and vv.tolist() == [[0.5, 0.1, 0.0]]
    with pytest.raises(ValueError, match="NaN or infinite"):
        clean_nonfinite(d, v, "a", nonfinite="error")


def test_stage_compare_ignores_readings_outside_the_simulation():
    t0 = D(2019, 10, 15)
    times = [t0 + timedelta(hours=6 * i) for i in range(20)]       # 15 to 19 October
    ws = 100.0 + np.linspace(6.0, 8.0, 20)
    obs = [(D(2019, 10, 10), 5.0), (D(2019, 10, 17), 7.0), (D(2019, 10, 25), 9.0)]
    r = stage.compare(times, ws, obs, [7.3], datum=100.0)
    assert [x["time"] for x in r["readings"]] == ["2019-10-17T00:00:00"]
    assert r["readings_outside_period"] == ["2019-10-10T00:00:00", "2019-10-25T00:00:00"]
    with pytest.raises(ValueError, match="No observed reading"):
        stage.compare(times, ws, [(D(2019, 11, 1), 8.0)], [7.3], datum=100.0)
