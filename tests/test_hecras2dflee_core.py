"""Unit tests for hecras2dflee (no HEC-RAS files needed)."""
import pytest

# Optional dependencies of flee.preprocessing.hecras2dflee (pip install flee[hydro]).
pytest.importorskip("shapely")
pytest.importorskip("scipy")

from datetime import datetime, timedelta

import numpy as np
import pytest

from flee.preprocessing.hecras2dflee.criteria import passable, speed_factor, xia_critical_depth, xia_incipient_velocity
from flee.preprocessing.hecras2dflee.coupling import _aggregate, closures_from_series, step_index


def test_depth_criterion_and_cap():
    d = np.array([0.0, 0.29, 0.31])
    assert passable(d, np.zeros(3), {"type": "depth", "max_depth": 0.3}).tolist() == [True, True, False]
    hr = {"type": "hazard_rating", "max_hr": 1.25, "max_depth": 1.0}
    # deep still water: HR = 1.2 * 0.5 = 0.6 < 1.25 but capped by max_depth
    assert not passable(np.array([1.2]), np.array([0.0]), hr)[0]
    # shallow fast water: HR = 0.5 * (3 + 0.5) = 1.75 >= 1.25
    assert not passable(np.array([0.5]), np.array([3.0]), hr)[0]
    assert passable(np.array([0.2]), np.array([0.5]), hr)[0]


def test_xia2014_published_values():
    # Xia et al. (2014) Table 2 (real subjects), adult 1.7 m / 65 kg
    p = {"calibration": "real"}
    uc = xia_incipient_velocity(np.array([0.4, 1.0]), p)
    assert uc[0] > uc[1] > 0                        # critical velocity falls with depth
    dc = xia_critical_depth(1.0, p)
    assert abs(xia_incipient_velocity(dc, p) - 1.0) < 1e-3
    assert 1.0 < dc < 1.3                           # about 1.17 m at 1 m/s
    # the flume-model calibration (Table 1) is more conservative
    assert xia_critical_depth(1.0, {"calibration": "model"}) < dc
    # a child is unstable in shallower water
    assert xia_critical_depth(1.0, {"h_p": 1.3, "m_p": 30.0}) < dc
    ok = passable(np.array([0.3, 1.4]), np.array([1.0, 1.0]), {"type": "xia2014"})
    assert ok.tolist() == [True, False]


def test_speed_factor_floor_and_pregnolato():
    g = speed_factor(np.array([0.0, 2.0]), {"type": "linear", "zero_speed_depth": 1.0})
    assert g.tolist() == [1.0, 0.1]
    p = speed_factor(np.array([0.0, 0.1, 0.35]), {"type": "pregnolato2017", "min_factor": 0.0})
    assert p[0] == 1.0 and 0 < p[1] < 1 and p[2] == 0.0


def test_dias2021_walking_speed():
    sf = {"type": "dias2021", "min_factor": 0.0}
    g = speed_factor(np.array([0.0, 0.08, 0.43, 0.90]), sf)
    # Dias et al. (2021): mean 1.18, 0.92, 0.70 m/s at 0.08, 0.43, 0.90 m -> ratios 1, 0.78, 0.59
    assert g[0] == 1.0 and abs(g[1] - 1.0) < 1e-9
    assert 0.65 < g[2] < 0.80 and 0.55 < g[3] < 0.65
    # flowing water slows walkers further
    assert speed_factor(np.array([0.43]), sf, np.array([1.0]))[0] < g[2]


def test_step_index_and_aggregate():
    t0 = datetime(2019, 10, 20)
    times = [t0 + timedelta(hours=h) for h in range(0, 24, 2)]   # 2-hourly over one day
    k = step_index(times, t0, 6, 4)
    assert k.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    ok = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], float)[:, None]
    # closed in step 1 because one output within it is closed
    assert (_aggregate(ok, k, 4, "min")[:, 0] > 0.5).tolist() == [True, False, True, True]


def test_closures_from_series():
    closed = {("A", "B", "foot"): np.array([0, 1, 1, 0, 1], bool),
              ("A", "B", "vehicle"): np.array([1, 1, 1, 1, 1], bool)}
    assert closures_from_series(closed, 5, "foot") == [["link", "A", "B", 1, 3], ["link", "A", "B", 4, 6]]
    assert closures_from_series(closed, 5, "vehicle") == [["link", "A", "B", 0, 6]]


def test_aggregate_raises_when_no_hydraulic_outputs_fall_in_simulation():
    values = np.array([[1.0], [2.0]])
    k = np.array([-1, -1])

    with pytest.raises(ValueError, match="No hydraulic outputs are available"):
        _aggregate(values, k, 3, "max")
