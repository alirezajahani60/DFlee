"""Bridge rules, out-of-domain treatments and the extent check on a synthetic mesh."""
import pytest

# Optional dependencies of flee.preprocessing.hecras2dflee (pip install flee[hydro]).
pytest.importorskip("geopandas")
pytest.importorskip("shapely")
pytest.importorskip("scipy")

from datetime import datetime, timedelta

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point, box

from flee.preprocessing.hecras2dflee.coupling import Domain, link_states, location_levels
from flee.preprocessing.hecras2dflee.rasread import MeshResults
from flee.preprocessing.hecras2dflee.validation import compare_extent, coverage

T0 = datetime(2019, 10, 20)


def _domain():
    """10 x 3 cells of 100 m over x in [0, 1000]; column x = 450-550 is the river (wet from t0).

    The river rises by 0, 1, 2, 3 m over four hourly outputs; the floodplain
    cells east of x = 700 flood to 0.5 m from the third output.
    """
    xs, ys = np.meshgrid(np.arange(50, 1000, 100), np.arange(50, 300, 100))
    c = np.c_[xs.ravel(), ys.ravel()].astype(float)
    T = 4
    d = np.zeros((T, len(c)))
    river = (c[:, 0] > 400) & (c[:, 0] < 600)
    for t in range(T):
        d[t, river] = 2.0 + t
    d[2:, c[:, 0] > 700] = 0.5
    m = MeshResults("a", [T0 + timedelta(hours=h) for h in range(T)], c, d, np.zeros_like(d),
                    np.array([[0, 0], [1000, 0], [1000, 300], [0, 300]], float), None)
    return Domain([m])


MODES = {"foot": {"criterion": {"type": "depth", "max_depth": 1.0}},
         "vehicle": {"criterion": {"type": "depth", "max_depth": 0.3}}}


def _links(**cols):
    g = gpd.GeoDataFrame(dict(name1=["W"], name2=["E"], bridge=[True], **cols),
                         geometry=[LineString([(100, 150), (650, 150)])])
    return g


def _closed(rows, mode):
    return [r["accessible"] == 0 for r in rows if r["mode"] == mode]


def test_bridge_rules():
    dom = _domain()
    kw = dict(modes=MODES, spacing=25.0, start=T0, step_hours=1.0, n_steps=4, exclude_initially_wet=1.0)
    rows, _ = link_states(dom, _links(), **kw)                            # channel_open (default)
    assert not any(_closed(rows, "foot"))
    rows, _ = link_states(dom, _links(), bridge_rule={"type": "rise", "rise": 2.0}, **kw)
    assert _closed(rows, "foot") == [False, False, True, True]           # rise of 2 m from the third output
    rows, _ = link_states(dom, _links(deck_clearance=[1.0]), bridge_rule={"type": "rise", "rise": 2.0}, **kw)
    assert _closed(rows, "foot") == [False, True, True, True]            # per-link clearance wins
    rows, _ = link_states(dom, _links(), bridge_rule={"type": "closed"}, **kw)
    assert all(_closed(rows, "vehicle"))
    assert all(r["bridge_closed"] == 1 for r in rows)


def test_outside_domain_links():
    dom = _domain()
    g = gpd.GeoDataFrame(dict(name1=["A"], name2=["B"]), geometry=[LineString([(1200, 150), (1800, 150)])])
    kw = dict(modes=MODES, spacing=50.0, start=T0, step_hours=1.0, n_steps=4)
    rows, _ = link_states(dom, g, **kw)
    assert not any(_closed(rows, "vehicle"))                             # open by default
    od = {"type": "extent", "polygons": box(1400, 0, 1600, 300), "depth": 0.5,
          "window": [str(T0 + timedelta(hours=2)), str(T0 + timedelta(hours=3))]}
    rows, _ = link_states(dom, g, outside_domain=od, **kw)
    assert _closed(rows, "vehicle") == [False, False, True, True]
    assert not any(_closed(rows, "foot"))                                # 0.5 m is passable on foot here
    cov = coverage(rows, {"X": "outside hydraulic domain"}, 3)
    assert cov["links_fully_outside"] == 1 and cov["locations_outside"] == 1


def test_outside_domain_locations():
    dom = _domain()
    locs = gpd.GeoDataFrame(dict(name=["in_east", "out_east", "far"]),
                            geometry=[box(750, 0, 950, 300), box(1100, 0, 1300, 300), Point(9000, 150)])
    kw = dict(start=T0, step_hours=1.0, n_steps=4, statistic="inundated_fraction",
              thresholds=[0.1, 0.5, 0.9], point_buffer=100.0)
    lv, notes = location_levels(dom, locs, **kw)
    assert lv["in_east"].tolist() == [0, 0, 3, 3] and lv["out_east"] is None
    lv, notes = location_levels(dom, locs, outside_domain={"type": "nearest", "max_km": 1.0, "level_offset": 1}, **kw)
    assert lv["out_east"].tolist() == [0, 0, 2, 2] and lv["far"] is None
    ext = {"type": "extent", "polygons": box(1000, 0, 1400, 300), "level": 2,
           "window": [str(T0 + timedelta(hours=1)), str(T0 + timedelta(hours=2))]}
    lv, notes = location_levels(dom, locs, outside_domain=ext, **kw)
    assert lv["out_east"].tolist() == [0, 2, 2, 0] and lv["far"] is None


def test_compare_extent():
    dom = _domain()
    obs = box(700, 0, 1000, 300)                                         # observed wet: x > 700
    r = compare_extent(dom, obs, box(0, 0, 1000, 300), T0 + timedelta(hours=2.2), wet_depth=0.1,
                       exclude_initially_wet=1.0)
    assert r["csi"] == 1.0 and r["cells"] == 24                          # river cells left out
    r = compare_extent(dom, obs, box(0, 0, 1000, 300), T0, wet_depth=0.1, exclude_initially_wet=1.0)
    assert r["hit_rate"] == 0.0


def test_dv_limits_criterion():
    import numpy as np
    from flee.preprocessing.hecras2dflee.criteria import passable
    small = dict(type="dv_limits", max_depth=0.3, max_velocity=2.0, max_dv=0.3)
    d = np.array([0.2, 0.2, 0.35, 0.25])
    v = np.array([1.0, 2.5, 0.1, 1.5])          # dv: 0.2, 0.5, 0.035, 0.375
    assert passable(d, v, small).tolist() == [True, False, False, False]


def test_point_depth_from_water_surface_and_terrain():
    xs, ys = np.meshgrid(np.arange(50, 1000, 100), np.arange(50, 300, 100))
    c = np.c_[xs.ravel(), ys.ravel()].astype(float)
    d = np.full((2, len(c)), 0.6)                      # every cell 0.6 m deep above its lowest point
    bed = np.zeros(len(c))                             # lowest point of each cell at 0 m
    m = MeshResults("a", [T0, T0 + timedelta(hours=1)], c, d, np.zeros_like(d),
                    np.array([[0, 0], [1000, 0], [1000, 300], [0, 300]], float), None, bed)
    dom = Domain([m])
    g = gpd.GeoDataFrame(dict(name1=["A"], name2=["B"]), geometry=[LineString([(100, 150), (900, 150)])])
    kw = dict(modes=MODES, spacing=50.0, start=T0, step_hours=1.0, n_steps=2)
    rows, _ = link_states(dom, g, **kw)
    assert all(_closed(rows, "vehicle"))               # cell depth 0.6 m closes it for vehicles
    raised = lambda xy: np.full(len(xy), 0.4)          # road on an embankment 0.4 m above the cell bed
    rows, _ = link_states(dom, g, road_terrain=raised, **kw)
    assert not any(_closed(rows, "vehicle"))           # 0.2 m on the road surface


def test_location_levels_covers_long_narrow_footprint():
    dom = _domain()
    locs = gpd.GeoDataFrame(
        dict(name=["long_narrow"]),
        geometry=[box(0, 149, 1000, 151)]
    )

    levels, _ = location_levels(
        dom,
        locs,
        start=T0,
        step_hours=1.0,
        n_steps=4,
        statistic="inundated_fraction",
        thresholds=[0.1, 0.5, 0.9],
        wet_depth=0.05,
        point_buffer=100.0,
    )

    # Ten cell centres lie inside the footprint.
    # Two are wet initially; five are wet from hour 2.
    assert levels["long_narrow"].tolist() == [1, 1, 2, 2]
