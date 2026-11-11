import pytest

# Optional dependencies of flee.preprocessing.hecras2dflee (pip install flee[hydro]).
pytest.importorskip("scipy")

import pandas as pd

from flee.preprocessing.hecras2dflee.fragmentation import fragmentation


def test_isolation_and_last_exit():
    # A - B - C; A floods from step 1; link A-B closes for foot from step 2, never for car.
    routes = pd.DataFrame({"name1": ["A", "B"], "name2": ["B", "C"], "distance": [10, 10]})
    levels = pd.DataFrame({"A": [0, 2, 3, 3], "B": [0, 0, 0, 0], "C": [0, 0, 0, 0]},
                          index=pd.Index(range(4), name="step"))
    ls = pd.DataFrame([dict(step=s, name1="A", name2="B", mode=m, accessible=int(not (m == "foot" and s >= 2)))
                       for s in range(4) for m in ("foot", "car")])
    by, last = fragmentation(ls, routes, levels, {"A": 100, "B": 50, "C": 10})
    foot = by[by["mode"] == "foot"].set_index("step")
    car = by[by["mode"] == "car"].set_index("step")
    assert foot.loc[1, "isolated_population"] == 0 and foot.loc[2, "isolated_population"] == 100
    assert foot.loc[2, "components"] == 2 and car.loc[3, "isolated_locations"] == 0
    lf = last.set_index(["origin", "mode"])
    assert lf.loc[("A", "foot"), "isolated_from_step"] == 2
    assert lf.loc[("A", "foot"), "last_exit_step"] == 1
    assert pd.isna(lf.loc[("A", "car"), "last_exit_step"])
