import pytest

# Optional dependencies of flee.preprocessing.hecras2dflee (pip install flee[hydro]).
pytest.importorskip("h5py")

import os
from datetime import datetime as D, timedelta

import numpy as np
import pytest

from flee.preprocessing.hecras2dflee import provenance, stage

# HEC-RAS example plan with reference points, from the rashdf test data (https://github.com/fema-ffrd/rashdf)
REF = os.path.join(os.environ.get("RASHDF_TESTDATA", ""), "BaldEagleDamBrk.reflines-refpts.p18.hdf")
needs_ref = pytest.mark.skipif(not os.path.exists(REF), reason="set RASHDF_TESTDATA to rashdf/tests/data/ras")


def test_compare_stage_with_datum_and_peak():
    t0 = D(2019, 10, 15)
    times = [t0 + timedelta(hours=6 * i) for i in range(60)]
    ws = 100.0 + np.interp(range(60), [0, 20, 40, 59], [5.0, 7.5, 8.4, 8.0])   # water surface, m
    obs = [(D(2019, 10, 18), 7.3), (D(2019, 10, 22), 7.8), (D(2019, 10, 25), 8.35)]
    r = stage.compare(times, ws, obs, [7.3, 8.3], datum=100.0)
    assert r["peak"]["modelled"] == 8.4 and not r["datum_from_peak"]
    assert r["thresholds"][0]["modelled"] is not None and r["thresholds"][0]["observed"] == "2019-10-18T00:00:00"
    r2 = stage.compare(times, ws, obs, [7.3, 8.3], datum="peak")
    assert r2["datum_from_peak"] and abs(r2["peak"]["modelled"] - 8.35) < 1e-9


@needs_ref
def test_reference_point_reader_and_provenance():
    import h5py
    with h5py.File(REF, "r") as h:
        name = provenance._s(h[f"{provenance.TS}/Reference Points/Name"][()][0])
    times, ws = stage.read_water_surface(REF, refpoint=name)
    assert len(times) == len(ws) == 37 and np.isfinite(ws).all()
    rec = provenance.collect(REF)
    assert rec["outputs"]["n_outputs"] == 37 and name in rec["reference_points"]
    assert "sha256" in next(iter(rec["files"].values()))
