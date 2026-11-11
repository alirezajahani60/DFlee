"""Modelled river stage at a gauge, compared with observed readings.

The modelled water surface is read from a HEC-RAS reference point (2D) or a 1D
cross section in the plan HDF and converted to gauge stage by subtracting the
gauge datum. If the datum is unknown, ``datum="peak"`` sets it so that the modelled
peak equals the observed peak; only timing is then tested, and the result says so.

Reported: modelled stage at each observation time inside the simulation period
(linear interpolation) and its error, with the readings outside the period listed
separately and left out of the RMSE; modelled and observed peak and its time; and the first time each
flood-risk threshold is reached, modelled and observed.

Command line::

    python -m flee.preprocessing.hecras2dflee.stage plan.hdf --refpoint NAME | --xs "River Reach RS" \
        --observed readings.csv --thresholds 6.5 7.3 8.3 [--datum 123.4 | --datum peak] -o stage
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import h5py
import numpy as np
import pandas as pd

from .gauge_warnings import crossing_times, read_gauge
from .provenance import TS, _s
from .rasread import _parse_time

FT = 0.3048


def _plan_times(h):
    for key in (f"{TS}/Time Date Stamp", f"{TS}/Time Date Stamp (ms)"):
        if key in h:
            return [_parse_time(x) for x in h[key][()]]
    raise ValueError("no output times in the plan HDF")


def read_water_surface(plan_hdf, refpoint=None, xs=None):
    """(times, water surface in m) at a reference point name or a 1D cross section.

    ``xs`` is matched against "River Reach RS" from Geometry/Cross Sections/Attributes.
    """
    with h5py.File(plan_hdf, "r") as h:
        us = _s(h.attrs.get("Units System", "")) == "US Customary"
        times = _plan_times(h)
        if refpoint is not None:
            names = [_s(x) for x in h[f"{TS}/Reference Points/Name"][()]]
            if refpoint not in names:
                raise KeyError(f"reference point {refpoint!r} not in {names}")
            ws = h[f"{TS}/Reference Points/Water Surface"][:, names.index(refpoint)].astype(float)
        elif xs is not None:
            att = h["Geometry/Cross Sections/Attributes"][()]
            labels = [" ".join(_s(r[k]) for k in ("River", "Reach", "RS")) for r in att]
            hits = [i for i, l in enumerate(labels) if " ".join(l.split()) == " ".join(xs.split())]
            if not hits:
                raise KeyError(f"cross section {xs!r} not found; e.g. {labels[:3]}")
            ws = h[f"{TS}/Cross Sections/Water Surface"][:, hits[0]].astype(float)
        else:
            raise ValueError("give refpoint or xs")
    return times, ws * FT if us else ws


def compare(times, ws, observed, thresholds, datum):
    """Compare modelled stage with observed readings; see the module docstring."""
    t = np.array([x.timestamp() for x in times])
    all_t = np.array([o[0].timestamp() for o in observed])
    # readings outside the simulation period have no modelled value; np.interp would repeat the end values
    inside = (all_t >= t[0]) & (all_t <= t[-1])
    excluded = [o[0].isoformat() for o, keep in zip(observed, inside) if not keep]
    if not inside.any():
        raise ValueError("No observed reading falls within the simulation period "
                         f"({times[0].isoformat()} to {times[-1].isoformat()}).")
    obs_in = [o for o, keep in zip(observed, inside) if keep]
    obs_t = all_t[inside]; obs_h = np.array([o[1] for o in obs_in])
    i_obs_peak = int(np.argmax(obs_h))
    peak_matched = datum == "peak"
    z0 = float(ws.max() - obs_h.max()) if peak_matched else float(datum)
    stage = ws - z0
    mod_at_obs = np.interp(obs_t, t, stage)
    i_pk = int(np.argmax(stage))
    mod_series = list(zip(times, stage))
    cross_m = crossing_times(mod_series, thresholds)
    cross_o = crossing_times(list(observed), thresholds)
    rows = [dict(time=o[0].isoformat(), observed=o[1], modelled=round(float(m), 3), error=round(float(m - o[1]), 3))
            for o, m in zip(obs_in, mod_at_obs)]
    return dict(datum=z0, datum_from_peak=peak_matched,
                readings=rows, readings_outside_period=excluded, rmse=round(float(np.sqrt(np.mean((mod_at_obs - obs_h) ** 2))), 3),
                peak=dict(modelled=round(float(stage[i_pk]), 3), modelled_time=times[i_pk].isoformat(),
                          observed=float(obs_h[i_obs_peak]), observed_time=obs_in[i_obs_peak][0].isoformat()),
                thresholds=[dict(threshold=th, modelled=(m.isoformat() if m else None),
                                 observed=(None if before else o.isoformat() if o else None),
                                 observed_note=(f"before the first reading ({observed[0][0]:%Y-%m-%d})" if before else None),
                                 lag_hours=(round((m - o).total_seconds() / 3600, 1) if m and o and not before else None))
                            for th, m, o, before in zip(thresholds, cross_m, cross_o,
                                                        [observed[0][1] > th for th in thresholds])])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("plan_hdf")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--refpoint"); g.add_argument("--xs")
    ap.add_argument("--observed", required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[6.5, 7.3, 8.3])
    ap.add_argument("--datum", default="peak")
    ap.add_argument("-o", "--output", default="stage")
    a = ap.parse_args(argv)
    times, ws = read_water_surface(a.plan_hdf, a.refpoint, a.xs)
    datum = a.datum if a.datum == "peak" else float(a.datum)
    res = compare(times, ws, read_gauge(a.observed), a.thresholds, datum)
    json.dump(res, open(a.output + ".json", "w"), indent=1)
    pd.DataFrame({"time": [x.isoformat() for x in times], "water_surface_m": ws, "stage_m": ws - res["datum"]}).to_csv(a.output + "_series.csv", index=False)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
