"""Checks of the hydraulic forcing against observations.

``compare_extent``: modelled against observed flood extent at one date, on the
model cells whose centres lie inside the observation's mapping area (AOI).
A cell is observed wet if its centre lies in the observed extent, and modelled
wet if its depth at the output nearest to the image time exceeds ``wet_depth``.
Cells already wet at the first output (the river) can be left out with
``exclude_initially_wet``. Reports hit rate H = TP/(TP+FN), false alarm ratio
FAR = FP/(TP+FP), critical success index CSI = TP/(TP+FP+FN) and the bias
(modelled/observed wet area).

``coverage``: how much of the DFlee graph the hydraulic domain covers.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
from shapely import contains_xy


def compare_extent(domain, observed, aoi, time: datetime, wet_depth: float = 0.1,
                   exclude_initially_wet: float | None = None) -> dict:
    t = int(np.argmin([abs((tt - time).total_seconds()) for tt in domain.times]))
    x, y = domain.centres[:, 0], domain.centres[:, 1]
    in_aoi = contains_xy(aoi, x, y)
    if exclude_initially_wet is not None:
        in_aoi &= domain.depth[0] <= exclude_initially_wet
    obs = contains_xy(observed, x, y)[in_aoi]
    mod = (domain.depth[t] > wet_depth)[in_aoi]
    tp = int((obs & mod).sum()); fp = int((~obs & mod).sum()); fn = int((obs & ~mod).sum())
    tn = int((~obs & ~mod).sum())
    return dict(output_time=domain.times[t].isoformat(), cells=int(in_aoi.sum()),
                tp=tp, fp=fp, fn=fn, tn=tn,
                hit_rate=round(tp / max(tp + fn, 1), 3), false_alarm_ratio=round(fp / max(tp + fp, 1), 3),
                csi=round(tp / max(tp + fp + fn, 1), 3), bias=round((tp + fp) / max(tp + fn, 1), 3))


def coverage(link_rows, location_notes: dict, n_locations: int) -> dict:
    """Share of links (fully, partly) and locations outside the hydraulic domain."""
    import pandas as pd
    df = pd.DataFrame(link_rows).groupby(["name1", "name2"])["share_outside_domain"].first()
    out_locs = sorted(n for n, v in location_notes.items() if v.startswith("outside hydraulic domain"))
    return dict(links=int(len(df)), links_fully_outside=int((df >= 1.0).sum()),
                links_partly_outside=int(((df > 0) & (df < 1.0)).sum()),
                mean_share_of_link_length_outside=round(float(df.mean()), 3),
                locations=n_locations, locations_outside=len(out_locs), locations_outside_names=out_locs)
