"""Map HEC-RAS 2D results onto a DFlee location graph.

Links: depth and speed are sampled at regular spacing along each road
polyline. At each hydraulic output a link is passable for a mode if the share
of impassable samples does not exceed ``max_impassable_share`` (0 means a
single impassable sample closes the link). A link is closed for a whole DFlee
timestep if it is closed at any hydraulic output within that timestep, and its
speed factor is the minimum over the timestep.

Locations: a summary statistic of depth over the settlement footprint is
taken at each output, its maximum over the DFlee timestep is discretised into
DFlee flood levels 0-3 using thresholds theta_1 < theta_2 < theta_3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings

import numpy as np
from shapely import contains_xy
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, Point
from shapely.prepared import prep

from .criteria import passable, speed_closed, speed_factor
from .rasread import MeshResults


@dataclass
class Domain:
    meshes: list[MeshResults]
    times: list[datetime] = field(init=False)
    depth: np.ndarray = field(init=False)     # (T, Ntot)
    speed: np.ndarray = field(init=False)     # (T, Ntot)
    centres: np.ndarray = field(init=False)   # (Ntot, 2)

    def __post_init__(self):
        t0 = self.meshes[0].times
        for m in self.meshes[1:]:
            if m.times != t0:
                raise ValueError("2D flow areas have different output times.")
        self.times = t0
        self.depth = np.hstack([m.depth for m in self.meshes])
        self.speed = np.hstack([m.speed for m in self.meshes])
        self.centres = np.vstack([m.centres for m in self.meshes])
        beds = [getattr(m, "bed", None) for m in self.meshes]
        self.bed = np.concatenate(beds) if all(b is not None for b in beds) else None
        self._offsets = np.cumsum([0] + [m.centres.shape[0] for m in self.meshes])
        self._polys = [prep(Polygon(m.perimeter)) for m in self.meshes]
        self._raw_polys = [Polygon(m.perimeter) for m in self.meshes]
        self._trees = [cKDTree(m.centres) for m in self.meshes]

    def locate(self, xy: np.ndarray) -> np.ndarray:
        """Global cell index for each point, or -1 if outside every flow area."""
        out = np.full(len(xy), -1, dtype=int)
        for k, (poly, tree) in enumerate(zip(self._polys, self._trees)):
            inside = contains_xy(self._raw_polys[k], xy[:, 0], xy[:, 1]) & (out < 0)
            if inside.any():
                _, idx = tree.query(xy[inside])
                out[inside] = idx + self._offsets[k]
        return out


def sample_line(geom, spacing: float) -> np.ndarray:
    n = max(2, int(np.ceil(geom.length / spacing)) + 1)
    return np.array([geom.interpolate(s).coords[0] for s in np.linspace(0, geom.length, n)])


def step_index(times: list[datetime], start: datetime, step_hours: float, n_steps: int) -> np.ndarray:
    dt = np.array([(t - start).total_seconds() / 3600.0 for t in times])
    k = np.floor(dt / step_hours).astype(int)
    k[(dt < 0) | (k >= n_steps)] = -1
    return k


def _aggregate(values: np.ndarray, k: np.ndarray, n_steps: int, how: str) -> np.ndarray:
    """Aggregate per-output values (T, ...) into DFlee steps (n_steps, ...)."""
    shape = (n_steps,) + values.shape[1:]
    out = np.full(shape, np.nan)
    for s in range(n_steps):
        sel = values[k == s]
        if len(sel):
            out[s] = sel[0] if how == "first" else sel.min(axis=0) if how == "min" else sel.max(axis=0)
    # forward-fill steps without hydraulic output, back-fill leading gaps
    missing = np.isnan(out.reshape(n_steps, -1)[:, 0])
    if missing.all():
        raise ValueError(
            "No hydraulic outputs are available within the requested DFlee simulation steps."
        )
    if missing.any():
        warnings.warn(f"{missing.sum()} DFlee steps have no hydraulic output; filling from neighbours.")
        last = None
        for s in range(n_steps):
            if not missing[s]:
                last = out[s]
            elif last is not None:
                out[s] = last
        first = next(s for s in range(n_steps) if not missing[s])
        out[:first] = out[first]
    return out


def _window_mask(times, window):
    """Boolean per hydraulic output: inside [start, end] (ISO strings); all True if no window."""
    if not window:
        return np.ones(len(times), dtype=bool)
    a, b = (datetime.fromisoformat(str(w)) for w in window)
    return np.array([a <= t <= b for t in times])


def _bridge_closed(rule: dict, d_channel: np.ndarray, link) -> np.ndarray:
    """Closure of the river crossing itself, per hydraulic output.

    ``channel_open``: the crossing never closes (only the approaches are tested).
    ``rise``: closed when the water at the crossing has risen by at least the deck
    clearance above its level at the first output; the clearance is the link's
    ``deck_clearance`` column if given, else ``rule["rise"]``. With depth outputs
    only, a rise in depth over a fixed bed equals the rise of the water surface.
    ``closed``: closed throughout (a bound for sensitivity tests).
    """
    kind = (rule or {}).get("type", "channel_open")
    T = d_channel.shape[0]
    if kind == "channel_open" or d_channel.shape[1] == 0:
        return np.zeros(T, dtype=bool)
    if kind == "closed":
        return np.ones(T, dtype=bool)
    if kind == "rise":
        c = link.get("deck_clearance", None)
        clearance = float(c) if c is not None and not pd.isna(c) else float(rule["rise"])
        rise = (d_channel - d_channel[0:1, :]).max(axis=1)
        return rise >= clearance
    raise ValueError(f"Unknown bridge rule: {kind}")


def link_states(domain: Domain, links, modes: dict, spacing: float, start: datetime,
                step_hours: float, n_steps: int, max_impassable_share: float = 0.0,
                percentile: float = 95.0, exclude_initially_wet: float | None = None,
                bridge_rule: dict | None = None, outside_domain: dict | None = None,
                road_terrain=None, step_rule: str = "worst"):
    """Per-link, per-mode accessibility and speed factor on DFlee steps.

    ``step_rule``: ``"worst"`` (default) closes a link for the whole DFlee step if it is
    closed at any hydraulic output within it, and takes the lowest speed factor;
    ``"start"`` takes the state at the first output of the step, as a check on how much
    the conservative rule adds.

    Returns a list of dict rows and a dict of boolean closure arrays keyed by
    (name1, name2, mode).

    ``exclude_initially_wet``: if set (a depth in m), samples already wetter
    than this at the first hydraulic output are treated as the river channel and
    taken out of the depth test, so the approaches are tested on their own and the
    crossing itself follows ``bridge_rule`` (see ``_bridge_closed``). Links flagged
    ``bridge = False`` in the input keep all samples.

    ``outside_domain`` sets the treatment of samples outside the hydraulic domain:
    ``{"type": "open"}`` (default: they are passable), or ``{"type": "extent",
    "polygons": <shapely geometry>, "depth": d, "window": [start, end]}``: samples
    inside the polygons (e.g. a satellite flood extent) take depth ``d`` and zero
    velocity during the window and are dry otherwise.

    ``road_terrain``: optional function returning terrain elevation (m) at sample
    points, e.g. a DEM sampler. If given, the depth on the road is the cell's water
    surface minus the terrain at the point, max(d_cell + bed_cell - z_point, 0),
    instead of the cell depth, which HEC-RAS measures from the lowest point of the
    cell and so overstates depth on a road that runs along higher ground in a
    partly wet cell. Needs the cells' minimum elevations.
    """
    if step_rule not in ("worst", "start"):
        raise ValueError(f"step_rule must be 'worst' or 'start', not {step_rule!r}")
    k = step_index(domain.times, start, step_hours, n_steps)
    od = outside_domain or {"type": "open"}
    if od.get("type", "open") not in ("open", "extent"):
        raise ValueError(f"Unknown outside_domain rule for links: {od.get('type')}")
    if od.get("type") == "extent":
        od_poly = prep(od["polygons"])
        od_win = _window_mask(domain.times, od.get("window"))
        od_depth = float(od["depth"])
    rows, closed = [], {}
    for _, link in links.iterrows():
        xy = sample_line(link.geometry, spacing)
        idx = domain.locate(xy)
        inside = idx >= 0
        share_out = 1.0 - inside.mean()
        bridge = link.get("bridge", None)
        keep_channel = bridge is not None and not pd.isna(bridge) and not bool(bridge)
        n_channel = 0
        d_channel = np.zeros((len(domain.times), 0))
        if exclude_initially_wet is not None and not keep_channel and inside.any():
            base = inside & (domain.depth[0, np.where(inside, idx, 0)] > exclude_initially_wet)
            n_channel = int(base.sum())
            d_channel = domain.depth[:, idx[base]]
            inside = inside & ~base
        bridge_shut = _bridge_closed(bridge_rule, d_channel, link)
        # samples outside the domain that a flood extent covers
        d_out = np.zeros((len(domain.times), 0))
        if od.get("type") == "extent":
            outside = idx < 0
            hit = np.array([od_poly.contains(Point(p)) for p in xy[outside]], dtype=bool)
            if hit.any():
                d_out = np.where(od_win[:, None], od_depth, 0.0) * np.ones((1, int(hit.sum())))
        if not inside.any() and d_out.shape[1] == 0 and not bridge_shut.any():
            for mode in modes:
                closed[(link.name1, link.name2, mode)] = np.zeros(n_steps, dtype=bool)
                for s in range(n_steps):
                    rows.append(dict(step=s, name1=link.name1, name2=link.name2, mode=mode,
                                     accessible=1, speed_factor=1.0, depth_p=0.0, speed_p=0.0,
                                     share_outside_domain=round(float(share_out), 3),
                                     channel_samples_dropped=n_channel, bridge_closed=0))
            continue
        d_in = domain.depth[:, idx[inside]]
        if road_terrain is not None:
            if domain.bed is None:
                raise ValueError("road_terrain needs the cells' minimum elevations (bed) in the results")
            z = np.asarray(road_terrain(xy[inside]), dtype=float)
            d_in = np.maximum(d_in + domain.bed[idx[inside]][None, :] - z[None, :], 0.0)
        d = np.hstack([d_in, d_out])       # (T, S)
        v = np.hstack([domain.speed[:, idx[inside]], np.zeros_like(d_out)])
        if d.shape[1] == 0:                      # only the crossing itself is tested
            d = np.zeros((len(domain.times), 1)); v = np.zeros_like(d)
        hi, lo = ("max", "min") if step_rule == "worst" else ("first", "first")
        dp = _aggregate(np.percentile(d, percentile, axis=1)[:, None], k, n_steps, hi)[:, 0]
        vp = _aggregate(np.percentile(v, percentile, axis=1)[:, None], k, n_steps, hi)[:, 0]
        bridge_s = _aggregate(bridge_shut.astype(float)[:, None], k, n_steps, hi)[:, 0] > 0.5
        for mode, spec in modes.items():
            sf = spec.get("speed", {"type": "none"})
            ok = passable(d, v, spec["criterion"])                    # (T, S)
            ok_t = ((1.0 - ok.mean(axis=1)) <= max_impassable_share) & ~bridge_shut    # (T,)
            # a speed function that reaches zero (e.g. Pregnolato closure depth) also closes the link
            ok_t &= ~speed_closed(np.percentile(d, percentile, axis=1), sf, np.percentile(v, percentile, axis=1))
            ok_s = _aggregate(ok_t.astype(float)[:, None], k, n_steps, lo)[:, 0] > 0.5
            g = speed_factor(np.percentile(d, percentile, axis=1), sf, np.percentile(v, percentile, axis=1))
            g_s = _aggregate(g[:, None], k, n_steps, lo)[:, 0]
            g_s = np.where(ok_s, g_s, 0.0)
            closed[(link.name1, link.name2, mode)] = ~ok_s
            for s in range(n_steps):
                rows.append(dict(step=s, name1=link.name1, name2=link.name2, mode=mode,
                                 accessible=int(ok_s[s]), speed_factor=round(float(g_s[s]), 4),
                                 depth_p=round(float(dp[s]), 3), speed_p=round(float(vp[s]), 3),
                                 share_outside_domain=round(float(share_out), 3),
                                 channel_samples_dropped=n_channel,
                                 bridge_closed=int(bridge_s[s])))
    return rows, closed


def location_levels(domain: Domain, locations, start: datetime, step_hours: float, n_steps: int,
                    statistic: str, thresholds, wet_depth: float = 0.05, point_buffer: float = 250.0,
                    outside_domain: dict | None = None):
    """DFlee flood levels (0-3) per location and step.

    Locations outside the domain are None (DFlee then treats them as dry) unless
    ``outside_domain`` says otherwise:
    ``{"type": "dry"}`` (default);
    ``{"type": "nearest", "max_km": x, "level_offset": n}``: the series of the
    nearest location inside the domain within ``max_km``, lowered by ``n`` levels;
    ``{"type": "extent", "polygons": geom, "level": L, "window": [start, end],
    "min_share": s}``: level ``L`` during the window if at least share ``s`` of the
    footprint lies in the extent (e.g. a satellite flood map), else 0.
    """
    k = step_index(domain.times, start, step_hours, n_steps)
    tree = cKDTree(domain.centres)
    levels, notes = {}, {}
    th = np.asarray(thresholds, dtype=float)
    for _, loc in locations.iterrows():
        geom = loc.geometry
        if geom.geom_type == "Point":
            geom = geom.buffer(point_buffer)
        pg = prep(geom)
        centre = np.array(geom.centroid.coords[0])
        minx, miny, maxx, maxy = geom.bounds
        corner_dx = max(abs(centre[0] - minx), abs(maxx - centre[0]))
        corner_dy = max(abs(centre[1] - miny), abs(maxy - centre[1]))
        search_radius = np.hypot(corner_dx, corner_dy)
        cand = tree.query_ball_point(centre, r=np.nextafter(search_radius, np.inf))
        cells = [c for c in cand if pg.contains(Point(domain.centres[c]))]
        if not cells:
            cen = np.array(geom.centroid.coords[0])
            if domain.locate(cen[None, :])[0] < 0:
                levels[loc["name"]] = None
                notes[loc["name"]] = "outside hydraulic domain"
                continue
            _, c = tree.query(cen)
            cells = [int(c)]
            notes[loc["name"]] = "footprint smaller than a cell; nearest cell used"
        d = domain.depth[:, cells]
        if statistic == "inundated_fraction":
            stat = (d > wet_depth).mean(axis=1)
        elif statistic == "median_depth":
            stat = np.median(d, axis=1)
        elif statistic == "p90_depth":
            stat = np.percentile(d, 90, axis=1)
        else:
            raise ValueError(f"Unknown location statistic: {statistic}")
        smax = _aggregate(stat[:, None], k, n_steps, "max")[:, 0]
        levels[loc["name"]] = np.searchsorted(th, smax, side="right").astype(int)
    od = outside_domain or {"type": "dry"}
    kind = od.get("type", "dry")
    missing = [n for n, v in levels.items() if v is None]
    if kind == "dry" or not missing:
        return levels, notes
    geoms = {r["name"]: (r.geometry.buffer(point_buffer) if r.geometry.geom_type == "Point" else r.geometry)
             for _, r in locations.iterrows()}
    if kind == "nearest":
        have = [n for n, v in levels.items() if v is not None]
        if not have:
            return levels, notes
        cen = np.array([geoms[n].centroid.coords[0] for n in have])
        tree2 = cKDTree(cen)
        max_d = float(od.get("max_km", np.inf)) * float(od.get("units_per_km", 1000.0))
        off = int(od.get("level_offset", 0))
        for n in missing:
            dist, j = tree2.query(np.array(geoms[n].centroid.coords[0]))
            if dist <= max_d:
                levels[n] = np.maximum(levels[have[j]] - off, 0)
                notes[n] = f"outside hydraulic domain; level of {have[j]} minus {off}"
        return levels, notes
    if kind == "extent":
        poly = od["polygons"]
        win = np.array([start + timedelta(hours=step_hours * s) for s in range(n_steps)])
        a, b = (datetime.fromisoformat(str(w)) for w in od["window"])
        on = np.array([a <= t <= b for t in win])
        for n in missing:
            g = geoms[n]
            share = g.intersection(poly).area / g.area if g.area > 0 else 0.0
            if share >= float(od.get("min_share", 0.1)):
                levels[n] = np.where(on, int(od["level"]), 0).astype(int)
                notes[n] = f"outside hydraulic domain; level {od['level']} from flood extent ({share:.0%} of footprint)"
        return levels, notes
    raise ValueError(f"Unknown outside_domain rule for locations: {kind}")


def closures_from_series(closed: dict, n_steps: int, mode: str) -> list[list]:
    """DFlee closures.csv rows: link,name1,name2,start,end (reopened at end)."""
    rows = []
    for (n1, n2, m), series in closed.items():
        if m != mode:
            continue
        s = 0
        while s < n_steps:
            if series[s]:
                e = s
                while e < n_steps and series[e]:
                    e += 1
                rows.append(["link", n1, n2, s, e if e < n_steps else n_steps + 1])
                s = e
            else:
                s += 1
    return rows
