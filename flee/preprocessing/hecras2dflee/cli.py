"""Command-line entry point: python -m flee.preprocessing.hecras2dflee config.yaml"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

from .coupling import Domain, closures_from_series, link_states, location_levels
from .validation import compare_extent, coverage
from .rasread import flow_area_names, read_mesh

import h5py


def _load_vector(path: str, crs_wkt: str | None, model_crs: str | None):
    gdf = gpd.read_file(path)
    target = model_crs or crs_wkt
    if target and gdf.crs is not None:
        gdf = gdf.to_crs(target)
    elif gdf.crs is None:
        print(f"Warning: {path} has no CRS; assuming the model CRS.", file=sys.stderr)
    return gdf


def _outside_rule(rule, crs_wkt, model_crs):
    """Load the polygons of an ``extent`` rule (file -> union geometry in the model CRS)."""
    if not rule or rule.get("type") != "extent":
        return rule
    r = dict(rule)
    r["polygons"] = _load_vector(rule["file"], crs_wkt, model_crs).geometry.buffer(0).union_all()
    return r


def _road_terrain(rule, hyd):
    """links.road_depth: "cell" (default, HEC-RAS cell depth) or "point" (water surface minus the
    terrain at each road sample, using hydraulics.dem)."""
    if rule in (None, "cell"):
        return None
    if rule != "point":
        raise ValueError(f"links.road_depth must be cell or point, not {rule}")
    if not hyd.get("dem"):
        raise ValueError("links.road_depth: point needs hydraulics.dem")
    from .rasread import sample_raster
    return lambda xy: sample_raster(hyd["dem"], xy)


def run(cfg: dict) -> dict:
    hyd = cfg["hydraulics"]
    with h5py.File(hyd["plan_hdf"], "r") as h:
        areas = hyd.get("flow_areas") or flow_area_names(h)
    meshes = [read_mesh(hyd["plan_hdf"], a, hyd.get("geometry_hdf"), hyd.get("dem"),
                      hyd.get("nonfinite", "dry")) for a in areas]
    domain = Domain(meshes)

    sim = cfg["simulation"]
    start = datetime.fromisoformat(sim["start"])
    step_h = float(sim.get("step_hours", 6))
    n_steps = int(sim["n_steps"])
    model_crs = hyd.get("crs")
    crs_wkt = meshes[0].crs_wkt
    # Mesh coordinates are in the model CRS units; geometry inputs are reprojected to it.
    out = cfg["output"]["dir"]
    os.makedirs(out, exist_ok=True)
    summary = {"flow_areas": areas, "n_outputs": len(domain.times),
               "first_output": domain.times[0].isoformat(), "last_output": domain.times[-1].isoformat()}
    summary["nonfinite_values"] = {m.name: getattr(m, "nonfinite", None) for m in meshes}

    # ---- links
    if "links" in cfg:
        lc = cfg["links"]
        links = _load_vector(lc["file"], crs_wkt, model_crs)
        spacing = float(lc.get("sample_spacing", 25.0))
        if getattr(meshes[0], "centres_units", "m") == "ft":
            spacing = spacing / 0.3048  # spacing given in metres; mesh coordinates in feet
        step_rule = str(lc.get("step_rule", "worst"))
        largs = (domain, links, lc["modes"], spacing, start, step_h, n_steps,
                 float(lc.get("max_impassable_share", 0.0)),
                 float(lc.get("percentile", 95)),
                 lc.get("exclude_initially_wet"),
                 lc.get("bridge_rule"),
                 _outside_rule(lc.get("outside_domain"), crs_wkt, model_crs),
                 _road_terrain(lc.get("road_depth"), hyd))
        rows, closed = link_states(*largs, step_rule=step_rule)
        pd.DataFrame(rows).to_csv(os.path.join(out, "link_states.csv"), index=False)
        summary["step_rule"] = step_rule
        if step_rule == "worst" and lc.get("within_step_check", True):
            # what the conservative rule adds: closed link-steps that are open at the start of the step
            srows, _ = link_states(*largs, step_rule="start")
            both = pd.DataFrame(rows).merge(pd.DataFrame(srows), on=["step", "name1", "name2", "mode"],
                                            suffixes=("", "_start"))
            shut = both.accessible == 0
            within = shut & (both.accessible_start == 1)
            dt = np.diff([t.timestamp() for t in domain.times])
            summary["within_step_closures"] = dict(
                output_interval_h=float(np.median(dt) / 3600) if len(dt) else None,
                share_of_closed_link_steps=round(float(within.sum() / max(int(shut.sum()), 1)), 4),
                by_mode={m: round(float(within[both["mode"] == m].sum() / max(int(shut[both["mode"] == m].sum()), 1)), 4)
                         for m in lc["modes"]})
        for mode in lc["modes"]:
            crow = closures_from_series(closed, n_steps, mode)
            with open(os.path.join(out, f"closures_{mode}.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["#closure_type", "name1", "name2", "closure_start", "closure_end"])
                w.writerows(crow)
            summary[f"closure_events_{mode}"] = len(crow)
        summary["links"] = len(links)
        summary["links_fully_outside_domain"] = int(
            pd.DataFrame(rows).groupby(["name1", "name2"])["share_outside_domain"].first().eq(1.0).sum())
        summary["bridge_rule"] = lc.get("bridge_rule", {"type": "channel_open"})
        link_rows = rows

    # ---- locations
    if "locations" in cfg:
        lc = cfg["locations"]
        locs = _load_vector(lc["file"], crs_wkt, model_crs)
        buf = float(lc.get("point_buffer", 250.0))
        if getattr(meshes[0], "centres_units", "m") == "ft":
            buf = buf / 0.3048
        od = _outside_rule(lc.get("outside_domain"), crs_wkt, model_crs)
        if od and od.get("type") == "nearest" and getattr(meshes[0], "centres_units", "m") == "ft":
            od = dict(od, units_per_km=1000 / 0.3048)
        levels, notes = location_levels(domain, locs, start, step_h, n_steps, lc["statistic"],
                                        lc["thresholds"], float(lc.get("wet_depth", 0.05)), buf, od)
        inside = {k: v for k, v in levels.items() if v is not None}
        df = pd.DataFrame(inside)
        df.insert(0, "#Day", range(n_steps))
        df.to_csv(os.path.join(out, "flood_level.csv"), index=False)
        summary["locations_written"] = list(inside)
        summary["location_notes"] = notes
        if "links" in cfg:
            summary["coverage"] = coverage(link_rows, notes, len(locs))

    # ---- check against an observed flood extent (e.g. Copernicus EMS)
    if "validation" in cfg:
        vc = cfg["validation"]
        obs = _load_vector(vc["extent_file"], crs_wkt, model_crs).geometry.buffer(0).union_all()
        aoi = _load_vector(vc["aoi_file"], crs_wkt, model_crs).geometry.buffer(0).union_all()
        summary["extent_check"] = compare_extent(domain, obs, aoi, datetime.fromisoformat(vc["time"]),
                                                 float(vc.get("wet_depth", 0.1)), vc.get("exclude_initially_wet"))

    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Convert HEC-RAS 2D results into DFlee inputs.")
    ap.add_argument("config")
    a = ap.parse_args(argv)
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    s = run(cfg)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
