"""Record the provenance of a HEC-RAS run used to force DFlee.

Collects what can be read from the files, so it does not have to be copied by hand:
HEC-RAS version and units; plan, geometry and flow file names and titles; the
simulation window and the computation, output and mapping intervals (from the
plan text file ``.pXX``); output times in the plan HDF and their spacing; the 2D
flow areas with cell counts, nominal cell spacing and default Manning's n; the
number of 1D cross sections; reference points and lines; structures by type,
including bridges; and SHA-256 checksums of every file. Items that need the
modeller (gauge datum, deck elevations, inundation threshold) are listed as
"to confirm".

Command line::

    python -m flee.preprocessing.hecras2dflee.provenance plan.pXX.hdf [--geometry g.hdf] [--plan-text plan.pXX] -o provenance
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime

import h5py
import numpy as np

TS = "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series"
# Items the plan files usually do not answer; to be confirmed with the model's authors.
TO_CONFIRM = [
    "gauge datum (to convert water-surface elevation to gauge stage)",
    "cross section or reference point at the gauge used for warnings",
    "deck and low-chord elevations of bridges on the network, if not in the geometry",
    "depth threshold used by the modellers to count a cell as inundated",
    "calibrated Manning's n by land cover, if not stored in the geometry HDF",
    "processing applied to the outputs before they were shared",
]


def _s(x):
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode(errors="replace").strip()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return [_s(v) for v in x.tolist()] if x.dtype.kind in "SO" else x.tolist()
    return x


def _attrs(obj):
    return {k: _s(v) for k, v in obj.attrs.items()}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_plan_text(path):
    """Key=value pairs of interest from a HEC-RAS plan text file (.pXX)."""
    keys = ("Plan Title", "Program Version", "Short Identifier", "Simulation Date", "Geom File",
            "Flow File", "Computation Interval", "Output Interval", "Instantaneous Interval",
            "Mapping Interval", "Run HTab", "Run UNet")
    out = {}
    with open(path, errors="replace") as f:
        for line in f:
            if "=" in line:
                k, v = line.rstrip("\n").split("=", 1)
                if k.strip() in keys:
                    out[k.strip()] = v.strip()
    return out


def _times(h):
    for key in (f"{TS}/Time Date Stamp", f"{TS}/Time Date Stamp (ms)"):
        if key in h:
            raw = [_s(x) for x in h[key][()]]
            ts = []
            for r in raw:
                for fmt in ("%d%b%Y %H:%M:%S:%f", "%d%b%Y %H:%M:%S"):
                    try:
                        ts.append(datetime.strptime(r, fmt)); break
                    except ValueError:
                        pass
            if ts:
                d = np.diff([t.timestamp() for t in ts]) / 60.0
                return dict(n_outputs=len(ts), first=ts[0].isoformat(), last=ts[-1].isoformat(),
                            interval_minutes=sorted(set(np.round(d, 3).tolist())))
    return None


def _flow_areas(g):
    out = []
    if "2D Flow Areas/Attributes" not in g:
        return out
    att = g["2D Flow Areas/Attributes"][()]
    names = att.dtype.names
    for row in att:
        d = {n: _s(row[n]) for n in names}
        out.append(dict(name=d.get("Name"), cell_count=d.get("Cell Count"),
                        spacing_dx=d.get("Spacing dx"), spacing_dy=d.get("Spacing dy"),
                        default_mannings_n=round(float(d["Mann"]), 4) if d.get("Mann") is not None else None))
    return out


def _structures(g):
    if "Structures/Attributes" not in g:
        return []
    att = g["Structures/Attributes"][()]
    names = att.dtype.names
    out = []
    for row in att:
        d = {n: _s(row[n]) for n in names}
        out.append({k: d.get(k) for k in ("Type", "Mode", "River", "Reach", "RS", "Connection") if k in d})
    return out


def collect(plan_hdf, geometry_hdf=None, plan_text=None):
    rec = dict(files={}, hec_ras={}, plan={}, outputs=None, flow_areas=[], cross_sections=0,
               reference_points=[], reference_lines=[], structures=[], to_confirm=TO_CONFIRM)
    for p in (plan_hdf, geometry_hdf, plan_text):
        if p:
            rec["files"][os.path.basename(p)] = dict(path=os.path.abspath(p), bytes=os.path.getsize(p), sha256=sha256(p))
    with h5py.File(plan_hdf, "r") as h:
        rec["hec_ras"] = _attrs(h)
        for grp in ("Plan Data/Plan Information", "Plan Data/Plan Parameters"):
            if grp in h:
                rec["plan"][grp.split("/")[-1]] = _attrs(h[grp])
        rec["outputs"] = _times(h)
        if f"{TS}/Reference Points" in h:
            rec["reference_points"] = [_s(x) for x in h[f"{TS}/Reference Points/Name"][()]]
        if f"{TS}/Reference Lines" in h and "Name" in h[f"{TS}/Reference Lines"]:
            rec["reference_lines"] = [_s(x) for x in h[f"{TS}/Reference Lines/Name"][()]]
        geoms = [h["Geometry"]] if "Geometry" in h else []
        gh = h5py.File(geometry_hdf, "r") if geometry_hdf else None
        try:
            if gh is not None and "Geometry" in gh:
                geoms.append(gh["Geometry"])
                rec["geometry_file"] = _attrs(gh)
            for g in geoms:
                rec["flow_areas"] = rec["flow_areas"] or _flow_areas(g)
                rec["structures"] = rec["structures"] or _structures(g)
                if "Cross Sections/Attributes" in g:
                    rec["cross_sections"] = max(rec["cross_sections"], int(g["Cross Sections/Attributes"].shape[0]))
        finally:
            if gh is not None:
                gh.close()
    if plan_text:
        rec["plan"]["text"] = read_plan_text(plan_text)
    rec["bridges"] = [s for s in rec["structures"] if str(s.get("Type", "")).lower().startswith("bridge")]
    return rec


def to_markdown(rec):
    L = ["# Hydraulic data provenance", ""]
    hr = rec["hec_ras"]
    L += [f"- HEC-RAS: {hr.get('File Version', '?')}, units {hr.get('Units System', '?')}"]
    pt = rec["plan"].get("text", {})
    pi = rec["plan"].get("Plan Information", {})
    for k in ("Plan Title", "Plan Name", "Geometry Title", "Flow Title"):
        if k in pi or k in pt:
            L.append(f"- {k}: {pi.get(k, pt.get(k))}")
    for k in ("Simulation Date", "Computation Interval", "Output Interval", "Mapping Interval", "Geom File", "Flow File"):
        if k in pt:
            L.append(f"- {k}: {pt[k]}")
    o = rec["outputs"]
    if o:
        L.append(f"- Outputs in the plan HDF: {o['n_outputs']} from {o['first']} to {o['last']}, every {o['interval_minutes']} min")
    for a in rec["flow_areas"]:
        L.append(f"- 2D area {a['name']}: {a['cell_count']} cells, nominal spacing {a['spacing_dx']} x {a['spacing_dy']}, default n {a['default_mannings_n']}")
    L.append(f"- 1D cross sections: {rec['cross_sections']}")
    L.append(f"- Reference points: {', '.join(rec['reference_points']) or 'none'}")
    L.append(f"- Bridges in the geometry: {len(rec['bridges'])}" + (" (" + "; ".join(f"{b.get('River','')} {b.get('Reach','')} {b.get('RS','')}".strip() for b in rec['bridges']) + ")" if rec['bridges'] else ""))
    L += ["", "## Files", ""] + [f"- `{n}`: {d['bytes']:,} bytes, SHA-256 `{d['sha256']}`" for n, d in rec["files"].items()]
    L += ["", "## To confirm with the modellers", ""] + [f"- [ ] {x}" for x in rec["to_confirm"]]
    L += ["", "## Processing decisions (fill in)", "", "| Date | Step | Decision |", "|---|---|---|", "| | | |"]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("plan_hdf")
    ap.add_argument("--geometry")
    ap.add_argument("--plan-text")
    ap.add_argument("-o", "--output", default="provenance")
    a = ap.parse_args(argv)
    rec = collect(a.plan_hdf, a.geometry, a.plan_text)
    json.dump(rec, open(a.output + ".json", "w"), indent=1, default=str)
    open(a.output + ".md", "w").write(to_markdown(rec))
    print(to_markdown(rec))


if __name__ == "__main__":
    main()
