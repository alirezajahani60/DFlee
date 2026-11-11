"""Read 2D flow-area results from HEC-RAS plan HDF files.

Supports HEC-RAS 6.x HDF output. Geometry (cell centres, faces) can come from
the plan HDF itself or from a separate geometry HDF (``.gXX.hdf``). Depth is
taken, in order of preference, from a ``Depth`` dataset, from water surface
minus ``Cells Minimum Elevation``, or from water surface minus a terrain
raster sampled at cell centres. Cell velocity is taken from ``Cell Velocity``
datasets if present, otherwise reconstructed from face-normal velocities by
least squares. All quantities are converted to SI units (m, m/s).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import warnings

import h5py
import numpy as np

TS = "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series"
GEOM = "Geometry/2D Flow Areas"
FT = 0.3048


@dataclass
class MeshResults:
    """Time series for one 2D flow area, in SI units."""

    name: str
    times: list[datetime]          # length T
    centres: np.ndarray            # (N, 2) cell centres, model CRS
    depth: np.ndarray              # (T, N) m
    speed: np.ndarray              # (T, N) m/s
    perimeter: np.ndarray          # (P, 2) polygon of the flow area
    crs_wkt: str | None
    bed: np.ndarray | None = None  # (N,) cell minimum elevation, m (for point depths on roads)


def _decode(x):
    return x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)


def _parse_time(s: str) -> datetime:
    s = _decode(s).strip()
    for fmt in ("%d%b%Y %H:%M:%S:%f", "%d%b%Y %H:%M:%S", "%d%b%Y %H%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    # HEC-RAS writes 24:00 for midnight in some versions
    if " 24:00" in s:
        base = datetime.strptime(s.split(" ")[0], "%d%b%Y")
        from datetime import timedelta
        return base + timedelta(days=1)
    raise ValueError(f"Unrecognised HEC-RAS time stamp: {s!r}")


def _is_us(h: h5py.File) -> bool:
    for key in ("Units System",):
        if key in h.attrs and "US" in _decode(h.attrs[key]):
            return True
    ws_units = None
    for name in h.get(f"{TS}/2D Flow Areas", {}):
        ds = h[f"{TS}/2D Flow Areas/{name}"].get("Water Surface")
        if ds is not None and "Units" in ds.attrs:
            ws_units = _decode(ds.attrs["Units"]).lower()
            break
    return ws_units in ("ft", "feet")


def _crs(h: h5py.File) -> str | None:
    if "Projection" in h.attrs:
        return _decode(h.attrs["Projection"])
    return None


def flow_area_names(h: h5py.File) -> list[str]:
    if f"{TS}/2D Flow Areas" not in h:
        return []
    return list(h[f"{TS}/2D Flow Areas"].keys())


def _cell_velocity_lsq(geom: h5py.Group, face_vel: np.ndarray) -> np.ndarray:
    """Reconstruct cell velocity magnitude from face-normal velocities.

    For each cell, solve min_u sum_f L_f (u . n_f - u_f)^2, where n_f is the
    face unit normal, L_f the face length and u_f the face-normal velocity.
    HEC-RAS reports u_f along the face's own normal, so the cell orientation
    sign is not applied (checked on the BaldEagleCrkMulti2D example: applying
    it raises the median relative residual from 0.27 to 0.88). Returns (T, N).
    """
    nl = geom["Faces NormalUnitVector and Length"][()]       # (F, 3)
    info = geom["Cells Face and Orientation Info"][()]      # (N, 2) start, count
    vals = geom["Cells Face and Orientation Values"][()]    # (K, 2) face, sign
    T = face_vel.shape[0]
    N = info.shape[0]
    speed = np.zeros((T, N), dtype=float)
    for c in range(N):
        start, count = int(info[c, 0]), int(info[c, 1])
        if count < 2:
            continue
        faces = vals[start:start + count, 0].astype(int)
        n = nl[faces, :2]
        w = np.sqrt(np.maximum(nl[faces, 2], 1e-9))
        A = n * w[:, None]
        # least squares for all time steps at once
        b = face_vel[:, faces] * w[None, :]                    # (T, count)
        sol, *_ = np.linalg.lstsq(A, b.T, rcond=None)          # (2, T)
        speed[:, c] = np.hypot(sol[0], sol[1])
    return speed


def clean_nonfinite(depth, speed, area: str = "", nonfinite: str = "dry"):
    """Depth (clipped at 0) and speed with NaN or infinite values set to 0, and their counts.

    ``nonfinite="error"`` raises instead; ``"dry"`` warns with the number of values.
    """
    bad_d, bad_v = ~np.isfinite(depth), ~np.isfinite(speed)
    counts = dict(depth=int(bad_d.sum()), velocity=int(bad_v.sum()), values=int(np.size(depth)))
    if nonfinite not in ("dry", "error"):
        raise ValueError(f"hydraulics.nonfinite must be 'dry' or 'error', not {nonfinite!r}")
    if counts["depth"] or counts["velocity"]:
        msg = (f"2D area {area}: {counts['depth']} depth and {counts['velocity']} velocity values of "
               f"{counts['values']} are NaN or infinite")
        if nonfinite == "error":
            raise ValueError(msg + " (hydraulics.nonfinite: error)")
        warnings.warn(msg + "; they are set to zero (dry, still water). Check the HEC-RAS output.")
    return np.where(bad_d, 0.0, np.maximum(depth, 0.0)), np.where(bad_v, 0.0, speed), counts


def read_mesh(plan_hdf: str, area: str, geom_hdf: str | None = None,
              dem_path: str | None = None, nonfinite: str = "dry") -> MeshResults:
    """Read depth and speed time series for one 2D flow area.

    ``nonfinite`` says what to do with NaN or infinite depths or velocities:
    ``"dry"`` (default) sets them to zero with a warning that gives their number,
    ``"error"`` stops. Their counts are kept in ``MeshResults.nonfinite`` either way,
    so a missing value is never silently taken as a dry cell.
    """
    with h5py.File(plan_hdf, "r") as hp:
        us = _is_us(hp)
        crs = _crs(hp)
        ts = hp[f"{TS}/2D Flow Areas/{area}"]
        times = [_parse_time(t) for t in hp[f"{TS}/Time Date Stamp"][()]] \
            if f"{TS}/Time Date Stamp" in hp else \
            [_parse_time(t) for t in hp[f"{TS}/Time Date Stamp (ms)"][()]]
        ws = ts["Water Surface"][()].astype(float)
        depth_ds = ts["Depth"][()].astype(float) if "Depth" in ts else None
        cvx = ts["Cell Velocity - Velocity X"][()] if "Cell Velocity - Velocity X" in ts else None
        cvy = ts["Cell Velocity - Velocity Y"][()] if "Cell Velocity - Velocity Y" in ts else None
        face_vel = ts["Face Velocity"][()].astype(float) if "Face Velocity" in ts else None

        gsrc = hp if f"{GEOM}/{area}/Cells Center Coordinate" in hp else None
        if gsrc is None:
            if geom_hdf is None:
                raise ValueError(
                    f"Plan file has no geometry for '{area}'; pass the matching geometry HDF.")
            hg = h5py.File(geom_hdf, "r")
            crs = crs or _crs(hg)
        else:
            hg = hp
        try:
            geom = hg[f"{GEOM}/{area}"]
            centres = geom["Cells Center Coordinate"][()].astype(float)
            perimeter = geom["Perimeter"][()].astype(float)
            minelev = geom["Cells Minimum Elevation"][()].astype(float) \
                if "Cells Minimum Elevation" in geom else None
            if cvx is not None and cvy is not None:
                speed = np.hypot(cvx, cvy).astype(float)
            elif face_vel is not None:
                speed = _cell_velocity_lsq(geom, face_vel)
            else:
                raise ValueError(f"No velocity output for '{area}'.")
        finally:
            if hg is not hp:
                hg.close()

    # Keep real computational cells only (HEC-RAS appends perimeter ghost cells)
    ncell = _real_cell_count(plan_hdf, geom_hdf, area, centres.shape[0])
    centres = centres[:ncell]
    ws = ws[:, :ncell]
    speed = speed[:, :ncell]
    if depth_ds is not None:
        depth_ds = depth_ds[:, :ncell]
    if minelev is not None:
        minelev = minelev[:ncell]

    if depth_ds is not None:
        depth = depth_ds
    else:
        if minelev is None:
            if dem_path is None:
                raise ValueError(
                    "No Depth output and no 'Cells Minimum Elevation'; pass a terrain raster (dem_path).")
            minelev = sample_raster(dem_path, centres)
        depth = ws - minelev[None, :]

    if us:
        depth = depth * FT
        speed = speed * FT
        centres_units = "ft"
    else:
        centres_units = "m"
    depth, speed, counts = clean_nonfinite(depth, speed, area, nonfinite)
    speed = np.where(depth > 0.0, speed, 0.0)
    bed = None if minelev is None else (minelev * FT if us else minelev)
    res = MeshResults(area, times, centres, depth, speed, perimeter, crs, bed)
    res.centres_units = centres_units  # type: ignore[attr-defined]
    res.nonfinite = counts  # type: ignore[attr-defined]
    return res


def _real_cell_count(plan_hdf: str, geom_hdf: str | None, area: str, default: int) -> int:
    for path in (plan_hdf, geom_hdf):
        if path is None:
            continue
        with h5py.File(path, "r") as h:
            key = f"{GEOM}/Attributes"
            if key in h:
                for row in h[key][()]:
                    if _decode(row["Name"]).strip() == area:
                        return int(row["Cell Count"])
    return default


def sample_raster(path: str, xy: np.ndarray) -> np.ndarray:
    """Sample a single-band raster at points (raster CRS must match the mesh)."""
    import rasterio
    with rasterio.open(path) as src:
        vals = np.array([v[0] for v in src.sample([tuple(p) for p in xy])], dtype=float)
        if src.nodata is not None:
            vals[vals == src.nodata] = np.nan
    return vals
