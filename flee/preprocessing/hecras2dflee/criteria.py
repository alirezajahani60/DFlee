"""Mode-specific passability criteria and depth-dependent speed factors.

Coefficients are deliberately not hard-coded for criteria whose published
values should be taken from the original papers; such criteria raise an
error until the coefficients are given in the configuration.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Pedestrian stability: Xia et al. (2014), J. Hydraul. Res. 52(1), 93-104.
# Incipient (critical) depth-averaged velocity U_c for water depth h_f, body height
# h_p (m) and mass m_p (kg); the person is unstable when U >= min(sliding, toppling).
#   Sliding,  Eq. (10): U_c = a (h_f/h_p)^b sqrt( m_p/(rho h_p h_f)
#                                - (a1 h_f/h_p + b1)(a2 m_p + b2)/h_p^2 )
#   Toppling, Eq. (13): U_c = a (h_f/h_p)^b sqrt( m_p/(rho h_f^2)
#                                - (a1/h_p^2 + b1/(h_f h_p))(a2 m_p + b2) )
# Body-shape coefficients (Eqs. 2-3): a1 = 0.633, b1 = 0.367 for a typical Chinese
# body (Guo and Wang 1995; 0.737 / 0.263 for an American body after Drillis et al.
# 1964), a2 = 1.015e-3 m3/kg, b2 = -4.927e-3 m3.
# Calibrated alpha (m^0.5/s), beta: model body, Table 1: sliding 7.975, 0.018;
# toppling 3.472, 0.188. Real human subjects, Table 2: sliding 10.253, 0.139
# (22 tests, R2 0.512); toppling 7.867, 0.462 (89 tests, R2 0.465).
# ---------------------------------------------------------------------------
XIA_DEFAULTS = dict(h_p=1.7, m_p=65.0, rho=1000.0, a1=0.633, b1=0.367, a2=1.015e-3, b2=-4.927e-3)
XIA_ALPHA_BETA = {
    "real": {"sliding": (10.253, 0.139), "toppling": (7.867, 0.462)},    # Table 2
    "model": {"sliding": (7.975, 0.018), "toppling": (3.472, 0.188)},    # Table 1
}


def _xia_params(p: dict) -> dict:
    q = dict(XIA_DEFAULTS)
    q.update({k: v for k, v in p.items() if v is not None})
    cal = XIA_ALPHA_BETA[q.get("calibration", "real")]
    q.setdefault("alpha_sliding", cal["sliding"][0]); q.setdefault("beta_sliding", cal["sliding"][1])
    q.setdefault("alpha_toppling", cal["toppling"][0]); q.setdefault("beta_toppling", cal["toppling"][1])
    return q


def xia_incipient_velocity(h, p: dict) -> np.ndarray:
    """Critical velocity (m/s): the lower of the sliding and toppling values."""
    q = _xia_params(p)
    h = np.asarray(h, dtype=float)
    hp, mp, rho = q["h_p"], q["m_p"], q["rho"]
    buoy = q["a2"] * mp + q["b2"]
    with np.errstate(divide="ignore", invalid="ignore"):
        s_in = mp / (rho * hp * h) - (q["a1"] * h / hp + q["b1"]) * buoy / hp ** 2
        t_in = mp / (rho * h ** 2) - (q["a1"] / hp ** 2 + q["b1"] / (h * hp)) * buoy
        us = q["alpha_sliding"] * (h / hp) ** q["beta_sliding"] * np.sqrt(np.maximum(s_in, 0.0))
        ut = q["alpha_toppling"] * (h / hp) ** q["beta_toppling"] * np.sqrt(np.maximum(t_in, 0.0))
    uc = np.minimum(us, ut)
    return np.where(h > 0, uc, np.inf)


def xia_critical_depth(v: float, p: dict, hmax: float = 3.0) -> float:
    """Depth at which U_c(depth) = v, by bisection (U_c decreases with depth)."""
    if v <= 0:
        v = 1e-6
    lo, hi = 1e-3, hmax
    if xia_incipient_velocity(hi, p) > v:
        return hi
    if xia_incipient_velocity(lo, p) < v:
        return lo
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if xia_incipient_velocity(mid, p) > v:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _check(p: dict, keys, name):
    missing = [k for k in keys if p.get(k) is None]
    if missing:
        raise ValueError(
            f"Criterion '{name}' needs coefficients {missing} in the config "
            "(take them from the original paper).")


def passable(depth, speed, crit: dict) -> np.ndarray:
    """Boolean array: True where (depth, speed) is passable for this criterion."""
    d = np.asarray(depth, dtype=float)
    v = np.asarray(speed, dtype=float)
    return _passable(d, v, crit) & (d <= crit.get("max_depth", np.inf))


def _passable(d, v, crit: dict) -> np.ndarray:
    kind = crit["type"]
    if kind == "depth":
        return d <= crit["max_depth"]
    if kind == "hazard_rating":
        # Flood hazard rating HR = d (v + 0.5) + DF (Defra/EA FD2320/FD2321)
        hr = d * (v + 0.5) + crit.get("debris_factor", 0.0)
        return (d <= crit.get("dry_depth", 0.0)) | (hr < crit["max_hr"])
    if kind == "dv_limits":
        # Combined depth, velocity and depth x velocity limits, e.g. the hazard classes of the
        # Australian flood guideline (AIDR 2017, Guideline 7-3): H1 limit (small vehicles)
        # d 0.3 m, v 2.0 m/s, dv 0.3 m2/s; H2 limit (larger vehicles) d 0.5, v 2.0, dv 0.6.
        _check(crit, ["max_depth", "max_velocity", "max_dv"], "dv_limits")
        return (d <= crit["max_depth"]) & (v <= crit["max_velocity"]) & (d * v <= crit["max_dv"])
    if kind == "xia2014":
        uc = xia_incipient_velocity(d, crit)
        return (d <= crit.get("dry_depth", 0.0)) | (v < uc)
    raise ValueError(f"Unknown criterion type: {kind}")


def speed_closed(depth, sf: dict, speed=None) -> np.ndarray:
    """True where the speed function itself reaches zero (e.g. above the Pregnolato
    closure depth, or the zero-speed depth of ``linear``): the link is then closed,
    whatever the passability criterion says."""
    if sf["type"] == "none":
        return np.zeros(np.shape(depth), dtype=bool)
    g = _speed_factor(np.asarray(depth, dtype=float), sf,
                      None if speed is None else np.asarray(speed, dtype=float))
    return g <= 0.0


def speed_factor(depth, sf: dict, speed=None) -> np.ndarray:
    """Factor in [0, 1] multiplying the maximum travel speed.

    ``min_factor`` (default 0.1) is a floor applied where the factor would
    otherwise become very small on a passable link. Where the speed function
    reaches zero the link is closed (``speed_closed``), and the caller sets the
    factor to zero on closed links.

    Thresholds are inclusive limits throughout: a depth equal to a limit
    (``max_depth``, ``closure_mm``, ``zero_speed_depth``) is still passable.
    """
    g = _speed_factor(np.asarray(depth, dtype=float), sf,
                      None if speed is None else np.asarray(speed, dtype=float))
    return np.maximum(g, sf.get("min_factor", 0.1)) if sf["type"] != "none" else g


# Dias et al. (2021), Int. J. Disaster Risk Reduct. 58, 102192, Fig. 11: walking speed
# v = a M^b (m/s) against the specific flood force per unit width
# M = D v_f^2 / g + D^2 / 2 (m3/m; Eq. 2). Fits to 14 participants in still water at
# 0.08, 0.43 and 0.90 m: all observations a = 0.6629, b = -0.103 (R2 0.67);
# minimum speeds a = 0.4922, b = -0.119 (R2 0.80). The factor is normalised to the
# speed at ankle depth (0.08 m, still water) and capped at 1, since the power law
# diverges as M -> 0.
DIAS_FITS = {"mean": (0.6629, -0.103), "minimum": (0.4922, -0.119)}


def dias_flood_force(depth, flow_speed=None, g=9.81):
    d = np.asarray(depth, dtype=float)
    v = np.zeros_like(d) if flow_speed is None else np.asarray(flow_speed, dtype=float)
    return d * v ** 2 / g + d ** 2 / 2.0


def _speed_factor(d, sf: dict, v=None) -> np.ndarray:
    kind = sf["type"]
    if kind == "dias2021":
        a, b = DIAS_FITS[sf.get("fit", "mean")]
        m0 = dias_flood_force(sf.get("reference_depth", 0.08))
        m = np.maximum(dias_flood_force(d, v), 1e-9)
        g = np.minimum(1.0, (m / m0) ** b)
        return np.where(d > 0, g, 1.0)
    if kind == "none":
        return np.ones_like(d)
    if kind == "linear":
        d0 = sf["zero_speed_depth"]
        return np.where(d > d0, 0.0, np.clip(1.0 - d / d0, 1e-9, 1.0))
    if kind == "pregnolato2017":
        # Pregnolato et al. (2017), Transp. Res. D 55, 67-81, Eq. (4): v(w) = 0.0009 w^2
        # - 0.5529 w + 86.9448 (w in mm, v in km/h, R2 = 0.95); roads closed at 300 mm,
        # "the average depth at which a passenger vehicle starts to float" (Table 2).
        c2, c1, c0 = sf.get("c2", 0.0009), sf.get("c1", -0.5529), sf.get("c0", 86.9448)
        w = d * 1000.0
        v = c2 * w ** 2 + c1 * w + c0
        g = np.clip(v / c0, 0.0, 1.0)
        return np.where(w > sf.get("closure_mm", 300.0), 0.0, np.maximum(g, 1e-9))
    raise ValueError(f"Unknown speed function type: {kind}")
