"""Warning schedules for DFlee from gauge observations and flood-risk thresholds.

DFlee's staged warnings (``hydro_rules.warnings``) read a ``warnings.csv`` with
columns ``stage,issue_step,locations``. This module builds that file from a
river-gauge record: stage k is issued when the observed (or forecast) stage
first reaches threshold k, optionally ``lead`` days earlier, which is how the
lead-time experiments shift the schedule.

The gauge record may be daily or sparse (e.g. readings quoted in bulletins).
With ``interpolate=True`` the crossing time is found by linear interpolation
between consecutive readings; otherwise it is the first reading at or above
the threshold. A stage whose threshold is never reached in the record can be
given explicitly with ``override`` (stage -> datetime), e.g. the issue date of
an advisory; stages neither reached nor overridden are left out.

Command line::

    python -m flee.preprocessing.hecras2dflee.gauge_warnings gauge.csv --thresholds 6.5 7.3 8.3 \
        --start 2019-10-01 --steps-per-day 4 --locations all \
        [--lead 0 2 0] [--override 1=2019-10-07] -o warnings.csv

``gauge.csv`` has columns ``date`` (ISO date or date-time) and ``level`` (m);
further columns are ignored.
"""
from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime, timedelta
from typing import Iterable, Sequence


def _parse_time(s: str) -> datetime:
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"unrecognised date: {s!r}")


def read_gauge(path: str) -> list[tuple[datetime, float]]:
    """Read (time, level) pairs, skipping rows with an empty level; sorted by time."""
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            v = (row.get("level") or "").strip()
            if v:
                out.append((_parse_time(row["date"]), float(v)))
    out.sort()
    if not out:
        raise ValueError(f"{path}: no readings")
    return out


def crossing_times(obs: Sequence[tuple[datetime, float]], thresholds: Sequence[float],
                   interpolate: bool = True) -> list[datetime | None]:
    """First time the stage reaches each threshold, or None if it never does.

    If the first reading is already at or above a threshold, that reading's
    time is returned (the crossing happened at or before the record starts).
    """
    out = []
    for th in thresholds:
        t_cross = None
        for i, (t, h) in enumerate(obs):
            if h >= th:
                if i == 0 or not interpolate:
                    t_cross = t
                else:
                    t0, h0 = obs[i - 1]
                    frac = (th - h0) / (h - h0) if h != h0 else 1.0
                    t_cross = t0 + (t - t0) * frac
                break
        out.append(t_cross)
    return out


def to_step(t: datetime, start: datetime, steps_per_day: int) -> int:
    """DFlee step containing time t (step 0 starts at ``start``)."""
    return math.floor((t - start).total_seconds() / 86400.0 * steps_per_day)


def schedule(obs, thresholds, start, steps_per_day=1, lead: float | Sequence[float] = 0.0,
             override: dict[int, datetime] | None = None, interpolate=True):
    """List of (stage, issue_time, issue_step); stages are numbered from 1.

    ``lead`` (days) is subtracted from each stage's crossing time; a single
    number applies to every stage. Issue steps are clipped at 0 and made
    non-decreasing in stage order, so a large lead on stage 2 cannot issue it
    before stage 1.
    """
    times = crossing_times(obs, thresholds, interpolate)
    override = override or {}
    leads = list(lead) if isinstance(lead, (list, tuple)) else [float(lead)] * len(thresholds)
    if len(leads) != len(thresholds):
        raise ValueError("lead: give one value or one per threshold")
    rows, last = [], 0
    for k, (t, L) in enumerate(zip(times, leads), start=1):
        t = override.get(k, t)
        if t is None:
            continue
        t = t - timedelta(days=L)
        step = max(0, to_step(t, start, steps_per_day))
        step = max(step, last)
        last = step
        rows.append((k, t, step))
    return rows


def write_warnings_csv(path: str, rows, locations: str | Iterable[str] = "all"):
    locs = locations if isinstance(locations, str) else ";".join(locations)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["#stage", "issue_step", "locations"])
        for k, _, step in rows:
            w.writerow([k, step, locs])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gauge")
    ap.add_argument("--thresholds", type=float, nargs="+", required=True)
    ap.add_argument("--start", required=True, help="date of DFlee step 0")
    ap.add_argument("--steps-per-day", type=int, default=1)
    ap.add_argument("--lead", type=float, nargs="+", default=[0.0], help="days, one value or one per stage")
    ap.add_argument("--override", nargs="*", default=[], help="stage=date, for stages not reached in the record")
    ap.add_argument("--no-interpolate", action="store_true")
    ap.add_argument("--locations", default="all", help="'all' or a ;-separated list")
    ap.add_argument("-o", "--output", default="warnings.csv")
    a = ap.parse_args(argv)
    ov = {int(k): _parse_time(v) for k, v in (x.split("=", 1) for x in a.override)}
    lead = a.lead[0] if len(a.lead) == 1 else a.lead
    rows = schedule(read_gauge(a.gauge), a.thresholds, _parse_time(a.start), a.steps_per_day,
                    lead, ov, not a.no_interpolate)
    write_warnings_csv(a.output, rows, a.locations)
    for k, t, s in rows:
        print(f"stage {k}: {t:%Y-%m-%d %H:%M}  step {s}")


if __name__ == "__main__":
    main()
