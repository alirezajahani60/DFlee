"""Summarise the per-agent evacuation outcomes of a DFlee run.

Reads the files written with ``log_levels: evacuation_outcomes: 1``
(``evacuation_outcomes.csv`` from run.py, ``evacuation_outcomes.N.csv`` per rank
from run_par.py) and, if present, the entrapment log of the hydro rules.

Usage:
    python -m flee.postprocessing.evacuation_outcomes RUN_DIR [--steps-per-day 4] [--entrapment entrapment.csv] [-o summary.json]

Definitions (unsafe = flood level >= hydro_rules.entrapment_level, default 2):
- exposed: an agent whose home location becomes unsafe during the run;
- evacuation failure: share of exposed agents that never stand at a safe location
  while their home is unsafe;
- time to safety: steps from the home becoming unsafe to the agent standing at a
  safe location (0 if it had left before), for the exposed agents that reach safety.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


def read_outcomes(run_dir: str) -> pd.DataFrame:
    """All per-agent outcome rows of a run (all ranks)."""
    files = sorted(glob.glob(os.path.join(run_dir, "evacuation_outcomes*.csv")))
    if not files:
        raise FileNotFoundError(f"no evacuation_outcomes*.csv in {run_dir}; set log_levels: evacuation_outcomes: 1")
    return pd.concat([pd.read_csv(f, keep_default_na=False) for f in files], ignore_index=True)


def _failure_and_time(df: pd.DataFrame, hours_per_step: float) -> dict:
    ex = df[df.home_unsafe_step >= 0]
    out = dict(agents=int(len(df)), exposed=int(len(ex)), departed=int((df.departed_step >= 0).sum()))
    if len(ex) == 0:
        return out
    ok = ex[ex.safe_step >= 0]
    out["evacuation_failure"] = float((ex.safe_step < 0).mean())
    if len(ok):
        tts = np.maximum(ok.safe_step - ok.home_unsafe_step, 0) * hours_per_step
        out["time_to_safety_h_median"] = float(np.median(tts))
        out["time_to_safety_h_p90"] = float(np.percentile(tts, 90))
    return out


def summarise(run_dir: str, steps_per_day: float = 1.0, entrapment: str | None = None) -> dict:
    """Evacuation failure and time to safety, overall and per travel mode; entrapment peaks if a log is given."""
    df = read_outcomes(run_dir)
    hps = 24.0 / float(steps_per_day)
    res = dict(all=_failure_and_time(df, hps))
    modes = [m for m in df["mode"].astype(str).unique() if m]
    if modes:
        res["by_mode"] = {m: _failure_and_time(df[df["mode"] == m], hps) for m in sorted(modes)}
    dep = df[df.departed_step >= 0].departed_step
    res["departures_per_step"] = np.bincount(dep.astype(int)).tolist() if len(dep) else []
    if entrapment and os.path.exists(entrapment):
        en = pd.read_csv(entrapment)
        cols = [c for c in ("entrapped", "physically_isolated") if c in en.columns]
        if "mode" in en.columns and cols:
            res["entrapment_peak"] = {m: {c: int(g[c].max()) for c in cols} for m, g in en.groupby("mode")}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--steps-per-day", type=float, default=1.0)
    ap.add_argument("--entrapment", default=None, help="entrapment log of the hydro rules")
    ap.add_argument("-o", "--output", default=None, help="JSON file (default: print)")
    a = ap.parse_args(argv)
    res = summarise(a.run_dir, a.steps_per_day, a.entrapment)
    text = json.dumps(res, indent=1)
    if a.output:
        with open(a.output, "w") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
