"""Network fragmentation over time, per travel mode, from DFlee inputs alone (no agents).

For each DFlee step and mode, using the links open for that mode:
- ``components``: number of connected components of the location graph;
- ``flooded_locations``: locations at or above ``unsafe_level``;
- ``isolated_locations`` / ``isolated_population``: flooded locations (and their
  population) from which no location below ``unsafe_level`` can be reached;
- ``reachable_safe``: mean number of safe locations reachable from a flooded one.
For each origin that floods at some point and each mode, every step (also before
the origin floods) is checked for a route to another safe location;
``last_exit_step`` is the last step with such a route before it is lost for the
rest of the run (empty if it is never lost). It marks the time after which
evacuation from that origin is physically impossible for that mode, which is
what a warning has to beat.

Command line::

    python -m flee.preprocessing.hecras2dflee.fragmentation link_states.csv routes.csv flood_level.csv \
        locations.csv [--unsafe-level 2] -o fragmentation

writes ``<o>_by_step.csv`` and ``<o>_last_exit.csv``.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def _read_levels(path):
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "step"}).set_index("step")
    return df


def fragmentation(link_states: pd.DataFrame, routes: pd.DataFrame, levels: pd.DataFrame,
                  population: dict, unsafe_level: int = 2, steps=None):
    """Return (by_step, last_exit) DataFrames; see the module docstring.

    ``link_states``: columns step, name1, name2, mode, accessible.
    ``routes``: first two columns are the linked location names.
    ``levels``: index step, one column per location with a flood level (others are 0).
    ``population``: location name -> population.
    Links missing from ``link_states`` for a step and mode are open.
    """
    names = sorted(set(routes.iloc[:, 0]) | set(routes.iloc[:, 1]) | set(population))
    ix = {n: i for i, n in enumerate(names)}
    edges = [(ix[a], ix[b]) for a, b in zip(routes.iloc[:, 0], routes.iloc[:, 1])]
    modes = sorted(link_states["mode"].unique()) if len(link_states) else ["all"]
    closed = defaultdict(set)                       # (step, mode) -> {frozenset(a, b)}
    for r in link_states.itertuples(index=False):
        if int(r.accessible) == 0:
            closed[(int(r.step), r.mode)].add(frozenset((r.name1, r.name2)))
    max_step = int(max(levels.index.max(), link_states["step"].max() if len(link_states) else 0))
    steps = range(max_step + 1) if steps is None else steps
    n = len(names)
    ever = np.array([float(levels[nm].max()) >= unsafe_level if nm in levels else False for nm in names])
    rows, iso_hist = [], defaultdict(list)          # (origin, mode) -> [(step, isolated)]
    for t in steps:
        lev_row = levels.loc[min(t, levels.index.max())] if len(levels) else pd.Series(dtype=float)
        lev = np.array([float(lev_row.get(nm, 0)) for nm in names])
        flooded = lev >= unsafe_level
        for m in modes:
            cl = closed.get((t, m), set())
            keep = [(a, b) for a, b in edges if frozenset((names[a], names[b])) not in cl]
            if keep:
                a, b = np.array(keep).T
                g = csr_matrix((np.ones(2 * len(keep)), (np.r_[a, b], np.r_[b, a])), shape=(n, n))
            else:
                g = csr_matrix((n, n))
            ncomp, lab = connected_components(g, directed=False)
            safe_per_comp = np.bincount(lab[~flooded], minlength=ncomp)
            reach = safe_per_comp[lab]               # safe locations in the same component
            iso = flooded & (reach == 0)
            pop = np.array([population.get(nm, 0) for nm in names], dtype=float)
            rows.append(dict(step=t, mode=m, components=int(ncomp), flooded_locations=int(flooded.sum()),
                             isolated_locations=int(iso.sum()), isolated_population=float(pop[iso].sum()),
                             reachable_safe=float(reach[flooded].mean()) if flooded.any() else np.nan))
            for i in np.flatnonzero(ever):
                other_safe = reach[i] - (0 if flooded[i] else 1)   # safe locations other than the origin
                iso_hist[(names[i], m)].append((t, bool(other_safe <= 0), bool(flooded[i])))
    by_step = pd.DataFrame(rows)
    last = []
    for (origin, m), hist in sorted(iso_hist.items()):
        # last step with an exit before isolation that lasts to the end of the record
        tail_iso = None
        for t, isolated, _ in reversed(hist):
            if isolated:
                tail_iso = t
            else:
                break
        exits = [t for t, isolated, _ in hist if not isolated and (tail_iso is None or t < tail_iso)]
        first_flooded = next((t for t, _, fl in hist if fl), None)
        last.append(dict(origin=origin, mode=m, first_flooded_step=first_flooded,
                         isolated_from_step=tail_iso,
                         last_exit_step=(exits[-1] if exits else None) if tail_iso is not None else None))
    return by_step, pd.DataFrame(last)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("link_states"); ap.add_argument("routes"); ap.add_argument("flood_level"); ap.add_argument("locations")
    ap.add_argument("--unsafe-level", type=int, default=2)
    ap.add_argument("-o", "--output", default="fragmentation")
    a = ap.parse_args(argv)
    loc = pd.read_csv(a.locations)
    pop = dict(zip(loc.iloc[:, 0], loc["population"]))
    by_step, last = fragmentation(pd.read_csv(a.link_states), pd.read_csv(a.routes), _read_levels(a.flood_level),
                                  pop, a.unsafe_level)
    by_step.to_csv(f"{a.output}_by_step.csv", index=False)
    last.to_csv(f"{a.output}_last_exit.csv", index=False)
    print(by_step.groupby("mode")[["isolated_locations", "isolated_population"]].max().to_string())


if __name__ == "__main__":
    main()
