"""DFlee hydrodynamically informed movement rules.

Implements the three mechanisms of the Beledweyne coupling paper on top of
DFlee, all switched off unless ``hydro_rules: enabled: true`` is set in
simsetting.yml:

1. Physically based link accessibility (Eqs. 2-3): each agent has a travel
   mode (foot or vehicle). Links closed for that mode at the current step
   (from ``link_states.csv``, produced by hecras2dflee) are excluded from
   route choice, an agent on a link that closes turns back to its start, and
   travel speed is the mode's maximum speed times the link's speed factor.
2. Staged warnings (Eq. 8): from the issue step of stage k, agents at warned
   locations move with probability max(P_flood, pi_k), and the cumulative
   share of forecast-aware agents there rises to rho_k.
3. Anticipatory route weighting (Eqs. 9-10): for forecast-aware agents, the
   weight of route r is multiplied by (1 - lambda * phi_r), phi_r being the
   share of its links forecast to close (for the agent's mode, within the
   horizon H) before the agent, at its current speed, can reach their end;
   destination flooding is evaluated at the maximum forecast flood level over
   the horizon.

Entrapment (Section 5.3) is logged per step and mode: agents at a location
with flood level >= entrapment_level that cannot reach a location below that
level through links open for their mode within the awareness range
(``entrapped``), and those that cannot reach one anywhere on the graph
(``physically_isolated``), which separates physical isolation from the
limits of the agents' knowledge.

Works with the serial Ecosystem (flee.py) and with pflee: under MPI every
rank holds the full network and its own share of agents, so per-agent rules
run unchanged and the entrapment counts are summed over ranks.
"""
from __future__ import annotations

import csv
import random
import sys

from flee.SimulationSettings import SimulationSettings

_STATE = None


class _Hydro:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        # ---- travel modes
        modes = cfg.get("modes") or {"foot": {"share": 1.0}}
        self.mode_names = list(modes)
        shares = [float(m.get("share", 0.0)) for m in modes.values()]
        tot = sum(shares)
        if tot <= 0:
            raise ValueError("hydro_rules.modes: shares must sum to a positive value")
        self.mode_shares = [s / tot for s in shares]
        self.mode_speed = {}
        for name, m in modes.items():
            default = SimulationSettings.move_rules["MaxWalkSpeed"] if name == "foot" \
                else SimulationSettings.move_rules["MaxMoveSpeed"]
            self.mode_speed[name] = float(m.get("max_speed", default))
        # ---- link states: (step, a, b, mode) -> (accessible, speed_factor)
        self.links = {}
        self.max_link_step = -1
        path = cfg.get("link_states_file")
        if path:
            self._read_link_states(path)
        # ---- flood levels (own copy, for forecasts and entrapment)
        self.flood = {}
        self.max_flood_step = -1
        if SimulationSettings.FloodLevelInputFile:
            self._read_flood_levels(SimulationSettings.FloodLevelInputFile)
        # ---- warnings
        w = cfg.get("warnings") or {}
        self.warn_enabled = bool(w.get("enabled", bool(w.get("file"))))
        self.stages = []   # list of (stage, issue_step, set(locations) or None for all)
        if self.warn_enabled:
            self._read_warnings(w["file"])
        self.pi = [float(x) for x in w.get("evac_prob", [])]
        self.rho = [float(x) for x in w.get("aware_share", [])]
        self.aware0 = float(cfg.get("aware_share_without_warnings", 0.0))
        # ---- anticipation
        a = cfg.get("anticipation") or {}
        self.ant_enabled = bool(a.get("enabled", False))
        self.H = int(a.get("horizon", 0))
        self.lam = float(a.get("strength", 0.0))
        # phi_r: share of links ("links", Eq. 10), of route length ("length") or of
        # travel time ("time") forecast to close before the agent gets there
        self.ant_weighting = str(a.get("weighting", "links"))
        if self.ant_weighting not in ("links", "length", "time"):
            raise ValueError("hydro_rules.anticipation.weighting must be links, length or time")
        # forecast timing error in steps: the forecast of the state at t + tau is the
        # true state at t + tau + shift (shift > 0: closures forecast too early)
        self.fc_shift = int(a.get("forecast_shift", 0))
        # ---- entrapment
        self.entrap_level = int(cfg.get("entrapment_level", 2))
        self.entrap_log = cfg.get("entrapment_log", "hydro_entrapment.csv")
        self._log_header_written = False
        # ---- per-agent state, keyed by id(agent): [mode, aware, last_stage_seen]
        self.agent = {}
        self.loc_by_name = {}

    # ------------------------------------------------------------- inputs
    def _read_link_states(self, path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                s = int(row["step"])
                acc = int(row["accessible"]) == 1
                g = float(row.get("speed_factor", 1.0) or 1.0)
                m = row["mode"]
                self.links[(s, row["name1"], row["name2"], m)] = (acc, g)
                self.links[(s, row["name2"], row["name1"], m)] = (acc, g)
                self.max_link_step = max(self.max_link_step, s)

    def _read_flood_levels(self, path):
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        header = [h.strip() for h in rows[0]]
        for name in header[1:]:
            self.flood[name] = []
        for r in rows[1:]:
            if not r:
                continue
            for name, v in zip(header[1:], r[1:]):
                self.flood[name].append(int(v.strip()))
        self.max_flood_step = len(rows) - 2

    def _read_warnings(self, path):
        with open(path, newline="") as f:
            for r in csv.reader(f):
                if not r or r[0].strip().startswith("#"):
                    continue
                stage, step, locs = int(r[0]), int(r[1]), r[2].strip()
                where = None if locs.lower() in ("all", "*", "") else {x.strip() for x in locs.split(";")}
                self.stages.append((stage, step, where))
        self.stages.sort(key=lambda x: (x[1], x[0]))

    # ------------------------------------------------------------- lookups
    def bind(self, e):
        if not self.loc_by_name:
            self.loc_by_name = {loc.name: loc for loc in e.locations}

    def level(self, name, t):
        series = self.flood.get(name)
        if not series:
            return 0
        return series[min(t, len(series) - 1)]

    def link_state(self, a, b, mode, t):
        if self.max_link_step < 0:
            return True, 1.0
        t = min(t, self.max_link_step)
        return self.links.get((t, a, b, mode), (True, 1.0))

    def state(self, agent):
        st = self.agent.get(id(agent))
        if st is None:
            mode = random.choices(self.mode_names, weights=self.mode_shares)[0]
            st = [mode, random.random() < self.aware0, 0]
            self.agent[id(agent)] = st
        return st

    def mode(self, agent):
        return self.state(agent)[0]

    def link_closed(self, agent, a, b, t):
        return not self.link_state(a, b, self.mode(agent), t)[0]

    # ------------------------------------------------------------- warnings
    def stage_at(self, loc_name, t):
        k = 0
        for stage, step, where in self.stages:
            if step <= t and (where is None or loc_name in where):
                k = max(k, stage)
        return k

    def apply_warning(self, agent, t, movechance):
        if not self.warn_enabled:
            return movechance
        k = self.stage_at(agent.location.name, t)
        if k == 0:
            return movechance
        st = self.state(agent)
        # awareness: raise the cumulative aware share to rho_k (once per new stage)
        if k > st[2]:
            prev = self.rho[st[2] - 1] if st[2] > 0 and st[2] <= len(self.rho) else self.aware0
            target = self.rho[k - 1] if k <= len(self.rho) else prev
            if not st[1] and target > prev:
                p = (target - prev) / max(1e-12, 1.0 - prev)
                st[1] = random.random() < p
            st[2] = k
        pik = self.pi[k - 1] if k <= len(self.pi) else 0.0
        return max(movechance, pik)

    # ------------------------------------------------------------- routes
    def filter_routes(self, agent, t, weights, routes):
        """Drop routes with a link closed for the agent's mode; apply anticipation.

        Anticipation counts a link against a route only if, at the agent's
        current travel speed, the agent would reach the end of that link
        after the step at which the link is forecast to close (within the
        horizon H). A road that will close later but can be crossed before
        then is not penalised, so aware agents are encouraged to use escape
        routes while they are still open rather than discouraged from them.
        """
        mode = self.mode(agent)
        origin = agent.location.name
        aware = self.ant_enabled and self.state(agent)[1] and self.H > 0
        speed = max(self.mode_speed[mode], 1e-9)
        out_w, out_r = [], []
        for w, r in zip(weights, routes):
            path = [origin] + list(r)
            pairs = list(zip(path[:-1], path[1:]))
            if any(not self.link_state(a, b, mode, t)[0] for a, b in pairs):
                continue
            if aware:
                blocked, total, cum = 0.0, 0.0, 0.0
                for a, b in pairs:
                    acc, g = self.link_state(a, b, mode, t)
                    d = self._distance(a, b)
                    cum += d / max(g, 1e-3)
                    reach = min(int(cum // speed), self.H)   # steps until the end of this link
                    wt = 1.0 if self.ant_weighting == "links" else (
                        d if self.ant_weighting == "length" else d / max(g, 1e-3))
                    total += wt
                    if any(not self._forecast_link(a, b, mode, t, tau) for tau in range(1, reach + 1)):
                        blocked += wt
                phi = blocked / total if total > 0 else 0.0
                w = w * (1.0 - self.lam * phi)
                w = w * self._destination_factor(path[-1], t)
            out_w.append(w)
            out_r.append(r)
        return out_w, out_r

    def _forecast_link(self, a, b, mode, t, tau):
        """Forecast accessibility of link (a, b) at t + tau, with the timing error."""
        return self.link_state(a, b, mode, max(t, t + tau + self.fc_shift))[0]

    def _distance(self, a, b):
        loc = self.loc_by_name.get(a)
        if loc is not None:
            for link in loc.links:
                if link.endpoint.name == b:
                    return float(link.get_distance())
        return 0.0

    def _destination_factor(self, name, t):
        """Replace W_f(F_j(t)) by W_f(max forecast level over the horizon)."""
        wf = SimulationSettings.move_rules["FloodLocWeights"]
        now = min(self.level(name, t), len(wf) - 1)
        fut = min(max(self.level(name, max(t, t + tau + self.fc_shift)) for tau in range(0, self.H + 1)),
                  len(wf) - 1)
        w_now = float(wf[now])
        if w_now <= 0.0:
            return 1.0
        return float(wf[fut]) / w_now

    # ------------------------------------------------------------- travel
    def travel_speed(self, agent, link, t):
        mode = self.mode(agent)
        acc, g = self.link_state(link.startpoint.name, link.endpoint.name, mode, t)
        return self.mode_speed[mode] * (g if acc else 0.0)

    # ------------------------------------------------------------- logging
    def _can_escape(self, loc, mode, t, depth=None):
        """Whether a location below the entrapment level is reachable over links open
        for this mode now: within ``depth`` links (the awareness range), or without
        limit if ``depth`` is None (physical reachability on the whole graph)."""
        safe = self.entrap_level
        seen = {loc.name}
        frontier = [loc]
        hops = 0
        while frontier and (depth is None or hops < max(1, depth)):
            hops += 1
            nxt = []
            for l in frontier:
                for link in l.links:
                    b = link.endpoint
                    if b.name in seen or not self.link_state(l.name, b.name, mode, t)[0]:
                        continue
                    if self.level(b.name, t) < safe:
                        return True
                    seen.add(b.name)
                    nxt.append(b)
            frontier = nxt
        return False

    def record(self, e, reduce=None, write=True):
        """Log entrapment for this step.

        ``reduce``: optional function summing a numpy array over MPI ranks
        (pflee passes MPIManager.CalcCommWorldTotal); ``write``: whether this
        process writes the log (rank 0 only under pflee).
        """
        t = e.time
        depth = int(round(SimulationSettings.move_rules.get("AwarenessLevel", 1)))
        counts = {m: [0, 0, 0] for m in self.mode_names}   # [in_flooded, entrapped, physically isolated]
        cache = {}
        for a in e.agents:
            if a.location is None or a.travelling:
                continue
            name = a.location.name
            if self.level(name, t) < self.entrap_level:
                continue
            m = self.mode(a)
            counts[m][0] += 1
            key = (name, m)
            if key not in cache:
                cache[key] = (self._can_escape(a.location, m, t, depth), self._can_escape(a.location, m, t, None))
            within, anywhere = cache[key]
            if not within:
                counts[m][1] += 1
            if not anywhere:
                counts[m][2] += 1
        if reduce is not None:
            import numpy as np
            flat = np.array([v for m in self.mode_names for v in counts[m]], dtype="i")
            flat = reduce(flat)
            counts = {m: [int(x) for x in flat[3 * i:3 * i + 3]] for i, m in enumerate(self.mode_names)}
        if not write:
            return
        mode_flag = "w" if not self._log_header_written else "a"
        with open(self.entrap_log, mode_flag, newline="") as f:
            w = csv.writer(f)
            if not self._log_header_written:
                w.writerow(["step", "mode", "agents_in_flooded_locations", "entrapped", "physically_isolated"])
                self._log_header_written = True
            for m, (n, k, iso) in counts.items():
                w.writerow([t, m, n, k, iso])


# ------------------------------------------------------------------ API
def enabled() -> bool:
    return bool(SimulationSettings.hydro_rules.get("enabled", False)) if hasattr(SimulationSettings, "hydro_rules") else False


def get() -> _Hydro:
    global _STATE
    if _STATE is None:
        _STATE = _Hydro(SimulationSettings.hydro_rules)
        print("Hydro rules enabled:", SimulationSettings.hydro_rules, file=sys.stderr)
    return _STATE


def reset() -> None:
    """Drop cached state (for tests or repeated runs in one process)."""
    global _STATE
    _STATE = None
