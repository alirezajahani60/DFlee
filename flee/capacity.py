"""DFlee capacity rules: reservations for agents en route, and host capacities.

Switched off unless ``capacity_rules: enabled: true`` is set in simsetting.yml,
so existing runs are unchanged.

Problem fixed
-------------
DFlee throttles a destination's attractiveness by its occupancy
(``moving.getCapMultiplier``), but occupancy counts only agents that have
already arrived. All agents that decide in the same timestep therefore see the
same (often empty) destination, and a sudden mass departure, as in a flash
flood, overfills camps many times over. Non-camp locations have no capacity at
all, so they absorb any overflow.

Rules
-----
1. ``count_en_route`` (default true). Occupancy used in route choice is
   arrived agents plus agents travelling to the location as their final
   destination ("reservations"). Reservations are recounted from the agents'
   routes at the start of every step and incremented as soon as an agent
   chooses a route, so later agents in the same step see the location filling.
   The arrival check (``Person.check_dest_is_full_camp``) keeps using arrived
   agents only, so an agent holding a reservation is not turned away by its
   own reservation.
2. ``host_capacity`` (optional). Non-camp locations of the listed types that
   have no capacity get ``capacity = factor * pop / people_per_agent``, the
   number of displaced agents they can host. ``people_per_agent`` converts
   populations given in people when each agent stands for several people;
   leave it at 1 if populations are already scaled (``hasten``).
   A ``host_capacity`` column in locations.csv (agents) sets the capacity of a
   non-camp location directly and takes precedence over the factor, e.g. to
   count only the dry part of a partly flooded town.
3. ``continue_en_route`` (default true). An agent that stopped part-way along
   its route (it ran out of travel time in the previous step) moves on with
   probability 1 instead of the stop's move chance. Otherwise agents passing
   through a camp (move chance near 0) stay there for good and overfill it,
   although they never chose it.
4. ``stay_if_no_room`` (default true). If every candidate route has zero
   weight, the agent stays where it is instead of choosing uniformly among
   them (DFlee's fallback, which sends agents to full locations).

Under pflee each rank holds its own agents. Reservations are summed over ranks
at the start of each step; within a step, a rank's new reservations are
counted ``commit_weight`` times (the number of ranks) as an estimate of the
reservations made on all ranks, since agents are spread evenly.
"""
from __future__ import annotations

import sys

from flee.SimulationSettings import SimulationSettings

_STATE = None


class _Capacity:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.count_en_route = bool(cfg.get("count_en_route", True))
        self.stay_if_no_room = bool(cfg.get("stay_if_no_room", True))
        self.continue_en_route = bool(cfg.get("continue_en_route", True))
        host = cfg.get("host_capacity") or {}
        self.host_factor = float(host.get("factor", 0.0))
        self.host_types = [t.lower() for t in host.get("location_types", ["flood_zone", "town"])]
        self.people_per_agent = float(host.get("people_per_agent", 1.0))
        self.commit_weight = 1
        self.release_on_arrival = True
        self._by_name = {}
        self._hosts_set = False

    # ------------------------------------------------------------- set-up
    def _set_host_capacities(self, e):
        for loc in e.locations:
            if loc.camp or loc.capacity > 0:
                continue
            given = str(loc.attributes.get("host_capacity", "")).strip()
            if given not in ("", "nan", "None"):
                loc.capacity = max(1, int(round(float(given))))
                continue
            if self.host_factor <= 0 or loc.pop <= 0:
                continue
            kinds = []
            if loc.flood_zone:
                kinds.append("flood_zone")
            if loc.town:
                kinds.append("town")
            if loc.conflict > 0:
                kinds.append("conflict")
            if any(k in self.host_types for k in kinds):
                loc.capacity = max(1, int(round(self.host_factor * loc.pop / self.people_per_agent)))

    def bind(self, e, reduce=None, commit_weight: int = 1, release_on_arrival: bool = True) -> None:
        """Called at the start of each step: recount reservations from agents' routes.

        ``reduce``: function summing an integer numpy array over MPI ranks (pflee).
        ``release_on_arrival``: drop a reservation when the agent arrives. True in
        serial DFlee, where arrivals update ``numAgents`` at once; False under pflee,
        where ``numAgents`` is only synchronised after the step, so the reservation
        stands in for the arrived agent until the next recount.
        """
        if not self._hosts_set:
            self._set_host_capacities(e)
            self._hosts_set = True
        self.commit_weight = int(commit_weight)
        self.release_on_arrival = bool(release_on_arrival)
        self._by_name = {loc.name: loc for loc in e.locations}
        for loc in e.locations:
            loc.pending_arrivals = 0
        if not self.count_en_route:
            return
        index = {loc.name: i for i, loc in enumerate(e.locations)}
        counts = [0] * len(e.locations)
        for a in e.agents:
            d = destination(a)
            if d is not None and d in index:
                counts[index[d]] += 1
        if reduce is not None:
            import numpy as np
            counts = reduce(np.array(counts, dtype="i")).tolist()
        for loc, c in zip(e.locations, counts):
            loc.pending_arrivals = int(c)

    # ------------------------------------------------------------- queries
    def occupancy(self, loc) -> float:
        return loc.numAgents + getattr(loc, "pending_arrivals", 0)

    def commit(self, name: str) -> None:
        """An agent has just chosen a route ending at ``name``."""
        if self.count_en_route:
            loc = self._by_name.get(name)
            if loc is not None:
                loc.pending_arrivals = getattr(loc, "pending_arrivals", 0) + self.commit_weight

    def release(self, name: str) -> None:
        """An agent has dropped a route ending at ``name`` before arriving."""
        if self.count_en_route:
            loc = self._by_name.get(name)
            if loc is not None:
                loc.pending_arrivals = max(0, getattr(loc, "pending_arrivals", 0) - self.commit_weight)


    def arrived(self, name: str) -> None:
        """An agent has reached its final destination ``name``."""
        if self.release_on_arrival:
            self.release(name)


def destination(a):
    """Final destination name of an agent that has not yet arrived, else None."""
    if a.location is None:
        return None
    if len(a.route) > 0:
        return a.route[-1]
    if a.travelling:
        return a.location.endpoint.name
    return None


# ------------------------------------------------------------------ API
def enabled() -> bool:
    rules = getattr(SimulationSettings, "capacity_rules", None) or {}
    return bool(rules.get("enabled", False))


def get() -> _Capacity:
    global _STATE
    if _STATE is None:
        _STATE = _Capacity(SimulationSettings.capacity_rules)
        print("Capacity rules enabled:", SimulationSettings.capacity_rules, file=sys.stderr)
    return _STATE


def reset() -> None:
    """Drop cached state (for tests or repeated runs in one process)."""
    global _STATE
    _STATE = None
