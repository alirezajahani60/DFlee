"""Tests for flee/capacity.py (reservations for agents en route, host capacities)."""
import random

import numpy as np
import pytest

from flee import capacity, flee, moving
from flee.SimulationSettings import SimulationSettings


def _world():
    """A (origin) linked to camp C (capacity 10) and village V (pop 40), both 50 km away.

    Agents need several steps to arrive, so every agent decides before anyone arrives.
    """
    SimulationSettings.ReadFromYML("empty.yml")
    SimulationSettings.move_rules["MaxMoveSpeed"] = 20.0
    SimulationSettings.move_rules["MaxWalkSpeed"] = 20.0
    SimulationSettings.move_rules["AwarenessLevel"] = 1
    e = flee.Ecosystem()
    e.addLocation("A", location_type="town", movechance=1.0, pop=1000)
    e.addLocation("C", location_type="camp", capacity=10, pop=0)
    e.addLocation("V", location_type="flood_zone", pop=40)
    e.linkUp("A", "C", 50.0)
    e.linkUp("A", "V", 50.0)
    for loc in e.locations:
        loc.movechance = 1.0 if loc.name == "A" else 0.0
    for _ in range(100):
        e.insertAgent(e.locations[0], {})
    return e


def _occupancy(e):
    """Agents at each location or travelling to it."""
    out = {loc.name: 0 for loc in e.locations}
    for a in e.agents:
        d = capacity.destination(a)
        out[d if d is not None else a.location.name] += 1
    return out


@pytest.fixture(autouse=True)
def _clean():
    yield
    SimulationSettings.capacity_rules = {}
    capacity.reset()


def _run(rules, steps=6, seed=1):
    random.seed(seed)
    np.random.seed(seed)
    e = _world()                            # reads empty.yml, which clears the rules
    SimulationSettings.capacity_rules = rules
    capacity.reset()
    for _ in range(steps):
        e.evolve()
    return e


def test_disabled_by_default():
    SimulationSettings.ReadFromYML("empty.yml")
    assert not capacity.enabled()


def test_without_rules_camp_overfills():
    e = _run({})
    assert _occupancy(e)["C"] > 20          # about half of 100 agents, against a capacity of 10


def test_reservations_respect_camp_capacity():
    e = _run({"enabled": True})
    occ = _occupancy(e)
    assert occ["C"] <= 10
    assert e.locations[1].numAgents <= 10   # arrived
    assert occ["C"] + occ["V"] + occ["A"] == 100


def test_host_capacity_and_stay_if_no_room():
    rules = {"enabled": True, "host_capacity": {"factor": 0.5, "people_per_agent": 1}}
    e = _run(rules)
    assert e.locations[2].capacity == 20    # 0.5 * 40
    occ = _occupancy(e)
    assert occ["C"] <= 10 and occ["V"] <= 20
    assert occ["A"] >= 70                   # no room left: the rest stay at the origin


def test_stay_off_falls_back_to_uniform_choice():
    rules = {"enabled": True, "stay_if_no_room": False, "host_capacity": {"factor": 0.5}}
    e = _run(rules)
    assert _occupancy(e)["A"] < 10          # DFlee's uniform fallback sends agents to full places


def test_arrival_check_ignores_own_reservation():
    e = _world()
    SimulationSettings.capacity_rules = {"enabled": True}
    capacity.reset()
    capacity.get().bind(e)
    camp = e.locations[1]
    camp.pending_arrivals = 10              # full by reservations, nobody arrived
    assert moving.getCapMultiplier(camp, 1) == 0.0
    assert moving.getCapMultiplier(camp, 1, include_pending=False) == 1.0
    a = e.agents[0]
    a.route = ["C"]
    assert a.check_dest_is_full_camp(e) is False


def _line_world():
    """A - B - C, 30 km per link; B and C are camps; agents stop at B after one step."""
    SimulationSettings.ReadFromYML("empty.yml")
    SimulationSettings.move_rules["MaxMoveSpeed"] = 40.0
    SimulationSettings.move_rules["MaxWalkSpeed"] = 40.0
    SimulationSettings.move_rules["AwarenessLevel"] = 2
    e = flee.Ecosystem()
    e.addLocation("A", location_type="town", movechance=1.0, pop=1000)
    e.addLocation("B", location_type="camp", capacity=100, pop=0)
    e.addLocation("C", location_type="camp", capacity=100, pop=0)
    e.linkUp("A", "B", 30.0)
    e.linkUp("B", "C", 30.0)
    for loc in e.locations:
        loc.movechance = 1.0 if loc.name == "A" else 0.0
    a = e.insertAgent(e.locations[0], {})
    return e, e.agents[-1]


@pytest.mark.parametrize("rules, end", [({}, "B"), ({"enabled": True}, "C")])
def test_continue_en_route(rules, end):
    e, a = _line_world()
    SimulationSettings.capacity_rules = rules
    capacity.reset()
    a.route = ["B", "C"]                     # planned, as if chosen earlier
    a.location.movechance = 1.0
    for _ in range(4):
        e.evolve()
        e.locations[0].movechance = 0.0      # no new departures from A
    assert a.location.name == end


def test_replan_when_destination_fills_on_the_way():
    e, a = _line_world()
    SimulationSettings.capacity_rules = {"enabled": True}
    capacity.reset()
    capacity.get().bind(e)
    e.locations[2].numAgents = 100           # C filled by others while the agent was at B
    a.handle_travel(e.locations[1], travelling=False)
    a.route = ["C"]
    link = a.take_next_step(e)
    assert link is not None and link.endpoint.name == "A"   # the only neighbour with room
    assert e.locations[2].pending_arrivals == 0             # reservation at C released
