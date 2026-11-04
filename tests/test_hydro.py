"""Tests for flee/hydro.py (hydrodynamically informed movement rules)."""
import random

import pytest

from flee import flee, hydro
from flee.SimulationSettings import SimulationSettings


def _write(path, text):
    path.write_text(text)
    return str(path)


@pytest.fixture
def world(tmp_path):
    """A - B - C line, plus a direct A - C link; C stays dry, A floods."""
    SimulationSettings.ReadFromYML("empty.yml")
    SimulationSettings.move_rules["FloodRulesEnabled"] = True
    SimulationSettings.move_rules["FloodLocWeights"] = [1.0, 0.5, 0.1, 0.0, 0.0]
    SimulationSettings.FloodLevelInputFile = _write(
        tmp_path / "flood_level.csv", "#Day,A,B,C\n0,0,0,0\n1,2,0,0\n2,3,1,0\n3,3,2,0\n")
    links = "step,name1,name2,mode,accessible,speed_factor\n"
    for s in range(4):
        links += f"{s},A,B,foot,1,1.0\n{s},B,C,foot,1,0.5\n"
        links += f"{s},A,C,foot,{1 if s < 2 else 0},1.0\n"
    ls = _write(tmp_path / "link_states.csv", links)
    warn = _write(tmp_path / "warnings.csv", "#stage,issue_step,locations\n1,1,A\n2,2,all\n")
    e = flee.Ecosystem()
    for n in "ABC":
        e.addLocation(n, location_type="flood_zone", movechance=0.3, pop=100)
    e.linkUp("A", "B", 10.0)
    e.linkUp("B", "C", 10.0)
    e.linkUp("A", "C", 30.0)
    cfg = {"enabled": True, "link_states_file": ls,
           "modes": {"foot": {"share": 1.0, "max_speed": 20}},
           "warnings": {"file": warn, "evac_prob": [0.5, 0.9], "aware_share": [0.0, 1.0]},
           "anticipation": {"enabled": True, "horizon": 2, "strength": 1.0},
           "entrapment_log": str(tmp_path / "entrap.csv")}
    SimulationSettings.hydro_rules = cfg
    hydro.reset()
    h = hydro.get()
    h.bind(e)
    yield e, h
    SimulationSettings.hydro_rules = {}
    hydro.reset()


def test_disabled_by_default():
    SimulationSettings.hydro_rules = {}
    assert not hydro.enabled()


def test_link_states_are_symmetric_and_clamped(world):
    e, h = world
    assert h.link_state("C", "A", "foot", 1) == (True, 1.0)
    assert h.link_state("A", "C", "foot", 2) == (False, 1.0)
    # beyond the last step the last state persists
    assert h.link_state("A", "C", "foot", 99) == (False, 1.0)
    # unknown links and modes are open
    assert h.link_state("A", "Z", "vehicle", 0) == (True, 1.0)


def test_speed_uses_mode_and_factor(world):
    e, h = world
    a = flee.Person(e.locations[1], {})
    link = [l for l in e.locations[1].links if l.endpoint.name == "C"][0]
    assert h.travel_speed(a, link, 0) == pytest.approx(10.0)   # 20 km * 0.5


def test_closed_routes_dropped(world):
    e, h = world
    a = flee.Person(e.locations[0], {})
    h.state(a)[1] = False   # not forecast-aware
    w, r = h.filter_routes(a, 2, [1.0, 1.0], [["C"], ["B", "C"]])
    assert r == [["B", "C"]]


def test_warning_stages_and_targeting(world):
    e, h = world
    a = flee.Person(e.locations[0], {})   # at A
    c = flee.Person(e.locations[2], {})   # at C
    assert h.apply_warning(a, 0, 0.1) == 0.1        # nothing issued yet
    assert h.apply_warning(a, 1, 0.1) == 0.5        # stage 1 covers A only
    assert h.apply_warning(c, 1, 0.1) == 0.1
    assert h.apply_warning(c, 2, 0.1) == 0.9        # stage 2 covers all
    assert h.state(c)[1] is True                    # aware share reaches 1.0 at stage 2


def test_anticipation_penalises_link_closing_before_arrival(world):
    e, h = world
    a = flee.Person(e.locations[0], {})
    h.state(a)[1] = True
    # At t=1 the direct A-C link (30 km) is open but closes at t=2; at 20 km/step
    # the agent cannot finish it before then, so the route is penalised to zero.
    w, r = h.filter_routes(a, 1, [1.0, 1.0], [["C"], ["B", "C"]])
    assert r == [["C"], ["B", "C"]]
    assert w[0] == pytest.approx(0.0)
    assert w[1] > 0.0


def test_entrapment_log(world, tmp_path):
    e, h = world
    for _ in range(5):
        e.addAgent(e.locations[0], {})
    e.time = 3
    # close everything out of A for this check
    for key in list(h.links):
        if key[0] == 3 and "A" in key[1:3]:
            h.links[key] = (False, 1.0)
    h.record(e)
    rows = (tmp_path / "entrap.csv").read_text().strip().splitlines()
    assert rows[0].endswith("entrapped,physically_isolated")
    assert rows[-1] == "3,foot,5,5,5"


def test_physical_isolation_differs_from_awareness_range(world, tmp_path):
    e, h = world
    for _ in range(4):
        e.addAgent(e.locations[0], {})
    e.time = 3                     # A level 3, B level 2 (unsafe), C dry
    SimulationSettings.move_rules["AwarenessLevel"] = 1
    for key in list(h.links):      # only A-B and B-C open: C is two links away
        if key[0] == 3 and set(key[1:3]) == {"A", "C"}:
            h.links[key] = (False, 1.0)
    h.record(e)
    last = (tmp_path / "entrap.csv").read_text().strip().splitlines()[-1]
    assert last == "3,foot,4,4,0"  # trapped within one link, but C is reachable


def test_length_weighted_anticipation(world):
    e, h = world
    a = flee.Person(e.locations[0], {})
    h.state(a)[1] = True
    h.ant_weighting = "length"
    # route A-B-C: close only B-C (10 km of 20 km) in the forecast
    for s in (2, 3):
        h.links[(s, "B", "C", "foot")] = (False, 1.0)
        h.links[(s, "C", "B", "foot")] = (False, 1.0)
    w, r = h.filter_routes(a, 1, [1.0], [["B", "C"]])
    assert w[0] == pytest.approx((1.0 - 0.5) * h._destination_factor("C", 1))


def test_forecast_shift_moves_forecast_closures(world):
    e, h = world
    a = flee.Person(e.locations[0], {})
    h.state(a)[1] = True
    # A-C closes at t=2. At t=1 a perfect forecast sees the closure before arrival;
    # a forecast one step late still shows the link open.
    h.fc_shift = 0
    w, r = h.filter_routes(a, 1, [1.0], [["C"]])
    assert w[0] == pytest.approx(0.0)
    h.fc_shift = -1
    w, r = h.filter_routes(a, 1, [1.0], [["C"]])
    assert w[0] > 0.0
    # a forecast one step early sees it closing at t=0 already
    h.fc_shift = 1
    w, r = h.filter_routes(a, 0, [1.0], [["C"]])
    assert w[0] == pytest.approx(0.0)
