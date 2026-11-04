import pytest

from flee import flee, moving


YML = """
log_levels:
  agent: 0
  camp: 0
  init: 0
spawn_rules:
  take_from_population: True
  flood_zone_spawning_only: True
  conflict_zone_spawning_only: False
  flood_driven_spawn_mode:
    flood_spawn_mode: "pop_ratio"
    displaced_per_flood_day: [0.0, 0.1, 0.2, 0.5, 0.9]
move_rules:
  flood_rules:
    flood_movechances: [0.0, 0.5, 0.9, 1.0, 1.0]
    flood_loc_weights: [1.0, 0.5, 0.2, 0.1, 0.0]
{extra}
"""


def _setup(tmp_path, extra=""):
    p = tmp_path / "simsetting.yml"
    p.write_text(YML.format(extra="\n".join("    " + x for x in extra.splitlines())))
    flee.SimulationSettings.ReadFromYML(str(p))
    e = flee.Ecosystem()
    a = e.addLocation(name="A", movechance=0.3, attributes={})
    b = e.addLocation(name="B", movechance=0.0, attributes={})
    e.linkUp(endpoint1="A", endpoint2="B", distance=10.0)
    e.addAgent(location=a, attributes={})
    return e, a, b


def _chance(e, level):
    agent = e.agents[0]
    agent.location.attributes["flood_level"] = level
    return moving.calculateMoveChance(agent, False, 0)[0]


def test_defaults_leave_flood_rules_unchanged(tmp_path):
    e, a, b = _setup(tmp_path)
    s = flee.SimulationSettings
    assert (s.move_rules["FloodSeverityScale"], s.move_rules["FloodMoveScale"],
            s.move_rules["FloodLocScale"], s.spawn_rules["FloodSpawnScale"]) == (1.0, 1.0, 1.0, 1.0)
    assert _chance(e, 1) == pytest.approx(0.5)
    a.attributes["flood_level"] = 0
    b.attributes["flood_level"] = 2
    assert moving.getEndPointScore(e.agents[0], b, 0) == pytest.approx(0.2 * moving.getEndPointScore(e.agents[0], a, 0))


def test_move_and_severity_scales(tmp_path):
    e, a, b = _setup(tmp_path, "flood_move_scale: 2.0\nflood_severity_scale: 2.0")
    assert _chance(e, 1) == pytest.approx(0.9 ** 2)   # level 1 x 2 = level 2
    assert _chance(e, 3) == pytest.approx(1.0)        # level 6 capped at level 4


def test_location_scale(tmp_path):
    e, a, b = _setup(tmp_path, "flood_loc_scale: 0.5")
    b.attributes["flood_level"] = 2
    assert moving.getEndPointScore(e.agents[0], b, 0) == pytest.approx(0.2 ** 0.5 * moving.getEndPointScore(e.agents[0], a, 0))
