"""Per-agent evacuation outcomes (log_levels.evacuation_outcomes) and their summary."""
import os
import shutil
import subprocess
import sys

import pytest

from flee import flee
from flee.postprocessing import evacuation_outcomes as eo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE = os.path.join(ROOT, "test_data", "test_data_dflee", "dflee_test_hecras_bald_eagle")


def test_tracking_records_departure_exposure_and_safety(tmp_path):
    flee.SimulationSettings.ReadFromYML("empty.yml")
    flee.SimulationSettings.log_levels["evacuation_outcomes"] = 1
    e = flee.Ecosystem()
    a = e.addLocation(name="A", movechance=0.0, attributes={})
    b = e.addLocation(name="B", movechance=0.0, attributes={})
    e.linkUp(endpoint1="A", endpoint2="B", distance=10.0)
    e.addAgent(location=a, attributes={})
    e.addAgent(location=a, attributes={})
    e.track_evacuation_outcomes()                 # t = 0: nothing happened
    e.time = 1
    a.attributes["flood_level"] = 2               # home floods
    e.track_evacuation_outcomes()
    e.time = 2
    e.agents[0].handle_travel(b, travelling=False)  # first agent reaches dry B
    e.track_evacuation_outcomes()
    p = tmp_path / "evacuation_outcomes.csv"
    e.write_evacuation_outcomes(str(p))
    s = eo.summarise(str(tmp_path), steps_per_day=4)["all"]
    assert s["exposed"] == 2 and s["departed"] == 1
    assert s["evacuation_failure"] == pytest.approx(0.5)
    assert s["time_to_safety_h_median"] == pytest.approx(6.0)   # one 6 h step after the home flooded


def _run_case(variant, tmp_path):
    d = tmp_path / variant
    shutil.copytree(CASE, d, ignore=shutil.ignore_patterns("hecras2dflee"))
    yml = (d / f"simsetting_{variant}.yml").read_text().replace("log_levels:\n", "log_levels:\n  evacuation_outcomes: 1\n", 1)
    (d / f"simsetting_{variant}.yml").write_text(yml)
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    subprocess.run([sys.executable, os.path.join(ROOT, "runscripts", "run.py"), "input_csv", "source_data", "0",
                    f"simsetting_{variant}.yml"], cwd=d, env=env, capture_output=True, check=True)
    ent = d / f"entrapment_{variant}.csv"
    return eo.summarise(str(d), steps_per_day=4, entrapment=str(ent) if ent.exists() else None)


def test_closed_roads_raise_evacuation_failure(tmp_path):
    v0h, v1 = _run_case("V0h", tmp_path), _run_case("V1", tmp_path)
    assert v1["all"]["evacuation_failure"] > v0h["all"]["evacuation_failure"]
    assert set(v1["by_mode"]) == {"foot", "vehicle"}
    assert "entrapment_peak" in v1
