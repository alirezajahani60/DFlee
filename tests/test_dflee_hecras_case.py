"""End-to-end run of the Bald Eagle hydrodynamic test case (link states from HEC-RAS via hecras2dflee)."""
import os
import subprocess
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE = os.path.join(ROOT, "test_data", "test_data_dflee", "dflee_test_hecras_bald_eagle")
RUN = os.path.join(ROOT, "runscripts", "run.py")
FLOODED = ["Flood_A", "Flood_B", "Flood_C"]


def _run(variant, tmp_path):
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    out = subprocess.run([sys.executable, RUN, "input_csv", "source_data", "0", f"simsetting_{variant}.yml"],
                         cwd=CASE, env=env, capture_output=True, text=True, check=True).stdout
    p = tmp_path / f"{variant}.csv"
    p.write_text(out)
    for f in os.listdir(CASE):
        if f.startswith("entrapment_"):
            os.replace(os.path.join(CASE, f), tmp_path / f)
    return pd.read_csv(p)


@pytest.mark.parametrize("variant", ["V0h", "V1", "V2", "V3"])
def test_variant_runs(variant, tmp_path):
    df = _run(variant, tmp_path)
    assert len(df) == 12
    if variant != "V0h":
        assert (tmp_path / f"entrapment_{variant}.csv").exists()


def test_closed_roads_keep_people_in_flooded_settlements(tmp_path):
    left = {v: sum(_run(v, tmp_path)[f"{n} sim"].iloc[-1] for n in FLOODED) for v in ("V0h", "V1")}
    assert left["V1"] > left["V0h"]
