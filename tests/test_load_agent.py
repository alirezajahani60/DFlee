from flee import flee, InputGeography
import os
"""
Generation 1 code. Incorporates only distance, travel always takes one day.
"""


def test_load_agent():
   
    ig = InputGeography.InputGeography()

    flee.SimulationSettings.ReadFromYML("empty.yml")

    csvname = os.path.join("test_data", "test_input_csv", "agents.csv")

    print("Testing basic data handling and simulation kernel.")
    e = flee.Ecosystem()

    l1 = e.addLocation(name="A", movechance=0.3)

    ig.ReadAgentsFromCSV(e, csvname)

    assert e.numAgents() == 1 
    # The agent in location A should be loaded.
    # The agent in location B should not be loaded, because B does not exist.

def test_load_agents_quietly_with_comments(tmp_path, capsys):
    """Comment lines before the header are skipped, and nothing is printed to stdout (out.csv)."""
    flee.SimulationSettings.ReadFromYML("empty.yml")
    p = tmp_path / "agents.csv"
    p.write_text("# two agents in A\nlocation,age\nA,30\nA,40\nB,50\n")
    e = flee.Ecosystem()
    e.addLocation(name="A", movechance=0.3, attributes={})
    capsys.readouterr()
    InputGeography.InputGeography().ReadAgentsFromCSV(e, str(p))
    assert e.numAgents() == 2
    assert [a.attributes["age"] for a in e.agents] == ["30", "40"]
    assert capsys.readouterr().out == ""


if __name__ == "__main__":
    test_load_agent()
