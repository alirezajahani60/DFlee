from flee import flee


def test_num_displaced_from_origin():
    flee.SimulationSettings.ReadFromYML("empty.yml")
    e = flee.Ecosystem()
    a = e.addLocation(name="A", movechance=1.0, attributes={})
    b = e.addLocation(name="B", movechance=0.0, attributes={})
    e.linkUp(endpoint1="A", endpoint2="B", distance=10.0)
    for _ in range(3):
        e.addAgent(location=a, attributes={})
    assert e.numDisplacedFromOrigin() == 0
    e.agents[0].handle_travel(b, travelling=False)
    e.agents[1].handle_travel(a.links[0], travelling=True)
    assert e.numDisplacedFromOrigin() == 2
    e.agents[1].location = None   # deactivated agents are not counted
    assert e.numDisplacedFromOrigin() == 1
