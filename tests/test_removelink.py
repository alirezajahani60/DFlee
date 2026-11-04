from flee import flee

"""
Generation 1 code. Incorporates only distance, travel always takes one day.
"""


def test_removelink():
    print("Testing basic data handling and simulation kernel.")

    flee.SimulationSettings.ReadFromYML("empty.yml")

    e = flee.Ecosystem()

    _ = e.addLocation(name="A", movechance=0.3)
    _ = e.addLocation(name="B", movechance=0.0)
    _ = e.addLocation(name="C", movechance=0.0)
    _ = e.addLocation(name="D", movechance=0.0)
    # l1 = e.addLocation(name="A", movechance=0.3)
    # l2 = e.addLocation(name="B", movechance=0.0)
    # l3 = e.addLocation(name="C", movechance=0.0)
    # l4 = e.addLocation(name="D", movechance=0.0)

    e.linkUp(endpoint1="A", endpoint2="B", distance=834.0)
    e.linkUp(endpoint1="A", endpoint2="C", distance=1368.0)
    e.linkUp(endpoint1="A", endpoint2="D", distance=536.0)

    assert e.remove_link(startpoint="A", endpoint="C")

    print("Test successful!")


def _links(e, name):
    loc = e.locations[e._convert_location_name_to_index(name=name)]
    return sorted(l.endpoint.name for l in loc.links), sorted(l.endpoint.name for l in loc.closed_links)


def test_twoway_remove_and_reopen_act_in_both_directions():
    flee.SimulationSettings.ReadFromYML("empty.yml")
    e = flee.Ecosystem()
    for n in ["A", "B", "C"]:
        e.addLocation(name=n, movechance=0.0)
    e.linkUp(endpoint1="A", endpoint2="B", distance=10.0)
    e.linkUp(endpoint1="A", endpoint2="C", distance=10.0)

    # endpoint name built at run time, so it is not the same string object as the stored name
    c = "".join(["C"])
    assert e.close_link(startpoint="A", endpoint=c)
    assert _links(e, "A") == (["B"], ["C"])
    assert _links(e, "C") == ([], ["A"])

    assert e.reopen_link(startpoint="A", endpoint=c)
    assert _links(e, "A") == (["B", "C"], [])
    assert _links(e, "C") == (["A"], [])

    assert e.remove_link(startpoint="A", endpoint="B")
    assert _links(e, "A") == (["C"], [])
    assert _links(e, "B") == ([], [])


if __name__ == "__main__":
    test_removelink()
