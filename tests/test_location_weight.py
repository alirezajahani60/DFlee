import pytest

from flee import flee, scoring


def test_location_weight_attribute_scales_score():
    flee.SimulationSettings.ReadFromYML("empty.yml")
    e = flee.Ecosystem()
    a = e.addLocation(name="A", location_type="town", pop=10, attributes={})
    b = e.addLocation(name="B", location_type="town", pop=10, attributes={"location_weight": "4"})
    c = e.addLocation(name="C", location_type="town", pop=10, attributes={"location_weight": ""})
    for loc in (a, b, c):
        scoring.updateLocationScore(0, loc)
    assert b.scores[0] == pytest.approx(4 * a.scores[0])
    assert c.scores[0] == a.scores[0]
