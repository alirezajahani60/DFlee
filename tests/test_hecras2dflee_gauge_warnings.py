from datetime import datetime as D

from flee.preprocessing.hecras2dflee import gauge_warnings as W

OBS = [(D(2019, 10, 18), 7.3), (D(2019, 10, 22), 7.8), (D(2019, 10, 24), 8.3), (D(2019, 10, 25), 8.35)]
START = D(2019, 10, 1)


def test_crossing_interpolates_between_readings():
    t = W.crossing_times(OBS, [7.3, 7.55, 9.0])
    assert t[0] == D(2019, 10, 18)
    assert t[1] == D(2019, 10, 20)          # halfway between 7.3 and 7.8
    assert t[2] is None


def test_first_reading_above_threshold_is_returned_as_is():
    assert W.crossing_times(OBS, [6.5])[0] == D(2019, 10, 18)


def test_schedule_steps_lead_override_and_order():
    rows = W.schedule(OBS, [6.5, 7.3, 8.3], START, steps_per_day=4,
                      override={1: D(2019, 10, 7)})
    assert [(k, s) for k, _, s in rows] == [(1, 24), (2, 68), (3, 92)]
    # a lead of 2 days on stage 3 only
    rows = W.schedule(OBS, [6.5, 7.3, 8.3], START, 4, lead=[0, 0, 2], override={1: D(2019, 10, 7)})
    assert rows[2][2] == 84
    # stage 2 cannot be issued before stage 1
    rows = W.schedule(OBS, [6.5, 7.3, 8.3], START, 4, lead=[0, 30, 0], override={1: D(2019, 10, 7)})
    assert rows[1][2] == rows[0][2] == 24


def test_unreached_stage_without_override_is_dropped(tmp_path):
    rows = W.schedule(OBS, [7.3, 9.0], START, 4)
    assert [k for k, _, _ in rows] == [1]
    p = tmp_path / "w.csv"
    W.write_warnings_csv(str(p), rows, ["A", "B"])
    assert p.read_text().splitlines()[1] == "1,68,A;B"
