import pytest

import flee.lib_math as lm


def test_interp_between_and_at_the_ends():
    levels = [0.0, 0.5, 0.9, 1.0, 1.0]
    assert lm.interp(levels, 0) == 0.0
    assert lm.interp(levels, 1.5) == pytest.approx(0.7)
    # the highest level, e.g. flood level 4 with five values, is valid
    assert lm.interp(levels, 4) == 1.0
    assert lm.interp(levels, 4.0) == 1.0


def test_interp_out_of_range():
    with pytest.raises(IndexError):
        lm.interp([0.0, 1.0], -0.1)
    with pytest.raises(IndexError):
        lm.interp([0.0, 1.0], 1.5)
