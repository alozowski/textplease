import pytest

from textplease.utils.time_utils import parse_time_str, format_time_precise


@pytest.mark.parametrize(("value", "expected"), [(59.9996, "00:01:00.000"), (3599.9996, "01:00:00.000")])
def test_format_rollover_round_trips(value, expected):
    assert format_time_precise(value) == expected
    assert abs(parse_time_str(expected) - value) < 0.001


@pytest.mark.parametrize("value", ["00:00:nan", "00:00:inf", "00:00:-inf"])
def test_nonfinite_time_is_rejected(value):
    with pytest.raises(ValueError):
        parse_time_str(value)
