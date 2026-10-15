"""Tests for the condition mini-language parser."""

import pytest

from pode.generation.conditional import parse_conditions, parse_conditions_config


def test_parse_simple_split():
    groups = parse_conditions("0.5:X=1;0.5:X=0", total_samples=100)
    assert len(groups) == 2
    assert groups[0].num_rows == 50
    assert groups[1].num_rows == 50
    assert groups[0].fixed_conditions == {"X": 1}
    assert groups[1].fixed_conditions == {"X": 0}


def test_parse_continuous_range():
    groups = parse_conditions("1.0:Age:3.0:10.0", total_samples=200)
    assert len(groups) == 1
    assert groups[0].num_rows == 200
    assert groups[0].rejection_conditions == [{"column": "Age", "min": 3.0, "max": 10.0}]
    assert groups[0].fixed_conditions == {}


def test_parse_mixed_constraints():
    groups = parse_conditions("0.3:X=1,Age:60:100;0.7:Age:20:45", total_samples=1000)
    assert groups[0].num_rows == 300
    assert groups[0].fixed_conditions == {"X": 1}
    assert groups[0].rejection_conditions == [{"column": "Age", "min": 60.0, "max": 100.0}]
    assert groups[1].num_rows == 700


def test_proportions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        parse_conditions("0.5:X=1;0.3:X=0", total_samples=100)


def test_value_casting():
    groups = parse_conditions("0.5:X=1.5;0.5:Y=abc", total_samples=100)
    assert groups[0].fixed_conditions == {"X": 1.5}
    assert groups[1].fixed_conditions == {"Y": "abc"}


def test_invalid_constraint_raises():
    with pytest.raises(ValueError, match="Invalid constraint"):
        parse_conditions("1.0:weird_constraint", total_samples=100)


def test_parse_conditions_config_simple_equality():
    groups = parse_conditions_config(
        {
            "groups": [
                {"proportion": 0.5, "equals": {"X": 1}},
                {"proportion": 0.5, "equals": {"X": 0}},
            ]
        },
        total_samples=100,
    )
    assert groups[0].num_rows == 50
    assert groups[1].num_rows == 50
    assert groups[0].fixed_conditions == {"X": 1}
    assert groups[1].fixed_conditions == {"X": 0}


def test_parse_conditions_config_mixed_constraints():
    groups = parse_conditions_config(
        {
            "groups": [
                {
                    "proportion": 0.3,
                    "equals": {"X": 1},
                    "ranges": {"Age": {"min": 60, "max": 100}},
                },
                {"proportion": 0.7, "ranges": {"Age": {"min": 20, "max": 45}}},
            ]
        },
        total_samples=1000,
    )
    assert groups[0].num_rows == 300
    assert groups[0].fixed_conditions == {"X": 1}
    assert groups[0].rejection_conditions == [{"column": "Age", "min": 60.0, "max": 100.0}]
    assert groups[1].rejection_conditions == [{"column": "Age", "min": 20.0, "max": 45.0}]


def test_parse_conditions_config_proportions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        parse_conditions_config(
            {"groups": [{"proportion": 0.5}, {"proportion": 0.3}]},
            total_samples=100,
        )


def test_parse_conditions_config_rejects_bad_shape():
    with pytest.raises(ValueError, match="non-empty 'groups' list"):
        parse_conditions_config({"groups": {}}, total_samples=100)

    with pytest.raises(ValueError, match="Unsupported condition config keys"):
        parse_conditions_config({"groups": [], "extra": True}, total_samples=100)

    with pytest.raises(ValueError, match="Unsupported condition group 0 keys"):
        parse_conditions_config(
            {"groups": [{"proportion": 1.0, "mean": {"Age": 5}}]},
            total_samples=100,
        )

    with pytest.raises(ValueError, match="min' and 'max"):
        parse_conditions_config(
            {"groups": [{"proportion": 1.0, "ranges": {"Age": {"min": 1}}}]},
            total_samples=100,
        )
