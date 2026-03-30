"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest

from inflammation.models import daily_mean, daily_min

@pytest.mark.parametrize(
    "test_mean, expected_mean",
    [
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
        ([ [1, 2], [3, 4], [5, 6] ], [3, 4]),
    ])

def test_daily_mean(test_mean, expected_mean):
    """Test that mean function works for an array of zeros."""

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(np.array(test_mean)), np.array(expected_mean))

@pytest.mark.parametrize( #Without parametrize, if the first test fails, pytest stops
        'test_min, expected_min',
        [
            ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
            ([ [1, 2], [3, 4], [5, 6] ], [1, 2]),
            ([ [1,-3], [-3,2], [5,-6] ], [-3, -6]),
            ([ [-4,5], [-2,2], [1,-5] ], [-4, -5])
        ]
)

def test_daily_min(test_min, expected_min):
    """Test that min function works for an array of zeros."""
    npt.assert_array_equal(daily_min(np.array(test_min)), np.array(expected_min))

def test_daily_min_string():
    """Test for TypeError when passing strings"""

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])