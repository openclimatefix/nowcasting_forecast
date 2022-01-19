""" Utils functions for test """
from datetime import datetime, timezone

import pytest

# Used constants
from nowcasting_forecast.database.utils import (convert_to_camelcase,
                                                datetime_must_have_timezone)


def test_datetime_must_have_timezone():
    """Test function datetime_must_have_timezone"""

    time_now = datetime.now(timezone.utc)

    # check functions works
    datetime_must_have_timezone(None, time_now)

    with pytest.raises(ValueError):
        time_now = datetime.now()
        datetime_must_have_timezone(None, time_now)


def test_convert_to_camelcase():
    """Test convert to camelcase works"""
    assert convert_to_camelcase("foo_bar") == "fooBar"
    assert convert_to_camelcase("foo_bar_baz") == "fooBarBaz"
