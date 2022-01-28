""" Test national forecast"""
from datetime import datetime, timezone

import pytest

# Used constants
from nowcasting_forecast.database.national import make_national_forecast
from nowcasting_forecast.database.fake import make_fake_forecasts
from nowcasting_forecast.database.models import Forecast


def test_make_national_forecast():

    f_sql = make_fake_forecasts(gsp_ids=list(range(0, 338)))
    forecasts = [Forecast.from_orm(f) for f in f_sql]

    national_forecast = make_national_forecast(forecasts=forecasts)


