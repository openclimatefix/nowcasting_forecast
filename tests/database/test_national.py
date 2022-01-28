""" Test national forecast"""
# Used constants
from nowcasting_forecast.database.national import make_national_forecast
from nowcasting_forecast.database.models import Forecast


def test_make_national_forecast(forecasts_all):

    forecasts_all = [Forecast.from_orm(f) for f in forecasts_all]

    national_forecast = make_national_forecast(forecasts=forecasts_all)

    assert type(national_forecast) == Forecast


