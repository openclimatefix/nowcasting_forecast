from nowcasting_forecast.database.fake import (
    make_fake_location,
    make_fake_input_data_last_updated,
    make_fake_forecast_value,
    make_fake_forecast,
)
from nowcasting_forecast.database.models import (
    LocationSQL,
    Location,
    InputDataLastUpdatedSQL,
    InputDataLastUpdated,
ForecastValue, ForecastValueSQL, Forecast, ForecastSQL
)
from datetime import datetime, timezone


def test_make_fake_location():
    location_sql: LocationSQL = make_fake_location(1)
    location = Location.from_orm(location_sql)
    _ = Location.to_orm(location)


def test_make_fake_input_data_last_updated():
    input_sql: InputDataLastUpdatedSQL = make_fake_input_data_last_updated()
    input = InputDataLastUpdated.from_orm(input_sql)
    _ = InputDataLastUpdated.to_orm(input)


def test_make_fake_forecast_value():

    target = datetime(2022, 1, 1, tzinfo=timezone.utc)

    forecast_value_sql: ForecastValueSQL = make_fake_forecast_value(target_time=target)
    forecast_value = ForecastValue.from_orm(forecast_value_sql)
    _ = ForecastValue.to_orm(forecast_value)


def test_make_fake_forecast():
    forecast_sql: ForecastSQL = make_fake_forecast(gsp_id=1)
    forecast = Forecast.from_orm(forecast_sql)
    _ = Forecast.to_orm(forecast)
