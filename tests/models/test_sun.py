from datetime import datetime
from freezegun import freeze_time

from nowcasting_forecast.models.sun import filter_forecasts_on_sun_elevation, drop_forecast_on_sun_elevation, warning_message
from nowcasting_datamodel.read.read import get_latest_status
from nowcasting_datamodel.models.models import StatusSQL


def test_filter_forecasts_on_sun_elevation(forecasts):

    # night time
    forecasts[0].forecast_values[0].target_time = datetime(2022, 1, 1)
    forecasts[0].forecast_values[0].expected_power_generation_megawatts = 1

    # day time
    forecasts[0].forecast_values[1].target_time = datetime(2022, 1, 1, 12)
    forecasts[0].forecast_values[1].expected_power_generation_megawatts = 1

    forecasts[-1].location.gsp_id = 338

    _ = filter_forecasts_on_sun_elevation(forecasts)

    assert forecasts[0].forecast_values[0].expected_power_generation_megawatts == 0
    assert forecasts[0].forecast_values[1].expected_power_generation_megawatts == 1


@freeze_time("2022-01-01 12:00:00")
def test_drop_forecast_on_sun_elevation_day(db_session):

    status = StatusSQL(message="",status="ok")
    db_session.add(status)

    forecasts = [1,2,3]

    filter_forecast = drop_forecast_on_sun_elevation(session=db_session,forecasts=forecasts)
    status = get_latest_status(session=db_session)

    assert len(filter_forecast) == 3
    assert status.status == "ok"


@freeze_time("2022-01-01 00:00:00")
def test_drop_forecast_on_sun_elevation_night(db_session):
    status = StatusSQL(message="", status="ok")
    db_session.add(status)

    forecasts = [1, 2, 3]

    filter_forecast = drop_forecast_on_sun_elevation(session=db_session, forecasts=forecasts)

    statuses = db_session.query(StatusSQL).all()

    assert len(filter_forecast) == 0
    assert len(statuses) == 2
    assert statuses[0].status == "ok"
    assert statuses[1].status == "warning"
    assert statuses[1].message == warning_message


@freeze_time("2022-01-01 12:00:00")
def test_drop_forecast_on_sun_elevation_first_day(db_session):
    status = StatusSQL(message=warning_message, status="warning")
    db_session.add(status)

    forecasts = [1, 2, 3]

    filter_forecast = drop_forecast_on_sun_elevation(session=db_session, forecasts=forecasts)
    statuses = db_session.query(StatusSQL).all()

    assert len(filter_forecast) == 3
    assert len(statuses) == 2
    assert statuses[0].status == "warning"
    assert statuses[1].status == "ok"




