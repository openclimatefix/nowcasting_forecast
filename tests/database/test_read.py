import logging
from datetime import datetime
from unittest import mock

import pytest
from freezegun import freeze_time

from nowcasting_forecast.database.fake import make_fake_forecasts
from nowcasting_forecast.database.models import ForecastSQL
from nowcasting_forecast.database.read import get_forecast
from nowcasting_forecast.database.save import save

logger = logging.getLogger(__name__)


def test_read(db_session):

    forecasts = make_fake_forecasts(gsp_ids=range(0,10))
    db_session.add_all(forecasts)

    forecasts_read = get_forecast(session=db_session)
    assert len(forecasts_read) == 10

    forecast_read = forecasts_read[0]
    assert forecast_read.location.gsp_id == forecasts[0].location.gsp_id
    assert forecast_read.forecast_values[0] == forecasts[0].forecast_values[0]


# def test_read_gsp_id(db_session):
#
#     forecasts = make_fake_forecasts(session=db_session)
#
#     forecasts_read = get_forecast(session=db_session, gsp_id=forecasts[0].gsp_id)
#     assert len(forecasts_read) == 4
#
#
# def test_read_t0_datetime(db_session):
#
#     forecasts = make_fake_forecasts(session=db_session)
#
#     forecasts_read = get_forecast(session=db_session, t0_datetime_utc=forecasts[0].t0_datetime_utc)
#     assert len(forecasts_read) == 20
#
#
# def test_read_target_datetime_utc(db_session):
#
#     forecasts = make_fake_forecasts(session=db_session)
#
#     forecasts_read = get_forecast(
#         session=db_session, target_datetime_utc=forecasts[0].target_datetime_utc
#     )
#     assert len(forecasts_read) == 10


@pytest.mark.skip("Need to have postgres database as test database - #20")
def test_read_latest(db_session):

    with freeze_time("2022-01-01 00:00:00"):
        _ = make_fake_forecasts(session=db_session)

    with freeze_time("2022-01-01 00:00:05"):
        forecasts_2 = make_fake_forecasts(session=db_session)

    forecasts_read = get_forecast(session=db_session)
    assert len(forecasts_read) == 40
    forecast_read = forecasts_read[0]
    assert forecast_read.gsp_id == forecasts_2[0].gsp_id
    assert forecast_read.t0_datetime_utc == forecasts_2[0].t0_datetime_utc
    assert forecast_read.target_datetime_utc == forecasts_2[0].target_datetime_utc
    assert forecast_read.created_utc == datetime(2022, 1, 1, 0, 0, 5)
    assert len(forecast_read.statistics) == 2
