import logging

from nowcasting_forecast.database.read import get_forecast_values, get_latest_forecast

logger = logging.getLogger(__name__)


def test_get_forecast(db_session, forecasts):

    forecast_read = get_latest_forecast(session=db_session)

    assert forecast_read.location.gsp_id == forecasts[0].location.gsp_id
    assert forecast_read.forecast_values[0] == forecasts[0].forecast_values[0]


def test_get_forecast_values(db_session, forecasts):

    forecast_values_read = get_forecast_values(session=db_session)
    assert len(forecast_values_read) == 20

    assert forecast_values_read[0] == forecasts[0].forecast_values[0]


def test_read_gsp_id(db_session, forecasts):

    forecast_read = get_latest_forecast(session=db_session, gsp_id=forecasts[1].location.gsp_id)
    assert forecast_read.location.gsp_id == forecasts[1].location.gsp_id


def test_get_forecast_values_gsp_id(db_session, forecasts):

    forecast_values_read = get_forecast_values(
        session=db_session, gsp_id=forecasts[0].location.gsp_id
    )
    assert len(forecast_values_read) == 2

    assert forecast_values_read[0] == forecasts[0].forecast_values[0]
