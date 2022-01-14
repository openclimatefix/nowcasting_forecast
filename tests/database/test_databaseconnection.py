from nowcasting_forecast.database.models import ForecastSQL
from typing import List


def test_get_session(db_connection):

    with db_connection.get_session() as _:
        pass


def test_read_forecast(db_session):
    forecasts = db_session.query(ForecastSQL).all()
    assert len(forecasts) == 0


def test_read_forecast_one(db_session, forecast_sql):

    # query
    forecasts: List[ForecastSQL] = db_session.query(ForecastSQL).all()
    assert len(forecasts) == 1
    assert forecast_sql == forecasts[0]
    assert len(forecasts[0].statistics) == 2
    assert forecast_sql.statistics[0] in forecasts[0].statistics
    assert forecast_sql.statistics[1] in forecasts[0].statistics