from typing import List

from nowcasting_forecast.app import run
from nowcasting_forecast.database.connection import DatabaseConnection
from nowcasting_forecast.database.models import ForecastSQL


def test_fake(db_connection: DatabaseConnection):

    run(url=db_connection.url, fake=True)

    with db_connection.get_session() as session:
        forecasts = session.query(ForecastSQL).all()
        assert len(forecasts) == 338 * 4
