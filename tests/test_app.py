import pytest
from click.testing import CliRunner

from nowcasting_forecast.app import run
from nowcasting_forecast.database.connection import DatabaseConnection
from nowcasting_forecast.database.models import Forecast, ForecastSQL


def test_fake(db_connection: DatabaseConnection):

    runner = CliRunner()
    response = runner.invoke(run, ["--db-url", db_connection.url, "--fake", "true"])
    assert response.exit_code == 0

    with db_connection.get_session() as session:
        forecasts = session.query(ForecastSQL).all()
        _ = Forecast.from_orm(forecasts[0])
        assert len(forecasts) == 338 + 1  # 338 gsp + national


@pytest.mark.skip("Skip for now #11")
def test_not_fake(db_connection: DatabaseConnection):

    runner = CliRunner()
    response = runner.invoke(run, ["--db-url", db_connection.url, "--fake", "false"])
    assert response.exit_code == 0

    with db_connection.get_session() as session:
        forecasts = session.query(ForecastSQL).all()
        _ = Forecast.from_orm(forecasts[0])
        assert len(forecasts) == 338 + 1  # 338 gsp + national
